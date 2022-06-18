import os
from textwrap import indent
from typing import List, Union
import pickle
import random

import torch
from torch.utils.data import IterableDataset

### You can import any Python standard libraries or pyTorch sub directories here
from torch.utils.data import DataLoader
### END YOUR LIBRARIES

import utils

from bpe import BytePairEncoding
from model import MLMandNSPmodel
from data import ParagraphDataset

# You can use tqdm to check your progress
from tqdm import tqdm, trange

class PretrainDataset(IterableDataset):
    def __init__(self, dataset: ParagraphDataset):
        """ Maked Language Modeling & Next Sentence Prediction dataset initializer
        Use below attributes when implementing the dataset

        Attributes:
        dataset -- Paragraph dataset to make a MLM & NSP sample
        """
        self.dataset = dataset

    @property
    def token_num(self):
        return self.dataset.token_num

    def __iter__(self):
        """ Masked Language Modeling & Next Sentence Prediction dataset
        Sample two sentences from the dataset, and make a self-supervised pretraining sample for MLM & NSP

        Note: You can use any sampling method you know.

        Yields:
        source_sentences -- Sampled sentences
        MLM_sentences -- Masked sentences
        MLM_mask -- Masking for MLM
        NSP_label -- NSP label which indicates whether the sentences is connected.

        Example: If 25% mask with 50 % <msk> + 25% random + 25% same -- this percentage is just a example.
        source_sentences = ['<cls>', 'He', 'bought', 'a', 'gallon', 'of', 'milk',
                            '<sep>', 'He', 'drank', 'it', 'all', 'on', 'the', 'spot', '<sep>']
        MLM_sentences = ['<cls>', 'He', '<msk>', 'a', 'gallon', 'of, 'milk',
                         '<sep>', 'He', 'drank', 'it', 'tree', 'on', '<msk>', 'spot', '<sep>']
        MLM_mask = [False, False, True, False, False, False, False,
                    False, True, False, False, True, False True, False, False]
        NSP_label = True
        """
        # Special tokens
        CLS = BytePairEncoding.CLS_token_idx
        SEP = BytePairEncoding.SEP_token_idx
        MSK = BytePairEncoding.MSK_token_idx

        # The number of tokens
        TOKEN_NUM = self.token_num

        while True:
            ### YOUR CODE HERE (~ 40 lines)
            source_sentences: List[int] = None
            MLM_sentences: List[int] = None
            MLM_mask: List[bool] = None
            NSP_label: bool = None
            
            global_mask_prob = 0.15
            local_mask_prob = 0.8
            local_rand_prob = 0.1
            
            # sample 2 sentences from dataset
            idxs = random.sample(range(len(self.dataset)), 2)
            
            sent1, sent2 = self.dataset[idxs[0]], self.dataset[idxs[1]]
            
            # dist = torch.FloatTensor([0.5, 0.5])
            # nsp = torch.multinomial(dist, 1)
            
            nsp_dist = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.5]))
            nsp = nsp_dist.sample()
            
            # 두개 문서 선택 
            doc_indices = random.sample(range(len(self.dataset)), 2)
            
            if nsp:
                NSP_label = True
                doc_idx = doc_indices[0]
                while len(self.dataset[doc_idx]) < 2:
                    doc_idx = random.randint(0, len(self.dataset)-1)
                    
                sent1_idx = random.randint(0, len(self.dataset[doc_idx])-2)
                sent2_idx = sent1_idx + 1
                sent1 = self.dataset[doc_idx][sent1_idx]
                sent2 = self.dataset[doc_idx][sent2_idx]
            else:
                NSP_label = False
                if len(self.dataset[doc_indices[0]]) < 2:
                    sent1 = self.dataset[doc_indices[0]][0]
                    sent2_idx = random.randint(0, len(self.dataset[doc_indices[1]])-1)
                    sent2 = self.dataset[doc_indices[1]][sent2_idx]
                else:
                    sent1_idx = random.randint(0, len(self.dataset[doc_indices[0]])-1)
                    sent1 = self.dataset[doc_indices[0]][sent1_idx]
                    # if sentence 1 index is the last of doc, not consider next sentence
                    if sent1_idx == len(self.dataset[doc_indices[0]])-1:
                        sent2_idx = random.randint(0, len(self.dataset[doc_indices[1]])-1)
                        sent2 = self.dataset[doc_indices[1]][sent2_idx]
                    # if sentence 1 isn't last sentence of doc, consider next sentence of sent1
                    else:
                        next_sent1 = self.dataset[doc_indices[0]][sent1_idx+1]
                        sent2_idx = random.randint(0, len(self.dataset[doc_indices[1]])-1)
                        sent2 = self.dataset[doc_indices[1]][sent2_idx]
                        
                        while next_sent1 == sent2:
                            sent2_idx = random.randint(0, len(self.dataset[doc_indices[1]])-1)
                            sent2 = self.dataset[doc_indices[1]][sent2_idx]

            # make source sentences
            pair = [sent1, sent2]

            source_sentences = [CLS]
            sent_length = 0
            for sent in pair:
                source_sentences += sent
                sent_length += len(sent)
                source_sentences += [SEP]
            
            masking_prob = torch.tensor(1/sent_length)
            src_sent = torch.tensor(source_sentences)
            special_token = (src_sent == SEP) + (src_sent == CLS)
            
            dist_mask = torch.zeros_like(src_sent, dtype=torch.float32)
            dist_mask[~special_token] = masking_prob

            num_glob_masked = int(sent_length*global_mask_prob)
            num_local_masked = int(num_glob_masked*local_mask_prob)
            num_rand_masked = int(num_glob_masked*local_rand_prob)
            
            glob_mask_idx = torch.multinomial(dist_mask, num_glob_masked)
            
            # makse MLM_mask tensor
            MLM_mask = torch.full(src_sent.shape, False)
            MLM_mask[glob_mask_idx] = True
            
            # perm
            indices = torch.randperm(glob_mask_idx.shape[0])
            loc_mask_idx = indices[:num_local_masked]
            loc_rand_idx = indices[num_local_masked:num_local_masked+num_rand_masked]
            
            mlm_sent = src_sent
            mlm_sent[glob_mask_idx[loc_mask_idx]] = MSK
            
            loc_rand_idx = loc_rand_idx.tolist()
            for idx in loc_rand_idx:
                rand_tok = random.randint(5, self.dataset.token_num-1)
                while mlm_sent[glob_mask_idx[idx]] == rand_tok:
                    rand_tok = random.randint(5, self.dataset.token_num-1)
                
                mlm_sent[glob_mask_idx[idx]] = rand_tok

            MLM_sentences = mlm_sent
            ### END YOUR CODE            

            assert len(source_sentences) == len(MLM_sentences) == len(MLM_mask)
            yield source_sentences, MLM_sentences, MLM_mask, NSP_label

def calculate_losses(
    model: MLMandNSPmodel,
    source_sentences: torch.Tensor,
    MLM_sentences: torch.Tensor,
    MLM_mask: torch.Tensor,
    NSP_label: torch.Tensor
):
    """ MLM & NSP losses calculation
    Use cross entropy loss to calculate both MLN and NSP losses.
    MLM loss should be an average loss of masked tokens.

    Arguments:
    model -- MLM & NSP model
    source_sentences -- Source sentences tensor in torch.long type
                        in shape (sequence_length, batch_size)
    MLM_sentences -- Masked sentences tensor in torch.long type
                        in shape (sequence_length, batch_size)
    MLM_mask -- MLM mask tensor in torch.bool type
                        in shape (sequence_length, batch_size)
    NSP_label -- NSP label tensor in torch.bool type
                        in shape (batch_size, )

    Returns:
    MLM_loss -- MLM loss in scala tensor
    NSP_loss -- NSP loss in scala tensor
    """
    ### YOUR CODE HERE (~4 lines)
    MLM_loss: torch.Tensor = None
    NSP_loss: torch.Tensor = None
    
    # mlm output shape : (seq len, batch size, token num)
    # nsp_output shape : (batch, nsp label)
    mlm_output, nsp_output = model(MLM_sentences)
    
    # declare cross entropy loss class
    ce_loss = torch.nn.CrossEntropyLoss()
    
    # calculate NSP loss
    NSP_label = NSP_label.to(torch.long)
    NSP_loss = ce_loss(nsp_output, NSP_label)
    
    # calculate MLM loss
    # use only masked token
    src_sentence = source_sentences[MLM_mask]
    mlm_output = mlm_output[MLM_mask]
    MLM_loss = ce_loss(mlm_output, src_sentence)

    ### END YOUR CODE
    assert MLM_loss.shape == NSP_loss.shape == torch.Size()
    return MLM_loss, NSP_loss

def pretraining(
    model: MLMandNSPmodel,
    model_name: str,
    train_dataset: PretrainDataset,
    val_dataset: PretrainDataset,
):
    """ MLM and NSP pretrainer
    Implement MLN & NSP pretrainer with the given model and datasets.

    Note 1: Don't forget setting model.train() and model.eval() before training / validation.
            It enables / disables the dropout layers of our model.

    Note 2: There are useful tools for your implementation in utils.py

    Note 3: Training takes almost 3 minutes per a epoch on TITAN RTX. Thus, 200 epochs takes 10 hours.
    For those who don't want to wait 10 hours, we attaches a model which has trained over 200 epochs.
    You can use it on the IMDB training later.

    Memory tip 1: If you delete the loss tensor explictly after every loss calculation like "del loss",
                  tensors are garbage-collected before next loss calculation so you can cut memory usage.

    Memory tip 2: If you use torch.no_grad when inferencing the model for validation,
                  you can save memory space of gradient. 

    Memory tip 3: If you want to keep batch_size while reducing memory usage,
                  creating a virtual batch is a good solution.
    Explanation: https://medium.com/@davidlmorton/increasing-mini-batch-size-without-increasing-memory-6794e10db672

    Useful readings: https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/ 

    Arguments:
    model -- Pretraining model which need to be trained
    model_name -- The model name. You can use this name to save your model
    train_dataset -- Pretraining dataset for training
    val_dataset -- Pretraining dataset for validation

    Variables:
    batch_size -- Batch size
    learning_rate -- Learning rate for the optimizer
    epochs -- The number of epochs
    steps_per_a_epoch -- The number of steps in a epoch.
                        Because there is no end in IterableDataset, you have to set the end of epoch explicitly.
    steps_for_val -- The number of steps for validation

    Returns:
    MLM_train_losses -- List of average MLM training loss per a epoch
    MLM_val_losses -- List of average MLM validation loss per a epoch
    NSP_train_losses -- List of average NSP training loss per a epoch
    NSP_val_losses -- List of average NSP validation loss per a epoch
    """
    # Below options are just our recommendation. You can choose different options if you want.
    batch_size = 16
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # epochs = 30 # 200 if you want to feel the effect of pretraining
    epochs = 5
    steps_per_a_epoch: int=30
    steps_for_val: int=10
    # steps_per_a_epoch: int=2000
    # steps_for_val: int=200

    ### YOUR CODE HERE
    MLM_train_losses: List[float] = None
    MLM_val_losses: List[float] = None
    NSP_train_losses: List[float] = None
    NSP_val_losses: List[float] = None
    
    training_dataset = PretrainDataset(train_dataset)
    validation_dataset = PretrainDataset(val_dataset)
    
    train_loader = DataLoader(training_dataset, batch_size=batch_size, collate_fn=utils.imdb_collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=utils.imdb_collate_fn)
    
    # initialize losses list
    MLM_train_losses = []
    MLM_val_losses = []
    NSP_train_losses = []
    NSP_val_losses = []
    
    for epoch in range(epochs):
        train_steps = 0
        val_steps = 0
        
        mlm_epoch_train_loss = 0
        nsp_epoch_train_loss = 0
        mlm_epoch_val_loss = 0
        nsp_epoch_val_loss = 0
        
        # train part : need to model.train()
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            train_steps += 1
            src_sent, mlm_sent, mlm_mask, nsp_label = batch
            mlm_loss, nsp_loss = calculate_losses(model, src_sent, mlm_sent, mlm_mask, nsp_label)
            
            mlm_epoch_train_loss += mlm_loss.item()
            nsp_epoch_train_loss += nsp_loss.item()
            loss = mlm_loss + nsp_loss
            loss.backward()
            optimizer.step()
            
            del loss, mlm_loss, nsp_loss
            
            if train_steps == steps_per_a_epoch:
                break
        
        # validation part : need to model.eval()
        model.eval()
        for batch in validation_loader:
            val_steps += 1
            src_sent, mlm_sent, mlm_mask, nsp_label = batch
            with torch.no_grad:
                mlm_loss, nsp_loss = calculate_losses(model, src_sent, mlm_sent, mlm_mask, nsp_label)
                mlm_epoch_val_loss += mlm_loss.item()
                nsp_epoch_val_loss += nsp_loss.item()
                
                del mlm_loss, nsp_loss
            
            if val_steps == steps_for_val:
                break
            
        # save loss per epoch
        MLM_train_losses.append(mlm_epoch_train_loss/steps_per_a_epoch)
        MLM_val_losses.append(mlm_epoch_val_loss/steps_for_val)
        NSP_train_losses.append(nsp_epoch_train_loss/steps_per_a_epoch)
        NSP_val_losses.append(nsp_epoch_val_loss/steps_for_val)

    ### END YOUR CODE

    assert len(MLM_train_losses) == len(MLM_val_losses) == epochs and \
           len(NSP_train_losses) == len(NSP_val_losses) == epochs

    assert all(isinstance(loss, float) for loss in MLM_train_losses) and \
           all(isinstance(loss, float) for loss in MLM_val_losses) and \
           all(isinstance(loss, float) for loss in NSP_train_losses) and \
           all(isinstance(loss, float) for loss in NSP_val_losses)

    return MLM_train_losses, MLM_val_losses, NSP_train_losses, NSP_val_losses

##############################################################
# Testing functions below.                                   #
#                                                            #
# We only checks MLM & NSP dataset and loss calculation.     #
# We do not tightly check the correctness of your trainer.   #
# You should attach the loss plot to the report              #
# and submit the pretrained model to validate your trainer.  #
# We will grade the score by running your saved model.       #
##############################################################

def test_MLM_and_NSP_dataset():
    print("======MLM & NSP Dataset Test Case======")
    CLS = BytePairEncoding.CLS_token_idx
    SEP = BytePairEncoding.SEP_token_idx
    MSK = BytePairEncoding.MSK_token_idx

    class Dummy(object):
        def __init__(self):
            self.paragraphs = [[[10] * 100, [11] * 100], [[20] * 100, [21] * 100], [[30] * 100]]

        @property
        def token_num(self):
            return 100

        def __len__(self):
            return len(self.paragraphs)

        def __getitem__(self, index):
            return self.paragraphs[index]
        
    dataset = PretrainDataset(Dummy())

    count = 0
    nsp_true_count = 0
    combinations = set()
    for src, mlm, mask, nsp in dataset:
        # First test
        assert src[0] == mlm[0] == CLS and src[101] == mlm[101] == src[-1] == mlm[101] == SEP and \
               not mask[0] and not mask[101] and not mask[-1], \
                "CLS and SEP should not be masked."
        
        # Second test
        assert not nsp ^ (src[1] // 10 == src[102] // 10 and src[1] % 10 + 1 == src[102] % 10), \
                "Your result does not match NSP label."
        
        # Third test
        assert all(src[1] == src[i] for i in range(2, 100)) and all(src[102] == src[i] for i in range(103, 201)), \
                "You should not modify the source sentence."

        # Forth test
        assert all((w1 == w2 or m) for w1, w2, m in zip(src, mlm, mask)), \
                "Only masked position can have a different token."

        # Fifth test
        assert .145 < sum(mask) / len(src) < .155, \
                "The number of the masked tokens should be 15%% of the total tokens."

        # Sixth test
        assert .795 < sum(word == MSK for word in mlm) / sum(mask) < .805, \
                "80%% of the masked tokens should be converted to MSK tokens"
        
        # Seventh test
        assert .095 < sum(w1 != w2 for w1, w2 in zip(src, mlm) if w2 != MSK) / sum(mask) < .105, \
                "10%% of the masked tokens should be converted to random tokens"
        
        combinations.add((src[1], src[102]))
        nsp_true_count += nsp
        
        count += 1
        if count > 10000:
            break

    # Eighth test
    assert .45 < nsp_true_count / 10000 < .55, \
            "Your NSP label is biased. Buy a lottery if you failed the test with a correct database."

    # Nineth test
    assert len(combinations) >= 18, \
            "The number of sentence combination is too limited."

    print("MLM & NSP dataset test passed!")

def test_loss_calculation():
    print("======MLM & NSP Loss Calculation Test Case======")
    CLS = BytePairEncoding.CLS_token_idx
    SEP = BytePairEncoding.SEP_token_idx
    MSK = BytePairEncoding.MSK_token_idx
    
    torch.manual_seed(1234)
    model = MLMandNSPmodel(100)

    samples = []

    src = [CLS] + [10, 10, 10, 10, 10, 10] + [SEP] + [20, 20, 20, 20, 20] + [SEP]
    mlm = [CLS] + [10, 10, 10, 10, MSK, 10] + [SEP] + [MSK, 20, 20, 15, 20] + [SEP]
    mask = [False, False, True, False, False, True, False, False, True, False, False, True, False, False]
    nsp = True
    samples.append((src, mlm, mask, nsp))

    src = [CLS] + [30, 30, 30] + [SEP] + [40, 40, 40, 40] + [SEP]
    mlm = [CLS] + [MSK, 30, 30] + [SEP] + [40, 45, 40, 40] + [SEP]
    mask = [False, True, False, True, False, False, True, False, False, False]
    nsp = False
    samples.append((src, mlm, mask, nsp))

    src = [CLS] + [10, 20, 30, 40] + [SEP] + [50, 40, 30, 20, 10] + [SEP]
    mlm = [CLS] + [10, MSK, 30, 40] + [SEP] + [50, MSK, 30, 25, 10] + [SEP]
    mask = [False, False, True, False, False, False, False, True, False, True, False ,False]
    nsp = True
    samples.append((src, mlm, mask, nsp))

    src, mlm, mask, nsp = utils.pretrain_collate_fn(samples)

    MLM_loss, NSP_loss = calculate_losses(model, src, mlm, mask, nsp)

    # First test
    assert MLM_loss.allclose(torch.scalar_tensor(5.12392426), atol=1e-2), \
        "Your MLM loss does not match the expected result"
    print("The first test passed!")

    # Second test
    assert NSP_loss.allclose(torch.scalar_tensor(0.59137219), atol=1e-2), \
        "Your NSP loss does not match the expected result"
    print("The second test passed!")

    print("All 2 tests passed!")

def pretrain_model():
    print("======MLM & NSP Pretraining======")
    """ MLM & NSP Pretraining 
    You can modify this function by yourself.
    This function does not affects your final score.
    """
    train_dataset = ParagraphDataset(os.path.join('data', 'imdb_train.csv'))
    train_dataset = PretrainDataset(train_dataset)
    val_dataset = ParagraphDataset(os.path.join('data', 'imdb_val.csv'))
    val_dataset = PretrainDataset(val_dataset)
    model = MLMandNSPmodel(train_dataset.token_num)

    model_name = 'pretrained'

    MLM_train_losses, MLM_val_losses, NSP_train_losses, NSP_val_losses \
            = pretraining(model, model_name, train_dataset, val_dataset)

    torch.save(model.state_dict(), model_name+'_final.pth')

    with open(model_name+'_result.pkl', 'wb') as f:
        pickle.dump((MLM_train_losses, MLM_val_losses, NSP_train_losses, NSP_val_losses), f)

    utils.plot_values(MLM_train_losses, MLM_val_losses, title=model_name + "_mlm")
    utils.plot_values(NSP_train_losses, NSP_val_losses, title=model_name + "_nsp")

    print("Final MLM training loss: {:06.4f}".format(MLM_train_losses[-1]))
    print("Final MLM validation loss: {:06.4f}".format(MLM_val_losses[-1]))
    print("Final NSP training loss: {:06.4f}".format(NSP_train_losses[-1]))
    print("Final NSP validation loss: {:06.4f}".format(NSP_val_losses[-1]))

if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    random.seed(1234)
    torch.manual_seed(1234)

    #test_MLM_and_NSP_dataset()
    test_loss_calculation()
    pretrain_model()
