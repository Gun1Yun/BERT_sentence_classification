import re
from typing import List, Dict, Set
from itertools import chain
from collections import Counter, defaultdict

### You can import any Python standard libraries here.
### Do not import external library such as numpy, torchtext, etc.

### END YOUR LIBRARIES

def build_bpe(
    corpus: List[str],
    max_vocab_size: int
) -> List[int]:
    """ BPE Vocabulary Builder
    Implement vocabulary builder for byte pair encoding.
    Please sort your idx2word by subword length in decsending manner.

    Hint: Counter in collection library would be helpful

    Note: If you convert sentences list to word frequence dictionary,
          building speed is enhanced significantly because duplicated words are preprocessed together

    Arguments:
    corpus -- List of words to build vocab
    max_vocab_size -- The maximum size of vocab

    Return:
    idx2word -- Subword list
    """
    # Special tokens
    PAD = BytePairEncoding.PAD_token # Index of <PAD> must be 0
    UNK = BytePairEncoding.UNK_token # Index of <UNK> must be 1
    CLS = BytePairEncoding.CLS_token # Index of <CLS> must be 2
    SEP = BytePairEncoding.SEP_token # Index of <SEP> must be 3
    MSK = BytePairEncoding.MSK_token # Index of <MSK> must be 4
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]

    WORD_END = BytePairEncoding.WORD_END # Use this token as the end of a word

    ### YOUR CODE HERE (~22 lines)
    idx2word: List[str] = SPECIAL    
    
    # append word_end token every word in corpus
    corpus = [word+WORD_END for word in corpus]

    # create set of vocaburary(only character)
    vocabs =  []
    for word in corpus:
        vocabs += word
    vocabs = set(vocabs)

    # split all characters by spacing
    # count the number of each word with Counter class
    corpus = list(map(" ".join, corpus))
    counter = Counter(corpus)
    
    # iteration
    while True:
        # constraint of maximum vocab size
        if max_vocab_size <= len(vocabs)+len(SPECIAL):
            break
        
        # make pair counter dict
        pairs = defaultdict(int)
        for word, freq in counter.items():
            tokens = word.split()
            
            # make pair with side token
            for i in range(len(tokens)-1):
                pairs[(tokens[i], tokens[i+1])] += freq
        
        # no addition pair
        if not pairs:
            break
        
        # get the most frequent pair
        new_pair = max(pairs, key=pairs.get)
        new_token = "".join(new_pair)
        vocabs.add(new_token)
        
        # update new counter with added token
        new_counter = {}
        old_token = " ".join(new_pair)
        for word in counter:
            changed_word = word.replace(old_token, new_token)
            new_counter[changed_word] = counter[word]
        
        counter = new_counter
        
    result = list(vocabs)
    result = sorted(result, reverse=True, key=len)
        
    idx2word+=result

    ### END YOUR CODE

    return idx2word

def encode(
    sentence: List[str],
    idx2word: List[str]
) -> List[int]:
    """ BPE encoder
    Implement byte pair encoder which takes a sentence and gives the encoded tokens

    Arguments:
    sentence -- The list of words which need to be encoded.
    idx2word -- The vocab that you have made on the above build_bpe function.
    
    Return:
    tokens -- The list of the encoded tokens
    """
    WORD_END = BytePairEncoding.WORD_END

    ### YOUR CODE HERE (~10 lines)
    tokens: List[int] = None
    # preprocessing
    tokens = []
    sentence = [word+WORD_END for word in sentence]

    # for each word
    for word in sentence:
        # split word by space
        word = list(map(" ".join, word))
        word = " ".join(word)
        
        # for each splitted token
        while True:
            splited_word = word.split()
            pair_idx = {}
            
            # get pair of token that in the vocaburary
            for i in range(len(splited_word)-1):
                pair = (splited_word[i], splited_word[i+1])
                token = "".join(pair)
                if token in idx2word:
                    pair_idx[pair]=idx2word.index(token)
            
            # if no more pairs in there, stop iteration
            if not pair_idx:
                break
            
            # get max index of vocab for merge two tokens
            new_pair = max(pair_idx, key=pair_idx.get)
            new_token = "".join(new_pair)
            word = word.replace(" ".join(new_pair), new_token)
        
        # get list of char in encoding word
        splited_word = word.split()
        
        # append token index
        for token in splited_word:
            if token in idx2word:
                tokens.append(idx2word.index(token))
            else:
                tokens.append(1)
    ### END YOUR CODE

    return tokens

def decode(
    tokens: List[int],
    idx2word: List[str]
) -> List[str]:
    """ BPE decoder
    Implement byte pair decoder which takes tokens and gives the decoded sentence.

    Arguments:
    tokens -- The list of tokens which need to be decoded
    idx2word -- the vocab that you have made on the above build_bpe function.

    Return:
    sentence  -- The list of the decoded words
    """
    WORD_END = BytePairEncoding.WORD_END

    ### YOUR CODE HERE (~1 lines)
    sentence: List[str] = None
    decoded= ""

    for token in tokens:
        decoded += idx2word[token]

    # split with end token
    sentence = decoded.split(WORD_END)
    sentence = sentence[:-1]

    ### END YOUR CODE
    return sentence


#############################################
# Helper functions below. DO NOT MODIFY!    #
#############################################

class BytePairEncoding(object):
    """ Byte Pair Encoding class
    We aren't gonna use this class for encoding. Because it is too slow......
    We will use sentence piece Google have made.
    Thus, this class is just for special token index reference.
    """
    PAD_token = '<pad>'
    PAD_token_idx = 0
    UNK_token = '<unk>'
    UNK_token_idx = 1
    CLS_token = '<cls>'
    CLS_token_idx = 2
    SEP_token = '<sep>'
    SEP_token_idx = 3
    MSK_token = '<msk>'
    MSK_token_idx = 4

    WORD_END = '_'

    def __init__(self, corpus: List[List[str]], max_vocab_size: int) -> None:
        self.idx2word = build_bpe(corpus, max_vocab_size)

    def encode(self, sentence: List[str]) -> List[int]:
        return encode(sentence, self.idx2word)

    def decoder(self, tokens: List[int]) -> List[str]:
        return decode(tokens, self.idx2word)
    
#############################################
# Testing functions below.                  #
#############################################

def test_build_bpe():
    print ("======Building BPE Vocab Test Case======")
    PAD = BytePairEncoding.PAD_token
    UNK = BytePairEncoding.UNK_token
    CLS = BytePairEncoding.CLS_token
    SEP = BytePairEncoding.SEP_token
    MSK = BytePairEncoding.MSK_token
    WORD_END = BytePairEncoding.WORD_END

    # First test
    corpus = ['abcde']
    vocab = build_bpe(corpus, max_vocab_size=15)
    assert vocab[:5] == [PAD, UNK, CLS, SEP, MSK], \
        "Please insert the special tokens properly"
    print("The first test passed!")

    # Second test
    assert sorted(vocab[5:], key=len, reverse=True) == vocab[5:], \
        "Please sort your idx2word by subword length in decsending manner."
    print("The second test passed!")

    # Third test
    corpus = ['low'] * 5 + ['lower'] * 2 + ['newest'] * 6 + ['widest'] * 3
    vocab = set(build_bpe(corpus, max_vocab_size=24))
    assert vocab > {PAD, UNK, CLS, SEP, MSK, 'est_', 'low', 'newest_', \
                    'i', 'e', 'n', 't', 'd', 's', 'o', 'l', 'r', 'w', WORD_END} and \
           "low_" not in vocab and "wi" not in vocab and "id" not in vocab, \
           "Your bpe result does not match expected result"
    print("The third test passed!")

    # forth test
    corpus = ['aaaaaaaaaaaa', 'abababab']
    vocab = set(build_bpe(corpus, max_vocab_size=13))
    assert vocab == {PAD, UNK, CLS, SEP, MSK, 'aaaaaaaa', 'aaaa', 'abab', 'aa', 'ab', 'a', 'b', WORD_END}, \
           "Your bpe result does not match expected result"
    print("The forth test passed!")

    # fifth test
    corpus = ['abc', 'bcd']
    vocab = build_bpe(corpus, max_vocab_size=10000)
    assert len(vocab) == 15, \
           "Your bpe result does not match expected result"
    print("The fifth test passed!")

    print("All 5 tests passed!")

def test_encoding():
    print ("======Encoding Test Case======")
    PAD = BytePairEncoding.PAD_token
    UNK = BytePairEncoding.UNK_token
    CLS = BytePairEncoding.CLS_token
    SEP = BytePairEncoding.SEP_token
    MSK = BytePairEncoding.MSK_token
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]
    WORD_END = BytePairEncoding.WORD_END

    # First test
    vocab = SPECIAL + ['bcc', 'bb', 'bc', 'a', 'b', 'c', WORD_END]
    assert encode(['abbccc'], vocab) == [8, 9, 5, 10, 11], \
           "Your bpe encoding does not math expected result"
    print("The first test passed!")

    # Second test
    vocab = SPECIAL + ['aaaa', 'aa', 'a', WORD_END]
    assert len(encode(['aaaaaaaa', 'aaaaaaa'], vocab)) == 7, \
           "Your bpe encoding does not math expected result"
    print("The second test passed!")

    print("All 2 tests passed!")

def test_decoding():
    print ("======Decoding Test Case======")
    PAD = BytePairEncoding.PAD_token
    UNK = BytePairEncoding.UNK_token
    CLS = BytePairEncoding.CLS_token
    SEP = BytePairEncoding.SEP_token
    MSK = BytePairEncoding.MSK_token
    SPECIAL = [PAD, UNK, CLS, SEP, MSK]
    WORD_END = BytePairEncoding.WORD_END

    # First test
    vocab = SPECIAL + ['bcc', 'bb', 'bc', 'a', 'b', 'c', WORD_END]
    assert decode([8, 9, 5, 10, 11], vocab) == ['abbccc'], \
           "Your bpe decoding does not math expected result"
    print("The first test passed!")

    # Second test
    vocab = SPECIAL + ['aaaa', 'aa', 'a', WORD_END]
    assert decode([5, 5, 8, 5, 6, 7, 8], vocab) == ['aaaaaaaa', 'aaaaaaa'], \
           "Your BPE decoding does not math expected result"
    print("The second test passed!")

def test_consistency():
    print ("======Consistency Test Case======")
    corpus = ['this is test corpus .',
              'we will check the consistency of your byte pairing encoding .', 
              'you have to pass this test to get full scores .',
              'we hope you to pass tests wihtout any problem .',
              'good luck .']

    vocab = build_bpe(chain.from_iterable(sentence.split() for sentence in corpus), 80)
    
    sentence = 'this is another sentence to test encoding and decoding .'.split()
    
    assert decode(encode(sentence, vocab), vocab) == sentence, \
            "Your BPE does not show consistency."
    print("The consistency test passed!")

if __name__ == "__main__":
    test_build_bpe()
    test_encoding()
    test_decoding()
    test_consistency()
    