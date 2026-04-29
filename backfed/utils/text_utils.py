                                                                 

import os
import torch
import json
import re

filter_symbols = re.compile('[a-zA-Z]*')

def get_tokenizer():
    """
    This tokenizer:
    1. Splits text on whitespace
    2. Extracts only alphabetic characters from each token
    3. Filters out tokens that are too short
    
    Returns:
        A function that tokenizes text according to these rules
    """
    def tokenize(text):
                             
        words = text.lower().split()
        
                                                              
        filtered_words = []
        for word in words:
                                           
            alpha_only = filter_symbols.search(word)
            if alpha_only:
                alpha_word = alpha_only[0]
                                            
                if len(alpha_word) > 1:
                    filtered_words.append(alpha_word)
        
        return filtered_words
    return tokenize

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        raise ValueError("Please don't call this method, so we won't break the dictionary :) ")

    def __len__(self):
        return len(self.idx2word)

def get_word_list(line, dictionary):
    try:
                                                                 
        text = json.loads(line.lower())
    except json.JSONDecodeError:
                                          
        text = line.lower()
    
    splitted_words = text.split()
    words = ['<bos>']
    for word in splitted_words:
        match = filter_symbols.search(word)
        if match:
            word = match[0]
            if len(word) > 1:
                if dictionary["word2idx"].get(word, False):
                    words.append(word)
                else:
                    words.append('<unk>')
    words.append('<eos>')
    return words

def batchify(data, bsz):
                                                                    
    nbatch = data.size(0) // bsz
                                                                         
    data = data.narrow(0, 0, nbatch * bsz)
                                                    
    data = data.view(bsz, -1).t().contiguous()
    return data.cuda()   

def get_batches(data_source: torch.Tensor, batch_size: int, seq_length: int):
    """
    Generate all batches for a client by first batchifying the data and then
    creating input-target pairs with the specified sequence length.
    
    Args:
        data_source: List of token IDs
        batch_size: Batch size for batchifying the data
        seq_length: Sequence length for creating input-target pairs
        
    Returns:
        List of (data, target) tuples where each is a batch for training
    """
                                                 
    batched_data = batchify(data_source, batch_size)

    batches = []    
                                                       
    for i in range(0, batched_data.size(0) - 1, seq_length):
                                                                        
        seq_len = min(seq_length, batched_data.size(0) - 1 - i)
            
        data = batched_data[i:i + seq_len]
        target = batched_data[i + 1:i + 1 + seq_len].view(-1)
        
                                    
        if data.size(0) > 0:
            batches.append((data, target))
    
    return batches

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
