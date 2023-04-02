import torch
import os
import requests
from tqdm import tqdm

def get_shakespeare_text():
    text_local_path = './shakespeare.txt'
    if not os.path.exists(text_local_path):
        print('no local shakespeare file found, downloading ...')
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(text_local_path, 'w') as f:
            f.write(requests.get(data_url).text)
    return open(text_local_path, 'r').read()

def tokenize(text):
    char_set = sorted(list(set(text)))
    char_set += ['<S>'] # sentinel, marking beginning and ending
    char2index = {}
    index2char = {}
    for i, c in enumerate(char_set):
        char2index[c] = i
        index2char[i] = c
    return char2index, index2char