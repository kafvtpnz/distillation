# -*- coding: utf-8 -*-
import torch
import argparse

import torch.nn as nn
from nltk.tokenize import RegexpTokenizer

def pred(text, max_len):
    if torch.cuda.is_available():    
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU')
    
    voc = torch.load('./data/cache/vocabulary.pth')
    
    tokenizer = RegexpTokenizer(r'\w+')
    V=[torch.tensor([i for i in range(max_len)])]
    L = voc(tokenizer.tokenize(text.lower()))
    V.append(torch.tensor(L))
    
    sent = [i[:max_len] for i in V]
    
    sent = nn.utils.rnn.pad_sequence(sent, batch_first = True)
    
      
    
    model = torch.load('./model_cnn/CNNet.ckpt')
    model.eval()
    if device.type == 'cuda':
        model.cuda()
    
    data = sent[1].unsqueeze(0).to(device)
    _, pred = torch.max(model(data), 1)
    print('Label is:', pred.cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=50, help="length of sequence")
    parser.add_argument("--text", type=str, default="Съешь еще этих мягких французских булочек", 
                        help="sequence for classification")
        
    opt = parser.parse_args()
    print(opt)
    pred(opt.text, opt.max_len)