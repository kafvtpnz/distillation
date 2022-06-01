# -*- coding: utf-8 -*-

import torch
import pickle
import time
import os
import argparse


import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torchtext.vocab import vocab
from nltk.tokenize import RegexpTokenizer

from collections import Counter, OrderedDict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup

from models import Teacher, CNN


def distill(alpha, batch_size, epochs, lr, max_len):
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        print('Available GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print('No GPU available, using the CPU')
    
    teacher = Teacher()
    
    print(teacher.predict('Съешь еще этих мягких французских булочек'))
    
    df = pd.read_csv("./data/labeled_rutoxic.csv", delimiter=',', 
                     header=0, names=['sentence', 'label'])
    
    
    sentences = df.sentence.values
    labels = df.label.values
    
    tokenizer = RegexpTokenizer(r'\w+')
    
    V=[]
    for j in sentences:
        L = [i for i in tokenizer.tokenize(j.lower())]
        V.extend(L)
    
    counter = Counter(V)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    voc = vocab(ordered_dict)
    voc.set_default_index(0)
    torch.save(voc, './data/cache/vocabulary.pth')
    v_size = len(voc)
    
    V=[]
    for j in sentences:
        L = voc(tokenizer.tokenize(j.lower()))
        V.append(torch.tensor(L))
    
    labl = [int(i) for i in labels]
    sent = [i[:max_len] for i in V]
    
    sent = nn.utils.rnn.pad_sequence(sent, batch_first = True)
    
    
    random_state=42
    tr_inp, val_inp, tr_labl, val_labl = train_test_split(sent, 
                                                          torch.tensor(labl), 
                                                          test_size=0.1, random_state=random_state)
    tr_sent, val_sent, _, _ = train_test_split(sentences, torch.tensor(labl),
                                               test_size=0.1, random_state=random_state)
    
    if len(os.listdir('./data/cache/')) == 0:
        with torch.no_grad():
            teachout_tr = np.vstack([teacher.predict(text) for text in tqdm(tr_sent)])
            teachout_vl = np.vstack([teacher.predict(text) for text in tqdm(val_sent)])
        with open('./data/cache/teachout_tr','wb') as fout: pickle.dump(teachout_tr,fout)
        with open('./data/cache/teachout_vl','wb') as fout: pickle.dump(teachout_vl,fout)
    else:
        with open('./data/cache/teachout_tr', 'rb') as fin:
            teachout_tr = pickle.load(fin)
        with open('./data/cache/teachout_vl', 'rb') as fin:
            teachout_vl = pickle.load(fin)
    
      
    train_data = TensorDataset(tr_inp, torch.tensor(teachout_tr), tr_labl)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, 
                                  batch_size=batch_size)
    
    
    validation_data = TensorDataset(val_inp, torch.tensor(teachout_vl), val_labl)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, 
                                       batch_size=batch_size)
    
    model = CNN(v_size,max_len,128,2)
    
    if device.type == 'cuda':
        model.cuda()
    
    
    total_steps = len(train_dataloader) * epochs  
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps = 0, 
                                                num_training_steps = total_steps)
    
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    
    
    for epoch in range(epochs):
        losses = []
        acc = []
        teachr_acc = []
        model.train()
        t0 = time.time()
        print("")
        print('Эпоха {:} из {:} '.format(epoch + 1, epochs))
        for step, batch in enumerate(train_dataloader):
            model.zero_grad()
    
            data = batch[0].to(device)
            lb = batch[2].to(device)
            pred = model(data)
            loss = alpha * ce_loss(pred, lb) + (1-alpha) * mse_loss(pred, batch[1].to(device))  
            loss.backward()
            opt.step()
            scheduler.step()
            losses.append(loss.item())
            
            if step % 200 == 0 and not step == 0:
                time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))
                print(' Батч {:>4,} из {:>4,}. Затраченное время: {:}. Loss: {:}.'.format(step, len(train_dataloader), time_elapsed, loss))
      
        model.eval()
        print('Валидация...')
        with torch.no_grad():
            for step, batch in enumerate(validation_dataloader):
                data = batch[0].to(device)
                lb = batch[2].to(device)
                _, teach_pred = torch.max(batch[1].to(device), 1)
                _, pred = torch.max(model(data), 1)
                acc.append((pred == lb).float().mean().item())
                teachr_acc.append((teach_pred == lb).float().mean().item())
                
        print('Mean_loss: {}, Accuracy: {}, Teacher_accuracy: {}'.format(np.mean(losses), 
                                                                         np.mean(acc), 
                                                                         np.mean(teachr_acc)))
        
    torch.save(model, './model_cnn/CNNet.ckpt')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha in loss function")
    parser.add_argument("--batch_size", type=int, default=4, help="size of batch")
    parser.add_argument("--epochs", type=int, default=10, help="epochs to train")
    parser.add_argument("--lr", type=float, default=0.002, help="start learning rate")
    parser.add_argument("--max_len", type=int, default=50, help="length of sequence")
        
    opt = parser.parse_args()
    print(opt)
    distill(opt.alpha, opt.batch_size, opt.epochs, opt.lr, opt.max_len)