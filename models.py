# -*- coding: utf-8 -*-
"""
Created on Tue May 31 21:59:20 2022

@author: AM4
"""
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# модель-донор
class Teacher(object):
    def __init__(self, max_length = 128, dir_pretrained = './model_bert/'):
        self.max_length = max_length
        
        self.tokenizer = BertTokenizer.from_pretrained(dir_pretrained)

        self.modelb = BertForSequenceClassification.from_pretrained(dir_pretrained)
        self.modelb.eval()
        if torch.cuda.is_available():
            self.modelb.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
    def predict(self, text):
        enc_s = self.tokenizer.encode(text,                      
                                add_special_tokens = True,
                                padding = 'max_length',
                                max_length = self.max_length,
                                truncation = True)

        input_ids = np.array(enc_s)
        attention_mask = [int(id_ > 0) for id_ in input_ids]
        batch = tuple(t.to(self.device) for t in torch.Tensor(np.array([input_ids, attention_mask])))
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            outputs = self.modelb(b_input_ids.unsqueeze(0).to(torch.long), token_type_ids=None, attention_mask=b_input_mask.unsqueeze(0))

        return torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu().numpy()

# модель-реципиент
class CNN(nn.Module):
    def __init__(self, v_size, e_size, hidden, output):
        super(CNN, self).__init__()
        self.emb = nn.Embedding(v_size, e_size, padding_idx=0)
        self.dropout = nn.Dropout(0.3)
        self.conv1 = nn.Conv2d(1, hidden, (3, e_size))
        self.conv2 = nn.Conv2d(1, hidden, (4, e_size))
        self.conv3 = nn.Conv2d(1, hidden, (5, e_size))
        self.conv4 = nn.Conv2d(1, hidden, (7, e_size))
        self.fc = nn.Linear(hidden * 4, output)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embed = self.dropout(self.emb(x)).unsqueeze(1)
        c1 = torch.relu(self.conv1(embed).squeeze(3))
        p1 = torch.max_pool1d(c1, c1.size()[2]).squeeze(2)
        c2 = torch.relu(self.conv2(embed).squeeze(3))
        p2 = torch.max_pool1d(c2, c2.size()[2]).squeeze(2)
        c3 = torch.relu(self.conv3(embed).squeeze(3))
        p3 = torch.max_pool1d(c3, c3.size()[2]).squeeze(2)
        c4 = torch.relu(self.conv4(embed).squeeze(3))
        p4 = torch.max_pool1d(c4, c4.size()[2]).squeeze(2)

        pool = self.dropout(torch.cat((p1, p2, p3, p4), 1))
        hidden = self.fc(pool)
        return self.softmax(hidden)