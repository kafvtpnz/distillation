# -*- coding: utf-8 -*-

import torch
import argparse
from models import Teacher

def bpred(text):
    
    teacher = Teacher()
    out = teacher.predict(text)
    _, pred = torch.max(torch.tensor(out), 1)
    
    print('Label is:', pred.cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="Съешь еще этих мягких французских булочек", 
                        help="sequence for classification")
        
    opt = parser.parse_args()
    print(opt)
    bpred(opt.text)