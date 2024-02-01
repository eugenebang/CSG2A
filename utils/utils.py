import numpy as np
import torch
from datetime import datetime

import os
from datetime import datetime

class Logger:
    def __init__(self, model_name):
        self.model_name=model_name
        self.date=str(datetime.now().date()).replace('-','')[2:]
        if not os.path.exists('log'):
            os.mkdir('log')
        self.logger_file = f'log/{self.date}_{self.model_name}'
        
    def __call__(self, text, verbose=True, log=True):
        if log:
            with open(f'{self.logger_file}.log', 'a') as f:
                f.write(f'[{datetime.now().replace(microsecond=0)}] {text}\n')
        if verbose:
            print(f'[{datetime.now().time().replace(microsecond=0)}] {text}')

import os
import random
import numpy as np
import torch

def set_seed(seed=42,logger=print, load_torch = True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if load_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger(f'random seed with {seed}')



class EarlyStopper:
    def __init__(self, patience=7, printfunc=print, verbose=True, delta=0, path='checkpoint.pt'):
        self.printfunc=printfunc
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.printfunc(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.printfunc(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

        
dose2int={
    0.5: 0,
    1.0: 1,
    10.0: 2,
    3.0: 3,
    5.0: 4,
    0.1: 5,
    60.0: 6,
    100.0: 7,
    40.0: 8,
    30.0: 9,
    20.0: 10,
    80.0: 11,
    90.0: 12,
    70.0: 13,
    50.0: 14,
    0.01: 15,
    0.001: 16,
    0.04: 17,
    0.12: 18,
    0.37: 19,
    1.11: 20,
    3.33: 21,
    0.41: 22,
    1.23: 23,
    11.11: 24,
    3.7: 25,
    33.33: 26
}

int2dose={0: 0.5,
     1: 1.0,
     2: 10.0,
     3: 3.0,
     4: 5.0,
     5: 0.1,
     6: 60.0,
     7: 100.0,
     8: 40.0,
     9: 30.0,
     10: 20.0,
     11: 80.0,
     12: 90.0,
     13: 70.0,
     14: 50.0,
     15: 0.01,
     16: 0.001,
     17: 0.04,
     18: 0.12,
     19: 0.37,
     20: 1.11,
     21: 3.33,
     22: 0.41,
     23: 1.23,
     24: 11.11,
     25: 3.7,
     26: 33.33
}

time2int={6:0,
         24:1,
         48:2
         }

int2time={
    0:6,
    1:24,
    2:48
}