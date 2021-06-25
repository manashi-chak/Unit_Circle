import pandas as pd
import csv
import os
import sys
import torch
import pdb
import shutil
import pickle

def get_id_label_map(meta_file):

    identity_list = meta_file
    df = pd.read_csv(identity_list, sep=',\s+', quoting=csv.QUOTE_ALL, encoding="utf-8")

    key = df["Class_ID"].values
    val = df["class"].values
    id_label_dict = dict(zip(key, val))
    return id_label_dict

def get_num_label_map(meta_file):

    identity_list = meta_file
    df = pd.read_csv(identity_list, sep=',\s+', quoting=csv.QUOTE_ALL, encoding="utf-8")
    key =  df.loc[df["Flag"] == 0, "Class_ID"].values
    val = df.loc[df["Flag"] == 0, "Sample_Num"].values
    id_label_dict = dict(zip(key, val))
    return id_label_dict

def load_state_dict(model, fname):
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def accuracy_pairwise(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    #pdb.set_trace()
    
    maxk = 1
    
    batch_size = target.size(0)

    output_sorted, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []

    mylist=[ output_sorted.data.cpu().numpy().item(),pred.data.cpu().numpy().item()]
    return mylist
    

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    #pdb.set_trace()
    maxk = max(topk)
    #maxk = 1
    
    batch_size = target.size(0)

    output_sorted, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
