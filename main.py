#########################
##   Same Class = 0    ##
## Different Class = 1 ##
#########################

#!/usr/bin/env python
from __future__ import print_function, division
import argparse
import os
import sys
import torch
import utils as ut
import torch.nn as nn
import Dataloader as cc

import UC_pairwise_radim as IrisNet
import torch.optim as optim
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np 
import pdb

from trainer import Trainer
from trainer import Verifier_pairwise



configurations = {
    1: dict(
        max_iteration=100000000000,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.01,
        gamma=0.1, # "lr_policy: step"
        step_size=1000000, # "lr_policy: step"
        interval_validate=1000,
    )
}

gen_imp={
    1:dict(
        genuine_count=14791,
        imposter_count=5743130
    ),
    2:dict(
        genuine_count=4478,
        imposter_count=848993
    ),
    3:dict(
        genuine_count=2098,
        imposter_count=550728
    )
}


             
N_IDENTITY = 1 #########Number of output classes for architecture. Here it is 1 since the output class is binary############

print('####################################################Options for Iris Recognition####################################################')

#Given below are the options to be given as input from command line:
#Mode of operation: Training or testing ; Data roots and files; Batch size for training and testing;
#Learning rate, momentum, regularisation decay factor,gpu to use, frequency of saving the checkpoint


cmd=input('Operation to Perform: train/test. All Small. Default: Train ')
if cmd=='':
    cmd= 'train'

root= input('Enter Root Directory. (Full Path). Default: /home/manashi/unit_circle_layer/ndiris/ ')
if root=='':
    root= '/home/manashi/radim/Unit_circle/'

chkdir=input('Name of checkpoint directory to load/create. Best checkpoint will be taken. (Name only) Default: chkpt_pairwise_unit_circle_layer ')
if chkdir=='':
    chkdir='chkpt_pairwise_unit_circle_layer'

data_root= input('Enter Root Directory to load Database from. (Full Path). Default: /home/manashi/ndiris_0405/ ')
if data_root=='':
    data_root='/home/manashi/Iris_Databases/ndiris_0405/'

if cmd=='train':
    train_img_list_file=input('Name of Train Image List (Name only) Default: train_left.txt ')
    if train_img_list_file=='':
        train_img_list_file='train_pairwise_left.txt'

    test_img_list_file=input('Name of Validation Image List (Name only) Default: validation_left.txt ')
    if test_img_list_file=='':
        test_img_list_file='validation_pairwise_left.txt'

    meta_file=input('Name of Meta File (Name only) Default: identity_meta_only_left.csv ')
    if meta_file=='':
        meta_file='identity_meta_train.csv'

elif (cmd=='test'):
    matching_list=input('Name of Matching Image List. (Name only) Default: test_right.txt  ')
    if matching_list=='':
        matching_list='test_pairwise_right.txt'
    
    meta_file=input('Name of Meta File (Name only) Default: identity_meta_only_right.csv ')
    if meta_file=='':
        meta_file='identity_meta_test.csv'
        
    
if cmd=='train':
    log_file=input('Name of Log File (Name only) Default: train_unit_circle_layer.log ')
    if log_file=='':
        log_file='train_unit_circle_layer.log'
else:
    log_file=input('Name of Log File (Name only) Default: test_unit_circle_layer.log ')
    if log_file=='':
        log_file='test_unit_circle_layer.log'

if cmd=='train':
    Batch_Sz=input('Batch Size for Train Mode (Name only) Default: 32 ')
    if Batch_Sz=='':
        Batch_Sz=32
    else:
        Batch_Sz=int(Batch_Sz)
else: 
    Batch_Sz=512

learning_rate=input('Enter Learning Rate for Training the Network. Default: 0.0001 ')
if learning_rate=='':
    learning_rate=0.01
else:
    learning_rate=float(learning_rate)

momentum=input('Enter Momentum for Training the Network. Default: 0.9 ')
if momentum=='':
    momentum=0.9
else:
    momentum=float(momentum)

wt_decay_factor=input('Enter Regularizer Parameter for Network (0 for no regularizer). Default: 0.0 ')
if wt_decay_factor=='':
    wt_decay_factor=0.01
else:
    wt_decay_factor=float(wt_decay_factor)


workers=input('Number of Workers(Number only) Default Value: 5. ')
if workers=='':
    workers=5
else:
    workers=int(workers)
    
gpu_touse=input('Enter GPU Number. Default Value: 1 ')
if gpu_touse=='':
    gpu_touse=str(0)

count=input('Enter 1 for NDIRIS-0405 dataset, 2 for Casia Interval dataset, 3 for IITD')
if count=='':
    count=1

print_freq=input('Please Enter the number of iterations after which status should be printed. Default Value: 1 ')
if print_freq=='':
    print_freq=500
else:
    print_freq=int(print_freq)
    
save_freq=input('Please Enter the number of epochs after which checkpoint should be saved. Default Value: 300 ')
if save_freq=='':
    save_freq=300
else:
    save_freq=int(save_freq)

resume=input('Path of checkpoint directory to resume from. (Full Path required) Optional field. ')

print('...')
print('...')
print('####################################################Inputs Received. Showing Entered Inputs.####################################################')
print('Operation: {}'.format(cmd))
print('Root Directory: {}'.format(root))
print('Database Root Directory: {}'.format(data_root))
print('Checkpoint filename to use for {} mode: {}'.format(cmd,chkdir))
if cmd=='train':
    print('Train Image List: {}'.format(train_img_list_file))
    print('Test Image List: {}'.format(test_img_list_file))
    print('Meta File: {}'.format(meta_file))
    
if (cmd=='test'):
    print('Matching Image List.: {}'.format(matching_list))
    print('Meta File: {}'.format(meta_file))
    

if cmd!='save':
    print('Log File: {}'.format(log_file))

print('Batch Size: {}'.format(Batch_Sz))

print('Learning Rate: {}'.format(learning_rate))
print('Momentum: {}'.format(momentum))
print('Regularizer: {}'.format(wt_decay_factor))
print('Number of Workers: {}'.format(workers))
print('GPU to Use: {}'.format(gpu_touse))
print('Count of Dataset to Use: {}'.format(count))
print('Print after {} Iterations.'.format(print_freq))
print('Save checkpoint after {} Epochs.'.format(save_freq))
print('Resume Directory: {}'.format(resume))
print('####################################################End.####################################################')
go=input('Proceed? (y/n). Default: Y ')
if go=='n':
    print('Process Exited')
    sys.exit()
    

cfg = configurations[1]

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_touse
cuda = torch.cuda.is_available()
if cuda:
    print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))
    
torch.manual_seed(1337)
if cuda:
    torch.cuda.manual_seed(1337)

include_top = False 
kwargs = {'num_workers': workers, 'pin_memory': True} if cuda else {}

config_num=1
model = IrisNet.Radim(num_classes=N_IDENTITY, include_top=include_top) 


if cmd=='train':
    checkpoint_dir=root+chkdir+'/'     
    ut.create_dir(checkpoint_dir)
    cfg = configurations[config_num]
    log_file = root+log_file 
    train_img_list_file = data_root+train_img_list_file
    target_img_list_file = data_root+test_img_list_file
    meta_file= data_root+meta_file
    nonclass_id_label_dict = ut.get_id_label_map(meta_file)
    num_label_dict = ut.get_num_label_map(meta_file)
    dt = cc.IrisLoader(data_root, train_img_list_file, nonclass_id_label_dict, split='train')
    train_loader = torch.utils.data.DataLoader(dt, batch_size=Batch_Sz, shuffle=False, **kwargs)
    dv = cc.IrisLoader(data_root, target_img_list_file, nonclass_id_label_dict, split='valid')
    val_loader = torch.utils.data.DataLoader(dv, batch_size=Batch_Sz, shuffle=False, **kwargs)

elif cmd=='test':
    pathtoload=root+chkdir+'/model_best.pth.tar'
    checkpoint = torch.load(pathtoload)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    log_file = root+log_file   
    matching_list = data_root+matching_list
    meta_file= data_root+meta_file
    nonclass_id_label_dict = ut.get_id_label_map(meta_file)
    num_label_dict = ut.get_num_label_map(meta_file)
    dm = cc.IrisLoader(data_root, matching_list, nonclass_id_label_dict, split='train')
    match_loader = torch.utils.data.DataLoader(dm, batch_size=Batch_Sz, shuffle=False, **kwargs)
    

start_epoch = 0
start_iteration = 0

if resume:
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    start_epoch = 0
    start_iteration = 0
    print("Resume from epoch: {}, iteration: {}".format(start_epoch, start_iteration))

 
if cuda:
    model = model.cuda()
    
criterion = nn.BCELoss()
if cuda:
    criterion = criterion.cuda()

#####################freezing unit circle layer for first 100 epochs################################

for name, param in model.named_parameters():
    if name=='bn1.weight':
        break
    else:
        param.requires_grad = False	


for name, param in model.named_parameters():        
    if param.requires_grad: 
        print(name)


        
gen_imp_count=gen_imp[count]


if cmd == 'train':
    temp = model.parameters()
    optim = optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=wt_decay_factor)
    last_epoch = start_iteration if resume else -1
    last_epoch = -1
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim,  cfg['step_size'],
                                                   gamma=cfg['gamma'], last_epoch=last_epoch)


if cmd == 'train':
    trainer = Trainer(
        cmd=cmd,
        cuda=cuda,
        model=model,
        criterion=criterion,
        optimizer=optim,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        log_file=log_file,
        max_iter=cfg['max_iteration'],
        checkpoint_dir=checkpoint_dir,
        print_freq=print_freq,
        save_freq=save_freq
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()
    

if cmd == 'test':
    validator=Verifier_pairwise(
        cmd=cmd,
        cuda=cuda,
        genuine_count=gen_imp_count['genuine_count'],
        imposter_count=gen_imp_count['imposter_count'],
        model=model,
        criterion=criterion,
        root=data_root,
        log_file=log_file,
        matching_list=matching_list,
        match_loader=match_loader,
        test_id_label_dict=nonclass_id_label_dict,
        test_num_label_dict=num_label_dict,
        print_freq=1,
    )
    validator.verifier()