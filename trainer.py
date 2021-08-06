from __future__ import division
import datetime
from multiprocessing import Pool
import math
import os
import shutil
import psutil
import gc
import time

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import utils
import tqdm
import pdb


class Trainer(object):

    def __init__(self, cmd, cuda, model, criterion, optimizer, lr_scheduler,
                 train_loader,val_loader, log_file, max_iter,
                 interval_validate=None,						#val loader, checkpoint dir, log file urie diechi
                 checkpoint_dir=None, print_freq=1, save_freq=300):

        
        self.cmd = cmd
        self.cuda = cuda

        self.model = model
        self.criterion = criterion
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.timestamp_start = datetime.datetime.now()

        if cmd == 'train':
            self.interval_validate = len(self.train_loader) if interval_validate is None else interval_validate

        self.epoch = 0
        self.iteration = 0

        self.max_iter = max_iter
        self.best_top1 = 0
        self.min_loss =100
        self.print_freq = print_freq
        self.save_freq = save_freq
   

        self.checkpoint_dir = checkpoint_dir
        self.log_file = log_file
        
        

    def print_log(self, log_str):
        with open(self.log_file, 'a') as f:
            f.write(log_str + '\n')
        

    def train(self):

        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader))) # 117
        for epoch in tqdm.trange(self.epoch, max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            if self.epoch==100:
                
                for name, param in self.model.named_parameters():
                    param.requires_grad = True	

            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
    
    def classparser(self, target):
        n=len(target)
        i=0
        final=[]        
        while i<(n-1):
            class1=target[i]
            i=i+1
            class2=target[i]
            i=i+1
            if class1==class2:
                final.append(0)
            else:
                final.append(1)
                
        final=torch.FloatTensor(final)
        return final           


    def validate(self):

        batch_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        training = self.model.training
        self.model.eval()

        end = time.time()
        for batch_idx, (imgs, target, img_files, class_ids) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration={} epoch={}'.format(self.iteration, self.epoch), ncols=80, leave=False):
            
            gc.collect()
            target=self.classparser(target)
            if self.cuda:
                imgs, target = imgs.cuda(), target.cuda()
            imgs = Variable(imgs, volatile=True)
            target = Variable(target, volatile=True)
            #############batch divided into two############
            myshape=imgs.shape
            imgs1=imgs[0:myshape[0]:2,:,:]
            imgs2=imgs[1:myshape[0]:2,:,:]
            img1=imgs1.cuda()
            img2=imgs2.cuda()
            ###############################################
            output = self.model(imgs1,imgs2)
            
            loss = self.criterion(output, target)

            if np.isnan(float(loss.data.item())):
                raise ValueError('loss is nan while validating')

            losses.update(loss.data.item(), imgs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % self.print_freq == 0:
                log_str = 'Test: [{0}/{1}]\tepoch: {epoch:}\titer: {iteration:}\t' \
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    batch_idx, len(self.val_loader), epoch=self.epoch, iteration=self.iteration,
                    batch_time=batch_time, loss=losses,top1=top1)
                print(log_str)
                self.print_log(log_str)

        if self.cmd == 'train':
            is_min= losses.avg < self.min_loss
            self.min_loss = min(losses.avg,self.min_loss)

            log_str = 'Test_summary: [{0}/{1}] epoch: {epoch:} iter: {iteration:}\t' \
                  'Time: {batch_time.avg:.3f}\tLoss: {loss.avg:.4f}\t'.format(
                batch_idx, len(self.val_loader), epoch=self.epoch, iteration=self.iteration,
                batch_time=batch_time, loss=losses)
            print(log_str)
            self.print_log(log_str)

            checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint.pth.tar')
            torch.save({
                'epoch': self.epoch,
                'iteration': self.iteration,
                'arch': self.model.__class__.__name__,
                'optim_state_dict': self.optim.state_dict(),
                'model_state_dict': self.model.state_dict(),
                'best_top1': self.best_top1,
                'batch_time': batch_time,
                'losses': losses,
            }, checkpoint_file)
            if is_min:
                shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, 'model_best.pth.tar'))
            if (self.epoch + 1) % self.save_freq == 0: # save each 300 epoch
                shutil.copy(checkpoint_file, os.path.join(self.checkpoint_dir, 'checkpoint-{}.pth.tar'.format(self.epoch)))

            if training:
                self.model.train()

    def train_epoch(self):
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        
        
        self.model.train()
        self.optim.zero_grad()
   
        end = time.time()
        for batch_idx, (imgs, target, img_files, class_ids) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch={}, iter={}'.format(self.epoch, str(self.iteration)), ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            data_time.update(time.time() - end)
            gc.collect()
            
            target=self.classparser(target)
            
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if (self.iteration + 1) % self.interval_validate == 0:
                self.validate()

            if self.cuda:
                imgs, target= imgs.cuda(), target.cuda()       

            imgs, target= Variable(imgs), Variable(target)
            
            myshape=imgs.shape
            imgs1=imgs[0:myshape[0]:2,:,:]
            imgs2=imgs[1:myshape[0]:2,:,:]
            img1=imgs1.cuda()
            img2=imgs2.cuda()
            ###############################################
            output = self.model(imgs1,imgs2)
            loss = self.criterion(output, target)

            if np.isnan(float(loss.data.item())):
                raise ValueError('loss is nan while training')

            
            
            losses.update(loss.data.item(), imgs.size(0))

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if self.iteration % self.print_freq == 0:
                log_str = 'Train: [{0}/{1}]\tepoch: {epoch:}\titer: {iteration:}\t' \
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    batch_idx, 
                    len(self.train_loader), 
                    epoch=self.epoch, 
                    iteration=self.iteration,
                    batch_time=batch_time, 
                    data_time=data_time, 
                    loss=losses)
                print(log_str)
                self.print_log(log_str)
            

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # update lr
        log_str = 'Train_summary: [{0}/{1}]\tepoch: {epoch:}\titer: {iteration:}\t' \
                      'Time: {batch_time.avg:.3f}\tData: {data_time.avg:.3f}\t' \
                      'Loss: {loss.avg:.4f}'.format(
                    batch_idx, len(self.train_loader), epoch=self.epoch, iteration=self.iteration,
                    lr=self.optim.param_groups[0]['lr'],
                    batch_time=batch_time, data_time=data_time, loss=losses)
        print(log_str)
        self.print_log(log_str)

    
    
class Verifier_pairwise(object):

    def __init__(self, cmd, cuda,genuine_count,imposter_count,model, criterion, root, log_file, matching_list, match_loader, test_id_label_dict, test_num_label_dict, print_freq=1):
        
        self.cmd = cmd
        self.cuda = cuda
        self.model = model
        self.criterion = criterion
        self.root = root
        self.log_file1 = log_file
        self.matchinglist = matching_list
        self.tar_loader = match_loader
        self.num_label_dict = test_num_label_dict
        self.log_file1 = log_file
        
        self.print_freq = print_freq   

        self.genuine=genuine_count

        self.imposter=imposter_count

        self.far_ham=0
        self.frr_ham=0
        
    def print_log(self, log_str):
        with open(self.log_file1, 'a') as f:
            f.write(log_str + '\n')
            
    def classparser(self, target):
        n=len(target)
        i=0
        final=[]        
        while i<(n-1):
            class1=target[i]
            i=i+1
            class2=target[i]
            i=i+1
            if class1==class2:
                final.append(0)
            else:
                final.append(1)
                
        final=torch.LongTensor(final)
        return final 
        
    def verifier(self):
        batch_time = utils.AverageMeter()
        
        self.model.eval()
        #step= float(0.0001)
        #init= float(0.00)
        #thres =[]  #############################vals??
        #for i in range(0,10001):
            #thres.append(init+(float(i)*step))
        
        #print(thres)
        t=[]
        p=[]
        end = time.time()
        #print(torch.cuda.memory_summary(abbreviated=True))
        for batch_idx, (imgs, target, img_files, class_ids) in tqdm.tqdm(
            enumerate(self.tar_loader), total= len(self.tar_loader),ncols= 80, leave= False):
            gc.collect()
            target= self.classparser(target)
            #print(target)
            #pdb.set_trace()
            #print(batch_idx)
           # print(torch.cuda.memory_summary(abbreviated=True))
        #    if batch_idx == 20:
                #pdb.set_trace()
        #        break
            for x in target:
                t.append(x.detach().cpu().numpy().item())
                    #print(x)
            if self.cuda:
                imgs= imgs.cuda()
                #pdb.set_trace()
            with torch.no_grad():
                output,_ = self.model(imgs)
                #pdb.set_trace()
            for x in output:
                p.append(x.detach().cpu().numpy().item())
            del target, imgs, output, _
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        fpr, tpr, thresholds = metrics.roc_curve(t, p)
        for a,b,c in zip(fpr, 1- tpr, thresholds):
            print('Thres: ',str(c),', FAR_ham: ',str(b),', FRR_ham: ',str(a),'\n')
            log_str = f'Thres: {c},\tFAR_ham: {b},\tFRR_ham: {a}\n'
            self.print_log(log_str)
        #pdb.set_trace()
        return None

        batch_time = utils.AverageMeter()
        
        self.model.eval()
        step=float(0.01)
        init=float(0.00)
        thres =[]  
        for i in range(0,101):
            thres.append(init+(float(i)*step))

        t=[]
        p=[]
        end = time.time()
        for batch_idx, (imgs, target, img_files, class_ids) in tqdm.tqdm(
            enumerate(self.tar_loader), total=len(self.tar_loader),ncols=80, leave=False):
            gc.collect()
            target=self.classparser(target)
            for x in target:
                t.append(x.data.cpu().numpy().item())
            if self.cuda:
                imgs, target= imgs.cuda(), target.cuda()    

            imgs, target= Variable(imgs), Variable(target)

            myshape=imgs.shape
            with torch.no_grad():
                
                imgs1=imgs[0:myshape[0]:2,:,:]
                imgs2=imgs[1:myshape[0]:2,:,:]
                img1=imgs1.cuda()
                img2=imgs2.cuda()
                output = self.model(imgs1,imgs2)

            for x in output:
                p.append(x.data.cpu().numpy().item())
      
        for th in thres:
            self.far_ham=0
            self.frr_ham=0
            for i in range(len(p)):
                if (p[i]>=th):
                    if (t[i]==0):
                        self.frr_ham=self.frr_ham+1
                else:
                    if (t[i]==1):
                        self.far_ham=self.far_ham+1
            self.far_ham=(self.far_ham/self.imposter)
            self.frr_ham=(self.frr_ham/self.genuine)

            print('Thres: ',str(th),', FAR_ham: ',str(self.far_ham),', FRR_ham: ',str(self.frr_ham),'\n')
            log_str = 'Thres: {thres:},\tFAR_ham: {FAR_ham:},\tFRR_ham: {FRR_ham:}\n'.format(thres=th,FAR_ham=self.far_ham,FRR_ham=self.frr_ham)
            self.print_log(log_str)
