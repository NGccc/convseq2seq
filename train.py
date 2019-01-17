# -*- coding: utf-8 -*-
import config
import torch 
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from preprocess import mydataset

#from BiLSTM_Attention_model import BLSTMA
#from mydataset import MyDataset
from convseq2seq import convseq2seq

from tensorboardX import SummaryWriter
import os
import pdb
import math

def hook(module, input, output):
    print(output[0].shape)
    print(output[0].data)


opts = config.opts
train_dataset = mydataset()
train_loader  = train_dataset.getIterator()
#model = convseq2seq()

# ==== tensorboard ====
writer = SummaryWriter(log_dir='logs_adam')

pre_model_path  = opts.model
epochs          = opts.epochs
device          = opts.DEVICE
lr              = opts.lr


if pre_model_path:
    if os.path.exists(pre_model_path):
        model = torch.load(pre_model_path)
        #model.load_state_dict(torch.load(pre_model_path))
        #torch.save(model,'./eye_brow_model.pth')
    else:
        print('error.check ur model path')
        exit()

model.to(device)
loss_func_c = nn.CrossEntropyLoss(reduce=False)

optimizer = optim.Adam(model.parameters(), lr=lr)

#when train
dloader  = [train_loader, ]
is_train = 1


tag = ['val', 'train']
for epoch in range(epochs):
    if (epoch + 1) % 100 == 0:
        lr = lr / 3.0
    running_loss = torch.tensor([0.0, 0.0]).to(device)
    running_acc = torch.tensor([0.0, 0.0]).to(device)
    running_total_len = torch.tensor([0.0, 0.0]).to(device)
    step = [1,1]
    print('epoch:[%d/%d]' % (epoch, epochs))
    for dataloader in dloader:
        if is_train:
            model.train()
        else:
            model.eval()
        for i, data in enumerate(dataloader, 1):
            x, ppx = data.src
            px = torch.zeros(x.shape)
            y, ppy = data.trg
            py = torch.zeros(y.shape)
            #print(x,ppx,y,ppy)
            mask1 = torch.zeros(x.shape)
            mask2 = torch.zeros((y.shape[0], y.shape[1]-1))
            sentence_len = (ppy.sum()-y.shape[0]).float()
            for idx in range(x.shape[0]):
                px[idx, :ppx[idx]] = torch.arange(0, ppx[idx])
                mask1[idx, :ppx[idx]] = torch.ones(ppx[idx].item())
            for idx in range(y.shape[0]):
                py[idx,:ppy[idx]] = torch.arange(0, ppy[idx])
                mask2[idx, :ppy[idx]-1] = torch.ones(ppy[idx].item()-1)
            x,px,y,py,mask1,mask2 = x.long().to(device),px.long().to(device),y.long().to(device),py.long().to(device),mask1.long().to(device),mask2.float().to(device)
            out  = model(x,px,y,py,mask1)
            loss = (loss_func_c(out[:,:-1,:].reshape((out[:,:-1,:].shape[0]*out[:,:-1,:].shape[1],-1)),y[:,1:].reshape(-1)).reshape((y[:,1:].shape[0],-1)) * mask2).sum().to(device)
            
            #import pdb
            #pdb.set_trace()
            _, acc = torch.max(out[:,:-1,:], 2)
            acc = ((acc == y[:,1:]).float() * mask2).sum()
            
            #wrong = (acc != y[:,1:]).float() * mask2
            #if wrong.sum() > 0:
            #    print(wrong)
            
            running_loss[is_train] += loss
            running_total_len[is_train] += sentence_len
            running_acc[is_train] += acc
            #log and print
            print('[%s]step %d: loss:%f| acc:%f%% lr:%f' % (tag[is_train], step[is_train], loss / sentence_len, 100.0 * acc / sentence_len, lr))
            writer.add_scalar('data/[%s]_loss' % tag[is_train], loss / sentence_len, step[is_train])
            writer.add_scalar('data/[%s]_class_acc' % tag[is_train], 100.0 * acc / sentence_len, step[is_train])

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #save model
            step[is_train] += 1
        
        #save model
        if is_train and (epoch + 1) % opts.save_frequence == 0:
            print('[%s]epoch [%d/%d] finish, loss:%f | acc:%f %%' % (tag[is_train],
            epoch, epochs, running_loss[is_train] / running_total_len[is_train].float(), 
            100.0 * (running_acc[is_train] / running_total_len[is_train].float()
            )))

            #torch.save(model.state_dict(), 
            torch.save(model,
            '{}/epoch_{}_acc_{:.2f}_loss_{:.2f}_model.pth'.format(
            opts.save_model,
            epoch+1,
            running_acc[is_train].float() / running_total_len[is_train].float() * 100.0,
            running_loss[is_train] / running_total_len[is_train].float(), 
        ))
        #is_train = 1 - is_train  