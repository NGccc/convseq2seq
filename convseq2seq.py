# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np
import pdb
from config import opts
import torch.nn.functional as F

'''
class OPTS(object):
    def __init__(self):
        super(OPTS, self).__init__()
        self.source_vocab_size = 100
        self.target_vocab_size = 80
        self.posi_length       = 15
        self.embed_dim         = 20
        self.kernel_size       = 3
        self.source_max_length = 10
        self.target_max_length = 12

        self.econv_num         = 3
        self.dconv_num         = 3
'''
class convseq2seq(nn.Module):
    def __init__(self):
        super(convseq2seq, self).__init__()
        self.src_vocab_size    = opts.src_vocab_size
        self.trg_vocab_size    = opts.trg_vocab_size
        self.posi_length       = opts.max_len    
        self.smax              = opts.max_len  
        self.tmax              = opts.max_len #frame to frame
        self.econv_num         = opts.encoder_layer_num
        self.dconv_num         = opts.decoder_layer_num

        self.embed_dim      = opts.embed_dim
        self.src_word_embedding = nn.Embedding(self.src_vocab_size, self.embed_dim)
        self.trg_word_embedding = nn.Embedding(self.trg_vocab_size, self.embed_dim)
        
        self.posi_embedding = nn.Embedding(self.posi_length, self.embed_dim)
        self.ekernel_size    = opts.encoder_kernel_width
        self.dkernel_size    = opts.decoder_kernel_width
        self.channel         = self.embed_dim * 2

        self.decoder_pad    = nn.ConstantPad2d((0, 0, self.dkernel_size - 1, 0), 0) #pad on top
        self.encoder_pad    = nn.ConstantPad2d((0, 0, int(self.ekernel_size / 2), int(self.ekernel_size / 2)), 0) #pad on both top and bottom
        self.sig            = nn.Sigmoid()
        
        #[batch_size, channel, height, width]
        self.econv_blocks   = [
            nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=(self.ekernel_size, self.embed_dim), stride=1)
            for i in range(self.econv_num)
        ]
        
        self.dconv_blocks   =[
            nn.Conv2d(in_channels=1, out_channels=self.channel, kernel_size=(self.dkernel_size, self.embed_dim), stride=1)
            for i in range(self.dconv_num)
        ]
        
        self.aconv_blocks    = [
            nn.Conv2d(in_channels=1, out_channels=self.embed_dim, kernel_size=(self.dkernel_size, self.embed_dim), stride=1) 
            for i in range(self.dconv_num)
        ]
        
        self.Attention      = self.AttentionLayer
        self.FC             = nn.Linear(self.embed_dim, self.trg_vocab_size)
    
    def encoder(self, x):
        #===========block1=============
        #[batch_size, source_sentence_length, embedding_size] => [batch_size, 1, int(k/2) + source_sentence_length + int(k/2), embedding_size]
        xlst = [x,]
        for i in range(self.econv_num):
            x1 = self.encoder_pad(xlst[i]).unsqueeze(1)
            x1 = self.econv_blocks[i].to(opts.DEVICE)(x1)
            #[batch_size, 2*embedding_size, source_sentence_length, 1]
            x1 , y1 = torch.split(x1, self.embed_dim, dim=1)
            #[batch_size, embedding_size, source_sentence_length, 1] [batch_size, embedding_size, source_sentence_length, 1] 
            x1 = x1 * self.sig(y1)
            #[batch_size, embedding_size, source_sentence_length, 1]

            x1 = x1.permute((0,3,2,1))
            #[batch_size, 1, source_sentence_length, embedding_size]
            #residual
            x1 = x1.squeeze() + xlst[i]
            xlst.append(x1)
            #[batch_size, source_sentence_length, embedding_size]
        return xlst[-1]
    
    def decoder(self, x, zu, e_s, e_t, mask1, mask2):
        #input: [batch_size, target_sentence_length, embedding_size]

        xlst = [x,]
        for i in range(self.dconv_num):
            #============== block i ====================
            x1 = self.decoder_pad(xlst[i]).unsqueeze(1)
            #[batch_size, 1, target_sentence_length, embedding_size]
            #padding to [batch_size, 1, (self.kernel_size - 1) + target_sentence_length, embedding_size]
            x1 = self.dconv_blocks[i].to(opts.DEVICE)(x1)
            #[batch_size, 2*embedding_size, target_sentence_length, 1]
            x1 , y1 = torch.split(x1, self.embed_dim, dim=1)
            #[batch_size, embedding_size, target_sentence_length, 1] [batch_size, embedding_size, target_sentence_length, 1] 
            
            x1 = x1 * self.sig(y1)
            x1 = x1.permute(0,3,2,1)
            #[batch_size, 1, target_sentence_length, embedding_size]
            c1 = self.Attention(x1, zu, e_s, e_t, mask1, mask2, i)
            x1 = x1.squeeze(1) + c1 
            xlst.append(x1)
        out = self.FC(xlst[-1])
        return out

    def AttentionLayer(self, h, zu, e_s, e_t, mask1, mask2, l):
        '''
        h:
            decoder hl
            shape : [batch_size, target_sentence_length, embedding_size]
        
        zu:
            encoder hL
            shape : [batch_size, source_sentence_length, embedding_size]
        
        e_s:
            source word embedding
            shape : [batch_size, source_sentence_length, embedding_size]

        e_t:
            target word embedding
            shape : [batch_size, target_sentence_length, embedding_size]
        l:
            layer number range [1, block_num]
            type : int
        
        Attention:
        dot product
        '''

        h  = self.decoder_pad(h)
        h  = self.aconv_blocks[l].to(opts.DEVICE)(h).squeeze(-1)
        #[batch_size, target_sentence_length, embedding_size]
        
        d = (h.transpose_(1,2) + e_t)
        #d[l,i] = W[l] * h[l,i] + b[l] + g[i]
        
        score = torch.matmul(d, zu.permute(0, 2, 1))
        #[batch_size, target_sentence_length, source_sentence_length]
        score = score * mask1 + mask2

        sf = nn.Softmax(dim=2)
        a  = sf(score) 
        #[batch_size, target_sentence_length, source_sentence_length]

        #[batch_size, 1, 1] * [batch_size, tmax, embedding_size]
        #c = mask1.sum(2).unsqueeze(2).mean(1).unsqueeze(2) * torch.matmul(a, (zu + e_s)) #broadcast matrix multiply
        c = torch.matmul(a, (zu + e_s))
        # 3.4 multiply by m to scale up the inputs to their original size
        # [batch_size, target_sentence_length, embedding_size]
        return c
    
    def forward(self, x, px, y, py, mask):
        # position embedding + word embedding
        x   = self.src_word_embedding(x)
        px  = self.posi_embedding(px)
        x   = x + px
        e_s = x
        #[batch_size, sentence_length, embedding_size]

        zu  = self.encoder(x)
        
        #encoder last layer output zu:[batch_size, sentence_length, embedding_size]
        
        y   = self.trg_word_embedding(y)
        py  = self.posi_embedding(py)
        y   = y + py
        e_t = y
        #[batch_size, sentence_length, embedding_size]

        #for attention softmax
        
        mask = mask.unsqueeze(1).repeat(1, y.shape[1], 1).float()
        mask_p = torch.zeros(mask.shape).float().to(opts.DEVICE)
        mask_p[mask==0] = -999
        out  = self.decoder(y, zu, e_s, e_t, mask, mask_p)
        return out

if __name__ == '__main__':
    opts = OPTS()
    model = convseq2seq(opts)
    batch_size   = 2
    source_sentence_len = 7
    targe_sentence_len  = 8
    t1 = [[1,2,1,1,9,0,0,0,0,0],
         [23,52,12,32,32,11,33,0,0,0]]

    t2 = [[76,42,42,23,11,3,22,5,0,0,0,0],
         [55,2,33,0,0,0,0,0,0,0,0,0]]
    
    x  = torch.LongTensor(t1)
    y  = torch.LongTensor(t2)

    px = [[1,2,3,4,5,6,7,8,9,10],
          [1,2,3,4,5,6,7,8,9,10]]
    
    py = [[1,2,3,4,5,6,7,8,9,10,11,12],
          [1,2,3,4,5,6,7,8,9,10,11,12]]
    px = torch.LongTensor(px)
    py = torch.LongTensor(py)

    mask1  =  torch.LongTensor([[1,1,1,1,1,0,0,0,0,0],[1,1,1,1,1,1,1,0,0,0]])
    mask2  =  torch.LongTensor([[1,1,1,1,1,1,1,1,0,0,0,0],[1,1,1,0,0,0,0,0,0,0,0,0]])
    out    =  model(x,px,y,py,mask1)
    print(out.shape)
    print(out)