# -*- coding: utf-8 -*-
import argparse

import torch
'''
self.source_vocab_size = 100
self.target_vocab_size = 80
self.source_max_length = 10
self.target_max_length = 12
self.posi_length       = max(self.source_max_length, self.target_max_length)

self.embed_dim         = 20
self.kernel_size       = 3
self.econv_num         = 3
self.dconv_num         = 3
'''
def all_args(parser):
    #model hyper parameters
    parser.add_argument('-embed_dim', type=int, default=300,
                       help='word embedding dim and position embedding dim')
    parser.add_argument('-encoder_kernel_width', type=int, default=3,
                       help='encoder kernel width')
    parser.add_argument('-encoder_layer_num', type=int, default=2,
                       help='encoder layer num')
    parser.add_argument('-decoder_kernel_width', type=int, default=3,
                       help='decoder kernel width')
    parser.add_argument('-decoder_layer_num', type=int, default=2,
                       help='decoder layer num')
    
    #dataset hyper parameters
    parser.add_argument('-path', type=str, default='./data',
                       help='data root')
    parser.add_argument('-filename', type=str, default='data',
                       help='train or label file name without suffix')
    parser.add_argument('-max_len', type=int, default=986,
                       help='maxinum length of the sequence')
    parser.add_argument('-min_freq', type=int, default=1,
                       help='mininum frequence of the word')

    #train hyper parameters
    parser.add_argument('-lr', type=float, default=5e-5,
                       help='learning rate')
    parser.add_argument('-epochs', type=int, default=100,
                       help='epochs')
    parser.add_argument('-batch_size', type=int, default=32,
                       help='batch size')
    parser.add_argument('-model', type=str, default='',
                       help='load model path')
    parser.add_argument('-save_model', type=str, default='./model',
                       help='save model directory')
    parser.add_argument('-save_frequence', type=int, default=1,
                       help='how many epoch we save model')
    parser.add_argument('-gpuid', type=int, default=0,
                       help='use gpu, -1 is cpu')
    

parser = argparse.ArgumentParser()
all_args(parser)
opts = parser.parse_args()

#other parameters

opts.DEVICE = torch.device('cuda:%d' % opts.gpuid)
#opts.vocab_size = 1643

#回归数
#opts.key_num    = 8

#分类数
#opts.class_num  = 4398