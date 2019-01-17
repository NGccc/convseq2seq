from torchtext import data, datasets
import torch
from config import opts
import pdb

class mydataset(object):
    def __init__(self):
        super(mydataset, self).__init__()
        
        self.UNK_TOKEN = "<unk>"
        self.PAD_TOKEN = "<pad>"
        self.SOS_TOKEN = "<S>"  #<S> is start token ,<s> is silence token.
        #EOS_TOKEN = "</s>"
        
        self.max_len   = opts.max_len
        self.min_freq  = opts.min_freq
        self.filename  = opts.filename
        self.path      = opts.path
        self.DEVICE    = opts.DEVICE
        
        DEVICE = torch.device('cuda:0')
    
    def getIterator(self):
        #Step 1
        #一般我用的是已经分好词的数据， 所以tokenize=None
        
        SRC = data.Field( 
                        sequential=True, batch_first=True, include_lengths=True,
                        unk_token=self.UNK_TOKEN, pad_token=self.PAD_TOKEN,
                        init_token=None, eos_token=None)
                            
        TRG = data.Field(
                        batch_first=True, include_lengths=True,
                        unk_token=self.UNK_TOKEN, pad_token=self.PAD_TOKEN,
                        init_token=self.SOS_TOKEN, eos_token=None)
        
        #Step 2
        train_data = datasets.TranslationDataset.splits(
        path=self.path, train= self.filename, validation=None, test=None, exts=('.train', '.label'), fields=(SRC, TRG),
        filter_pred=lambda x: len(vars(x)['src']) <= self.max_len and len(vars(x)['trg']) <= self.max_len)
        train_data = train_data[0]
        #Step 3
        SRC.build_vocab(train_data.src, min_freq = self.min_freq)
        TRG.build_vocab(train_data.trg, min_freq = self.min_freq)
        
        #Step 4
        opts.src_vocab_size = len(SRC.vocab.stoi)
        opts.trg_vocab_size = len(TRG.vocab.stoi)

        f = open('s2i.train','w',encoding='utf-8-sig')
        f.write(str(SRC.vocab.stoi))
        f.close()

        f = open('s2i.label','w',encoding='utf-8-sig')
        f.write(str(TRG.vocab.stoi))
        f.close()

        train_iter = data.BucketIterator(train_data, batch_size=64, train=True, 
                                        sort_within_batch=True, 
                                        sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                        device=self.DEVICE)
        
        # 如果用一些未排序的外部文件进行valid，经常有问题。
        #为了方便，将batch大小设置为1
        #valid_iter = data.Iterator(valid_data, batch_size=1, train=False, sort=False, repeat=False,  device=DEVICE)
        return train_iter