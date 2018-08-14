import torch as T
import torch.nn as nn
from torch.autograd import Variable as V

import numpy as np

class vsumG(nn.Module):
    def __init__(self, feat_size, hid_size, max_out_len):
        super(vsumG, self).__init__()
        
        self.feat_size = feat_size
        self.hid_size = hid_size
        self.max_out_len = max_out_len
        
        self.enc = nn.LSTM(self.feat_size, self.hid_size, bidirectional=True)
        self.dec = nn.LSTMCell(self.feat_size, 2*self.hid_size)
        
        self.enc_w = nn.Linear(2*self.hid_size, self.hid_size)
        self.dec_w = nn.Linear(2*self.hid_size, self.hid_size)
        self.v = nn.Linear(self.hid_size, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, feats, gts=None, is_train=False):
        enc_out, (h, c) = self.enc(feats)
        h = h.view((h.shape[1], -1))
        c = c.view((c.shape[1], -1))
        inp = feats[0, :, :]
            
        dec_out = []
        for i in range(self.max_out_len):
            h, c = self.dec(inp, (h, c))
                
            tmp = []
            for j in range(enc_out.shape[0]):
                tmp.append(self.v(self.tanh(self.dec_w(h)+self.enc_w(enc_out[j, :, :]))))
            tmp = T.cat(tmp, dim=1)
            dec_out.append(tmp)
            
            if is_train==True:
                inp = T.cat([feats[gts[k][i], k, :].view(1, -1) for k in range(feats.shape[1])], dim=0)
            
            else:
                idx = tmp.max(dim=1)[1]
                inp = T.cat([feats[idx[k], k, :].view(1, -1) for k in range(feats.shape[1])], dim=0)
            
        dec_out = T.cat(dec_out, dim=0)
            
        return dec_out
    
    def sample(self, feats, size=4):
        idx_outs = []
        feat_outs = []
        pb_outs = []
        
        for _ in range(size):
            enc_out, (h, c) = self.enc(feats)
            h = h.view((h.shape[1], -1))
            c = c.view((c.shape[1], -1))
            inp = feats[0, :, :]
            
            idx_out = []
            feat_out = []
            pb_out = []
            for i in range(self.max_out_len):
                h, c = self.dec(inp, (h, c))
                
                tmp = []
                for j in range(enc_out.shape[0]):
                    tmp.append(self.v(self.tanh(self.dec_w(h)+self.enc_w(enc_out[j, :, :]))))
                tmp = T.cat(tmp, dim=1)
                idxs = T.multinomial(T.exp(tmp), 1)
                
                inp = T.cat([feats[idxs[k][0], k, :].view(1, -1) for k in range(feats.shape[1])], dim=0)
                
                feat_out.append(inp)
                idx_out.append(idxs)
                pb_out.append(nn.functional.log_softmax(tmp))
            
            feat_out = T.cat(feat_out, dim=0)
            feat_outs.append(feat_out.view(1, feat_out.shape[0], feat_out.shape[1]))
            idx_out = T.cat(idx_out, dim=0)
            idx_outs.append(idx_out.view(1, idx_out.shape[0], idx_out.shape[1]))
            pb_out = T.cat(pb_out, dim=0)
            pb_outs.append(pb_out.view(1, pb_out.shape[0], pb_out.shape[1]))
        
        feat_outs = T.cat(feat_outs, dim=0)
        idx_outs = T.cat(idx_outs, dim=0)
        pb_outs = T.cat(pb_outs, dim=0)
        
        return feat_outs, idx_outs, pb_outs

class vsumD(nn.Module):
    def __init__(self, window=5):
        super(vsumD, self).__init__()
        
        self.window = window
        
        conv1 = nn.Conv1d(self.window, 32, 5, stride=2)
        conv2 = nn.Conv1d(32, 32, 5, stride=2)
        conv3 = nn.Conv1d(32, 64, 5, stride=2)
        conv4 = nn.Conv1d(64, 64, 5, stride=2)
        conv5 = nn.Conv1d(64, 128, 3, stride=2)
        conv6 = nn.Conv1d(128, 128, 3, stride=1)
        
        pool = nn.MaxPool1d(3)
        
        linear1 = nn.Linear(512, 64)
        linear2 = nn.Linear(64, 64)
        linear3 = nn.Linear(64, 1)
        
        leaky = nn.LeakyReLU()
        sigmoid = nn.Sigmoid()
        
        self.conv = nn.Sequential(*[conv1, leaky, conv2, leaky, pool, 
                                    conv3, leaky, conv4, leaky, pool, 
                                    conv5, leaky, conv6, leaky])
        self.linear = nn.Sequential(*[linear1, leaky, 
                                      linear2, leaky, 
                                      linear3, sigmoid])
    
    def forward(self, feats):
        out = self.conv(feats)
        out = out.view((out.shape[0], -1))
        out = self.linear(out)
        
        return out
