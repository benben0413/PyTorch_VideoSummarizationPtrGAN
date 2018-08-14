import torch as T
import torch.nn as nn
from torch.autograd import Variable as V

import numpy as np

def f1_score(pd, gt):
    def cross(lst_a, lst_b):
        ret = 0
        
        for i in lst_a:
            if i in lst_b:
                ret += 1
                
        return ret
    
    p = cross(pd, gt)/len(pd) # precision
    r = cross(gt, pd)/len(gt) # recall
    f1 = 2*((p*r)/(p+r))
    
    return f1

def evaluation(G, loader):
    f1s = 0
    
    for f, g, w in loader:
        f = f.view((f.shape[1], f.shape[0], f.shape[2]))
        f = V(f.float()).cuda()
        
        out = G(feats=f, is_train=False)
        out = np.argmax(out.cpu().detach().numpy(), axis=1)
        out = [x for x in out]
        
        g = g.view((-1))
        g = g.numpy()
        g = [x for x in g]
        ed = g[-1]
        
        if ed in out:
            out = out[:out.index(ed)+1]
        g = g[:g.index(ed)+1]
        
        f1 = f1_score(pd=out, gt=g)
        f1s += f1
    
    f1 = f1s/len(loader)
    
    return f1
        