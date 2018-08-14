import numpy as np
import pickle

from torch.utils.data import Dataset, DataLoader

class FeatureDataset(Dataset):
    def __init__(self, feats, gts, user=0):
        super(FeatureDataset, self).__init__()
        
        self.feats = feats
        self.gts = gts
        self.user = user
        
        self.keys = list(feats.keys())
        
        self.max_out_len = 0
        for k in self.keys:
            self.max_out_len = max(self.max_out_len, len(gts[k][self.user]))
        self.max_out_len += 2
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        feat = [np.ones((2048, ))]+self.feats[self.keys[idx]]
        feat.append(np.zeros((2048, )))
        feat = np.array(feat)
        
        gt = [x for x in self.gts[self.keys[idx]][self.user]]+[(feat.shape[0]-1) for _ in range(self.max_out_len-len(self.gts[self.keys[idx]][self.user]))]
        gt = np.array(gt)
        
        weight = [1 for _ in range(len(self.gts[self.keys[idx]][self.user])+1)]+[0 for _ in range(self.max_out_len-len(self.gts[self.keys[idx]][self.user])-1)]
        weight = np.array(weight)
        
        return feat, gt, weight

def get_loader():
    groundtruth = pickle.load(open('Dataset/Youtube/groundtruth.pkl', 'rb'))
    features = pickle.load(open('Dataset/Youtube/features.pkl', 'rb'))
    
    data = FeatureDataset(feats=features, gts=groundtruth)
    loader = DataLoader(data, batch_size=1, shuffle=True)
    
    return data, loader
