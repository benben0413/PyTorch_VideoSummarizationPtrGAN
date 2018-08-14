import numpy as np
from tqdm import tqdm

import torch as T
import torch.nn as nn
from torch.autograd import Variable as V

def train_G(G, data, loader, EPOCHS=100, save_func=None):
    # train generator with groudtruth using teacher forcing

    loss_G = nn.CrossEntropyLoss().cuda()
    opt_G = T.optim.Adam(G.parameters(), lr=0.005)

    for e in tqdm(range(EPOCHS)):
        ep_loss = 0

        for f, g, w in loader:
            f = f.view((f.shape[1], f.shape[0], f.shape[2]))

            f = V(f.float()).cuda()
            g = V(g.long()).cuda()
            w = V(w.float()).cuda()

            out = G(feats=f, gts=g, is_train=True)
            g = g.view(-1)
            w = w.view(-1)

            loss = loss_G(out, g)
            loss = (loss*w).mean()

            G.zero_grad()
            loss.backward()
            opt_G.step()

            loss = loss.cpu().detach().numpy()
            ep_loss += loss

        ep_loss /= len(loader)
        print('Epoch %d: %.4f' % (e+1, ep_loss))

        if not save_func==None:
            save_func(G, 'Model/onlyG.pt')

class PGLoss(nn.Module):
    def __init__(self):
        super(PGLoss, self).__init__()

        self.b = 0 # baselines
        self.n = 0

    def forward(self, idx_outs, pb_outs, reward):
        loss = 0

        tmp = 0
        for i in range(idx_outs.shape[0]):
            tmp += reward[i, 0]

            for j in range(idx_outs.shape[1]):
                idx = idx_outs[i, j]
                loss += -(reward[i, 0]-self.b)*pb_outs[i, j, idx]

        tmp /= idx_outs.shape[0]
        self.b = (self.b*self.n+tmp)/(self.n+1)
        self.n += 1

        return loss

def train_GD(G, D, loader, EPOCHS=40, save_func=None):
    # train generator using policy gradient and train discriminator

    def update_D(feats, gd_func):
        out = D(feats)
        gd = V(gd_func(out.shape)).cuda()

        loss = loss_D(out, gd)

        D.zero_grad()
        loss.backward()
        opt_D.step()

        return out, loss

    WINDOW = 5

    loss_G = PGLoss().cuda()
    opt_G = T.optim.Adam(G.parameters(), lr=0.005)

    loss_D = nn.MSELoss().cuda()
    opt_D = T.optim.Adam(D.parameters(), lr=0.002)

    for e in tqdm(range(EPOCHS)):
        ep_loss_D = 0
        ep_loss_G = 0

        for f, g, w in loader:
            # train negaive example
            f = f.view((f.shape[1], f.shape[0], f.shape[2]))
            f = V(f.float()).cuda()

            feat_outs, idx_outs, pb_outs = G.sample(f)
            idx_outs = idx_outs.view((idx_outs.shape[0], idx_outs.shape[1]))
            ed_len = T.max(idx_outs)

            feats = []
            for i in range(feat_outs.shape[0]):
                ed = 0
                while (ed+1)<feat_outs.shape[1] and not idx_outs[i, ed+1]==ed_len:
                    ed += 1

                idxs = np.random.randint(ed+1, size=WINDOW)
                idxs = sorted(idxs)
                feat = T.cat([feat_outs[i, idxs[j], :].view(1, 1, feat_outs.shape[2]) for j in range(WINDOW)], dim=1)

                feats.append(feat)

            feats = T.cat(feats, dim=0)
            feats = V(feats).cuda()

            out, loss = update_D(feats, T.zeros)
            ep_loss_D += loss.cpu().detach().numpy()

            # train G using PG
            reward = out
            loss_g = loss_G(idx_outs, pb_outs, reward)

            G.zero_grad()
            loss_g.backward()
            opt_G.step()

            ep_loss_G += loss_g.cpu().detach().numpy()

            # train positive example
            g = g.view(-1).numpy()

            ed = 0
            while not g[ed+1]==g[-1]:
                ed += 1

            feats = []
            for _ in range(4):
                tmp_loss = 0

                idxs = np.random.randint(ed+1, size=WINDOW)
                idxs = sorted(idxs)
                feat = T.cat([f[idxs[k], :, :].view((f.shape[1], 1, f.shape[2])) for k in range(WINDOW)], dim=1)

                feats.append(feat)

            feats = T.cat(feats, dim=0)
            feats = V(feats).cuda()

            _, loss = update_D(feat, T.ones)
            ep_loss_D += loss.cpu().detach().numpy()

        ep_loss_D /= 2*len(loader)
        ep_loss_G /= len(loader)
        print('Epoch %d: loss_D: %.4f, loss_G: %.4f' % (e+1, ep_loss_D, ep_loss_G))

        if not save_func==None:
            save_func(G, 'Model/G.pt')
            save_func(D, 'Model/D.pt')
