import torch
import torch.nn as nn

from Params import args

import sys

class PropModel(nn.Module):
    def __init__(self):
        super(PropModel, self).__init__()

        # self.n_users = user_emb.shape[1]
        self.emb_dim = args.hidden_dim
        self.prop_weight = self._init_weights()
        self.leakyrelu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(args.drop_rate1)

        self.mh_model = MultiHeadAttentionLayer(self.emb_dim, args.head_num, args.drop_rate1, torch.device('cuda'))

    def _init_weights(self):
        initializer = nn.init.xavier_uniform_
        
        all_weights = nn.ParameterDict({
            'pv_2_fav' : nn.Parameter(initializer(torch.empty([self.emb_dim, self.emb_dim]))),
            'pv_2_cart' : nn.Parameter(initializer(torch.empty([self.emb_dim, self.emb_dim]))),
            'pv_2_buy' : nn.Parameter(initializer(torch.empty([self.emb_dim, self.emb_dim]))),
            'fav_2_cart' : nn.Parameter(initializer(torch.empty([self.emb_dim, self.emb_dim]))),
            'fav_2_buy' : nn.Parameter(initializer(torch.empty([self.emb_dim, self.emb_dim]))),
            'cart_2_buy' : nn.Parameter(initializer(torch.empty([self.emb_dim, self.emb_dim]))),
        }
        )

        return all_weights
    
    def forward(self, user_emb):
        attRepList = []
        tmpWeight = []
        # print(self.prop_weight)
        # sys.exit()
        for x in self.prop_weight.values():
            tmpWeight.append(x)

        attRepList.append(user_emb[0])
        for i in range(1, user_emb.shape[0]): # 1(0) 2(0 1) 3(0 1 2)
            tmpRepList = []
            for j in range(i):
                score, _ = self.mh_model(user_emb[i], user_emb[j])
                score = score.view(user_emb.shape[1], self.emb_dim)
                if j == 0:
                    tmp = torch.mul(user_emb[i], user_emb[j])
                    tmp = self.dropout(tmp)
                else:
                    tmp = torch.mul(tmpRepList[0], user_emb[j])
                    tmp = self.dropout(tmp)
                    tmpRepList = []
                
                tmp  = torch.matmul(tmp, tmpWeight[(i-1)+j])
                # tmp = self.leakyrelu(tmp)
                mes_rep = self.leakyrelu(torch.mul(tmp, score))
                
                # user_emb[j] = user_emb[j] + mes_rep
                # tmp2 = user_emb[j]
                final_mes_rep = user_emb[i] + mes_rep
                tmpRepList.append(final_mes_rep)
            attRepList.append(final_mes_rep)
        
        new_user_emb = torch.stack(attRepList, dim=0) 
        return new_user_emb

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, q_user_emb, k_user_emb):
        batch_size = q_user_emb.shape[0]
        # batch_size = self.hid_dim

        Q = self.fc_q(q_user_emb)
        K = self.fc_k(k_user_emb)
        V = self.fc_v(k_user_emb)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        attention = torch.softmax(energy, dim = -1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        
        return x, attention

