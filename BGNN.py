import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from Params import args

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class myModel(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats): 
        super(myModel, self).__init__()  
        
        self.userNum = userNum
        self.itemNum = itemNum
        self.behavior = behavior
        self.behavior_mats = behavior_mats
        
        self.embedding_dict = self.init_embedding() 
        self.weight_dict = self.init_weight()
        self.gcn = GCN(self.userNum, self.itemNum, self.behavior, self.behavior_mats)


    def init_embedding(self):
        
        embedding_dict = {  
            'user_embedding': None,
            'item_embedding': None,
            'user_embeddings': None,
            'item_embeddings': None,
        }
        return embedding_dict

    def init_weight(self):  
        initializer = nn.init.xavier_uniform_
        
        weight_dict = nn.ParameterDict({
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([args.hidden_dim, args.hidden_dim]))),
            'w_self_attention_cat': nn.Parameter(initializer(torch.empty([args.head_num*args.hidden_dim, args.hidden_dim]))),
            'alpha': nn.Parameter(torch.ones(2)),
        })      
        return weight_dict  


    def forward(self):

        user_embed, item_embed, user_embeds, item_embeds = self.gcn()


        return user_embed, item_embed, user_embeds, item_embeds 

    def para_dict_to_tenser(self, para_dict): 
    
        tensors = []
        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)

        return tensors.float()

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_parameters(), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_parameters()(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  
                    self.set_param(self, name, param)


class GCN(nn.Module): # 각각의 GCN layer의 임베딩을 합치는 부분
    def __init__(self, userNum, itemNum, behavior, behavior_mats):
        super(GCN, self).__init__()  
        self.userNum = userNum
        self.itemNum = itemNum
        self.hidden_dim = args.hidden_dim

        self.behavior = behavior
        self.behavior_mats = behavior_mats

        self.user_embedding, self.item_embedding = self.init_embedding()         
        
        self.alpha, self.i_concatenation_w, self.u_concatenation_w, self.i_input_w, self.u_input_w = self.init_weight()
        
        self.sigmoid = torch.nn.Sigmoid()
        self.act = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(args.drop_rate)

        self.gnn_layer = eval(args.gnn_layer)
        self.layers = nn.ModuleList()
        for i in range(0, len(self.gnn_layer)): # [0, 1, 2] => 총 3개 => 3-hop ; hidden_dim = 16
            self.layers.append(GCNLayer(args.hidden_dim, args.hidden_dim, self.userNum, self.itemNum, self.behavior, self.behavior_mats))  

    def init_embedding(self):
        user_embedding = torch.nn.Embedding(self.userNum, args.hidden_dim)
        item_embedding = torch.nn.Embedding(self.itemNum, args.hidden_dim)
        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)

        return user_embedding, item_embedding

    def init_weight(self):
        alpha = nn.Parameter(torch.ones(2))  
        i_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim)) # item's transformation matrix
        u_concatenation_w = nn.Parameter(torch.Tensor(len(eval(args.gnn_layer))*args.hidden_dim, args.hidden_dim)) # user's transformation matrix
        i_input_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        u_input_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        init.xavier_uniform_(i_concatenation_w)
        init.xavier_uniform_(u_concatenation_w)
        init.xavier_uniform_(i_input_w)
        init.xavier_uniform_(u_input_w)
        # init.xavier_uniform_(alpha)

        return alpha, i_concatenation_w, u_concatenation_w, i_input_w, u_input_w

    def forward(self, user_embedding_input=None, item_embedding_input=None):
        all_user_embeddings = []
        all_item_embeddings = []
        all_user_embeddingss = []
        all_item_embeddingss = []

        user_embedding = self.user_embedding.weight
        item_embedding = self.item_embedding.weight

        for i, layer in enumerate(self.layers):
            
            user_embedding, item_embedding, user_embeddings, item_embeddings = layer(user_embedding, item_embedding)

            norm_user_embeddings = F.normalize(user_embedding, p=2, dim=1)
            norm_item_embeddings = F.normalize(item_embedding, p=2, dim=1)  

            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)
            all_user_embeddingss.append(user_embeddings)
            all_item_embeddingss.append(item_embeddings)
            
        user_embedding = torch.cat(all_user_embeddings, dim=1) # [#user, 16*3(#GNN Layer)]
        item_embedding = torch.cat(all_item_embeddings, dim=1) # [#item, 16*3(#GNN Layer)]
        user_embeddings = torch.cat(all_user_embeddingss, dim=2) # [4, #user, 16*3(#GNN Layer)]
        item_embeddings = torch.cat(all_item_embeddingss, dim=2) # [4, #item, 16*3(#GNN Layer)]
        
        user_embedding = torch.matmul(user_embedding , self.u_concatenation_w) # [#user, 16]
        item_embedding = torch.matmul(item_embedding , self.i_concatenation_w) # [#item, 16]
        user_embeddings = torch.matmul(user_embeddings , self.u_concatenation_w) # [4, #user, 16]
        item_embeddings = torch.matmul(item_embeddings , self.i_concatenation_w) # [4, #item, 16]

        return user_embedding, item_embedding, user_embeddings, item_embeddings  #[31882, 16], [31882, 16], [4, 31882, 16], [4, 31882, 16]


class GCNLayer(nn.Module): # 각각의 GCN(GNN) layer를 설명
    def __init__(self, in_dim, out_dim, userNum, itemNum, behavior, behavior_mats):
        super(GCNLayer, self).__init__()

        self.behavior = behavior
        self.behavior_mats = behavior_mats

        self.userNum = userNum
        self.itemNum = itemNum

        self.act = torch.nn.Sigmoid()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim)) # GCN - weight about item
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim)) # GCN - weight about user
        self.ii_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        init.xavier_uniform_(self.i_w) # 초기 가중치 설정 ; Xavier Initialization ; item's transformation matrix
        init.xavier_uniform_(self.u_w) # 초기 가중치 설정 ; Xavier Initialization ; user's transformation matrix

    def forward(self, user_embedding, item_embedding):

        user_embedding_list = [None]*len(self.behavior)
        item_embedding_list = [None]*len(self.behavior)

        for i in range(len(self.behavior)): # 각 behavior type 별로 embedding propagation layer
            user_embedding_list[i] = torch.spmm(self.behavior_mats[i]['A'], item_embedding)
            item_embedding_list[i] = torch.spmm(self.behavior_mats[i]['AT'], user_embedding)

        user_embeddings = torch.stack(user_embedding_list, dim=0) # TMall) user_embeddings.shape = [4, #user, 16]
        item_embeddings = torch.stack(item_embedding_list, dim=0) # TMall) item_embeddings.shape = [4, #item, 16]
        # print(user_embeddings.shape)
        # print(item_embeddings.shape)
        # print("before")
        
        # embedding - message aggregation
        user_embedding = self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w)) # user's aggregated feature representation ; [#user, 16]
        item_embedding = self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w)) # item's aggregated feature representation ; [#item, 16]

        user_embeddings = self.act(torch.matmul(user_embeddings, self.u_w)) # user's not aggregated feature representation ; [4, #user, 16]
        item_embeddings = self.act(torch.matmul(item_embeddings, self.i_w)) # item's not aggregated feature representation ; [4, #item, 16]
        # import sys
        # print(user_embedding.shape)
        # print(user_embeddings.shape)
        # print(user_embedding.shape)
        # print(item_embeddings.shape)
        # print("BGNN - GCNLayer- forward")
        # sys.exit()
        return user_embedding, item_embedding, user_embeddings, item_embeddings             

#------------------------------------------------------------------------------------------------------------------------------------------------

