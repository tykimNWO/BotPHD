import numpy as np
from numpy import random
import pickle
from scipy.sparse import csr_matrix
import math
import gc
import time
import random
import datetime

import torch
import torch.nn as nn
import torch.utils.data as dataloader
import torch.nn.functional as F
from torch.nn import init

import graph_utils
import DataHandler


import BGNN

from Params import args
from Utils.TimeLogger import log
from tqdm import tqdm

from setproctitle import *
from prop import PropModel

import sys

torch.backends.cudnn.benchmark=True

if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

MAX_FLAG = 0x7FFFFFFF

now_time = datetime.datetime.now()
modelTime = datetime.datetime.strftime(now_time,'%Y_%m_%d__%H_%M_%S')

torch.autograd.set_detect_anomaly(True)

class Model():
    def __init__(self):

        self.trn_file = args.path + args.dataset + '/trn_'
        self.tst_file = args.path + args.dataset + '/tst_int' # test data     
        # self.tst_file = args.path + args.dataset + '/BST_tst_int_59' 
        #Tmall: 3,4,5,6,8,59
        #IJCAI_15: 5,6,8,10,13,53

        self.meta_multi_single_file = args.path + args.dataset + '/meta_multi_single_beh_user_index_shuffle'
        
        self.meta_multi_single = pickle.load(open(self.meta_multi_single_file, 'rb')) # len을 하였을 때, TMall=11690
        
        self.t_max = -1 
        self.t_min = 0x7FFFFFFF
        self.time_number = -1

        self.user_num = -1
        self.item_num = -1
        self.behavior_mats = {} 
        self.behaviors = []
        self.behaviors_data = {}

        #history
        self.train_loss = []
        self.his_hr = []
        self.his_ndcg = []
        gc.collect()  #

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.curEpoch = 0

            
        if args.dataset == 'Tmall':
            self.behaviors_SSL = ['pv','fav', 'cart', 'buy']
            self.behaviors = ['pv','fav', 'cart', 'buy']
            # self.behaviors = ['buy']
        elif args.dataset == 'IJCAI_15':
            self.behaviors = ['click','fav', 'cart', 'buy']
            # self.behaviors = ['buy']
            self.behaviors_SSL = ['click','fav', 'cart', 'buy']

        elif args.dataset == 'JD':
            self.behaviors = ['review','browse', 'buy']
            self.behaviors_SSL = ['review','browse', 'buy']

        elif args.dataset == 'retailrocket':
            self.behaviors = ['view','cart', 'buy']
            # self.behaviors = ['buy']
            self.behaviors_SSL = ['view','cart', 'buy']

        # interaction value(or id?) = data.data
        # self.behaviors_data 에는 target + auxiliary edge type의 interaction data가 들어감!
        for i in range(0, len(self.behaviors)):
            with open(self.trn_file + self.behaviors[i], 'rb') as fs:  
                data = pickle.load(fs)
                # print(data)
                # print(type(data))
                # sys.exit()
                self.behaviors_data[i] = data 

                if data.get_shape()[0] > self.user_num:  
                    self.user_num = data.get_shape()[0]  
                if data.get_shape()[1] > self.item_num:  
                    self.item_num = data.get_shape()[1]  

                if data.data.max() > self.t_max:
                    self.t_max = data.data.max()
                if data.data.min() < self.t_min:
                    self.t_min = data.data.min()

                # target edge type 같은 경우, self.trainMat에 해당 interaction data가 들어감!!
                if self.behaviors[i]==args.target:
                    self.trainMat = data
                    self.trainLabel = 1*(self.trainMat != 0)  
                    self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0))) # item 개수와 동일 / test에서 쓰이나?
                    
        # self.behavior_mats => 모델 학습에 이용
        time = datetime.datetime.now()
        print("Start building:  ", time)
        for i in range(0, len(self.behaviors)):
            self.behavior_mats[i] = graph_utils.get_use(self.behaviors_data[i])
        time = datetime.datetime.now()
        print("End building:", time)


        print("user_num: ", self.user_num)
        print("item_num: ", self.item_num)
        print("\n")


        #---------------------------------------------------------------------------------------------->>>>>
        #train_data --> data in target edge type(보류)
        train_u, train_v = self.trainMat.nonzero()
        train_data = np.hstack((train_u.reshape(-1,1), train_v.reshape(-1,1))).tolist()
        train_dataset = DataHandler.RecDataset_beh(self.behaviors, train_data, self.item_num, self.behaviors_data, True)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

        #valid_data


        # test_data  
        with open(self.tst_file, 'rb') as fs:
            data = pickle.load(fs)
        
        test_user = np.array([idx for idx, i in enumerate(data) if i is not None])
        test_item = np.array([i for idx, i in enumerate(data) if i is not None])
        # tstUsrs = np.reshape(np.argwhere(data!=None), [-1])
        test_data = np.hstack((test_user.reshape(-1,1), test_item.reshape(-1,1))).tolist()
        # testbatch = np.maximum(1, args.batch * args.sampNum 
        test_dataset = DataHandler.RecDataset(test_data, self.item_num, self.trainMat, 0, False)
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)  

        # -------------------------------------------------------------------------------------------------->>>>>

    def prepareModel(self):
        self.modelName = self.getModelName()  
        self.setRandomSeed()
        self.gnn_layer = eval(args.gnn_layer)  
        self.hidden_dim = args.hidden_dim
        

        if args.isload == True:
            self.loadModel(args.loadModelPath)
        # else:
            # self.model = BGNN.myModel(self.user_num, self.item_num, self.behaviors, self.behavior_mats).cuda()
            # self.meta_weight_net = MV_Net.MetaWeightNet(len(self.behaviors)).cuda()
        
        self.beh_model = BGNN.myModel(self.user_num, self.item_num, self.behaviors, self.behavior_mats).cuda()
        
        # self.beh_model_opt = torch.optim.AdamW(self.beh_model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
        # self.beh_model_scheduler = torch.optim.lr_scheduler.CyclicLR(self.beh_model_opt, args.opt_base_lr, args.opt_max_lr, step_size_up=5, step_size_down=20, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        
        
        self.prop_model = PropModel().cuda()
        self.prop_model_opt = torch.optim.AdamW(self.prop_model.parameters(), lr = args.prop_opt_base_lr, weight_decay = args.opt_weight_decay)
        self.prop_model_scheduler = torch.optim.lr_scheduler.CyclicLR(self.prop_model_opt, args.prop_opt_base_lr, args.prop_opt_max_lr, step_size_up=3, step_size_down=7, mode='triangular', gamma=0.98, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.9, max_momentum=0.99, last_epoch=-1)            
        # self.prop_model.load_state_dict(self.beh_model.state_dict())
        
        # for name, param in self.beh_model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
                
        # for name, param in self.prop_model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        
        # sys.exit()


        # #IJCAI_15
        self.beh_model.opt = torch.optim.AdamW(self.beh_model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
        # self.beh_model.meta_opt =  t.optim.AdamW(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay)
        # # self.meta_opt =  t.optim.RMSprop(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay, momentum=0.95, centered=True)
        self.beh_model.scheduler = torch.optim.lr_scheduler.CyclicLR(self.beh_model.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=5, step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        # self.meta_scheduler = t.optim.lr_scheduler.CyclicLR(self.meta_opt, args.meta_opt_base_lr, args.meta_opt_max_lr, step_size_up=2, step_size_down=3, mode='triangular', gamma=0.98, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.9, max_momentum=0.99, last_epoch=-1)
        # #       


        #Tmall
        # self.opt = torch.optim.AdamW(self.model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
        # self.meta_opt =  torch.optim.AdamW(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay)
        # # self.meta_opt =  t.optim.RMSprop(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay, momentum=0.95, centered=True)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=5, step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        # self.meta_scheduler = torch.optim.lr_scheduler.CyclicLR(self.meta_opt, args.meta_opt_base_lr, args.meta_opt_max_lr, step_size_up=3, step_size_down=7, mode='triangular', gamma=0.98, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.9, max_momentum=0.99, last_epoch=-1)
        #                                                                                                                                                                           0.993                                             

        # # retailrocket
        # self.opt = t.optim.AdamW(self.model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
        # # self.meta_opt =  t.optim.AdamW(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay)
        # self.meta_opt =  t.optim.SGD(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay, momentum=0.95, nesterov=True)
        # self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=1, step_size_down=2, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        # self.meta_scheduler = t.optim.lr_scheduler.CyclicLR(self.meta_opt, args.meta_opt_base_lr, args.meta_opt_max_lr, step_size_up=1, step_size_down=2, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.9, max_momentum=0.99, last_epoch=-1)
        # #                                                                                                                                      exp_range


        # if use_cuda:
        #     self.model = self.model.cuda()

    def innerProduct(self, u, i, j):  
        pred_i = torch.sum(torch.mul(u,i), dim=1)*args.inner_product_mult  # Tmall) pred_i.shape = ([8141])
        pred_j = torch.sum(torch.mul(u,j), dim=1)*args.inner_product_mult  # Tmall) pred_i.shape = ([8141])
        
        return pred_i, pred_j

    def SSL(self, user_embeddings, item_embeddings, target_user_embeddings, target_item_embeddings, user_step_index, e):
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        def multi_neg_sample_pair_index(neg_cl_index, batch_index, step_index, embedding1, embedding2):  #small, big, target, beh: [100], [1024], [31882, 16], [31882, 16]

            # index_set = set(np.array(step_index.cpu())) #length->Random
            # batch_index_set = set(np.array(batch_index.cpu())) #length->18
            
            # neg2_index_set = index_set - batch_index_set                         #beh
            
            # neg2_index = torch.as_tensor(np.array(list(neg2_index_set))).long().cuda()  #[910]
            neg2_index = neg_cl_index.long().cuda()
            neg2_index = torch.unsqueeze(neg2_index, 0)                              #[1, 910]
            neg2_index = neg2_index.repeat(len(batch_index), 1)                  #[100, 910]
            neg2_index = torch.reshape(neg2_index, (1, -1))                          #[1, 91000]
            neg2_index = torch.squeeze(neg2_index)                                   #[91000]
                                                                                 #target
            neg1_index = batch_index.long().cuda()     #[100]
            neg1_index = torch.unsqueeze(neg1_index, 1)                              #[100, 1]
            neg1_index = neg1_index.repeat(1, len(neg_cl_index))               #[100, 910]
            neg1_index = torch.reshape(neg1_index, (1, -1))                          #[1, 91000]           
            neg1_index = torch.squeeze(neg1_index)                                   #[91000]

            # neg2(neg1)_index <- len(neg2_index_set) * batch_index.shape(18)
            # print("test - multi_neg")
            # tmp_neg1 = neg1_index.tolist()
            # tmp_neg2 = neg2_index.tolist()

            # for i in range(batch_index.size(dim=0)):
            #     euc_dist = []

            #     for j in range(len(neg2_index_set)):
            #         euclidean_dist = sum(((embedding1[neg1_index[i*batch_index.size(dim=0)+j]] - embedding2[neg2_index[i*batch_index.size(dim=0)+j]])**2)).cuda()
            #         euc_dist.append((tmp_neg1[i*batch_index.size(dim=0)+j],tmp_neg2[i*batch_index.size(dim=0)+j], euclidean_dist.item()))

            #     euc_dist = sorted(euc_dist, key=lambda x:x[1], reverse=True)
            #     real_neg = euc_dist[:int(len(neg2_index_set) * args.neg_rate)]

            #     for j in range(len(real_neg)):
            #         del tmp_neg1[tmp_neg1.index(real_neg[j][0])]
            #         del tmp_neg2[tmp_neg2.index(real_neg[j][1])]
            
            # neg1_index = torch.tensor(tmp_neg1)
            # neg2_index = torch.tensor(tmp_neg2)
                
            neg_score_pre = torch.sum(compute(embedding1, embedding2, neg1_index, neg2_index, sign="neg", ep=e).squeeze().view(len(batch_index), -1), -1)  #[91000,1]==>[91000]==>[100, 910]==>[100]
            
            return neg_score_pre  #[100]

        def compute(x1, x2, neg1_index=None, neg2_index=None, sign=None, ep=None ,τ = 0.05):  #[1024, 16], [1024, 16]

            if neg1_index!=None:
                x1 = x1[neg1_index]
                x2 = x2[neg2_index]

            N = x1.shape[0]  
            D = x1.shape[1]

            # x1 = x1
            # x2 = x2
            
            # torch.bmm => batch matrix multiplication(two operands : batch)
            
            # tmp = torch.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1)
            # print("{} input test(isnan): ".format(sign), torch.any(torch.isnan(tmp)))
            # print("{} input test(isinf): ".format(sign), torch.any(torch.isinf(tmp)))
            # if torch.any(torch.isnan(tmp)) == True or torch.any(torch.isinf(tmp)) == True:
            #     print("nan or inf detected")
            #     sys.exit()
            
            tmp = torch.div(torch.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1), np.power(D, 10)+1e-8)
            # print("{} input test(isnan): ".format(sign), torch.any(torch.isnan(tmp)))
            # print("{} input test(isinf): ".format(sign), torch.any(torch.isinf(tmp)))
            # if torch.any(torch.isnan(tmp)) == True or torch.any(torch.isinf(tmp)) == True:
            #     print("nan or inf detected")
            #     sys.exit()
            
            scores = torch.div(torch.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1), np.power(D, 10)+1e-8)  #[1024, 1]
            # scores = torch.div(scores, np.power(D,2)+1e-8)
            scores = torch.div(scores, 10000000)
            scores = torch.exp(scores)  #[1024, 1]
            # scores = torch.exp(torch.div(torch.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1), np.power(D, 1)+1e-8))  #[1024, 1]
            
            # print("{} input score test(isnan): ".format(sign), torch.any(torch.isnan(scores)))
            
            # print("{} input score test(isinf): ".format(sign), torch.any(torch.isinf(scores)))
            
            if torch.any(torch.isnan(scores)) == True or torch.any(torch.isinf(scores)) == True:
                # scores = torch.nan_to_num(scores, nan=1, posinf=1, neginf=1)
                print("{} input score test(isinf): ".format(sign), tmp)
                print("{} input score test(isinf): ".format(sign), scores)
                scores = torch.nan_to_num(scores)
                
            scores = scores.squeeze()
            
            if torch.any(torch.isnan(scores)) == True or torch.any(torch.isinf(scores)) == True:
                print("scores: nan or inf detected")
                sys.exit()

            scores = scores.view(N,1)
            return scores

        def single_infoNCE_loss_one_by_one(embedding1, embedding2, step_index, e):  #target, aux
            N = step_index.shape[0] # TMall) N = 819
            D = embedding1.shape[1] # TMall) D = 16(hidden_dim)

            pos_score = compute(embedding1[step_index], embedding2[step_index], sign="pos", ep=e).squeeze()  #[1024]
            # pos_score.shape = torch.Size([819])??
            neg_score = torch.zeros((N,), dtype = torch.float64).cuda()  #[1024]

            #-------------------------------------------------multi version-----------------------------------------------------
            # np.ceil : 올림 연산
            steps = int(np.ceil(N / args.SSL_batch))  #separate the batch to smaller one ; TMall) steps = 46
            for i in range(steps):
                st = i * args.SSL_batch # Tmall) args.SSL_batch = 18
                ed = min((i+1) * args.SSL_batch, N)
                
                batch_index = step_index[st: ed]

                # print("test - batch_index")
                index_set = set(np.array(step_index.cpu())) #length->Random
                batch_index_set = set(np.array(batch_index.cpu())) #length->18
            
                neg2_index_set = index_set - batch_index_set 
                tmp_neg1 = list(batch_index_set)
                tmp_neg1 = random.sample(tmp_neg1, int(len(tmp_neg1) * 0.1))
                tmp_neg2 = list(neg2_index_set)
                tmp_neg2 = random.sample(tmp_neg2, int(len(neg2_index_set) * args.neg_rate * 1.2))
                
                euc_dist = []
                # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                # cos = nn.PairwiseDistance(p=2)
                # import time 
                # bef_time = time.time()
                for j in range(len(tmp_neg1)):
                    for k in range(len(tmp_neg2)):
                        cos_dist = torch.norm(embedding1[tmp_neg1[j]] - embedding2[tmp_neg2[k]], 2)
                        # cos_dist = cos(embedding1[tmp_neg1[j]], embedding2[tmp_neg2[k]])
                        euc_dist.append((tmp_neg2[k], cos_dist.item()))
                # aft_time = time.time()
                # print(aft_time - bef_time)
                euc_dist = sorted(euc_dist, key=lambda x:x[1])
                euc_dist = euc_dist[:int(len(neg2_index_set) * args.neg_rate)]
                real_neg = [euc_dist[i][0] for i in range(len(euc_dist))]
                
                neg_cl_index = torch.tensor(real_neg).cuda()
                
                neg_score_pre = multi_neg_sample_pair_index(neg_cl_index, batch_index, step_index, embedding1, embedding2)
                
                if i ==0:
                    neg_score = neg_score_pre
                else:
                    neg_score = torch.cat((neg_score, neg_score_pre), 0).cuda()
            #-------------------------------------------------multi version-----------------------------------------------------
            # print("pos")
            # print(pos_score)
            # print("neg")
            # print(neg_score)
            # if torch.any(torch.isnan(pos_score)).item() == True or torch.any(torch.isinf(pos_score)).item() == True:
            #     pos_score = torch.nan_to_num(pos_score, nan=0.05, posinf=0.05, neginf=0.05)
            # if torch.any(torch.isnan(neg_score)).item() == True or torch.any(torch.isinf(neg_score)).item() == True:
            #     neg_score = torch.nan_to_num(neg_score, nan=0.05, posinf=0.05, neginf=0.05)
            con_tmp1 = torch.div(pos_score, neg_score+1e-8)
            # if torch.any(torch.isnan(con_tmp1)).item() == True or torch.any(torch.isinf(con_tmp1)).item() == True:
            #     con_tmp1 = torch.nan_to_num(con_tmp1, nan=0.05, posinf=0.05, neginf=0.05)
            
            con_loss = -torch.log(1e-8 +con_tmp1).cuda()  #[1024]/[1024]==>1024
            
            # if torch.any(torch.isinf(con_loss)).item() == True:
            #     con_loss = torch.nan_to_num(con_loss, nan=0.05, posinf=0.05, neginf=0.05)
            
            # print("torch-log")
            # print(torch.div(pos_score, neg_score+1e-8))
            # print("con_loss")
            # print(con_loss) # nan 발견
            
            
            assert not torch.any(torch.isnan(con_loss))
            assert not torch.any(torch.isinf(con_loss))

            return torch.where(torch.isnan(con_loss), torch.full_like(con_loss, 0+1e-8), con_loss)

        torch.autograd.set_detect_anomaly(True)
        
        user_con_loss_list = []
        max_loss_layer_lower = torch.tensor(float('-inf')).cuda()

        user_con_loss_list.append(user_embeddings[0])

        SSL_len = int(user_step_index.shape[0]/10)
        user_step_index = torch.as_tensor(np.random.choice(user_step_index.cpu(), size=SSL_len, replace=False, p=None)).cuda()

        for i in range(1, len(self.behaviors_SSL)):
            for j in range(i):
                layer_loss = single_infoNCE_loss_one_by_one(user_embeddings[i], user_embeddings[j], user_step_index, e)
                # layer_loss = torch.max(max_loss_layer_lower.to(torch.device('cuda'), layer_loss))
                layer_loss = torch.max(max_loss_layer_lower, layer_loss).cuda()
            
            user_con_loss_list.append(layer_loss)     
        
        return user_con_loss_list, user_step_index  #4*[1024]

    def run(self):
     
        self.prepareModel()
        if args.isload == True:
            print("----------------------pre test:") 
            HR, NDCG = self.testEpoch(self.test_loader)
            print(f"HR: {HR} , NDCG: {NDCG}")
            
        log('Model Prepared')
        cvWait = 0  
        self.best_HR = 0 
        self.best_NDCG = 0
        flag = 0

        self.user_embed = None 
        self.item_embed = None
        self.user_embeds = None
        self.item_embeds = None


        # print("Test before train:")
        # HR, NDCG = self.testEpoch(self.test_loader)
        self.curEpoch = 0
        for e in range(self.curEpoch, args.epoch+1):  
            # if e == 10:
            #     sys.exit()
            self.curEpoch = e

            self.meta_flag = 0
            if e%args.meta_slot == 0:
                self.meta_flag=1


            log("*****************Start epoch: %d ************************"%e)  
            if args.isJustTest == False:
                # epoch_loss, user_embed, item_embed, user_embeds, item_embeds, final_lst, final_lst3 = self.trainEpoch()
                epoch_loss, user_embed, item_embed, user_embeds, item_embeds = self.trainEpoch(e)
                self.train_loss.append(epoch_loss)  
                print(f"epoch {e/args.epoch},  epoch loss{epoch_loss}")
                self.train_loss.append(epoch_loss)
            else:
                break

            HR, NDCG = self.testEpoch(self.test_loader)
            self.his_hr.append(HR)
            self.his_ndcg.append(NDCG)

            self.beh_model.scheduler.step()
            # self.prop_model_scheduler.step()
            # self.meta_scheduler.step()

            if HR > self.best_HR:
                self.best_HR = HR
                self.best_epoch = self.curEpoch 
                cvWait = 0
                print("--------------------------------------------------------------------------------------------------------------------------best_HR", self.best_HR)
                # print("--------------------------------------------------------------------------------------------------------------------------NDCG", self.best_NDCG)
                self.user_embed = user_embed 
                self.item_embed = item_embed
                self.user_embeds = user_embeds
                self.item_embeds = item_embeds

                self.saveHistory()
                self.saveModel()
            
            if NDCG > self.best_NDCG:
                self.best_NDCG = NDCG
                self.best_epoch = self.curEpoch 
                cvWait = 0
                # print("--------------------------------------------------------------------------------------------------------------------------HR", self.best_HR)
                print("--------------------------------------------------------------------------------------------------------------------------best_NDCG", self.best_NDCG)
                self.user_embed = user_embed 
                self.item_embed = item_embed
                self.user_embeds = user_embeds
                self.item_embeds = item_embeds

                self.saveHistory()
                self.saveModel()

            if (HR<self.best_HR) and (NDCG<self.best_NDCG): 
                cvWait += 1


            if cvWait == args.patience:
                print(f"Early stop at {self.best_epoch} :  best HR: {self.best_HR}, best_NDCG: {self.best_NDCG} \n")
                # self.saveHistory()
                # self.saveModel()
                break
               
        HR, NDCG = self.testEpoch(self.test_loader)
        self.his_hr.append(HR)
        self.his_ndcg.append(NDCG)

    def negSamp(self, temLabel, sampSize, nodeNum):
        negset = [None] * sampSize
        cur = 0
        while cur < sampSize:
            rdmItm = np.random.choice(nodeNum)
            if temLabel[rdmItm] == 0:
                negset[cur] = rdmItm
                cur += 1
        return negset

    def sampleTrainBatch(self, batIds, labelMat):
        # batchIds : meta data(user) <-- using meta_multi_single_beh_user_index_shuffle
        # labelMat : behavior data about each behavior type
        temLabel = labelMat[batIds.cpu()].toarray()
        batch = len(batIds)
        user_id = [] 
        item_id_pos = [] 
        item_id_neg = [] 
 
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
            # args.sampNum = 40
            sampNum = min(args.sampNum, len(posset))   
            if sampNum == 0:
                poslocs = [np.random.choice(labelMat.shape[1])]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = self.negSamp(temLabel[i], sampNum, labelMat.shape[1])

            for j in range(sampNum):
                user_id.append(batIds[i].item())
                item_id_pos.append(poslocs[j].item()) 
                item_id_neg.append(neglocs[j])
                cur += 1

        return torch.as_tensor(np.array(user_id)).cuda(), torch.as_tensor(np.array(item_id_pos)).cuda(), torch.as_tensor(np.array(item_id_neg)).cuda() 

    def trainEpoch(self, e):   
        train_loader = self.train_loader
        time = datetime.datetime.now()
        print("start_ng_samp:  ", time)
        train_loader.dataset.ng_sample()
        time = datetime.datetime.now()
        print("end_ng_samp:  ", time)
        
        epoch_loss = 0
        
#-----------------------------------------------------------------------------------
        self.behavior_loss_list = [None]*len(self.behaviors)      

        self.user_id_list = [None]*len(self.behaviors)
        self.item_id_pos_list = [None]*len(self.behaviors)
        self.item_id_neg_list = [None]*len(self.behaviors)

        self.meta_start_index = 0
        self.meta_end_index = self.meta_start_index + args.meta_batch
#----------------------------------------------------------------------------------

        cnt = 0
        tmp_cnt = 0
        for user, item_i, item_j in tqdm(train_loader):
            user = user.long().cuda()
            self.user_step_index = user
            # self.meta_user = torch.as_tensor(self.meta_multi_single[self.meta_start_index:self.meta_end_index]).cuda()  
            
            # if self.meta_end_index == self.meta_multi_single.shape[0]:
            #     self.meta_start_index = 0  
            # else:
            #     self.meta_start_index = (self.meta_start_index + args.meta_batch) % (self.meta_multi_single.shape[0] - 1)
            # self.meta_end_index = min(self.meta_start_index + args.meta_batch, self.meta_multi_single.shape[0])
        

#---round one---------------------------------------------------------------------------------------------

            beh_bpr_loss_list = [None]*len(self.behaviors)
            beh_user_index_list = [None]*len(self.behaviors)  #---

            # beh_model = BGNN.myModel(self.user_num, self.item_num, self.behaviors, self.behavior_mats).cuda()
            # beh_model_opt = torch.optim.AdamW(beh_model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
            # beh_model_scheduler = torch.optim.lr_scheduler.CyclicLR(beh_model_opt, args.opt_base_lr, args.opt_max_lr, step_size_up=5, step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
            # # beh_model.load_state_dict(self.model.state_dict())
            
            # prop_model = PropModel().cuda()
            # prop_model_opt = torch.optim.AdamW(prop_model.parameters(), lr = args.prop_opt_base_lr, weight_decay = args.opt_weight_decay)
            # prop_model_scheduler = torch.optim.lr_scheduler.CyclicLR(prop_model_opt, args.prop_opt_base_lr, args.prop_opt_max_lr, step_size_up=3, step_size_down=7, mode='triangular', gamma=0.98, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.9, max_momentum=0.99, last_epoch=-1)            
            
            beh_user_embed, beh_item_embed, beh_user_embeds, beh_item_embeds = self.beh_model()
            
            for index in range(len(self.behaviors)):

                not_zero_index = np.where(item_i[index].cpu().numpy()!=-1)[0]

                self.user_id_list[index] = user[not_zero_index].long().cuda()
                beh_user_index_list[index] = self.user_id_list[index]
                self.item_id_pos_list[index] = item_i[index][not_zero_index].long().cuda()
                self.item_id_neg_list[index] = item_j[index][not_zero_index].long().cuda()
                
                beh_userEmbed = beh_user_embed[self.user_id_list[index]] 
                beh_posEmbed = beh_item_embed[self.item_id_pos_list[index]] 
                beh_negEmbed = beh_item_embed[self.item_id_neg_list[index]] 
                
                beh_pred_i, beh_pred_j = 0, 0
                beh_pred_i, beh_pred_j = self.innerProduct(beh_userEmbed, beh_posEmbed, beh_negEmbed)
                beh_bpr_loss_list[index] = - (beh_pred_i.view(-1) - beh_pred_j.view(-1)).sigmoid().log()

            new_userEmbed = self.prop_model(beh_user_embeds)
            
            # meta_infoNCELoss_list => infoNCE Loss with target behavior type <-> target + auxililary behavior type
            # meta_infoNCELoss_list, SSL_user_step_index = self.SSL(beh_user_embeds, beh_item_embeds, beh_user_embed, beh_item_embed, self.user_step_index)  
            meta_infoNCELoss_list, SSL_user_step_index = self.SSL(new_userEmbed, beh_item_embeds, beh_user_embed, beh_item_embed, self.user_step_index, e)  
            
            tmp_bpr_loss = 0
            tmp_info_loss = 0 

            for i in range(len(beh_bpr_loss_list)):
                # tmp_bpr_loss = tmp_bpr_loss + torch.sum(beh_bpr_loss_list[i]).cuda() * (1/len(beh_bpr_loss_list))
                tmp_bpr_loss = tmp_bpr_loss + torch.sum(beh_bpr_loss_list[i]).cuda() * 0.129
            for i in range(len(meta_infoNCELoss_list)):
                # tmp_info_loss = tmp_info_loss + torch.sum(meta_infoNCELoss_list[i]).cuda() * (1/len(beh_bpr_loss_list))
                tmp_info_loss = tmp_info_loss + torch.sum(meta_infoNCELoss_list[i]).cuda() * 0.129
            
            meta_bprloss = tmp_bpr_loss / len(beh_bpr_loss_list)
            meta_infoNCELoss = tmp_info_loss / len(meta_infoNCELoss_list)
            # meta_bprloss = tmp_bpr_loss
            # meta_infoNCELoss = tmp_info_loss
            
            meta_regLoss = (torch.norm(beh_userEmbed) ** 2 + torch.norm(beh_posEmbed) ** 2 + torch.norm(beh_negEmbed) ** 2)            

            meta_model_loss = (meta_bprloss + args.reg * meta_regLoss + args.beta*meta_infoNCELoss) / args.batch
            
            # print("loss test")
            # print(meta_model_loss)
            epoch_loss = meta_model_loss.item()
            
            self.beh_model.opt.zero_grad(set_to_none=True)
            # self.prop_model_opt.zero_grad(set_to_none = True)
            
            # print(meta_model_loss)
            meta_model_loss.backward()
            
            nn.utils.clip_grad_norm_(self.beh_model.parameters(), max_norm=20, norm_type=2)
            # nn.utils.clip_grad_norm_(self.prop_model.parameters(), max_norm=20, norm_type=2)
            
            self.beh_model.opt.step()
            # self.prop_model_opt.step()
            
#---round one---------------------------------------------------------------------------------------------

            cnt+=1
            tmp_cnt += 1
        
        return epoch_loss, beh_user_embed, beh_item_embed, new_userEmbed, beh_item_embeds


    def testEpoch(self, data_loader, save=False):
        #data_loader = tst_int in each dataset
        epochHR, epochNDCG = [0]*2
        with torch.no_grad():
            user_embed, item_embed, user_embeds, item_embeds = self.beh_model()

        cnt = 0
        tot = 0
        # data_loader => user와 pos_item만 존재
        for user, item_i in data_loader: #TMall) len(user) = len(item_i) = 8,192
            # user_compute : 테스트 데이터셋의 사용자 인덱스(batch * 100)
            # item_compute : user_compute의 인덱스 사용자와 연결된 99개의 neg_item + 1개의 pos_item
            # user_item1 : test data에서 사용자 당 실제로 연결된 pos_item - list : 사용자 당 1개의 아이템
            # user_item100 : 사용자 당 실제로 연결된 pos_item과 neg_item : 사용자 당 100개의 아이템
            user_compute, item_compute, user_item1, user_item100 = self.sampleTestBatch(user, item_i)
            
            userEmbed = user_embed[user_compute]  #[614400, 16], [147894, 16]
            itemEmbed = item_embed[item_compute]
            # TMall) userEmbed.shape = itemEmbed.shape = torch.Size[819200, 16]
            pred_i = torch.sum(torch.mul(userEmbed, itemEmbed), dim=1) # TMall) pred_i.shape = torch.Size[819200]
            # t.reshape(pred_i, [user.shape[0], 100]) => [8192, 100]

            hit, ndcg = self.calcRes(torch.reshape(pred_i, [user.shape[0], 100]), user_item1, user_item100, user)  
            epochHR = epochHR + hit  
            epochNDCG = epochNDCG + ndcg  #
            cnt += 1 
            tot += user.shape[0]


        result_HR = epochHR / tot
        result_NDCG = epochNDCG / tot
    
        print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}")

        return result_HR, result_NDCG

    def calcRes(self, pred_i, user_item1, user_item100, user):  

        hit = 0
        ndcg = 0
        err_cnt =0
        print("Starting test for each user")
        pos_score, neg_score = [], []
        pos_user, neg_user = [], []
        # pos_user_a, neg_user_a = [], []
        for j in range(pred_i.shape[0]): # Tmall) pred_i.shape[0] = 8192
            # 아랫 라인 : KGIN의 evaluate.py > ranklist_by_heapq의 K_max_item_score를 뽑는 부분과 동일
            # shoot_index : 임베딩 점수가 높은 최상위 10개 tensor의 인덱스
            # args.shoot = 10
            # shoot_value, shoot_index = t.topk(pred_i[j], args.shoot)
            shoot_value, shoot_index = torch.topk(pred_i[j], 10)
            shoot_value, tmp_shoot_index = torch.topk(pred_i[j], 100)
            
            shoot_index = shoot_index.cpu() # len(shoot_index) = 10
            shoot = user_item100[j][shoot_index]
            shoot = shoot.tolist()

            tmp_shoot_index = tmp_shoot_index.cpu()
            tmp_shoot = user_item100[j][tmp_shoot_index]
            tmp_shoot = tmp_shoot.tolist()

            tmp_ndcg = np.reciprocal( np.log2( tmp_shoot.index( user_item1[j])+2))
            if type(shoot)!=int and (user_item1[j] in shoot):  
                hit += 1 
                ndcg += np.reciprocal( np.log2( shoot.index( user_item1[j])+2))
                # print("ndcg: ", np.reciprocal( np.log2( shoot.index( user_item1[j])+2)))
                # print("Case 1")
            elif type(shoot)==int and (user_item1[j] == shoot):
                hit += 1  
                ndcg += np.reciprocal( np.log2( 0+2))
            else:
                # print("Error!")
                err_cnt += 1
                # sys.exit()
        
        print("#Total Error: ", err_cnt)
        print("Error rate: ", err_cnt / pred_i.shape[0])
        print("hit: ", hit)
        print("ndcg: ", ndcg)
    
        return hit, ndcg  #int, float


    def sampleTestBatch(self, batch_user_id, batch_item_id):
       # batch_user_id : user index 
       # batch_item_id : positive item index
        batch = len(batch_user_id) # TMall) batch = 8192
        tmplen = (batch*100) # TMall) tmplen = 819200

        sub_trainMat = self.trainMat[batch_user_id].toarray()
        # sub_trainMat.shape = (8192, #item(=31232))
        user_item1 = batch_item_id 
        user_compute = [None] * tmplen
        item_compute = [None] * tmplen
        user_item100 = [None] * (batch)

        cur = 0
        for i in range(batch):
            pos_item = user_item1[i]
            negset = np.reshape(np.argwhere(sub_trainMat[i]==0), [-1])
            # negset => target edge type에서 positive item을 제외한 negative item의 candidate set(negative item의 index)
            
            # numpy.random.permutation => 무작위로 섞인 숫자 배열 return
            random_neg_sam = np.random.permutation(negset)[:99]
            user_item100_one_user = np.concatenate(( random_neg_sam, np.array([pos_item]))) 
            user_item100[i] = user_item100_one_user
            
            for j in range(100):
                user_compute[cur] = batch_user_id[i]
                item_compute[cur] = user_item100_one_user[j]
                cur += 1
        
        return user_compute, item_compute, user_item1, user_item100


    def setRandomSeed(self):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)

    def getModelName(self):  
        title = args.title
        ModelName = \
        args.point + \
        "_" + title + \
        "_" +  args.dataset +\
        "_" + modelTime + \
        "_lr_" + str(args.lr) + \
        "_reg_" + str(args.reg) + \
        "_batch_size_" + str(args.batch) + \
        "_gnn_layer_" + str(args.gnn_layer)

        return ModelName

    def saveHistory(self):  
        history = dict()
        history['loss'] = self.train_loss  
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        ModelName = self.modelName

        with open(r'./History/' + args.dataset + r'/' + ModelName + '.his', 'wb') as fs: 
            pickle.dump(history, fs)

    def saveModel(self):  
        ModelName = self.modelName

        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        savePath = r'./Model/' + args.dataset + r'/' + ModelName + r'.pth'
        params = {
            'epoch': self.curEpoch,
            # 'lr': self.lr,
            'model': self.beh_model,
            'prop_model': self.prop_model,
            # 'meta_weight_net' : self.meta_weight_net,
            # 'reg': self.reg,
            'history': history,
            'user_embed': self.user_embed,
            'user_embeds': self.user_embeds,
            'item_embed': self.item_embed,
            'item_embeds': self.item_embeds,
        }
        torch.save(params, savePath)

    def loadModel(self, loadPath):      
        ModelName = self.modelName
        # loadPath = r'./Model/' + args.dataset + r'/' + ModelName + r'.pth'
        loadPath = loadPath
        checkpoint = torch.load(loadPath)
        self.model = checkpoint['model']
        # self.meta_weight_net = checkpoint['meta_weight_net']
        self.prop_model = checkpoint['prop_model']
        self.curEpoch = checkpoint['epoch'] + 1
        # self.lr = checkpoint['lr']
        # self.args.reg = checkpoint['reg']
        history = checkpoint['history']
        self.train_loss = history['loss']
        self.his_hr = history['HR']
        self.his_ndcg = history['NDCG']
        # log("load model %s in epoch %d"%(modelPath, checkpoint['epoch']))
        self.user_embed = checkpoint['user_embed']
        self.user_embeds = checkpoint['user_embeds']
        self.item_embed = checkpoint['item_embed']
        self.item_embeds = checkpoint['item_embeds']
    

if __name__ == '__main__':
    setproctitle("TY-model")
    my_model = Model()  
    my_model.run()
    # my_model.test()

