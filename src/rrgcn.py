import math
import torch
# smc
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm
from torch_scatter import scatter

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.hgls.hrgnn import HRGNN
from src.model import BaseRGCN
from src.decoder import *
from HNN import Hnn


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "convgcn":
            return UnionRGCNLayer(self.out_dim, self.out_dim, self.num_rels, self.num_bases,
                                  activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc,
                                  rel_emb=self.rel_emb)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb, method):
        if self.encoder_name == "convgcn":
            if method == 0:
                # print(g.ndata['id'])
                node_id = g.ndata['id'].squeeze()
                g.ndata['h'] = init_ent_emb[node_id]
            elif method == 1:
                g.ndata['h'] = init_ent_emb
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                # print("层数", self.layers)
                layer(g, [], r[i])  # 关键代码！！！！！！！！！！！！！
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, num_times,
                 time_interval, h_dim, opn, history_rate, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_prelearning=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu=0, alpha=0.2, analysis=False, graph=None, long_con=None):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.history_rate = history_rate
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.num_times = num_times
        self.time_interval = time_interval
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_prelearning = use_prelearning
        self.alpha = alpha
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        # self.emb_rel = None
        self.gpu = gpu
        self.sin = torch.sin
        self.linear_0 = nn.Linear(num_times, 1)
        self.linear_1 = nn.Linear(num_times, self.h_dim - 1)
        self.tanh = nn.Tanh()
        self.use_cuda = None

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        # self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2 + 1, self.h_dim), requires_grad=True).float()
        # torch.nn.init.xavier_normal_(self.emb_rel)
        #
        # self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, self.h_dim), requires_grad=True).float()
        # torch.nn.init.xavier_normal_(self.dynamic_emb)

        self.dynamic_emb = nn.Embedding(self.num_ents, self.h_dim)
        self.emb_rel = nn.Embedding(self.num_rels * 2 + 1, self.h_dim)
        # torch.nn.init.xavier_normal_(self.dynamic_emb.weight)
        torch.nn.init.normal_(self.dynamic_emb.weight)
        torch.nn.init.xavier_normal_(self.emb_rel.weight)

        # smc
        self.time_emb = torch.nn.Parameter(torch.Tensor(sequence_len, 50), requires_grad=True).float()
        torch.nn.init.normal_(self.time_emb)

        # 用于计算时间向量
        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))

        if self.use_prelearning:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, self.h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels * 2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False,
                                                    skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()
        # 时序图
        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)
        # 时序超图
        self.rgcn0 = RGCNCell(num_ents,
                              h_dim,
                              h_dim,
                              num_rels * 2,
                              num_bases,
                              num_basis,
                              num_hidden_layers,
                              dropout,
                              self_loop,
                              skip_connect,
                              encoder_name,
                              self.opn,
                              self.emb_rel,
                              use_cuda,
                              analysis)

        # 用于预学习图的超图卷积
        self.hnn = Hnn(self.num_ents, self.num_rels, self.dynamic_emb.weight, self.emb_rel.weight, self.h_dim, dropout, num_hidden_layers, use_cuda, gpu)

        # hgls粗粒度历史
        self.model_t = HRGNN(graph=graph, num_nodes=num_ents, num_rels=num_rels, **long_con)
        # 解开下面的注释使用多头注意力机制
        # self.model_t1 = HRGNN(graph=graph, num_nodes=num_ents, num_rels=num_rels, **long_con)
        # self.model_t2 = HRGNN(graph=graph, num_nodes=num_ents, num_rels=num_rels, **long_con)
        # self.model_t3 = HRGNN(graph=graph, num_nodes=num_ents, num_rels=num_rels, **long_con)
        # self.model_t4 = HRGNN(graph=graph, num_nodes=num_ents, num_rels=num_rels, **long_con)
        self.model_t.aggregator = self.rgcn   # 并没有用到
        self.model_t.en_embedding = self.dynamic_emb
        # 解开下面的注释使用多头注意力机制
        # self.model_t1.en_embedding = self.dynamic_emb
        # self.model_t2.en_embedding = self.dynamic_emb
        # self.model_t3.en_embedding = self.dynamic_emb
        # self.model_t4.en_embedding = self.dynamic_emb
        self.model_t.rel_embedding = self.emb_rel
        # 解开下面的注释使用多头注意力机制
        # self.model_t1.rel_embedding = self.emb_rel
        # self.model_t2.rel_embedding = self.emb_rel
        # self.model_t3.rel_embedding = self.emb_rel
        # self.model_t4.rel_embedding = self.emb_rel
        self.hgls_gate = GatingMechanism(self.num_ents, self.h_dim)
        self.hgls_gate_r = GatingMechanism(self.num_rels * 2, self.h_dim)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        # add
        self.global_weight = nn.Parameter(torch.Tensor(self.num_ents, 1))
        nn.init.xavier_uniform_(self.global_weight, gain=nn.init.calculate_gain('relu'))
        self.global_bias = nn.Parameter(torch.Tensor(1))
        nn.init.zeros_(self.global_bias)

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim * 2, self.h_dim)
        self.entity_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)

        # decoder
        if decoder_name == "timeconvtranse":
            self.decoder_ob1 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob2 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob3 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re1 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re2 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re3 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        elif decoder_name == "conve":
            self.decoder_ob1 = ConvE(num_ents)
            self.decoder_ob2 = ConvE(num_ents)
            self.rdecoder_re1 = ConvR(num_rels * 2)
            self.rdecoder_re2 = ConvR(num_rels * 2)
        else:
            raise NotImplementedError

        # smc
        # self.h_staic = torch.nn.Parameter(torch.Tensor(num_ents, self.h_dim), requires_grad=True).float()
        # torch.nn.init.xavier_normal_(self.h_staic)

        # smc
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 1)
        self.bceloss = nn.BCELoss()

        self.f_cat = nn.Linear(self.h_dim * 5, self.h_dim)

        # # 超图卷积
        # self.g_h_update = nn.Linear(200, 200, bias=False)
        # self.g_h_0_update = nn.Linear(200, 200, bias=False)
        # self.loop = nn.Linear(200, 200, bias=False)
        # self.evolve = nn.Linear(200, 200, bias=False)
        # self.w_u2r = nn.Linear(200, 200, bias=False)
        # self.w_r = nn.Linear(200, 200, bias=False)
        # self.w_r2u = nn.Linear(200, 200, bias=False)
        # self.w_u = nn.Linear(200, 200, bias=False)
        # self.entity_cell_g = nn.GRUCell(200, 200)
        # self.relation_cell_g = nn.GRUCell(200, 200)
        # self.u_gate = GatingMechanism(self.num_ents, 200)
        # self.r_gate = GatingMechanism(self.num_rels * 2, 200)
        # self.u_dp = nn.Dropout(0.2)
        # self.r_dp = nn.Dropout(0.2)
        # self.gate = nn.Parameter(torch.Tensor(1, 200))

    def forward(self, g_list, static_graph, use_cuda, g_list0, predict_time, index_list, glist0):  # g_list0 背景图、 predict_time 预测的时间、 index_list 超图神经网络的索引、 glist0 时空同步图、 batch 供查询使用的batch
        gate_list = []
        degree_list = []
        if self.use_prelearning:
            h_rel = F.normalize(self.emb_rel.weight[0:self.num_rels*2]) if self.layer_norm else self.emb_rel
            # self.h = self.rgcn0.forward(g_list0.to(self.gpu), self.h, [h_rel, h_rel], self.time_emb[0], type=0)
            self.h = self.rgcn0.forward(g_list0.to(self.gpu), self.dynamic_emb.weight, [h_rel, h_rel], 0)
            # self.h = self.rgcn0.forward(g_list0, self.dynamic_emb.weight, [h_rel, h_rel], self.time_emb[0])
            self.h = F.normalize(self.h) if self.layer_norm else self.h

            self.h_gnn = self.hnn(index_list)
            self.h = (1-self.alpha) * self.h + self.alpha * self.h_gnn
            self.h = F.normalize(self.h) if self.layer_norm else self.h
        else:
            self.h = F.normalize(self.dynamic_emb.weight) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None
        # self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
        # self.h_0 = F.normalize(self.emb_rel.weight[0:self.num_rels * 2]) if self.layer_norm else self.emb_rel

        history_embs = []
        # tcn_emb_times = []
        global_embs = []

        # 主要计算local图
        for i, g in enumerate(g_list):
            g = g.to(self.gpu) if use_cuda else g
            if i == 0:
                # x_input = torch.cat((self.emb_rel, x_input), dim=1)
                # self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
                # self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                self.h_0 = F.normalize(self.emb_rel.weight[0:self.num_rels*2]) if self.layer_norm else self.emb_rel
            else:
                # x_input = torch.cat((self.emb_rel, x_input), dim=1)
                # self.h_0 = self.relation_cell_1(x_input, self.h_0)
                # 最优秀模型使用
                self.h_0 = F.normalize(self.emb_rel.weight[0:self.num_rels*2]) if self.layer_norm else self.emb_rel
                # self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0

            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0], 0)
            current_h = F.normalize(current_h) if self.layer_norm else current_h

            self.h = self.entity_cell_1(current_h, self.h)
            self.h = F.normalize(self.h) if self.layer_norm else self.h  # 这个叫归一化，就是将数据映射到0-1
            # self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0

            history_embs.append(self.h)

        return history_embs, history_embs[0], self.h_0, gate_list, degree_list, global_embs

        # gate_list = []
        # degree_list = []
        #
        #
        # self.h = F.normalize(self.dynamic_emb.weight)
        # self.h_0 = F.normalize(self.emb_rel.weight[0:self.num_rels * 2])
        #
        # history_embs = []
        # # tcn_emb_times = []
        # global_embs = []
        #
        # history_embs.append(self.h)
        #
        # return history_embs, history_embs[0], self.h_0, gate_list, degree_list, global_embs

    def predict(self, test_graph, num_rels, static_graph, test_triplets, entity_history_vocabulary,
                rel_history_vocabulary, use_cuda, g_list, predict_time, index_list, glist0, entity_local_vocabulary, data_list, node_id_new, time_gap):
        self.use_cuda = use_cuda
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            # all_triples = test_triplets

            evolve_embs, _, r_emb, _, _, _ = self.forward(test_graph, static_graph, use_cuda, g_list, predict_time,
                                                          index_list, glist0)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            time_embs = self.get_init_time(all_triples)

            # smc
            # p = self.caculate_show(embedding)
            # /smc

            # smc hgls粗粒度历史
            new_embedding = F.normalize(self.model_t(data_list, node_id_new, time_gap, self.gpu, mode='test'))
            # 解开下面的代码将会使用多头注意力
            # new_embedding1 = F.normalize(self.model_t1(data_list, node_id_new, time_gap, self.gpu, mode='test'))
            # new_embedding2 = F.normalize(self.model_t2(data_list, node_id_new, time_gap, self.gpu, mode='test'))
            # new_embedding3 = F.normalize(self.model_t3(data_list, node_id_new, time_gap, self.gpu, mode='test'))
            # new_embedding4 = F.normalize(self.model_t4(data_list, node_id_new, time_gap, self.gpu, mode='test'))
            # new_embedding = torch.cat((new_embedding, new_embedding1, new_embedding2, new_embedding3, new_embedding4),
            #                           dim=1)
            # new_embedding = self.f_cat(new_embedding)
            new_r_embedding = self.model_t.rel_embedding.weight[0:self.num_rels * 2]
            # 融合
            embedding, e_cof = self.hgls_gate(embedding, new_embedding)
            r_emb, e_cof = self.hgls_gate_r(r_emb, new_r_embedding)
            # /smc

            # # 去掉时序图
            # embedding = new_embedding
            # r_emb = new_r_embedding

            # # smc 构造实体显现信息
            # index = torch.unique(all_triples[:, [0, 2]])
            # target = torch.zeros(self.num_ents)
            # target[index] = 1  # 这个就是出现的实体
            # target = target.to(self.gpu) if use_cuda else target
            # # /smc

            score_rel_r = self.rel_raw_mode(embedding, r_emb, time_embs, all_triples)
            score_rel_h = self.rel_history_mode(embedding, r_emb, time_embs, all_triples, rel_history_vocabulary)
            # score_r = self.raw_mode(embedding, r_emb, time_embs, all_triples) * target
            score_r = self.raw_mode(embedding, r_emb, time_embs, all_triples)
            # score_h = self.history_mode(embedding, r_emb, time_embs, all_triples, entity_history_vocabulary) * target
            score_h = self.history_mode(embedding, r_emb, time_embs, all_triples, entity_history_vocabulary)
            # score_l = self.history_local_mode(embedding, r_emb, time_embs, all_triples, entity_local_vocabulary) * target
            score_l = self.history_local_mode(embedding, r_emb, time_embs, all_triples, entity_local_vocabulary)

            # score_rel = score_rel_r                                     # !!!!!!!!!!
            score_rel = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
            score_rel = torch.log(score_rel)
            # score = self.history_rate * score_h + (1 - self.history_rate) * score_r
            # score = score_r     # !!!!!!!!!!!!!!!!!
            score = 0.3 * score_h + 0.5 * score_r + 0.2 * score_l
            # score = 0.7 * score_r + 0.3 * score_h
            score = torch.log(score)

            return all_triples, score, score_rel

    def get_loss(self, glist, triples, static_graph, entity_history_vocabulary, rel_history_vocabulary, use_cuda,
                 g_list, predict_time, index_list, glist0, entity_local_vocabulary,
                 data_list, node_id_new, time_gap):
        self.use_cuda = use_cuda
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0, 3]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu) if use_cuda else all_triples
        # all_triples = triples.to(self.gpu)

        # # 测试预测实体显现概率的上限
        # unique = torch.unique(all_triples[:, 0])
        # p0 = torch.zeros(self.num_ents, dtype=torch.float32)
        # p0 = p0.to(self.gpu)
        # p0.scatter_(0, unique, 1)

        # # smc 构造实体显现信息
        # index = torch.unique(triples[:, [0, 2]])
        # target = torch.zeros(self.num_ents)
        # target[index] = 1  # 这个就是出现的实体
        # target = target.to(self.gpu) if use_cuda else target
        # # /smc

        # 短期
        evolve_embs, static_emb, r_emb, _, _, global_embs = self.forward(glist, static_graph, use_cuda, g_list,
                                                                         predict_time, index_list, glist0)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        # smc hgls粗粒度历史
        new_embedding = F.normalize(self.model_t(data_list, node_id_new, time_gap, self.gpu, mode='test'))
        # 解开下面的代码将会使用多头注意力
        # new_embedding1 = F.normalize(self.model_t1(data_list, node_id_new, time_gap, self.gpu, mode='test'))
        # new_embedding2 = F.normalize(self.model_t2(data_list, node_id_new, time_gap, self.gpu, mode='test'))
        # new_embedding3 = F.normalize(self.model_t3(data_list, node_id_new, time_gap, self.gpu, mode='test'))
        # new_embedding4 = F.normalize(self.model_t4(data_list, node_id_new, time_gap, self.gpu, mode='test'))
        # new_embedding = torch.cat((new_embedding, new_embedding1, new_embedding2, new_embedding3, new_embedding4), dim=1)
        # new_embedding = self.f_cat(new_embedding)
        new_r_embedding = self.model_t.rel_embedding.weight[0:self.num_rels * 2]

        # 融合
        pre_emb, e_cof = self.hgls_gate(pre_emb, new_embedding)
        r_emb, e_cof = self.hgls_gate_r(r_emb, new_r_embedding)
        # /smc

        # # 去掉时序图
        # pre_emb = new_embedding
        # r_emb = new_r_embedding

        time_embs = self.get_init_time(all_triples)

        # smc
        # p = self.caculate_show(pre_emb)
        # loss_show = self.bceloss(p, target)
        # /smc

        if self.entity_prediction:
            score_r = self.raw_mode(pre_emb, r_emb, time_embs, all_triples)
            # score_r = self.raw_mode(pre_emb, r_emb, time_embs, all_triples) * target
            score_h = self.history_mode(pre_emb, r_emb, time_embs, all_triples, entity_history_vocabulary)
            # score_h = self.history_mode(pre_emb, r_emb, time_embs, all_triples, entity_history_vocabulary) * target
            score_l = self.history_local_mode(pre_emb, r_emb, time_embs, all_triples, entity_local_vocabulary)
            # score_l = self.history_local_mode(pre_emb, r_emb, time_embs, all_triples, entity_local_vocabulary) * target
            # score_en = self.history_rate * score_h + (1 - self.history_rate) * score_r
            # score_en = score_r   # !!!!!!!!!!!!!!!!!!!!
            #
            score_en = 0.3 * score_h + 0.5 * score_r + 0.2 * score_l
            # score_en = 0.3 * score_h + 0.7 * score_r
            # score_en = 0.7 * score_r + 0.3 * score_h

            scores_en = torch.log(score_en)
            loss_ent += F.nll_loss(scores_en, all_triples[:, 2])

        if self.relation_prediction:
            score_rel_r = self.rel_raw_mode(pre_emb, r_emb, time_embs, all_triples)
            score_rel_h = self.rel_history_mode(pre_emb, r_emb, time_embs, all_triples, rel_history_vocabulary)
            # score_re = score_rel_r  # !!!!!!!!!!!!!!!!!!!
            score_re = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
            scores_re = torch.log(score_re)
            loss_rel += F.nll_loss(scores_re, all_triples[:, 1])

        return loss_ent, loss_rel, loss_static #, loss_show

    def get_init_time(self, quadrupleList):
        T_idx = quadrupleList[:, 3] // self.time_interval
        T_idx = T_idx.unsqueeze(1).float()
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = self.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2

    def raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_ob = self.decoder_ob1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, self.num_ents)
        score = F.softmax(scores_ob, dim=1)
        return score

    def history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.decoder_ob2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=global_index)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h

    # smc
    def history_local_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.decoder_ob3.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=global_index)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h

    def rel_raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_re = self.rdecoder_re1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, 2 * self.num_rels)
        score = F.softmax(scores_re, dim=1)
        return score

    def rel_history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.rdecoder_re2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=global_index)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h

    # def caculate_show(self, ent_emb):
    #     x = self.fc1(ent_emb)
    #     x = F.sigmoid(x)
    #     x = self.fc2(x)
    #     x = F.sigmoid(x).view(-1)
    #     return x


class GatingMechanism(nn.Module):  # 这里是使用的对于每一个实体或者每一个关系，都使用特定的参数，可以借鉴下
    def __init__(self, entity_num, hidden_dim):
        super(GatingMechanism, self).__init__()
        # gating 的参数
        self.gate_theta = nn.Parameter(torch.empty(entity_num, hidden_dim))
        nn.init.xavier_uniform_(self.gate_theta)
        # self.dropout = nn.Dropout(self.params.dropout)

    def forward(self, X: torch.FloatTensor, Y: torch.FloatTensor):
        '''
        :param X:   LSTM 的输出tensor   |E| * H
        :param Y:   Entity 的索引 id    |E|,
        :return:    Gating后的结果      |E| * H
        '''
        gate = torch.sigmoid(self.gate_theta)
        output = torch.mul(gate, X) + torch.mul(-gate + 1, Y)
        return output, gate


# class Self_Attention(nn.Module):
#     # input : batch_size * seq_len * input_dim
#     # q : batch_size * input_dim * dim_k
#     # k : batch_size * input_dim * dim_k
#     # v : batch_size * input_dim * dim_v
#     def __init__(self, input_dim, dim_k, dim_v):
#         super(Self_Attention, self).__init__()
#         self.q = nn.Linear(input_dim, dim_k)
#         self.k = nn.Linear(input_dim, dim_k)
#         self.v = nn.Linear(input_dim, dim_v)
#         self._norm_fact = 1 / math.sqrt(dim_k)
#
#         self.l1 = nn.Linear(input_dim, input_dim)
#         self.l2 = nn.Linear(input_dim, input_dim)
#
#     def forward(self, x):
#         Q = self.q(x)  # Q: batch_size * seq_len * dim_k
#         K = self.k(x)  # K: batch_size * seq_len * dim_k
#         V = self.v(x)  # V: batch_size * seq_len * dim_v
#
#         atten = nn.Softmax(dim=-1)(
#             torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact  # Q * K.T() # batch_size * seq_len * seq_len
#
#         output = torch.bmm(atten, V)  # Q * K.T() * V # batch_size * seq_len * dim_v
#
#         output = F.normalize(output + x, dim=2)
#         output = self.l2(F.relu(self.l1(output)))
#         output = F.normalize(output + x, dim=2)
#
#         return output