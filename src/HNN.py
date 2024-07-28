#  超图，用来聚合静态图，来发现实体之间的相似性
import torch
# smc
torch.backends.cudnn.enabled = False
import torch.nn as nn
from torch_scatter import scatter
import torch.nn.functional as F


class Hnn(nn.Module):
    def __init__(self, num_ents, num_rels, ent_emb, rel_emb, hidden_dim, dropout, layers, use_cuda, gpu):
        super(Hnn, self).__init__()
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.ent_emb = ent_emb
        self.rel_emb = rel_emb
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.layers = layers
        self.use_cuda = use_cuda
        self.gpu = gpu

        self.w_r = nn.Linear(self.hidden_dim,  self.hidden_dim)
        self.w_u2r = nn.Linear(self.hidden_dim,  self.hidden_dim)
        self.w_u = nn.Linear(self.hidden_dim,  self.hidden_dim)
        self.w_r2u = nn.Linear(self.hidden_dim,  self.hidden_dim)
        self.u_dp = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, index_list):
        index = index_list[0]
        self.g_h_last = F.normalize(self.ent_emb)
        self.g_h_0_last = F.normalize(self.rel_emb)
        for i in range(self.layers):
            # self.g_h = self.g_h_update(self.g_h)
            # self.g_h_0 = self.g_h_0_update(self.g_h_0)
            # print(self.rel_emb.device)
            self.g_h = self.g_h_last
            self.g_h_0 = self.g_h_0_last
            # self.g_h_last = self.g_h.clone()
            # self.g_h_0_last = self.g_h_0.clone()
            # 第一步，self_loop
            # self.g_h = self.loop(self.g_h)
            # 第二步，节点消息传入到边上
            index = index.long().to(self.gpu) if self.use_cuda else index.long()  # index[[s,s,s,s,s], [r,r,r,r,r]]
            u_unique = torch.unique(index[0])
            r_unique = torch.unique(index[1])
            u_emb = self.g_h[index[0]]       # 头实体的嵌入
            r_emb = self.g_h_0[r_unique]  # 出现的边的嵌入
            r = index[1]
            u2r = scatter(u_emb, r, dim=0, dim_size=self.num_rels * 2, reduce='mean')    # u2r表示的是头节点传入每个边的消息，如果对应的消息为0，则表示没有头实体传入该边
            u2r = self.w_u2r(u2r)

            r_emb = self.w_r(r_emb)  # 更新出现的边的嵌入
            r_res = u2r[r_unique]
            r_res = F.relu(r_res)
            self.g_h_0[r_unique] = r_res * 1/2 + r_emb * 1/2
            # 第三步，边传入节点
            r_emb = self.g_h_0[index[1]]
            u_emb = self.g_h[u_unique]
            u = index[0]
            r2u = scatter(r_emb, u, dim=0, dim_size=self.num_ents, reduce='mean')
            r2u = self.w_r2u(r2u)
            u_emb = self.w_u(u_emb)
            u_res = r2u[u_unique]
            u_res = F.relu(u_res)
            self.g_h[u_unique] = 1 / 2 * u_res+ 1 / 2 * u_emb
            # self.g_h = F.relu(self.u_dp(self.g_h))

            self.g_h_last = F.normalize(self.g_h)
            self.g_h_0_last = F.normalize(self.g_h_0)

        return self.g_h_last