import random

import torch.autograd

from layers import *


class VLTransformer(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(VLTransformer, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.d_out = self.d_v * self.n_head * self.modal_num

        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []

        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)

            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        d_in = self.d_v * self.n_head * self.modal_num
        self.Outputlayer = OutputLayer(d_in, self.d_v * self.n_head, hyperpm=self.hyperpm)

    # feat_attn是可视化时用的探针，使用时请传入一个空的Tensor
    def forward(self, x, feat_attn=None):
        bs = x.size(0)
        attn_map = []
        x, _attn = self.InputLayer(x)
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())

        # 随机选择一个模态
        mask_idx = random.randint(0, self.modal_num - 1)
        x_mask = x.clone()
        x_mask[:, mask_idx, :] = 0

        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())

        for i in range(self.n_layer):
            x_mask, _ = self.Encoder[i](q=x_mask, k=x_mask, v=x_mask, modal_num=self.modal_num)
            x_mask = self.FeedForward[i](x_mask)

        x = x.view(bs, -1)
        x_mask = x.view(bs, -1)
        if feat_attn is not None:
            feat_attn.data = x.data
        attn_embedding = attn.view(bs, -1)
        output, hidden = self.Outputlayer(x, attn_embedding)
        _, hidden_mask = self.Outputlayer(x_mask, attn_embedding)
        # output为Ysp结果,hidden为Hsp拼接Hsh,attn_map为attention结果序列
        return output, hidden, hidden_mask, attn_map


class GraphLearn(nn.Module):
    def __init__(self, input_dim, hyperpm):
        super(GraphLearn, self).__init__()
        self.mode = hyperpm.GC_mode
        self.th = hyperpm.th
        self.dataname = hyperpm.dataname
        self.w = nn.Linear(input_dim, 1)
        self.t = nn.Parameter(torch.ones(1))

        self.Linears = []
        self.gl_lin_layer = hyperpm.gl_lin_layer
        for i in range(self.gl_lin_layer):
            linear = nn.Linear(input_dim, input_dim)

            self.add_module('gl_linear_%d' % i, linear)
            self.Linears.append(linear)

        self.Encoder = []
        self.FeedForward = []
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.gl_attn_layer = hyperpm.gl_attn_layer

        for i in range(self.gl_attn_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)

            self.add_module('gl_encode_%d' % i, encoder)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.add_module('gl_feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)

        self.threshold = nn.Parameter(torch.zeros(1))

    def forward(self, x, feat_gllin=None, feat_glattn=None):
        initial_x = x.clone()
        num, feat_dim = x.size(0), x.size(1)

        output = None
        if self.mode == "sigmoid-like":
            x = x.repeat_interleave(num, dim=0)
            x = x.view(num, num, feat_dim)
            diff = abs(x - initial_x)
            diff = diff.pow(2).sum(dim=2).pow(1 / 2)
            diff = (diff + self.threshold) * self.t
            output = 1 - torch.sigmoid(diff)

        elif self.mode == "adaptive-learning":
            x = x.repeat_interleave(num, dim=0)
            x = x.view(num, num, feat_dim)
            diff = abs(x - initial_x)
            diff = F.relu(self.w(diff)).view(num, num)
            output = F.softmax(diff, dim=1)

        elif self.mode == 'weighted-cosine':
            th = self.th
            for i in range(self.gl_lin_layer):
                x = self.Linears[i](x)
            if feat_gllin is not None:
                feat_gllin.data = x.data
            x = x.unsqueeze(1)
            for i in range(self.gl_attn_layer):
                x, _ = self.Encoder[i](q=x, k=x, v=x, modal_num=1)  # 模态已融合
                x = self.FeedForward[i](x)
            x = x.view(num, -1)
            if feat_glattn is not None:
                feat_glattn.data = x.data
            x_norm = F.normalize(x, dim=-1)
            score = torch.matmul(x_norm, x_norm.T)
            mask = (score > th).detach().float()
            markoff_value = 0
            output = score * mask + markoff_value * (1 - mask)

        return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x3 = self.gc2(x2, adj)
        return F.log_softmax(x3, dim=1), x2


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttConv(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttConv(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1), x
