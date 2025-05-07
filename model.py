import torch.autograd
import torch.optim as optim

from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from utils import *
from network import *


class EvalHelper:
    def __init__(self, input_data_dims, feat, label, hyperpm, train_index, test_index):
        use_cuda = torch.cuda.is_available()
        dev = torch.device('cuda' if use_cuda else 'cpu')
        feat = torch.from_numpy(feat).float().to(dev)
        label = torch.from_numpy(label).long().to(dev)
        self.dev = dev
        self.hyperpm = hyperpm
        self.GC_mode = hyperpm.GC_mode
        self.MP_mode = hyperpm.MP_mode
        self.marl_mode = hyperpm.marl_mode
        self.CL_mode = hyperpm.CL_mode
        self.d_v = hyperpm.n_hidden
        self.modal_num = hyperpm.nmodal
        self.nclass = hyperpm.nclass
        self.dropout = hyperpm.dropout
        self.alpha = hyperpm.alpha
        self.CL_mse = hyperpm.CL_mse
        self.CL_weight = hyperpm.CL_weight
        self.CL_rate = hyperpm.CL_rate
        self.n_head = hyperpm.n_head
        self.th = hyperpm.th
        self.lg = hyperpm.lg
        self.feat = feat
        self.targ = label
        self.trn_idx = train_index
        self.val_idx = np.array(test_index)
        self.tst_idx = np.array(test_index)

        # 标签权重计算
        trn_label = label[self.trn_idx].cpu().numpy()
        counter = Counter(trn_label)
        weight = len(trn_label) / np.array(list(counter.values())) / self.nclass

        self.weight = torch.from_numpy(weight).float().to(dev)

        self.ModalFusion = VLTransformer(input_data_dims, hyperpm).to(dev)

        # 图结构
        if self.marl_mode == 0:
            self.marl_out_dim = self.d_v * self.n_head + self.modal_num ** 2
        elif self.marl_mode == 1:
            self.marl_out_dim = self.modal_num ** 2
        elif self.marl_mode == 2:
            self.marl_out_dim = self.d_v * self.n_head
        self.GraphConstruct = GraphLearn(self.marl_out_dim, hyperpm=self.hyperpm).to(dev)
        if self.MP_mode == 'GCN':
            self.MessagePassing = GCN(self.marl_out_dim, self.marl_out_dim, self.nclass, self.dropout).to(dev)
        elif self.MP_mode == 'GAT':
            self.MessagePassing = GAT(self.marl_out_dim, self.marl_out_dim, self.nclass, self.dropout, self.alpha,
                                      nheads=2).to(dev)

        self.optimizer_MF = optim.Adam(self.ModalFusion.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.optimizer_GC = optim.Adam(self.GraphConstruct.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.optimizer_MP = optim.Adam(self.MessagePassing.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)

        # 对比图
        self.CL_MessagePassing = None
        self.optimizer_CL = None

        self.CL_MessagePassing = GCN(self.marl_out_dim, self.marl_out_dim, self.nclass, self.dropout).to(dev)
        self.optimizer_CL = optim.Adam(self.CL_MessagePassing.parameters(), lr=hyperpm.lr, weight_decay=hyperpm.reg)

        self.ModalFusion.apply(my_weight_init)

    def run_epoch(self, mode):
        dev = self.dev
        if mode == 'local-only':
            self.ModalFusion.train()
            self.optimizer_MF.zero_grad()

            prob, fusion_feat, mask_feat, attn = self.ModalFusion(self.feat)
            cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
            mse_loss = F.mse_loss(fusion_feat, mask_feat)
            loss = self.CL_mse * mse_loss + cls_loss
            loss.backward()

            self.optimizer_MF.step()
            print('trn-loss-MF: %.4f ' % cls_loss, end=' ')

        elif mode == 'global-only':
            self.ModalFusion.train()
            self.GraphConstruct.train()
            self.MessagePassing.train()

            self.optimizer_MF.zero_grad()
            self.optimizer_GC.zero_grad()
            self.optimizer_MP.zero_grad()

            # 对比学习
            self.CL_MessagePassing.train()
            self.optimizer_CL.zero_grad()

            _, fusion_feat, _, attn = self.ModalFusion(self.feat)
            adj = self.GraphConstruct(fusion_feat)

            graph_loss = GraphConstructLoss(fusion_feat, adj, self.hyperpm.theta_smooth, self.hyperpm.theta_degree,
                                            self.hyperpm.theta_sparsity)
            normalized_adj = normalize_adj(adj + torch.eye(adj.size(0)).to(dev))

            constrast_loss = None
            if self.CL_mode == "random_mask_edge":
                normalized_adj_mask = random_mask_adj(normalized_adj, self.CL_rate)
                _, hidden_1 = self.CL_MessagePassing(fusion_feat, normalized_adj)
                _, hidden_2 = self.CL_MessagePassing(fusion_feat, normalized_adj_mask)
                combine_hidden = torch.cat((hidden_1, hidden_2), dim=-1)
                constrast_loss = GCL_loss(combine_hidden)
            elif self.CL_mode == "random_mask_edge_2":
                normalized_adj_mask_1 = random_mask_adj(normalized_adj, self.CL_rate)
                normalized_adj_mask_2 = random_mask_adj(normalized_adj, self.CL_rate)
                _, hidden_1 = self.CL_MessagePassing(fusion_feat, normalized_adj_mask_1)
                _, hidden_2 = self.CL_MessagePassing(fusion_feat, normalized_adj_mask_2)
                combine_hidden = torch.cat((hidden_1, hidden_2), dim=-1)
                constrast_loss = GCL_loss(combine_hidden)

            prob, xx = self.MessagePassing(fusion_feat, normalized_adj)
            cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)

            loss = cls_loss
            if self.lg:
                loss += graph_loss
            loss += constrast_loss * self.CL_weight

            loss.backward()

            self.optimizer_GC.step()
            self.optimizer_MP.step()

            self.optimizer_CL.step()
            print('trn-loss-MF: %.4f trn-loss-GC: %.4f' % (cls_loss, graph_loss), end=' ')

        elif mode == 'local-global':
            self.ModalFusion.train()
            self.GraphConstruct.eval()
            self.MessagePassing.eval()

            self.optimizer_MF.zero_grad()
            self.optimizer_GC.zero_grad()
            self.optimizer_MP.zero_grad()

            self.CL_MessagePassing.eval()
            self.optimizer_CL.zero_grad()

            prob, fusion_feat, mask_feat, attn = self.ModalFusion(self.feat)
            cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
            mse_loss = F.mse_loss(fusion_feat, mask_feat)
            loss = self.CL_mse * mse_loss + cls_loss
            loss.backward()

            self.optimizer_MF.step()
            # trick, 虽然无法解释，但效果好
            if self.hyperpm.dataname == "ABIDE":
                self.optimizer_GC.step()
                self.optimizer_MP.step()
            print('trn-loss-MF: %.4f ' % cls_loss, end=' ')

            self.ModalFusion.eval()
            self.GraphConstruct.train()
            self.MessagePassing.train()

            self.optimizer_MF.zero_grad()
            self.optimizer_GC.zero_grad()
            self.optimizer_MP.zero_grad()

            # 对比学习
            self.CL_MessagePassing.train()
            self.optimizer_CL.zero_grad()

            _, embedding, _, attn = self.ModalFusion(self.feat)
            # 阻断反向转播
            fusion_feat = embedding.detach()

            adj = self.GraphConstruct(fusion_feat)

            graph_loss = GraphConstructLoss(fusion_feat, adj, self.hyperpm.theta_smooth, self.hyperpm.theta_degree,
                                            self.hyperpm.theta_sparsity)
            normalized_adj = normalize_adj(adj + torch.eye(adj.size(0)).to(dev))

            constrast_loss = None
            if self.CL_mode == "random_mask_edge":
                normalized_adj_mask = random_mask_adj(normalized_adj, self.CL_rate)
                _, hidden_1 = self.CL_MessagePassing(fusion_feat, normalized_adj)
                _, hidden_2 = self.CL_MessagePassing(fusion_feat, normalized_adj_mask)
                combine_hidden = torch.cat((hidden_1, hidden_2), dim=-1)
                constrast_loss = GCL_loss(combine_hidden)
            elif self.CL_mode == "random_mask_edge_2":
                normalized_adj_mask_1 = random_mask_adj(normalized_adj, self.CL_rate)
                normalized_adj_mask_2 = random_mask_adj(normalized_adj, self.CL_rate)
                _, hidden_1 = self.CL_MessagePassing(fusion_feat, normalized_adj_mask_1)
                _, hidden_2 = self.CL_MessagePassing(fusion_feat, normalized_adj_mask_2)
                combine_hidden = torch.cat((hidden_1, hidden_2), dim=-1)
                constrast_loss = GCL_loss(combine_hidden)
            elif self.CL_mode == "random_mask_feat":
                _, hidden_1 = self.CL_MessagePassing(fusion_feat, normalized_adj)
                fusion_feat = random_mask_feat(fusion_feat, self.CL_rate)
                _, hidden_2 = self.CL_MessagePassing(fusion_feat, normalized_adj)
                combine_hidden = torch.cat((hidden_1, hidden_2), dim=-1)
                constrast_loss = GCL_loss(combine_hidden)
            elif self.CL_mode == "random_mask_feat_2":
                fusion_feat_1 = random_mask_feat(fusion_feat, self.CL_rate)
                _, hidden_1 = self.CL_MessagePassing(fusion_feat_1, normalized_adj)
                fusion_feat_2 = random_mask_feat(fusion_feat, self.CL_rate)
                _, hidden_2 = self.CL_MessagePassing(fusion_feat_2, normalized_adj)
                combine_hidden = torch.cat((hidden_1, hidden_2), dim=-1)
                constrast_loss = GCL_loss(combine_hidden)
            elif self.CL_mode == "random_mask_edge_feat":
                fusion_feat_1 = random_mask_feat(fusion_feat, self.CL_rate)
                _, hidden_1 = self.CL_MessagePassing(fusion_feat_1, normalized_adj)
                normalized_adj_mask = random_mask_adj(normalized_adj, self.CL_rate)
                _, hidden_2 = self.CL_MessagePassing(fusion_feat, normalized_adj_mask)
                combine_hidden = torch.cat((hidden_1, hidden_2), dim=-1)
                constrast_loss = GCL_loss(combine_hidden)

            prob, xx = self.MessagePassing(fusion_feat, normalized_adj)
            cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)

            loss = cls_loss
            if self.lg:
                loss += graph_loss
            loss += constrast_loss * self.CL_weight

            loss.backward()

            self.optimizer_GC.step()
            self.optimizer_MP.step()

            self.optimizer_CL.step()
            print('trn-loss-MF: %.4f trn-loss-GC: %.4f' % (cls_loss, graph_loss), end=' ')
        elif mode == 'local-global-jointly':
            self.ModalFusion.train()
            self.GraphConstruct.train()
            self.MessagePassing.train()
            self.CL_MessagePassing.train()

            self.optimizer_MF.zero_grad()
            self.optimizer_GC.zero_grad()
            self.optimizer_MP.zero_grad()
            self.optimizer_CL.zero_grad()

            prob, fusion_feat, mask_feat, attn = self.ModalFusion(self.feat)
            cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)
            print('trn-loss-MF: %.4f ' % cls_loss, end=' ')
            mse_loss = F.mse_loss(fusion_feat, mask_feat)
            loss = self.CL_mse * mse_loss + cls_loss

            _, embedding, _, attn = self.ModalFusion(self.feat)
            # 阻断反向转播
            fusion_feat = embedding.detach()

            adj = self.GraphConstruct(fusion_feat)

            graph_loss = GraphConstructLoss(fusion_feat, adj, self.hyperpm.theta_smooth, self.hyperpm.theta_degree,
                                            self.hyperpm.theta_sparsity)
            normalized_adj = normalize_adj(adj + torch.eye(adj.size(0)).to(dev))

            constrast_loss = None
            if self.CL_mode == "random_mask_edge":
                normalized_adj_mask = random_mask_adj(normalized_adj, self.CL_rate)
                _, hidden_1 = self.CL_MessagePassing(fusion_feat, normalized_adj)
                _, hidden_2 = self.CL_MessagePassing(fusion_feat, normalized_adj_mask)
                combine_hidden = torch.cat((hidden_1, hidden_2), dim=-1)
                constrast_loss = GCL_loss(combine_hidden)
            elif self.CL_mode == "random_mask_edge_2":
                normalized_adj_mask_1 = random_mask_adj(normalized_adj, self.CL_rate)
                normalized_adj_mask_2 = random_mask_adj(normalized_adj, self.CL_rate)
                _, hidden_1 = self.CL_MessagePassing(fusion_feat, normalized_adj_mask_1)
                _, hidden_2 = self.CL_MessagePassing(fusion_feat, normalized_adj_mask_2)
                combine_hidden = torch.cat((hidden_1, hidden_2), dim=-1)
                constrast_loss = GCL_loss(combine_hidden)
            elif self.CL_mode == "random_mask_feat":
                _, hidden_1 = self.CL_MessagePassing(fusion_feat, normalized_adj)
                fusion_feat = random_mask_feat(fusion_feat, self.CL_rate)
                _, hidden_2 = self.CL_MessagePassing(fusion_feat, normalized_adj)
                combine_hidden = torch.cat((hidden_1, hidden_2), dim=-1)
                constrast_loss = GCL_loss(combine_hidden)
            elif self.CL_mode == "random_mask_feat_2":
                fusion_feat_1 = random_mask_feat(fusion_feat, self.CL_rate)
                _, hidden_1 = self.CL_MessagePassing(fusion_feat_1, normalized_adj)
                fusion_feat_2 = random_mask_feat(fusion_feat, self.CL_rate)
                _, hidden_2 = self.CL_MessagePassing(fusion_feat_2, normalized_adj)
                combine_hidden = torch.cat((hidden_1, hidden_2), dim=-1)
                constrast_loss = GCL_loss(combine_hidden)
            elif self.CL_mode == "random_mask_edge_feat":
                fusion_feat_1 = random_mask_feat(fusion_feat, self.CL_rate)
                _, hidden_1 = self.CL_MessagePassing(fusion_feat_1, normalized_adj)
                normalized_adj_mask = random_mask_adj(normalized_adj, self.CL_rate)
                _, hidden_2 = self.CL_MessagePassing(fusion_feat, normalized_adj_mask)
                combine_hidden = torch.cat((hidden_1, hidden_2), dim=-1)
                constrast_loss = GCL_loss(combine_hidden)

            prob, xx = self.MessagePassing(fusion_feat, normalized_adj)
            cls_loss = ClsLoss(prob, self.targ, self.trn_idx, self.weight)

            loss += cls_loss
            if self.lg:
                loss += graph_loss
            loss += constrast_loss * self.CL_weight

            loss.backward()

            self.optimizer_MF.step()
            self.optimizer_GC.step()
            self.optimizer_MP.step()
            self.optimizer_CL.step()
            print('trn-loss-MF: %.4f trn-loss-GC: %.4f' % (cls_loss, graph_loss), end=' ')

    def print_trn_acc(self, mode='local-only'):
        print('trn-', end='')
        trn_acc, trn_auc, targ_trn, pred_trn = self._print_acc(self.trn_idx, mode, end=' val-')
        val_acc, val_auc, targ_val, pred_val = self._print_acc(self.val_idx, mode)
        # print('pred:',pred_val[:10], 'targ:',targ_val[:10])
        return trn_acc, val_acc

    def print_tst_acc(self, mode='local-only'):
        print('tst-', end='')
        tst_acc, tst_auc, targ_tst, pred_tst = self._print_acc(self.tst_idx, mode, tst=True)
        conf_mat = confusion_matrix(targ_tst.detach().cpu().numpy(), pred_tst.detach().cpu().numpy())
        return tst_acc, tst_auc, conf_mat

    def acc_compute(self, prob, eval_idx):
        prob = prob[eval_idx]
        targ = self.targ[eval_idx]
        pred = prob.max(1)[1].type_as(targ)
        acc = pred.eq(targ).double().sum() / len(targ)
        acc = acc.item()
        auc = roc_auc_score(one_hot(targ, self.nclass).cpu().numpy(), one_hot(pred, self.nclass).cpu().numpy())
        return acc, auc, targ, pred

    def _print_acc(self, eval_idx, mode, tst=False, end='\n'):
        self.ModalFusion.eval()
        self.GraphConstruct.eval()
        self.MessagePassing.eval()
        # 对比学习
        self.CL_MessagePassing.eval()

        _adj = None
        fusion_feat = None
        MP_feat = None
        if mode == 'local-only':
            prob, _, _, attn = self.ModalFusion(self.feat)
        else:
            prob_MF, fusion_feat, _, attn = self.ModalFusion(self.feat)
            adj = self.GraphConstruct(fusion_feat)
            _adj = adj.clone().detach()
            adj = normalize_adj(adj + torch.eye(adj.size(0)).to(self.dev))
            prob, MP_feat = self.MessagePassing(fusion_feat, adj)
        prob = prob[eval_idx]
        targ = self.targ[eval_idx]
        pred = prob.max(1)[1].type_as(targ)
        acc = pred.eq(targ).double().sum() / len(targ)
        acc = acc.item()
        auc = roc_auc_score(one_hot(targ, self.nclass).cpu().numpy(), one_hot(pred, self.nclass).cpu().numpy())
        print('auc: %.4f  acc: %.4f' % (auc, acc), end=end)
        if tst is True and mode != 'local-only':
            print('attention maps have been saved.')
            np.save('./attn/attn_map_{}.npy'.format(self.hyperpm.dataname), attn)
            np.savez('./graph/{}_{}_graph'.format(self.hyperpm.dataname, self.GC_mode),
                     adj=_adj.detach().cpu().numpy(),
                     feat=self.feat.detach().cpu().numpy(),
                     fused=fusion_feat.detach().cpu().numpy(),
                     embedding=MP_feat.detach().cpu().numpy(),
                     label=self.targ.detach().cpu().numpy())
        return acc, auc, targ, pred
