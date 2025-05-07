import numpy as np
import matplotlib.pyplot as plt

import torch.autograd
import torch.nn as nn
import torch.nn.functional as F

from sklearn.manifold import TSNE


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]


def GraphConstructLoss(feat, adj, theta_smooth, theta_degree, theta_sparsity):
    # Graph regularization
    use_cuda = torch.cuda.is_available()
    dev = torch.device('cuda' if use_cuda else 'cpu')
    L = torch.diagflat(torch.sum(adj, -1)) - adj
    vec_one = torch.ones(adj.size(-1)).to(dev)

    smoothess_penalty = torch.trace(torch.mm(feat.T, torch.mm(L, feat))) / int(np.prod(adj.shape))
    degree_penalty = torch.mm(vec_one.unsqueeze(0), torch.log(torch.mm(adj, vec_one.unsqueeze(-1)) + 1e-5)).squeeze() / \
                     adj.shape[-1]
    sparsity_penalty = torch.sum(torch.pow(adj, 2)) / int(np.prod(adj.shape))

    return theta_smooth * smoothess_penalty - theta_degree * degree_penalty + theta_sparsity * sparsity_penalty


def GCL_loss(hidden, hidden_norm=True, temperature=1.0):
    batch_size = hidden.shape[0] // 2
    LARGE_NUM = 1e9
    # inner dot or cosine
    if hidden_norm:
        hidden = F.normalize(hidden, p=2, dim=-1)
    hidden_list = torch.split(hidden, batch_size, dim=0)
    hidden1, hidden2 = hidden_list[0], hidden_list[1]

    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = torch.from_numpy(np.arange(batch_size)).to(hidden.device)
    masks = F.one_hot(torch.from_numpy(np.arange(batch_size)).to(hidden.device), batch_size)

    logits_aa = torch.matmul(hidden1, hidden1_large.transpose(1, 0)) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, hidden2_large.transpose(1, 0)) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, hidden2_large.transpose(1, 0)) / temperature
    logits_ba = torch.matmul(hidden2, hidden1_large.transpose(1, 0)) / temperature

    loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], 1), labels)
    loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], 1), labels)
    loss = (loss_a + loss_b)
    return loss


def ClsLoss(output, labels, idx, weight):
    return F.nll_loss(output[idx], labels[idx], weight)


def ClsLoss_noweight(output, labels, idx):
    return F.nll_loss(output[idx], labels[idx])


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    D = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(D, -0.5)
    d_inv_sqrt = torch.diagflat(d_inv_sqrt)
    adj = torch.mm(d_inv_sqrt, torch.mm(adj, d_inv_sqrt))
    return adj


def random_mask_adj(adj, mask_ratio=0.1):
    """
    参数：
    adj (tensor): 输入的邻接矩阵

    返回：
    res_adj (tensor): 随机掩码的邻接矩阵
    """
    # 复制输入矩阵，以避免修改原始矩阵
    res_adj = adj.clone()

    # 获取存在的边的索引（只取上三角部分，因为是无向图）
    edge_indices = torch.triu_indices(res_adj.size(0), res_adj.size(1), offset=1)
    existing_edges_mask = res_adj[edge_indices[0], edge_indices[1]] != 0

    # 获取实际存在的边的索引
    present_edges_indices = edge_indices[:, existing_edges_mask]

    # 计算总边数
    num_edges = present_edges_indices.size(1)

    # 计算需要掩盖的边数
    num_to_mask = int(num_edges * mask_ratio)

    # 如果没有边或者需要掩码的边数为0，则直接返回原始矩阵
    if num_edges == 0 or num_to_mask == 0:
        return res_adj

    # 随机选择需要掩盖的边
    perm = torch.randperm(num_edges)[:num_to_mask]
    edges_to_mask = present_edges_indices[:, perm]

    # 将选择的边设为0
    res_adj[edges_to_mask[0], edges_to_mask[1]] = 0
    res_adj[edges_to_mask[1], edges_to_mask[0]] = 0
    return res_adj


def random_mask_feat(feat, mask_ratio=0.1):
    """
    参数：
    feat (tensor): 输入的特征张量

    返回：
    res_feat (tensor): 随机掩码的特征张量
    """
    # 复制输入矩阵，以避免修改原始矩阵
    res_feat = feat.clone()

    # 获取张量的形状
    B, D = res_feat.shape

    # 计算需要遮蔽的元素总数
    mask_num = int(D * mask_ratio)

    # 对每个样本进行遮蔽
    for i in range(B):
        # 随机选择mask_num个特征索引
        mask_indices = torch.randperm(D)[:mask_num]
        # 将这些索引位置的特征置零
        res_feat[i, mask_indices] = 0

    return res_feat


def my_weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


# 这被设计为一个可视化的探针,在需要的地方调用它来获取可视化视图
def visualize_tsne(features, labels, filename, label_map=None):
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("jet", len(unique_labels))  # 选择颜色映射

    if label_map is None:
        label_map = {label: str(label) for label in unique_labels}  # 默认标签为原值

    for i, label in enumerate(unique_labels):
        # 获取标签的名称，替换原始标签为新标签
        label_name = label_map.get(label, str(label))

        plt.scatter(tsne_results[labels == label, 0],
                    tsne_results[labels == label, 1],
                    label=label_name,
                    alpha=0.7,
                    color=colors(i))

    # 设置图例字体大小
    plt.legend(fontsize=16)

    # 设置X轴和Y轴的字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # 设置边框粗细
    ax = plt.gca()  # 获取当前坐标轴
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # 设置边框粗细为2

    # 保存图片
    # plt.savefig(filename, dpi=600)
    plt.savefig(filename+'.pdf', format = 'pdf')
    plt.close()  # 关闭当前图形，以释放内存
