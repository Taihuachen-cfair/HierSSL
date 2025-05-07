import os
import gc
import sys
import time
import argparse
import tempfile
import torch.autograd

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from model import *

from utils import *


class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_and_eval(datadir, dataname, hyperpm):
    set_rng_seed(hyperpm.seed)
    if dataname == 'TADPOLE':
        hyperpm.nclass = 3
        hyperpm.nmodal = 6
    elif dataname == 'ABIDE':
        hyperpm.nclass = 2
        hyperpm.nmodal = 4
    elif hyperpm.nclass == 0 and hyperpm.nmodal == 0:
        print("For other datasets, you must provide nclass and nmodal!")
        print("Use --help for more information")
        return
    path = datadir + dataname + '/'
    modal_feat_dict = np.load(path + 'modal_feat_dict.npy', allow_pickle=True).item()
    data = pd.read_csv(path + 'processed_standard_data.csv').values
    print('data shape: ', data.shape)

    input_data_dims = []
    for i in modal_feat_dict.keys():
        input_data_dims.append(len(modal_feat_dict[i]))
    print('Modal dims ', input_data_dims)
    input_data = data[:, :-1]
    label = data[:, -1] - 1
    skf = StratifiedKFold(n_splits=10, random_state=hyperpm.seed, shuffle=True)
    val_acc, tst_acc, tst_auc, tst_sen, tst_spe = [], [], [], [], []
    clk = 0
    visualize_tsne(input_data, label, "Init")
    model = None
    for train_index, test_index in skf.split(input_data, label):
        clk += 1
        agent = EvalHelper(input_data_dims, input_data, label, hyperpm, train_index, test_index)
        tm = time.time()
        best_val_acc, wait_cnt = 0.0, 0
        model_sav = tempfile.TemporaryFile()
        for t in range(hyperpm.nepoch):
            print('%3d/%d' % (t, hyperpm.nepoch), end=' ')
            agent.run_epoch(mode=hyperpm.mode)
            _, cur_val_acc = agent.print_trn_acc(hyperpm.mode)
            if cur_val_acc > best_val_acc:
                wait_cnt = 0
                best_val_acc = cur_val_acc
                model_sav.close()
                model_sav = tempfile.TemporaryFile()
                dict_list = [agent.ModalFusion.state_dict(),
                             agent.GraphConstruct.state_dict(),
                             agent.MessagePassing.state_dict()]
                torch.save(dict_list, model_sav)
            else:
                wait_cnt += 1
                if wait_cnt > hyperpm.early:
                    break
        print("time: %.4f sec." % (time.time() - tm))
        model_sav.seek(0)
        dict_list = torch.load(model_sav)
        agent.ModalFusion.load_state_dict(dict_list[0])
        agent.GraphConstruct.load_state_dict(dict_list[1])
        agent.MessagePassing.load_state_dict(dict_list[2])
        model = agent
        val_acc.append(best_val_acc)
        cur_tst_acc, cur_tst_auc, conf_mat = agent.print_tst_acc(hyperpm.mode)
        tst_acc.append(cur_tst_acc)
        tst_auc.append(cur_tst_auc)
        if hyperpm.nclass == 2:
            tn, fp, fn, tp = conf_mat.ravel()
            spe = tn / (tn + fp)
            sen = tp / (tp + fn)
            tst_spe.append(spe)
            tst_sen.append(sen)

        if np.array(tst_acc).mean() < 0.6 and clk == 5:
            break
    use_cuda = torch.cuda.is_available()
    dev = torch.device('cuda' if use_cuda else 'cpu')
    if dataname == 'TADPOLE':
        label_map = {0: 'NC', 1: 'MCI', 2: 'ND'}
    elif dataname == 'ABIDE':
        label_map = {0: 'NC', 1: 'ASD'}
    else:
        label_map = None

    feat = torch.from_numpy(input_data).float().to(dev)
    attn_feat = torch.tensor([])
    _, fusion_feat, _, _ = model.ModalFusion(feat, feat_attn=attn_feat)
    attn_feat_vis = attn_feat.cpu().detach().numpy()
    visualize_tsne(attn_feat_vis, label, "Hattn",label_map=label_map)

    attn_feat = attn_feat.reshape(feat.size(0), hyperpm.nmodal, hyperpm.n_head * hyperpm.n_hidden)
    if hyperpm.dataname == "TADPOLE":
        for i in range(hyperpm.nmodal):
            attn_i = attn_feat[:, i, :]
            attn_i_vis = attn_i.cpu().detach().numpy()
            visualize_tsne(attn_i_vis, label, f"modalAttn/Hattn_TAD_{i}",label_map=label_map)
    else:
        for i in range(hyperpm.nmodal):
            attn_i = attn_feat[:, i, :]
            attn_i_vis = attn_i.cpu().detach().numpy()
            visualize_tsne(attn_i_vis, label, f"modalAttn/Hattn_ABI_{i}",label_map=label_map)

    fusion_feat_vis = fusion_feat.cpu().detach().numpy()
    visualize_tsne(fusion_feat_vis, label, "Hsh",label_map=label_map)
    gllin_feat = torch.tensor([])
    glattn_feat = torch.tensor([])
    adj = model.GraphConstruct(fusion_feat, feat_gllin=gllin_feat, feat_glattn=glattn_feat)
    gllin_feat_vis = gllin_feat.cpu().detach().numpy()
    glattn_feat_vis = glattn_feat.cpu().detach().numpy()
    visualize_tsne(gllin_feat_vis, label, "Hgllin",label_map=label_map)
    visualize_tsne(glattn_feat_vis, label, "Hglattn",label_map=label_map)
    normalized_adj = normalize_adj(adj + torch.eye(adj.size(0)).to(dev))
    _, Hg = model.MessagePassing(fusion_feat, normalized_adj)
    Hg_vis = Hg.cpu().detach().numpy()
    visualize_tsne(Hg_vis, label, "Hg",label_map=label_map)

    return (np.array(val_acc).mean(),
            np.array(tst_acc).mean(), np.array(tst_acc).std(),
            np.array(tst_auc).mean(), np.array(tst_auc).std(),
            np.array(tst_sen).mean(), np.array(tst_sen).std(),
            np.array(tst_spe).mean(), np.array(tst_spe).std())


def main(args_str=None):
    assert float(torch.__version__[:3]) + 1e-3 >= 0.4
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data/')
    parser.add_argument('--dataname', type=str)
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Insist on using CPU instead of CUDA.')
    parser.add_argument('--nepoch', type=int, default=1000,
                        help='Max number of epochs to train.')
    parser.add_argument('--early', type=int, default=50,
                        help='Extra iterations before early-stopping.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--reg', type=float, default=0.0036,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.65,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of VLTransformer Encoder layers.')
    parser.add_argument('--n_hidden', type=int, default=16,
                        help='Number of hidden units per modal.')
    parser.add_argument('--n_head', type=int, default=8,
                        help='Number of attention head.')
    parser.add_argument('--th', type=float, default=0.9,
                        help='threshold of weighted cosine')
    parser.add_argument('--MP_mode', type=str, default='GAT',
                        help='Massage Passing mode')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='alpha for LeakyRelu in GAT')
    parser.add_argument('--theta_smooth', type=float, default=1.0,
                        help='graph_loss_smooth')
    parser.add_argument('--theta_degree', type=float, default=0.5,
                        help='graph_loss_degree')
    parser.add_argument('--theta_sparsity', type=float, default=0.0,
                        help='graph_loss_degree')
    parser.add_argument('--mode', type=str, default='local-only',
                        help='training mode')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed setting')
    parser.add_argument('--nmodal', type=int, default=0,
                        help='modal number')
    parser.add_argument('--nclass', type=int, default=0,
                        help='class number')
    parser.add_argument('--marl_mode', type=int, default=2,
                        help='marl_mode: 0 is Hsh concat Hsp, 1 is Hsp only, 2 is Hsh only')
    parser.add_argument('--GC_mode', type=str, default='weighted-cosine',
                        help='graph constrcution mode: \n '
                             'weighted-consine is the A-graph implementation of the original HierSSL')
    parser.add_argument('--lg', action='store_true', default=False,
                        help='Whether to use Loss g, default is false')
    # 实验超参数
    parser.add_argument('--CL_mode', type=str, default='random_mask_edge',
                        help='Contrast learning mode: \n'
                             'random_mask_edge : random mask some edges from A \n '
                        )
    parser.add_argument('--CL_mse', type=float, default=1.0,
                        help='The weights of the contrast learning loss function in marl, this value should be '
                             'between 0 and 1')
    parser.add_argument('--CL_weight', type=float, default=0.2,
                        help='The weights of the contrast learning loss function, this value should be between 0 and 1')
    parser.add_argument('--CL_rate', type=float, default=0.2,
                        help='The rate of mask in contrast learning, this value should be between 0 and 1')
    parser.add_argument('--gl_attn_layer', type=int, default=2,
                        help='N/A')
    parser.add_argument('--gl_lin_layer', type=int, default=2,
                        help='N/A')

    if args_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_str.split())
    with ((RedirectStdStreams(stdout=sys.stderr))):
        print('GC_mode:', args.GC_mode, 'MP_mode:', args.MP_mode)
        val_acc, tst_acc, tst_acc_std, tst_auc, tst_auc_std, tst_sen, tst_sen_std, tst_spe, tst_spe_std = \
            train_and_eval(os.path.expanduser(args.datadir), args.dataname, args)
        print('val=%.2f%% tst_acc=%.2f%% tst_auc=%.2f%%' % (val_acc * 100, tst_acc * 100, tst_auc * 100))
        print('tst_sen=%.2f%% tst_spe=%.2f%%' % (tst_sen * 100, tst_spe * 100))
        print('tst_acc_std=%.4f tst_auc_std=%.4f' % (tst_acc_std, tst_auc_std))
    return val_acc, tst_acc


if __name__ == '__main__':
    print(str(main()))
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
