
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import svds


def entropy(net_output):
    p = F.softmax(net_output, dim=1)
    logp = F.log_softmax(net_output, dim=1)
    plogp = p * logp
    entropy = - torch.sum(plogp, dim=1)
    return entropy


def compute_ETF(W, device):  # W [K, 512]
    K = W.shape[0]
    # W = W - torch.mean(W, dim=0, keepdim=True)
    WWT = torch.mm(W, W.T)            # [K, 512] [512, K] -> [K, K]
    WWT /= torch.norm(WWT, p='fro')   # [K, K]

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device) / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_W_H_relation(W, H, device):  # W:[K, 512] H:[512, K]
    """ H is already normalized"""
    K = W.shape[0]

    # W = W - torch.mean(W, dim=0, keepdim=True)
    WH = torch.mm(W, H.to(device))   # [K, 512] [512, K]
    WH /= torch.norm(WH, p='fro')
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K))).to(device)

    res = torch.norm(WH - sub, p='fro')
    return res.detach().cpu().numpy().item()


def analysis(model, loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = [0 for _ in range(args.C)]  # within class sample size
    mean = [0 for _ in range(args.C)]
    Sw_cls = [0 for _ in range(args.C)]

    # get the logit, label, feats
    model.eval()
    logits_list = []
    labels_list = []
    feats_list = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits, feats = model(data, ret_feat=True)
            logits_list.append(logits)
            labels_list.append(target)
            feats_list.append(feats)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
        feats = torch.cat(feats_list).to(device)

    loss = torch.nn.CrossEntropyLoss(reduction='mean')(logits, labels).item()
    acc = (logits.argmax(dim=-1) == labels).sum().item() / len(labels)

    # ====== compute mean and var for each class
    for c in range(args.C):
        idxs = (labels == c).nonzero(as_tuple=True)[0]
        h_c = feats[idxs, :]  # [B, 512]

        N[c] = h_c.shape[0]
        mean[c] = torch.sum(h_c, dim=0) / h_c.shape[0]  # [512]

        # update within-class cov
        z = h_c - mean[c].unsqueeze(0)  # [B, 512]
        cov_c = torch.matmul(z.unsqueeze(-1), z.unsqueeze(1))  # [B 512 1] [B 1 512] -> [B, 512, 512]
        Sw_cls[c] = torch.sum(cov_c, dim=0)  # [512, 512]

    # global mean
    M = torch.stack(mean).T   # [512, C]
    muG = torch.mean(M, dim=1, keepdim=True)  # [512, C]
    Sw = sum(Sw_cls) / sum(N)

    # between-class covariance
    M_ = M - muG  # [512, C]
    Sb = torch.matmul(M_, M_.T) / args.C

    # ============ NC1: tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=args.C - 1)
    inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
    nc1 = np.trace(Sw @ inv_Sb)

    # ========== NC2.1 and NC2.2
    W = model.classifier.weight.detach().T  # [512, C]
    M_norms = torch.norm(M_, dim=0)  # [C]
    W_norms = torch.norm(W , dim=0)  # [C]

    norm_M_CoV = (torch.std(M_norms) / torch.mean(M_norms)).item()
    norm_W_CoV = (torch.std(W_norms) / torch.mean(W_norms)).item()

    # mutual coherence
    def coherence(V):
        G = V.T @ V  # [C, D] [D, C]
        G += torch.ones((args.C, args.C), device=device) / (args.C - 1)
        G -= torch.diag(torch.diag(G))  # [C, C]
        return torch.norm(G, 1).item() / (args.C * (args.C - 1))

    cos_M = coherence(M_ / M_norms)  # [D, C]
    cos_W = coherence(W  / W_norms)

    # =========== NC2
    nc2_h = compute_ETF(M_.T, device)
    nc2_w = compute_ETF(W.T, device)

    # =========== NC2/NC3 (all losses are equal paper)
    nc2 = compute_W_H_relation(W.T, M_, device)

    # =========== NC3  ||W^T - M_||
    normalized_M = M_ / torch.norm(M_, 'fro')
    normalized_W = W  / torch.norm(W, 'fro')
    W_M_dist = (torch.norm(normalized_W - normalized_M) ** 2).item()

    return {
        'loss': loss,
        'acc': acc,
        'nc1': nc1,
        'nc2_norm_h': norm_M_CoV,
        'nc2_norm_w': norm_W_CoV,
        'nc2_cos_h': cos_M,
        'nc2_cos_w': cos_W,
        'nc2_h': nc2_h,
        'nc2_w': nc2_w,
        'nc2': nc2,
        'nc3': W_M_dist,
        "w_mnorm": np.mean(W_norms.cpu().numpy()),
        "h_mnorm": np.mean(M_norms.cpu().numpy()),
    }


def analysis_nc1(logits, targets, feats, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N, N_inc, N_cor = [0] * num_classes, [0] * num_classes, [0] * num_classes
    mean = [0] * num_classes
    Sw_all, Sw_cor, Sw_inc = 0, 0, 0

    loss, n_correct, n_match = 0, 0, 0
    logits, targets, feats = logits.to(device), targets.to(device), feats.to(device)
    preds = logits.argmax(dim=-1)

    criterion = nn.CrossEntropyLoss().cuda()
    idxs_cor = (targets == preds).nonzero(as_tuple=True)[0]
    idxs_inc = (targets != preds).nonzero(as_tuple=True)[0]
    acc = (logits.argmax(dim=-1) == targets).sum().item() / len(targets)

    # === loss
    loss = criterion(logits, targets).item()
    loss_cor = criterion(logits[idxs_cor], targets[idxs_cor]).item()
    loss_inc = criterion(logits[idxs_inc], targets[idxs_inc]).item()

    # === entropy
    ent = torch.mean(entropy(logits))
    ent_cor = torch.mean(entropy(logits[idxs_cor]))
    ent_inc = torch.mean(entropy(logits[idxs_inc]))

    for c in range(num_classes):
        idxs = (preds == c).nonzero(as_tuple=True)[0]
        idxs_cor = ((preds == c) * (targets == preds)).nonzero(as_tuple=True)[0]
        idxs_inc = ((preds == c) * (targets != preds)).nonzero(as_tuple=True)[0]

        # update class means
        h_c = feats[idxs, :]  # [B, 512]
        N[c] = h_c.shape[0]
        mean[c] = torch.sum(h_c, dim=0) / h_c.shape[0]  # [512]

        # update within-class cov
        z = feats[idxs, :] - mean[c].unsqueeze(0)  # [B, 512]
        cov = torch.matmul(z.unsqueeze(-1), z.unsqueeze(1))   # [B 512 1] [B 1 512] -> [B, 512, 512]
        Sw_all += torch.sum(cov, dim=0)  # [512, 512]

        z = feats[idxs_cor, :] - mean[c].unsqueeze(0)  # [B, 512]
        cov = torch.matmul(z.unsqueeze(-1), z.unsqueeze(1))  # [B 512 1] [B 1 512] -> [B, 512, 512]
        Sw_cor += torch.sum(cov, dim=0)  # [512, 512]
        N_cor[c] += z.shape[0]

        z = feats[idxs_inc, :] - mean[c].unsqueeze(0)  # [B, 512]
        cov = torch.matmul(z.unsqueeze(-1), z.unsqueeze(1))  # [B 512 1] [B 1 512] -> [B, 512, 512]
        Sw_inc += torch.sum(cov, dim=0)  # [512, 512]
        N_inc[c] += z.shape[0]

        M = torch.stack(mean).T
        muG = torch.mean(M, dim=1, keepdim=True)  # [512, C]

        # between-class covariance
        M_ = M - muG  # [512, C]
        Sb = torch.matmul(M_, M_.T) / num_classes
        Sb = Sb.cpu().numpy()

        # within-class varaince
        Sw_all /= sum(N)
        Sw_inc /= sum(N_inc)
        Sw_cor /= sum(N_cor)
        Sw_all = Sw_all.cpu().numpy()
        Sw_inc = Sw_inc.cpu().numpy()
        Sw_cor = Sw_cor.cpu().numpy()

        # compute NC1
        eigvec, eigval, _ = svds(Sb, k=num_classes - 1)
        inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
        nc1_all = np.trace(Sw_all @ inv_Sb)
        nc1_cor = np.trace(Sw_cor @ inv_Sb)
        nc1_inc = np.trace(Sw_inc @ inv_Sb)

    return {
        'acc': acc,

        'loss': loss,
        'loss_inc': loss_inc,
        'loss_cor': loss_cor,

        'nc1': nc1_all,
        'nc1_inc': nc1_inc,
        'nc1_cor': nc1_cor,

        'ent': ent,
        'ent_cor': ent_cor,
        'ent_inc': ent_inc
    }
