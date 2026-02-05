from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import  random
import networkx as nx
from itertools import permutations
import matplotlib.pyplot as plt

def to_np(x):
    """Convert numpy, list, or tensor (CPU/GPU) to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    try:
        return x.detach().cpu().numpy()
    except:
        return np.array(x)

def plot_map(X, X_pgd, eta, mask):
    """
    Plots clean input and PGD-perturbed input on top subplot,
    and perturbation η on bottom subplot.
    """

    # Slice 512:1024
    X      = to_np(X)
    X_pgd  = to_np(X_pgd)
    eta    = to_np(eta)
    mask   = to_np(mask)


    # print("shape of X", X.shape,X_pgd)

    T = len(X)
    t = np.arange(T)

    # # Create figure with 2 rows
    fig, ax = plt.subplots(2, 1, figsize=(6, 3), sharex=True)

    # # ---- (1) Clean vs Perturbed Input ----
    ax[0].plot(t, X, linewidth=2, label="Clean Input (X)")
    ax[0].plot(t, X_pgd, linestyle="--", linewidth=2, label="Perturbed Input (X_pgd)")
    ax[0].set_ylabel("Value")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    # # ---- (2) Perturbation (η) ----
    ax[1].plot(t, eta, linestyle=":", linewidth=2, label="Perturbation (η)")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("η")
    ax[1].grid(alpha=0.3)
    ax[1].legend()
    batch_idx = 4
    # # Extract batch and remove channel dim → shape (T, F)
    #data = X[batch_idx, :, 0, :]  # (325, 12)

    #print("shape of mask", mask.shape)
    m1 = mask[batch_idx, :, 0, :]  # (325, 12)

    #num_ones = m1.sum().item()
    #num_zeros = m1.numel() - num_ones

    #print(num_ones)
    
    # map values

    plt.figure(figsize=(10, 4))
    plt.figure(figsize=(2.5,3))
    plt.pcolormesh(m1, cmap="viridis")
    plt.colorbar(label="Mask Values")
    plt.xlabel("Timestamp",fontsize = 12)
    plt.ylabel("Nodes",fontsize = 12)
    num_t = m1.shape[1]
    plt.xticks(np.arange(0, num_t, 3),fontsize = 12)
    plt.tight_layout()

    # Save figure
    #plt.savefig("plots/stta_pems.pdf", bbox_inches='tight')

    plt.show()

def attack_set_by_degree(adj, attack_nodes):
    adj = np.asarray(adj)
    G = nx.from_numpy_array(adj)
    D = G.degree()
    Degree = np.zeros(adj.shape[0])
    for i in range(adj.shape[0]):
        Degree[i] = D[i]
    # print(Degree)
    Dsort = Degree.argsort()[::-1]
    l = Dsort
    chosen_nodes = [l[i] for i in range(attack_nodes)]
    return chosen_nodes

def attack_set_by_pagerank(adj, attack_nodes):
    adj = np.asarray(adj)
    G = nx.from_numpy_array(adj)
    result = nx.pagerank(G)
    d_order = sorted(result.items(), key=lambda x: x[1], reverse=True)
    l = [x[0] for x in d_order]  # The sequence produced by pagerank algorithm
    chosen_nodes = [l[i] for i in range(attack_nodes)]
    return chosen_nodes

def attack_set_by_pagerank_budget(adj, B):
    """
    adj : adjacency matrix (numpy array or torch tensor)
    B   : total budget (sum of node degrees allowed)

    Returns:
        chosen_nodes : list of selected node indices within budget B
    """

    # Convert to numpy
    adj = np.asarray(adj)

    # Build graph
    G = nx.from_numpy_array(adj)

    # ---- Compute node-degree cost vector b ----
    # b[i] = degree of node i (in numeric form)
    b = np.array([deg for _, deg in G.degree()])   # shape: [N]

    # ---- Compute PageRank ----
    pr_scores = nx.pagerank(G)                    # dict: node -> score

    # ---- Adjust PR score by cost = PR / degree ----
    cost_adjusted_scores = {i: pr_scores[i] / b[i] if b[i] > 0 else 0.0
                            for i in range(len(b))}

    # ---- Sort nodes by descending score ----
    ranked_nodes = sorted(cost_adjusted_scores, key=cost_adjusted_scores.get, reverse=True)

    # ---- Budget-aware greedy selection ----
    chosen_nodes = []
    total_cost = 0.0

    for node in ranked_nodes:
        node_cost = b[node]

        if total_cost + node_cost > B:
            continue   # skip if over budget

        chosen_nodes.append(node)
        total_cost += node_cost

        if total_cost >= B:
            break

    return chosen_nodes



def attack_set_by_betweenness(adj, attack_nodes):
    adj = np.asarray(adj)
    G = nx.from_numpy_array(adj)
    result = nx.betweenness_centrality(G)
    # print(result)
    d_order = sorted(result.items(), key=lambda x: x[1], reverse=True)
    l = [x[0] for x in d_order]
    chosen_nodes = [l[i] for i in range(attack_nodes)]
    return chosen_nodes

def batch_saliency_map(input_grads):

    input_grads = input_grads.mean(dim=0)
    node_saliency_map = []
    #print("input grads shape", input_grads.shape)
    input_grads = input_grads.permute(1,0,2) # for adv test only
    for n in range(input_grads.shape[0]): # nth node
        #node_grads = input_grads[:,n]
        node_grads = input_grads[n,:]
        node_saliency = torch.norm(F.relu(node_grads)).item()
        node_saliency_map.append(node_saliency)
    #print("shape of grads node", input_grads.shape )
    sorted_id = sorted(range(len(node_saliency_map)), key=lambda k: node_saliency_map[k], reverse=False)
    return node_saliency_map, sorted_id

def attack_set_by_saliency_map(input_grads, attack_nodes):
    node_saliency_map, sorted_id = batch_saliency_map(input_grads)
    #print("in other attacks")
    #print(sorted_id, attack_nodes)
    chosen_nodes = [sorted_id[i] for i in range(attack_nodes)]
    
    return  chosen_nodes

def attack_set_by_betweenness_budget(adj, B):
    """
    adj : adjacency matrix (numpy array or torch tensor)
    B   : total budget (sum of node degrees allowed)

    Returns:
        chosen_nodes : list of selected node indices within budget B
    """

    # Convert to numpy
    adj = np.asarray(adj)

    # Build graph
    G = nx.from_numpy_array(adj)

    # ---- Compute node-degree cost vector b ----
    # b[i] = degree of node i (in numeric form)
    b = np.array([deg for _, deg in G.degree()])   # shape: [N]

    # ---- Compute Betweenness Centrality ----
    # Returns a dict: node -> centrality score
    bc_scores = nx.betweenness_centrality(G, normalized=True)  

    # ---- Adjust centrality score by cost = BC / degree ----
    cost_adjusted_scores = {
        i: (bc_scores[i] / b[i]) if b[i] > 0 else 0.0
        for i in range(len(b))
    }

    # ---- Sort nodes by descending adjusted centrality ----
    ranked_nodes = sorted(
        cost_adjusted_scores, key=cost_adjusted_scores.get, reverse=True
    )

    # ---- Budget-aware greedy selection ----
    chosen_nodes = []
    total_cost = 0.0

    for node in ranked_nodes:
        node_cost = b[node]

        if total_cost + node_cost > B:
            continue   # skip if over budget

        chosen_nodes.append(node)
        total_cost += node_cost

        if total_cost >= B:
            break

    return chosen_nodes

def attack_set_by_Kg_betweenness(adj, b, B):
    adj = np.asarray(adj)
    G = nx.from_numpy_array(adj)
    result = nx.betweenness_centrality(G)
    # print(result)
    b = b.mean(dim=0)
    #print("I am node grad and graph centrality", b.shape, result.shape)
    for i in range(adj.shape[0]):
        node_grads = b[:,i]
        node_saliency = torch.norm(F.relu(node_grads)).item()
        result[i] = result[i] / node_saliency
    d_order = sorted(result.items(), key=lambda x: x[1], reverse=True)
    l = [x[0] for x in d_order]
    chosen_nodes = l[:B]
    #print(chosen_nodes)
    return chosen_nodes


def temporal_attention_from_data(x):  # x: [B,T,N,C]
    # energy per timestep = average L2 over batch, nodes, channels
    e_t = torch.norm(x, dim=-1).mean(dim=(0,2))         # [T]
    a_t = torch.softmax(e_t, dim=0)                      # [T]
    return a_t


def spatial_attention_from_data(x, pairwise=True):       # x: [B,T,N,C]
    # node descriptors by averaging over B and T
    H = x.mean(dim=(0,1))                                # [N,C]
    # per-node saliency (vector)
    e_n = torch.norm(H, dim=-1)                          # [N]
    a_n = torch.softmax(e_n, dim=0)                      # [N]

    if not pairwise:
        return a_n, None

    # pairwise attention via cosine similarity
    Hn = F.normalize(H, dim=-1)                          # [N,C]
    A = Hn @ Hn.t()                                      # [N,N], in [-1,1]
    A = A.masked_fill(torch.eye(A.size(0), device=A.device).bool(), 0.0)
    # row-normalize to get outgoing attention (can also softmax row-wise)
    A = torch.softmax(A, dim=-1)                         # [N,N]
    return a_n, A


def batch_weighted_map(input_grads):
    #print("shape of grads node before", input_grads.shape )
    B, T, N, C = input_grads.shape
    device = input_grads.device
    attn_t = temporal_attention_from_data(input_grads)           # [T]
    attn_s_vec, attn_s_mat = spatial_attention_from_data(input_grads, pairwise=True)  # [N], [N,N]

    attn_t = torch.softmax(attn_t.to(device), dim=0)      # [T]
    attn_s_vec = torch.softmax(attn_s_vec.to(device), dim=0)      # [N]

    W = attn_t[:, None] * attn_s_vec[None, :] 
    
    input_grads = input_grads.mean(dim=0)
    node_saliency_map = []

    #print("shape of W", W.shape) 

    for n in range(input_grads.shape[1]): # nth node
        node_grads = input_grads[:,n]
        w_part = W[:,n]
        node_saliency = torch.norm(F.relu(node_grads)*w_part[:, None]).item()
        node_saliency_map.append(node_saliency)
    #print("shape of grads node after", input_grads.shape )
    #print(node_saliency_map)
    sorted_id = sorted(range(len(node_saliency_map)), key=lambda k: node_saliency_map[k], reverse=False)
    print(sorted_id)
    return node_saliency_map, sorted_id


def attack_set_by_weighted_attention(input_grads, attack_nodes):
    node_saliency_map, sorted_id = batch_weighted_map(input_grads)
    chosen_nodes = [sorted_id[i] for i in range(attack_nodes)]

    return chosen_nodes

def _safe_std(a, axis=0):
    return np.sqrt(np.nanmean((a - np.nanmean(a, axis=axis, keepdims=True))**2, axis=axis))


def _mean_abs_derivative(X_TN):
    # finite differences along time, mean absolute value
    d = np.diff(X_TN, axis=0)
    return np.nanmean(np.abs(d), axis=0)

def _peak_rate(X_TN, z=2.0):
    # Count z-score exceedances per node / T
    mu = np.nanmean(X_TN, axis=0, keepdims=True)
    sd = np.nanstd(X_TN, axis=0, keepdims=True) + 1e-8
    Z = (X_TN - mu) / sd
    peaks = np.nanmean((np.abs(Z) > z).astype(float), axis=0)  # fraction of timesteps
    return peaks

def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

def _highfreq_power_ratio(X_TN, frac_cut=0.25):
    """
    Ratio of spectral power in higher frequencies (above frac_cut of Nyquist) per node.
    Uses real FFT along time.
    """
    T, N = X_TN.shape
    # demean to remove DC bias
    Xc = X_TN - np.nanmean(X_TN, axis=0, keepdims=True)
    # Replace remaining NaNs with 0 for FFT stability (assumes sparse NaNs)
    Xc = np.where(np.isnan(Xc), 0.0, Xc)

    # rFFT: frequencies from 0..Nyquist inclusive
    F = np.fft.rfft(Xc, axis=0)
    P = (F.real**2 + F.imag**2)  # power spectrum
    K = P.shape[0]               # number of freq bins
    cut = int(np.floor(frac_cut * (K-1)))  # index threshold
    total = P.sum(axis=0) + 1e-12
    high = P[cut+1: , :].sum(axis=0) if cut+1 < K else np.zeros(N)
    return high / total

def _lag1_autocorr(X_TN):
    """
    1-lag autocorrelation per node. Returns value in [-1,1].
    """
    T, N = X_TN.shape
    x = X_TN - np.nanmean(X_TN, axis=0, keepdims=True)
    x0 = x[:-1, :]
    x1 = x[1:, :]
    num = np.nansum(x0 * x1, axis=0)
    den = (np.sqrt(np.nansum(x0**2, axis=0)) * np.sqrt(np.nansum(x1**2, axis=0))) + 1e-12
    return num / den

def spectral_entropy(x, fs=1.0, normalize=True):
    x = x - np.mean(x)
    psd = np.abs(np.fft.rfft(x))**2
    psd /= psd.sum() + 1e-12
    se = -(psd * np.log(psd + 1e-12)).sum()
    if normalize:
        se /= np.log(len(psd))
    return se

def sample_entropy(x, m=2, r=None):
    x = np.asarray(x)
    n = len(x)
    if r is None:
        r = 0.2 * np.std(x)
    def _phi(m):
        x_m = np.array([x[i:i+m] for i in range(n - m + 1)])
        C = np.sum(np.abs(x_m[:, None] - x_m[None, :]).max(axis=2) <= r, axis=0) - 1
        return np.sum(C) / (n - m + 1)
    return -np.log((_phi(m + 1) + 1e-12) / (_phi(m) + 1e-12))

# temporal feature difference
def _to_TN(X):
    """
    Normalize input shape to (T, N).
    Accepts (T, N) or (B, T, N, F). If batched, it averages over batch & feature dims.
    """
    X = X.detach().cpu().numpy()
    if X.ndim == 2:            # (T, N)
        return X
    elif X.ndim == 4:          # (B, T, N, F)
        return X.mean(axis=(0, 3))  # -> (T, N)
    else:
        raise ValueError(f"X must be (T,N) or (B,T,N,F), got shape {X.shape}")

def _to_new_TN(X):
    """
    Normalize input shape to (T, N).
    Accepts (T, N) or (B, T, N, F). If batched, it averages over batch & feature dims.
    """
    X = X.detach().cpu().numpy()
    if X.ndim == 2:            # (T, N)
        return X
    elif X.ndim == 4:          # (B, T, N, F)
        return X.mean(axis=(0, 2))  # -> (T, N)
    else:
        raise ValueError(f"X must be (T,N) or (B,T,N,F), got shape {X.shape}")


def _shannon_entropy(X_TN, bins=32):
    # Entropy via histogram per node
    N = X_TN.shape[1]
    ent = np.empty(N)
    for j in range(N):
        x = X_TN[:, j]
        x = x[~np.isnan(x)]
        if x.size == 0:
            ent[j] = 0.0
            continue
        hist, _ = np.histogram(x, bins=bins, density=True)
        p = hist / (hist.sum() + 1e-12)
        p = p[p > 0]
        ent[j] = -(p * np.log(p + 1e-12)).sum()  # nats
    # Normalize to [0,1] by dividing by log(bins)
    return ent / (np.log(bins) + 1e-12)

def permutation_entropy(x, order=3, delay=1):
    x = np.asarray(x)
    n = len(x)
    if n < order * delay:
        return np.nan
    # Build ordinal patterns
    patterns = []
    for i in range(n - delay * (order - 1)):
        window = x[i:(i + delay * order):delay]
        patterns.append(tuple(np.argsort(window)))
    # Count patterns
    counts = {p: 0 for p in permutations(range(order))}
    for p in patterns:
        counts[p] += 1
    p = np.array(list(counts.values()), dtype=float)
    p /= p.sum() + 1e-12
    p = p[p > 0]
    return -np.sum(p * np.log(p)) / np.log(len(counts))


def _minmax_norm(v):
    v = np.asarray(v, dtype=float)
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if np.isclose(vmax, vmin):
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)

def compute_degree(adj, directed=False, weight=True):
    """
    Return degree vector (length N). If directed=True, returns out-degree.
    If weight=False, uses binary adjacency.
    """
    A = np.asarray(adj).copy()
    if not weight:
        A = (A != 0).astype(float)
    if directed:
        # out-degree (sum of row)
        return A.sum(axis=1)
    else:
        # undirected degree
        return (A + A.T).sum(axis=1) / 2.0

def compute_centrality_measures(adj, weighted=True, directed=False):
    """
    Compute multiple centrality vectors for each node.
    Returns dict of numpy arrays, each of length N.
    """
    A = np.asarray(adj)
    N = A.shape[0]
    if directed:
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_array(A)

    # Handle weights
    weight_key = 'weight' if weighted else None

    # Compute centralities
    degree_c = np.array([v for k, v in nx.degree_centrality(G).items()])
    between_c = np.array([v for k, v in nx.betweenness_centrality(G, weight=weight_key).items()])
    close_c = np.array([v for k, v in nx.closeness_centrality(G).items()])
    try:
        eigen_c = np.array([v for k, v in nx.eigenvector_centrality_numpy(G, weight=weight_key).items()])
    except nx.NetworkXException:
        eigen_c = np.zeros(N)

    return between_c


def compute_pagerank(adj, directed=False, weight=True, alpha=0.85, tol=1e-6, max_iter=100):
    A = np.asarray(adj, dtype=float).copy()
    N = A.shape[0]
    np.fill_diagonal(A, 0.0)

    if not weight:
        A = (A != 0).astype(float)

    if directed:
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    else:
        # Symmetrize for undirected PageRank
        A = np.maximum(A, A.T)
        G = nx.from_numpy_array(A)

    pr = nx.pagerank(
        G,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        weight="weight" if weight else None
    )

    # Convert dict to numpy array
    pr_vec = np.array([pr[i] for i in range(N)], dtype=float)
    return pr_vec
    

def select_timestamp(model, A_wave, edges, edge_weights, X, y):
    k = 3
    X_ = X.clone().detach().requires_grad_(True)
    with torch.enable_grad():
            loss = nn.MSELoss()(model(X_,A_wave, edges, edge_weights), y)
    loss.backward()
    sal = X_.grad.abs()
    sal = sal.mean(dim = 2)

    # sal: (B, T, N)
    #print("shape of saliency", sal.shape)
    sal_mean = sal.mean(dim=0)   # -> (T, N)

    # now do top-k along time for each node
    scores_topk, topk_time_idx = torch.topk(sal_mean.T, k=k, dim=0)

    # transpose if you want (K, N)
    topk_time_idx = topk_time_idx.T
    
    return topk_time_idx, scores_topk


import torch

def select_timestamp_local_energy(X, k=3):
    """
    High local variance / energy based timestamp selection.
    Idea: pick timestamps with large local absolute change |x[t] - x[t-1]|.

    Args:
        X: input tensor with shape (B, T, N) or (B, T, N, F)
           B: batch, T: time, N: nodes, F: features (optional)
        k: top-k timestamps per node

    Returns:
        topk_time_idx: (k, N) tensor of time indices for each node
        scores_topk:  (N, k) tensor of energy scores for those timestamps
    """

    # Ensure we don't modify original
    X_ = X.detach()

    # Handle both (B, T, N) and (B, T, N, F)
    if X_.dim() == 4:
        # (B, T, N, F)
        # local change along time
        diff = X_[:, 1:, :, :] - X_[:, :-1, :, :]   # (B, T-1, N, F)
        # energy = mean over batch + feature dims -> (T-1, N)
        energy = diff.abs().mean(dim=(0, -1))
    elif X_.dim() == 3:
        # (B, T, N)
        diff = X_[:, 1:, :] - X_[:, :-1, :]         # (B, T-1, N)
        # energy = mean over batch -> (T-1, N)
        energy = diff.abs().mean(dim=0)
    else:
        raise ValueError(f"Expected X to have 3 or 4 dims, got {X_.dim()}")

    # Optional: pad to get a score for the first time step as well
    # Here we just copy the first diff value
    # energy: (T-1, N) -> (T, N)
    energy = torch.cat([energy[0:1, :], energy], dim=0)

    # Now: energy has shape (T, N), similar to sal_mean in your original code

    # top-k along time per node
    # energy.T: (N, T)
    scores_topk, topk_time_idx = torch.topk(energy.T, k=k, dim=1)

    # match your saliency function's output style: (k, N) indices
    topk_time_idx = topk_time_idx.T

    return topk_time_idx, scores_topk


def select_timestamp_edge_detector(X, k=3):
 
    # Don't touch the original tensor
    X_ = X.detach()

    # 1) Collapse over batch (+ features if present) to get a clean time series per node
    if X_.dim() == 4:
        # (B, T, N, F) -> mean over batch and feature dims: (T, N)
        signal = X_.mean(dim=(0, -1))
    elif X_.dim() == 3:
        # (B, T, N) -> mean over batch: (T, N)
        signal = X_.mean(dim=0)
    else:
        raise ValueError(f"Expected X to have 3 or 4 dims, got {X_.dim()}")

    # 2) First derivative along time: simple temporal edge detector
    # diff[t] ≈ signal[t+1] - signal[t]
    diff = signal[1:, :] - signal[:-1, :]   # (T-1, N)

    # Edge strength = magnitude of derivative
    edge_strength = diff.abs()              # (T-1, N)

    # 3) Optionally pad the first time step so we get (T, N) scores
    # Here we just repeat the first derivative value
    edge_strength = torch.cat([edge_strength[0:1, :], edge_strength], dim=0)  # (T, N)

    # 4) Top-k along time per node
    # edge_strength.T: (N, T)
    scores_topk, topk_time_idx = torch.topk(edge_strength.T, k=k, dim=1)

    # Match your saliency function's style: (k, N) indices
    topk_time_idx = topk_time_idx.T

    return topk_time_idx, scores_topk


def select_timestamp_random(X, k=3):
    # Determine T, N from the input
    if X.dim() == 4:
        # (B, T, N, F)
        _, T, N, _ = X.shape
    elif X.dim() == 3:
        # (B, T, N)
        _, T, N = X.shape
    else:
        raise ValueError(f"Expected X to have 3 or 4 dims, got {X.dim()}")

    # Randomly pick k timestamps *per node*
    # Shape desired = (k, N)
    topk_time_idx = torch.randint(low=0, high=T, size=(k, N))

    # Dummy scores (since random has no score)
    # Some baselines use zeros; some use uniform random values.
    scores_topk = torch.zeros(N, k)

    return topk_time_idx, scores_topk


def select_timestamp_frequency(X, k=7, high_freq_ratio=0.3):
    # Don't modify original
    X_ = X.detach()

    # 1) Collapse over batch (+ features) → per-node time series: (T, N)
    if X_.dim() == 4:
        # (B, T, N, F) -> mean over batch and feature dims
        signal = X_.mean(dim=(0, -1))  # (T, N)
    elif X_.dim() == 3:
        # (B, T, N) -> mean over batch
        signal = X_.mean(dim=0)        # (T, N)
    else:
        raise ValueError(f"Expected X to have 3 or 4 dims, got {X_.dim()}")

    T, N = signal.shape

    # 2) FFT along time dimension
    # Use rfft as signal is real-valued: output shape (T_fft, N)
    freq = torch.fft.rfft(signal, dim=0)   # (T_fft, N)
    T_fft = freq.size(0)

    # 3) Simple high-pass filter in frequency domain
    # Keep only the highest 'high_freq_ratio' portion of freq bins
    keep_bins = max(1, int(T_fft * high_freq_ratio))
    cutoff = T_fft - keep_bins

    # Zero out low frequencies [0:cutoff)
    high_freq = freq.clone()
    high_freq[:cutoff, :] = 0.0

    # 4) Inverse FFT to reconstruct high-frequency component in time domain
    hf_signal = torch.fft.irfft(high_freq, n=T, dim=0)  # (T, N)

    # 5) Spectral energy per timestamp & node
    # Use absolute value as energy measure
    energy = hf_signal.abs()  # (T, N)

    # 6) Top-k along time per node
    # energy.T: (N, T)
    scores_topk, topk_time_idx = torch.topk(energy.T, k=k, dim=1)

    # Match your other selectors: (k, N) indices
    topk_time_idx = topk_time_idx.T  # (k, N)

    return topk_time_idx, scores_topk


def select_timestamp_edge_bias(X, k=3):

    # Determine T, N from the input
    if X.dim() == 4:
        # (B, T, N, F)
        _, T, N, _ = X.shape
    elif X.dim() == 3:
        # (B, T, N)
        _, T, N = X.shape
    else:
        raise ValueError(f"Expected X to have 3 or 4 dims, got {X.dim()}")

    device = X.device

    # ---- 1) Define edge-bias scores over time ----
    # time_idx: 0, 1, ..., T-1
    time_idx = torch.arange(T, device=device)

    # Distance to nearest boundary (start or end)
    # dist_to_start = t
    # dist_to_end   = T-1 - t
    dist_to_start = time_idx
    dist_to_end   = (T - 1) - time_idx
    dist_to_edge  = torch.minimum(dist_to_start, dist_to_end)  # smaller near edges

    # Convert distance (small at edges) -> score (large at edges)
    # max_dist is at center; subtract distances from max to get highest at edges.
    max_dist = dist_to_edge.max()
    edge_scores_time = max_dist - dist_to_edge  # shape (T,)

    # ---- 2) Make this per-node: (T, N) ----
    # Same edge pattern for every node
    scores = edge_scores_time.unsqueeze(1).expand(T, N)  # (T, N)

    # ---- 3) Top-k timestamps per node ----
    # scores.T: (N, T)
    scores_topk, topk_time_idx = torch.topk(scores.T, k=min(k, T), dim=1)

    # Match your convention: (k, N) indices
    topk_time_idx = topk_time_idx.T  # (k, N)

    return topk_time_idx, scores_topk


def attack_set_by_kgspsa(score_vector, b, B):
    """
    score_vector : tensor of shape [N] - influence per cost for each node
    b            : tensor/list of shape [N] - cost of each node
    B            : total budget

    Returns:
        selected_nodes : list of node indices selected under budget B
    """

    # Convert to CPU numpy for sorting (safe, faster)
    scores = score_vector.detach().cpu().numpy()
    costs  = b.detach().cpu().numpy() if torch.is_tensor(b) else np.array(b)

    # Sort nodes by DESCENDING score (highest influence first)
    sorted_idx = np.argsort(scores)[::-1]

    selected_nodes = []
    total_cost = 0.0

    for idx in sorted_idx:
        node_cost = costs[idx]

        # If adding this node exceeds budget, skip
        if total_cost + node_cost > B:
            continue

        # Otherwise add the node
        selected_nodes.append(idx)
        total_cost += node_cost

        # Budget exhausted → stop
        if total_cost >= B:
            break

    return selected_nodes



def select_timestamp_temporal_centrality(X, A, k=1):
    X_ = X.detach()

    # ---- 1) Local temporal activity per node: |x[t] - x[t-1]| ----
    if X_.dim() == 4:
        # (B, T, N, F)
        diff = X_[:, 1:, :, :] - X_[:, :-1, :, :]   # (B, T-1, N, F)
        # average over batch and feature dims -> (T-1, N)
        activity = diff.abs().mean(dim=(0, -1))
    elif X_.dim() == 3:
        # (B, T, N)
        diff = X_[:, 1:, :] - X_[:, :-1, :]         # (B, T-1, N)
        # average over batch -> (T-1, N)
        activity = diff.abs().mean(dim=0)
    else:
        raise ValueError(f"Expected X to have 3 or 4 dims, got {X_.dim()}")

    T_minus_1, N = activity.shape

    # Pad to get a score for all T time steps: (T, N)
    activity = torch.cat([activity[0:1, :], activity], dim=0)  # (T, N)
    T = activity.size(0)

    # ---- 2) Graph-aware spreading: neighbors' activity -> temporal centrality ----
    # Normalize adjacency row-wise (simple random-walk style)
    A = A.to(X_.device)
    row_sum = A.sum(dim=1, keepdim=True) + 1e-8
    A_norm = A / row_sum                      # (N, N)

    # centrality[t, i] ≈ avg activity at node i and its neighbors at time t
    # activity: (T, N); A_norm.T: (N, N)
    # (activity @ A_norm.T)[t, i] = sum_j activity[t, j] * A_norm[i, j]
    neighbor_activity = activity @ A_norm.T   # (T, N)

    # Combine self activity + neighbor activity (you can tune weights)
    centrality = 0.5 * activity + 0.5 * neighbor_activity  # (T, N)

    # ---- 3) Top-k timestamps per node ----
    # centrality.T: (N, T)
    scores_topk, topk_time_idx = torch.topk(centrality.T, k=min(k, T), dim=1)

    # Match your convention: (k, N)
    topk_time_idx = topk_time_idx.T

    return topk_time_idx, scores_topk


@torch.no_grad()
def zo_estimate_grad(
    model,
    X_adv,
    y,
    A_wave,
    edges,
    edge_weights,
    mu=0.01,
    q=1,
    mask=None,
):
    """
    Two-sided zeroth-order gradient estimate:
        g ≈ (1/q) * Σ_i [(L(x+μu_i) - L(x-μu_i)) / (2μ)] * u_i
    """
    device = X_adv.device
    if mask is None:
        mask = torch.ones_like(X_adv, device=device)
    else:
        mask = mask.to(device)

    grad_hat = torch.zeros_like(X_adv, device=device)

    for _ in range(q):
        # Rademacher direction u ∈ {-1, +1}^d, masked
        u = torch.randint_like(X_adv, low=0, high=2, dtype=torch.float32, device=device)
        u = (u * 2.0 - 1.0) * mask

        X_plus  = X_adv + mu * u
        X_minus = X_adv - mu * u

        pred_plus  = model(X_plus,  A_wave, edges, edge_weights)
        pred_minus = model(X_minus, A_wave, edges, edge_weights)

        loss_plus  = F.mse_loss(pred_plus,  y)
        loss_minus = F.mse_loss(pred_minus, y)

        grad_hat += (loss_plus - loss_minus) / (2.0 * mu) * u

    grad_hat = grad_hat / float(q)
    return grad_hat

def zo_sgd_attack(
    model,
    X,
    y,
    A_wave,
    edges,
    edge_weights,
    epsilon=0.1,
    step_size=0.01,
    num_steps=20,
    mu=0.01,
    q=1,
    mask=None,
    Random=True,
    clamp_min=0.0,
    clamp_max=1.0,
):
    """
    Zeroth-Order SGD-based black-box attack for traffic forecasting.
    PGD-style updates using numerically estimated gradients (no .backward()).
    """

    device = X.device
    X = X.to(device)
    y = y.to(device)

    if mask is None:
        mask = torch.ones_like(X, device=device)
    else:
        mask = mask.to(device)

    # Perturbation variable (what we optimize)
    delta = torch.zeros_like(X, device=device)

    # Random init in L_inf-ball if desired
    if Random:
        delta.uniform_(-epsilon, epsilon)
        delta = delta * mask

    # Project initial delta and build X_adv
    delta = torch.clamp(delta, -epsilon, epsilon)
    X_adv = torch.clamp(X + delta, clamp_min, clamp_max)

    for _ in range(num_steps):
        # 1) Estimate zeroth-order gradient wrt X_adv
        grad_hat = zo_estimate_grad(
            model,
            X_adv,
            y,
            A_wave,
            edges,
            edge_weights,
            mu=mu,
            q=q,
            mask=mask,
        )

        # 2) ZO-SGD step on delta
        #    You can use grad_hat directly or its sign (for L_inf)
        delta = delta + step_size * torch.sign(grad_hat) * mask

        # 3) Project to epsilon-ball and clamp to valid data range
        delta = torch.clamp(delta, -epsilon, epsilon)
        X_adv = torch.clamp(X + delta, clamp_min, clamp_max)

    return delta, X_adv


def zo_adam_attack(
    model,
    X,
    y,
    A_wave,
    edges,
    edge_weights,
    epsilon=0.1,
    step_size=0.01,
    num_steps=20,
    mu=0.01,
    q=1,
    mask=None,
    Random=True,
    clamp_min=0.0,
    clamp_max=1.0,
    beta1=0.9,
    beta2=0.999,
    adam_eps=1e-8,
):
    """
    Zeroth-Order Adam-based black-box attack for traffic forecasting.
    Uses Adam on the perturbation delta with numerically estimated gradients.
    """

    device = X.device
    X = X.to(device)
    y = y.to(device)

    if mask is None:
        mask = torch.ones_like(X, device=device)
    else:
        mask = mask.to(device)

    # Perturbation variable
    delta = torch.zeros_like(X, device=device)

    # Random init
    if Random:
        delta.uniform_(-epsilon, epsilon)
        delta = delta * mask

    # Project and form X_adv
    delta = torch.clamp(delta, -epsilon, epsilon)
    X_adv = torch.clamp(X + delta, clamp_min, clamp_max)

    # Adam states
    m = torch.zeros_like(delta, device=device)
    v = torch.zeros_like(delta, device=device)

    for t in range(1, num_steps + 1):
        # 1) Zeroth-order gradient estimate at current X_adv
        grad_hat = zo_estimate_grad(
            model,
            X_adv,
            y,
            A_wave,
            edges,
            edge_weights,
            mu=mu,
            q=q,
            mask=mask,
        )

        # 2) Adam update on delta
        m = beta1 * m + (1.0 - beta1) * grad_hat
        v = beta2 * v + (1.0 - beta2) * (grad_hat * grad_hat)

        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)

        # Adam step; optionally multiply by sign if you want strict L_inf step
        delta = delta + step_size * m_hat / (torch.sqrt(v_hat) + adam_eps)
        delta = delta * mask

        # 3) Project to epsilon-ball and valid domain
        delta = torch.clamp(delta, -epsilon, epsilon)
        X_adv = torch.clamp(X + delta, clamp_min, clamp_max)

    return delta, X_adv


def evolutionary_attack(
    model,
    X,
    y,
    A_wave,
    edges,
    edge_weights,
    epsilon=0.1,
    clamp_min=0.0,
    clamp_max=1.0,
    mask=None,
    Random=True,
    pop_size=20,
    num_generations=30,
    elite_frac=0.2,
    mutation_std=0.02,
    crossover_prob=0.5, 
    ):
  
    device = X.device
    X = X.to(device)
    y = y.to(device)

    if mask is None:
        mask = torch.ones_like(X, device=device)
    else:
        mask = mask.to(device)

    # ---- 1) Initialize population of perturbations delta ----
    # Shape: (pop_size, *X.shape)
    pop_shape = (pop_size,) + X.shape
    if Random:
        delta_pop = torch.empty(pop_shape, device=device).uniform_(-epsilon, epsilon)
        delta_pop = delta_pop * mask.unsqueeze(0)
    else:
        delta_pop = torch.zeros(pop_shape, device=device)

    # Project initial deltas to L_inf ball
    delta_pop = torch.clamp(delta_pop, -epsilon, epsilon)

    # Helper for evaluating fitness of entire population
    @torch.no_grad()
    def evaluate_population(delta_population):
        """
        delta_population: (P, *X.shape)
        Returns: losses (P,)
        """
        P = delta_population.size(0)
        # Broadcast X: (1, ...) -> (P, ...)
        X_batch = X.unsqueeze(0).expand(P, *X.shape)
        X_adv_batch = torch.clamp(X_batch + delta_population, clamp_min, clamp_max)

        # Flatten population into batch dimension for model
        # Assumes model can handle batch size P * B
        B = X.shape[0]
        # Merge population and batch dimensions: (P, B, ...) -> (P*B, ...)
        X_adv_flat = X_adv_batch.reshape(P * B, *X.shape[1:])

        preds_flat = model(X_adv_flat, A_wave, edges, edge_weights)
        # Reshape back: (P*B, ...) -> (P, B, ...)
        preds = preds_flat.view(P, B, *preds_flat.shape[1:])

        # Compute MSE per individual, average over batch and all prediction dims
        # y: (B, ...) -> (1, B, ...) -> (P, B, ...)
        y_exp = y.unsqueeze(0).expand(P, *y.shape)
        losses = F.mse_loss(preds, y_exp, reduction='none')
        # Average over all dims except population dim
        while losses.dim() > 1:
            losses = losses.mean(dim=-1)
        # losses: (P,)
        return losses

    # Track best
    best_delta = None
    best_loss = None

    num_elite = max(1, int(pop_size * elite_frac))

    for gen in range(num_generations):
        # ---- 2) Evaluate fitness (loss) ----
        losses = evaluate_population(delta_pop)  # (pop_size,)

        # Update best individual
        max_loss, max_idx = torch.max(losses, dim=0)
        if best_loss is None or max_loss > best_loss:
            best_loss = max_loss
            best_delta = delta_pop[max_idx].clone().detach()

        # ---- 3) Select elites ----
        # Higher loss = stronger attack
        sorted_idx = torch.argsort(losses, descending=True)
        elite_idx = sorted_idx[:num_elite]
        elites = delta_pop[elite_idx]  # (num_elite, *X.shape)

        # ---- 4) Create new population via crossover + mutation ----
        new_pop = []

        # Always keep elites
        new_pop.append(elites)

        # How many offspring to generate
        num_offspring = pop_size - num_elite

        for _ in range(num_offspring):
            # Pick two parents from elites
            parents_idx = torch.randint(0, num_elite, (2,), device=device)
            p1 = elites[parents_idx[0]]
            p2 = elites[parents_idx[1]]

            # Crossover: gene-wise blend or uniform crossover
            # Here: uniform crossover on each element
            # mask_c: Bernoulli(crossover_prob)
            mask_c = (torch.rand_like(p1, device=device) < crossover_prob).float()
            child = mask_c * p1 + (1.0 - mask_c) * p2

            # Mutation: Gaussian noise
            mutation = torch.randn_like(child, device=device) * mutation_std * epsilon
            child = child + mutation * mask

            # Projection: keep within L_inf ball
            child = torch.clamp(child, -epsilon, epsilon)

            new_pop.append(child.unsqueeze(0))

        delta_pop = torch.cat(new_pop, dim=0)  # (pop_size, *X.shape)

    # ---- Final best adversarial example ----
    X_adv_best = torch.clamp(X + best_delta, clamp_min, clamp_max).detach()
    return best_delta, X_adv_best


def _ST_fgsm_whitebox(model,
                  X,
                  y,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  Random,
                  find_type,
                  **kwargs):

    #X: Input data of shape (batch_size, num_nodes,
    # num_features=in_channels, num_timesteps,).

    X_fgsm = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1,1, num_features, steps_length).cuda() # [1,1 ,num of channel, number of time length]

    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'kg_betweeness':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 1
        step_size = epsilon
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)


        index = attack_set_by_Kg_betweenness(A_wave.cpu().detach.numpy, inputs_grad, k)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 1
        step_size = epsilon
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError



    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:,index,:,:] = ones_mat

    opt = optim.SGD([X_fgsm], lr=1e-3)
    opt.zero_grad()

    with torch.enable_grad():
        loss = nn.MSELoss()(model(X_fgsm,A_wave, edges, edge_weights), y)
    loss.backward()
    # how to clamp

    # second clamp the value according to  the neighbourhood value [min, max]
    # define the epsilon: stds: parameter free
    #X_fgsm = X_fgsm.data + epsilon * chosen_attack_nodes * X_fgsm.grad.data.sign()
    #print('X_fgsm', X_fgsm.shape)
    X_fgsm = Variable(torch.clamp(X_fgsm.data + epsilon * chosen_attack_nodes * X_fgsm.grad.data.sign(), 0.0, 1.0), requires_grad=True)

    # print('err fgsm (white-box): ', err_pgd)
    return X, X_fgsm, index

def _ST_fgsm_semibox(model,
                  X,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  Random,
                  find_type,
                  output_len,
                  transform_ground_truth='no-linear',
                  **kwargs):

    #X: Input data of shape (batch_size, num_nodes,
    # num_features=in_channels, num_timesteps,).

    X_fgsm = Variable(X.data, requires_grad=True)
    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1,1, num_features, steps_length).cuda() # [1,1 ,num of channel, number of time length]

    if transform_ground_truth == 'linear':
        if num_features > 1:
            y = Variable(X.data[:, :,0,:], requires_grad=True)
        else:
            y = Variable(X.data, requires_grad=True).squeeze(2)

        if output_len < steps_length:

            y = Variable(y.data[:,:,:output_len], requires_grad=True)
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
        else:
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
    elif transform_ground_truth == 'no-linear':
        y_bar = model(X_fgsm,A_wave, edges, edge_weights)
        y = y_bar + torch.FloatTensor(*y_bar.shape).uniform_(-epsilon / 10 , epsilon / 10 ).cuda()
    else:
        raise  NameError


    y = Variable(torch.clamp(y, 0, 1.0), requires_grad=True)

    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'shannon_entropy':
        x_f = X.data
        X_TN = _to_new_TN(x_f)  # (T, N)
        #print(X_TN.shape)
        X_TN = X_TN.transpose(1, 0)
        #T, N = X_TN.shape
        bins_entropy = 32
        score  = _shannon_entropy(X_TN, bins=bins_entropy)
        index = np.argsort(score)[-K:][::-1]
        #index = np.argsort(score)[:K]
        index = index.tolist()

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 1
        step_size = epsilon
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError



    #chosen_attack_nodes = torch.zeros_like(X)
    #chosen_attack_nodes[:,index,:,:] = ones_mat

    X = X.permute(0,3,1,2) # for freq

    #time_idx, time_score = select_timestamp(model, A_wave, edges, edge_weights, X, y)
    #time_idx, time_score = select_timestamp_local_energy(X, k=3)
    #time_idx, time_score = select_timestamp_edge_detector(X, k=3)
    #time_idx, time_score = select_timestamp_random(X, k=3)
    time_idx, time_score = select_timestamp_frequency(X, k=3, high_freq_ratio=0.3)
    #time_idx, time_score = select_timestamp_edge_bias(X, k=3)
    #time_idx, time_score = select_timestamp_temporal_centrality(X, A_wave, k=3)

    #print("time_idx ", time_idx.shape)
    
    # # pgd attack
    #print("shape of x", X.shape)

    X = X.permute(0,2,3,1) # for freq
    #X = X.permute(0, 3, 1, 2)

    #print("shape of x", X.shape)

    B, T, N, F = X.shape
    mask = torch. zeros_like(X)   
    
    for n in index:                     # n is a node id (scalar)
        #print(time_idx[n,:])
        #t_sel = time_idx[n,:]                # shape (K,) -> timesteps for this node
        t_sel = time_idx[:,n]        # for freq
        # set mask=1 for all batches, all features, selected times for this node
        mask[:, n, :, t_sel] = 1.0
        #mask[:, t_sel, n, :] = 1.0

    num_ones = mask.sum().item()
    num_zeros = mask.numel() - num_ones

    #print("Number of zeros and ones",num_zeros, num_ones)
    
    #chosen_attack_nodes = torch.zeros_like(X)
    #chosen_attack_nodes[:, index , : , :] = ones_mat

    #print("shape of x_pgd and mask", X_pgd.shape, mask.shape )

    opt = optim.SGD([X_fgsm], lr=1e-3)
    opt.zero_grad()

    with torch.enable_grad():
        loss = nn.MSELoss()(model(X_fgsm,A_wave, edges, edge_weights), y)
    loss.backward()


    # second clamp the value according to  the neighbourhood value [min, max]

    eta = epsilon * mask * X_fgsm.grad.data.sign()
    X_fgsm = Variable(torch.clamp(X_fgsm.data + eta, 0.0, 1.0), requires_grad=True)


    return eta, X_fgsm, index



def _ST_pgd_whitebox(model,
                  X,
                  y,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  num_steps,
                  Random,
                  step_size,
                  find_type,
                  **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)

    # first choose the nodes: randomly choose K nodes from total nodes
    #print("size of x in pgd",X.size())
    #X = X.permute(0,2,3,1)
    batch_size_x, steps_length, num_nodes, num_features  = X.size()
    ones_mat = torch.ones(1, steps_length, 1, num_features).cuda()  # [1,1 ,num of channel, number of time length]

    #selected_nodes = []
    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        #print("list and k", list, K)
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'kg_betweeness':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 1
        step_size = epsilon
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)


        index = attack_set_by_Kg_betweenness(A_wave.cpu().detach().numpy(), inputs_grad,  K)


    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5 
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                #y_truth = y[..., 0]              # -> [512, 12, 207]
                 # reorder to [B, N, T] to match y_pred
                #y_truth = y_truth.permute(0, 2, 1)
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        #print("the value of K in saliency", K)
        index = attack_set_by_saliency_map(inputs_grad, K)
        #print("slected nodes for purturbation")
        #print(index)
        #selected_nodes.append(index)
        #index = attack_set_by_weighted_attention(inputs_grad, K)
    
    elif find_type == 'shannon_entropy':
        x_f = X.data
        X_TN = _to_TN(x_f)  # (T, N)
        #T, N = X_TN.shape
        bins_entropy = 32
        score  = _shannon_entropy(X_TN, bins=bins_entropy)
        index = np.argsort(score)[-K:][::-1]
        #index = np.argsort(score)[:K]
        index = index.tolist()
    
    elif find_type == 'safe_std':
        x_f = X.data
        X_TN = _to_TN(x_f)  # (T, N)
        score = _safe_std(X_TN, axis=0)**2 
        index = np.argsort(score)[-K:][::-1]
        #index = np.argsort(score)[:K]
        index = index.tolist()
    
    elif find_type == 'mean_abs_derivative':
        x_f = X.data
        X_TN = _to_TN(x_f)  # (T, N)
        score = _mean_abs_derivative(X_TN)
        index = np.argsort(score)[-K:][::-1]
        #index = np.argsort(score)[:K]
        index = index.tolist()

    elif find_type == 'peak_rate':
        x_f = X.data
        X_TN = _to_TN(x_f)  # (T, N)
        z_for_peaks = 2.0
        score = _peak_rate(X_TN, z=z_for_peaks)
        index = np.argsort(score)[-K:][::-1]
        #index = np.argsort(score)[:K]
        index = index.tolist()

    elif find_type == 'highfreq':
        x_f = X.data
        X_TN = _to_TN(x_f)  # (T, N)
        frac_cut_highfreq=0.25
        score = _highfreq_power_ratio(X_TN, frac_cut=frac_cut_highfreq)
        index = np.argsort(score)[-K:][::-1]
        #index = np.argsort(score)[:K]
        index = index.tolist()

    elif find_type == 'auto_corr':
        x_f = X.data
        X_TN = _to_TN(x_f)  # (T, N)
        score = _lag1_autocorr(X_TN)
        index = np.argsort(score)[-K:][::-1]
        #index = np.argsort(score)[:K]
        index = index.tolist()

    elif find_type == 'spectral_entropy_nodes':
        x_f = X.data
        X_TN = _to_TN(x_f)  # (T, N)
        N = X_TN.shape[1]
        se = np.zeros(N)
        for j in range(N):
            se[j] = spectral_entropy(X_TN[:, j])

        index = np.argsort(se)[-K:][::-1]
        #index = np.argsort(score)[:K]
        index = index.tolist()

    elif find_type == 'sample_entropy_nodes':
        x_f = X.data
        X_TN = _to_TN(x_f)  # (T, N)
        N = X_TN.shape[1]
        m = 2
        r = None
        ent = np.zeros(N)
        for j in range(N):
            x = X_TN[:, j]
            ent[j] = sample_entropy(x, m, r)
        index = np.argsort(ent)[-K:][::-1]
        #index = np.argsort(score)[:K]
        index = index.tolist()

    elif find_type == 'permutation_entropy_nodes':
        x_f = X.data
        X_TN = _to_TN(x_f)  # (T, N)
        N = X_TN.shape[1]
        order = 3
        delay = 1
        pe = np.zeros(N)
        for j in range(N):
            pe[j] = permutation_entropy(X_TN[:, j], order, delay)

        index = np.argsort(pe)[-K:][::-1]
        #index = np.argsort(score)[:K]
        index = index.tolist()

    elif find_type == 'entropy_and_degree':
        x_f = X.data
        X_TN = _to_TN(x_f)  # (T, N)
        #A = np.asarray(A_wave)
        #N = A.shape[0]
        #T, N = X_TN.shape
        bins_entropy = 32
        alpha = 0.5
        beta = 0.5
        directed=False,
        degree_weighted=True,

        ent  = _shannon_entropy(X_TN, bins=bins_entropy)
        ent_n = _minmax_norm(ent)
        
        #index = np.argsort(score)[:K]
        #deg = compute_degree(A_wave.cpu().detach().numpy(), directed=directed, weight=degree_weighted)
        #deg = compute_centrality_measures(A_wave.cpu().detach().numpy(), weighted = degree_weighted, directed = directed)
        deg = compute_pagerank(A_wave.cpu().detach().numpy(), directed=False, weight=True)
        deg_n = _minmax_norm(deg)

        score = alpha * ent_n + beta * deg_n
        index = np.argsort(score)[-K:][::-1]
        index = index.tolist()

    else:
        raise  NameError

    #X = X.permute(0,3,1,2) # for freq
    #print("shape of input and output in attack method", X.shape, y.shape)
    #time_idx, time_score = select_timestamp(model, A_wave, edges, edge_weights, X, y)
    #time_idx, time_score = select_timestamp_local_energy(X, k=3)
    #time_idx, time_score = select_timestamp_edge_detector(X, k=3)
    #time_idx, time_score = select_timestamp_random(X, k=3)
    time_idx, time_score = select_timestamp_frequency(X, k=3, high_freq_ratio=0.3)
    #time_idx, time_score = select_timestamp_edge_bias(X, k=3)
    #time_idx, time_score = select_timestamp_temporal_centrality(X, A_wave, k=3)

    #print("time_idx ", time_idx.shape)
    #X = X.permute(0,2,3,1) # for freq
    
    # # pgd attack
    B, T, N, F = X.shape
    mask = torch. zeros_like(X)   
    for n in index:                         # n is a node id (scalar)
        t_sel = time_idx[:,n]                # shape (K,) -> timesteps for this node
        # set mask=1 for all batches, all features, selected times for this node
        mask[:, t_sel, n, :] = 1.0

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, :, index, :] = ones_mat

    # pgd based attack
    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).cuda()
        X_pgd = Variable(X_pgd.data + chosen_attack_nodes * random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward()
        eta = step_size * mask * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        delta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + delta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    
    # l2 ball regularization
    # for _ in range(num_steps):
    #     opt = optim.SGD([X_pgd], lr=1e-3)
    #     opt.zero_grad()

    #     with torch.enable_grad():
    #         loss = nn.MSELoss()(model(X_pgd, A_wave, edges, edge_weights), y)
    #     loss.backward()

    #     g = X_pgd.grad.data * mask
    #     dim_extra = [1] * (g.dim() - 1)
    #     g_norm = g.view(g.size(0), -1).norm(p=2, dim=1).view(-1, *dim_extra) + 1e-8
    #     eta = step_size * g / g_norm

    #     X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

    #     # still L∞-projected
    #     delta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
    #     X_pgd = Variable(X.data + delta, requires_grad=True)
    #     X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)


    ## loss pgd
    # lambda_smooth = 0.1  # tune this
    # for _ in range(num_steps):
    #     opt = optim.SGD([X_pgd], lr=1e-3)
    #     opt.zero_grad()

    #     with torch.enable_grad():
    #         pred = model(X_pgd, A_wave, edges, edge_weights)
    #         mse_loss = nn.MSELoss()(pred, y)

    #         # perturbation
    #         delta = X_pgd - X

    #         # temporal smoothness: penalize |delta_t - delta_{t-1}|
    #         # assume X shape: (B, T, N, F) or (B, T, N)
    #         if delta.dim() == 4:
    #             diff_t = delta[:, 1:, :, :] - delta[:, :-1, :, :]
    #         else:
    #             diff_t = delta[:, 1:, :] - delta[:, :-1, :]

    #         smooth_loss = diff_t.pow(2).mean()  # L2 smoothness (TV-like)

    #         loss = mse_loss + lambda_smooth * smooth_loss

    #     loss.backward()
    #     eta = step_size * chosen_attack_nodes * X_pgd.grad.data.sign()
    #     X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

    #     delta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
    #     X_pgd = Variable(X.data + delta, requires_grad=True)
    #     X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    # SPSA based attack
    # X_adv = X.clone().detach()
  
    # # Optional random init inside L_inf ball (like your PGD code)
    # if Random:
    #     random_noise = torch.empty_like(X_adv).uniform_(-epsilon, epsilon)
    #     X_adv = X_adv + mask * random_noise
    #     # Project back to epsilon-ball and valid range
    #     delta = torch.clamp(X_adv - X, -epsilon, epsilon)
    #     X_adv = torch.clamp(X + delta, 0, 1.0).detach()

    # for _ in range(num_steps):
    #     # Sample Rademacher vector Δ ∈ {+1, -1}^d ----
    #     perturb = torch.randint_like(X_adv, low=0, high=2, dtype=torch.float32).cuda()
    #     perturb = (perturb * 2.0 - 1.0)  # 0/1 -> -1/+1
    #     perturb = perturb * mask         # respect mask (no change where mask == 0)
    #     X_plus  = X_adv + 0.01 * perturb
    #     X_minus = X_adv - 0.01 * perturb
    #     with torch.no_grad():
    #         pred_plus  = model(X_plus,  A_wave, edges, edge_weights)
    #         pred_minus = model(X_minus, A_wave, edges, edge_weights)

    #         loss_plus  = nn.MSELoss()(pred_plus,  y)
    #         loss_minus = nn.MSELoss()(pred_minus, y)

    #     # SPSA gradient estimate ----
    #     grad_hat = (loss_plus - loss_minus) / (2.0 * spsa_delta) * perturb
    #     # Update step (PGD-style, L_inf) ----
    #     X_adv = X_adv + step_size * torch.sign(grad_hat) * mask
    #     # Projection to epsilon-ball around X & clamp to valid range ----
    #     delta = torch.clamp(X_adv - X, -epsilon, epsilon)
    #     X_adv = X + delta
    #     X_adv = torch.clamp(X_adv, clamp_min, clamp_max).detach()

    # X_zo_sgd = zo_sgd_attack(
    # model,
    # X,
    # y,
    # A_wave,
    # edges,
    # edge_weights,
    # epsilon=epsilon,
    # step_size=step_size,
    # num_steps=num_steps,
    # mu=0.01,
    # q=4,
    # mask=mask,
    # Random=True,
    # )

    # X_zo_adam = zo_adam_attack(
    # model,
    # X,
    # y,
    # A_wave,
    # edges,
    # edge_weights,
    # epsilon=epsilon,
    # step_size=step_size,
    # num_steps=num_steps,
    # mu=0.01,
    # q=4,
    # mask=mask,
    # Random=True,
    # )

    # X_ea = evolutionary_attack(
    # model,
    # X,
    # y,
    # A_wave,
    # edges,
    # edge_weights,
    # epsilon=epsilon,
    # clamp_min=0.0,
    # clamp_max=1.0,
    # mask=mask,
    # Random=True,
    # pop_size=30,
    # num_generations=40,
    # elite_frac=0.2,
    # mutation_std=0.05,
    # crossover_prob=0.5,
    # )

    return X, X_pgd, index




def _pgd_whitebox(model,
                  X,
                  y,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  num_steps,
                  Random,
                  step_size,
                  find_type,
                  **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, steps_length, num_nodes, num_features = X.size() 
    ones_mat = torch.ones(1, steps_length, 1, num_features).cuda()  # [1,1 ,num of channel, number of time length]

    


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)


    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, :, index, :] = ones_mat
    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).cuda()
        X_pgd = Variable(X_pgd.data + chosen_attack_nodes * random_noise, requires_grad=True)


    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon) * chosen_attack_nodes
    X_pgd = Variable(X.data + eta, requires_grad=True)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    return X, X_pgd, index



def _ST_pgd_semibox(model,
                    X,
                    A_wave,
                    edges,
                    edge_weights,
                    K,
                    epsilon,
                    num_steps,
                    Random,
                    step_size,
                    find_type,
                    output_len,
                    transform_ground_truth='no-linear', **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]
    if transform_ground_truth == 'linear':
        if num_features > 1:
            y = Variable(X.data[:, :,0,:], requires_grad=True)
        else:
            y = Variable(X.data, requires_grad=True).squeeze(2)

        if output_len < steps_length:

            y = Variable(y.data[:,:,:output_len], requires_grad=True)
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
        else:
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
    elif transform_ground_truth == 'no-linear':
        y_bar = model(X_pgd,A_wave, edges, edge_weights)
        y = y_bar + torch.FloatTensor(*y_bar.shape).uniform_(-epsilon / 10, epsilon / 10 ).cuda()
    else:
        raise  NameError


    y = Variable(torch.clamp(y, 0, 1.0), requires_grad=True)


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'kgpagerank':
        index = attack_set_by_pagerank_budget(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'kgbetweeness':
         index = attack_set_by_betweenness_budget(A_wave.cpu().detach().numpy(), K)


    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 1
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)

    elif find_type == 'kgspsa':

        # --- PARAMETERS ---
      node_num = X.shape[1]          # B, T, N, F → using N = number of nodes
      seq_len  = X.shape[3]          # time steps
      iter_spsa = 5                 # SPSA iterations per node (adjust)
      w = torch.ones(seq_len).cuda() # weight vector for time dimension
      score_vector = torch.zeros(node_num).cuda()
      
      b = torch.ones(node_num).cuda()  # cost per node = 1.0

        # LOOP OVER ALL NODES TO ESTIMATE INFLUENCE
      for node in range(node_num):
        # Initialize perturbation vector for THIS node: shape [T]
        x_init = torch.zeros(seq_len).cuda()
        x_init = Variable(x_init, requires_grad=True)

        # SPSA optimization for THIS node
        for _ in range(iter_spsa):

            # --- SPSA random direction ---
            delta = torch.sign(torch.randn_like(x_init))

            # Generate ± perturbations
            x_plus  = x_init + epsilon * delta
            x_minus = x_init - epsilon * delta

            # Clamp range
            x_plus  = torch.clamp(x_plus,  -epsilon, epsilon)
            x_minus = torch.clamp(x_minus, -epsilon, epsilon)

            # INPUT CONSTRUCTION FOR ± perturbation
            Xp_plus  = X.clone()
            Xp_minus = X.clone()

            # Apply perturbations ONLY to node "node"
            Xp_plus[:, node, :, :]  += x_plus.view(1, 1, seq_len)
            Xp_minus[:, node, :, :] += x_minus.view(1, 1, seq_len)

            # Compute model outputs
            yp_plus  = model(Xp_plus,  A_wave, edges, edge_weights)
            yp_minus = model(Xp_minus, A_wave, edges, edge_weights)

            # SPSA gradient estimate (scalar loss difference × direction)
            loss_plus  = nn.MSELoss()(yp_plus, y)
            loss_minus = nn.MSELoss()(yp_minus, y)
            g_spsa = (loss_plus - loss_minus) / (2 * epsilon) * delta

            # Gradient step
            x_init = x_init + step_size * g_spsa
            x_init = torch.clamp(x_init, -epsilon, epsilon)
            x_init = Variable(x_init, requires_grad=True)

        # AFTER SPSA: Evaluate final adversarial effect for node "node"

        Xp_final = X.clone()
        Xp_final[:, node, :, :] += x_init.view(1, 1, seq_len)

        yp_final = model(Xp_final, A_wave, edges, edge_weights)
        yp_clean = model(X, A_wave, edges, edge_weights).detach()

        #print("shape of yp_final and clearn", yp_final.shape, yp_clean.shape)

        # yp_final, yp_clean: [B, N, T]
        diff = yp_final - yp_clean                    # [64, 325, 12]

        # w: [12]
        weighted = diff * w.view(1, 1, -1)            # broadcast over time dimension

        influence = weighted.sum() / b[node]
        score_vector[node] = influence


        #print(f"[KG-SPSA] Node {node} influence score = {score_vector[node].item():.4f}")

      # --- SELECT TOP-K NODES WITHIN BUDGET USING SCORES ---
      index = attack_set_by_kgspsa(score_vector, b, B)


    elif find_type == 'shannon_entropy':
        x_f = X.data
        X_TN = _to_new_TN(x_f)  # (T, N)
        #print(X_TN.shape)
        X_TN = X_TN.transpose(1, 0)
        #T, N = X_TN.shape
        bins_entropy = 32
        score  = _shannon_entropy(X_TN, bins=bins_entropy)
        index = np.argsort(score)[-K:][::-1]
        #index = np.argsort(score)[:K]
        index = index.tolist()

    elif find_type == 'peak_rate':
        x_f = X.data
        X_TN = _to_new_TN(x_f)  # (T, N)
        z_for_peaks = 2.0
        score = _peak_rate(X_TN, z=z_for_peaks)
        index = np.argsort(score)[-K:][::-1]
        #index = np.argsort(score)[:K]
        index = index.tolist()

    elif find_type == 'highfreq':
        x_f = X.data
        X_TN = _to_new_TN(x_f)  # (T, N)
        frac_cut_highfreq=0.25
        score = _highfreq_power_ratio(X_TN, frac_cut=frac_cut_highfreq)
        index = np.argsort(score)[-K:][::-1]
        #index = np.argsort(score)[:K]
        index = index.tolist()

    elif find_type == 'auto_corr':
        x_f = X.data
        X_TN = _to_new_TN(x_f)  # (T, N)
        score = _lag1_autocorr(X_TN)
        index = np.argsort(score)[-K:][::-1]
        #index = np.argsort(score)[:K]
        index = index.tolist()


    else:
        raise  NameError

    X = X.permute(0,3,1,2) # for freq

    #time_idx, time_score = select_timestamp(model, A_wave, edges, edge_weights, X, y)
    #time_idx, time_score = select_timestamp_local_energy(X, k=3)
    #time_idx, time_score = select_timestamp_edge_detector(X, k=3)
    #time_idx, time_score = select_timestamp_random(X, k=3)
    time_idx, time_score = select_timestamp_frequency(X, k=3, high_freq_ratio=0.3)
    #time_idx, time_score = select_timestamp_edge_bias(X, k=3)
    #time_idx, time_score = select_timestamp_temporal_centrality(X, A_wave, k=3)
    #print("time_idx ", time_idx.shape)
    
    # # pgd attack
    #print("shape of x", X.shape)

    X = X.permute(0,2,3,1) # for freq
    #X = X.permute(0, 3, 1, 2)

    #print("shape of x", X.shape)

    B, T, N, F = X.shape
    mask = torch. zeros_like(X)   
    
    for n in index:                     # n is a node id (scalar)
        #print(time_idx[n,:])
        #t_sel = time_idx[n,:]                # shape (K,) -> timesteps for this node
        t_sel = time_idx[:,n]        # for freq
        # set mask=1 for all batches, all features, selected times for this node
        mask[:, n, :, t_sel] = 1.0
        #mask[:, t_sel, n, :] = 1.0

    num_ones = mask.sum().item()
    num_zeros = mask.numel() - num_ones

    #print("Number of zeros and ones",num_zeros, num_ones)
    
    #chosen_attack_nodes = torch.zeros_like(X)
    #chosen_attack_nodes[:, index , : , :] = ones_mat

    #print("shape of x_pgd and mask", X_pgd.shape, mask.shape )
    

    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).cuda()
        X_pgd = Variable(X_pgd.data + mask * random_noise, requires_grad=True)


    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward()
        eta = step_size * mask * X_pgd.grad.data.sign()

        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    #eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon) * mask
    #X_pgd = Variable(X.data + eta, requires_grad=True)
    #X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)



    #SPSA based attack
    # X_adv = X.clone().detach()
    # spsa_delta = 0.01
    # clamp_min = 0.0
    # clamp_max = 1.0
  
    # # Optional random init inside L_inf ball (like your PGD code)
    # if Random:
    #     random_noise = torch.empty_like(X_adv).uniform_(-epsilon, epsilon)
    #     X_adv = X_adv + mask * random_noise
    #     # Project back to epsilon-ball and valid range
    #     delta = torch.clamp(X_adv - X, -epsilon, epsilon)
    #     X_adv = torch.clamp(X + delta, 0, 1.0).detach()

    # for _ in range(num_steps):
    #     # Sample Rademacher vector Δ ∈ {+1, -1}^d ----
    #     perturb = torch.randint_like(X_adv, low=0, high=2, dtype=torch.float32).cuda()
    #     perturb = (perturb * 2.0 - 1.0)  # 0/1 -> -1/+1
    #     perturb = perturb * mask         # respect mask (no change where mask == 0)
    #     X_plus  = X_adv + 0.01 * perturb
    #     X_minus = X_adv - 0.01 * perturb
    #     with torch.no_grad():
    #         pred_plus  = model(X_plus,  A_wave, edges, edge_weights)
    #         pred_minus = model(X_minus, A_wave, edges, edge_weights)

    #         loss_plus  = nn.MSELoss()(pred_plus,  y)
    #         loss_minus = nn.MSELoss()(pred_minus, y)

    #     # SPSA gradient estimate ----
    #     grad_hat = (loss_plus - loss_minus) / (2.0 * spsa_delta) * perturb
    #     # Update step (PGD-style, L_inf) ----
    #     X_adv = X_adv + step_size * torch.sign(grad_hat) * mask
    #     # Projection to epsilon-ball around X & clamp to valid range ----
    #     delta = torch.clamp(X_adv - X, -epsilon, epsilon)
    #     X_adv = X + delta
    #     X_adv = torch.clamp(X_adv, clamp_min, clamp_max).detach()

    # delta, X_zo_sgd = zo_sgd_attack(
    # model,
    # X,
    # y,
    # A_wave,
    # edges,
    # edge_weights,
    # epsilon=epsilon,
    # step_size=step_size,
    # num_steps=num_steps,
    # mu=0.01,
    # q=4,
    # mask=mask,
    # Random=True,
    # )

    # delta, X_zo_adam = zo_adam_attack(
    # model,
    # X,
    # y,
    # A_wave,
    # edges,
    # edge_weights,
    # epsilon=epsilon,
    # step_size=step_size,
    # num_steps=num_steps,
    # mu=0.01,
    # q=4,
    # mask=mask,
    # Random=True,
    # )

    # delta, X_ea = evolutionary_attack(
    # model,
    # X,
    # y,
    # A_wave,
    # edges,
    # edge_weights,
    # epsilon=epsilon,
    # clamp_min=0.0,
    # clamp_max=1.0,
    # mask=mask,
    # Random=True,
    # pop_size=30,
    # num_generations=40,
    # elite_frac=0.2,
    # mutation_std=0.05,
    # crossover_prob=0.5,
    # )
    #plot_map(X, X_pgd, eta, mask)
    #plot_attack_defense(X, y, X_pgd)
    return eta, X_pgd, index

def _pgd_semibox(model,
                    X,
                    A_wave,
                    edges,
                    edge_weights,
                    K,
                    epsilon,
                    num_steps,
                    Random,
                    step_size,
                    find_type,
                    output_len,
                    transform_ground_truth='no-linear', **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]
    if transform_ground_truth == 'linear':
        if num_features > 1:
            y = Variable(X.data[:, :,0,:], requires_grad=True)
        else:
            y = Variable(X.data, requires_grad=True).squeeze(2)

        if output_len < steps_length:

            y = Variable(y.data[:,:,:output_len], requires_grad=True)
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
        else:
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
    elif transform_ground_truth == 'no-linear':
        y_bar = model(X_pgd,A_wave, edges, edge_weights)
        y = y_bar + torch.FloatTensor(*y_bar.shape).uniform_(-epsilon / 10, epsilon / 10 ).cuda()
    else:
        raise  NameError


    y = Variable(torch.clamp(y, 0, 1.0), requires_grad=True)


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 1
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, index, :, :] = ones_mat
    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).cuda()
        X_pgd = Variable(X_pgd.data + chosen_attack_nodes * random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd, A_wave, edges, edge_weights), y)
        loss.backward()
        eta = step_size  * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon) * chosen_attack_nodes
    X_pgd = Variable(X.data + eta, requires_grad=True)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    return eta, X_pgd, index




def _uniformnoise_whitebox(model,
                  X,
                  y,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  Random,
                  step_size,
                  find_type,
                  **kwargs):

    X_randnoise = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_randnoise.shape).uniform_(-epsilon, epsilon).cuda()



    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, index, :, :] = ones_mat


    eta =  chosen_attack_nodes * random_noise
    X_randnoise = Variable(X_randnoise.data + eta, requires_grad=True)
    eta = torch.clamp(X_randnoise.data - X.data, -epsilon, epsilon)
    X_randnoise = Variable(X.data + eta, requires_grad=True)
    X_randnoise = Variable(torch.clamp(X_randnoise, 0, 1.0), requires_grad=True)




    return eta, X_randnoise, index

def _normalnoise_whitebox(model,
                  X,
                  y,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  Random,
                  step_size,
                  find_type,
                  **kwargs):

    X_randnoise = Variable(X.data, requires_grad=True)
    random_noise = epsilon * torch.normal(mean = 0, std= 1, size=X_randnoise.shape).cuda()
    #random_noise = torch.FloatTensor(*X_randnoise.shape).uniform_(-epsilon, epsilon).cuda()


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, index, :, :] = ones_mat


    eta =  chosen_attack_nodes * random_noise
    X_randnoise = Variable(X_randnoise.data + eta, requires_grad=True)
    eta = torch.clamp(X_randnoise.data - X.data, -epsilon, epsilon)
    X_randnoise = Variable(X.data + eta, requires_grad=True)
    X_randnoise = Variable(torch.clamp(X_randnoise, 0, 1.0), requires_grad=True)



    # print('err pgd (white-box): ', err_pgd)
    return eta, X_randnoise, index



def _ST_mim_whitebox(model,
                  X,
                  y,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  num_steps,
                  Random,
                  step_size,
                  find_type,
                  decay_factor=1.0,
                  **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)

    #print("I am x shape in ST_mim", X.shape)


    # first choose the nodes: randomly choose K nodes from total nodes
    #batch_size_x, num_nodes, num_features, steps_length = X.size()
    batch_size_x, steps_length, num_nodes, num_features = X.size() 
    ones_mat = torch.ones(1, steps_length, 1, num_features).cuda()  # [1,1 ,num of channel, number of time length]


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():

                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, :, index, :] = ones_mat
    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).cuda()
        X_pgd = Variable(X_pgd.data + chosen_attack_nodes * random_noise, requires_grad=True)


    previous_grad = torch.zeros_like(X.data)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward()
        grad = X_pgd.grad.data / torch.mean(torch.abs(X_pgd.grad.data), [1,2,3], keepdim=True)
        previous_grad = decay_factor * previous_grad + grad
        X_pgd = Variable(X_pgd.data + step_size * chosen_attack_nodes * previous_grad.sign(), requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)



    return X, X_pgd, index


def _mim_whitebox(model,
                  X,
                  y,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  num_steps,
                  Random,
                  step_size,
                  find_type,
                  decay_factor=1.0,
                  **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, steps_length, num_nodes, num_features = X.size() 
    ones_mat = torch.ones(1, steps_length, 1, num_features).cuda() # [1,1 ,num of channel, number of time length]


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward()

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, :, index, :] = ones_mat
    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).cuda()
        X_pgd = Variable(X_pgd.data + chosen_attack_nodes * random_noise, requires_grad=True)




    previous_grad = torch.zeros_like(X.data)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward()
        grad = X_pgd.grad.data / torch.mean(torch.abs(X_pgd.grad.data), [1,2,3], keepdim=True)
        previous_grad = decay_factor * previous_grad + grad
        X_pgd = Variable(X_pgd.data + step_size  * previous_grad.sign(), requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon) * chosen_attack_nodes
    X_pgd = Variable(X.data + eta, requires_grad=True)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)



    return X, X_pgd, index






def _ST_mim_semibox(model,
                  X,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  num_steps,
                  Random,
                  step_size,
                  find_type,
                  decay_factor=1.0,
                  output_len = 12,
                  transform_ground_truth='no-linear',
                  **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]
    if transform_ground_truth == 'linear':
        if num_features > 1:
            y = Variable(X.data[:, :,0,:], requires_grad=True)
        else:
            y = Variable(X.data, requires_grad=True).squeeze(2)

        if output_len < steps_length:

            y = Variable(y.data[:,:,:output_len], requires_grad=True)
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
        else:
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
    elif transform_ground_truth == 'no-linear':
        y_bar = model(X_pgd,A_wave, edges, edge_weights)
        y = y_bar + torch.FloatTensor(*y_bar.shape).uniform_(-epsilon/ 10 , epsilon/ 10 ).cuda()
    else:
        raise  NameError


    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'shannon_entropy':
        x_f = X.data
        X_TN = _to_new_TN(x_f)  # (T, N)
        #print(X_TN.shape)
        X_TN = X_TN.transpose(1, 0)
        #T, N = X_TN.shape
        bins_entropy = 32
        score  = _shannon_entropy(X_TN, bins=bins_entropy)
        index = np.argsort(score)[-K:][::-1]
        #index = np.argsort(score)[:K]
        index = index.tolist()


    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward(retain_graph=True)

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    #chosen_attack_nodes = torch.zeros_like(X)
    #chosen_attack_nodes[:, index, :, :] = ones_mat

    X = X.permute(0,3,1,2) # for freq

    #time_idx, time_score = select_timestamp(model, A_wave, edges, edge_weights, X, y)
    #time_idx, time_score = select_timestamp_local_energy(X, k=3)
    #time_idx, time_score = select_timestamp_edge_detector(X, k=3)
    #time_idx, time_score = select_timestamp_random(X, k=3)
    time_idx, time_score = select_timestamp_frequency(X, k=3, high_freq_ratio=0.3)
    #time_idx, time_score = select_timestamp_edge_bias(X, k=3)
    #time_idx, time_score = select_timestamp_temporal_centrality(X, A_wave, k=3)

    #print("time_idx ", time_idx.shape)
    
    # # pgd attack
    #print("shape of x", X.shape)

    X = X.permute(0,2,3,1) # for freq
    #X = X.permute(0, 3, 1, 2)

    #print("shape of x", X.shape)

    B, T, N, F = X.shape
    mask = torch. zeros_like(X)   
    
    for n in index:                     # n is a node id (scalar)
        #print(time_idx[n,:])
        #t_sel = time_idx[n,:]                # shape (K,) -> timesteps for this node
        t_sel = time_idx[:,n]        # for freq
        # set mask=1 for all batches, all features, selected times for this node
        mask[:, n, :, t_sel] = 1.0
        #mask[:, t_sel, n, :] = 1.0

    num_ones = mask.sum().item()
    num_zeros = mask.numel() - num_ones

    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).cuda()
        X_pgd = Variable(X_pgd.data + mask * random_noise, requires_grad=True)

    previous_grad = torch.zeros_like(X.data)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward(retain_graph=True)
        grad = X_pgd.grad.data / torch.mean(torch.abs(X_pgd.grad.data), [1,2,3], keepdim=True)
        previous_grad = decay_factor * previous_grad + grad
        X_pgd = Variable(X_pgd.data + step_size * mask * previous_grad.sign(), requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    return eta, X_pgd, index


def _mim_semibox(model,
                  X,
                  A_wave,
                  edges,
                  edge_weights,
                  K,
                  epsilon,
                  num_steps,
                  Random,
                  step_size,
                  find_type,
                  decay_factor=1.0,
                  output_len = 12,
                  transform_ground_truth='no-linear',
                  **kwargs):

    X_pgd = Variable(X.data, requires_grad=True)


    # first choose the nodes: randomly choose K nodes from total nodes
    batch_size_x, num_nodes, num_features, steps_length = X.size()
    ones_mat = torch.ones(1, 1, num_features, steps_length).cuda()  # [1,1 ,num of channel, number of time length]
    if transform_ground_truth == 'linear':
        if num_features > 1:
            y = Variable(X.data[:, :,0,:], requires_grad=True)
        else:
            y = Variable(X.data, requires_grad=True).squeeze(2)

        if output_len < steps_length:

            y = Variable(y.data[:,:,:output_len], requires_grad=True)
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
        else:
            y = y + torch.FloatTensor(*y.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
    elif transform_ground_truth == 'no-linear':
        y_bar = model(X_pgd,A_wave, edges, edge_weights)
        y = y_bar + torch.FloatTensor(*y_bar.shape).uniform_(-epsilon/ 10 , epsilon/ 10 ).cuda()
    else:
        raise  NameError






    if find_type == 'random':
        list = [i for i in range(num_nodes)]
        index = random.sample(list, K)
    elif find_type == 'pagerank':
        index = attack_set_by_pagerank(A_wave.cpu().detach().numpy(), K)
    elif find_type == 'betweeness':
        index = attack_set_by_betweenness(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'degree':
        index = attack_set_by_degree(A_wave.cpu().detach().numpy(), K)

    elif find_type == 'saliency':
        X_saliency = Variable(X.data, requires_grad=True)
        saliency_steps = 5
        for _ in range(saliency_steps):

            if Random:
                random_noise = torch.FloatTensor(*X_saliency.shape).uniform_(-epsilon / 10, epsilon / 10).cuda()
                X_saliency = Variable(X_saliency.data + random_noise, requires_grad=True)

            opt_saliency = optim.SGD([X_saliency], lr=1e-3)
            opt_saliency.zero_grad()
            with torch.enable_grad():
                loss_saliency = nn.MSELoss()(model(X_saliency, A_wave, edges, edge_weights), y)
            loss_saliency.backward(retain_graph=True)

            eta = step_size * X_saliency.grad.data.sign()
            inputs_grad = X_saliency.grad.data

            X_saliency = Variable(X_saliency.data + eta, requires_grad=True)
            eta = torch.clamp(X_saliency.data - X.data, -epsilon, epsilon)
            X_saliency = Variable(X.data + eta, requires_grad=True)
            X_saliency = Variable(torch.clamp(X_saliency, 0, 1.0), requires_grad=True)

        index = attack_set_by_saliency_map(inputs_grad, K)
    else:
        raise  NameError

    chosen_attack_nodes = torch.zeros_like(X)
    chosen_attack_nodes[:, index, :, :] = ones_mat
    if Random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon/10, epsilon/10).cuda()
        X_pgd = Variable(X_pgd.data + chosen_attack_nodes * random_noise, requires_grad=True)




    previous_grad = torch.zeros_like(X.data)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.MSELoss()(model(X_pgd,A_wave, edges, edge_weights), y)
        loss.backward(retain_graph=True)
        grad = X_pgd.grad.data / torch.mean(torch.abs(X_pgd.grad.data), [1,2,3], keepdim=True)
        previous_grad = decay_factor * previous_grad + grad
        X_pgd = Variable(X_pgd.data + step_size * previous_grad.sign(), requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)


    eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon) * chosen_attack_nodes
    X_pgd = Variable(X.data + eta, requires_grad=True)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    # print('err pgd (white-box): ', err_pgd)
    return eta, X_pgd, index









