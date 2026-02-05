from __future__ import print_function
import argparse
import yaml
import os
import shutil
import torch
from mmengine.config import Config
from mmengine.utils import mkdir_or_exist
import time
import pywt
from torch.autograd import Variable
from utils.env import get_root_logger, set_default_configs, init_dist, logger_info, set_random_seed


from datasets.datasets import  SemiMETRLA, SemiPeMS
from attacks.other_attacks import  _ST_fgsm_semibox,  _ST_pgd_semibox, \
     _ST_mim_semibox, _pgd_semibox, _mim_semibox
from torch.utils.data import DataLoader

from models.GraphWaveNet import gwnet #stable
from models.STAWnet import  stawnet #stable
from models.AGCRN import agcrn
from models.AGCnet import STWN

import  numpy as np
import matplotlib.pyplot as plt
from  datasets.datasets import  DataLoaderX
from utils.data_utils import  mae, All_Metrics, All_Local_Metrics, load_la_locations, per_node_metrics, k_hop_mask
from utils.statistics_tools import  log_test_results

parser = argparse.ArgumentParser(description='PyTorch ST PGD Attack Evaluation')
parser.add_argument('config',
                    default='./configs/METRLA-train0.6-val0.2-test0.2-pgd0.1nodes-stgcn.yaml',
                    help='path to config file')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu', default=0, type=int,
                    help='which gpu to use')
parser.add_argument('--seed', type=int, default= 24, metavar='S',
                    help='random seed (default: 24)')
parser.add_argument('--rename', '-r', action='store_true', default=False,
                    help='whether allow renaing the checkpoints parameter to match')
parser.add_argument('--from_file', '-f', action='store_true', default=False,
                    help='analysis data from file')
parser.add_argument('--eval_train_data', action='store_true', default=False,
                    help='whether eval train data')
parser.add_argument('--save_features', '-s', action='store_true', default=True,
                    help='whether save features')
parser.add_argument('--individual', action='store_true', default=False,
                    help='whether to perform individual aa')
parser.add_argument('--attacker', '-a', default='ALL', # ['ALL', 'PGD']
                    help='which attack to perform')
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
    default='none', help='job launcher')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--device_id',  '-d', default= 2, type=int,# ['TRAIN', 'TEST']
                    help='device ID')
args = parser.parse_args()
set_random_seed(args.seed)

# settings
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
if  torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


# set configs
with open(args.config) as cf:
    cfgs = Config(yaml.safe_load(cf))
mkdir_or_exist(cfgs.model_dir)
shutil.copyfile(args.config, os.path.join(cfgs.model_dir, "config_test.yaml"))
set_default_configs(cfgs)
# torch.backends.cudnn.benchmark = True
# torch.cuda.set_device(args.device_id)

# setup logger
logger = get_root_logger(cfgs.log_level, cfgs.model_dir)
logger.info("Loading config file from {}".format(args.config))
logger.info("Work_dir: {}".format(cfgs.model_dir))

# init distributed env first, since logger depends on the dist info.
if args.launcher == 'none':
    distributed = False
else:
    distributed = True
    init_dist(args.launche)

print("Testing on",device)

if cfgs.dataset == 'SemiMETRLA':


    test_data = SemiMETRLA(mode='test',
                       split_train=cfgs.split_train,
                       split_val=cfgs.split_val,
                       num_timesteps=cfgs.num_timesteps)
    adj = test_data.A.numpy()
    #locations = load_la_locations()

elif cfgs.dataset == 'SemiPeMS':

    test_data = SemiPeMS(mode='test',
                       split_train=cfgs.split_train,
                       split_val=cfgs.split_val,
                       num_timesteps=cfgs.num_timesteps)
    adj = test_data.A.numpy()

else:
    raise  NameError


def plot_time(val_target, val_predict, adv_val_predict):
    """
    Plots the clean target, clean prediction, and adversarial prediction
    for the same time window.

    Parameters:
        val_target:      1D array-like — ground truth time series
        val_predict:     1D array-like — clean model prediction
        adv_val_predict: 1D array-like — adversarial model prediction
    """
    #print("shape of arrays", val_target.shape, val_predict.shape, adv_val_predict.shape)
    # Convert to numpy arrays
    val_target      = np.array(val_target).flatten()[512:1024]
    val_predict     = np.array(val_predict).flatten()[512:1024]
    adv_val_predict = np.array(adv_val_predict).flatten()[512:1024]

    T = len(val_target)
    x = np.arange(T)

    plt.figure(figsize=(6, 2))
    plt.plot(x, val_target,   label="Ground Truth")
    plt.plot(x, val_predict,  label="Prediction")
    plt.plot(x, adv_val_predict,  label="Adversarial Prediction")

    plt.xlabel("Time Slot")
    plt.ylabel("Traffic Speed")
    #plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/time3", bbox_inches='tight')


def log_test_csv(val_predict , val_target , adv_val_predict, cfgs, file_name, method):
    metric_list = []
    data_set = cfgs.dataset
    metric_list.append(data_set)
    model_name = cfgs.backbone
    metric_list.append(model_name)


    method_name = method
    metric_list.append(method_name)

    clean_MAE, clean_RMSE, clean_MAPE = All_Metrics(val_predict, val_target)
    #metric_list.append(clean_MAPE)
    adv_MAE, adv_RMSE, adv_MAPE = All_Metrics(adv_val_predict, val_target)
    #metric_list.append(adv_MAPE)
    local_adv_MAE, local_adv_RMSE = All_Local_Metrics(adv_val_predict, val_predict)



    metric_list.append(clean_MAE)
    metric_list.append(adv_MAE)
    metric_list.append(local_adv_MAE)

    metric_list.append(clean_RMSE)
    metric_list.append(adv_RMSE)
    metric_list.append(local_adv_RMSE)

    log_test_results(cfgs.model_dir, metric_list, file_name)


def log_test_black_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, method):
    metric_list = []
    data_set = cfgs.dataset
    metric_list.append(data_set)
    model_name = cfgs.backbone
    metric_list.append(model_name)

    target_model_name = cfgs.target_model
    metric_list.append(target_model_name)

    method_name = method
    metric_list.append(method_name)

    clean_MAE, clean_RMSE, clean_MAPE = All_Metrics(val_predict, val_target)
    # metric_list.append(clean_MAPE)
    adv_MAE, adv_RMSE, adv_MAPE = All_Metrics(adv_val_predict, val_target)
    # metric_list.append(adv_MAPE)
    local_adv_MAE, local_adv_RMSE = All_Local_Metrics(adv_val_predict, val_predict)

    metric_list.append(clean_MAE)
    metric_list.append(adv_MAE)
    metric_list.append(local_adv_MAE)

    metric_list.append(clean_RMSE)
    metric_list.append(adv_RMSE)
    metric_list.append(local_adv_RMSE)

    log_test_results(cfgs.model_dir, metric_list, file_name)

def compute_MVCS_MKCS(y_true, y_pred, y_adv, max_speed, victims, adj, k=1, metric='mae',
                      dims_to_reduce=(0,1), directed=False, exclude_victims_in_MKCS=True):
    """
    y_true, y_pred: tensors (e.g., [B, T, N]) — set dims_to_reduce accordingly.
    victims: list/1D LongTensor of victim node indices
    adj: [N, N] adjacency (float or int). Will be binarized internally.
    k: hop distance for MKCS
    metric: 'mae' or 'rmse'
    dims_to_reduce: which dims to average over to yield per-node errors
    directed: treat adjacency as directed or not
    exclude_victims_in_MKCS: standard definition excludes victims from neighbor set
    Returns: MVCS (scalar), MKCS (scalar), plus useful diagnostics
    """
    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)
    y_adv = np.vstack(y_adv)
    
    y_pred = y_pred * max_speed
    y_true = y_true * max_speed
    y_adv = y_adv * max_speed
  
    per_node_err = per_node_metrics(y_pred, y_true)  # [N]
    #victims = torch.as_tensor(victims, dtype=torch.long)
    per_node_adv_err = per_node_metrics(y_adv, y_true)

    MVCS_clean = per_node_err[victims].mean()
    MVCS_adv = per_node_adv_err[victims].mean()

    MVCS = (MVCS_adv - MVCS_clean)/ (MVCS_adv + MVCS_clean)

    neigh_mask = k_hop_mask(adj, victims, k=k, include_self=not exclude_victims_in_MKCS, directed=directed)
    # Guard against empty neighborhood (can happen if k too small)
    neigh_mask = neigh_mask.detach().cpu()
    if neigh_mask.any():
        MKCS_clean = per_node_err[neigh_mask].mean()
        MKCS_adv = per_node_adv_err[neigh_mask].mean()
    else:
        MKCS = torch.tensor(float('nan'))
    
    MKCS = (MKCS_adv - MKCS_clean)/ (MKCS_adv + MKCS_clean) 

    return MVCS.item(), MKCS.item()

def compute_l2(val_predict, adv_val_predict):
    val_predict_torch = torch.tensor(val_predict, dtype=torch.float32)
    adv_val_predict_torch = torch.tensor(adv_val_predict, dtype=torch.float32)
    delta = adv_val_predict_torch - val_predict_torch      # (B, N, T)
    B, N, T = delta.shape
    delta_flat = delta.reshape(B, -1)             # (B, N*T)
    l2_per_sample = torch.norm(delta_flat, p=2, dim=1)   # (B,)
    l2_mean = l2_per_sample.mean().item()
    return l2_mean

def compute_dwt(clean, adv, wavelet='haar', level=None):
    """
    clean, adv: (B, C, N, T)
    Returns L2 of wavelet-coeff difference per sample.
    """
    #assert clean.shape == adv.shape

    # c = np.array(clean)
    # a = np.array(adv)   

    c = torch.tensor(clean, dtype=torch.float32)
    a = torch.tensor(adv, dtype=torch.float32)

    
    #c_np = c.permute(1, 0, 2, 3) 
    #a_np   = a.permute(1, 0, 2, 3)
    
    # ----- 1. take batch mean -----
    clean_np = c.mean(dim=0)   # (C, N, T)
    adv_np   = a.mean(dim=0)     # (C, N, T)
    
    #print("shape of clean", clean_mean.shape)
    N, T = clean_np.shape 

    # ----- 2. compute DWT L2 per node -----
    dwt_l2_per_node = np.zeros(N, dtype=np.float32)

    for n in range(N):
        # take mean across channels -> (T,)
        x_c = clean_np[n, :]
        x_a = adv_np[n, :]

        c_c = pywt.wavedec(x_c, wavelet=wavelet, level=level)
        c_a = pywt.wavedec(x_a, wavelet=wavelet, level=level)

        c_c_flat = np.concatenate([c.ravel() for c in c_c])
        c_a_flat = np.concatenate([c.ravel() for c in c_a])

        diff = c_a_flat - c_c_flat
        dwt_l2_per_node[n] = np.linalg.norm(diff, ord=2)

    # ----- 3. return both per-node and global mean -----
    #dwt_l2_per_node_t = torch.from_numpy(dwt_l2_per_node)   # (N,)
    dwt_l2_total = dwt_l2_per_node.mean()                 # scalar

    return dwt_l2_total


def batch_eval(val_target, val_predict, adv_val_predict, max_speed):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average mae.
    """
    val_predict = np.vstack(val_predict)
    val_target = np.vstack(val_target)
    adv_val_predict = np.vstack(adv_val_predict)

    val_predict = val_predict * max_speed
    adv_val_predict = adv_val_predict * max_speed
    val_target = val_target * max_speed

    clean_MAE, clean_RMSE, clean_MAPE  = All_Metrics(val_predict, val_target)
    adv_MAE, adv_RMSE, adv_MAPE = All_Metrics(adv_val_predict, val_target)

    return clean_MAE, clean_RMSE, clean_MAPE, adv_MAE, adv_RMSE, adv_MAPE

def batch_eval_local(val_predict, adv_val_predict, max_speed):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average mae.
    """
    val_predict = np.vstack(val_predict)
    adv_val_predict = np.vstack(adv_val_predict)
    val_predict = val_predict * max_speed
    adv_val_predict = adv_val_predict * max_speed



    local_adv_MAE, local_adv_RMSE = All_Local_Metrics(adv_val_predict, val_predict)

    return local_adv_MAE, local_adv_RMSE



def eval_black_attack(cfgs, val_loader, source_model, target_model,A_wave, edges, edge_weights, attacker, max_speed,  find_type = 'random'):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    # with torch.no_grad():
    start = time.time()
    source_model.eval()
    target_model.eval()
    val_predict = []
    val_target = []
    adv_val_predict = []
    samples_total = len(val_loader) * cfgs.test_batch_size
    attack_nodes = int(cfgs.test_attack_nodes * len(A_wave))
    for batch_idx, (past_data, current_data, future_target) in enumerate(val_loader):
        X_batch, y_batch = past_data, current_data

        data_past_batch = X_batch.to(device=args.device)
        data_current_batch = y_batch.to(device=args.device)
        data_future_batch = future_target.to(device=args.device)



        # pre-trained model is net, generate current data based on past data
        past_out = source_model(data_past_batch.unsqueeze(2), A_wave, edges, edge_weights)
        # val predict
        current_out = target_model(data_current_batch.unsqueeze(2), A_wave, edges, edge_weights)
        if attacker == 'ST_FGSM_semi':
            eta, _, index = _ST_fgsm_semibox(source_model,
                  past_out.unsqueeze(2),
                  A_wave,
                  edges,
                  edge_weights,
                  attack_nodes,
                  cfgs.test_epsilon,
                  cfgs.random,
                  find_type,
                  cfgs.num_timesteps,
                  transform_ground_truth='no-linear')
        elif attacker == 'ST_MIM_semi':
            decay_factor = 1
            eta, _, index = _ST_mim_semibox(source_model,
                             past_out.unsqueeze(2),
                             A_wave,
                             edges,
                             edge_weights,
                             attack_nodes,
                             cfgs.test_epsilon,
                             cfgs.test_num_steps,
                             cfgs.random,
                             cfgs.test_step_size,
                             find_type,
                             decay_factor,
                             cfgs.num_timesteps,
                             transform_ground_truth='no-linear')
        elif attacker == 'MIM_semi':
            decay_factor = 1
            eta, _, index = _mim_semibox(source_model,
                             past_out.unsqueeze(2),
                             A_wave,
                             edges,
                             edge_weights,
                             attack_nodes,
                             cfgs.test_epsilon,
                             cfgs.test_num_steps,
                             cfgs.random,
                             cfgs.test_step_size,
                             find_type,
                             decay_factor,
                             cfgs.num_timesteps,
                             transform_ground_truth='no-linear')

        elif attacker == 'ST_PGD_semi':
            eta, _, index = _ST_pgd_semibox(source_model,
                             past_out.unsqueeze(2),
                             A_wave,
                             edges,
                             edge_weights,
                             attack_nodes,
                             cfgs.test_epsilon,
                             cfgs.test_num_steps,
                             cfgs.random,
                             cfgs.test_step_size,
                             find_type,
                             cfgs.num_timesteps,
                             transform_ground_truth='no-linear')
        elif attacker == 'PGD_semi':
            eta, _, index = _pgd_semibox(source_model,
                             past_out.unsqueeze(2),
                             A_wave,
                             edges,
                             edge_weights,
                             attack_nodes,
                             cfgs.test_epsilon,
                             cfgs.test_num_steps,
                             cfgs.random,
                             cfgs.test_step_size,
                             find_type,
                             cfgs.num_timesteps,
                             transform_ground_truth='no-linear')
        else:
            raise  NameError



        X_adv = Variable(torch.clamp(data_current_batch.unsqueeze(2)+eta, 0, 1.0))

        adv_out = target_model(X_adv,A_wave, edges, edge_weights)
        val_predict.append(current_out.cpu().detach().numpy())
        val_target.append(data_future_batch.cpu().detach().numpy())
        adv_val_predict.append(adv_out.cpu().detach().numpy())

        clean_MAE, clean_RMSE, clean_RRSE, adv_MAE, adv_RMSE, adv_RRSE = batch_eval(val_target, val_predict, adv_val_predict, max_speed)
        local_adv_MAE, local_adv_RMSE = batch_eval_local(val_predict, adv_val_predict, max_speed)
        torch.cuda.empty_cache()
        if batch_idx % cfgs.log_interval == 0:
            logger_info(logger, distributed, 'Info:  [{}/{} ({:.0f}%)]\t  MAE: {:.4f} RMSE: {:.4f} RRSE: {:.4f} Glocal: Adv MAE: {:.4f} Adv RMSE: {:.4f} Adv RRSE: {:.4f} Local: Adv MAE: {:.4f} Adv RMSE: {:.4f}  time:{:.3f}'.format(
                       batch_idx * len(current_data), samples_total,
                       100. * batch_idx / len(val_loader),
                       clean_MAE, clean_RMSE, clean_RRSE, adv_MAE, adv_RMSE, adv_RRSE,
                       local_adv_MAE, local_adv_RMSE,
                       time.time() - start))
        break
        #vis_attack_nodes(index, locations, adj, cfgs.dataset, attacker,find_type, cfgs.model_dir, batch_idx)
    val_predict = np.vstack(val_predict)
    val_target = np.vstack(val_target)
    adv_val_predict = np.vstack(adv_val_predict)
    clean_MAE, clean_RMSE, clean_RRSE, adv_MAE, adv_RMSE, adv_RRSE = batch_eval(val_target, val_predict, adv_val_predict, max_speed)
    local_adv_MAE, local_adv_RMSE = batch_eval_local(val_predict, adv_val_predict, max_speed)
    logger_info(logger, distributed,
               'MAE: {:.4f} RMSE: {:.4f} RRSE: {:.4f} Global: Adv MAE: {:.4f} Adv RMSE: {:.4f} Adv RRSE: {:.4f} Local: Adv MAE: {:.4f} Adv RMSE: {:.4f} '.format(
               clean_MAE, clean_RMSE, clean_RRSE, adv_MAE, adv_RMSE, adv_RRSE,
               local_adv_MAE, local_adv_RMSE))

    MVCS, MKCS = compute_MVCS_MKCS(
               val_target, val_predict, adv_val_predict,
               max_speed = max_speed,
               victims=index,
               adj=A_wave,          # torch.FloatTensor [207, 207]
               k=2,                     # 2-hop neighbors
               metric='mae',            # or 'rmse'
               dims_to_reduce=(0,1),    # reduce over batch and time for [B,T,N]
               directed=False,           # set True if you want strictly directed hops  
               )

    print(f"Mean Victim Corruption Score (MVCS): {MVCS:.4f} | Mean k-hop Corruption Score (MKCS): {MKCS:.4f}")

    l2_norm = compute_l2(val_predict, adv_val_predict)
    print(f"l2 norm: {l2_norm:.4f}")

    dwt = compute_dwt(val_predict, adv_val_predict, wavelet='haar', level=None)
    print(f"dwt: {dwt:.4f}")    

    return val_predict * max_speed , val_target * max_speed, adv_val_predict * max_speed



def eval_val(cfgs, val_loader, net, A_wave, edges, edge_weights, attacker, max_speed,  find_type = 'random'):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    # with torch.no_grad():
    start = time.time()
    net.eval()
    val_predict = []
    val_target = []
    adv_val_predict = []
    samples_total = len(val_loader) * cfgs.test_batch_size
    attack_nodes = int(cfgs.test_attack_nodes * len(A_wave))
    for batch_idx, (past_data, current_data, future_target) in enumerate(val_loader):
        data_past_batch = past_data.to(device=args.device)
        data_current_batch = current_data.to(device=args.device)
        data_future_batch = future_target.to(device=args.device)



        # pre-trained model is net, generate current data based on past data
        past_out = net(data_past_batch.unsqueeze(2), A_wave, edges, edge_weights)
        # val predict
        current_out = net(data_current_batch.unsqueeze(2), A_wave, edges, edge_weights)
        if attacker == 'ST_FGSM_semi':
            eta, _, index = _ST_fgsm_semibox(net,
                  past_out.unsqueeze(2),
                  A_wave,
                  edges,
                  edge_weights,
                  attack_nodes,
                  cfgs.test_epsilon,
                  cfgs.random,
                  find_type,
                  cfgs.num_timesteps,
                  transform_ground_truth='no-linear')
        elif attacker == 'ST_MIM_semi':
            decay_factor = 1
            eta, _, index = _ST_mim_semibox(net,
                             past_out.unsqueeze(2),
                             A_wave,
                             edges,
                             edge_weights,
                             attack_nodes,
                             cfgs.test_epsilon,
                             cfgs.test_num_steps,
                             cfgs.random,
                             cfgs.test_step_size,
                             find_type,
                             decay_factor,
                             cfgs.num_timesteps,
                             transform_ground_truth='no-linear')
        elif attacker == 'MIM_semi':
            decay_factor = 1
            eta, _, index = _mim_semibox(net,
                             past_out.unsqueeze(2),
                             A_wave,
                             edges,
                             edge_weights,
                             attack_nodes,
                             cfgs.test_epsilon,
                             cfgs.test_num_steps,
                             cfgs.random,
                             cfgs.test_step_size,
                             find_type,
                             decay_factor,
                             cfgs.num_timesteps,
                             transform_ground_truth='no-linear')

        elif attacker == 'ST_PGD_semi':
            eta, _, index = _ST_pgd_semibox(net,
                             past_out.unsqueeze(2),
                             A_wave,
                             edges,
                             edge_weights,
                             attack_nodes,
                             cfgs.test_epsilon,
                             cfgs.test_num_steps,
                             cfgs.random,
                             cfgs.test_step_size,
                             find_type,
                             cfgs.num_timesteps,
                             transform_ground_truth='no-linear')
        elif attacker == 'PGD_semi':
            eta, _, index = _pgd_semibox(net,
                             past_out.unsqueeze(2),
                             A_wave,
                             edges,
                             edge_weights,
                             attack_nodes,
                             cfgs.test_epsilon,
                             cfgs.test_num_steps,
                             cfgs.random,
                             cfgs.test_step_size,
                             find_type,
                             cfgs.num_timesteps,
                             transform_ground_truth='no-linear')
        else:
            raise  NameError

        X_adv = Variable(torch.clamp(data_current_batch.unsqueeze(2)+eta, 0, 1.0))

        adv_out = net(X_adv,A_wave, edges, edge_weights)
        val_predict.append(current_out.cpu().detach().numpy())
        val_target.append(data_future_batch.cpu().detach().numpy())
        adv_val_predict.append(adv_out.cpu().detach().numpy())

        clean_MAE, clean_RMSE, clean_RRSE, adv_MAE, adv_RMSE, adv_RRSE = batch_eval(val_target, val_predict, adv_val_predict, max_speed)
        local_adv_MAE, local_adv_RMSE = batch_eval_local(val_predict, adv_val_predict, max_speed)
        torch.cuda.empty_cache()
        if batch_idx % cfgs.log_interval == 0:
            logger_info(logger, distributed, 'Info:  [{}/{} ({:.0f}%)]\t  MAE: {:.4f} RMSE: {:.4f} RRSE: {:.4f} Glocal: Adv MAE: {:.4f} Adv RMSE: {:.4f} Adv RRSE: {:.4f} Local: Adv MAE: {:.4f} Adv RMSE: {:.4f}  time:{:.3f}'.format(
                       batch_idx * len(current_data), samples_total,
                       100. * batch_idx / len(val_loader),
                       clean_MAE, clean_RMSE, clean_RRSE, adv_MAE, adv_RMSE, adv_RRSE,
                       local_adv_MAE, local_adv_RMSE,
                       time.time() - start))
        #plot_time(val_target, val_predict, adv_val_predict)
        #break
        #vis_attack_nodes(index, locations, adj, cfgs.dataset, attacker,find_type, cfgs.model_dir, batch_idx)
    val_predict = np.vstack(val_predict)
    val_target = np.vstack(val_target)
    adv_val_predict = np.vstack(adv_val_predict)
    clean_MAE, clean_RMSE, clean_RRSE, adv_MAE, adv_RMSE, adv_RRSE = batch_eval(val_target, val_predict, adv_val_predict, max_speed)
    local_adv_MAE, local_adv_RMSE = batch_eval_local(val_predict, adv_val_predict, max_speed)
    logger_info(logger, distributed,
               'MAE: {:.4f} RMSE: {:.4f} RRSE: {:.4f} Global: Adv MAE: {:.4f} Adv RMSE: {:.4f} Adv RRSE: {:.4f} Local: Adv MAE: {:.4f} Adv RMSE: {:.4f} '.format(
               clean_MAE, clean_RMSE, clean_RRSE, adv_MAE, adv_RMSE, adv_RRSE,
               local_adv_MAE, local_adv_RMSE))

    MVCS, MKCS = compute_MVCS_MKCS(
               val_target, val_predict, adv_val_predict,
               max_speed = max_speed,
               victims=index,
               adj=A_wave,          # torch.FloatTensor [207, 207]
               k=2,                     # 2-hop neighbors
               metric='mae',            # or 'rmse'
               dims_to_reduce=(0,1),    # reduce over batch and time for [B,T,N]
               directed=False,           # set True if you want strictly directed hops  
               )

    print(f"Mean Victim Corruption Score (MVCS): {MVCS:.4f} | Mean k-hop Corruption Score (MKCS): {MKCS:.4f}")

    l2_norm = compute_l2(val_predict, adv_val_predict)
    print(f"l2 norm: {l2_norm:.4f}")

    dwt = compute_dwt(val_predict, adv_val_predict, wavelet='haar', level=None)
    print(f"dwt: {dwt:.4f}")    

    return val_predict * max_speed , val_target * max_speed, adv_val_predict * max_speed

def main():
    # set up data loader
    logger.info("Building test datasets {}".format(cfgs.dataset))


    test_loader = DataLoader(test_data, batch_size=cfgs.test_batch_size, shuffle=False)
    #test_loader = DataLoaderX(test_data, batch_size=cfgs.test_batch_size, shuffle=False, num_workers=6, pin_memory=True)
    A_wave = test_data.A_wave.to(device=args.device)
    edges = test_data.edges.to(device=args.device)
    edge_weights = test_data.edge_weights.to(device=args.device)
    max_speed = test_data.max_speed

    if cfgs.backbone == 'GWNET':
        dropout = 0.3
        supports = None
        gcn_bool = True
        addaptadj = True
        aptinit = None
        nhid = 32
        model = gwnet(device, num_nodes=cfgs.num_nodes, dropout=dropout, supports=supports, gcn_bool=gcn_bool,
                      addaptadj=addaptadj, aptinit=aptinit, in_dim=cfgs.num_features, out_dim=cfgs.num_timesteps,
                      residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,
                      end_channels=nhid * 16).to(device=args.device)

    elif cfgs.backbone == 'STAWNET':
        nhid = 32
        num_nodes = cfgs.num_nodes
        dropout = 0.3
        gat_bool = True
        addaptadj = True
        aptonly = False
        aptinit = None
        in_dim = cfgs.num_features
        out_dim = cfgs.num_timesteps
        emb_length = 16
        noapt = False

        model = stawnet(device, num_nodes, dropout, gat_bool=gat_bool, addaptadj=addaptadj,
                        aptonly=aptonly, aptinit=aptinit, in_dim=in_dim, out_dim=out_dim,
                        residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,
                        end_channels=nhid * 16, emb_length=emb_length, noapt=noapt).to(device=args.device)

    elif cfgs.backbone == 'AGCNET':
        num_nodes = cfgs.num_nodes     
        default_graph = True
        input_dim = cfgs.num_features
        output_dim = cfgs.num_output_features
        horizon = 12
        num_layers = 2
    
        model = STWN(device, num_nodes, A_wave.to(device), input_dim = input_dim, output_dim=output_dim, 
                       horizon = horizon).to(device=args.device)

    else:
        raise NameError

    load_path = cfgs.model_path
    logger.info('Loading checkpoint from %s', load_path)
    model.load_state_dict(torch.load(load_path))
    model.eval()

    print("load source complete")



    metric_list = []
    if args.attacker == 'black-attack':

        if cfgs.target_model== 'GWNET':
            dropout = 0.3
            supports = None
            gcn_bool = True
            addaptadj = True
            aptinit = None
            nhid = 32
            target_model = gwnet(device, num_nodes=cfgs.num_nodes, dropout=dropout, supports=supports, gcn_bool=gcn_bool,
                          addaptadj=addaptadj, aptinit=aptinit, in_dim=cfgs.num_features, out_dim=cfgs.num_timesteps,
                          residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,
                          end_channels=nhid * 16).to(device=args.device)

        elif cfgs.target_model == 'STAWNET':
            nhid = 32
            num_nodes = cfgs.num_nodes
            dropout = 0.3
            gat_bool = True
            addaptadj = True
            aptonly = False
            aptinit = None
            in_dim = cfgs.num_features
            out_dim = cfgs.num_timesteps
            emb_length = 16
            noapt = False

            target_model = stawnet(device, num_nodes, dropout, gat_bool=gat_bool, addaptadj=addaptadj,
                            aptonly=aptonly, aptinit=aptinit, in_dim=in_dim, out_dim=out_dim,
                            residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8,
                            end_channels=nhid * 16, emb_length=emb_length, noapt=noapt).to(device=args.device)
         
        elif cfgs.target_model == 'AGCRN':
            num_nodes = cfgs.num_nodes     
            default_graph = True
            input_dim = cfgs.num_features
            output_dim = cfgs.num_output_features
            embed_dim = 10
            horizon = 12
            num_layers = 2
            rnn_units= 64
            cheb_k = 2

            target_model = agcrn(device, num_nodes,  input_dim=input_dim, rnn_units=rnn_units, output_dim=output_dim, 
                       horizon = horizon, cheb_k = cheb_k, num_layers = num_layers , embed_dim = embed_dim,  
                       default_graph = True).to(device=args.device)

        elif cfgs.target_model == 'AGCNET':
            num_nodes = cfgs.num_nodes     
            default_graph = True
            input_dim = cfgs.num_features
            output_dim = cfgs.num_output_features
            horizon = 12
            num_layers = 2
    
            target_model = STWN(device, num_nodes, A_wave.to(device), input_dim = input_dim, output_dim=output_dim, 
                       horizon = horizon).to(device=args.device)
        
        else:
            raise NameError

        load_path = cfgs.target_model_path
        logger.info('Loading checkpoint from %s', load_path)
        target_model.load_state_dict(torch.load(load_path))
        target_model.eval()



        header = ['dataset', 'source model', 'target model','method',"test batch size",'clean_MAE','adv_MAE', 'local_adv_MAE', 'clean_RMSE',
                   'adv_RMSE',  'local_adv_RMSE']

        file_name = 'black_box-data-{}_num-nodes{}_eps{}-source_model-{}-target_model-{}'.format(cfgs.dataset,cfgs.test_attack_nodes,cfgs.test_epsilon,cfgs.backbone, cfgs.target_model)
        log_test_results(cfgs.model_dir, header, file_name)


        logger.info('STPGD-TNDS black_box attack. -Info: num attack nodes pro: {}, outputs length: {}'.format(cfgs.test_attack_nodes, cfgs.num_timesteps))
        val_predict , val_target , adv_val_predict = eval_black_attack(cfgs, test_loader, model,target_model, A_wave, edges, edge_weights, 'ST_PGD_semi', max_speed, find_type = 'shannon_entropy')
        log_test_black_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'STPGD-TNDS black_box attack, {}'.format(cfgs.test_batch_size))
 
        # logger.info('STPGD-TNDS black_box attack. -Info: num attack nodes pro: {}, outputs length: {}'.format(cfgs.test_attack_nodes, cfgs.num_timesteps))
        # val_predict , val_target , adv_val_predict = eval_black_attack(cfgs, test_loader, model,target_model, A_wave, edges, edge_weights, 'ST_MIM_semi', max_speed, find_type = 'shannon_entropy')
        # log_test_black_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'STPGD-TNDS black_box attack, {}'.format(cfgs.test_batch_size))


        # logger.info('STMIM-TNDS black_box attack. -Info: num attack nodes pro: {}'.format(cfgs.test_attack_nodes))
        # val_predict , val_target , adv_val_predict = eval_black_attack(cfgs, test_loader, model,target_model, A_wave, edges, edge_weights, 'ST_MIM_semi', max_speed, find_type = 'saliency')
        # log_test_black_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'st_mim black_box attack: saliency,  {}'.format(cfgs.test_batch_size))


        # logger.info('PGD-Random black_box attack. -Info: num attack nodes pro: {}, outputs length: {}'.format(
        #     cfgs.test_attack_nodes, cfgs.num_timesteps))
        # val_predict, val_target, adv_val_predict = eval_black_attack(cfgs, test_loader, model, target_model, A_wave,
        #                                                              edges, edge_weights, 'PGD_semi', max_speed,
        #                                                              find_type='random')
        # log_test_black_csv(val_predict, val_target, adv_val_predict, cfgs, file_name,
        #                    'PGD-Random black_box attack,  {}'.format(cfgs.test_batch_size))

        # logger.info('PGD-PR black_box attack. -Info: num attack nodes pro: {}, outputs length: {}'.format(
        #     cfgs.test_attack_nodes, cfgs.num_timesteps))
        # val_predict, val_target, adv_val_predict = eval_black_attack(cfgs, test_loader, model, target_model, A_wave,
        #                                                              edges, edge_weights, 'PGD_semi', max_speed,
        #                                                              find_type='pagerank')
        # log_test_black_csv(val_predict, val_target, adv_val_predict, cfgs, file_name,
        #                    'PGD-PR black_box attack, {}'.format(cfgs.test_batch_size))

        # logger.info('PGD-Centrlity black_box attack. -Info: num attack nodes pro: {}, outputs length: {}'.format(
        #     cfgs.test_attack_nodes, cfgs.num_timesteps))
        # val_predict, val_target, adv_val_predict = eval_black_attack(cfgs, test_loader, model, target_model, A_wave,
        #                                                              edges, edge_weights, 'PGD_semi', max_speed,
        #                                                              find_type='betweeness')
        # log_test_black_csv(val_predict, val_target, adv_val_predict, cfgs, file_name,
        #                    'PGD-Centrlity black_box attack,  {}'.format(cfgs.test_batch_size))

        # logger.info('PGD-Degree black_box attack. -Info: num attack nodes pro: {}, outputs length: {}'.format(
        #     cfgs.test_attack_nodes, cfgs.num_timesteps))
        # val_predict, val_target, adv_val_predict = eval_black_attack(cfgs, test_loader, model, target_model, A_wave,
        #                                                              edges, edge_weights, 'PGD_semi', max_speed,
        #                                                              find_type='degree')
        # log_test_black_csv(val_predict, val_target, adv_val_predict, cfgs, file_name,
        #                    'PGD-Degree black_box attack,{}'.format(cfgs.test_batch_size))


        # logger.info('MIM-Random black_box attack. -Info: num attack nodes pro: {}'.format(cfgs.test_attack_nodes))
        # val_predict , val_target , adv_val_predict = eval_black_attack(cfgs, test_loader, model,target_model, A_wave, edges, edge_weights, 'MIM_semi', max_speed, find_type = 'random')
        # log_test_black_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'MIM-Random black_box attack,  {}'.format(cfgs.test_batch_size))


        # logger.info(
        #     'MIM-PR black_box attack. -Info: num attack nodes pro: {}'.format(cfgs.test_attack_nodes))
        # val_predict, val_target, adv_val_predict = eval_black_attack(cfgs, test_loader, model,target_model, A_wave, edges, edge_weights,
        #                                                     'MIM_semi', max_speed, find_type='pagerank')
        # log_test_black_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'MIM-PR black_box attack, {}'.format(cfgs.test_batch_size))



        # logger.info(
        #     'MIM-Centrality black_box attack. -Info: num attack nodes pro: {}'.format(cfgs.test_attack_nodes))
        # val_predict, val_target, adv_val_predict = eval_black_attack(cfgs, test_loader, model,target_model, A_wave, edges, edge_weights,
        #                                                     'MIM_semi', max_speed, find_type='betweeness')
        # log_test_black_csv(val_predict, val_target, adv_val_predict, cfgs, file_name,
        #              'MIM-Centrality black_box attack.,  {}'.format(cfgs.test_batch_size))


        # logger.info(
        #     'MIM-Degree black_box attack. -Info: num attack nodes pro: {}'.format(cfgs.test_attack_nodes))
        # val_predict, val_target, adv_val_predict = eval_black_attack(cfgs, test_loader, model,target_model, A_wave, edges, edge_weights,
        #                                                     'MIM_semi', max_speed, find_type='degree')
        # log_test_black_csv(val_predict, val_target, adv_val_predict, cfgs, file_name,
        #              'MIM-Degree black_box attack., {}'.format(cfgs.test_batch_size))


        # logger_info(logger, distributed,
        #             '[Remarks] {} | End of testing, saved at {}'.format(cfgs.remark, cfgs.model_dir))


    elif args.attacker == 'grey-attack':
        header = ['dataset', 'model', 'method', 'batch size','clean_MAE','adv_MAE', 'local_adv_MAE', 'clean_RMSE',
                   'adv_RMSE',  'local_adv_RMSE']

        file_name = 'grey_box-data-{}_num-nodes{}_eps{}-model-{}'.format(cfgs.dataset,cfgs.test_attack_nodes,cfgs.test_epsilon,cfgs.backbone)
        log_test_results(cfgs.model_dir, header, file_name)

        logger.info('STPGD-TNDS grey-box attack. -Info: num attack nodes pro: {}, outputs length: {}'.format(cfgs.test_attack_nodes, cfgs.num_timesteps))
        val_predict , val_target , adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights, 'ST_PGD_semi', max_speed, find_type = 'shannon_entropy')
        log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'STPGD-TNDS grey-box attack, {}'.format(cfgs.test_batch_size))

        # logger.info('STMIM-TNDS grey-box attack. -Info: num attack nodes pro: {}, outputs length: {}'.format(cfgs.test_attack_nodes, cfgs.num_timesteps))
        # val_predict , val_target , adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights, 'ST_MIM_semi', max_speed, find_type = 'saliency')
        # log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'STMIM-TNDS grey-box attack, {}'.format(cfgs.test_batch_size))

        # logger.info('PGD-Random grey-box attack. -Info: num attack nodes pro: {}, outputs length: {}'.format(cfgs.test_attack_nodes, cfgs.num_timesteps))
        # val_predict , val_target , adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights, 'PGD_semi', max_speed, find_type = 'random')
        # log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'PGD-Random grey-box attack, {}'.format(cfgs.test_batch_size))

        # logger.info('PGD-PR grey-box attack. -Info: num attack nodes pro: {}, outputs length: {}'.format(cfgs.test_attack_nodes, cfgs.num_timesteps))
        # val_predict , val_target , adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights, 'PGD_semi', max_speed, find_type = 'pagerank')
        # log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'PGD-PR grey-box attack, {}'.format(cfgs.test_batch_size))


        # logger.info('PGD-Centrality grey-box. -Info: num attack nodes pro: {}, outputs length: {}'.format(cfgs.test_attack_nodes, cfgs.num_timesteps))
        # val_predict , val_target , adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights, 'PGD_semi', max_speed, find_type = 'betweeness')
        # log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'PGD-Centrality grey-box,  {}'.format(cfgs.test_batch_size))

        # logger.info('PGD-Degree grey-box. -Info: num attack nodes pro: {}, outputs length: {}'.format(cfgs.test_attack_nodes, cfgs.num_timesteps))
        # val_predict , val_target , adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights, 'PGD_semi', max_speed, find_type = 'degree')
        # log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'PGD-Degree grey-box, {}'.format(cfgs.test_batch_size))


        # logger.info('MIM-Random grey-box attack. -Info: num attack nodes pro: {}'.format(cfgs.test_attack_nodes))
        # val_predict , val_target , adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights, 'MIM_semi', max_speed, find_type = 'random')
        # log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'MIM-Random grey-box attack, {}'.format(cfgs.test_batch_size))


        # logger.info(
        #     'MIM-PR grey-box attack. -Info: num attack nodes pro: {}'.format(cfgs.test_attack_nodes))
        # val_predict, val_target, adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights,
        #                                                     'MIM_semi', max_speed, find_type='pagerank')
        # log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name, 'PGD-PR grey-box attack,  {}'.format(cfgs.test_batch_size))



        # logger.info(
        #     'MIM-Centrality grey-box. -Info: num attack nodes pro: {}'.format(cfgs.test_attack_nodes))
        # val_predict, val_target, adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights,
        #                                                     'MIM_semi', max_speed, find_type='betweeness')
        # log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name,
        #              'MIM-Centrality grey-box, {}'.format(cfgs.test_batch_size))


        # logger.info(
        #     'MIM-Degree grey-box. -Info: num attack nodes pro: {}'.format(cfgs.test_attack_nodes))
        # val_predict, val_target, adv_val_predict = eval_val(cfgs, test_loader, model, A_wave, edges, edge_weights,
        #                                                     'MIM_semi', max_speed, find_type='degree')
        # log_test_csv(val_predict, val_target, adv_val_predict, cfgs, file_name,
        #              'MIM-Degree grey-box,  {}'.format(cfgs.test_batch_size))


        logger_info(logger, distributed,
                    '[Remarks] {} | End of testing, saved at {}'.format(cfgs.remark, cfgs.model_dir))

    else:
        raise NameError




if __name__ == '__main__':
    main()
