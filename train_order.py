import argparse
import os
import random
import torch
import pandas as pd
import numpy as np
import torch.nn.parallel
import torch.utils.data

import pandas as pd
from castle.metrics import MetricsDAG

from cdt.metrics import SID

import dataset
from utils import *
from pathlib import Path
from cdt.utils.R import launch_R_script
import tempfile

torch.set_printoptions(linewidth=1000)
np.set_printoptions(linewidth=1000)
def blue(x): return '\033[94m' + x + '\033[0m'
def red(x): return '\033[31m' + x + '\033[0m'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model', type=str, default='CaPS')
    parser.add_argument('--pre_pruning', type=str, default=True)
    parser.add_argument('--add_edge', type=str, default=True)
    parser.add_argument('--lambda1', type=float, default=50.0)
    parser.add_argument('--lambda2', type=float, default=50.0)

    parser.add_argument('--dataset', type=str, default='sachs')
    parser.add_argument('--norm', type=bool, default=False)
    parser.add_argument('--num_nodes', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--method', type=str, default='mixed')
    parser.add_argument('--linear_sem_type', type=str, default='gauss')
    parser.add_argument('--nonlinear_sem_type', type=str, default='gp')
    parser.add_argument('--linear_rate', type=float, default=1.0)

    parser.add_argument('--manualSeed', type=str, default='False')
    parser.add_argument('--runs', type=int, default=5)

    args = parser.parse_args()
    args.manualSeed = True if args.manualSeed == 'True' else False
    return args

def order_divergence(order, adj):
    err = 0
    for i in range(len(order)):
        err += adj[order[i+1:], order[i]].sum()
    return err

def evaluate(args, dag, GT_DAG):
    print('pred_dag:\n', dag)
    print('gt_dag:\n', GT_DAG.astype(int))
    print(blue('edge_num: ' + str(np.sum(dag))))
    mt = MetricsDAG(dag, GT_DAG)
    sid = SID(GT_DAG, dag)
    mt.metrics['sid'] = sid.item()
    print(blue(str(mt.metrics)))
    return mt.metrics

def cam_pruning(A, X, cutoff, only_pruning=True):
    with tempfile.TemporaryDirectory() as save_path:
        if only_pruning:
            pruning_path = Path(__file__).parent / "pruning_R_files/cam_pruning.R"
        
        data_np = X #np.array(X.detach().cpu().numpy())
        data_csv_path = np_to_csv(data_np, save_path)
        dag_csv_path = np_to_csv(A, save_path)

        arguments = dict()
        arguments['{PATH_DATA}'] = data_csv_path
        arguments['{PATH_DAG}'] = dag_csv_path
        arguments['{PATH_RESULTS}'] = os.path.join(save_path, "results.csv")
        arguments['{ADJFULL_RESULTS}'] = os.path.join(save_path, "adjfull.csv")
        arguments['{CUTOFF}'] = str(cutoff)
        arguments['{VERBOSE}'] = "FALSE" # TRUE, FALSE


        def retrieve_result():
            A = pd.read_csv(arguments['{PATH_RESULTS}']).values
            os.remove(arguments['{PATH_RESULTS}'])
            os.remove(arguments['{PATH_DATA}'])
            os.remove(arguments['{PATH_DAG}'])
            return A
        
        dag = launch_R_script(str(pruning_path), arguments, output_function=retrieve_result)

    return dag, adj2order(dag)


def train_test(args, train_set, GT_DAG, data_ls, runs):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.dataset in ['sachs', 'syntren']:
        x_in = train_set[0].shape[-1]-1
    elif args.dataset.startswith('Syn'):
        x_in = args.num_nodes
    else:
        raise Exception('Dataset not recognized.')

    if args.model == 'CaPS':
        def Stein_hess(X, eta_G, eta_H, s = None):
            """
            Estimates the diagonal of the Hessian of log p_X at the provided samples points
            X, using first and second-order Stein identities
            """
            n, d = X.shape
            X = X.to(device)
            X_diff = X.unsqueeze(1)-X
            if s is None:
                D = torch.norm(X_diff, dim=2, p=2)
                s = D.flatten().median()
            K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
            
            nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s**2
            G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n).to(device)), nablaK)
            
            nabla2K = torch.einsum('kij,ik->kj', -1/s**2 + X_diff**2/s**4, K)
            return (-G**2 + torch.matmul(torch.inverse(K + eta_H * torch.eye(n).to(device)), nabla2K)).to('cpu')

        def compute_top_order(X, eta_G, eta_H, dispersion="mean"):
            n, d = X.shape
            full_X = X
            order = []
            layer = [0 for _ in range(d)]
            active_nodes = list(range(d))
            for i in range(d-1):
                H = Stein_hess(X, eta_G, eta_H)
                if dispersion == "mean": # Lemma 1 of CaPS
                    l = int(H.mean(axis=0).argmax())
                else:
                    raise Exception("Unknown dispersion criterion")

                order.append(active_nodes[l])
                active_nodes.pop(l)

                X = torch.hstack([X[:,0:l], X[:,l+1:]])
            order.append(active_nodes[0])
            order.reverse()

            # compute parents score
            active_nodes = list(range(d))
            full_H = Stein_hess(full_X, eta_G, eta_H).mean(axis=0)
            parents_score = np.zeros((d,d))
            for i in range(d):
                curr_X = torch.hstack([full_X[:,0:i], full_X[:,i+1:]])
                curr_H = Stein_hess(curr_X, eta_G, eta_H).mean(axis=0)
                parents_score[i] = get_parents_score(curr_H, full_H, i)
            print(parents_score)

            return order, parents_score
        
        cutoff = 0.001
        train_set = torch.Tensor(train_set[:, 1:])

        order, parents_score = compute_top_order(train_set, eta_G=0.001, eta_H=0.001, dispersion="mean")
        print(blue(str(order)))

        if args.pre_pruning:
            init_dag = pre_pruning_with_parents_score(full_DAG(order), parents_score, args.lambda1)
        else:
            init_dag = full_DAG(order)
        
        dag, _ = cam_pruning(init_dag, train_set, cutoff)
        
        if args.add_edge:
            dag = add_edge_with_parents_score(dag, parents_score, args.lambda2)

    return evaluate(args, dag, GT_DAG), order


if __name__ == '__main__':
    args = get_args()
    metrics_list_dict = {'fdr':[], 'tpr':[], 'fpr':[], 'shd':[], 'sid':[], 'nnz':[], 'precision':[], 'recall':[], 'F1':[], 'gscore':[]}
    metrics_res_dict = {}
    t_ls = []
    D_top = []
    simulation_seeds = [1, 2, 3, 4, 5]

    for i in range(args.runs):
        print(red('runs {}:'.format(i)))
        if args.dataset in ['sachs', 'syntren']:
            train_set, GT_DAG, data_ls = dataset.load_data(args.dataset, n=853, norm=args.norm, simulation_seed=42, num_nodes=args.num_nodes, num_samples=args.num_samples,
                                                            method=args.method, linear_sem_type=args.linear_sem_type, nonlinear_sem_type=args.nonlinear_sem_type, linear_rate=args.linear_rate, runs=i+1)
        elif args.dataset.startswith('Syn'):
            train_set, GT_DAG, data_ls = dataset.load_data(args.dataset, n=853, norm=args.norm, simulation_seed=simulation_seeds[i], num_nodes=args.num_nodes, num_samples=args.num_samples,
                                                            method=args.method, linear_sem_type=args.linear_sem_type, nonlinear_sem_type=args.nonlinear_sem_type, linear_rate=args.linear_rate, runs=i+1)
        else:
            raise Exception('Dataset not recognized.')
        
        if args.manualSeed:
            Seed = args.random_seed
        else:
            Seed = random.randint(1, 10000)
        print('Random Seed:', Seed)
        random.seed(Seed)
        torch.manual_seed(Seed)
        np.random.seed(Seed)
        if args.model in ['CaPS']:
            metrics_dict, order = train_test(args, train_set, GT_DAG, data_ls, runs=i)
        else:
            raise Exception('No such model: {}'.format(args.model))
        for k in metrics_list_dict:
            if np.isnan(metrics_dict[k]):
                metrics_list_dict[k].append(0.0)
            else:
                metrics_list_dict[k].append(metrics_dict[k])

    for k in metrics_list_dict:
        metrics_list_dict[k] = np.array(metrics_list_dict[k])
        metrics_res_dict[k] = '{}±{}'.format(np.around(np.mean(metrics_list_dict[k]), 5), np.around(np.std(metrics_list_dict[k]), 5))
        print(blue(str(k) + ': ' + metrics_res_dict[k]))
