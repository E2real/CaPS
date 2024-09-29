import os
import os.path
import numpy as np
import pickle as pk
import copy
from dag_simulation import DAG, IIDSimulation


def load_data(dataset, n=7466, norm=True, seed=42, simulation_seed=42, num_nodes=10, num_samples=1000, method='nonlinear', linear_sem_type='gauss', nonlinear_sem_type='gp', linear_rate=0.5, runs=-1, id_first=True):
    
    np.set_printoptions(suppress=True)
    
    root = 'Datasets'
    np.random.seed(seed)
    
    if dataset in ['sachs']:
        GT_DAG = np.load(os.path.join(root, dataset, "DAG.npy"))
        with open(os.path.join(root, dataset, "data_n{}.pkl".format(n)), 'rb') as f:
            data_ls = pk.load(f)
        ori_data_ls = copy.deepcopy(data_ls)
        if len(data_ls) != 1:
            full_data = np.concatenate(data_ls, axis=0)
        else:
            full_data = data_ls[0]
        mu = np.mean(full_data, axis=0)
        sigma = np.std(full_data, axis=0)
        for i in range(len(data_ls)):
            if norm:
                data_ls[i] = (data_ls[i] - mu) / sigma
            idx = np.full((data_ls[i].shape[0], 1), i)
            if id_first:
                data_ls[i] = np.concatenate([idx, data_ls[i]], axis=-1)
                # print(data_ls[i][0], data_ls[i].shape)
        
        if len(data_ls) != 1:
            data = np.concatenate(data_ls, axis=0)
        else:
            data = data_ls[0]
        np.random.shuffle(data)  # return None, inplace
        # print(data.shape)

    elif dataset in ['syntren']:
        GT_DAG = np.load(os.path.join(root, dataset, "DAG{}.npy".format(runs)))
        data = np.load(os.path.join(root, dataset, "data{}.npy".format(runs)))

        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        if norm:
            data = (data - mu) / sigma
        if id_first:
            idx = np.full((data.shape[0], 1), runs)
            data = np.concatenate([idx, data], axis=-1)
        
        np.random.shuffle(data)  # return None, inplace
        ori_data_ls = None

    elif dataset.startswith('Syn'):
        gen_graph = dataset[3:5]
        edge_sparsity = int(dataset[-1])
        print('generate', gen_graph, edge_sparsity)
        if gen_graph=='ER':
            weighted_random_dag = DAG.erdos_renyi(n_nodes=num_nodes, n_edges=edge_sparsity*num_nodes, seed=simulation_seed)
            # print('weighted_random_dag:\n', weighted_random_dag)
        elif gen_graph=='SF':
            weighted_random_dag = DAG.scale_free(n_nodes=num_nodes, n_edges=edge_sparsity*num_nodes, seed=simulation_seed)
            # print('weighted_random_dag:\n', weighted_random_dag)

        low_weight_scale = 0.1
        high_weight_scale = 1.0
        # noise_scale = np.random.uniform(0.4, 0.8, (num_nodes))
        noise_scale = 1.0 

        # method: str, (linear or nonlinear), default='linear'
        #     Distribution for standard trainning dataset.
        # sem_type: str
        #     gauss, exp, gumbel, uniform, logistic (linear); 
        #     mlp, mim, gp, gp-add, quadratic (nonlinear).

        weighted_random_dag = weighted_random_dag * np.random.uniform(low=low_weight_scale, high=high_weight_scale, size=(num_nodes, num_nodes)) * np.random.choice([-1, 1], size=(num_nodes, num_nodes))
        # np.save('./log/parents_score_discussion/{}_{}_runs{}_DAG.npy'.format(dataset, linear_rate, runs), weighted_random_dag)
        dataset = IIDSimulation(W=weighted_random_dag, n=num_samples, method=method, linear_sem_type=linear_sem_type, nonlinear_sem_type=nonlinear_sem_type, noise_scale=noise_scale, linear_rate=linear_rate)

        GT_DAG, data = dataset.B, dataset.X
        # print(weighted_random_dag, '\n', GT_DAG)
        if norm:
            mu = data.mean(axis=0)
            sigma = data.std(axis=0)
            data = (data-mu) / sigma
        if id_first:
            data = np.concatenate([np.zeros([num_samples, 1]), data], axis=1)
        ori_data_ls = None

    else:
        raise Exception('Dataset not recognized.')

    return data, GT_DAG, ori_data_ls

# if __name__=='__main__':
#     data, GT_DAG, ori_data_ls = load_data('sachs', n=853, norm=False)
#     print(data.shape, (len(ori_data_ls), len(ori_data_ls[0]), len(ori_data_ls[0][0])), '\n', GT_DAG)
#     data, GT_DAG, ori_data_ls = load_data('SynER1', num_nodes=10, num_samples=1000)
#     print(data.shape, '\n', GT_DAG)