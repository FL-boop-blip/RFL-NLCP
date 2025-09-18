import torch
from client import *
from .server import *
import numpy as np
from collections import Counter

class RFLNLCP(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):   
        super(RFLNLCP, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        self.dual_steps = torch.ones(args.total_client, dtype=torch.float32)
        self.h_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        print(" Dual Variable Param List  --->  {:d} * {:d}".format(
                self.clients_updated_params_list.shape[0], self.clients_updated_params_list.shape[1]))
        
        # rebuild
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
            'Local_dual_correction': torch.zeros((init_par_list.shape[0])),
        }
        self.client_models = list(range(self.args.total_client))
        self.Client = rflnlcp

    def _select_clients_(self, t,cluster_labels_m,counts):
        # select active clients ID
        inc_seed = 0
        while(True):
            np.random.seed(t + self.args.seed + inc_seed)
            act_list = np.random.uniform(size=self.args.total_client)
            act_clients = act_list <= self.args.active_ratio
            selected_clients = np.sort(np.where(act_clients)[0])
            selected_clusters_labels = cluster_labels_m[selected_clients]
            count_per_cluster = Counter(selected_clusters_labels)
            print('count per cluster is {}'.format(count_per_cluster))
            all_categories = list(range(self.args.num_cluster))
            for category in all_categories:
                if category not in count_per_cluster:
                    count_per_cluster[category] = 1
            sorted_counts = sorted(count_per_cluster.items())
            print("sorted_counts is {}".format(sorted_counts))
            counts_values = [count for _, count in sorted_counts]
            count_array = np.array(counts_values)
            enlarge = counts / count_array * self.args.beta3
            
            inc_seed += 1
            if len(selected_clients) != 0:
                return selected_clients, enlarge
            
    def _cluster_clients_(self, t):
        Averaged_update = torch.zeros(self.server_model_params_list.shape)
        for client in self.client_models:
            print('pre-training for k-means {}'.format(client))
            dataset = data.DataLoader(Dataset(self.datasets.client_x[client],\
                         self.datasets.client_y[client], train=True, dataset_name=self.args.dataset),\
                             batch_size=self.args.batchsize, shuffle=True)
            self.process_for_communication(client, Averaged_update)
            _edge_device = self.Client(device=self.device, model_func=self.model_func, received_vecs=self.comm_vecs,
                                          dataset=dataset, lr=self.lr, args=self.args)
            self.received_vecs = _edge_device.train()
            self.clients_updated_params_list[client] = self.received_vecs['local_update_list']
            # self.clients_params_list[client] = self.received_vecs['local_model_param_list']
            # self.postprocess(client)  
                
            # release the salloc
            del _edge_device
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        pca = PCA(n_components=10)
        reduced_updates = pca.fit_transform(self.clients_updated_params_list.cpu().numpy())
        k_means = KMeans(n_clusters=self.args.num_cluster, random_state=0).fit(reduced_updates)
        cluster_labels_m = k_means.labels_
        print("Cluster labels for each client:")
        print(cluster_labels_m)
        unique, counts = np.unique(cluster_labels_m, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Cluster label {label}: {count}")
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(reduced_updates,cluster_labels_m)
        print(f"Silhouette Score: {silhouette_avg}")
        self.clients_updated_params_list = torch.zeros((self.args.total_client,self.server_model_params_list.shape[0]))
        return cluster_labels_m, counts


    
    
    def process_for_communication(self, client, Averaged_update):
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list)
        else:
            # RI adopts the w(i,t) = w(t) + beta[w(t) - w(i,K,t-1)] as initialization
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list + self.args.beta\
                                    * (self.server_model_params_list - self.clients_params_list[client]))
        
        # combination of dual variable and global server model
        # local gradient is g - hi + alpha(wk - wt)
        #                  ---       --------
        #                grad      weight-decay
        # Therefore, -hi - alpha*wt are communicated as Local_dual_correction term
        # self.comm_vecs['Local_dual_correction'].copy_(self.h_params_list[client] - self.server_model_params_list)
        self.comm_vecs['Local_dual_correction'].copy_(self.h_params_list[client] - self.comm_vecs['Params_list'])

    
    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        # FedDyn (ServerOpt)
        # w(t+1) = average_s[wi(t)] + average_c[h(t)]
        return Averaged_model + torch.mean(self.h_params_list, dim=0)
    
    
    def postprocess(self, client,received_vecs,d):
        self.h_params_list[client] += d * self.clients_updated_params_list[client]                             


    def train(self):
        print("##=============================================##")
        print("##           Training Process Starts           ##")
        print("##=============================================##")
        
        Averaged_update = torch.zeros(self.server_model_params_list.shape)
        
        for t in range(self.args.comm_rounds):
            if t == 0:
                cluster_labels_m, counts = self._cluster_clients_(t)
            start = time.time()
            # select active clients list
            selected_clients,enlarge = self._select_clients_(t,cluster_labels_m,counts)
            print('============= Communication Round', t + 1, '=============', flush = True)
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clients])))
            for client in selected_clients:
                d = enlarge[cluster_labels_m[client]]
                dataset = data.DataLoader(Dataset(self.datasets.client_x[client],\
                        self.datasets.client_y[client], train=True, dataset_name=self.args.dataset),\
                            batch_size=self.args.batchsize, shuffle=True)
        
                    #dataset = (self.datasets.client_x[client], self.datasets.client_y[client])
                self.process_for_communication(client, Averaged_update)
                _edge_device = self.Client(device=self.device, model_func=self.model_func, received_vecs=self.comm_vecs,
                                          dataset=dataset, lr=self.lr, args=self.args)
                self.received_vecs = _edge_device.train()
                self.clients_updated_params_list[client] = self.received_vecs['local_update_list']
                self.clients_params_list[client] = self.received_vecs['local_model_param_list']
                self.postprocess(client,self.received_vecs,d)
                
                # release the salloc
                del _edge_device
            
            # calculate averaged model
            Averaged_update = torch.mean(self.clients_updated_params_list[selected_clients], dim=0)
            Averaged_model  = torch.mean(self.clients_params_list[selected_clients], dim=0)
            
            self.server_model_params_list = self.global_update(selected_clients, Averaged_update, Averaged_model)
            set_client_from_params(self.device, self.server_model, self.server_model_params_list)
            
            
            self._test_(t, selected_clients)
            self._lr_scheduler_()
            
            # time
            end = time.time()
            self.time[t] = end-start
            print("            ----    Time: {:.2f}s".format(self.time[t]), flush = True)
    
            
        
        self._save_results_()
        self._summary_()