
import copy
import numpy as np
from loaddata import generate_dataloader
import torch
from metric import server_meters
from sklearn.cluster import KMeans,SpectralClustering
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster,to_tree
from sklearn.metrics import silhouette_score,calinski_harabasz_score,davies_bouldin_score
from collections import deque
from itertools import groupby

class Server():
    def __init__(self,args):
        self.time=[]
        self.args=args
        self.grow_layer=True
        self.add=self.args.add_layer
        self.structure= {0:{}}
        self.num_layers=1
        self.leaf_only=True
        self.act_eval_loss_queue = [999]*args.acc_queue_length
        self.old_act_eval_loss_queue = [999]*args.acc_queue_length
        # self.train_acc_queue = [0]*args.acc_queue_length
        self.eval_acc=[]
        self.test_acc = []
        self.mean_test_acc=0
        self.meters=server_meters(args)
        self.process_mask=[0,True]*self.args.num_client
        # self.layer_istrain_map={}
    def init_model(self, model):
        self.structure[0][0] = [np.arange(self.args.num_client), copy.deepcopy(model)]
        self.keys = self.structure[0][0][1].state_dict().keys()
        # self.layer_istrain_map={0:np.ones(self.args.num_client,dtype=bool)}

    # def select_client_fun(self):
    #     self.select_client=np.arange(self.args.num_client)

    def model_distribute(self, epoch,clients):
        self.epoch=epoch
        for client in clients:
            client.load_model(self.structure)
    def train(self,clients):
        # acc_train = []
        for client in clients:
            if client.stop_layer is True:
                continue
            client.train()
            # meter = client.train()
            # acc_train.append(meter.accuracy_score)
        # if self.leaf_only:
        #     num=np.sum(np.array([not client.stop_layer for client in clients]))
        # else:
        #     num=self.args.num_client
        # self.train_acc_queue.pop(0)
        # self.train_acc_queue.append(np.sum(np.array(acc_train)) / num)
        # print('train_accuracy:', acc_train)
        # print('overall_train_accuracy:', np.sum(np.array(acc_train)) / num)

    def eval(self, clients):
        acc = []
        loss=[]
        activate_acc=[]
        activate_loss=[]
        act_client=[]
        for client in clients:
            eval_loader = generate_dataloader(client.args.data_name, client.data['eval'], batch_size=32)
            meter = client.eval(eval_loader,'eval')
            acc.append(meter.accuracy_score)
            loss.append(meter.new_loss)
            if not client.stop_layer:
                act_client.append(client.id)
                activate_acc.append(meter.accuracy_score)
                activate_loss.append(meter.new_loss)
        self.act_eval_loss_queue.pop(0)
        self.act_eval_loss_queue.append(np.sum(np.array(activate_loss)) / len(act_client))
        self.eval_acc.append(np.array(acc))
        print('eval_accuracy:', acc)
        print('overall_eval_accuracy:', np.sum(np.array(acc)) / self.args.num_client)
        print('eval_loss:', loss)
        print('overall_eval_loss:', np.sum(np.array(loss)) / self.args.num_client)
        # print('activate_eval_accuracy:', activate_acc)
        # print('overall_activate_eval_accuracy:', np.sum(np.array(activate_acc)) / len(act_client))
        print('activate_eval_loss:', activate_loss)
        print('overall_activate_eval_loss:', np.sum(np.array(activate_loss)) / len(act_client))
        # print('activate client list:',act_client)
        print('activate client list num:', len(act_client))
        self.meters.update_optimal(copy.deepcopy(self.structure), np.sum(np.array(loss)) / self.args.num_client,
                                   copy.deepcopy(self.epoch), copy.deepcopy(self.mean_test_acc),
                                   copy.deepcopy(self.test_acc))
        self.meters.update_structure_list(copy.deepcopy(self.structure), copy.deepcopy(self.epoch), act_client)
        self.meters.update(self.epoch, acc, np.sum(np.array(acc)) / self.args.num_client, loss,
                           np.sum(np.array(loss)) / self.args.num_client, activate_acc,
                           np.sum(np.array(activate_acc)) / len(act_client), activate_loss,
                           np.sum(np.array(activate_loss)) / len(act_client))
        # if self.add and (self.epoch+1)%2==0:
        if self.add:
            if self.epoch < self.args.init_epoch:
                return False
            # if self.args.data_name == 'cifar100' and self.epoch < 350:
            #     return False
            return self.add_layer(clients)
        return False
    def test(self,clients):
        self.test_acc = []
        for client in clients:
            test_loader = generate_dataloader(client.args.data_name, client.data['test'], batch_size=32)
            meter = client.eval(test_loader,'test')
            self.test_acc.append(meter.accuracy_score)
        print('test_accuracy:',  self.test_acc)
        self.mean_test_acc=np.sum(np.array( self.test_acc)) / self.args.num_client
        print('overall_test_accuracy:',  self.mean_test_acc)

    def add_layer(self,clients):
        redistribute = False
        # if self.epoch==self.args.training_round[self.num_layers-1]:
        if np.std(self.act_eval_loss_queue)<self.args.std_threshold:
            redistribute=True
            print('act_eval_loss_queue:',self.act_eval_loss_queue)
            self.act_eval_loss_queue = [999] * self.args.acc_queue_length

            if self.num_layers>1:
                for client in clients:
                    client.process_mask.append(None)
                    client.process_mask.append(len(client.eval_loss))
                    indexes = [index for index, value in enumerate(client.eval_loss) if value == 999]
                    index1 = indexes[-1]
                    a = client.eval_loss[index1 + 1:]
                    client.eval_loss.append(999)
                    if len(a) > self.args.acc_queue_length:
                        a = a[-self.args.acc_queue_length:]
                    for i,item in enumerate(reversed(client.process_mask)):
                        if item==True:
                            index2=client.process_mask[len(client.process_mask)-1-i-1]
                            index2_1=client.process_mask[len(client.process_mask)-1-i+1]
                            break
                    b = client.eval_loss[index2 + 1:index2_1]
                    if len(b) > self.args.acc_queue_length:
                        b = b[-self.args.acc_queue_length:]
                    surpass = (sum(b) / len(b) - sum(a) / len(a) > 0)
                    if surpass:
                        client.process_mask[-2]=True
                    else:
                        client.process_mask[-2]=False
                        for level,value in self.structure[self.num_layers - 1].items():
                            value[0]=value[0][value[0]!=client.id]
                    self.process_mask[client.id]=client.process_mask
                new_structure={}
                i=0
                for key,value in self.structure[self.num_layers - 1].items():
                    if self.structure[self.num_layers - 1][key][0].size==0:
                        continue
                    else:
                        new_structure[i]=self.structure[self.num_layers - 1][key]
                        i=i+1
                if len(new_structure)==0:
                    del self.structure[self.num_layers - 1]
                else:
                    self.structure[self.num_layers-1]=new_structure
            else:
                for client in clients:
                    client.process_mask.append(len(client.eval_loss))
                    client.eval_loss.append(999)
                    self.process_mask[client.id] = client.process_mask

            if self.num_layers >= self.args.structure_length:
                for i in range(5):
                    self.model_distribute(self.epoch,clients)
                    self.eval(clients)
                    self.test(clients)
                    self.meters.update_final(copy.deepcopy(self.structure),
                                               copy.deepcopy(self.epoch), copy.deepcopy(self.mean_test_acc),
                                               copy.deepcopy(self.test_acc))
                self.leaf_only = False
                self.grow_layer=False
                return redistribute
            self.meters.update_mask(self.process_mask)
            self.num_layers += 1
            self.structure[self.num_layers - 1] = {}
            self.structure[self.num_layers - 1][0] = [[], copy.deepcopy(self.structure[0][0][1])]
            for client in clients:
                client.stop_layer = False
                self.structure[self.num_layers - 1][0][0].append(client.id)
            self.structure[self.num_layers-1][0][0]=np.array(self.structure[self.num_layers-1][0][0])
            # skip_aggregate = True
            #
            # if len(self.structure)>=a:
            #     # self.grow_layer=False
            #     skip_aggregate = True
            # if self.structure[self.num_layers-1][0][0].size==0:
            #     del self.structure[self.num_layers-1]
            #     self.grow_layer=False
            #     self.leaf_only=False
            #     skip_aggregate=False
            # self.meters.update_vital_epoch(copy.deepcopy(self.epoch),self.grow_layer,self.leaf_only)
        return redistribute

    def aggregate_id(self,clients):
        clusters=[self.args.cluster_num[layer_id] for layer_id in range(self.num_layers)]
        if not self.args.add_layer:
            if self.epoch<3:
                clusters=[1]
        layers = [list(self.structure.keys())[-1]] if self.leaf_only else list(self.structure.keys())
        # layers = [len(self.structure) - 1] if self.leaf_only else list(range(len(self.structure)))
        if self.leaf_only:
            clusters=[clusters[-1]]
        self.diff_sample=[]
        aggregate_id=[]
        for client in clients:
            if client.stop_layer is True:
                continue
            self.diff_sample.append(client.diff_sample)
            aggregate_id.append(client.id)
        aggregate_id=np.array(aggregate_id)
        for layer,cluster_num in zip(layers,clusters):
            kmeans = KMeans(n_clusters=cluster_num, init='random', n_init=10, max_iter=300, tol=1e-4)
            a = []
            for sample in self.diff_sample:
                a.append(sample[layer])
            if self.num_layers>=self.args.structure_length:
            # if len(a)<self.args.least_layer_size*self.args.num_client:
                kmeans.n_clusters = len(a)
            kmeans.fit(a)
            labels = kmeans.labels_
            self.structure[layer] = {}
            for i in set(labels):
                # a=np.squeeze(np.argwhere(labels == i), axis=-1)
                self.structure[layer][i] = [aggregate_id[np.squeeze(np.argwhere(labels == i), axis=-1)], None]
        for layer in list(self.structure.keys()):
            group_ids = []
            groups = []
            for group_id, group in self.structure[layer].items():
                group_ids.append(str(group_id))
                groups.append(group[0].tolist())
            print(group_ids)
            print(groups)

    def aggregate_model(self,clients):
        def weight_sum_param(models, weights):
            new_model = copy.deepcopy(models[0])
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            new_param = {}
            for key in self.keys:
                for (model, weight) in zip(models, weights):
                    if key not in new_param:
                        new_param[key] = model.state_dict()[key] * weight
                    else:
                        new_param[key] = new_param[key] + model.state_dict()[key] * weight
            new_model.load_state_dict(new_param)
            return new_model
        layers = [list(self.structure.keys())[-1]] if self.leaf_only else list(self.structure.keys())

        # layers = [len(self.structure) - 1] if self.leaf_only else list(range(len(self.structure)))
        for layer in layers:
            for group_id, group in self.structure[layer].items():
                model = []
                weight = []
                label = group[0]
                for client in clients:
                    if client.id in label:
                        model.append(client.structure[layer].cpu())
                        weight.append(client.num_data)
                new_model = weight_sum_param(model, weight)
                self.structure[layer][group_id][1] = new_model

    def aggregate(self, clients):
        self.aggregate_id(clients)
        self.aggregate_model(clients)


