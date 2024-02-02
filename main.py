import copy
import sys

import torch,torchvision
from loaddata import *
from server_no_loss_thresh import Server
import torch.nn as nn
import time
import math
import os
from metric import save_meters
from clients import client
# from aggregate_module import *
import argparse
import torchsummary
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score,accuracy_score
from pympler import asizeof
import pdb
import pickle
import ast
from metric import meters

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--mark', default='new_disign', type=str, help='')
parser.add_argument('--data-name', default='cifar100', type=str, help='choose the data')
parser.add_argument('--local-epoch', default=2, type=int, help='number of local epoch')
parser.add_argument('--model_name', default='resnet', type=str, help='choose the model')
parser.add_argument('--hierarchical-data', default=False, type=str2bool, help='')
parser.add_argument('--hierarchical-dis', default=6, type=int, help='')
parser.add_argument('--num-client', default=50, type=int, help='choose the number of clients')
parser.add_argument('--num-class-per-cluster', default='[10]', type=str, help='choose the number of class per cluster in dataset')
parser.add_argument('--use-class-partition', default=False, type=str2bool, help='whether to partition data by class')
parser.add_argument('--cluster-num', default='[1,5,5,5,5,5,5,5]', type=str, help='number of clusters for methods with fixed clusters')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='')
parser.add_argument('--dirichlet-dis', default=10, type=float, help='choose the paramter of dirichlet distribution of data')
parser.add_argument('--global-epoch', default=700, type=int, help='number of global epo ch')
parser.add_argument('--init-epoch', default=400, type=int, help='number of init epoch')
# parser.add_argument('--cluster-keys', default=2, type=int, help='select some model layers to group id')
parser.add_argument('--use-diff', default=False, type=str2bool, help='whether use gradient as group sample')
parser.add_argument('--add-layer', default=True, type=str2bool, help='whether add layer')
parser.add_argument('--acc-queue-length', default=50, type=int, help='number of rounds to judge convergence')
parser.add_argument('--std-threshold', default=1, type=float, help='')
# parser.add_argument('--threshold1', default=0.01, type=float, help='')
# parser.add_argument('--threshold2', default=1.5, type=float, help='')
parser.add_argument('--structure-length', default=5, type=float, help='')
# parser.add_argument('--least-layer-size', default=0.2, type=float, help='')
parser.add_argument('--alpha', default=0.1, type=float, help='')
parser.add_argument('--device', default=0, type=int, help='')
if __name__=='__main__':
    args = parser.parse_args()
    args.cluster_num=ast.literal_eval(args.cluster_num)
    # data, client_by_class, class_by_client, gd_cluster = produce_data(args, args.use_class_partition, args.data_name,
    #                                                                args.num_client,
    #                                                                ast.literal_eval(args.num_class_per_cluster),
    #                                                                args.dirichlet_dis, args.alpha,
    #                                                                args.hierarchical_data)
    # sys.exit()
    data,client_by_class, class_by_client,gd_cluster=load_data(args,args.use_class_partition,args.data_name,args.num_client,ast.literal_eval(args.num_class_per_cluster),args.dirichlet_dis,args.alpha,args.hierarchical_data)
    args.client_by_class=client_by_class
    args.class_by_client=class_by_client
    args.gd_cluster=gd_cluster
    # gd_cluster=[gd_cluster[2]]
    model=load_model(args.model_name,class_by_client,args.data_name,data['client' + str(0)]['train'],32,image_size=None)
    clients=[]
    for i in range(args.num_client):
        clients.append(client(i,args,data['client' + str(i)],model))
    server=Server(args)
    server.init_model(model)
    server.time.append(time.time())
    for epoch in range(args.global_epoch):
        print('epoch:', epoch)

        server.model_distribute(epoch,clients)
        add_layer=server.eval(clients)

        if server.leaf_only==False:
            break

        if add_layer:
            server.model_distribute(epoch, clients)

        server.train(clients)
        server.aggregate(clients)
        server.model_distribute(epoch,clients)
        server.test(clients)
    server.meters.update_clients(clients)
    save_meters(args,server.meters)
    server.time.append(time.time())



