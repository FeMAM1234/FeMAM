import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy.stats import dirichlet
import random
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d
from collections import OrderedDict
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import cv2
from multiprocessing import Pool
import os
import pickle
from PIL import Image
import pandas as pd
from models import resnet,vit_small
import sys
sys.path.append('/home/shutoche/Data/AAAI_model/PFL-Non-IID-master')
from dataset.utils.dataset_utils import separate_data, split_data_val
from torchvision.datasets import ImageFolder, DatasetFolder
from sklearn.model_selection import train_test_split
from PIL import Image
random.seed(11)
np.random.seed(11)
torch.manual_seed(11)
# def set_seed():
#     seed_value=0
#     np.random.seed(seed_value)
#     random.seed(seed_value)

def memmap_clients(clients,name,args,client_by_class,class_by_client,gd_cluster,statistic,num_per_class):
    dir = '/home/shutoche/Data/AAAI_dataset'+'/'+name+'_' + str(args.num_class_per_cluster) + '_' + '_' + str(
        args.use_class_partition) + '_' + str(args.hierarchical_data) + '_' + str(args.hierarchical_dis)+ '_' + str(args.alpha)+'/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # if client_by_class is not None:
    #     for i,item in enumerate(client_by_class):
    #         item=np.array(item)
    #         if not os.path.exists(dir + 'client_by_class'):
    #             os.makedirs(dir + 'client_by_class')
    #         np.save(dir  + 'client_by_class'+'/'+str(i)+'shape.npy', item.shape)
    #         # # 从硬盘加载shape
    #         fp = np.memmap(dir  + 'client_by_class'+'/'+str(i)+'.dat', dtype=item.dtype, mode='w+', shape=item.shape)
    #         fp[:] = item[:]
    #         fp.flush()
    # if class_by_client is not None:
    #     for i,item in enumerate(class_by_client):
    #         item = np.array(item)
    #         if not os.path.exists(dir + 'class_by_client'):
    #             os.makedirs(dir + 'class_by_client')
    #         np.save(dir + 'class_by_client' + '/' + str(i) + 'shape.npy', item.shape)
    #         fp = np.memmap(dir  + 'class_by_client'+'/'+str(i)+'.dat', dtype=item.dtype, mode='w+', shape=item.shape)
    #         fp[:] = item[:]
    #         fp.flush()
    if client_by_class is not None:
        with open(dir + 'client_by_class.pkl', 'wb') as file:
            pickle.dump(client_by_class, file)
    if class_by_client is not None:
        with open(dir + 'class_by_client.pkl', 'wb') as file:
            pickle.dump(class_by_client, file)
    if gd_cluster is not None:
        with open(dir + 'gd_cluster.pkl', 'wb') as file:
            pickle.dump(gd_cluster, file)
    if statistic is not None:
        with open(dir + 'statistic.pkl','wb') as file:
            pickle.dump(statistic,file)
    if num_per_class is not None:
        with open(dir +'num_per_class.pkl', 'wb') as file:
            pickle.dump(num_per_class, file)

    for client_id,client in clients.items():
        if not os.path.exists(dir + client_id):
            os.makedirs(dir + client_id)
        for data_id,datas in client.items():
            np.save(dir + client_id + '/' + data_id + '_data.npy', datas[0].shape)
            fp = np.memmap(dir + client_id+'/'+data_id+'_data.dat', dtype=datas[0].dtype, mode='w+', shape=datas[0].shape)
            fp[:] = datas[0][:]
            fp.flush()
            np.save(dir + client_id + '/' + data_id + '_label.npy', datas[1].shape)
            fp = np.memmap(dir + client_id + '/' + data_id + '_label.dat', dtype=datas[1].dtype, mode='w+',
                           shape=datas[1].shape)
            fp[:] = datas[1][:]
            fp.flush()
def load_memmap_clients(name,args,num_client):
    dir = '/home/shutoche/Data/AAAI_dataset' + '/' + name + '_' + str(args.num_class_per_cluster) + '_' + '_' + str(
        args.use_class_partition) + '_' + str(args.hierarchical_data) + '_' + str(args.hierarchical_dis) + '_' + str(
        args.alpha) + '/'
    client_by_class = None
    if os.path.exists(dir + 'client_by_class.pkl'):
        with open(dir + 'client_by_class.pkl', 'rb') as file:
            client_by_class = pickle.load(file)
    class_by_client=None
    if os.path.exists(dir + 'class_by_client.pkl'):
        with open(dir + 'class_by_client.pkl', 'rb') as file:
            class_by_client=pickle.load(file)
    gd_cluster = None
    if os.path.exists(dir + 'gd_cluster.pkl'):
        with open(dir + 'gd_cluster.pkl', 'rb') as file:
            gd_cluster = pickle.load(file)
    statistic = None
    if os.path.exists(dir + 'statistic.pkl'):
        with open(dir + 'statistic.pkl', 'rb') as file:
            statistic = pickle.load(file)
    num_per_class = None
    if os.path.exists(dir + 'num_per_class.pkl'):
        with open(dir + 'num_per_class.pkl', 'rb') as file:
            num_per_class = pickle.load(file)
    # client_by_class=[]
    # for i in range(num_client):
    #     loaded_shape = tuple(np.load(dir + 'client_by_class' + '/' + str(i) +'shape.npy'))
    #     fp = np.memmap(dir + 'client_by_class' + '/' + str(i) + '.dat', dtype=np.int64, mode='r', shape=loaded_shape)
    #     client_by_class.append(fp)
    # class_by_client=[]
    # if name=='cifar100':
    #     a=100
    # if name=='tinyimagenet':
    #     a=200
    # for i in range(a):
    #     loaded_shape = tuple(np.load(dir + 'class_by_client' + '/' + str(i) +'shape.npy'))
    #     fp = np.memmap(dir + 'class_by_client' + '/' + str(i) + '.dat', dtype=np.int64, mode='r', shape=loaded_shape)
    #     class_by_client.append(fp)
    clients=OrderedDict()
    for i in range(num_client):
        clients['client' + str(i)]={}
        for data_id in ['train','eval','test']:
            clients['client'+str(i)][data_id]=[]
            shapes=tuple(np.load(dir + 'client' + str(i) + '/' + data_id + '_data.npy'))
            clients['client'+str(i)][data_id].append(np.memmap(dir + 'client' + str(i) + '/' + data_id + '_data.dat', dtype=np.uint8, mode='r',
                           shape=shapes))
            shapes = tuple(np.load(dir + 'client' + str(i) + '/' + data_id + '_label.npy'))
            clients['client' + str(i)][data_id].append(
                np.memmap(dir + 'client' + str(i) + '/' + data_id + '_label.dat', dtype=np.int64, mode='r',
                          shape=shapes))
    return clients, client_by_class,class_by_client,gd_cluster,statistic,num_per_class


def train2trainval(trainset,num_per_class,ratio=0.1):
    new_trainset=[[],[]]
    new_valset=[[],[]]
    for i in range(len(num_per_class)):
        cur_dataset=trainset[0][trainset[1]==i]
        cur_label=trainset[1][trainset[1]==i]
        cur_size=len(cur_label)
        permute=np.random.permutation(np.arange(cur_size))
        val_ind=permute[:int(ratio*cur_size)]
        train_ind=permute[int(ratio*cur_size):]
        if i==0:
            new_trainset[0]=cur_dataset[train_ind]
            new_trainset[1]=cur_label[train_ind]
            new_valset[0]=cur_dataset[val_ind]
            new_valset[1]=cur_label[val_ind]
        else:
            new_trainset[0]=np.concatenate([new_trainset[0],cur_dataset[train_ind]],axis=0)
            new_trainset[1] = np.concatenate([new_trainset[1], cur_label[train_ind]], axis=0)
            new_valset[0] = np.concatenate([new_valset[0], cur_dataset[val_ind]], axis=0)
            new_valset[1] = np.concatenate([new_valset[1], cur_label[val_ind]], axis=0)
    permute=np.random.permutation(len(new_trainset[0]))
    new_trainset[0]=new_trainset[0][permute]
    new_trainset[1]=new_trainset[1][permute]
    permute = np.random.permutation(len(new_valset[0]))
    new_valset[0] = new_valset[0][permute]
    new_valset[1] = new_valset[1][permute]
    return new_trainset,new_valset


import cv2
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool


def load_val_images(args):
    val_path, file_to_class = args
    # data, labels = [], []
    data=np.empty((len(file_to_class),64,64,3),dtype=np.uint8)
    labels=[]
    for i,file_name in enumerate(os.listdir(val_path)):
        img = cv2.imread(os.path.join(val_path, file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from OpenCV's BGR to RGB
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        label = file_to_class[file_name]
        data[i]=img
        # data.append(img)
        labels.append(label)
    return data, labels
def load_images(args):
    path, wnid = args
    dir_path=os.listdir(os.path.join(path, wnid, 'images'))
    # data, labels = [], []
    data=np.empty((len(dir_path),64,64,3),dtype=np.uint8)
    labels=[]
    for i,img_path in enumerate(dir_path):
        img = cv2.imread(os.path.join(path, wnid, 'images', img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from OpenCV's BGR to RGB
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        data[i] = img
        # data.append(img)
        labels.append(wnid)
    return data, labels

def val_tinyimagenet(val_path, val_annotations_path):
    df = pd.read_csv(val_annotations_path, sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'H', 'W'])
    file_to_class = {row[1]['File']: row[1]['Class'] for row in df.iterrows()}

    # val_image_path = os.path.join(val_path, 'images')
    args = [(val_path, file_to_class)]

    # Use multiprocessing to load images in parallel
    # with Pool(os.cpu_count()) as pool:
    #     results = pool.map(load_val_images, args)

    testdata = []
    testlabel = []
    for arg in args:
        data, label =load_val_images(arg)
        testdata.extend(data)
        testlabel.extend(label)
    testset = [np.array(testdata), testlabel]
    return testset

def train_tinyimagenet(path, wnids_path):
    with open(wnids_path, 'r') as f:
        wnids = [x.strip() for x in f]
    args = [(path, wnid) for wnid in wnids]
    traindata = []
    trainlabel = []
    # Use multiprocessing to load images in parallel
    for arg in tqdm(args):
        data, label = load_images(arg)
        traindata.extend(data)
        trainlabel.extend(label)
    trainset = [np.array(traindata), trainlabel]
    return trainset
    # with Pool(os.cpu_count()) as pool:
    #     results = pool.map(load_images, args)
    #
    # traindata = []
    # trainlabel = []
    #
    # for data, labels in results:
    #     traindata.extend(data)
    #     trainlabel.extend(labels)
    # trainset = [np.array(traindata), trainlabel]
    # return trainset


# def val_tinyimagenet(val_path, val_annotations_path):
#     # 从注释文件中读取图像名和对应的标签
#     df = pd.read_csv(val_annotations_path, sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'H', 'W'])
#     file_to_class = {row[1]['File']: row[1]['Class'] for row in df.iterrows()}
#
#     data, labels = [], []
#     for file_name in os.listdir(val_path):
#         # 打开图像
#         img = Image.open(os.path.join(val_path, file_name))
#         img_array = np.array(img)
#         if len(img_array.shape) == 2:
#             img_array = np.stack([img_array] * 3, axis=-1)
#         # 获取图像的类别
#         label = file_to_class[file_name]
#         data.append(img_array)
#         labels.append(label)
#     testset=[np.array(data),labels]
#     return testset


# def train_tinyimagenet(path,wnids_path):
#     with open(wnids_path,'r') as f:
#         wnids=[x.strip() for x in f]
#     traindata,trainlabel=[],[]
#     for i, wnid in enumerate(wnids):
#         for img_path in os.listdir(os.path.join(path,wnid,'images')):
#             img=Image.open(os.path.join(path,wnid,'images',img_path))
#             img=np.array(img)
#             if len(img.shape) == 2:
#                 img = np.stack([img] * 3, axis=-1)
#             traindata.append(img)
#             trainlabel.append(wnid)
#     trainset=[np.array(traindata),trainlabel]
#     return trainset

def load_dataset(name):
    dir='/home/shutoche/Data/AAAI_dataset'
    if name == 'mnist':
        trainset = torchvision.datasets.MNIST(root=dir+'/mnist/', train=True, download=True)
        testset = torchvision.datasets.MNIST(root=dir+'/mnist/', train=False, download=True)
    elif name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=dir+'/cifar10/', train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root=dir+'/cifar10/', train=False, download=True)
    elif name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=dir+'/cifar100/', train=True, download=True)
        testset = torchvision.datasets.CIFAR100(root=dir+'/cifar100/', train=False, download=True)
    elif name == 'tinyimagenet':
        testset = val_tinyimagenet(dir +'/tinyimagenet/' + 'tiny-imagenet-200/val/images',
                                   dir +'/tinyimagenet/'+ 'tiny-imagenet-200/val/val_annotations.txt')
        trainset = train_tinyimagenet(dir +'/tinyimagenet/'+'tiny-imagenet-200/train', dir +'/tinyimagenet/'+ 'tiny-imagenet-200/wnids.txt')

        # trainset=[]
        # testset=[]
        # filename = dir + '/' + name + '/train_data.dat'
        # # a=np.memmap(filename, dtype=np.uint8, mode='r', shape=(100000, 64, 64, 3))
        # trainset.append(np.memmap(filename, dtype=np.uint8, mode='r', shape=(100000, 64, 64, 3)))
        # filename = dir + '/' + name + '/train_label.dat'
        # trainset.append(np.memmap(filename, dtype=np.int64, mode='r', shape=100000))
        # filename = dir + '/' + name + '/test_data.dat'
        # testset.append(np.memmap(filename, dtype=np.uint8, mode='r', shape=(10000, 64, 64, 3)))
        # filename = dir + '/' + name + '/test_label.dat'
        # testset.append(np.memmap(filename, dtype=np.int64, mode='r', shape=10000))
        # filename = dir + '/' + name + '/num_per_class.dat'
        # num_per_class = [550]*200
        # return trainset,testset,num_per_class

        wnid_to_int = {wnid: i for i, wnid in enumerate(set(trainset[1]))}
        train_labels_int = [wnid_to_int[wnid] for wnid in trainset[1]]
        trainset[1] = np.array(train_labels_int)
        val_labels_int = [wnid_to_int[wnid] for wnid in testset[1]]
        testset[1] = np.array(val_labels_int)
        num_class=len(set(trainset[1]))
        num_per_class = []
        for i in range(num_class):
            num_per_class.append(trainset[1].tolist().count(i)  + \
                                 testset[1].tolist().count(i))
        return trainset, testset, np.array(num_per_class)
    else:
        return None
    num_class=len(trainset.classes)
    num_per_class=[]
    for i in range(num_class):
        num_per_class.append(trainset.targets.count(i)+\
                             testset.targets.count(i))
    trainset = [trainset.data, np.array(trainset.targets)]
    testset = [testset.data, np.array(testset.targets)]
    return trainset,testset,np.array(num_per_class)

class randompicker:
    def __init__(self,arr):
        self.arr=arr
        self.available=arr.copy()
    def pick(self):
        if not self.available:
            self.available=self.arr.copy()
        chosen=random.choice(self.available)
        self.available.remove(chosen)
        return chosen



def produce_cluster_by_class(num_per_class,num_class_per_cluster):
    num_class=len(num_per_class)
    if sum(num_class_per_cluster) < num_class or any(i>num_class for i in num_class_per_cluster):
        raise Exception('The number of classes in clusters is not appropriate.')

    picker=randompicker(np.arange(num_class).tolist())
    cluster_by_class=[]
    for i in num_class_per_cluster:
        classes = []
        for j in range(i):
            classes.append(picker.pick())
        cluster_by_class.append(classes)
    return cluster_by_class

def produce_client_by_class(ran,num_per_class,num_client,cluster_by_class,num_class_per_cluster):
    if ran:
        random_numbers=uniform_filter1d(np.random.rand(len(num_class_per_cluster)),size=3)
        num_client_per_cluster=np.round(random_numbers*num_client/np.sum(random_numbers)).astype(int)
        num_client_per_cluster[num_client_per_cluster<2]=2
        diff=np.sum(num_client_per_cluster)-num_client
        if diff>0:
            while diff!=0:
                ind=np.argmax(num_client_per_cluster)
                num_client_per_cluster[ind]-=1
                diff=diff-1
        elif diff<0:
            while diff!=0:
                 ind=np.argmin(num_client_per_cluster)
                 num_client_per_cluster[ind]+=1
                 diff=diff+1
        assert np.sum(num_client_per_cluster)==num_client
    client_by_class=[elem for i,elem in enumerate(cluster_by_class) for _ in range(num_client_per_cluster[i])]
    class_by_client=[]
    for i in range(len(num_per_class)):
        arr=[]
        for j,elem in enumerate(client_by_class):
            if i in elem:
                arr.append(j)
        class_by_client.append(arr)


    return client_by_class,class_by_client

def generate_hierarchical_data(num_per_class,num_client,hs=0,min_cluster_unit=2):
    if num_client>=50:
        if len(num_per_class)==100:
            with open('/home/shutoche/PycharmProjects/her_old/cifar100_hierarchical_data.pkl','rb') as f:
                possible_data_clusters=pickle.load(f)
        elif len(num_per_class)==200:
            with open('/home/shutoche/PycharmProjects/her_old/tinyimagenet_hierarchical_data.pkl', 'rb') as f:
                possible_data_clusters = pickle.load(f)
    else:
        raise Exception(
            'You must use cifar100 or tinyimagenet, and not less than 50 clients to achieve hierarchical data.')
    # ind=np.random.choice(range(len(possible_data_clusters)))
    ind=hs
    # data_cluster = possible_data_clusters[0]
    data_cluster=possible_data_clusters[ind]
    hierarchical_clusters=data_cluster[0]
    hierarchical_labels=data_cluster[1]
    hierarchical_cluster_index_per_client=[]
    for i,num_cluster in enumerate(hierarchical_clusters[::-1]):
        if i==0:
            cluster_index_per_client=np.concatenate([np.array(list(range(num_cluster))*min_cluster_unit),np.random.choice(list(range(num_cluster)),size=num_client-(min_cluster_unit*num_cluster),replace=True)])
            cluster_index_per_client = np.sort(cluster_index_per_client)
        else:
            cluster_index_per_client=np.concatenate([np.array(list(range(num_cluster))),np.random.choice(range(num_cluster),size=len(set(hierarchical_cluster_index_per_client[-1]))-num_cluster,replace=True)])
            unique_elements, counts = np.unique(hierarchical_cluster_index_per_client[-1], return_counts=True)
            cluster_index_per_client = np.sort(cluster_index_per_client)
            cluster_index_per_client=np.repeat(cluster_index_per_client,counts)
        hierarchical_cluster_index_per_client.append(cluster_index_per_client.tolist())
    hierarchical_cluster_index_per_client=hierarchical_cluster_index_per_client[::-1]
    hierarchical_label_index_per_client=[[] for i in range(len(hierarchical_cluster_index_per_client))]
    classes=np.arange(len(num_per_class))

    for i,cluster_index_per_client in enumerate(hierarchical_cluster_index_per_client):
        ind=0
        for n,j in enumerate(cluster_index_per_client):
            if n==0:
                rand_indices = np.random.choice(len(classes), size=hierarchical_labels[i], replace=False)
                select_element = classes[rand_indices]
                hierarchical_label_index_per_client[i].append(select_element)
                classes = np.delete(classes, rand_indices)
            else:
                if j==ind:
                    hierarchical_label_index_per_client[i].append(select_element)
                else:
                    rand_indices = np.random.choice(len(classes), size=hierarchical_labels[i], replace=False)
                    select_element = classes[rand_indices]
                    hierarchical_label_index_per_client[i].append(select_element)
                    classes = np.delete(classes, rand_indices)
                    ind=ind+1

    client_by_class=[]
    for i in range(num_client):
        client_by_class.append(np.concatenate([hierarchical_label_index_per_client[j][i] for j in range(len(hierarchical_label_index_per_client))]).tolist())

    class_by_client = []
    for i in range(len(num_per_class)):
        arr = []
        for j, elem in enumerate(client_by_class):
            if i in elem:
                arr.append(j)
        class_by_client.append(arr)
    return client_by_class,class_by_client,[np.array(cluster) for cluster in hierarchical_cluster_index_per_client]

def get_sample(clients,dirichlet_dis,is_train,dataset,classid,client_by_class,class_by_client):
    ind=np.random.permutation(len(dataset[1][dataset[1]==classid]))
    cur_data=dataset[0][dataset[1]==classid][ind]
    cur_label=dataset[1][dataset[1]==classid][ind]
    prob=dirichlet([dirichlet_dis]*len(class_by_client[classid]),seed=classid).rvs()
    # print(prob)
    class_num_by_client=np.round(len(ind)*prob)[0]
    diff=np.sum(class_num_by_client)-len(ind)
    count=0
    if diff>0:
        while diff!=0:
            class_num_by_client[np.argmax(class_num_by_client)]-=1
            count=count+1
            if count>20:
                break
            diff-=1
    elif diff<0:
        while diff!=0:
            class_num_by_client[np.argmin(class_num_by_client)]+=1
            count = count + 1
            if count > 20:
               break
            diff+=1
    assert np.sum(class_num_by_client)==len(ind)
    count=0
    while np.any(class_num_by_client<1):
        # breakpoint()
        count+=1
        if count > 20:
            break
        class_num_by_client[np.argmax(class_num_by_client)] -= 1
        class_num_by_client[np.argmin(class_num_by_client)] += 1
    cur_cumsum=np.concatenate(([0],np.cumsum(class_num_by_client))).astype(int)
    for j in range(len(class_by_client[classid])):
        data_dict=[cur_data[cur_cumsum[j]:cur_cumsum[j+1]],cur_label[cur_cumsum[j]:cur_cumsum[j+1]]]
        if 'client'+str(class_by_client[classid][j]) not in clients:
            clients['client' + str(class_by_client[classid][j])] = {}
            clients['client' + str(class_by_client[classid][j])]['train']={}
            clients['client' + str(class_by_client[classid][j])]['eval'] = {}
            clients['client' + str(class_by_client[classid][j])]['test'] = {}
        if is_train==0:
            if clients['client' + str(class_by_client[classid][j])]['train']=={}:
                clients['client' + str(class_by_client[classid][j])]['train'] = data_dict
            else:
                clients['client'+str(class_by_client[classid][j])]['train'][0]=np.concatenate([clients['client'+str(class_by_client[classid][j])]['train'][0],data_dict[0]],axis=0)
                clients['client'+str(class_by_client[classid][j])]['train'][1]=np.concatenate([clients['client'+str(class_by_client[classid][j])]['train'][1],data_dict[1]])
        elif is_train==2:
            if clients['client' + str(class_by_client[classid][j])]['test'] == {}:
                clients['client' + str(class_by_client[classid][j])]['test'] = data_dict
            else:
                clients['client' + str(class_by_client[classid][j])]['test'][0] = np.concatenate(
                    [clients['client' + str(class_by_client[classid][j])]['test'][0], data_dict[0]], axis=0)
                clients['client' + str(class_by_client[classid][j])]['test'][1] = np.concatenate(
                    [clients['client' + str(class_by_client[classid][j])]['test'][1], data_dict[1]])
        elif is_train==1:
            if clients['client' + str(class_by_client[classid][j])]['eval'] == {}:
                clients['client' + str(class_by_client[classid][j])]['eval'] = data_dict
            else:
                clients['client' + str(class_by_client[classid][j])]['eval'][0] = np.concatenate(
                    [clients['client' + str(class_by_client[classid][j])]['eval'][0], data_dict[0]], axis=0)
                clients['client' + str(class_by_client[classid][j])]['eval'][1] = np.concatenate(
                    [clients['client' + str(class_by_client[classid][j])]['eval'][1], data_dict[1]])

    return clients

def produce_client_by_data(trainset,valset,testset,num_per_class,client_by_class,class_by_client,dirichlet_dis):
    num_class=len(num_per_class)
    clients={}
    for i in range(num_class):
        clients=get_sample(clients,dirichlet_dis,0,trainset,i,client_by_class,class_by_client)
        clients = get_sample(clients, dirichlet_dis, 1, valset, i, client_by_class, class_by_client)
        clients=get_sample(clients,dirichlet_dis,2,testset,i,client_by_class,class_by_client)
    clients=OrderedDict(clients)
    for i in range(len(clients)):
        value=clients.pop('client'+str(i))
        clients['client'+str(i)]=value
    return clients

def generate_dataloader(name,data,batch_size,transform=None):

    class Mydataset(Dataset):
        def __int__(self, data, label, transform=None):
            self.data = data
            self.label = label
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample_data = self.data[idx]
            if self.transform:
                sample_data = self.transform(sample_data)

            return sample_data, self.label[idx]


    if transform is None:
        if name=='cifar10':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))])
        if name=='cifar100':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761))])
        if name=='tinyimagenet':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4802,0.4481,0.3975), (0.2302,0.2265,0.2262))])
        if name=='emnist':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))])

    class domain_dataset(Dataset):
        def __int__(self, data, label, transform=None):
            self.data = data
            self.label = label
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            path = self.data[idx]
            target = self.label[idx]
            target = int(target)
            img = Image.open(path)
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def convert_to_rgb(img):
        """将单通道图像转换为三通道图像"""
        if img.mode == 'L':  # 检查图像是否为单通道
            return img.convert('RGB')
        return img

    if transform is None:
        if name=='office_caltech_10':
            transform = transforms.Compose([transforms.Lambda(lambda img: convert_to_rgb(img)),transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),
                                                                                          (0.229, 0.224, 0.225))])

    dataset=Mydataset()
    if name=='office_caltech_10':
        dataset=domain_dataset()
    dataset.data=data[0]
    dataset.label=data[1]
    dataset.transform=transform
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=False)
    # for data in dataloader:
    #     print()
    return dataloader

def load_data_without_class(name):
    trainset, testset, num_per_class = load_dataset(name)
    trainset, evalset = train2trainval(trainset, num_per_class)


def produce_data(args, use_class_partition, name, num_client, num_class_per_cluster, dirichlet_dis, alpha,
              hierarchical_data=False):
    if name == 'office_caltech_10':
        clients, domains_num, num_class_per_cluster_num, _ = load_domain_adaption_dataset('office_caltech_10')
        return clients, domains_num, num_class_per_cluster_num, None

    trainset, testset, num_per_class = load_dataset(name)
    if not use_class_partition:
        X, y, statistic = separate_data(
            (np.concatenate([trainset[0], testset[0]], axis=0), np.concatenate([trainset[1], testset[1]], axis=0)),
            num_client, len(num_per_class),
            True, False, 'dir',None, alpha)
        if name == 'tinyimagenet':
            train_data, eval_data, test_data = split_data_val(X, y, val_size=0.1, test_size=1 / 11)
        else:
            train_data, eval_data, test_data = split_data_val(X, y, val_size=0.1, test_size=1 / 6)
        clients = {}
        for i in range(num_client):
            clients['client' + str(i)] = {}
            clients['client' + str(i)]['train'] = []
            clients['client' + str(i)]['eval'] = []
            clients['client' + str(i)]['test'] = []
            clients['client' + str(i)]['train'].append(train_data[i]['x'])
            clients['client' + str(i)]['train'].append(train_data[i]['y'])
            clients['client' + str(i)]['eval'].append(eval_data[i]['x'])
            clients['client' + str(i)]['eval'].append(eval_data[i]['y'])
            clients['client' + str(i)]['test'].append(test_data[i]['x'])
            clients['client' + str(i)]['test'].append(test_data[i]['y'])
        del X, y, train_data, eval_data, test_data
        memmap_clients(clients, name, args, None, None, None, statistic, num_per_class)
        clients, _, _, _, statistic, num_per_class = load_memmap_clients(name, args, num_client)
        return clients, statistic, num_per_class, None

    trainset, evalset = train2trainval(trainset, num_per_class)
    if hierarchical_data:
        client_by_class, class_by_client, gd_cluster = generate_hierarchical_data(num_per_class, num_client,
                                                                                  args.hierarchical_dis)
        gd_cluster = ([len(set(cluster.tolist())) for cluster in gd_cluster], gd_cluster)
    else:
        gd_cluster = None
        cluster_by_class = produce_cluster_by_class(num_per_class, num_class_per_cluster)
        client_by_class, class_by_client = produce_client_by_class(True, num_per_class, num_client, cluster_by_class,
                                                                   num_class_per_cluster)
    clients = produce_client_by_data(trainset, evalset, testset, num_per_class, client_by_class, class_by_client,
                                     dirichlet_dis)
    memmap_clients(clients, name, args, client_by_class, class_by_client, gd_cluster, None, None)
    clients, client_by_class, class_by_client, gd_cluster, _, _ = load_memmap_clients(name, args, num_client)
    return clients, client_by_class, class_by_client, gd_cluster

def load_data(args,use_class_partition,name,num_client,num_class_per_cluster,dirichlet_dis,alpha,hierarchical_data=False):
    if use_class_partition:
        clients, client_by_class, class_by_client, gd_cluster, _, _ = load_memmap_clients(name, args, num_client)
        return clients, client_by_class, class_by_client, gd_cluster
    else:
        clients, _, _, _, statistic, num_per_class = load_memmap_clients(name, args, num_client)
        return clients, statistic, num_per_class, None


def load_model(name,num_class,data_name,sample_data,batch_size,image_size=None):
    if name == 'resnet':
        model = resnet.ResNet18(num_class=len(num_class))
    elif name == 'vit':
        model = vit_small.ViT(
            # image_size=data['client' + str(0)]['train'][0].shape[1],
            image_size=image_size,
            patch_size=4,
            num_classes=len(num_class),
            dim=int(512),
            depth=4,
            heads=6,
            mlp_dim=256,
            dropout=0.1,
            emb_dropout=0.1
        )
    train_loader = generate_dataloader(data_name, sample_data, batch_size)
    model.eval()
    for (input, target) in train_loader:
        output = model(input)
        break
    return model



def load_domain_adaption_dataset(name):
    dir = '/home/shutoche/Data/AAAI_dataset'
    if name == 'office_caltech_10':
        domains_num=[6,8,4,2]
        num_class_per_cluster_num=[[5,5],[5,5],[5,5],[5,5]]
        num_client=sum(domains_num)
        clients= {}
        full_domains=[]
        for ind,domain_name in enumerate(['amazon','caltech','webcam','dslr']):
            imagefolder_obj = ImageFolder(root=dir + '/Office_Caltech_10/'+domain_name)
            full_domains.append(imagefolder_obj)
            num_per_class=[imagefolder_obj.targets.count(i) for i in set(imagefolder_obj.targets)]
            num_class_per_cluster=num_class_per_cluster_num[ind]
            num_client = domains_num[ind]
            cluster_by_class = produce_cluster_by_class(num_per_class, num_class_per_cluster)
            client_by_class, class_by_client = produce_client_by_class(True, num_per_class, num_client, cluster_by_class,
                                                                       num_class_per_cluster)
            dataset=[imagefolder_obj.imgs[i][0] for i in range(len(imagefolder_obj.imgs))]
            label=[imagefolder_obj.imgs[i][1] for i in range(len(imagefolder_obj.imgs))]
            X_train,X_test,y_train,y_test=train_test_split(dataset,label,test_size=0.3)
            X_test,X_val,y_test,y_val=train_test_split(X_test, y_test, test_size=1/3)
            train_ind=np.arange(len(y_train))
            test_ind=np.arange(len(y_test))
            val_ind=np.arange(len(y_val))
            del dataset,label
            domain_client = produce_client_by_data([train_ind,np.array(y_train)], [val_ind,np.array(y_val)], [test_ind,np.array(y_test)], num_per_class, client_by_class, class_by_client,
                                         10)
            for i in range(len(domain_client)):
                for name,value in domain_client['client'+str(i)].items():
                    if name == 'train':
                        a=[X_train[n] for n in value[0]]
                        value[0]=a
                    if name == 'eval':
                        a=[X_val[n] for n in value[0]]
                        value[0]=a
                    if name == 'test':
                        a=[X_test[n] for n in value[0]]
                        value[0]=a
            all_ind=len(clients)
            new_domain_client={}
            for i in range(domains_num[ind]):
                new_domain_client['client'+str(all_ind+i)]=domain_client['client' + str(i)]
                # domain_client['client'+str(all_ind+i)]=domain_client.pop('client' + str(i))
            del(domain_client)
            clients.update(new_domain_client)
    return clients,client_by_class, class_by_client,None

if __name__=='__main__':
    clients,client_by_class, class_by_client,_=load_domain_adaption_dataset('office_caltech_10')
    dataloader=generate_dataloader('office_caltech_10',clients['client'+str(0)]['train'],32)
    for img,lab in dataloader:
        print()
    print()
