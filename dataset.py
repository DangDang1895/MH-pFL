import os
import random
import numpy as np
import torch.utils.data
from utils import set_seed
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, EMNIST

def dirichlet_distribution(data_name, dataroot, num_clients, batch_size, seed, least_nums, alpha=0.01):
    set_seed(seed)
    if data_name =='cifar10':
        normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        data_obj = CIFAR10
        alpha=0.1
    elif data_name == 'cifar100':
        normalization = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        data_obj = CIFAR100
        if num_clients>100:
            least_nums=15
    elif data_name == 'tiny_imageNet':
        data_obj = ImageFolder
    elif data_name == 'emnist':
        data_obj = EMNIST
        least_nums=10
    else:
        raise ValueError("choose data_name from ['emnist', 'cifar10', 'tiny_imageNet' ,'cifar100']")

    if data_name=='tiny_imageNet':
        transform = transforms.Compose([  
            transforms.Resize((32, 32)),  
            transforms.ToTensor(), 
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),  
        ])  
        train_data = ImageFolder(os.path.join(dataroot, 'tiny-imagenet-200/train'), transform=transform)  
        targets = torch.tensor(train_data.targets)

    elif(data_name == 'emnist'):
        transform = transforms.Compose([  
            transforms.Resize((32, 32)),  
            transforms.ToTensor(),  
            transforms.Normalize((0.5,), (0.5,))  
        ])
        train_data = data_obj(root=dataroot, split='byclass', train=True, download=True, transform=transform)  
        targets = train_data.targets  

    else:
        transform =  transforms.Compose([transforms.ToTensor(), normalization])
        train_data = data_obj(dataroot, train=True, download=True, transform=transform)
        targets = torch.tensor(train_data.targets)

    min_value = 0.0
    max_value = 0.0
    while least_nums >= min_value:
        n_classes = len(train_data.classes)
        label_distribution = np.random.dirichlet([alpha]*num_clients, n_classes)
    
        data_id = [i for i in range(len(train_data))]
        class_idcs = [np.argwhere(targets[data_id]==y).flatten() for y in range(n_classes)]
        clients_idcs = [[] for _ in range(num_clients)] 

        for c, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
                clients_idcs[i] += [idcs]
        clients_idcs = [np.concatenate(idcs) for idcs in clients_idcs]
        min_value = min([len(client_size) for client_size in clients_idcs])
        if min_value>max_value:
            max_value = min_value

    train_set,test_set,data_len = [],[],[]  
    for _, data_idcs in enumerate(clients_idcs):
        n_test = int(len(data_idcs)*0.25)
        n_train = len(data_idcs) -  n_test
        client_train_data_idcs, client_test_data_idcs = torch.utils.data.random_split(data_idcs, [n_train, n_test])
        data_len.append(len(client_train_data_idcs))
        loader = DataLoader(Subset(train_data, client_train_data_idcs), batch_size=batch_size, shuffle=True)
        train_set.append(loader)
        test_loader = DataLoader(Subset(train_data, client_test_data_idcs), batch_size=batch_size, shuffle=False)
        test_set.append(test_loader)
    return train_set, test_set, data_len


def get_classes(data_name, dataroot, num_clients, batch_size, num_classes_per_client, seed):
    set_seed(seed)
    if data_name =='cifar10':
        normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        data_obj = CIFAR10
        total_classes =10
        class_nums = [5000] * total_classes
    elif data_name == 'cifar100':
        normalization = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        data_obj = CIFAR100
        total_classes =100
        class_nums = [500] * total_classes
    elif data_name == 'tiny_imageNet':
        data_obj = ImageFolder
        total_classes =200
        class_nums = [500] * total_classes
    elif data_name == 'emnist':
        data_obj = EMNIST
        total_classes =62
    else:
        raise ValueError("choose data_name from ['emnist', 'cifar10', 'tiny_imageNet', 'cifar100']")

    if data_name=='tiny_imageNet':
        transform = transforms.Compose([  
            transforms.Resize((32, 32)), 
            transforms.ToTensor(),  
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)), 
        ])  
        train_data = ImageFolder(os.path.join(dataroot, 'tiny-imagenet-200/train'), transform=transform) 
        targets = torch.tensor(train_data.targets)  

    elif(data_name == 'emnist'):
        transform = transforms.Compose([  
            transforms.Resize((32, 32)),  
            transforms.ToTensor(),  
            transforms.Normalize((0.5,), (0.5,))  
        ])
        train_data = data_obj(root=dataroot, split='byclass', train=True, download=True, transform=transform)  
        targets = train_data.targets
        class_nums = torch.bincount(targets)  
    
    else:
        transform =  transforms.Compose([transforms.ToTensor(), normalization])
        train_data = data_obj(
            dataroot,
            train=True,
            download=True,
            transform=transform
        )
        targets = torch.tensor(train_data.targets) 

    client_classes = {}  
    for client_id in range(num_clients):  
        classes = np.random.choice(total_classes, size=num_classes_per_client, replace=False)  
        client_classes[client_id] = classes  

    client_weights = {}  
    for client_id,_ in client_classes.items():
        client_weights[client_id] = {}  
        for class_label in client_classes[client_id]:   
            weight = np.random.uniform(0.4, 0.6)  
            client_weights[client_id][class_label] = weight  

    weights_sum = {id: 0 for id in range(total_classes)}  
    for _, weights in client_weights.items():  
        for key,value in weights.items():
            weights_sum[key] += value

    for _, weights in client_weights.items():  
        for key,value in weights.items():
            weights[key] = (value / weights_sum[key])

    client_label_sum = {id: 0 for id in range(total_classes)}
    for client_id, weights in client_weights.items():  
        for key,value in weights.items():
            weights[key] = int(value * int(class_nums[key]))
            client_label_sum[key] += weights[key]

    flag = [0 for _ in range(total_classes)]
    for key,value in client_label_sum.items():
        if(value==0 or value==int(class_nums[key])):
            pass
        else:
            for _, weights in client_weights.items():  
                for label,value in weights.items():
                    if(key==label and flag[key]==0):
                        weights[label] += (int(class_nums[key]) - client_label_sum[key])
                        flag[key]=1

    label_idcs = {label_id: [] for label_id in range(total_classes)}  
    for label in range(total_classes):
        for j in range(len(targets)):
            if(targets[j]==label):
                label_idcs[label].append(j)
        random.shuffle(label_idcs[label])  

    clients_idcs = []
    idx = {label:0 for label in range(total_classes)}
    for client_id, weights in client_weights.items():
        client_idcs = []
        for key,value in weights.items():
            client_idcs += label_idcs[key][idx[key]:idx[key]+value]
            idx[key]+=value
        print(client_id,'\t',weights)
        clients_idcs.append(client_idcs)
    client_data = [Subset(train_data, idcs) for idcs in clients_idcs] 

    train_set,test_set,data_len = [],[],[]  
    for _, data in enumerate(client_data):
        n_test = int(len(data)*0.25)
        n_train = len(data) -  n_test
    
        client_train_data, client_test_data = torch.utils.data.random_split(data, [n_train, n_test])
        data_len.append(len(client_train_data))

        loader = DataLoader(client_train_data, batch_size=batch_size, shuffle=True)
        train_set.append(loader)

        test_loader = DataLoader(client_test_data, batch_size=batch_size, shuffle=False)
        test_set.append(test_loader)

    return train_set, test_set, data_len









    
    







