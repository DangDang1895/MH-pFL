import yaml  
import torch
import argparse
import numpy as np
import os,json,random
from tqdm import trange
from torchvision import models  
from collections import defaultdict
from utils import get_args,set_seed
from HyperNetworks import CNNHyper,Hyper
from models import CNN,VGG8,ResNet10,ResNet18,MLP
from dataset import get_classes, dirichlet_distribution

def subtract_(W, W_old):
    dW = {key : None for key in W_old}
    for name in W_old:
        dW[name] = W[name]-W_old[name]
    return dW

class LocalTrainer:
    def __init__(self, args, nets, nets_name, device):

        if args.data_distribution == 'incomplete_label':
            self.train_loaders, self.test_loaders, self.data_len = get_classes(
                args.data_name, args.data_path, args.num_nodes, args.batch_size, args.classes_per_node, args.seed)
        else:
            self.train_loaders, self.test_loaders, self.data_len = dirichlet_distribution(
                args.data_name, args.data_path, args.num_nodes, args.batch_size, args.seed, args.least_nums)
            
        self.device = device
        self.args = args
        self.nets = nets
        self.criteria = torch.nn.CrossEntropyLoss()

        with open('config/All_Layers.json', 'r') as f:
            self.layer_list = json.load(f) 
        self.personalized_layer_name = nets_name

    def train(self, weights, client_id):

        client_weight = {}
        idx = 0
        for key,value in self.nets[client_id].named_parameters():
            if key in self.layer_list[self.personalized_layer_name[client_id]]:
                client_weight[key]=weights[idx:idx+value.numel()].reshape_as(value)
                idx += value.numel()
        self.nets[client_id].load_state_dict(client_weight, strict=False)
        self.nets[client_id].train()
        optimizer = torch.optim.SGD(self.nets[client_id].parameters(), lr=self.args.inner_lr, momentum=.9, weight_decay=self.args.inner_wd)

        for _ in range(self.args.inner_steps):
            for x, y in self.train_loaders[client_id + (self.args.num_nodes - self.args.test_clients)]: 
                x, y = x.to(device), y.to(device)
                pred = self.nets[client_id](x)
                loss = self.criteria(pred, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nets[client_id].parameters(), 50)
                optimizer.step()

        final_state = self.nets[client_id].state_dict()
        final_state_weight = torch.cat([(final_state[k]).view(-1) for k in  self.layer_list[self.personalized_layer_name[client_id]]],dim=0)
        dW = weights - final_state_weight
        return dW
    
    @torch.no_grad()
    def evalute(self, weights, client_id):
        running_loss, running_correct, running_samples = 0., 0., 0.
        client_weights = {}
        idx = 0
        for key,value in self.nets[client_id].named_parameters():
            if key in self.layer_list[self.personalized_layer_name[client_id]]:
                client_weights[key]=weights[idx:idx+value.numel()].reshape_as(value)
                idx += value.numel()
        self.nets[client_id].load_state_dict(client_weights,strict=False)
        self.nets[client_id].eval()

        for x, y in trainer.test_loaders[client_id + (self.args.num_nodes - self.args.test_clients)]:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.nets[client_id](x)
            running_loss += self.criteria(pred, y).item()
            running_correct += pred.argmax(1).eq(y).sum().item()
            running_samples += len(y)

        return running_loss/(len(trainer.test_loaders[client_id + (self.args.num_nodes - self.args.test_clients)]) + 1), running_correct, running_samples

def evaluate(hnet, trainer, clients):
    results = defaultdict(lambda: defaultdict(list))
    for client_id in clients:
        hnet.eval()
        weights = hnet(client_id)
        running_loss, running_correct, running_samples = trainer.evalute(weights, client_id)
        results[client_id]['loss'] = running_loss
        results[client_id]['correct'] = running_correct
        results[client_id]['total'] = running_samples
    
    total_correct = sum([val['correct'] for val in results.values()])
    total_samples = sum([val['total'] for val in results.values()])
    avg_loss = np.mean([val['loss'] for val in results.values()])
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = get_args(parser)
    set_seed(args.seed)
    if args.cuda == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:%d" % args.cuda)
    if args.data_name == 'cifar100':
        args.classes_per_node = 10
        out_dim = 100
        in_channels=3
    else:    
        raise ValueError("choose cifar100")
    
    if args.test_clients == -1:
        raise ValueError("Please choose test clients")
    train_list = range(args.test_clients)

    if not os.path.exists(args.save_dir):  
        os.makedirs(args.save_dir)  
    if args.only_cnn:
        save_title = 'cnn_'+args.data_name+'_'+'cn_'+str(args.num_nodes)+'_test_'+str(len(train_list))
    else:
        save_title = 'more_model_'+args.data_name+'_'+'cn_'+str(args.num_nodes)+'_test_'+str(len(train_list))
    
    file_path = os.path.join(args.save_dir, save_title + '.json') 
    if not os.path.exists(file_path):  
        with open(file_path, 'w') as json_file:  
            json_file.write('[]\n')
    
    nets, nets_name, nets_param_nums = [],[],[]
    with open('config/defaults.yaml', 'r') as file:  
        config = yaml.safe_load(file) 

    if args.test_more_model:
        model_nums = 2
        basic_value = len(train_list) // model_nums
        remainder = len(train_list)  % model_nums
        result = [basic_value] * model_nums
        for i in range(remainder):  
            result[i] += 1  
        for _ in range(result[0]):
            nets.append(ResNet18(in_channels,out_dim=out_dim, bias_flag=True).to(device))
            nets_name.append("ResNet18")
            nets_param_nums.append(config[args.data_name]["ResNet18"])
        for _ in range(result[1]):
            nets.append(models.squeezenet1_1(weights=None,num_classes=100).to(device))
            nets_name.append("squeezenet1_1")
            nets_param_nums.append(config[args.data_name]["squeezenet1_1"])
        hnet = Hyper(nets_param_nums, args.embed_dim, args.hnet_output_size,  args.hidden_layers, args.hidm, norm_var=args.norm_var).to(device)
    elif args.only_cnn:
        for _ in train_list:
            nets.append(CNN(in_channels,16,out_dim=out_dim, bias_flag=True).to(device))
            nets_name.append("CNN")
            nets_param_nums.append(config[args.data_name]["CNN"])
        hnet = CNNHyper(nets_param_nums, args.embed_dim, args.hnet_output_size,  args.hidden_layers, args.hidm, norm_var=args.norm_var).to(device)
    else:
        model_nums = 5
        basic_value = len(train_list) // model_nums
        remainder = len(train_list)  % model_nums
        result = [basic_value] * model_nums
        for i in range(remainder):  
            result[i] += 1  
        for _ in range(result[0]):
            nets.append(CNN(in_channels,out_channels=16,out_dim=out_dim, bias_flag=True).to(device))
            nets_name.append("CNN")
            nets_param_nums.append(config[args.data_name]["CNN"])
        for _ in range(result[1]):
            nets.append(VGG8(in_channels,16,out_dim=out_dim, bias_flag=True).to(device))
            nets_name.append("VGG8")
            nets_param_nums.append(config[args.data_name]["VGG8"])
        for _ in range(result[2]):
            nets.append(ResNet10(in_channels,out_dim=out_dim, bias_flag=True).to(device))
            nets_name.append("ResNet10")
            nets_param_nums.append(config[args.data_name]["ResNet10"])
        for _ in range(result[3]):
            nets.append(MLP(3,100,True).to(device))
            nets_name.append("MLP")
            nets_param_nums.append(config[args.data_name]["MLP"])
        for _ in range(result[4]):
            nets.append(models.squeezenet1_0(weights=None,num_classes=100).to(device))
            nets_name.append("squeezenet1_0")
            nets_param_nums.append(config[args.data_name]["squeezenet1_0"])
            
        hnet = Hyper(nets_param_nums, args.embed_dim, args.hnet_output_size,  args.hidden_layers, args.hidm, norm_var=args.norm_var).to(device)
    
    if args.test_more_model:
        hnet.hynet.load_state_dict(torch.load(args.hynet_dir,weights_only=True,map_location=torch.device('cpu')))
    else:
        hnet.hynet.load_state_dict(torch.load(args.hynet_dir,weights_only=True,map_location=torch.device('cpu')))
        hnet.fc.load_state_dict(torch.load(args.fc_dir,weights_only=True,map_location=torch.device('cpu')))
    trainer = LocalTrainer(args, nets, nets_name, device)

    hnet.train()
    if args.test_more_model:
        params_to_optimize = list(hnet.embeddings.parameters()) + list(hnet.fc.parameters())
        optimizers = {
            'adam': torch.optim.Adam(params=params_to_optimize, lr=args.lr),
            'adamw': torch.optim.AdamW(params=params_to_optimize, lr=args.lr)
        }
    else:
        optimizers = {
            'adam': torch.optim.Adam(params=hnet.embeddings.parameters(), lr=args.lr),
            'adamw': torch.optim.AdamW(params=hnet.embeddings.parameters(), lr=args.lr)
        }
    
    optimizer = optimizers[args.optim]

    for step in trange(args.num_steps): 
        idc = random.sample(train_list, len(train_list))    
        for client_id in idc:
            weights = hnet(client_id)
            delta = trainer.train(weights, client_id)
          
            optimizer.zero_grad()
            hnet_grads = torch.autograd.grad(weights, hnet.parameters(), grad_outputs=delta, allow_unused=True)
            for p, g in zip(hnet.parameters(), hnet_grads):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(hnet.parameters(), args.grad_clip)
            optimizer.step()

        avg_loss, avg_acc = evaluate(hnet, trainer, train_list)
        print(f"Step: {step+1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

        result = {"Rounds":step, "best_average":avg_acc}  
        with open(file_path, 'r+') as json_file:  
            data = json.load(json_file)  
            data.append(result) 
            json_file.seek(0)  
            json.dump(data, json_file, indent=4)  
    




        
        
       
        




        
 

