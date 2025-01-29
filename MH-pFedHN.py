import yaml  
import torch
import argparse
import numpy as np
import os,json,random
from tqdm import trange
from collections import defaultdict
from utils import get_args,set_seed
from HyperNetworks import CNNHyper,Hyper
from display import show_plot,ExperimentLogger
from models import CNN,VGG8,ResNet10,ResNet12,ResNet18
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
            for x, y in self.train_loaders[client_id]: 
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

        if self.args.topk:
            with torch.no_grad(): 
                _, index = torch.topk(torch.abs(dW), int(len(dW)*0.3))
                values = dW[index].view(index.shape)  
                sparse_tensor = torch.sparse_coo_tensor(index.unsqueeze(0), values, size=(dW.numel(),), dtype=torch.float32)
                return sparse_tensor
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

        for x, y in trainer.test_loaders[client_id]:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.nets[client_id](x)
            running_loss += self.criteria(pred, y).item()
            running_correct += pred.argmax(1).eq(y).sum().item()
            running_samples += len(y)

        return running_loss/(len(trainer.test_loaders[client_id]) + 1), running_correct, running_samples
    
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
    all_acc = [val['correct'] / val['total'] for val in results.values()]
    all_loss = [val['loss'] for val in results.values()]
    return avg_loss, avg_acc, all_acc, all_loss

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
    elif args.data_name ==  'tiny_imageNet':
        args.classes_per_node = 20
        out_dim = 200
        in_channels=3
    elif args.data_name == 'cifar10':
        args.classes_per_node = 2  
        out_dim = 10    
        in_channels=3
    elif args.data_name == 'emnist':
        args.classes_per_node = 6
        out_dim = 62    
        in_channels=1   
    else:
        raise ValueError("choose data_name from ['emnist', 'cifar10', 'tiny_imageNet' ,'cifar100']")
    
    if args.train_clients == -1:
        train_list = range(args.num_nodes)
    else:
        train_list = range(args.train_clients)

    if not os.path.exists(args.save_dir):  
        os.makedirs(args.save_dir)  
    if args.only_cnn:
        save_title = 'cnn_'+args.data_name+'_'+args.data_distribution+'_cn_'+str(args.num_nodes)+'_tr_'+str(len(train_list))
    else:
        save_title = 'more_model_'+args.data_name+'_'+args.data_distribution+'_cn_'+str(args.num_nodes)+'_tr_'+str(len(train_list))
   
    file_path = os.path.join(args.save_dir, save_title + '.json') 
    if not os.path.exists(file_path):  
        with open(file_path, 'w') as json_file:  
            json_file.write('[]\n')
    
    nets, nets_name, nets_param_nums = [],[],[]

    with open('config/defaults.yaml', 'r') as file:  
        config = yaml.safe_load(file) 

    if args.only_cnn:
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
            nets.append(ResNet12(in_channels,out_dim=out_dim, bias_flag=True).to(device))
            nets_name.append("ResNet12")
            nets_param_nums.append(config[args.data_name]["ResNet12"])
           
        for _ in range(result[4]):
            nets.append(ResNet18(in_channels,out_dim=out_dim, bias_flag=True).to(device))
            nets_name.append("ResNet18")
            nets_param_nums.append(config[args.data_name]["ResNet18"])
            
        hnet = Hyper(nets_param_nums, args.embed_dim, args.hnet_output_size,  args.hidden_layers, args.hidm, norm_var=args.norm_var).to(device)
    
    trainer = LocalTrainer(args, nets, nets_name, device)

    hnet.train()
    hnet_optim = {
        'adam': torch.optim.Adam(params=hnet.parameters(), lr=args.lr),
        'adamw': torch.optim.AdamW(params=hnet.parameters(), lr=args.lr)
    }[args.optim]

    best_acc = -1
    client_stats = ExperimentLogger() 
 
    for step in trange(args.num_steps): 
        idc = random.sample(list(train_list), len(train_list))         

        for client_id in idc:
            weights = hnet(client_id)
            delta = trainer.train(weights, client_id)
            
            if args.topk:
                delta = delta.to_dense()

            hnet_optim.zero_grad()
            hnet_grads = torch.autograd.grad(weights, hnet.parameters(), grad_outputs=delta, allow_unused=True)
            for p, g in zip(hnet.parameters(), hnet_grads):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(hnet.parameters(), args.grad_clip)
            hnet_optim.step()

        avg_loss, avg_acc, all_acc, all_loss = evaluate(hnet, trainer, train_list)
        print(f"Step: {step+1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")
        client_stats.log({"rounds": step,'client_acc':all_acc,'client_loss':all_loss})

        if(avg_acc>best_acc): 
            best_acc = avg_acc
            result = {"Rounds":step, "best_average": best_acc}  
            with open(file_path, 'r+') as json_file:  
                data = json.load(json_file)  
                data.append(result) 
                json_file.seek(0)  
                json.dump(data, json_file, indent=4)  

        png_path = os.path.join(args.save_dir, save_title + '.png') 
        show_plot(client_stats, args.num_steps, png_path) 
    
    if args.save_model:
        hynet_dir = os.path.join(args.save_dir, save_title+'_'+'hynet.pt') 
        fc_dir = os.path.join(args.save_dir, save_title+'_'+'fc.pt') 
        torch.save(hnet.hynet.state_dict(), hynet_dir)  
        torch.save(hnet.fc.state_dict(), fc_dir) 




        
        
       
        




        
 

