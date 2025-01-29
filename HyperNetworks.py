import math  
import torch
from torch import nn

class CNNHyper(nn.Module):
    def __init__(self, clients_message, embedding_dim, hnet_output_size, hidden_layers, hidm, norm_var=0.002):
        super().__init__()
        self.clients_message = clients_message
        self.hnet_output_size = hnet_output_size
        self.embeddings = nn.ParameterList([nn.Parameter(torch.normal(0, norm_var, (math.ceil(parameter_nums/self.hnet_output_size), embedding_dim)))
                                            for parameter_nums in self.clients_message])
        layers = []  
        layers.append(nn.Linear(embedding_dim, hidm))  
        layers.append(nn.ReLU(inplace=True))  
        for _ in range(hidden_layers - 1): 
            layers.append(nn.Linear(hidm, hidm))  
            layers.append(nn.ReLU(inplace=True))  
        self.hynet = nn.Sequential(*layers)  

        self.fc = nn.ModuleList(nn.Linear(hidm,self.hnet_output_size)  for _ in range(math.ceil(self.clients_message[0]/self.hnet_output_size)))
        
    def forward(self,idx): 
        embed_params = self.embeddings[idx]
        ft = self.hynet(embed_params)
        client_param = []
        for i in range(math.ceil(self.clients_message[idx]/self.hnet_output_size)):
            client_param.append(self.fc[i](ft[i]))
        client_param = torch.cat(client_param,dim=0)
        return client_param[0:self.clients_message[idx]]
    
class Hyper(nn.Module):
    def __init__(self, clients_message, embedding_dim, hnet_output_size, hidden_layers, hidm, norm_var=0.002):
        super().__init__()
        self.hnet_output_size = hnet_output_size
        self.clients_message = clients_message
        self.embeddings = nn.ParameterList([nn.Parameter(torch.normal(0, norm_var, (math.ceil(parameter_nums/self.hnet_output_size), embedding_dim)))
                                            for parameter_nums in self.clients_message])
        layers = []  
        layers.append(nn.Linear(embedding_dim, hidm))  
        layers.append(nn.ReLU(inplace=True))  
        for _ in range(hidden_layers - 1): 
            layers.append(nn.Linear(hidm, hidm))  
            layers.append(nn.ReLU(inplace=True))  
        self.hynet = nn.Sequential(*layers)  

        self.fc = nn.ModuleList()

        result = [math.ceil(x /self.hnet_output_size) for x in self.clients_message]  

        for parameter_nums in list(dict.fromkeys(result)) :
            self.fc.append(nn.ModuleList(nn.Linear(hidm,self.hnet_output_size)  for _ in range(parameter_nums)))  
        
        fc_index = {value: index for index, value in enumerate(sorted(set(result)))}  
  
        self.idx_to_fc = {idx: fc_index[value] for idx, value in enumerate(result)}  
      
           
    def forward(self, idx):
        embed_params = self.embeddings[idx]
        ft = self.hynet(embed_params)
        client_param = []
        for i in range(math.ceil(self.clients_message[idx]/self.hnet_output_size)):
            client_param.append(self.fc[self.idx_to_fc[idx]][i](ft[i]))
        client_param = torch.cat(client_param,dim=0)
        return client_param[0:self.clients_message[idx]] 


      
        
    




        
      
