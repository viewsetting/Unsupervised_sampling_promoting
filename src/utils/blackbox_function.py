import torch
import torch.nn as nn
    
class BlackBoxFunctionPECNET():
    def __init__(self, model,initial_pos,device,x):
        self.model = model
        self.initial_pos = initial_pos
        self.device = device
        self.x = x
        self.dest_recon_label = self.model(self.x, self.initial_pos,  noise=torch.zeros((x.size(0), 16)).to(device), device = self.device) # [batchsize,2]

    
    def __call__(self, noise_recon):
        dest_recon = self.model(self.x, self.initial_pos,  noise=noise_recon, device = self.device) # [batchsize,2]
        return torch.norm(dest_recon - self.dest_recon_label, dim=-1)