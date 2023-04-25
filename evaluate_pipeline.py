import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
from torch import Tensor
import yaml
import random

from src.models.pecnet import PECNet
from src.bo.botorch_funcitons import one_step_BO
from src.data.pecnet import PECNETTrajectoryDataset,pecnet_traj_collate_fn
from src.utils.sample import box_muller_transform
from src.utils.blackbox_function import BlackBoxFunctionPECNET
from src.utils.pecnet import model_forward_post_hook


class ConfigExtractor():
    def __init__(self,file_path) -> None:
        self.file_path = file_path
        with open(file_path, 'r') as stream:
            self.data = yaml.safe_load(stream)
    
    def __str__(self) -> str:
        return str(self.data)

    def __call__(self, ) -> dict:
        return self.data

def compute_batch_metric(pred, gt):
    """Get ADE, FDE, TCC scores for each pedestrian"""
    # Calculate ADEs and FDEs
    temp = (pred - gt).norm(p=2, dim=-1)
    ADEs = temp.mean(dim=1).min(dim=0)[0]
    FDEs = temp[:, -1, :].min(dim=0)[0]
    return ADEs,FDEs            
        
def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)
    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_        

class BayesianEvaluator():
    def __init__(self,model,bbfunction,dataloader,model_name='pecnet',
                 num_of_warmup:int=5,num_of_bo:int=15,bound_factor:float=1.0,
                 config=None,
                 index=0,
                 ) -> None:
        self.model = model
        self.blackbox_function = bbfunction
        self.num_of_warmup= num_of_warmup
        self.num_of_bo = num_of_bo
        self.dataloader = dataloader
        self.bound_factor = bound_factor
        self.model_name = model_name
        self.batchsize = dataloader.batch_size

        self.ades = []
        self.fdes = []
        
        self.ade_cache = []
        self.fde_cache = []
        
        self.config = config
        self.index = index
        
        # self.bb_f_x = []
        # self.bb_f_y = []
    
    def single_step_bo(self,train_x,train_obj,bb_function,bounds):
        next_to_probe = one_step_BO(train_x,train_obj,
                                    bounds=bounds,
                                    max_iter=self.config['bo']['max_iter'][self.index],
                                    acq_factor = self.config['bo']['acq_factor'][self.index],
                                    acq_type=self.config['bo']['acq_type'],
                                    lr=self.config['bo']['lr'][self.index])
        target = bb_function(next_to_probe)
        return next_to_probe,target

    
    def single_step_warmup(self,):
        pass
    
    def unpack_batch(self,batch,pecnet_datascale=170):
        if self.model_name == 'pecnet':
            obs_traj, pred_traj, _, _, mask, _ = [data.cuda(non_blocking=True) for data in batch]
    

            x = obs_traj.permute(0, 2, 1).clone()
            y = pred_traj.permute(0, 2, 1).clone()

            # starting pos is end of past, start of future. scaled down.
            initial_pos = x[:, 7, :].clone() / 1000

            # shift origin and scale data
            origin = x[:, :1, :].clone()
            x -= origin
            y -= origin
            x *= pecnet_datascale  # hyper_params["data_scale"]

            # reshape the data
            x = x.reshape(-1, x.shape[1] * x.shape[2])
            x = x.to(obs_traj.device)
            return obs_traj, pred_traj, mask, x, y, initial_pos, pecnet_datascale

        else:
            raise NotImplementedError
    
    def generate_bounds(self,bound_factor,*args):       
        if self.model_name == 'pecnet':
            sigma=1.3
            bounds = torch.ones((2,16)) * bound_factor
            bounds[0,:] = -bounds[0,:]
            return bounds
        else:
            raise NotImplementedError
        
    def pecnet_evaluate(self):
        self.model.eval()
        with torch.no_grad ():
            for batch in tqdm(self.dataloader):
                ade_cache = []
                fde_cache = []
                            
                # unpack batch
                obs_traj, pred_traj, mask, x, y, initial_pos, pecnet_datascale = self.unpack_batch(batch,data_scale)
                
                y *= data_scale  # hyper_params["data_scale"]
                y = y.cpu().numpy()
                dest = y[:, -1, :]
                
                # sampling by mc/qmc
                if self.config['qmc'] is True:
                    sobol_generator = torch.quasirandom.SobolEngine(dimension=16, scramble=True)
                    loc = box_muller_transform(sobol_generator.draw(self.num_of_warmup).cuda()).unsqueeze(dim=1).expand((self.num_of_warmup, 
                                                                                                                         x.size(0), 16))
                
                
                
                all_dest_recon = []
                
                # get black box function
                bb_function = BlackBoxFunctionPECNET(self.model,initial_pos,obs_traj.device,x)
                num_of_all_ped = x.shape[0]
                # get initial target
                train_x =  torch.zeros((num_of_all_ped,self.num_of_warmup,16)).to(x.device) #(batch_size, num_of_warmup, 16)
                train_obj = torch.zeros((num_of_all_ped,self.num_of_warmup,1)).to(x.device) #(batch_size, num_of_warmup, 1)
                
                for i in range(self.num_of_warmup):
                    if self.config['qmc'] is True:
                        dest_recon = self.model.forward(x, initial_pos,  noise=loc[i], device = obs_traj.device )
                        bb_val = bb_function(loc[i])
                        train_x[:,i,:] = loc[i]
                        train_obj[:,i,0] = bb_val
                    else:
                        noise = torch.randn((x.shape[0],16)).to(obs_traj.device )
                        dest_recon = self.model.forward(x, initial_pos, noise = noise,device = obs_traj.device )
                        bb_val = bb_function(noise)
                        train_x[:,i,:] = noise
                        train_obj[:,i,0] = bb_val
                
                    all_dest_recon.append(dest_recon)
                
                # get bounds
                bound = self.generate_bounds(self.bound_factor).to(x.device)
                
                # bo
                for i in range(self.num_of_bo):
                    next_to_probe,target = self.single_step_bo(train_x,train_obj,bb_function,bound)
                    next_to_probe=next_to_probe.unsqueeze(dim=1) #[num_of_all_ped,1,16]
                    target = target.unsqueeze(dim=1).unsqueeze(dim=1) #[num_of_all_ped,1,1]
                    train_x = torch.cat((train_x,next_to_probe),dim=1)
                    train_obj = torch.cat((train_obj,target),dim=1)
                    dest_recon = self.model.forward(x, initial_pos, noise = next_to_probe.squeeze(dim=1),device = obs_traj.device )
                    all_dest_recon.append(dest_recon)
                
                ades,fdes = model_forward_post_hook(self.model,all_dest_recon,mask,x,y,initial_pos,dest,)
                ade_cache.append(np.array(ades))
                fde_cache.append(np.array(fdes))
                self.ades.append(ade_cache)
                self.fdes.append(fde_cache)

                
  
    def evaluate(self,):
        exp_ades = []
        exp_fdes = []
        self.model.eval()
        for i in range(self.config['eval_times']):
            self.ades = []
            self.fdes = []

            if self.model_name == 'pecnet':
                self.pecnet_evaluate()
                self.ades = np.concatenate(self.ades, axis=1)
                self.fdes = np.concatenate(self.fdes, axis=1)                
            else:
                raise NotImplementedError
            ade = self.ades.mean()
            fde = self.fdes.mean()
            print(ade,fde)
            
            exp_ades.append(ade)
            exp_fdes.append(fde)
        exp_ades = np.array(exp_ades)
        exp_fdes = np.array(exp_fdes)    
        print("[AVG] [ADE] {} [FDE] {}".format(exp_ades.mean(),exp_fdes.mean()))
        return exp_ades.mean(),exp_fdes.mean()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
               
class LoadModel():
    def __init__(self,model_name,model_weight,) -> None:
        self.model_name = model_name
        self.model_weight = model_weight
    def load(self,):
        if self.model_name == 'pecnet':
            def get_hyperparams():
                global data_scale
                with open("./configs/pecnet/optimal_settings.yaml", 'r') as file:
                    hyper_params = yaml.load(file, Loader=yaml.FullLoader)
                    data_scale = hyper_params["data_scale"]
                    return (hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"],
                            hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'],
                            hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"],
                            hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'],
                            hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], False)
            model = PECNet(*get_hyperparams())
            model.load_state_dict(torch.load(self.model_weight))
            model = model.cuda()
            model.eval()
            return model
        else:
            raise NotImplementedError

def load_dataset(model_name,dataset_path,batch_size,obs_len=8,pred_len=12,skip=1,
                 delim='\t',loader_num_workers=20,):    
    if model_name == 'pecnet':
        dset_train = PECNETTrajectoryDataset(dataset_path, obs_len=obs_len, pred_len=pred_len)
        loader_phase = DataLoader(dset_train, batch_size,collate_fn=pecnet_traj_collate_fn, shuffle=False)
        return loader_phase
    else:
        raise NotImplementedError


def main(config):

    datasets = config['datasets']
    model_name = config['model_name']
    print("Baseline model: {}".format(model_name))
    ade_results = []
    fde_results = []
    
    for i,dataset in enumerate(datasets):
        print("*"*100)
        print("Evaluating on dataset {}".format(dataset))
        # load model
        model_path = config['model_path'][i]

        m = LoadModel(model_name=model_name,model_weight=model_path,)
        model = m.load()
        # load data
        data_loader = load_dataset(model_name=model_name,dataset_path=config['dataset_path'][i],
                                batch_size=config['batch_size'][i],
                                obs_len=config['obs_len'],pred_len=config['pred_len'],
                                skip=config['skip'],
                                loader_num_workers=config['loader_num_workers'])
        
        if model_name == 'pecnet':
            evaluator = BayesianEvaluator(model=model,bbfunction=BlackBoxFunctionPECNET,
                                          dataloader=data_loader,model_name=model_name,
                                          num_of_warmup=config['bo']['num_of_warmup'][i],
                                          num_of_bo=config['bo']['num_of_bo'][i],
                                          bound_factor=config['bo']['bound_factor'][i],
                                          config=config,index=i)
            ade,fde = evaluator.evaluate()
            ade_results.append(ade)
            fde_results.append(fde)
        else:
            raise NotImplementedError
    
    print("ADE results {}".format(np.array(ade_results).mean()))
    print("FDE results {}".format(np.array(fde_results).mean()))

    pass
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/pecnet/zara2.yaml')
    args = parser.parse_args()
    configs = ConfigExtractor(args.config)
    
    # exp_configs = ConfigExtractor(args.exp_config)
    
    # comment below if using evaluation script
    torch.cuda.set_device(configs.data['gpu_idx'])
    
    #print(configs)
    main(configs())
    
        