"""
Training and testing util functions
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr 
from tqdm import tqdm
import torch
import eval_metrics
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device = {device}')

class Trainer():
    
    def __init__(self, nnArchitecture, dataloaders, optimizer_lr=0.01):
        self.nnArchitecture = nnArchitecture.to(device)
        self.dataloaders = dataloaders
        
        self.optimizer = torch.optim.Adam(self.nnArchitecture.parameters(), lr=optimizer_lr)
        # reduce the optimizer lr if loss is not decreasing for 5 (patience) epochs
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=4, verbose=True)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        # self.corr_stats = []
        self.loss_dict = {'train': [],
                            'val':[],
                            'test':[]}

    def nnTrain(self, num_epochs, lossFunction=torch.nn.MSELoss(), phases=['train', 'test'], plot_after_epoch=(False, 1)):
        '''
        Args:
            num_epochs: number of training epochs
            lossFunction: choose between 'mse' or 'weighted_mse'
            phases: list (example ['train', 'val', 'test'])
            plot_after_epoch: tuple
                first arg: to plot or not
                second arg: after how many epochs the plot is needed
        '''
        for epoch in tqdm(range(num_epochs)):
            for phase in phases:
                if phase == 'train':
                    self.nnArchitecture.train()  # Set model to training mode
                else:
                    self.nnArchitecture.eval()   # Set model to evaluate mode
                loss_list_epoch = []
                for j, sampled_batch in enumerate(self.dataloaders[phase]):
                    resp, co2 = sampled_batch
                    resp = resp.to(device, dtype=torch.float)
                    co2 = co2.to(device, dtype=torch.float)
                    self.optimizer.zero_grad()            
                    with torch.set_grad_enabled(phase != 'test'):
                        co2_pred = self.nnArchitecture(resp)
                        loss = lossFunction(co2_pred, co2)
                        if phase == 'train':
                            ## backprop
                            loss.backward()
                            self.optimizer.step()
                        loss_list_epoch.append(loss.detach().item())
                epoch_mean_loss = np.mean(loss_list_epoch)
                self.loss_dict[phase].append(epoch_mean_loss)
                if phase=='train':
                    self.scheduler.step(epoch_mean_loss)
                
            if plot_after_epoch[0] and epoch%plot_after_epoch[1]==0:
                co2_pred = np.squeeze(co2_pred.cpu().detach().numpy())
                co2 = np.squeeze(co2.cpu().detach().numpy())
                plt.figure()
                t = 0
                plt.plot(co2_pred[t + 0:t + 1000], label = 'co2 pred')
                plt.plot(co2[t + 0:t+ 1000], linestyle='dashed', label = 'co2_gt')
                plt.title(f'plotting at epoch: {epoch}')
                plt.legend()
                plt.show()
        return self.loss_dict
    
    def nnTrain_peaks(self, num_epochs, lossFunction=torch.nn.MSELoss(), phases=['train', 'test'], plot_after_epoch=(False, 1)):
        '''
        Training with weighted loss function
        Args:
            num_epochs: number of training epochs
            lossFunction: USELESS PARAMETER. I wrote it initially, but forgot to remove. TODO - fix
            phases: list (example ['train', 'val', 'test'])
            plot_after_epoch: tuple
                first arg: to plot or not
                second arg: after how many epochs the plot is needed
        '''
        for epoch in tqdm(range(num_epochs)):
            for phase in phases:
                if phase == 'train':
                    self.nnArchitecture.train()  # Set model to training mode
                else:
                    self.nnArchitecture.eval()   # Set model to evaluate mode
                loss_list_epoch = []
                for j, sampled_batch in enumerate(self.dataloaders[phase]):
                    resp, co2 = sampled_batch
                    # co2_peak_index, peak_amplitude = utils.get_peaks(np.squeeze(co2.numpy()), Fs=10, thres=0.3)
                    # print(f'co2_peak_index: {co2_peak_index}')
                    resp = resp.to(device, dtype=torch.float)
                    co2 = co2.to(device, dtype=torch.float)
                    self.optimizer.zero_grad()            
                    with torch.set_grad_enabled(phase != 'test'):
                        co2_pred = self.nnArchitecture(resp)
                        # print(co2_pred[...,co2_peak_index])
                        # print(co2[...,co2_peak_index])
                        # loss = lossFunction(co2_pred, co2) + 0.5*lossFunction(co2_pred[...,co2_peak_index], co2[...,co2_peak_index])
                        # loss = lossFunction(co2_pred, co2)
                        # loss = torch.mean((co2**2)*((co2_pred - co2)**2)) + lossFunction(co2_pred[...,co2_peak_index], co2[...,co2_peak_index])
                        loss = torch.mean(torch.abs(co2 - torch.mean(co2))*((co2_pred - co2)**2))
                        if phase == 'train':
                            ## backprop
                            loss.backward()
                            self.optimizer.step()
                        loss_list_epoch.append(loss.detach().item())
                epoch_mean_loss = np.mean(loss_list_epoch)
                self.loss_dict[phase].append(epoch_mean_loss)
                if phase=='train':
                    self.scheduler.step(epoch_mean_loss)
                
            if plot_after_epoch[0] and epoch%plot_after_epoch[1]==0:
                co2_pred = np.squeeze(co2_pred.cpu().detach().numpy())
                co2 = np.squeeze(co2.cpu().detach().numpy())
                plt.figure()
                t = 0
                plt.plot(co2_pred[t + 0:t + 1000], label = 'co2 pred')
                plt.plot(co2[t + 0:t+ 1000], linestyle='dashed', label = 'co2_gt')
                plt.title(f'plotting at epoch: {epoch}')
                plt.legend()
                plt.show()
        return self.loss_dict
    
    def nnTest(self, phases=['test'], plots=False, output_smoothing = False, output_stdnorm = False):
        self.nnArchitecture.eval()
        corr_stats = {'corr_co2':[], 
                      'corr_petco2':[],
                      'z_co2': [],
                      'z_petco2': [],
                      'mse_co2':[],
                      'mse_petco2':[],
                      'mae_co2':[],
                      'mae_petco2':[],
                      'mape_co2':[],
                      'mape_petco2':[],
                      }
        for phase in phases:
            for j, sampled_batch in enumerate(self.dataloaders[phase]):
                resp, co2 = sampled_batch
                resp = resp.to(device, dtype=torch.float)
                co2 = co2.to(device, dtype=torch.float)
                with torch.set_grad_enabled(False):
                    co2_pred = self.nnArchitecture(resp)
                    
                resp = np.squeeze(resp.cpu().detach().numpy())
                co2 = np.squeeze(co2.cpu().detach().numpy())
                co2_pred = np.squeeze(co2_pred.cpu().detach().numpy())

                if output_smoothing:
                    # co2 = utils.moving_average(co2, w = 7, mode = 'same')
                    co2_pred = utils.moving_average(co2_pred, w = 7, mode = 'same')
                    co2, co2_pred, _ = utils.delay_correction(co2, co2_pred, negative_relationship=False) # because moving avg shifts the signal

                petco2_gt = utils.get_petco2_interpolated(co2, Fs=10, len_interpolated=len(co2))
                petco2_pred = utils.get_petco2_interpolated(co2_pred, Fs=10, len_interpolated=len(co2))
                if output_stdnorm:
                    petco2_pred = utils.std_normalise(petco2_pred)*np.std(petco2_gt) + np.mean(petco2_gt)

                corr_stats['corr_co2'].append(pearsonr(co2, co2_pred)[0])
                corr_stats['corr_petco2'].append(pearsonr(petco2_gt, petco2_pred)[0])
                corr_stats['z_co2'].append(utils.get_z_score(pearsonr(co2, co2_pred)[0], len(co2)))
                corr_stats['z_petco2'].append(utils.get_z_score(pearsonr(petco2_gt, petco2_pred)[0], len(petco2_gt)))
                corr_stats['mse_co2'].append(eval_metrics.mse(co2, co2_pred))
                corr_stats['mse_petco2'].append(eval_metrics.mse(petco2_gt, petco2_pred))
                corr_stats['mae_co2'].append(eval_metrics.mae(co2, co2_pred))
                corr_stats['mae_petco2'].append(eval_metrics.mae(petco2_gt, petco2_pred))
                corr_stats['mape_co2'].append(eval_metrics.mape(co2, co2_pred))
                corr_stats['mape_petco2'].append(eval_metrics.mape(petco2_gt, petco2_pred))
                
                if plots:
                    plt.figure()
                    t = 1000
                    plt.plot(co2_pred[t + 0:t + 1000], label = 'co2 pred')
                    plt.plot(petco2_pred[t + 0:t+ 1000], label = 'petco2_pred')
                    plt.plot(co2[t + 0:t+ 1000], linestyle='dashed', label = 'co2_gt')
                    plt.plot(petco2_gt[t + 0:t+ 1000], linestyle='dashed', label = 'petco2_gt')
                    plt.plot(-resp[t + 0:t + 1000], label = 'resp gt')
                    plt.legend()
                    plt.title(f"{j}; corr_co2: {corr_stats['corr_co2'][-1]:0.3f}, corr_petco2: {corr_stats['corr_petco2'][-1]:0.3f}")
                    plt.show()
        return corr_stats
                    
    def save_model(self, save_path):
        torch.save({'model_state_dict': self.nnArchitecture.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss_dict': self.loss_dict,
            }
            , save_path)
        print('Saved the model successfully!')

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        self.nnArchitecture.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.loss_dict = checkpoint['loss_dict']
        print('Loaded the model successfully!')