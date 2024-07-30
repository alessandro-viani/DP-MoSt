import torch
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from Sigmoid import Sigmoid
from SigmoidGroup import SigmoidGroup
from utility import plot_solution

class DPMoSt(object):
    
    def __init__(self, data=None, device='cpu', log_noise_std=None, prior_mean=None, prior_std=None, 
                 name_biomarkers=None, lambda_reg=None, lambda_reg_noise=None, lambda_reg_theta=None, stopping_criteria=True,
                 n_prints=5, benchmarks=False, verbose=False, time_shift_eval=True, prior_time_shift='gaussian', 
                 name_path='example'):
        
        super(DPMoSt, self).__init__()  # Call the parent constructor
        
        #input parameters
        self.data=data
        
        self.y=torch.tensor(self.data.iloc[:,2:].values, dtype=torch.float32, device=device)
        self.t=torch.tensor(self.data['time'].values, dtype=torch.float32, device=device)
        self.n_samples=self.y.shape[0]
        self.n_features=self.y.shape[1]
        self.subjecs=self.data['subj_id'].unique()
        self.n_subjects=len(self.subjecs)

        self.device=device

        if log_noise_std is None:
            log_noise_std=torch.tensor([torch.log(self.y.std(dim=0)[fdx]).item() for fdx in range(self.n_features)], dtype=torch.float32, device=self.device, requires_grad=True)
        self.log_noise_std=log_noise_std

        if prior_mean is None:
            prior_mean=torch.tensor([10, 0.7, 2], dtype=torch.float32, device=self.device)
        self.prior_mean=prior_mean
        if prior_std is None:
            prior_std=torch.tensor([3.5, 0.2, 0.5], dtype=torch.float32, device=self.device)
        self.prior_std=prior_std

        if name_biomarkers is None:
            name_biomarkers=self.data.columns[2:]
        self.name_biomarkers=name_biomarkers

        if lambda_reg is None:
            lambda_reg=0.15*self.n_subjects
        self.lambda_reg=torch.tensor([lambda_reg], dtype=torch.float32, device=self.device)
        
        if lambda_reg_noise is None:
            lambda_reg_noise=0.15*self.n_samples #0.15*self.n_subjects
        self.lambda_reg_noise=torch.tensor([lambda_reg_noise], dtype=torch.float32, device=self.device)

        if lambda_reg_theta is None:
            lambda_reg_theta=0.001
        self.lambda_reg_theta=torch.tensor([lambda_reg_theta], dtype=torch.float32, device=self.device)

        if prior_time_shift=='gamma':
            self.shape_parameter=1
            self.rate_parameter=0.4

        self.stopping_criteria=stopping_criteria
        self.n_prints=n_prints
        self.benchmarks=benchmarks
        self.name_path=name_path
        if self.benchmarks:
            Path(f'{self.name_path}/fig_benchmarks').mkdir(parents=True, exist_ok=True)
        self.verbose=verbose

        self.time_shift_eval=time_shift_eval
        self.prior_time_shift=prior_time_shift

        if self.verbose:
            print('Initialisation:')
            print(f'    Num samples: {self.n_samples}')
            print(f'    Num subjects: {self.n_subjects}')
            print(f'    Num features: {self.n_features}')
            print(f'    Evaluation time-shift: {self.time_shift_eval}')
            if self.time_shift_eval: print(f'    Prior time-shift: {self.prior_time_shift}')
            print(f'    Lambda regression: {lambda_reg}')
            print(f'    Lambda regression noise: {lambda_reg_noise}')
            print(f'    Lambda regression theta: {lambda_reg_theta}\n')

        #attributes
        self.n_max=2
        if not hasattr(mpl, 'colormaps') or len(mpl.colormaps['tab10'].colors)<self.n_features:
            self.colors=cm.spring(np.linspace(0, 1, self.n_features))
        else:
            self.colors = mpl.colormaps['tab20'].colors
        
        #parameters to be estimated        
        self.xi=None
        self.pi=None
        self.particle=None

        #estimates
        self.est_num=None
        self.est_theta=None
        self.est_subpop=None
        self.est_noise=None

        #performance metrics
        self.time_elapsed=0
        self.all_loss=[]
        self.n_outer_iterations=None
        self.n_inner_iterations=None
        self.lr=None
    
        #initialisation
        self.initialisation()

        self.regression_parameters=[{'params': self.particle[fdx][ndx].sigmoids[sdx].theta} for fdx in range(self.n_features) for ndx in range(self.n_max) for sdx in range(ndx+1)]
        if self.log_noise_std.requires_grad:
            self.regression_parameters.append({'params': self.log_noise_std})


    def get_rate_grouth(self):
        return torch.cat([self.particle[fdx][n_subpop].get_rate_grouth() for fdx in range(self.n_features) for n_subpop in range(2)], dim=0)

    #this function samples from a Gamma distribution with shape parameter alpha and rate parameter beta
    def random_gamma(self, shape, alpha, beta=1.0):
        alpha = torch.ones(shape) * torch.tensor(alpha)
        beta = torch.ones(shape) * torch.tensor(beta)
        gamma_distribution = torch.distributions.gamma.Gamma(alpha, beta)

        return gamma_distribution.sample()
        
    #this function initialise the value for xi and pi setting each value to 0.5, 
    #i.e. assuming no prior on the initial configuration and initialise the particles (sigmoids configurations)
    def initialisation(self):
        self.xi=torch.tensor([0.5 for _ in range(self.n_features)], dtype=torch.float32, device=self.device)
        self.pi=torch.tensor([0.5 for _ in range(self.n_subjects)], dtype=torch.float32, device=self.device)
        #added by ale 24/06/2024
        if not self.time_shift_eval: self.time_shift=torch.tensor([0 for _ in range(self.n_subjects)], dtype=torch.float32, device=self.device, requires_grad=False)
        else: 
            if self.prior_time_shift=='gamma':
                self.time_shift=self.random_gamma(shape=self.n_subjects, alpha=self.shape_parameter, beta=self.rate_parameter).to(self.device).requires_grad_(requires_grad=True)
            if self.prior_time_shift=='gaussian':
                self.time_shift=torch.randn((self.n_subjects), dtype=torch.float32, device=self.device, requires_grad=True)
        
        self.particle=[[SigmoidGroup(number=n_sig+1, 
                                     y=self.y[:,fdx],
                                     t=self.t, 
                                     prior_mean=self.prior_mean, 
                                     prior_std=self.prior_std, 
                                     device=self.device) 
                                     for n_sig in range(self.n_max)] 
                                     for fdx in range(self.n_features)]

    #this function evaluates the sigmoids configurations in the data, 
    #it returns a list containing the all different evaluations
    def eval(self, t):
        return [[self.particle[n_feat][n_sig].eval(t) for n_sig in range(self.n_max)] for n_feat in range(self.n_features)]
      
    
    def aux_log_like(self, particle, y, t, pi, xi, log_noise_std):
        log_like_no_split=particle[0].log_like(y, t, log_noise_std=log_noise_std)[0] + torch.log(xi)
        log_like_split=particle[1].log_like(y, t, log_noise_std=log_noise_std)
        
        log_like_split_1=log_like_split[0] + torch.log(pi) + torch.log(1-xi)
        log_like_split_2=log_like_split[1] + torch.log((1-pi)) + torch.log(1-xi)
        
        x_concat = torch.stack((log_like_no_split, log_like_split_1, log_like_split_2)).unsqueeze(1)
        return torch.logsumexp(x_concat,dim=0)
    

    def log_like_single_subject(self, subj_id):

        y_single_subject= torch.tensor(self.data[self.data['subj_id']==self.subjecs[subj_id]].iloc[:,2:].values, dtype=torch.float32, device=self.device)
        t_single_subject= torch.tensor(self.data[self.data['subj_id']==self.subjecs[subj_id]]['time'].values, dtype=torch.float32, device=self.device)
        if self.prior_time_shift=='gamma':
            l=torch.stack([self.aux_log_like(self.particle[feature], y_single_subject[t_subj,feature], t_single_subject[t_subj]+torch.abs(self.time_shift[subj_id]), 
                                            self.pi[subj_id], self.xi[feature], self.log_noise_std[feature]) 
                        for t_subj in range(len(t_single_subject)) for feature in range(self.n_features)]).sum()
        if self.prior_time_shift=='gaussian':
            l=torch.stack([self.aux_log_like(self.particle[feature], y_single_subject[t_subj,feature], t_single_subject[t_subj]+self.time_shift[subj_id], 
                                            self.pi[subj_id], self.xi[feature], self.log_noise_std[feature]) 
                        for t_subj in range(len(t_single_subject)) for feature in range(self.n_features)]).sum()
        return l

    
    def log_like(self):
        l=torch.stack([self.log_like_single_subject(subj_id) for subj_id in range(self.n_subjects)]).sum()

        return l
    
    #this function evaluates the posterior for the split and no split, it is used for the e-step for xi

    def gamma(self, subj_id, feature):
        #log likelihood for a single subject with no split
        y_single_subject= torch.tensor(self.data[self.data['subj_id']==self.subjecs[subj_id]].iloc[:,2:].values, dtype=torch.float32, device=self.device)
        t_single_subject= torch.tensor(self.data[self.data['subj_id']==self.subjecs[subj_id]]['time'].values, dtype=torch.float32, device=self.device)

        y_single_subject_single_feature=y_single_subject[:,feature]

        like_no_split=torch.exp(self.particle[feature][0].log_like(y_single_subject_single_feature, t_single_subject, self.log_noise_std[feature])[0])

        log_like_split=self.particle[feature][1].log_like(y_single_subject_single_feature, t_single_subject, self.log_noise_std[feature])
        like_split=self.pi[subj_id]*torch.exp(log_like_split[0]) + (1-self.pi[subj_id])*torch.exp(log_like_split[1])

        norm_cost=like_no_split*self.xi[feature] + like_split*(1-self.xi[feature])

        post_no_split=like_no_split*self.xi[feature]/norm_cost

        return post_no_split, 1-post_no_split
    

    def e_step_xi(self):
        gamma_values=torch.tensor([[self.gamma(subj_id, feature) 
                                    for feature in range(self.n_features)] 
                                    for subj_id in range(self.n_subjects)], dtype=torch.float32, device=self.device)
        
        aux_gamma1 = gamma_values[:, :, 0]
        aux_gamma2 = gamma_values[:, :, 1]

        self.xi=aux_gamma1.sum(axis=0)/(aux_gamma1.sum(axis=0)+aux_gamma2.sum(axis=0)+self.lambda_reg*(self.xi-1))

        for fdx in range(self.n_features):
            if self.xi[fdx] > 1 or self.xi[fdx] < 0:
                print('Attention prboability out of range [0,1] ... change the regularisation parameter for avoiding errors.')


    #this function evaluates the posterior for belongin to one or another split, it is used for the e-step for pi
    def chi(self, subj_id, feature):
        y_single_subject= torch.tensor(self.data[self.data['subj_id']==self.subjecs[subj_id]].iloc[:,2:].values, dtype=torch.float32, device=self.device)
        t_single_subject= torch.tensor(self.data[self.data['subj_id']==self.subjecs[subj_id]]['time'].values, dtype=torch.float32, device=self.device)

        y_single_subject_single_feature=y_single_subject[:,feature]

        like_1=torch.exp(self.particle[feature][0].log_like(y_single_subject_single_feature, t_single_subject, self.log_noise_std[feature])[0])
        like_2_0=torch.exp(self.particle[feature][1].log_like(y_single_subject_single_feature, t_single_subject, self.log_noise_std[feature])[0])
        like_2_1=torch.exp(self.particle[feature][1].log_like(y_single_subject_single_feature, t_single_subject, self.log_noise_std[feature])[1])

        chi_1 = like_2_0*self.pi[subj_id]/(like_1*self.xi[feature] + (like_2_0*self.pi[subj_id] + like_2_1*(1-self.pi[subj_id]))*(1-self.xi[feature]))
        chi_2 = like_2_1*(1-self.pi[subj_id])/(like_1*self.xi[feature] + (like_2_0*self.pi[subj_id] + like_2_1*(1-self.pi[subj_id]))*(1-self.xi[feature]))

        return chi_1, chi_2

    #this function performs one e-step for the parameter pi
    def e_step_pi(self):
        aux_chi=torch.tensor([[self.chi(subj_id, feature) for feature in range(self.n_features)] 
                               for subj_id in range(self.n_subjects)], dtype=torch.float32, device=self.device)
        self.pi=(aux_chi[:,:,0]/(aux_chi[:,:,0]+aux_chi[:,:,1])).mean(axis=1)


    def log_prior_noise(self):
        return -self.lambda_reg_noise*torch.sum(self.log_noise_std+1/torch.exp(self.log_noise_std))


    def log_prior_xi(self):
        return -self.lambda_reg*(self.n_features-self.xi.sum())
    

    def log_prior_time_shift(self):
        if self.time_shift_eval:
            if self.prior_time_shift=='gamma':
                log_prior_time_shift=((self.shape_parameter-1)*torch.log(torch.abs(self.time_shift))-self.rate_parameter*torch.abs(self.time_shift)).sum()
            if self.prior_time_shift=='gaussian':
                log_prior_time_shift=-((self.time_shift)**2).sum()
        else:
            log_prior_time_shift = 0

        return log_prior_time_shift


    def log_prior_theta(self, outer_iter):
        rate_grouth=torch.abs(self.get_rate_grouth())
        return -1/self.lambda_reg_theta*torch.sum(torch.square((rate_grouth-3))) if outer_iter<5 and self.lambda_reg_theta>0 else torch.tensor([0], dtype=torch.float32, device=self.device)

    #this function evaluates the loss function for a given configuration as -log_like+constraints
    def loss_eval(self, tdx):
        return -(self.log_like()+self.log_prior_xi()+self.log_prior_noise()+self.log_prior_time_shift()+self.log_prior_theta(tdx))
 
    #this function print benchmarks every 5 iterations, if verbose==True it also display on the workspace, 
    #otherwise it only save the plots in the folder fig_benchmarks
    def print_benchmarks(self, loss, tdx, n_outer_iterations):
        if (tdx+1)%self.n_prints==0:
            if self.verbose:
                print(f'iter {tdx+1}/{n_outer_iterations} -- loss: {loss.item():.4f}')
                for fdx in range(self.n_features):
                    print(f'    P(split {self.name_biomarkers[fdx]}) = {1-self.xi[fdx].cpu().item():.4f}', end=' ')
                if self.log_noise_std.requires_grad:
                    print('')
                    for fdx in range(self.n_features):
                        print(f'    noisestd {self.name_biomarkers[fdx]} = {torch.exp(self.log_noise_std[fdx]).item():.4f}', end=' ') 
            print('\n')
        if self.benchmarks:
            self.estimates()
            show=True if (tdx+1)%self.n_prints==0 and self.verbose else False
            plot_solution(self, show=show, name_path=f'{self.name_path}/fig_benchmarks/dp-most_sol_iter_{tdx}')


    #this function performs the m-step for the optimisation problem
    def optimise(self, n_outer_iterations=30, n_inner_iterations=30, lr=1e-1):
        self.n_outer_iterations=n_outer_iterations
        self.n_inner_iterations=n_inner_iterations
        self.lr=lr
        start=time.time()
        optimizer=torch.optim.Adam(self.regression_parameters, lr=lr)
        #added by ale 24/06/2024
        if self.time_shift.requires_grad: optimizer_time_shift=torch.optim.Adam([self.time_shift], lr=lr)

        for tdx in range(n_outer_iterations):
            self.e_step_xi()
            self.e_step_pi()

            #added by ale 24/06/2024
            if self.time_shift.requires_grad:
                for epoch in range(n_inner_iterations):
                    loss_time_shift=self.loss_eval(tdx)
                    optimizer_time_shift.zero_grad()
                    loss_time_shift.backward()
                    optimizer_time_shift.step()

            for epoch in range(n_inner_iterations):
                loss=self.loss_eval(tdx)
                self.all_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if self.stopping_criteria and epoch>1 and np.abs(self.all_loss[-1]-self.all_loss[-2])/self.all_loss[-2]<1e-3:
                    break

            if self.verbose:
                self.print_benchmarks(loss, tdx, n_outer_iterations)

        self.estimates()
        self.time_elapsed=time.time()-start
        if self.verbose:
            print(f'Elapsed time: {self.time_elapsed:.4f}s')


    #this function estimates the ML configuration
    def estimates(self):
        self.est_num=np.asarray([1 if self.xi[fdx] > 0.5 else 2 for fdx in range(self.n_features)])
        self.est_theta=[[self.particle[fdx][self.est_num[fdx]-1].sigmoids[sdx].theta.cpu().detach()
                        for sdx in range(self.est_num[fdx])]
                        for fdx in range(self.n_features)]
        self.est_subpop=torch.zeros(self.n_samples, dtype=torch.int, device='cpu')
        self.est_time=torch.zeros(self.n_samples, dtype=torch.float32, device='cpu')
        for idx in range(self.n_samples):
            indices = torch.nonzero(torch.tensor(self.subjecs) == self.data['subj_id'].values[idx], as_tuple=True)[0].item()
            self.est_subpop[idx]= 0 if self.pi[indices] > 0.5 and self.est_num.max()>1 else 1
            #added by ale 24/06/2024
            if self.prior_time_shift=='gamma':
                self.est_time[idx]=self.t[idx]+torch.abs(self.time_shift[indices]).cpu().detach()
            if self.prior_time_shift=='gaussian':
                self.est_time[idx]=self.t[idx]+self.time_shift[indices].cpu().detach()
            
        self.est_noise=[torch.exp(self.log_noise_std[fdx].cpu().detach()) for fdx in range(self.n_features)]


    #this function saves the model in the indicated folder
    def save(self, name_file='dpmost_sol'):
        with open(f'{self.name_path}/{name_file}.pkl', 'wb') as f:
            pickle.dump(self, f)