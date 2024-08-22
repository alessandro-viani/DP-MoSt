import torch
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import torch.nn as nn

from pathlib import Path
from utility import plot_solution, sigmoid_eval

class DPMoSt(object):
    
    def __init__(self, data=None, device='cpu', 
                 log_noise_std=None, 
                 prior_mean=None, prior_std=None, 
                 name_biomarkers=None, 
                 lambda_reg=None, 
                 lambda_reg_noise=None, 
                 lambda_reg_theta=None, 
                 stopping_criteria=True,
                 n_prints=5, 
                 benchmarks=False, 
                 verbose=False, 
                 time_shift_eval=True, 
                 noise_std_eval=True, 
                 theta_eval=True, 
                 xi_eval=True, 
                 pi_eval=True):
        
        super(DPMoSt, self).__init__()  # Call the parent constructor
        
        #input parameters
        self.data=data
        
        self.y=torch.tensor(self.data.iloc[:,2:].values, dtype=torch.float32, device=device)
        self.t=torch.tensor(self.data['time'].values, dtype=torch.float32, device=device)
        self.n_samples=self.y.shape[0]
        self.n_features=self.y.shape[1]
        self.subjects=self.data['subj_id'].unique()
        self.n_subjects=len(self.subjects)

        self.device=device

        if log_noise_std is None:
            log_noise_std=torch.tensor([torch.log(self.y[:,fdx][~torch.isnan(self.y[:,fdx])].std(dim=0)).item() for fdx in range(self.n_features)], dtype=torch.float32, device=self.device, requires_grad=noise_std_eval)
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
            lambda_reg_noise=0.15*self.n_samples
        self.lambda_reg_noise=torch.tensor([lambda_reg_noise], dtype=torch.float32, device=self.device)

        if lambda_reg_theta is None:
            lambda_reg_theta=0.001
        self.lambda_reg_theta=torch.tensor([lambda_reg_theta], dtype=torch.float32, device=self.device)

        self.stopping_criteria=stopping_criteria
        self.n_prints=n_prints
        self.benchmarks=benchmarks
        
        if self.benchmarks: Path('./fig_benchmarks/').mkdir(parents=True, exist_ok=True)
        self.verbose=verbose

        self.time_shift_eval=time_shift_eval
        self.noise_std_eval=noise_std_eval
        self.theta_eval=theta_eval
        self.xi_eval=xi_eval
        self.pi_eval=pi_eval

        if self.verbose:
            print('Initialisation:')
            print(f'    Num samples: {self.n_samples}')
            print(f'    Num subjects: {self.n_subjects}')
            print(f'    Num features: {self.n_features}')
            print(f'    Evaluation time-shift: {self.time_shift_eval}')
            print(f'    Lambda regression: {lambda_reg}')
            print(f'    Lambda regression noise: {lambda_reg_noise}')
            print(f'    Lambda regression theta: {lambda_reg_theta}\n')

        #attributes
        if not hasattr(mpl, 'colormaps') or len(mpl.colormaps['tab10'].colors)<self.n_features:
            self.colors=cm.spring(np.linspace(0, 1, self.n_features))
        else:
            self.colors = mpl.colormaps['tab20'].colors
        
        #parameters to be estimated        
        self.xi=None
        self.pi=None
        self.theta=None

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
        self.loss=torch.nn.GaussianNLLLoss(full=False, reduction='mean')
    
        #initialisation
        self.initialisation()

        self.regression_parameters=[]
        if self.theta_eval:
            for fdx in range(self.n_features):
                for ndx in range(2):
                    for sdx in range(ndx+1):
                        self.regression_parameters.append({'params': self.theta[f'{self.name_biomarkers[fdx]}_{ndx}_split'][sdx]})
        if self.log_noise_std.requires_grad:
            self.regression_parameters.append({'params': self.log_noise_std})


    #this function initialise the value for xi and pi setting each value to 0.5, 
    #i.e. assuming no prior on the initial configuration and initialise the particles (sigmoids configurations)
    def initialisation(self, epochs=1000, lr=1e-3):
        self.xi=torch.tensor([0.5 for _ in range(self.n_features)], dtype=torch.float32, device=self.device)
        self.pi=torch.tensor([0.5 for _ in range(self.n_subjects)], dtype=torch.float32, device=self.device)

        self.theta={}
        for fdx in range(self.n_features):
            if self.y[:,fdx] is None:
                self.theta.update({f'{self.name_biomarkers[fdx]}_0_split': [torch.tensor([(self.prior_mean[tdx].cpu()+self.prior_std[tdx].cpu()*torch.randn((1))).item() for tdx in range(len(self.prior_mean))], dtype=torch.float32, device=self.device, requires_grad=self.theta_eval)]})
                self.theta.update({f'{self.name_biomarkers[fdx]}_1_split': [torch.tensor([(self.prior_mean[tdx].cpu()+self.prior_std[tdx].cpu()*torch.randn((1))).item() for tdx in range(len(self.prior_mean))], dtype=torch.float32, device=self.device, requires_grad=self.theta_eval) for _ in range(2)]})

            else:
                theta_aux=torch.tensor([(self.prior_mean[tdx].cpu()+self.prior_std[tdx].cpu()*torch.randn((1))).item() for tdx in range(len(self.prior_mean))], dtype=torch.float32, device=self.device, requires_grad=True)
                outputs=torch.zeros(self.n_samples, dtype=torch.float32, device=self.device)

                criterion=nn.MSELoss()
                optimizer=torch.optim.Adam([theta_aux], lr=lr)

                targets=self.y[:,fdx][~torch.isnan(self.y[:,fdx])]
                t_aux=self.t[~torch.isnan(self.y[:,fdx])]

                for epoch in range(epochs):
                    outputs = torch.cat([theta_aux[2]/(1+torch.exp(-torch.abs(theta_aux[1])*(t_aux-theta_aux[0])))]).flatten()
                    loss = criterion(outputs, targets.flatten())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if self.theta_eval:
                    self.theta.update({f'{self.name_biomarkers[fdx]}_0_split': [theta_aux.detach().requires_grad_()]})
                    self.theta.update({f'{self.name_biomarkers[fdx]}_1_split': [(theta_aux.detach()+torch.randn((1), dtype=torch.float32, device=self.device)).requires_grad_() for _ in range(2)]})
                else:
                    self.theta.update({f'{self.name_biomarkers[fdx]}_0_split': [theta_aux.detach()]})
                    self.theta.update({f'{self.name_biomarkers[fdx]}_1_split': [(theta_aux.detach()+torch.randn((1), dtype=torch.float32, device=self.device)) for _ in range(2)]})

            
        if not self.time_shift_eval: self.time_shift=torch.tensor([0 for _ in range(self.n_subjects)], dtype=torch.float32, device=self.device, requires_grad=False)
        else: self.time_shift=torch.randn((self.n_subjects), dtype=torch.float32, device=self.device, requires_grad=True)

    def log_like(self):
        l_new = 0
        log_noise_std_exp = torch.exp(self.log_noise_std)  # Pre-compute exponentiated noise std
        xi_log = torch.log(self.xi)
        one_minus_xi_log = torch.log(1 - self.xi)
        
        for subj_id in range(self.n_subjects):
            subj_indices = self.data.index[self.data['subj_id'] == self.subjects[subj_id]]
            y_single_subject = self.y[subj_indices]
            t_single_subject = self.t[subj_indices]+self.time_shift[subj_id]
            
            pi_log = torch.log(self.pi[subj_id])
            one_minus_pi_log = torch.log(1 - self.pi[subj_id])
            
            for feature in range(self.n_features):
                mask = ~torch.isnan(y_single_subject[:, feature])
                y_aux = y_single_subject[:, feature][mask]
                t_aux = t_single_subject[mask]
                
                ll_no_split = -self.loss(sigmoid_eval(t_aux, self.theta[f'{self.name_biomarkers[feature]}_{0}_split'][0]), y_aux, log_noise_std_exp[feature]) + xi_log[feature]
                ll_split1 = -self.loss(sigmoid_eval(t_aux, self.theta[f'{self.name_biomarkers[feature]}_{1}_split'][0]), y_aux, log_noise_std_exp[feature]) + pi_log + one_minus_xi_log[feature]
                ll_split2 = -self.loss(sigmoid_eval(t_aux, self.theta[f'{self.name_biomarkers[feature]}_{1}_split'][1]), y_aux, log_noise_std_exp[feature]) + one_minus_pi_log + one_minus_xi_log[feature]

                l_new += torch.logsumexp(torch.stack((ll_no_split, ll_split1, ll_split2), dim=0), dim=0).sum()
        
        return l_new


    def log_like_old(self):
        # Pre-compute constants
        log_noise_std_exp = torch.exp(self.log_noise_std)  # Shape: (n_features,)
        xi_log = torch.log(self.xi)  # Shape: (n_features,)
        one_minus_xi_log = torch.log1p(-self.xi)  # More stable computation

        # Prepare data mappings
        subj_to_indices = {subj: self.data.index[self.data['subj_id'] == subj] for subj in self.subjects}
        pi_log = torch.log(self.pi)  # Shape: (n_subjects,)
        one_minus_pi_log = torch.log1p(-self.pi)  # Shape: (n_subjects,)

        total_log_likelihood = 0

        for subj_idx, subj in enumerate(self.subjects):
            indices = subj_to_indices[subj]
            y_subj = self.y[indices]  # Shape: (n_timepoints, n_features)
            t_subj = self.t[indices] + self.time_shift[subj_idx]  # Shape: (n_timepoints,)

            # Create masks for valid observations across all features
            #valid_mask = ~torch.isnan(y_subj)  # Shape: (n_timepoints, n_features)

            # Expand time for broadcasting
            t_expanded = t_subj.unsqueeze(1).expand_as(y_subj)  # Shape: (n_timepoints, n_features)

            # Prepare theta parameters
            theta_no_split = torch.stack([self.theta[f'{self.name_biomarkers[feat]}_0_split'][0] for feat in range(self.n_features)])  # Shape: (n_features, theta_dim)
            theta_split1 = torch.stack([self.theta[f'{self.name_biomarkers[feat]}_1_split'][0] for feat in range(self.n_features)])  # Shape: (n_features, theta_dim)
            theta_split2 = torch.stack([self.theta[f'{self.name_biomarkers[feat]}_1_split'][1] for feat in range(self.n_features)])  # Shape: (n_features, theta_dim)

            # Evaluate sigmoid functions
            sigmoid_no_split = sigmoid_eval(t_expanded, theta_no_split)  # Shape: (n_timepoints, n_features)
            sigmoid_split1 = sigmoid_eval(t_expanded, theta_split1)
            sigmoid_split2 = sigmoid_eval(t_expanded, theta_split2)

            # Compute loss terms
            print(y_subj.shape)
            print(sigmoid_no_split.shape)
            print(xi_log.shape)
            loss_no_split = -self.loss(sigmoid_no_split, y_subj, log_noise_std_exp) + xi_log  # Shape: (n_timepoints, n_features)
            loss_split1 = -self.loss(sigmoid_split1, y_subj, log_noise_std_exp) + pi_log[subj_idx] + one_minus_xi_log
            loss_split2 = -self.loss(sigmoid_split2, y_subj, log_noise_std_exp) + one_minus_pi_log[subj_idx] + one_minus_xi_log

            # Stack and compute logsumexp across splits
            loss_stack = torch.stack([loss_no_split, loss_split1, loss_split2], dim=0)  # Shape: (3, n_timepoints, n_features)
            logsumexp_loss = torch.logsumexp(loss_stack, dim=0)  # Shape: (n_timepoints, n_features)

            # Apply valid mask and sum over all observations
            total_log_likelihood += logsumexp_loss.sum()

        return total_log_likelihood


    #this function evaluates the posterior for the split and no split, it is used for the e-step for xi

    def gamma(self, subj_id, feature):
        #log likelihood for a single subject with no split
        subj_indices = self.data.index[self.data['subj_id'] == self.subjects[subj_id]]
        y_single_subject = self.y[subj_indices]
        t_single_subject = self.t[subj_indices]+self.time_shift[subj_id]
            
        y_single_subject_single_feature=y_single_subject[:,feature]

        theta_no_split=self.theta[f'{self.name_biomarkers[feature]}_{0}_split'][0]
        output=sigmoid_eval(t_single_subject, theta_no_split)

        like_no_split=torch.exp(-self.loss(output, y_single_subject_single_feature, torch.exp(self.log_noise_std[feature])))
        
        
        theta_split=self.theta[f'{self.name_biomarkers[feature]}_{1}_split']
        
        theta_split_1=theta_split[0]
        theta_split_2=theta_split[1]

        output_split_1=sigmoid_eval(t_single_subject, theta_split_1)
        output_split_2=sigmoid_eval(t_single_subject, theta_split_2)
        
        log_like_split_1=-self.loss(output_split_1, y_single_subject_single_feature, torch.exp(self.log_noise_std[feature]))
        log_like_split_2=-self.loss(output_split_2, y_single_subject_single_feature, torch.exp(self.log_noise_std[feature]))
                            
        like_split=self.pi[subj_id]*torch.exp(log_like_split_1) + (1-self.pi[subj_id])*torch.exp(log_like_split_2)

        norm_cost=like_no_split*self.xi[feature] + like_split*(1-self.xi[feature])

        post_no_split=like_no_split*self.xi[feature]/norm_cost

        return post_no_split, 1-post_no_split
    

    def e_step_xi(self):
        gamma_values=torch.tensor([[self.gamma(subj_id, feature) for feature in range(self.n_features)] for subj_id in range(self.n_subjects)], dtype=torch.float32, device=self.device)
        
        aux_gamma1 = gamma_values[:, :, 0]
        aux_gamma2 = gamma_values[:, :, 1]

        self.xi=aux_gamma1.sum(axis=0)/(aux_gamma1.sum(axis=0)+aux_gamma2.sum(axis=0)+self.lambda_reg*(self.xi-1))

        for fdx in range(self.n_features):
            if self.xi[fdx] > 1 or self.xi[fdx] < 0:
                print('Attention prboability out of range [0,1] ... change the regularisation parameter for avoiding errors.')


    #this function evaluates the posterior for belongin to one or another split, it is used for the e-step for pi
    def chi(self, subj_id, feature):

        subj_indices = self.data.index[self.data['subj_id'] == self.subjects[subj_id]]
        y_single_subject = self.y[subj_indices]#torch.tensor(subject_data.iloc[:, 2:].values, dtype=torch.float32, device=self.device)
        t_single_subject = self.t[subj_indices]+self.time_shift[subj_id]#torch.tensor(subject_data['time'].values, dtype=torch.float32, device=self.device)
            
        y_single_subject_single_feature=y_single_subject[:,feature]

        like_no_split=torch.exp(-self.loss(sigmoid_eval(t_single_subject, self.theta[f'{self.name_biomarkers[feature]}_{0}_split'][0]), y_single_subject_single_feature, torch.exp(self.log_noise_std[feature])))

        like_split1=torch.exp(-self.loss(sigmoid_eval(t_single_subject, self.theta[f'{self.name_biomarkers[feature]}_{1}_split'][0]), y_single_subject_single_feature, torch.exp(self.log_noise_std[feature])))
        like_split2=torch.exp(-self.loss(sigmoid_eval(t_single_subject, self.theta[f'{self.name_biomarkers[feature]}_{1}_split'][1]), y_single_subject_single_feature, torch.exp(self.log_noise_std[feature])))

        chi_1_new = like_split1*self.pi[subj_id]/(like_no_split*self.xi[feature] + (like_split1*self.pi[subj_id] + like_split2*(1-self.pi[subj_id]))*(1-self.xi[feature]))
        chi_2_new = like_split2*(1-self.pi[subj_id])/(like_no_split*self.xi[feature] + (like_split1*self.pi[subj_id] + like_split2*(1-self.pi[subj_id]))*(1-self.xi[feature]))

        return chi_1_new, chi_2_new

    #this function performs one e-step for the parameter pi
    def e_step_pi(self):
        aux_chi=torch.tensor([[self.chi(subj_id, feature) for feature in range(self.n_features)] 
                               for subj_id in range(self.n_subjects)], dtype=torch.float32, device=self.device)
        self.pi=(aux_chi[:,:,0]/(aux_chi[:,:,0]+aux_chi[:,:,1])).mean(axis=1)

    
    def log_prior(self, outer_iter):
        log_prior_noise=-self.lambda_reg_noise*torch.sum(self.log_noise_std+1/torch.exp(self.log_noise_std))
        log_prior_xi=-self.lambda_reg*(self.n_features-self.xi.sum())
        log_prior_time_shift=-((self.time_shift/3)**2).sum() if self.time_shift_eval else 0

        rate_grouth=torch.tensor([self.theta[f'{self.name_biomarkers[fdx]}_{sdx}_split'][ndx].cpu().detach()[1] for fdx in range(self.n_features) for sdx in range(2) for ndx in range(sdx+1)])
        log_prior_theta=-1/self.lambda_reg_theta*torch.sum(torch.square((torch.abs(rate_grouth)-3))) if outer_iter<0 and self.lambda_reg_theta>0 else torch.tensor([0], dtype=torch.float32, device=self.device)

        return log_prior_xi+log_prior_noise+log_prior_time_shift+log_prior_theta
    

    #this function evaluates the loss function for a given configuration as -log_like+constraints
    def loss_eval(self, outer_iter):
        return -(self.log_like()+self.log_prior(outer_iter))
 
    #this function print benchmarks every 5 iterations, if verbose==True it also display on the workspace, 
    #otherwise it only save the plots in the folder fig_benchmarks
    def print_benchmarks(self, tdx):
        if (tdx+1)%self.n_prints==0:
            if self.verbose:
                print(f'iter {tdx+1}/{self.n_outer_iterations} -- loss: {self.all_loss[-1]:.4f}')
                for fdx in range(self.n_features):
                    print(f'    P(split {self.name_biomarkers[fdx]}) = {1-self.xi[fdx].cpu().item():.4f}', end=' ')
                if self.log_noise_std.requires_grad:
                    print('')
                    for fdx in range(self.n_features):
                        print(f'    noisestd {self.name_biomarkers[fdx]} = {torch.exp(self.log_noise_std[fdx]).item():.4f}', end=' ') 
            print('\n')
            if self.benchmarks:
                self.estimates()
                plot_solution(self, save=self.benchmarks, show=self.verbose, show_alternatives=True, dpi=100, name_path=f'fig_benchmarks/sol_iter_{tdx+1}')


    #this function performs the m-step for the optimisation problem
    def optimise(self, n_outer_iterations=30, n_inner_iterations_time_shift=30, n_inner_iterations_theta_and_noise=30, 
                 lr_theta_and_noise=1e-1, lr_time_shift=1e-2):
        self.n_outer_iterations=n_outer_iterations
        self.n_inner_iterations_time_shift=n_inner_iterations_time_shift
        self.n_inner_iterations_theta_and_noise=n_inner_iterations_theta_and_noise
        self.lr_theta_and_noise=lr_theta_and_noise
        self.lr_time_shift=lr_time_shift
        start=time.time()
        if self.regression_parameters: optimizer=torch.optim.Adam(self.regression_parameters, lr=lr_theta_and_noise)
        if self.time_shift.requires_grad: optimizer_time_shift=torch.optim.Adam([self.time_shift], lr=lr_time_shift)

        for tdx in range(n_outer_iterations):
            if self.xi_eval: self.e_step_xi()
            if self.pi_eval: self.e_step_pi()

            if self.time_shift.requires_grad:
                for epoch in range(n_inner_iterations_time_shift):
                    loss_time_shift=self.loss_eval(tdx)
                    self.all_loss.append(loss_time_shift.item())
                    optimizer_time_shift.zero_grad()
                    loss_time_shift.backward()
                    optimizer_time_shift.step()

            if self.theta_eval or self.noise_std_eval:
                for epoch in range(n_inner_iterations_theta_and_noise):
                    loss=self.loss_eval(tdx)
                    self.all_loss.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if self.stopping_criteria and epoch>1 and np.abs(self.all_loss[-1]-self.all_loss[-2])/self.all_loss[-2]<1e-3:
                        break

            if self.verbose:
                self.print_benchmarks(tdx)

        self.estimates()
        self.time_elapsed=time.time()-start
        if self.verbose:
            print(f'Elapsed time: {self.time_elapsed:.4f}s')


    #this function estimates the ML configuration
    def estimates(self):
        self.est_num=np.asarray([1 if self.xi[fdx] > 0.5 else 2 for fdx in range(self.n_features)])

        self.est_theta=[[self.theta[f'{self.name_biomarkers[fdx]}_{self.est_num[fdx]-1}_split'][sdx].cpu().detach()
                        for sdx in range(self.est_num[fdx])] for fdx in range(self.n_features)]
        self.est_subpop=torch.zeros(self.n_samples, dtype=torch.int, device='cpu')
        self.est_time=torch.zeros(self.n_samples, dtype=torch.float32, device='cpu')
        for idx in range(self.n_samples):
            indices = torch.nonzero(torch.tensor(self.subjects) == self.data['subj_id'].values[idx], as_tuple=True)[0].item()
            self.est_subpop[idx]= 0 if self.pi[indices] > 0.5 and self.est_num.max()>1 else 1
            self.est_time[idx]=self.t[idx]+self.time_shift[indices].cpu().detach()
            
        self.est_noise=[torch.exp(self.log_noise_std[fdx].cpu().detach()) for fdx in range(self.n_features)]


    #this function saves the model in the indicated folder
    def save(self, name_path='dpmost_sol'):
        with open(f'{name_path}.pkl', 'wb') as f:
            pickle.dump(self, f)