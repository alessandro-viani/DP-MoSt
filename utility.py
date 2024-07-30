import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Sigmoid import Sigmoid
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from scipy import interpolate
import os



def log_normal(x,mean,std):
    return -torch.log(std)-0.5*torch.log(torch.tensor(torch.pi))-0.5*((x-mean)/std)**2


def initialise_sigmoid(prior_mean, prior_std, device):
    return [(prior_mean[tdx].cpu()+prior_std[tdx].cpu()*torch.randn((1), dtype=torch.float32, device=device)) for tdx in range(len(prior_mean))]


def data_parameters(n_features, max_dist, device='cpu'):
    t = torch.linspace(0, 20, 1000, dtype=torch.float32, device=device)
    prior_mean=torch.tensor([10, 0.7, 2], dtype=torch.float32, device=device)
    prior_std=torch.tensor([3.5, 0.2, 0.5], dtype=torch.float32, device=device)
    n_sig=torch.tensor([1 if _ < int(0.5*n_features) else 2 for _ in range(n_features)], dtype=torch.int, device=device)
    xi_true=torch.tensor([1 if _ < int(0.5*n_features) else 0 for _ in range(n_features)], dtype=torch.int, device=device)
    dict_data={'n_sig': n_sig, 't':t, 'sub_pop':[], 'xi_true':xi_true.cpu().numpy().astype(int)}
    for fdx in range(n_features):
        if n_sig[fdx]>1:
            theta_10 = initialise_sigmoid(prior_mean, prior_std, device)
            theta_11 = initialise_sigmoid(prior_mean, prior_std, device)
            while 1/len(t)*((Sigmoid(theta=theta_10).eval(t) - Sigmoid(theta=theta_11).eval(t))**2).sum()<max_dist:
                theta_10 = initialise_sigmoid(prior_mean, prior_std, device)
                theta_11 = initialise_sigmoid(prior_mean, prior_std, device)
            theta_aux=[theta_10, theta_11]
        else:
            theta_aux=[initialise_sigmoid(prior_mean, prior_std, device)]
        dict_data.update({f'Biomarker_{fdx}': torch.tensor(theta_aux, dtype=torch.float32, device=device)})
    
    return dict_data


def data_creation(n_subjects, n_time_points, n_features, noise_std=None, max_dist=1, time_shifted=False, name_path='example/dpmost_data', device='cpu'):
    
    columns = ['subj_id', 'time'] + [f'Biomarker_{i}' for i in range(n_features)]
    data = pd.DataFrame(columns=columns)

    noise_std=torch.tensor([noise_std for fdx in range(n_features)], dtype=torch.float32, device=device)
    
    dict_data=data_parameters(n_features, max_dist, device)

    n_first=int(n_subjects/2)
    index = np.random.choice(n_subjects, size=n_subjects, replace=False)
    subjects_0=index[n_first:]
    subjects_1=index[:n_first]

    count = 0
    for _sdx in range(n_subjects):
        time, _ = torch.sort(dict_data['t'][torch.randint(low=0, high=dict_data['t'].shape[0], size=(n_time_points,))])
        for _t in time:
            data.loc[count, 'subj_id']=_sdx
            dict_data['sub_pop'].append(0 if _sdx in subjects_0 else 1)
            data.loc[count, 'time']=_t.item()-torch.min(time).item() if time_shifted else _t.item()
            for fdx in range(n_features):
                if dict_data['n_sig'][fdx]==1 or (dict_data['n_sig'][fdx]>1 and data.loc[count, 'subj_id'] in subjects_0):
                    data.loc[count, f'Biomarker_{fdx}'] = (Sigmoid(theta=dict_data[f'Biomarker_{fdx}'][0]).eval(_t) + noise_std[fdx]*torch.randn(1, device=device)).item()
                elif dict_data['n_sig'][fdx]>1 and data.loc[count, 'subj_id'] in subjects_1:
                    data.loc[count, f'Biomarker_{fdx}'] = (Sigmoid(theta=dict_data[f'Biomarker_{fdx}'][1]).eval(_t) + noise_std[fdx]*torch.randn(1, device=device)).item()
            count += 1

    data[data.columns]=data[data.columns].apply(pd.to_numeric)
    dict_data['data']=data
    dict_data['noise_std']=noise_std
    dict_data['n_features']=n_features
    dict_data['n_subjects']=n_subjects
    dict_data['n_time_points']=n_time_points
    dict_data['max_dist']=max_dist
    dict_data['pi_true']=torch.tensor([dict_data['sub_pop'][data[data['subj_id']==_sdx].index[0]] for _sdx in data['subj_id'].unique()]).cpu().numpy().astype(int)

    with open(f'./{name_path}.pkl', 'wb') as f:
            pickle.dump(dict_data, f)

    return dict_data


def plot_data(data, dict_data=None, plt_style='seaborn-v0_8-darkgrid', colors=['steelblue','lightcoral'], 
              colors_trajectory=['darkblue', 'darkred'], s=50, fontsize=15, alpha=0.7, save=True, show=True, name_path='example/data_fig', dpi=50):
    plt.style.use(plt_style)
    
    name_biomarkers=data.columns[2:]
    n_features=len(name_biomarkers)
    data_aux=data.copy()
    data_aux['sub_pop']=dict_data['sub_pop'] if dict_data is not None else np.zeros(data_aux.shape[0])

    if n_features>1:
        fig, ax=plt.subplots(1, n_features, figsize=(5*n_features,5), sharex=True, sharey=True)
    else:
        plt.figure(figsize=(5*n_features,5))
    
    for fdx in range(n_features):
        if n_features>1: plt.sca(ax[fdx])
        sns.scatterplot(data=data_aux, x='time', y=f'{name_biomarkers[fdx]}', hue='sub_pop', s=s, legend=False, palette=colors[:len(data_aux['sub_pop'].unique())], alpha=0.7)
        for _ in data_aux['subj_id'].unique():
            sorted_index=np.argsort(data_aux[data_aux['subj_id']==_]['time'].values)
            t=data_aux[data_aux['subj_id']==_]['time'].values[sorted_index]
            y=data_aux[data_aux['subj_id']==_][name_biomarkers[fdx]]
            plt.plot(t, y, linewidth=1, color='k', alpha=0.1)
        
        if dict_data is not None:
            for sdx in range(dict_data['n_sig'][fdx].item()):
#                sorted_time, sorted_index=torch.sort(torch.tensor(data_aux['time'].values))
                y=Sigmoid(theta=dict_data[f'{name_biomarkers[fdx]}'][sdx]).eval(dict_data['t']).cpu().detach().numpy()
                plt.plot(dict_data['t'].numpy(), y, color=colors_trajectory[sdx], linestyle=':', 
                         linewidth=3, alpha=alpha, label=f'Sub-pop {sdx}')
            
        plt.title(name_biomarkers[fdx].replace("_", " "), fontsize=fontsize)
        plt.xlabel('t', fontsize=fontsize)
        plt.xticks([0,4,8,12,16,20], fontsize=fontsize)
        plt.xlim([-1,21])
        plt.yticks(fontsize=fontsize)
        if fdx==0: plt.ylabel('Biomarker severity', fontsize=fontsize)
        if fdx>0 or dict_data is None: plt.legend().remove() 
        if fdx==n_features-1 and dict_data is not None: plt.legend(fontsize=fontsize)

    plt.tight_layout()
    if save: plt.savefig(name_path, dpi=dpi)
    plt.show() if show else plt.close()


def plot_solution(dpmost, plt_style='seaborn-v0_8-darkgrid', colors=['steelblue','lightcoral'], 
                  colors_trajectory=['darkblue', 'darkred'], s=50, fontsize=15, alpha=0.7, save=True, 
                  show=True, name_path='example/dpmost_fig', dpi=50):
    
    dpmost.estimates()
    
    plt.style.use(plt_style)
    data_aux=dpmost.data.copy()
    data_aux['time']=dpmost.est_time.numpy()
    data_aux['sub_pop']=dpmost.est_subpop

    if dpmost.n_features>1: 
        fig, ax=plt.subplots(1, dpmost.n_features, figsize=(5*dpmost.n_features,5), sharex=True, sharey=True)
    else:
        plt.figure(figsize=(5*dpmost.n_features,5))
    
    for fdx in range(dpmost.n_features):
        if dpmost.n_features>1: plt.sca(ax[fdx])
        sns.scatterplot(data=data_aux, x='time', y=f'{dpmost.name_biomarkers[fdx]}', hue='sub_pop', s=s, 
                        legend=False, palette=colors[:len(data_aux['sub_pop'].unique())], alpha=alpha)
        for _ in data_aux['subj_id'].unique(): 
            sorted_index=np.argsort(data_aux[data_aux['subj_id']==_]['time'].values)
            t=data_aux[data_aux['subj_id']==_]['time'].values[sorted_index]
            y=data_aux[data_aux['subj_id']==_][dpmost.name_biomarkers[fdx]]
            plt.plot(t, y, linewidth=1, color='k', alpha=0.1)
            
        for sdx in range(dpmost.est_num[fdx]):
                sorted_time, sorted_index=torch.sort(dpmost.est_time)
                y=Sigmoid(theta=dpmost.est_theta[fdx][sdx]).eval(sorted_time)
                plt.plot(sorted_time.numpy(), y.numpy(), color=colors_trajectory[sdx],  linestyle=':',
                         linewidth=3, alpha=1, label=f'Sub-pop {sdx}')
                
        if dpmost.log_noise_std.requires_grad: name_title=f'P(split {dpmost.name_biomarkers[fdx].replace("_", " ")}) = {1-dpmost.xi[fdx].cpu().item():.2f} -- noise std: {torch.exp(dpmost.log_noise_std[fdx]).cpu().item():.2f}'
        else: name_title=f'P(split {dpmost.name_biomarkers[fdx].replace("_", " ")}) = {1-dpmost.xi[fdx].cpu().item():.2f}'
        plt.title(name_title, fontsize=fontsize)
        plt.xlabel('t', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        #plt.xlim([-1,21])
        plt.yticks(fontsize=fontsize)
        if fdx==0: plt.ylabel('Biomarker severity', fontsize=fontsize)
        if fdx>0: plt.legend().remove() 
        if fdx==dpmost.n_features-1: plt.legend(fontsize=fontsize)

    plt.tight_layout()
    if save: plt.savefig(name_path, dpi=dpi)
    plt.show() if show else plt.close()


def eval_roc_curve_xi(dpmost, dict_data, t):
    if not torch.isnan(dpmost.xi).any():
        fpr, tpr, _=roc_curve(dict_data['xi_true'], dpmost.xi.numpy())
        f=interpolate.interp1d(fpr, tpr)
        roc=f(t)
    else:
        roc=np.zeros(t.shape)

    roc[0]=0
    roc[-1]=1
    return roc


def eval_roc_curve_pi(dpmost, dict_data, t):
    
    if not torch.isnan(dpmost.pi).any():
        fpr, tpr, _=roc_curve(dict_data['pi_true'], dpmost.pi)
        fpr_2, tpr_2, _=roc_curve(dict_data['pi_true'], 1-dpmost.pi)

        auc_1 = auc(fpr, tpr)
        auc_2 = auc(fpr_2, tpr_2)
        if auc_1<auc_2 and tpr_2[0]==0:
            tpr=tpr_2
            fpr=fpr_2
        f = interpolate.interp1d(fpr, tpr)
        roc=f(t)
    else:
        roc=np.zeros(t.shape)

    roc[0]=0
    roc[-1]=1
    return roc


def eval_all_roc(name_folder, n_sim):
    t=np.linspace(0,1,100)
    roc_xi=0
    roc_pi=0
    for idx in range(n_sim):
        if os.path.isfile(f'{name_folder}/sol_{idx}.pkl'):
            with open(f'{name_folder}/data_{idx}.pkl', 'rb') as f:
                dict_data = pickle.load(f) 
            with open(f'{name_folder}/sol_{idx}.pkl', 'rb') as f:
                dpmost = pickle.load(f)
            roc_xi+=eval_roc_curve_xi(dpmost, dict_data, t)
            roc_pi+=eval_roc_curve_pi(dpmost, dict_data, t)
    return roc_xi/n_sim, roc_pi/n_sim, t


def error_eval(dpmost, dict_data):
    loss = torch.nn.MSELoss()
    error_ospa=np.zeros(dict_data['n_features'])
    for fdx in range(dict_data['n_features']):
        if dpmost.est_num[fdx]==1 and dict_data['n_sig'][fdx].item()==1:
            input = Sigmoid(theta=dpmost.est_theta[fdx][0]).eval(dict_data['t']).flatten()
            target = Sigmoid(theta=dict_data[f'Biomarker_{fdx}'][0]).eval(dict_data['t']).flatten()
            error_ospa[fdx]=loss(input, target).item()

        if dpmost.est_num[fdx]==2 and dict_data['n_sig'][fdx].item()==2:
            input_0 = Sigmoid(theta=dpmost.est_theta[fdx][0]).eval(dict_data['t']).flatten()
            input_1 = Sigmoid(theta=dpmost.est_theta[fdx][1]).eval(dict_data['t']).flatten()
            target_0 = Sigmoid(theta=dict_data[f'Biomarker_{fdx}'][0]).eval(dict_data['t']).flatten()
            target_1 = Sigmoid(theta=dict_data[f'Biomarker_{fdx}'][1]).eval(dict_data['t']).flatten()
            error_ospa[fdx]=np.min(np.asarray([loss(input_0, target_0).item()+loss(input_1, target_1).item(), loss(input_0, target_1).item()+loss(input_1, target_0).item()]))

        if dpmost.est_num[fdx]==2 and dict_data['n_sig'][fdx].item()==1:
            input_0 = Sigmoid(theta=dpmost.est_theta[fdx][0]).eval(dict_data['t']).flatten()
            input_1 = Sigmoid(theta=dpmost.est_theta[fdx][1]).eval(dict_data['t']).flatten()
            target = Sigmoid(theta=dict_data[f'Biomarker_{fdx}'][0]).eval(dict_data['t']).flatten()
            error_ospa[fdx]=np.min(np.asarray([loss(input_0, target).item(), loss(input_1, target).item()]))

        if dpmost.est_num[fdx]==1 and dict_data['n_sig'][fdx].item()==2:
            input = Sigmoid(theta=dpmost.est_theta[fdx][0]).eval(dict_data['t']).flatten()
            target_0 = Sigmoid(theta=dict_data[f'Biomarker_{fdx}'][0]).eval(dict_data['t']).flatten()
            target_1 = Sigmoid(theta=dict_data[f'Biomarker_{fdx}'][1]).eval(dict_data['t']).flatten()
            error_ospa[fdx]=np.min(np.asarray([loss(input, target_0).item(), loss(input, target_1).item()]))
    
    error_noise=((torch.exp(dpmost.log_noise_std.detach())-dict_data['noise_std'])/dict_data['noise_std']).cpu().numpy()

    return error_ospa, error_noise


def new_error_eval(dpmost, dict_data):

    error_rate_grouth=np.zeros(dpmost.n_features)
    for fdx in range(dpmost.n_features):
        if dpmost.est_num[fdx]==1 and dict_data['n_sig'][fdx].item()==1:
            true_grouth=dict_data[f'Biomarker_{fdx}'][0][1]
            est_grouth=dpmost.est_theta[fdx][0][1]
            error_rate_grouth[fdx]=(1-true_grouth/est_grouth).item()

        if dpmost.est_num[fdx]==2 and dict_data['n_sig'][fdx].item()==2:
            true_grouth_1=dict_data[f'Biomarker_{fdx}'][0][1]
            true_grouth_2=dict_data[f'Biomarker_{fdx}'][1][1]
            est_grouth_1=dpmost.est_theta[fdx][0][1]
            est_grouth_2=dpmost.est_theta[fdx][1][1]
            error_rate_grouth[fdx]=(true_grouth_1/true_grouth_2-est_grouth_1/est_grouth_2).item()
        if dpmost.est_num[fdx]==2 and dict_data['n_sig'][fdx].item()==1:
            true_grouth=dict_data[f'Biomarker_{fdx}'][0][1]
            est_grouth_1=dpmost.est_theta[fdx][0][1]
            est_grouth_2=dpmost.est_theta[fdx][1][1]
            error_rate_grouth[fdx]=np.min(np.asarray([(1-true_grouth/est_grouth_1).item(), (1-true_grouth/est_grouth_2).item()]))

        if dpmost.est_num[fdx]==1 and dict_data['n_sig'][fdx].item()==2:
            true_grouth_1=dict_data[f'Biomarker_{fdx}'][0][1]
            true_grouth_2=dict_data[f'Biomarker_{fdx}'][1][1]
            est_grouth=dpmost.est_theta[fdx][0][1]
            error_rate_grouth[fdx]=np.min(np.asarray([(1-true_grouth_1/est_grouth).item(), (1-true_grouth_2/est_grouth).item()]))

    return error_rate_grouth