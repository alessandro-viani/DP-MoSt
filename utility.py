import torch # type: ignore
import pickle
import numpy as np 
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

def sigmoid_eval(t, theta):
    """
    Evaluate the sigmoid function for given time points `t` and parameters `theta`.

    Parameters:
    ----------
    t : torch.Tensor
        The input time points. Shape: (n_timepoints,)
    theta : torch.Tensor
        Parameters for the sigmoid function. If multi-dimensional, the first dimension corresponds
        to different features. Shape: (n_features, 3) or (3,) for a single feature.
        - theta[0]=midpoint: The midpoint of the sigmoid curve.
        - theta[1]=growth_rate: The growth rate of the sigmoid.
        - theta[2]=supremum: The upper asymptote (maximum value) of the sigmoid.

    Returns:
    -------
    torch.Tensor
        The evaluated sigmoid values. Shape: (n_timepoints, n_features) or (n_timepoints,) for a single feature.
    """
    if len(theta.shape) > 1:
        midpoint = theta[:, 0]  # Shape: (n_features,)
        growth_rate = theta[:, 1]  # Shape: (n_features,)
        supremum = theta[:, 2]  # Shape: (n_features,)
    else:
        midpoint = theta[0]  # Shape: (1,)
        growth_rate = theta[1]  # Shape: (1,)
        supremum = theta[2]  # Shape: (1,)

    # Compute sigmoid
    sigmoid_value = supremum / (1 + torch.exp(-(torch.abs(growth_rate) * (t - midpoint))))

    return sigmoid_value  # Shape: (n_timepoints, n_features)


def log_normal(x,mean,std):
    return -torch.log(std)-0.5*torch.log(torch.tensor(torch.pi))-0.5*((x-mean)/std)**2


def evaluate_distance(theta_10, theta_11, t):
    p = sigmoid_eval(theta=torch.tensor(theta_10), t=t)
    q = sigmoid_eval(theta=torch.tensor(theta_11), t=t)
    return torch.norm(p-q,p=2)


def initialise_sigmoid(prior_mean, prior_std, device):
    return [(prior_mean[tdx].cpu()+prior_std[tdx].cpu()*torch.randn((1), dtype=torch.float32, device=device)) for tdx in range(len(prior_mean))]


def data_parameters(n_features, max_dist, min_dist, device='cpu'):
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
            distance = evaluate_distance(theta_10, theta_11, t)
            while distance<=min_dist or distance >= max_dist:
                theta_10 = initialise_sigmoid(prior_mean, prior_std, device)
                theta_11 = initialise_sigmoid(prior_mean, prior_std, device)
                distance = evaluate_distance(theta_10, theta_11, t)
            theta_aux=[theta_10, theta_11]
        else:
            theta_aux=[initialise_sigmoid(prior_mean, prior_std, device)]
        dict_data.update({f'Biomarker_{fdx}': torch.tensor(theta_aux, dtype=torch.float32, device=device)})
    
    return dict_data


def data_creation(n_subjects, n_time_points, n_features, noise_std=None, 
                  max_dist=1, min_dist=0, time_shifted=False, 
                  save=True, name_path='dpmost_data', device='cpu'):
    
    columns = ['subj_id', 'time'] + [f'Biomarker_{i}' for i in range(n_features)]
    data = pd.DataFrame(columns=columns)

    noise_std=torch.tensor([noise_std for fdx in range(n_features)], dtype=torch.float32, device=device)
    
    dict_data=data_parameters(n_features, max_dist, min_dist, device)

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
                    data.loc[count, f'Biomarker_{fdx}'] = (sigmoid_eval(theta=dict_data[f'Biomarker_{fdx}'][0], t=_t) + noise_std[fdx]*torch.randn(1, device=device)).item()
                elif dict_data['n_sig'][fdx]>1 and data.loc[count, 'subj_id'] in subjects_1:
                    data.loc[count, f'Biomarker_{fdx}'] = (sigmoid_eval(theta=dict_data[f'Biomarker_{fdx}'][1], t=_t) + noise_std[fdx]*torch.randn(1, device=device)).item()
            count += 1

    data[data.columns]=data[data.columns].apply(pd.to_numeric)
    dict_data['data']=data
    dict_data['noise_std']=noise_std
    dict_data['n_features']=n_features
    dict_data['n_subjects']=n_subjects
    dict_data['n_time_points']=n_time_points
    dict_data['max_dist']=max_dist
    dict_data['pi_true']=torch.tensor([dict_data['sub_pop'][data[data['subj_id']==_sdx].index[0]] for _sdx in data['subj_id'].unique()]).cpu().numpy().astype(int)

    if save:
        with open(f'./{name_path}.pkl', 'wb') as f:
            pickle.dump(dict_data, f)

    return dict_data


def plot_data(data, dict_data=None, plt_style='seaborn-v0_8-darkgrid', colors=['steelblue','lightcoral'], 
              colors_trajectory=['darkblue', 'darkred'], s=50, n_for_col=5, figsize=None, fontsize=15, alpha=0.7, save=False, show=True, 
              name_path='data_fig', x_lim=[-3, 23], y_lim=[-1, 4], dpi=50, plot_x_axis=True, plot_y_axis=True):
    plt.style.use(plt_style)
    
    name_biomarkers=data.columns[2:]
    n_features=len(name_biomarkers)
    data_aux=data.copy()
    data_aux['sub_pop']=dict_data['sub_pop'] if dict_data is not None else np.zeros(data_aux.shape[0])


     # Create subplots if more than one feature is present
    if n_features == 1: 
        if figsize is None: figsize=[5*n_features, 5]
        plt.figure(figsize=(figsize[0], figsize[1]))
    if 1 < n_features <= n_for_col: 
        if figsize is None: figsize=[n_features, n_for_col]
        fig, ax = plt.subplots(1, n_features, figsize=(figsize[0], figsize[1]), sharex=True, sharey=True)
    elif n_features > n_for_col:
        if figsize is None: figsize=[5*n_for_col, 5*int(np.ceil(n_features/n_for_col))]
        fig, ax = plt.subplots(int(np.ceil(n_features/5)), n_for_col, figsize=(figsize[0], figsize[1]), sharex=True, sharey=True)
    
    # Loop through each feature to plot
    for fdx in range(n_features):
        if 1 < n_features <= n_for_col: plt.sca(ax[fdx])  # Set the current axis
        if n_features > n_for_col:
            row=0 if fdx < ax.shape[1] else 1
            plt.sca(ax[int(np.floor(fdx/n_for_col)), fdx%n_for_col])  # Set the current axis

        sns.scatterplot(data=data_aux, x='time', y=f'{name_biomarkers[fdx]}', hue='sub_pop', s=s, legend=False, palette=colors[:len(data_aux['sub_pop'].unique())], alpha=0.7)
        for _ in data_aux['subj_id'].unique():
            sorted_index=np.argsort(data_aux[data_aux['subj_id']==_]['time'].values)
            t=data_aux[data_aux['subj_id']==_]['time'].values[sorted_index]
            y=data_aux[data_aux['subj_id']==_][name_biomarkers[fdx]]
            plt.plot(t, y, linewidth=1, color='k', alpha=0.1)
        
        if dict_data is not None:
            for sdx in range(dict_data['n_sig'][fdx].item()):
#                sorted_time, sorted_index=torch.sort(torch.tensor(data_aux['time'].values))
                y=sigmoid_eval(theta=dict_data[f'{name_biomarkers[fdx]}'][sdx], t=dict_data['t']).cpu().detach().numpy()
                plt.plot(dict_data['t'].cpu().numpy(), y, color=colors_trajectory[sdx], linestyle=':', 
                         linewidth=3, alpha=alpha, label=f'Sub-pop {sdx}')
            
        plt.title(name_biomarkers[fdx].replace("_", " "), fontsize=fontsize)
        plt.xlabel('t', fontsize=fontsize)
        if plot_x_axis: plt.xticks(fontsize=fontsize)
        else: plt.xticks(fontsize=0)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        if plot_y_axis: plt.yticks(fontsize=fontsize)
        else: plt.yticks(fontsize=0)
        if 1 <= n_features <= n_for_col: 
            if fdx==0: plt.ylabel('Biomarker severity', fontsize=fontsize)
        if n_features > n_for_col: 
            if fdx==0 or fdx%n_for_col==0: plt.ylabel('Biomarker severity', fontsize=fontsize)
        #if fdx>0 or dict_data is None: plt.legend(fontsize=fontsize).remove() 
        if fdx==n_features-1 and dict_data is not None: plt.legend(fontsize=fontsize)

    plt.tight_layout()
    if save: plt.savefig(name_path, dpi=dpi)
    plt.show() if show else plt.close()


def plot_solution(dpmost, plt_style='seaborn-v0_8-darkgrid', 
                  colors=['steelblue','lightcoral'], 
                  colors_trajectory=['darkblue', 'darkred'], 
                  x_lim=[-20, 30], y_lim=[-2, 5], s=50, fontsize=15, alpha=0.7,
                  show_alternatives=True, show_noise_band=True, show_data=True,
                  save=False, show=True, n_label=None, name_path='dpmost_fig', 
                  n_for_col=5, figsize=None, dpi=100, 
                  plot_x_axis=True, plot_y_axis=True,
                  print_noise=None, label=['sub-type 0', 'sub-type 1'],
                  xticks_points=None, xticks_names=None, loc='best'):

    time=torch.linspace(np.asarray(x_lim).min(), np.asarray(x_lim).max(), 1000, device=dpmost.device, dtype=torch.float32)
    dpmost.estimates()

    if print_noise == None: print_noise=dpmost.log_noise_std.requires_grad
    
    # Set the plot style
    plt.style.use(plt_style)
    
    # Create a copy of the data and add estimated time and subpopulation
    data_aux = dpmost.data.copy()
    data_aux['time'] = dpmost.est_time.numpy()
    data_aux[data_aux.columns[2:]] = dpmost.y
    data_aux['sub_pop'] = dpmost.est_subpop

    if n_label == None: n_label= dpmost.n_features - 1

     # Create subplots if more than one feature is present
    if dpmost.n_features == 1: 
        if figsize is None: figsize=[5*dpmost.n_features, 5]
        plt.figure(figsize=(figsize[0], figsize[1]))
    if 1 < dpmost.n_features <= n_for_col: 
        if figsize is None: figsize=[n_for_col*dpmost.n_features, n_for_col]
        fig, ax = plt.subplots(1, dpmost.n_features, figsize=(figsize[0], figsize[1]), sharex=True, sharey=True)
    elif dpmost.n_features > n_for_col:
        if figsize is None: figsize=[5*n_for_col, 5*int(np.ceil(dpmost.n_features/n_for_col))]
        fig, ax = plt.subplots(int(np.ceil(dpmost.n_features/5)), n_for_col, figsize=(figsize[0], figsize[1]), sharex=True, sharey=True)
    
    # Loop through each feature to plot
    for fdx in range(dpmost.n_features):
        if 1 < dpmost.n_features <= n_for_col: plt.sca(ax[fdx])  # Set the current axis
        if dpmost.n_features > n_for_col:
            row=0 if fdx < ax.shape[1] else 1
            plt.sca(ax[int(np.floor(fdx/n_for_col)), fdx%n_for_col])  # Set the current axis

        # Scatter plot of the data points, colored by subpopulation
        if show_data:
            sns.scatterplot(data=data_aux, x='time', y=f'{dpmost.name_biomarkers[fdx]}', hue='sub_pop', s=s, 
                        legend=False, palette=colors[:len(data_aux['sub_pop'].unique())], alpha=alpha)
        
        # Plot the trajectory of each subject
        for subj_id in data_aux['subj_id'].unique(): 
            sorted_index = np.argsort(data_aux[data_aux['subj_id'] == subj_id]['time'].values)
            t = data_aux[data_aux['subj_id'] == subj_id]['time'].values[sorted_index]
            y = data_aux[data_aux['subj_id'] == subj_id][dpmost.name_biomarkers[fdx]]
            if show_data: plt.plot(t, y, linewidth=1, color='k', alpha=0.1)
            else: plt.plot(t, y, linewidth=1, color='green', alpha=0.5)
            
        # Plot the estimated subpopulation trajectories
        for sdx in range(dpmost.est_num[fdx]):
            sorted_time, sorted_index = torch.sort(dpmost.est_time)
            y = sigmoid_eval(theta=dpmost.est_theta[fdx][sdx], t=time)
            plt.plot(time.numpy(), y.numpy(), color=colors_trajectory[sdx], linestyle='-', linewidth=3, alpha=1, label=label[sdx])
            if show_noise_band:
                plt.fill_between(x=time.numpy(), 
                             y1=y.numpy() - 3*np.exp(dpmost.log_noise_std.detach().numpy())[fdx], 
                             y2=y.numpy() + 3*np.exp(dpmost.log_noise_std.detach().numpy())[fdx], 
                             color=colors_trajectory[sdx], alpha=0.1)
                
        # Optionally plot alternative trajectories
        if show_alternatives:
            for sdx in range(3 - dpmost.est_num[fdx]):
                #sorted_time, sorted_index = torch.sort(dpmost.est_time)
                y = sigmoid_eval(theta=dpmost.theta[f'{dpmost.name_biomarkers[fdx]}_{3 - dpmost.est_num[fdx] - 1}_split'][sdx].detach().cpu(), t=time)
                #plt.plot(sorted_time.numpy(), y.numpy(), color='k', linestyle='--', linewidth=3, alpha=0.3) 
                plt.plot(time.numpy(), y.numpy(), color='k', linestyle='--', linewidth=3, alpha=0.3) 
                
        # Add titles and labels to the plot
        if print_noise: 
            name_title = f'P(split {dpmost.name_biomarkers[fdx].replace("_", " ")}) = {1 - dpmost.xi[fdx].cpu().item():.2f} -- noise std: {torch.exp(dpmost.log_noise_std[fdx]).cpu().item():.2f}'
        else: 
            name_title = f'P(split {dpmost.name_biomarkers[fdx].replace("_", " ")}) = {1 - dpmost.xi[fdx].cpu().item():.2f}'
        plt.title(name_title, fontsize=fontsize)
        plt.xlabel('Time to Conversion', fontsize=fontsize)
        if plot_x_axis: 
            plt.xticks(xticks_points, xticks_names, fontsize=fontsize)
        else: plt.xticks(fontsize=0)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        if plot_y_axis: plt.yticks(fontsize=fontsize)
        else: plt.yticks(fontsize=0)
        if fdx%n_for_col == 0: 
            plt.ylabel('Biomarker severity', fontsize=fontsize)
        if fdx == n_label: 
            plt.legend(fontsize=fontsize, loc=loc)
        else: plt.legend().remove() 

    # Adjust layout and save or display the plot
    plt.tight_layout()
    if save: 
        plt.savefig(name_path, dpi=dpi)
    plt.show() if show else plt.close()


def plot_subpop(dpmost, colors=['steelblue','lightcoral'], plt_style='seaborn-v0_8-darkgrid', sharex=False, sharey=False,
                fontsize=15, n_for_col=5, figsize=None, split=True, bw_adjust=1, cut=2, log_scale=None, kind='violinplot',
                linewidth=1.5, gap=0, show=True, save=False, name_path='./', dpi=200):

    plt.style.use(plt_style)
    data_aux = dpmost.data.copy()
    data_aux['subpop'] = dpmost.est_subpop

     # Create subplots if more than one feature is present
    if dpmost.n_features == 1: 
        if figsize is None: figsize=[5*dpmost.n_features, 5]
        plt.figure(figsize=(figsize[0], figsize[1]))
    if 1 < dpmost.n_features <= n_for_col: 
        if figsize is None: figsize=[n_for_col*dpmost.n_features, n_for_col]
        fig, ax = plt.subplots(1, dpmost.n_features, figsize=(figsize[0], figsize[1]), sharex=sharex, sharey=sharey)
    elif dpmost.n_features > n_for_col:
        if figsize is None: figsize=[5*n_for_col, 5*int(np.ceil(dpmost.n_features/n_for_col))]
        fig, ax = plt.subplots(int(np.ceil(dpmost.n_features/5)), n_for_col, figsize=(figsize[0], figsize[1]), sharex=sharex, sharey=sharey)
    
    # Loop through each feature to plot
    for fdx in range(dpmost.n_features):
        if 1 < dpmost.n_features <= n_for_col: plt.sca(ax[fdx])  # Set the current axis
        if dpmost.n_features > n_for_col:
            row=0 if fdx < ax.shape[1] else 1
            plt.sca(ax[int(np.floor(fdx/n_for_col)), fdx%n_for_col])  # Set the current axis

        legend=True if fdx==0 else False
        if kind=='violinplot':
            sns.violinplot(data=data_aux, y=dpmost.name_biomarkers[fdx], hue='subpop', 
                        saturation=0.7, orient='v', split=split, gap=gap, cut=cut, log_scale=log_scale,
                        bw_adjust=bw_adjust, inner='quartile', legend=legend, linewidth=linewidth, palette=colors)
        elif kind=='boxplot':
            sns.boxplot(data=data_aux, y=dpmost.name_biomarkers[fdx], hue='subpop', 
                saturation=0.7, log_scale=log_scale,legend=legend, linewidth=linewidth, palette=colors)
        else:
            print('Only: violinplot ---- boxplot')
                
        if fdx==0: 
            if dpmost.n_features == 1 or 1 < dpmost.n_features <= 5: 
                handles, _ = ax[fdx%5].get_legend_handles_labels()
            elif dpmost.n_features > 5:
                handles, _ = ax[int(np.floor(fdx/5)), fdx%5].get_legend_handles_labels()
            plt.legend(handles, ['Sub-population 1', 'Sub-population 2'], loc='upper left', fontsize=fontsize)

        plt.title(dpmost.name_biomarkers[fdx].replace('_', ' '), fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlabel('', fontsize=0)
        plt.ylabel('', fontsize=0)

    plt.tight_layout()
    if save: plt.savefig(name_path, dpi=dpi)
    plt.show() if show else plt.close()
