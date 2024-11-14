import torch # type: ignore
import pickle
import time
import itertools

import numpy as np
import matplotlib.pyplot as plt # type: ignore
import matplotlib.cm as cm # type: ignore
import matplotlib as mpl # type: ignore
import torch.nn as nn # type: ignore

from pathlib import Path
from utility import plot_solution, sigmoid_eval # type: ignore
from joblib import Parallel, delayed # type: ignore

class DPMoSt(object):
    """
    DPMoSt Model for Bayesian hierarchical analysis of longitudinal data.

    This class implements a Bayesian hierarchical model suitable for analyzing 
    longitudinal data with multiple features and subpopulations. The model incorporates 
    time-varying effects, noise modeling, and allows for flexible regularization 
    through hyperparameters.

    Attributes:
    ----------
    data : pandas.DataFrame
        The input data containing samples, subjects, and features.
    device : str
        The device for computation, either 'cpu' or 'cuda'.
    y : torch.Tensor
        Tensor representation of the feature data from the input DataFrame.
    t : torch.Tensor
        Tensor representation of the time data from the input DataFrame.
    n_samples : int
        Number of samples in the dataset.
    n_features : int
        Number of features (biomarkers) in the dataset.
    subjects : array-like
        Unique subject identifiers extracted from the input data.
    n_subjects : int
        Number of unique subjects.
    log_noise_std : torch.Tensor
        Logarithm of the standard deviation of noise for each feature.
    prior_mean : torch.Tensor
        Prior mean values for model parameters.
    prior_std : torch.Tensor
        Prior standard deviation values for model parameters.
    name_biomarkers : list of str
        Names of the biomarkers/features used in the model.
    lambda_reg_xi : torch.Tensor
        Regularization parameter for the model probability of split.
    lambda_reg_noise : torch.Tensor
        Regularization parameter for noise modeling.
    lambda_reg_theta : torch.Tensor
        Regularization parameter for theta values.
    n_prints : int
        Frequency of printing updates during optimization.
    benchmarks : bool
        Flag to save benchmarking figures during optimization.
    verbose : bool
        Flag for verbose output during initialization and optimization.
    time_shift_eval : bool
        Flag to evaluate time shift parameters during optimization.
    noise_std_eval : bool
        Flag to evaluate noise standard deviation parameters during optimization.
    theta_eval : bool
        Flag to evaluate theta parameters during optimization.
    xi_eval : bool
        Flag to evaluate xi parameters during optimization.
    pi_eval : bool
        Flag to evaluate pi parameters during optimization.
    do_normalisation : bool
        Flag to normalize the data.
    initialise : bool
        Flag to initialise the model.
    colors : array-like
        Color mapping for visualizing features.
    xi, pi, theta : torch.Tensor
        Model parameters to be estimated.
    est_num, est_theta, est_subpop, est_noise : array-like
        Estimated parameters and noise values after fitting the model.
    time_elapsed : float
        Time taken for model fitting and estimation.
    all_loss : list
        List to store loss values during optimization.
    n_outer_iterations, n_inner_iterations, lr : int, float
        Variables for tracking optimization iterations and learning rates.
    loss : torch.nn.Module
        Loss function used for model fitting.

    Methods:
    -------
    - normalise_data: Normalize the input data features.
    - normalise_time: Adjust the time values to a standard scale.
    - initialisation: Initialize model parameters and states.
    - log_like_single: Compute the log-likelihood for a single observation.
    - log_like: Compute the overall log-likelihood for all observations.
    - gamma: Compute specific gamma-related values for the model.
    - em_step_xi: Perform the Expectation-Maximization step for xi parameters.
    - chi: Calculate the chi values for model fitting.
    - em_step_pi: Perform the Expectation-Maximization step for pi parameters.
    - log_prior: Evaluate the log-prior probability of model parameters.
    - loss_eval: Evaluate the total loss for the current model state.
    - print_benchmarks: Print or log benchmarking information during optimization.
    - optimise: Run the optimization process for model parameter estimation.
    - estimates: Generate estimates of model parameters based on the fitted model.
    - save: Save the current state of the model to a file.

    Notes:
    -----
    - The model is designed to be flexible and can be adapted for different datasets 
      by adjusting the input parameters.
    - It is recommended to ensure that the input DataFrame is structured correctly, 
      with subjects and time information appropriately defined for accurate modeling.
    """

    
    def __init__(self, data=None, 
                device='cpu', 
                modality_scale=None,
                log_noise_std=None, 
                prior_mean=None, 
                prior_std=None, 
                name_biomarkers=None, 
                lambda_reg_xi=None, 
                lambda_reg_noise=None, 
                lambda_reg_theta=None, 
                n_prints=5, 
                benchmarks=False, 
                verbose=False, 
                time_shift_eval=True, 
                noise_std_eval=True, 
                theta_eval=True, 
                xi_eval=True, 
                pi_eval=True,
                do_normalisation=False, 
                initialise=True):
        """
        Initializes the DPMoSt model with input data and various parameters.

        Parameters:
        ----------
        data : pandas.DataFrame, optional
            The input data containing samples, subjects, and features. The second column should be time values.
        device : str, optional
            The device to use for computations, either 'cpu' or 'cuda' for GPU acceleration. Default is 'cpu'.
        modality_scale : array-like, optional
            A scaling factor for each biomarker modality. If None, defaults to an array of ones.
        log_noise_std : torch.Tensor, optional
            Logarithm of the standard deviation of noise for each feature. If None, calculated from the input data.
        prior_mean : torch.Tensor, optional
            Prior mean values for the model parameters. Default is a tensor of [10, 0.7, 2].
        prior_std : torch.Tensor, optional
            Prior standard deviation values for the model parameters. Default is a tensor of [3.5, 0.2, 0.5].
        name_biomarkers : list of str, optional
            Names of the biomarkers/features used in the model. If None, extracted from the input data columns.
        lambda_reg_xi : float, optional
            Regularization parameter for the model. Default is 0.1 times the number of subjects.
        lambda_reg_noise : float, optional
            Regularization parameter for the noise. Default is 0.15 times the number of subjects.
        lambda_reg_theta : float, optional
            Regularization parameter for the theta values. Default is 0.001.
        n_prints : int, optional
            Frequency of printing updates during optimization. Default is 5.
        benchmarks : bool, optional
            If True, saves benchmarking figures during optimization.
        verbose : bool, optional
            If True, enables verbose output during initialization and optimization.
        time_shift_eval : bool, optional
            If True, evaluates time shift parameters during optimization.
        noise_std_eval : bool, optional
            If True, evaluates noise standard deviation parameters during optimization.
        theta_eval : bool, optional
            If True, evaluates theta parameters during optimization.
        xi_eval : bool, optional
            If True, evaluates xi parameters during optimization.
        pi_eval : bool, optional
            If True, evaluates pi parameters during optimization.
        do_normalisation : bool, optional
            If True, data nomralization is performed.
        initialise : bool, optional
            If True, initialization is performed.

        Returns:
        -------
        None
            This method does not return any values.

        Details:
        -------
        - This constructor initializes various attributes needed for the DPMoSt model, including the input data, 
        noise parameters, priors, regularization parameters, and feature names.
        
        - It converts the input data into PyTorch tensors and assigns them to the appropriate attributes for 
        later processing.

        - The method also handles the initialization of various parameters, prints initialization details if 
        verbose mode is enabled, and prepares the model for further optimization and estimation.

        - Additionally, it ensures that the necessary directories for saving benchmarks are created if required.
        """

        super(DPMoSt, self).__init__()  # Call the parent constructor
        
        #input parameters
        self.data=data
        self.do_normalisation=do_normalisation
        self.initialise=initialise
        
        self.y=torch.tensor(self.data.iloc[:,2:].values, dtype=torch.float32, device=device)
        self.normalise_data()
        self.t=torch.tensor(self.data['time'].values, dtype=torch.float32, device=device)
        self.normalise_time()
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

        self.modality_scale=torch.tensor(modality_scale, dtype=torch.float32, device=self.device)/len(self.name_biomarkers) if modality_scale is not None else torch.ones(len(self.name_biomarkers), dtype=torch.float32, device=self.device)


        if lambda_reg_xi is None:
            lambda_reg_xi=0.1*self.n_subjects
        self.lambda_reg_xi=torch.tensor([lambda_reg_xi], dtype=torch.float32, device=self.device)
        
        if lambda_reg_noise is None:
            lambda_reg_noise=0.15*self.n_subjects
        self.lambda_reg_noise=torch.tensor([lambda_reg_noise], dtype=torch.float32, device=self.device)

        if lambda_reg_theta is None:
            lambda_reg_theta=0.001
        self.lambda_reg_theta=torch.tensor([lambda_reg_theta], dtype=torch.float32, device=self.device)

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
            if self.initialise: print('Initialisation:')
            print(f'    Num samples: {self.n_samples}')
            print(f'    Num subjects: {self.n_subjects}')
            print(f'    Num features: {self.n_features}')
            print(f'    Evaluation time-shift: {self.time_shift_eval}')
            print(f'    Lambda regression xi: {lambda_reg_xi}')
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
        if self.initialise: self.initialisation()

        self.regression_parameters=[]
        if self.theta_eval:
            for fdx in range(self.n_features):
                for ndx in range(2):
                    for sdx in range(ndx+1):
                        self.regression_parameters.append({'params': self.theta[f'{self.name_biomarkers[fdx]}_{ndx}_split'][sdx]})


    def normalise_data(self, new_min_data=0, new_max_data=3):
        """
        Normalizes the data in `self.y` for each feature (column) to a specified range.

        Parameters:
        ----------
        new_min_data : float, optional (default=0)
            The minimum value of the new normalized range.

        new_max_data : float, optional (default=3)
            The maximum value of the new normalized range.

        Returns:
        -------
        None
            The method modifies `self.y` in place by normalizing each feature (column) to the range [new_min_data, new_max_data].

        Details:
        -------
        - For each feature in `self.y`, the method computes the minimum (`y_min`) and maximum (`y_max`) values.
        - The values in each column are then rescaled to fit the specified range `[new_min_data, new_max_data]`.
        - The formula used for normalization is:
            normalized_value = (value - y_min) / (y_max - y_min) * (new_max_data - new_min_data) + new_min_data
        """
        if self.do_normalisation:
            for col in range(self.y.shape[1]):
                y_min, y_max = self.y[:,col][~torch.isnan(self.y[:,col])].min(), self.y[:,col][~torch.isnan(self.y[:,col])].max()
                self.y[:,col] = (self.y[:,col] - y_min)/(y_max - y_min)*(new_max_data - new_min_data) + new_min_data


    def normalise_time(self, new_min_time=0, new_max_time=20):
        """
        Normalizes the time data in `self.t` to a specified range.

        Parameters:
        ----------
        new_min_time : float, optional (default=0)
            The minimum value of the new normalized time range.

        new_max_time : float, optional (default=20)
            The maximum value of the new normalized time range.

        Returns:
        -------
        None
            The method modifies `self.t` in place by normalizing it to the range [new_min_time, new_max_time].

        Details:
        -------
        - The method computes the minimum (`t_min`) and maximum (`t_max`) values of the time data (`self.t`).
        - The values of `self.t` are rescaled to fit the specified range `[new_min_time, new_max_time]`.
        - The formula used for normalization is:
            normalized_time = (time - t_min) / (t_max - t_min) * (new_max_time - new_min_time) + new_min_time
        """
        if self.t.var()>0:
            t_min, t_max = self.t[~torch.isnan(self.t)].min(), self.t[~torch.isnan(self.t)].max()
            self.t = (self.t - t_min)/(t_max - t_min)*(new_max_time - new_min_time) + new_min_time


    def initialisation(self, epochs=1000, lr=1e-3):
        """
        Initializes model parameters including `xi`, `pi`, and `theta`, as well as optional time shift parameters.

        Parameters:
        ----------
        epochs : int, optional (default=1000)
            The number of epochs to run the optimization for the estimation of `theta` parameters.
        
        lr : float, optional (default=1e-3)
            Learning rate for the optimizer used in the parameter estimation process.
        
        Details:
        -------
        - `xi` is initialized based on whether `xi_eval` is set or not, with different initial values for each feature.
        - `pi` is initialized for all subjects as a tensor of 0.5 values.
        - `theta` is updated based on the feature data availability, using MSELoss and Adam optimizer if data exists.
        - The time-shift parameter is either fixed at zero or learned depending on the value of `time_shift_eval`.

        Returns:
        -------
        None
        """
        if self.xi_eval:
            self.xi=torch.tensor([0.5 for _ in range(self.n_features)], dtype=torch.float32, device=self.device)
        else:
            self.xi=torch.tensor([1 for _ in range(self.n_features)], dtype=torch.float32, device=self.device)
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


    def log_like_old(self):
        log_like = 0
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
                if any(mask):
                    y_aux = y_single_subject[:, feature][mask]
                    t_aux = t_single_subject[mask]
                
                    ll_no_split = -self.loss(sigmoid_eval(t_aux, self.theta[f'{self.name_biomarkers[feature]}_{0}_split'][0]), y_aux, log_noise_std_exp[feature]) + xi_log[feature]
                    ll_split1 = -self.loss(sigmoid_eval(t_aux, self.theta[f'{self.name_biomarkers[feature]}_{1}_split'][0]), y_aux, log_noise_std_exp[feature]) + pi_log + one_minus_xi_log[feature]
                    ll_split2 = -self.loss(sigmoid_eval(t_aux, self.theta[f'{self.name_biomarkers[feature]}_{1}_split'][1]), y_aux, log_noise_std_exp[feature]) + one_minus_pi_log + one_minus_xi_log[feature]

                    log_like += torch.logsumexp(torch.stack((ll_no_split, ll_split1, ll_split2), dim=0), dim=0).sum()
                else:
                    log_like += 0   
        
        return log_like
    

    def log_like_single(self, subj_id, feature):
        """
        Computes the log-likelihood for a single subject and feature, considering both split and non-split models.

        Parameters:
        ----------
        subj_id : int
            Index of the subject for whom the log-likelihood is being computed.

        feature : int
            Index of the feature (biomarker) for which the log-likelihood is being computed.

        Returns:
        -------
        log_like : torch.Tensor
            The computed log-likelihood value for the given subject and feature.

        Details:
        -------
        - Computes likelihood terms for three cases:
            1. No split (using `xi` parameter).
            2. First split (using `pi` and `1 - xi` parameters).
            3. Second split (using `1 - pi` and `1 - xi` parameters).
        - The log-likelihood is based on MSE loss with respect to the model outputs (`sigmoid_eval`) and targets (`y_aux`).
        - Uses `torch.logsumexp` to combine the log-likelihoods from the three cases in a numerically stable way.
        """
            
        log_like = 0

        log_noise_std_exp = torch.exp(self.log_noise_std)  # Pre-compute exponentiated noise std
        xi_log = torch.log(self.xi)
        one_minus_xi_log = torch.log(1 - self.xi)
 
        subj_indices = self.data.index[self.data['subj_id'] == self.subjects[subj_id]]
        y_single_subject = self.y[subj_indices]
        t_single_subject = self.t[subj_indices]+self.time_shift[subj_id]
        
        pi_log = torch.log(self.pi[subj_id])
        one_minus_pi_log = torch.log(1 - self.pi[subj_id])
        
        mask = ~torch.isnan(y_single_subject[:, feature])
        if any(mask):
            y_aux = y_single_subject[:, feature][mask]
            t_aux = t_single_subject[mask]
        
            ll_no_split = -self.loss(sigmoid_eval(t_aux, self.theta[f'{self.name_biomarkers[feature]}_{0}_split'][0]), y_aux, log_noise_std_exp[feature]) + xi_log[feature]
            ll_split1 = -self.loss(sigmoid_eval(t_aux, self.theta[f'{self.name_biomarkers[feature]}_{1}_split'][0]), y_aux, log_noise_std_exp[feature]) + pi_log + one_minus_xi_log[feature]
            ll_split2 = -self.loss(sigmoid_eval(t_aux, self.theta[f'{self.name_biomarkers[feature]}_{1}_split'][1]), y_aux, log_noise_std_exp[feature]) + one_minus_pi_log + one_minus_xi_log[feature]

            log_like = self.modality_scale[feature] * torch.logsumexp(torch.stack((ll_no_split, ll_split1, ll_split2), dim=0), dim=0).sum()

        return log_like


    def log_like(self, n_jobs=1):
        """
        Computes the total log-likelihood across all subjects and features, using parallel computation.

        Parameters:
        ----------
        n_jobs : int, optional (default=1)
            Number of parallel jobs to use for computing the log-likelihood. If set to 1, no parallelism is used.

        Returns:
        -------
        log_like : torch.Tensor
            The total log-likelihood computed across all subjects and features.

        Details:
        -------
        - Uses parallel processing (via `joblib.Parallel`) to compute the log-likelihood for each subject-feature pair.
        - The `log_like_single` method is applied to each combination of subject and feature in parallel.
        - The results are summed to obtain the total log-likelihood.
        """
        log_like=Parallel(n_jobs=n_jobs, prefer="threads", require='sharedmem')(delayed(self.log_like_single)(subj_id, feature) for subj_id, feature in itertools.product(range(self.n_subjects), range(self.n_features)))
        return sum(log_like)


    def gamma(self, subj_id, feature):
        """
        Computes the posterior probabilities (gamma) for a single subject and feature, determining the probability 
        that the subject follows a no-split vs split model.

        Parameters:
        ----------
        subj_id : int
            Index of the subject for whom the gamma values are being computed.

        feature : int
            Index of the feature (biomarker) for which the gamma values are being computed.

        Returns:
        -------
        post_no_split : torch.Tensor
            The posterior probability that the subject follows the no-split model for the given feature.

        post_split : torch.Tensor
            The posterior probability that the subject follows the split model for the given feature.

        Details:
        -------
        - The function retrieves the data for the given subject and feature and checks for missing values (NaNs).
        - It computes the likelihood for the no-split model and the split model:
            - The no-split model uses `theta_no_split` and a sigmoid evaluation of the time data.
            - The split model uses `theta_split` and evaluates two possible splits, `theta_split_1` and `theta_split_2`.
        - The posteriors are then computed using the likelihoods, `xi`, and `pi`, normalizing them based on the combined likelihoods.
        - If the data contains only missing values, both posterior probabilities are set to 0.
        """
        #log likelihood for a single subject with no split
        subj_indices = self.data.index[self.data['subj_id'] == self.subjects[subj_id]]
        y_single_subject = self.y[subj_indices]
        
        t_single_subject = self.t[subj_indices]+self.time_shift[subj_id]
            
        y_single_subject_single_feature=y_single_subject[:,feature]

        mask = ~torch.isnan(y_single_subject_single_feature)

        if any(mask):
            theta_no_split=self.theta[f'{self.name_biomarkers[feature]}_{0}_split'][0]
            output=sigmoid_eval(t_single_subject[mask], theta_no_split)

            like_no_split=torch.exp(-self.loss(output, y_single_subject_single_feature[mask], torch.exp(self.log_noise_std[feature])))
            
            theta_split=self.theta[f'{self.name_biomarkers[feature]}_{1}_split']
            
            theta_split_1=theta_split[0]
            theta_split_2=theta_split[1]

            output_split_1=sigmoid_eval(t_single_subject[mask], theta_split_1)
            output_split_2=sigmoid_eval(t_single_subject[mask], theta_split_2)
            
            log_like_split_1=-self.loss(output_split_1, y_single_subject_single_feature[mask], torch.exp(self.log_noise_std[feature]))
            log_like_split_2=-self.loss(output_split_2, y_single_subject_single_feature[mask], torch.exp(self.log_noise_std[feature]))
                                
            like_split=self.pi[subj_id]*torch.exp(log_like_split_1) + (1-self.pi[subj_id])*torch.exp(log_like_split_2)

            norm_cost=like_no_split*self.xi[feature] + like_split*(1-self.xi[feature])

            post_no_split=like_no_split*self.xi[feature]/norm_cost
            post_split = 1-post_no_split

        else:
            post_no_split = 0
            post_split = 0 

        return post_no_split, post_split
    

    def em_step_xi(self):
        """
        Executes the Expectation-Maximization (EM) step to update the xi probabilities based on computed gamma values.

        Returns:
        -------
        None
            The method updates the `self.xi` attribute in place based on the calculated gamma values.

        Details:
        -------
        - The function calculates the gamma values for all subjects and features by calling the `gamma` method.
        - It separates the computed gamma values into two auxiliary tensors (`aux_gamma1` and `aux_gamma2`) for the no-split and split models, respectively.
        - The xi probabilities are updated using the formula:
            xi[feature] = aux_gamma1_sum / (aux_gamma1_sum + aux_gamma2_sum + lambda_reg_xi * (xi - 1))
        This incorporates a regularization term (`lambda_reg_xi`) to stabilize the estimation.
        - A check is performed to ensure that the updated `xi` values are within the valid range of [0, 1]. If any value is out of this range, a warning message is printed, suggesting the user adjust the regularization parameter to prevent errors.
        """
        gamma_values=torch.tensor([[self.gamma(subj_id, feature) for feature in range(self.n_features)] for subj_id in range(self.n_subjects)], dtype=torch.float32, device=self.device)
        
        aux_gamma1 = gamma_values[:, :, 0]
        aux_gamma2 = gamma_values[:, :, 1]

        self.xi=aux_gamma1.sum(axis=0)/(aux_gamma1.sum(axis=0)+aux_gamma2.sum(axis=0)+self.lambda_reg_xi*(self.xi-1))

        for fdx in range(self.n_features):
            if self.xi[fdx] > 1 or self.xi[fdx] < 0:
                print('Attention prboability out of range [0,1] ... change the regularisation parameter for avoiding errors.')


    def chi(self, subj_id, feature):
        """
        Computes the posterior probabilities (chi) for a single subject and feature, representing the updated probabilities 
        for the split models.

        Parameters:
        ----------
        subj_id : int
            Index of the subject for whom the chi values are being computed.

        feature : int
            Index of the feature (biomarker) for which the chi values are being computed.

        Returns:
        -------
        chi_1_new : torch.Tensor
            The updated probability for the first split model for the given subject and feature.

        chi_2_new : torch.Tensor
            The updated probability for the second split model for the given subject and feature.

        Details:
        -------
        - The function retrieves the data for the specified subject and feature, applying a mask to handle NaN values.
        - It computes the likelihoods for the no-split model and two split models using the `sigmoid_eval` function and the provided model parameters (`theta`).
        - The chi values are updated based on the likelihoods and the current values of `pi` and `xi`:
            - chi_1_new is the probability for the first split model.
            - chi_2_new is the probability for the second split model.
        - If all feature values are NaN, both chi values are set to 0.
        """
        subj_indices = self.data.index[self.data['subj_id'] == self.subjects[subj_id]]
        y_single_subject = self.y[subj_indices]
        t_single_subject = self.t[subj_indices]+self.time_shift[subj_id]
            
        y_single_subject_single_feature=y_single_subject[:,feature]
        mask = ~torch.isnan(y_single_subject_single_feature)

        if any(mask):
            like_no_split=torch.exp(-self.loss(sigmoid_eval(t_single_subject[mask], self.theta[f'{self.name_biomarkers[feature]}_{0}_split'][0]), y_single_subject_single_feature[mask], torch.exp(self.log_noise_std[feature])))

            like_split1=torch.exp(-self.loss(sigmoid_eval(t_single_subject[mask], self.theta[f'{self.name_biomarkers[feature]}_{1}_split'][0]), y_single_subject_single_feature[mask], torch.exp(self.log_noise_std[feature])))
            like_split2=torch.exp(-self.loss(sigmoid_eval(t_single_subject[mask], self.theta[f'{self.name_biomarkers[feature]}_{1}_split'][1]), y_single_subject_single_feature[mask], torch.exp(self.log_noise_std[feature])))

            chi_1_new = like_split1*self.pi[subj_id]/(like_no_split*self.xi[feature] + (like_split1*self.pi[subj_id] + like_split2*(1-self.pi[subj_id]))*(1-self.xi[feature]))
            chi_2_new = like_split2*(1-self.pi[subj_id])/(like_no_split*self.xi[feature] + (like_split1*self.pi[subj_id] + like_split2*(1-self.pi[subj_id]))*(1-self.xi[feature]))

        else:
            chi_1_new=0
            chi_2_new=0

        return chi_1_new, chi_2_new


    def em_step_pi(self):
        """
        Executes the Expectation-Maximization (EM) step to update the pi probabilities based on computed chi values.

        Returns:
        -------
        None
            The method updates the `self.pi` attribute in place based on the calculated chi values.

        Details:
        -------
        - The function calculates the chi values for all subjects and features by calling the `chi` method.
        - It extracts the first chi values from the resulting tensor to compute probabilities for the split model:
            aux_aux_chi = chi_1 / (chi_1 + chi_2)
        - Any NaN values in the resulting probabilities are set to 0.
        - The updated `pi` probabilities are calculated as the mean of the computed probabilities across all features for each subject.
        """
        aux_chi=torch.tensor([[self.chi(subj_id, feature) for feature in range(self.n_features)] 
                               for subj_id in range(self.n_subjects)], dtype=torch.float32, device=self.device)

        aux_aux_chi = aux_chi[:,:,0]/(aux_chi[:,:,0]+aux_chi[:,:,1])
        aux_aux_chi[torch.isnan(aux_aux_chi)] = 0
  
        self.pi=(aux_aux_chi).mean(axis=1)

    
    def log_prior(self, outer_iter):
        """
        Computes the log prior probabilities for the model parameters, incorporating regularization terms.

        Parameters:
        ----------
        outer_iter : int
            The current iteration number of the outer optimization loop, used to conditionally apply regularization to theta.

        Returns:
        -------
        log_prior_value : torch.Tensor
            The total log prior probability for the model parameters, summing the contributions from xi, noise standard deviation, 
            and time shift.

        Details:
        -------
        - The function calculates the log prior contributions for:
            - Noise standard deviations: A regularization term penalizing large values of `log_noise_std`.
            - xi probabilities: A regularization term encouraging the sum of `xi` to be close to the number of features.
            - Time shift: A regularization term that penalizes the squared values of `time_shift` if `time_shift_eval` is True.
        - The function returns the total log prior probability as a sum of the calculated contributions.
        """
        log_prior_noise=-self.lambda_reg_noise*torch.sum(self.log_noise_std+1/torch.exp(self.log_noise_std))
        log_prior_xi=-self.lambda_reg_xi*(self.n_features-self.xi.sum())
        log_prior_time_shift=-((self.time_shift/60)**2).sum() if self.time_shift_eval else 0

        #rate_grouth=torch.tensor([self.theta[f'{self.name_biomarkers[fdx]}_{sdx}_split'][ndx].cpu().detach()[1] for fdx in range(self.n_features) for sdx in range(2) for ndx in range(sdx+1)])
        #log_prior_theta=-torch.sum(torch.log(torch.abs(rate_grouth)))
        #log_prior_theta=-1/self.lambda_reg_theta*torch.sum(torch.square((torch.abs(rate_grouth)-3))) if outer_iter<10 and self.lambda_reg_theta>0 else torch.tensor([0], dtype=torch.float32, device=self.device)

        return log_prior_xi+log_prior_noise+log_prior_time_shift#+log_prior_theta
    

    def loss_eval(self, outer_iter):
        """
        Evaluates the loss function for the model by combining the negative log likelihood and log prior probabilities.

        Parameters:
        ----------
        outer_iter : int
            The current iteration number of the outer optimization loop, which may influence prior calculations.

        Returns:
        -------
        loss_value : torch.Tensor
            The computed loss value, representing the negative of the sum of the log likelihood and log prior.

        Details:
        -------
        - The function calculates the log likelihood using the `log_like` method and the log prior using the `log_prior` method.
        - It then computes the loss as the negative of the combined log likelihood and log prior, which is a common formulation in probabilistic models to maximize the posterior distribution.
        - This loss value is used in optimization routines to adjust model parameters during training.
        """
        return -(self.log_like()+self.log_prior(outer_iter))
 

    def print_benchmarks(self, tdx):
        """
        Prints benchmarking information during the optimization process, including loss and model parameters.

        Parameters:
        ----------
        tdx : int
            The current iteration index in the outer optimization loop.

        Returns:
        -------
        None
            The method outputs information to the console and may generate visualizations based on the current state of the model.

        Details:
        -------
        - The function checks if the current iteration (`tdx`) is a multiple of `n_prints`, which determines how often to print updates.
        - If verbose mode is enabled (`self.verbose`), it prints the current iteration number and the most recent loss value from `self.all_loss`.
        - It then iterates through each feature to print the estimated probability of the split for each biomarker, calculated as `1 - xi`.
        - If the noise standard deviation (`log_noise_std`) requires gradients, it prints the exponentiated noise standard deviations for each feature.
        - After printing the information, if benchmarks are enabled (`self.benchmarks`), it calls the `estimates` method to compute estimates and generates a plot using `plot_solution`, saving it if specified.
        """
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


    def optimise(self, n_outer_iterations=30, n_final_iterations=50, 
                n_inner_iterations_time_shift=30, n_inner_iterations_theta=30, n_inner_iterations_noise=30, 
                lr_theta=1e-1, lr_noise=1e-1, lr_time_shift=1e-2, stopping_criteria=True, threshold=1e-3):
        """
        Optimizes model parameters using an iterative Expectation-Maximization (EM) approach.

        Parameters:
        ----------
        n_outer_iterations : int, optional
            Number of outer iterations for the optimization process (default is 30).
            
        n_final_iterations : int, optional
            Number of final iterations for the E-step optimization (default is 50).
            
        n_inner_iterations_time_shift : int, optional
            Number of inner iterations for optimizing the time shift parameters (default is 30).
            
        n_inner_iterations_theta : int, optional
            Number of inner iterations for optimizing the regression parameters (theta) (default is 30).
            
        n_inner_iterations_noise : int, optional
            Number of inner iterations for optimizing the noise parameters (default is 30).
            
        lr_theta : float, optional
            Learning rate for the theta optimizer (default is 1e-1).
            
        lr_noise : float, optional
            Learning rate for the noise standard deviation optimizer (default is 1e-1).
            
        lr_time_shift : float, optional
            Learning rate for the time shift optimizer (default is 1e-2).
            
        stopping_criteria : bool, optional
            Whether to apply stopping criteria based on convergence (default is True).
            
        threshold : float, optional
            Convergence threshold for the stopping criteria (default is 1e-3).

        Returns:
        -------
        None
            The method updates the model parameters in place and logs the optimization process.

        Details:
        -------
        - The function initializes various parameters and optimizers for the regression coefficients, noise standard deviation, and time shift.
        - It enters an outer loop where it performs the Expectation step for the parameters `xi` and `pi`.
        - For each parameter type (time shift, theta, and noise), it performs inner optimization loops using gradient descent.
        - Loss values are computed and appended to `self.all_loss`, which is monitored for convergence based on the specified stopping criteria.
        - The function provides verbose logging of the optimization process and prints benchmarking information at specified intervals.
        - After completing the outer iterations, it performs final E-steps for `xi` and `pi`, followed by a final estimation of model parameters.
        - The total elapsed time for the optimization process is recorded and optionally printed if verbosity is enabled.
        """

        # Initialize parameters
        self.n_outer_iterations = n_outer_iterations
        self.n_final_iterations = n_final_iterations
        self.n_inner_iterations_time_shift = n_inner_iterations_time_shift
        self.n_inner_iterations_theta = n_inner_iterations_theta
        self.n_inner_iterations_noise = n_inner_iterations_noise
                
        self.lr_theta = lr_theta
        self.lr_noise = lr_noise
        self.lr_time_shift = lr_time_shift

        self.stopping_criteria=stopping_criteria
        self.threshold = threshold

        # Timer
        start = time.time()
        
        # Optimizer Initialization
        if self.theta_eval: optimizer_theta = torch.optim.Adam(self.regression_parameters, lr=lr_theta)
        else: optimizer_theta = None

        if self.noise_std_eval: optimizer_noise = torch.optim.Adam([self.log_noise_std], lr=lr_noise)
        else: optimizer_noise = None

        if self.time_shift_eval: optimizer_time_shift = torch.optim.Adam([self.time_shift], lr=lr_time_shift)
        else: optimizer_time_shift = None

        # Optimization Loop
        for tdx in range(n_outer_iterations):
            # Expectation Step for xi and pi
            if self.xi_eval: self.em_step_xi()
            if self.pi_eval: self.em_step_pi()

            # Time-shift Optimization
            if self.time_shift_eval:
                for epoch in range(n_inner_iterations_time_shift):
                    loss_time_shift = self.loss_eval(tdx)
                    if torch.isnan(loss_time_shift):
                        print('Attention!!! loss time shift is nan')
                        break
                    self.all_loss.append(loss_time_shift.item())
                    optimizer_time_shift.zero_grad()
                    loss_time_shift.backward()
                    optimizer_time_shift.step()

            # Parameter Optimization
            if self.theta_eval:
                for epoch in range(n_inner_iterations_theta):
                    loss = self.loss_eval(outer_iter=tdx)
                    if torch.isnan(loss):
                        print('Attention!!! loss theta is nan')
                        break
                    self.all_loss.append(loss.item())
                    optimizer_theta.zero_grad()
                    loss.backward()
                    optimizer_theta.step()
                    if self.stopping_criteria and epoch > 1 and np.abs(self.all_loss[-1] - self.all_loss[-2]) / self.all_loss[-2] < threshold:
                        break

            # Noise Optimization
            if self.noise_std_eval:
                for epoch in range(n_inner_iterations_noise):
                    loss = self.loss_eval(outer_iter=tdx)
                    if torch.isnan(loss):
                        print('Attention!!! loss noise is nan')
                        break
                    self.all_loss.append(loss.item())
                    optimizer_noise.zero_grad()
                    loss.backward()
                    optimizer_noise.step()
                    if self.stopping_criteria and epoch > 1 and np.abs(self.all_loss[-1] - self.all_loss[-2]) / self.all_loss[-2] < threshold:
                        break
                    
            # Verbose Logging
            if self.verbose:
                self.print_benchmarks(tdx)


        for tdx in range(n_final_iterations):
            if self.xi_eval: self.em_step_xi()
            if self.pi_eval: self.em_step_pi()

        # Final Estimation
        self.estimates()
        self.time_elapsed = time.time() - start

        if self.verbose: print(f'Elapsed time: {self.time_elapsed:.4f}s')


    def estimates(self):
        """
        Computes estimates of model parameters and classifications for subpopulations based on the optimized parameters.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
            The method updates the following attributes in place:
            - `est_num`: Estimated number of splits for each feature.
            - `est_theta`: Estimated parameters (theta) for each feature based on the split.
            - `est_subpop`: Estimated subpopulation classifications for each sample.
            - `est_time`: Estimated time values adjusted for time shift for each sample.
            - `est_noise`: Estimated noise standard deviations for each feature.

        Details:
        -------
        - `est_num`: For each feature, determines if it has a split based on the value of `xi`. If `xi` is greater than 0.5, the feature is considered to have 1 split; otherwise, it is considered to have 2 splits.
        
        - `est_theta`: Constructs a list of estimated parameters for each feature based on the number of splits determined in `est_num`. The parameters are retrieved from the model's `theta` attribute for the appropriate split.

        - `est_subpop`: Initializes an array to hold estimated subpopulation classifications for each sample. For each sample, it checks the value of `pi` for the corresponding subject. If `pi` is greater than 0.5 and the maximum number of splits across features is greater than 1, it assigns the sample to subpopulation 0; otherwise, it assigns it to subpopulation 1.

        - `est_time`: Initializes an array to hold adjusted time values for each sample. The adjusted time is computed by adding the original time `t` with the time shift for the corresponding subject.

        - `est_noise`: Computes the estimated noise standard deviation for each feature by exponentiating the `log_noise_std` values, which are stored in the model.

        - The method does not return any values, but it updates various attributes of the class instance that can be used for further analysis or reporting.
        """

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


    def save(self, name_path='dpmost_sol'):
        """
        Saves the current state of the object to a file using Python's pickle module.

        Parameters:
        ----------
        name_path : str, optional
            The name of the file (without extension) to which the object will be saved. 
            Default is 'dpmost_sol'.

        Returns:
        -------
        None
            This method does not return any values.

        Details:
        -------
        - The method creates a binary file with the specified `name_path` and 
        the '.pkl' extension. The object's state, including all attributes and 
        methods, is serialized and stored in this file.

        - Using `pickle` allows for easy saving and loading of Python objects, 
        facilitating the preservation of the current model state for future use 
        or analysis.

        - If the file already exists, it will be overwritten without warning.
        """
        with open(f'{name_path}.pkl', 'wb') as f:
            pickle.dump(self, f)
