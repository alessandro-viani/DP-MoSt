DP-MoSt is based on the optimization of two complementary problems: (i) estimating an absolute long-term disease time axis from short-term observations, and (ii) identifying along this disease time axis the existence of sub-populations with respective sub-trajectories.

Considering problem (i), for each individual $j$ we define the observations across all biomarkers as $\bm x^j = (\bm x^j_b)_{b=1}^B$; where $\bm x^j_b = (\bm x_b^j(\tilde t_{1}), \ldots, x_{1}^{j}(\tilde t_{k_j}))$ and $B$ is the number of biomarkers. Without loss of generality, for notational convenience, we assume that the sampling times are common among all subjects and biomarkers. To map the individual observations to a common disease time scale, we parameterize the individual time axis via a translation by a time-shift $\delta {\tilde t}_j$, i.e.\;$\bm t_j = \tilde t_{1:k_j} + \delta {\tilde t}_j$.
In this work, we evaluate the time shifts relying on the Gaussian process theory of GPPM \cite{lorenzi2019probabilistic}, which is based on the monotonic description of biomarkers trajectories from normal to pathological stages.

Considering problem (ii), given the measured observations $\bm x= \bm x^{1:J}$, where $J$ is the number of subjects, and the estimated absolute time $\bm t= \bm t_{1:J}$, we define a trajectory mixture model to identify the existence of sub-populations.

To achieve our goal, we assume that the evolution of each biomarker $b$ can be split into multiple sub-trajectories with probability $\xi_b$ (${b=1}, \ldots, B$). Once a sub-trajectory is considered, we assume that each subject $j$ is issued from this trajectory with probability $\pi_j$ (${j=1}, \ldots, J$). We observe that both $\xi = (\xi_b)_{1}^B$ and $\pi = (\pi_j)_1^J$ are independent with respect to time. This allows our model to link information deriving from longitudinal data; we also note that the probability for a subject belonging to one sub-trajectory must be consistent across all the biomarkers.

To simplify the inference process, compatibly with the monotonic assumption of GPPM, we adopt a parametric approach for the disease trajectories assuming that biomarkers follow increasing sigmoidal functions %with parameters $\theta=(\theta_b)_{b=1}^B$
over time. %, where $\theta_b$ is a vector consisting of the supremum, mid-point and rate of growth.
We furthermore assume that the given measures are perturbed by additive Gaussian noise with standard deviation $\sigma=(\sigma_{b})_{b=1}^B$. Given the assumptions above, the posterior distribution for our model can be written as:
\begin{equation}
\begin{split}
	p(\theta, \sigma, \xi, \pi \mid \bm x) \propto p(\theta, \sigma, \xi, \pi)\prod_{j,b} p(\bm x_b^j \mid \theta_b, \sigma_b, \xi_b, \pi_j)
	\end{split}
	\label{eq:post_full}
\end{equation}
where for simplicity we omitted the conditioning on the time points. We observe that Equation \eqref{eq:post_full} implicitly assumes independence between the unknown parameters as well as independence between different subjects and biomarkers.

We can rewrite the equation by expanding the likelihood function in order to highlight the two-level mixture model formulation. In this setting, a first level deals with the sub-trajectory discovery task, while a second one determines the probability of a subject to belong to the sub-trajectory:
\begin{equation}
\begin{split}
	p(\bm x \mid \theta, \sigma, \xi, \pi)=&\prod_{j,b}\biggl[p(\bm x_b^j \mid \theta_b^0, \sigma_b)\xi_b +\\& \left(\pi_jp(\bm x_b^j \mid \theta_b^1, \sigma_b) + (1-\pi_j) p(\bm x_b^j \mid \theta_b^2, \sigma_b) \right)(1-\xi_b)\biggl],
	\end{split}
	\label{eq:like_full}
\end{equation}
where $p(\bm x_b^j \mid \theta_b^i, \sigma_b) = \prod_{\ell=1}^{k_j} \text{NormPDF}\left(x_b^j(t_\ell), f(t_\ell\mid \theta_b^i), \sigma_b\right)$ due to the assumption of additive Gaussian noise, and $f(t_\ell\mid \theta_b^i)$ is a Sigmoid function with parameters $\theta_b^i$ evaluated at $t_\ell$.
