"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import os
import abc
from re import I, T
import torch
import numpy as np
import logging
import scipy
import h5py

class importance_sampler():
  def __init__(self, N, h=50):
    """Construct a loss scaler for importance sampling
     
    Args:
      N: number of discretization time steps.
      h: number of historic steps to average over
    """
    
    self.N = N
    self.history_2 = np.ones((N, h)) + np.nan
    self.pt = np.ones(N, dtype=np.float) / N
    self.t = np.linspace(0,1,num=self.N)

  def dump_state(self):
    if "importance_sampler_weighting" in os.environ.get('DEBUG'):
      ds='training/importance/pts'
      try:
        with h5py.File("debug_data.h5", 'a') as F:
          F[ds].resize((F[ds].shape[0] + 1), axis=0)
          F[ds][-1] = self.pt
      except KeyError:
        with h5py.File("debug_data.h5", "a") as F:
          print(self.pt[np.newaxis, :].shape)
          F.create_dataset(ds, data=self.pt[np.newaxis, :], chunks=True, maxshape=(None, self.N))

  def add(self, tee, ell):
    """
       tee ~ [batch_size]
       ell ~ [batch_size]
    """
    tee = tee.cpu().numpy()
    ell = ell.cpu().detach().numpy() 
    for t, L in zip(tee, ell):
      t=int(t)
      self.history_2[t, :] = np.roll(self.history_2[t,:], 1)
      self.history_2[t, 0] = L**2
     
    self.update_pt()

  def update_pt(self):
    pt = np.sqrt(np.nanmean(self.history_2, axis=1))
    
    self.pt = pt / np.nansum(pt)
    #print(self.pt)
    
    #if not np.isnan(pt).any():
    #  self.pt = pt / np.sum(pt)

  def sample_t(self, batch_size, device):
    self.dump_state() #write the state to the debug file if debugging on
    if np.isnan(self.history_2).any():
      mean = np.mean(self.history_2, axis=1)  #will be nan in columns that have Nans
      nan_idx = np.where(np.isnan(mean))[0]
      print(f"There are still NaNs in the history buffer in {len(nan_idx)} columns ({np.isnan(self.history_2).sum()} NaN elements)")
      t_idx = torch.Tensor(np.random.choice(nan_idx, batch_size, replace=True)).to(device)
    else:
      t_idx = torch.multinomial(input=torch.Tensor(self.pt), num_samples=batch_size, replacement=True).to(device)

    slice = len(t_idx) // 2
    t_idx[0:slice] = torch.Tensor(np.random.randint(0, self.N, slice))

    t = t_idx / self.N
    return t, t_idx  #return a sample from [0, 1), weighted by p_t, and original bucket as well


  def get_normalization(self, T_idx):
    if np.isnan(self.history_2).any():
      return torch.ones(len(T_idx)), 0.0
    
    t_ = T_idx.cpu().numpy().astype(int)
    ret_ = torch.tensor(self.pt[t_])

    return ret_, 1.0






















class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N
    #self.importance_sampler = fit_importance_sampler(self.N)
    self.importance_sampler = importance_sampler(self.N, h=10)


  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  


  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        """THIS IS THE ONE THAT IS USED"""
        f, G = discretize_fn(x, t)

        score = score_fn(x, t)
        print("x:", x.shape)
        print("score:", score.shape)
        print("t:", t.shape)

        if "backward_score_evaluations" in os.environ.get('DEBUG'):
          chkpt = os.environ.get("INTERNAL_CURRENT_CHECKPOINT")
          ds = f'backward/score_evaluation_chkpt_{chkpt}/'
          try:
            with h5py.File("debug_data.h5", 'a') as F:
              F[ds+'xs'].resize((F[ds+'xs'].shape[0] + 1), axis=0)
              F[ds+'xs'][-1, :] = x.detach().cpu().numpy()              

              F[ds+'score'].resize((F[ds+'score'].shape[0] + 1), axis=0)
              F[ds+'score'][-1, :] = score.detach().cpu().numpy()              

          except KeyError:
            with h5py.File("debug_data.h5", "a") as F:
              F.create_dataset(ds+'xs', data=x.cpu().numpy()[np.newaxis, :], chunks=True, maxshape=[None,]+list(x.shape))
              F.create_dataset(ds+'score', data=score.cpu().numpy()[np.newaxis, :], chunks=True, maxshape=[None,]+list(score.shape))



        rev_f = f - G[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()

class VPSDE_working_sampling(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
    self.dt = torch.tensor(self.T / self.N)
    #self.betas = torch.linspace(beta_min, beta_max, N)


  @property
  def T(self):
    return 1

  def numerical_sample(self, x0s, ts):
    """
    Params: 
      x0s : a batch of images of shape [batch, width, height, channels]
      ts : a batch of times, each interpreted as being between 0 and sde.T
    
    Numerically evolves the SDE according to the VP-SDE equation, returning
    the "perturbed", or forward-noised samples.   

    The VP-SDE is given by Eq. (32) in https://arxiv.org/pdf/2011.13456.pdf


    Assumes the following are set as class attributes:
      self.beta_0
      self.beta_1
      self.dt 
    
    """
    #we only need to evolve the batch to the greatest value of t in the batch
    maxt = (ts.max().item())  
    t = 0
    xs = torch.clone(x0s).to(x0s.device)
    while t < maxt:
      dw = torch.randn_like(x0s) * torch.sqrt(self.dt) 
      drift, diffusion = self.sde(xs, ts)
      dx = drift*self.dt + diffusion*dw
      #I mask the update so that any batch members that have reached their respective
      #time are not updated further (since dx * False = 0)
      xs = xs + dx*(t<=ts)[:, None, None, None]
      t += self.dt.item()
    
    return xs


  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None, None] * x - x
    G = sqrt_beta
    return f, G


class VPSDE(VPSDE_working_sampling):
  """
  First attempt at CIM SDE
   - for CIM SDE, x = [mu, sigma]
  """
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.N = N
    self.dt = torch.tensor(self.T / self.N) 
    self.j = 10. 
    self._lambda = self.j #Farhad said lambda~j should be good
    self._last_dw = None
    self._prior_params = dict()

    self.debug_sampling_counter = 0


  def sde(self, x, t):
    #x is a stack of mu/sigma
    drift = torch.zeros_like(x)
    diffusion = torch.zeros_like(x)
    j = self.j
    
    # slice out mu and sigma
    mu = x[:,:,:,:,0]
    sigma = x[:,:,:,:,1]

    # make some noise!
    # save it in the object because we'll need to use the
    # same noise for numerical sampling
    Wt = torch.randn_like(x)
    dw = Wt * torch.sqrt(self.dt)
    self._last_dw = dw

    #compute mu_tilde
    mu_tilde = mu + np.sqrt(1.0/(4*j)) * (dw[..., 0] / self.dt)
    
    #drift term, gradient wrt xi of Sum(xi,xj)= sum xj
    #sum_mu = torch.mean(mu_tilde, dim=(1,2,3), keepdim=True)
    #print("sum_mu.shape=", sum_mu.shape)
    
    drift[..., 0] = (-(1+j)*mu + (1+j)*mu_tilde - self._lambda*mu_tilde)
    drift[..., 1] = -2*(1+j)*sigma - 2*j*(sigma-0.5)**2 + (1+j)

    diffusion[..., 0] = np.sqrt(j)*(sigma - 0.5)
    diffusion[..., 1] = torch.zeros_like(sigma)

    return drift, diffusion

  def prior_sampling(self, shape):
    pT = torch.zeros(shape)
    pT[..., 0] = 0.035443030297756195 * (torch.randn(*shape[:-1]))
    pT[..., 1] = 0.50000
    return pT

  def numerical_sample(self, x0s, ts, return_full=False):
    """
    Params: 
      x0s : a batch of images of shape [batch, width, height, channels]
      ts : a batch of times, each interpreted as being between 0 and sde.T
    
    Numerically evolves the SDE according to the VP-SDE equation, returning
    the "perturbed", or forward-noised samples.   

    The VP-SDE is given by Eq. (32) in https://arxiv.org/pdf/2011.13456.pdf


    Assumes the following are set as class attributes:
      self.beta_0
      self.beta_1
      self.dt 
    
    """

    #we only need to evolve the batch to the greatest value of t in the batch
    last_dim = len(x0s.shape)
   # print("x0s", x0s.shape)
    mu_sigma = x0s #stacked in loss function## torch.stack((x0s, torch.zeros_like(x0s)), dim=last_dim)
    #print("mu_sigma", mu_sigma.shape)
    maxt = (ts.max().item())  
    t = 0
    xs = torch.clone(mu_sigma).to(x0s.device)

    debug_all_xs = []
    debug_all_ts = []

    while t < maxt:
#      print("xs_in", xs.shape)
      #torch.randn_like(xs) * torch.sqrt(self.dt)

      drift, diffusion = self.sde(xs, ts)
      
      dw = self._last_dw
      assert dw is not None, "dw shouldn't be None. There's a problem."
#      print("drift", drift.shape)
#      print("diffusion", diffusion.shape)
      dx = drift*self.dt + diffusion*dw
      #I mask the update so that any batch members that have reached their respective
      #time are not updated further (since dx * False = 0)

#      print("dx", dx.shape)
      bcast_mask = (t<=ts)[:, None, None, None, None]
#      print("mask", bcast_mask.shape)

      xs = xs + dx*bcast_mask
#      print("xs_out", xs.shape)
      #error()

      debug_all_xs.append(xs.cpu().numpy()[np.newaxis])
      debug_all_ts.append(np.array([t]))

      t += self.dt.item()
    

    if "full_forward_trajectories" in os.environ.get('DEBUG'):
      if self.debug_sampling_counter % 100 == 0:
        with h5py.File("debug_data.h5", 'a') as F:
          F.create_dataset(f'forward/trajectories/{str(self.debug_sampling_counter).zfill(5)}/xs', data=np.concatenate(debug_all_xs))
          F.create_dataset(f'forward/trajectories/{str(self.debug_sampling_counter).zfill(5)}/ts', data=np.concatenate(debug_all_ts))
    if "terminal_forward_samples" in os.environ.get('DEBUG'):
      try:
        with h5py.File("debug_data.h5", 'a') as F:
          F['forward/terminal/xTs'].resize((F['forward/terminal/xTs'].shape[0] + xs.shape[0]), axis=0)
          F['forward/terminal/xTs'][-xs.shape[0]:] = xs.cpu().numpy()
          F['forward/terminal/ts'].resize((F['forward/terminal/ts'].shape[0] + ts.shape[0]), axis=0)
          F['forward/terminal/ts'][-ts.shape[0]:] = ts.cpu().numpy()
      except KeyError:
        with h5py.File("debug_data.h5", "a") as F:
          F.create_dataset('forward/terminal/xTs', data=xs.cpu().numpy(), chunks=True, maxshape=(None,)*len(xs.shape))
          F.create_dataset('forward/terminal/ts', data=ts.cpu().numpy(), chunks=True, maxshape=(None,)*len(ts.shape))
          

    self.debug_sampling_counter += 1    
    
    if return_full:
      return xs
    else:
      return xs[:,:,:,:,0] #return only mu



class VPSDE_original(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None, None] * x - x
    G = sqrt_beta
    return f, G


class subVPSDE(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct the sub-VP SDE that excels at likelihoods.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
    std = 1 - torch.exp(2. * log_mean_coeff)
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.


class VESDE(SDE):
  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
    """Construct a Variance Exploding SDE.

    Args:
      sigma_min: smallest sigma.
      sigma_max: largest sigma.
      N: number of discretization steps
    """
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def marginal_prob(self, x, t):
    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    mean = x
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape) * self.sigma_max

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

  def discretize(self, x, t):
    """SMLD(NCSN) discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    sigma = self.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas[timestep - 1].to(t.device))
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
    return f, G



















class fit_importance_sampler():
  def __init__(self, N):

    self.N = N
    self.sampler = importance_sampler(N=N, h=25)

    self.guess = [1.0, 6.0, -10.0]  #empirically obtained
    self.params = [i for i in self.guess]

    self.update_freq = 10
    self.update_count = 0

    self.update_pt()

  def template(self, t, a, b, c):
    return a*np.exp(b*np.exp(c*t))
  
  def update_pt(self):
    print("Tuning importance sampler")
    if self.sampler.counter < 4000:
      print(f"  x-> skipping because we have only obtained {self.sampler.counter} samples so far")
      self.pt = self.template(np.linspace(0, 1.0, num=self.N), *self.guess)
      #self.pt = np.ones(self.N) / self.N
      return
    else:
      if self.update_count == self.update_freq:
        self.update_count = 0
        try:
          popt, _ = scipy.optimize.curve_fit(self.template, self.sampler.t, self.sampler.pt, p0=self.guess)
          self.params = popt
        except RuntimeError:
          print("Failure to fit")

    print(f" --> Importance sampler parameters: {self.params}")
    pt = self.template(np.linspace(0,1.0,num=self.N), *self.params)
    self.pt = pt/np.sum(pt)
    
    debug=True 
    if debug:
      with h5py.File("debug_pt.h5","w") as F:
        print(self.sampler.t.shape)
        F.create_dataset("t", data=self.sampler.t)
        F.create_dataset("pt", data=self.sampler.pt)
        F.create_dataset("params", data=self.params)
        F.create_dataset("pt_fit", data=self.pt)

   
  def print_status(self):
    pass

  def add(self, tee, ell):
    """
       tee ~ [batch_size]
       ell ~ [batch_size]
    """

    self.sampler.add(tee, ell)
    
    self.update_pt()
    self.print_status()
  
  def sample_t(self, batch_size, device):
    t_idx = torch.multinomial(input=torch.Tensor(self.pt), num_samples=batch_size, replacement=True).to(device)
    
    
    t_idx[0] = np.random.randint(0,self.N)

    t = t_idx / self.N
    return t, t_idx  #return a sample from [0, 1), weighted by p_t, and original bucket as well

  def pluck_pt(self, t):
    nans = False #for compatibility
    t_ = t.cpu().numpy().astype(int)
    return torch.tensor(self.pt[t_]), nans
class static_importance_sampler():
  def __init__(self, N):
    self.N = N
    self.pt = np.ones(N, dtype=np.float) / N
    self.update_pt()

  def add(self, tee, ell):
    pass

  def update_pt(self):
    t = np.linspace(0., 1., num=self.N)
    p = np.exp(-2.*t)
    self.pt = p / np.sum(p)

  def sample_t(self, batch_size, device):
    t_idx = torch.multinomial(input=torch.Tensor(self.pt), num_samples=batch_size, replacement=True).to(device)
    t = t_idx / self.N
    return t, t_idx  #return a sample from [0, 1), weighted by p_t, and original bucket as well

  def pluck_pt(self, t):
    nans = False #for compatibility
    t_ = t.cpu().numpy().astype(int)
    return torch.tensor(self.pt[t_]), nans












