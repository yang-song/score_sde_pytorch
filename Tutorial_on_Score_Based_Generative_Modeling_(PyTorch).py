#!/usr/bin/env python
# coding: utf-8

import typer

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import os
import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
import matplotlib.pyplot as plt
from scipy import integrate
from torchvision.utils import make_grid
import seaborn as sns
from enum import Enum

from ml_downscaling_emulator.utils import cp_model_rotated_pole

class Device(str, Enum):
  cuda = "cuda"
  cpu = "cpu"

def main(n_epochs: int = 10, dataset_name: str = '2.2km-coarsened-2x_london_pr_random', device: Device = Device.cuda, batch_size: int = 32, lr: float = 1e-4, sampling_steps: int = 500, snr: float = 0.1, error_tolerance: float = 1e-5, sigma:float =  25.0):
  # device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
  # ## size of a mini-batch
  # batch_size =  32 #@param {'type':'integer'}
  # ## learning rate
  # lr=1e-4 #@param {'type':'number'}
  # ## The number of sampling steps.
  # num_steps =  500#@param {'type':'integer'}
  # signal_to_noise_ratio = 0.16 #@param {'type':'number'}
  # ## The error tolerance for the black-box ODE solver
  # error_tolerance = 1e-5 #@param {'type': 'number'}
  num_steps = sampling_steps
  signal_to_noise_ratio = snr

  class XRDataset(Dataset):
      def __init__(self, ds, variables):
          self.ds = ds
          self.variables = variables

      def __len__(self):
          return len(self.ds.time)

      def __getitem__(self, idx):
          subds = self.ds.isel(time=idx)

          X = torch.tensor(np.stack([subds[var].values for var in self.variables], axis=0))
          y = torch.tensor(np.stack([subds["target_pr"].values], axis=0))
          return X, y

  class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
      super().__init__()
      # Randomly sample weights during initialization. These weights are fixed
      # during optimization and are not trainable.
      self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
      x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
      return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


  class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
      super().__init__()
      self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
      return self.dense(x)[..., None, None]

  class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
      """Initialize a time-dependent score-based network.

      Args:
        marginal_prob_std: A function that takes time t and gives the standard
          deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
        channels: The number of channels for feature maps of each resolution.
        embed_dim: The dimensionality of Gaussian random feature embeddings.
      """
      super().__init__()
      # Gaussian random feature embedding layer for time
      self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
          nn.Linear(embed_dim, embed_dim))
      # Encoding layers where the resolution decreases
      self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
      self.dense1 = Dense(embed_dim, channels[0])
      self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
      self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
      self.dense2 = Dense(embed_dim, channels[1])
      self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
      self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
      self.dense3 = Dense(embed_dim, channels[2])
      self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
      self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
      self.dense4 = Dense(embed_dim, channels[3])
      self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

      # Decoding layers where the resolution increases
      self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
      self.dense5 = Dense(embed_dim, channels[2])
      self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
      self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
      self.dense6 = Dense(embed_dim, channels[1])
      self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
      self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
      self.dense7 = Dense(embed_dim, channels[0])
      self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
      self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

      # The swish activation function
      self.act = lambda x: x * torch.sigmoid(x)
      self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
      # Obtain the Gaussian random feature embedding for t
      embed = self.act(self.embed(t))
      # Encoding path
      h1 = self.conv1(x)
      ## Incorporate information from t
      h1 += self.dense1(embed)
      ## Group normalization
      h1 = self.gnorm1(h1)
      h1 = self.act(h1)
      h2 = self.conv2(h1)
      h2 += self.dense2(embed)
      h2 = self.gnorm2(h2)
      h2 = self.act(h2)
      h3 = self.conv3(h2)
      h3 += self.dense3(embed)
      h3 = self.gnorm3(h3)
      h3 = self.act(h3)
      h4 = self.conv4(h3)
      h4 += self.dense4(embed)
      h4 = self.gnorm4(h4)
      h4 = self.act(h4)

      # Decoding path
      h = self.tconv4(h4)
      ## Skip connection from the encoding path
      h += self.dense5(embed)
      h = self.tgnorm4(h)
      h = self.act(h)
      h = self.tconv3(torch.cat([h, h3], dim=1))
      h += self.dense6(embed)
      h = self.tgnorm3(h)
      h = self.act(h)
      h = self.tconv2(torch.cat([h, h2], dim=1))
      h += self.dense7(embed)
      h = self.tgnorm2(h)
      h = self.act(h)
      h = self.tconv1(torch.cat([h, h1], dim=1))

      # Normalize output
      h = h / self.marginal_prob_std(t)[:, None, None, None]
      return h

  def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    # t = torch.tensor(t, device=device)
    t = t.clone().detach().to(device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

  def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device=device)


  def loss_fn(model, x, marginal_prob_std, eps=1e-5):
      """The loss function for training score-based generative models.

      Args:
        model: A PyTorch model instance that represents a
          time-dependent score-based model.
        x: A mini-batch of training data.
        marginal_prob_std: A function that gives the standard deviation of
          the perturbation kernel.
        eps: A tolerance value for numerical stability.
      """
      random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
      z = torch.randn_like(x)
      std = marginal_prob_std(random_t)
      perturbed_x = x + z * std[:, None, None, None]
      score = model(perturbed_x, random_t)
      loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
      return loss

  def Euler_Maruyama_sampler(score_model,
                              marginal_prob_std,
                              diffusion_coeff,
                              batch_size=64,
                              num_steps=num_steps,
                              device='cuda',
                              eps=1e-3):
      """Generate samples from score-based models with the Euler-Maruyama solver.

      Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the standard deviation of
          the perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        num_steps: The number of sampling steps.
          Equivalent to the number of discretized time steps.
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        eps: The smallest time step for numerical stability.

      Returns:
        Samples.
      """
      t = torch.ones(batch_size, device=device)
      init_x = torch.randn(batch_size, 1, 28, 28, device=device)     * marginal_prob_std(t)[:, None, None, None]
      time_steps = torch.linspace(1., eps, num_steps, device=device)
      step_size = time_steps[0] - time_steps[1]
      x = init_x
      with torch.no_grad():
        # for time_step in tqdm.notebook.tqdm(time_steps):
        for time_step in tqdm.tqdm(time_steps):
          batch_time_step = torch.ones(batch_size, device=device) * time_step
          g = diffusion_coeff(batch_time_step)
          mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
          x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
      # Do not include any noise in the last sampling step.
      return mean_x

  def pc_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64,
                num_steps=num_steps,
                snr=signal_to_noise_ratio,
                device='cuda',
                eps=1e-3):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
      score_model: A PyTorch model that represents the time-dependent score-based model.
      marginal_prob_std: A function that gives the standard deviation
        of the perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient
        of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
      device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
      eps: The smallest time step for numerical stability.

    Returns:
      Samples.
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = np.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
      # for time_step in tqdm.notebook.tqdm(time_steps):
      for time_step in tqdm.tqdm(time_steps):
        batch_time_step = torch.ones(batch_size, device=device) * time_step
        # Corrector step (Langevin MCMC)
        grad = score_model(x, batch_time_step)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = np.sqrt(np.prod(x.shape[1:]))
        langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
        x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

        # Predictor step (Euler-Maruyama)
        g = diffusion_coeff(batch_time_step)
        x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
        x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)

      # The last step does not include any noise
      return x_mean

  def ode_sampler(score_model,
                  marginal_prob_std,
                  diffusion_coeff,
                  batch_size=64,
                  atol=error_tolerance,
                  rtol=error_tolerance,
                  device='cuda',
                  z=None,
                  eps=1e-3):
    """Generate samples from score-based models with black-box ODE solvers.

    Args:
      score_model: A PyTorch model that represents the time-dependent score-based model.
      marginal_prob_std: A function that returns the standard deviation
        of the perturbation kernel.
      diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      atol: Tolerance of absolute errors.
      rtol: Tolerance of relative errors.
      device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
      z: The latent code that governs the final sample. If None, we start from p_1;
        otherwise, we start from the given z.
      eps: The smallest time step for numerical stability.
    """
    t = torch.ones(batch_size, device=device)
    # Create the latent code
    if z is None:
      init_x = torch.randn(batch_size, 1, 28, 28, device=device)       * marginal_prob_std(t)[:, None, None, None]
    else:
      init_x = z

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
      """A wrapper of the score-based model for use by the ODE solver."""
      sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
      time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
      with torch.no_grad():
        score = score_model(sample, time_steps)
      return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
      """The ODE function for use by the ODE solver."""
      time_steps = np.ones((shape[0],)) * t
      g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
      return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x

  def prior_likelihood(z, sigma):
    """The likelihood of a Gaussian distribution with mean zero and
        standard deviation sigma."""
    shape = z.shape
    N = np.prod(shape[1:])
    return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1,2,3)) / (2 * sigma**2)

  def ode_likelihood(x,
                    score_model,
                    marginal_prob_std,
                    diffusion_coeff,
                    batch_size=64,
                    device='cuda',
                    eps=1e-5):
    """Compute the likelihood with probability flow ODE.

    Args:
      x: Input data.
      score_model: A PyTorch model representing the score-based model.
      marginal_prob_std: A function that gives the standard deviation of the
        perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient of the
        forward SDE.
      batch_size: The batch size. Equals to the leading dimension of `x`.
      device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
      eps: A `float` number. The smallest time step for numerical stability.

    Returns:
      z: The latent code for `x`.
      bpd: The log-likelihoods in bits/dim.
    """

    # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
    epsilon = torch.randn_like(x)

    def divergence_eval(sample, time_steps, epsilon):
      """Compute the divergence of the score-based model with Skilling-Hutchinson."""
      with torch.enable_grad():
        sample.requires_grad_(True)
        score_e = torch.sum(score_model(sample, time_steps) * epsilon)
        grad_score_e = torch.autograd.grad(score_e, sample)[0]
      return torch.sum(grad_score_e * epsilon, dim=(1, 2, 3))

    shape = x.shape

    def score_eval_wrapper(sample, time_steps):
      """A wrapper for evaluating the score-based model for the black-box ODE solver."""
      sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
      time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
      with torch.no_grad():
        score = score_model(sample, time_steps)
      return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def divergence_eval_wrapper(sample, time_steps):
      """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
      with torch.no_grad():
        # Obtain x(t) by solving the probability flow ODE.
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
        # Compute likelihood.
        div = divergence_eval(sample, time_steps, epsilon)
        return div.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
      """The ODE function for the black-box solver."""
      time_steps = np.ones((shape[0],)) * t
      sample = x[:-shape[0]]
      logp = x[-shape[0]:]
      g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
      sample_grad = -0.5 * g**2 * score_eval_wrapper(sample, time_steps)
      logp_grad = -0.5 * g**2 * divergence_eval_wrapper(sample, time_steps)
      return np.concatenate([sample_grad, logp_grad], axis=0)

    init = np.concatenate([x.cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
    # Black-box ODE solver
    res = integrate.solve_ivp(ode_func, (eps, 1.), init, rtol=1e-5, atol=1e-5, method='RK45')
    zp = torch.tensor(res.y[:, -1], device=device)
    z = zp[:-shape[0]].reshape(shape)
    delta_logp = zp[-shape[0]:].reshape(shape[0])
    sigma_max = marginal_prob_std(1.)
    prior_logp = prior_likelihood(z, sigma_max)
    bpd = -(prior_logp + delta_logp) / np.log(2)
    N = np.prod(shape[1:])
    bpd = bpd / N + 8.
    return z, bpd

  marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
  diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

  score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
  if device == "cuda":
    typer.echo("DataParallelizing score_model...")
    score_model = torch.nn.DataParallel(score_model)
  score_model = score_model.to(device)

  optimizer = Adam(score_model.parameters(), lr=lr)

  # ## Show some samples pre-training
  sample_batch_size = 64 #@param {'type':'integer'}
  sampler = Euler_Maruyama_sampler# ode_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

  ## Generate samples using the specified sampler.
  samples = sampler(score_model,
                    marginal_prob_std_fn,
                    diffusion_coeff_fn,
                    sample_batch_size,
                    device=device)


  data_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'nc-datasets', dataset_name)

  xr_data = xr.load_dataset(os.path.join(data_dirpath, 'train.nc')).isel(grid_longitude=slice(0,28), grid_latitude=slice(0,28))

  dataset = XRDataset(xr_data, ['target_pr'])
  data_loader = DataLoader(dataset, batch_size=batch_size)

  # xarray dataset to hold generated samples
  coords = {"sample_id": np.arange(64), "grid_longitude": xr_data.coords["grid_longitude"], "grid_latitude": xr_data.coords["grid_latitude"]}
  ds = xr.Dataset(data_vars={key: xr_data.data_vars[key] for key in ["grid_latitude_bnds", "grid_longitude_bnds", "rotated_latitude_longitude"]}, coords=coords, attrs={})
  ds['pr'] = xr.DataArray(samples.cpu().squeeze(1), dims=["sample_id", "grid_latitude", "grid_longitude"])



  nrows = int(np.sqrt(sample_batch_size))

  fig, axes = plt.subplots(nrows, nrows, figsize=(24,24), subplot_kw={'projection': cp_model_rotated_pole})

  for sample in range(sample_batch_size):
      ax = axes[sample // nrows][sample % nrows]
      ax.coastlines()
      ds["pr"].isel(sample_id=sample).plot(ax=ax, vmax=ds['pr'].max().values)#, vmin=0)
      ax.set_title("")

  plt.savefig("pre-trained.png")

  # Train model



  # tqdm_epoch = tqdm.notebook.trange(n_epochs)
  tqdm_epoch = tqdm.trange(n_epochs)

  for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    for x, y in data_loader:
      x = x.to(device)
      loss = loss_fn(score_model, x, marginal_prob_std_fn)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      avg_loss += loss.item() * x.shape[0]
      num_items += x.shape[0]
    # Print the averaged training loss so far.
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    # Update the checkpoint after each epoch of training.
    torch.save(score_model.state_dict(), 'ckpt.pth')



  #@title Sampling (double click to expand or collapse)

  ## Load the pre-trained checkpoint from disk.
  # device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
  ckpt = torch.load('ckpt.pth', map_location=device)
  score_model.load_state_dict(ckpt)

  sample_batch_size = 64 #@param {'type':'integer'}
  sampler = Euler_Maruyama_sampler# ode_sampler #@param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}

  ## Generate samples using the specified sampler.
  samples = sampler(score_model,
                    marginal_prob_std_fn,
                    diffusion_coeff_fn,
                    sample_batch_size,
                    device=device)


  ## Sample visualization.
  # samples = samples.clamp(0.0, 1.0)
  sample_grid = make_grid(samples.clamp(0.0), nrow=int(np.sqrt(sample_batch_size)))

  plt.figure(figsize=(16,16))
  plt.axis('off')
  # plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)import seaborn as sns
  sns.heatmap(sample_grid[0].cpu(), cmap='viridis')#.permute(1, 2, 0).cpu())
  plt.show()
  plt.savefig("sample-grid.png")


  xr_data = xr.load_dataset(os.path.join(data_dirpath, 'train.nc')).isel(grid_longitude=slice(0, 28), grid_latitude=slice(0, 28))

  coords = {"sample_id": np.arange(64), "grid_longitude": xr_data.coords["grid_longitude"], "grid_latitude": xr_data.coords["grid_latitude"]}
  ds = xr.Dataset(data_vars={key: xr_data.data_vars[key] for key in ["grid_latitude_bnds", "grid_longitude_bnds", "rotated_latitude_longitude"]}, coords=coords, attrs={})
  ds['pr'] = xr.DataArray(samples.cpu().squeeze(1), dims=["sample_id", "grid_latitude", "grid_longitude"])

  sample_batch_size = 64

  nrows = int(np.sqrt(sample_batch_size))

  fig, axes = plt.subplots(nrows, nrows, figsize=(24,24), subplot_kw={'projection': cp_model_rotated_pole})

  for sample in range(sample_batch_size):
      ax = axes[sample // nrows][sample % nrows]
      ax.coastlines()
      ds["pr"].isel(sample_id=sample).plot(ax=ax)#, vmax=ds['pr'].max().values)#, vmin=0)
      ax.set_title("")

  plt.savefig("post-training samples.png")

  nrows = int(np.sqrt(sample_batch_size))

  fig, axes = plt.subplots(nrows, nrows, figsize=(24,24), subplot_kw={'projection': cp_model_rotated_pole})

  for sample in range(sample_batch_size):
      itime = np.random.randint(0, len(xr_data.time.values))
      ax = axes[sample // nrows][sample % nrows]
      ax.coastlines()
      xr_data["target_pr"].isel(time=itime).plot(ax=ax)#, vmax=1)#sample_grid.max(), vmin=0)
      ax.set_title("")

  plt.savefig("examples.png")

  # ## Likelihood Computation

  #@title Define the likelihood function (double click to expand or collapse)

  #@title Compute likelihood on the dataset (double click to expand or collapse)

  # comment out the likelihood stuff for now

  # batch_size = 32 #@param {'type':'integer'}

  # dataset = MNIST('.', train=False, transform=transforms.ToTensor(), download=True)
  # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

  # ckpt = torch.load('ckpt.pth', map_location=device)
  # score_model.load_state_dict(ckpt)

  # all_bpds = 0.
  # all_items = 0
  # try:
  #   tqdm_data = tqdm.notebook.tqdm(data_loader)
  #   for x, _ in tqdm_data:
  #     x = x.to(device)
  #     # uniform dequantization
  #     x = (x * 255. + torch.rand_like(x)) / 256.
  #     _, bpd = ode_likelihood(x, score_model, marginal_prob_std_fn,
  #                             diffusion_coeff_fn,
  #                             x.shape[0], device=device, eps=1e-5)
  #     all_bpds += bpd.sum()
  #     all_items += bpd.shape[0]
  #     tqdm_data.set_description("Average bits/dim: {:5f}".format(all_bpds / all_items))

  # except KeyboardInterrupt:
  #   # Remove the error message when interuptted by keyboard or GUI.
  #   pass


  # ## Further Resources
  #
  # If you're interested in learning more about score-based generative models, the following papers would be a good start:
  #
  # * Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. "[Score-Based Generative Modeling through Stochastic Differential Equations.](https://arxiv.org/pdf/2011.13456.pdf)" International Conference on Learning Representations, 2021.
  # * Jonathan Ho, Ajay Jain, and Pieter Abbeel. "[Denoising diffusion probabilistic models.](https://arxiv.org/pdf/2006.11239.pdf)" Advances in Neural Information Processing Systems. 2020.
  # *    Yang Song, and Stefano Ermon. "[Improved Techniques for Training Score-Based Generative Models.](https://arxiv.org/pdf/2006.09011.pdf)" Advances in Neural Information Processing Systems. 2020.
  # *   Yang Song, and Stefano Ermon. "[Generative modeling by estimating gradients of the data distribution.](https://arxiv.org/pdf/1907.05600.pdf)" Advances in Neural Information Processing Systems. 2019.
  #
  #

if __name__ == "__main__":
    typer.run(main)
