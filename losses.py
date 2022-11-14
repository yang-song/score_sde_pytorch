# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE, importance_sampler
from tqdm import tqdm
import os
import uuid
import h5py

#import wandb

def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, model, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if os.environ.get('DISABLE_WEIGHT_UPDATE', "false") == "true":
      print("Skipping weight update")
      return
      
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      #params = [p for p in params] #make a copy of the generator contents so we can use it twice
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()

    lipschitz = False
    if lipschitz:
      with torch.no_grad():
        for param in model.parameters():
          param.clamp_(-0.01, 0.01)
      #for param in model.parameters():
      #  print(param.name, param.data.min(), param.data.max())
        

  return optimize_fn

def get_sde_loss_fn_original(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t)

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    return loss

  return loss_fn



def get_dsm_sde_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """


  Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    if not train:
      #I don't care about the loss during eval, especially because we don't want to do autograd during eval mode
      return torch.zeros(1)

    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    #t_frac = torch.rand(batch.shape[0], device=batch.device)

    t_frac, T_idx = sde.importance_sampler.sample_t(batch_size=batch.shape[0], device=batch.device)

    print("Timesteps in this batch: ", T_idx)

    t = t_frac * (sde.T - eps) + eps

    z = torch.randn_like(batch)
    
    with torch.no_grad():
      mean, std = sde.marginal_prob(batch, t)
      perturbed_data = mean + std[:, None, None, None] * z
    perturbed_data = perturbed_data.detach()

    xs = perturbed_data
    nbatch, nchannels, width, height = xs.shape
    xs.requires_grad_(True)

    sigma=0.4 # torch.tanh(-3.0*(1-t_frac))+1
    xs_corrupt = xs + torch.randn_like(xs)*sigma

    score_corrupt = score_fn(xs_corrupt, t)
    grad = 1.0/(sigma**2) * (xs-xs_corrupt)

    grad = grad.reshape((nbatch, nchannels*width*height)) 
    score_corrupt = score_corrupt.reshape((nbatch, nchannels*width*height)) 

    loss = torch.norm(score_corrupt - grad, dim=-1)**2
    
    sde.importance_sampler.add(tee=T_idx, ell=loss)
    normalization, mask = sde.importance_sampler.get_normalization(T_idx)
    loss = loss / normalization.to(batch.device)

    #nans = False 
    #write loss vs. t to disk:
    #os.makedirs("t_sample_dump", exist_ok=True)
    #if not(os.path.exists("loss_vs_t_dump/disable")):
      #data = np.stack((t.cpu(), loss.cpu().detach()), axis=1)
      #np.save(f"loss_vs_t_dump/{uuid.uuid4()}_data", data)
    #np.save(f"t_sample_dump/{uuid.uuid4()}_t", T_idx.cpu().numpy())
    np.save(f"importance_sampling_distribution", sde.importance_sampler.pt)

    if mask == 0.0:
      os.environ['DISABLE_WEIGHT_UPDATE'] = "true"
    else:
      os.environ['DISABLE_WEIGHT_UPDATE'] = "false"


    return loss.mean() * mask  #mask will be zero until the history buffer is full

  return loss_fn



def get_sliced_score_matching_loss_fn(sde, train, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  """Create a loss function for training with arbirary SDEs.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.
    Args:
      model: A score model.
      batch: A mini-batch of training data.
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    #print("Using sliced score matching loss")
    if not train:
      return torch.zeros(1)
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)

#    T_sampling = "linear"
    T_sampling = "importance"
#    T_sampling = "terminal"
#    T_sampling = "uniform"

    if T_sampling == "linear": #override T with linearly space time...no good for training but useful for debugging sampling
      T_idx = torch.linspace(0,sde.N,batch.shape[0], dtype=int).to(batch.device)
      t = T_idx / sde.N
    elif T_sampling == "importance":
      t, T_idx = sde.importance_sampler.sample_t(batch_size=batch.shape[0], device=batch.device)
    elif T_sampling == "terminal":
      T_idx = 999 * torch.ones(batch.shape[0], dtype=int).to(batch.device)
      t = T_idx / sde.N
    elif T_sampling == "uniform":
      T_idx = torch.randint(low=0, high=sde.N, size=(batch.shape[0],), dtype=int).to(batch.device)
      t = T_idx / sde.N
    else:
      raise NotImplementedError

    #print("Timesteps in this batch: ", T_idx)
    
    t = t * (sde.T - eps) + eps

    mu_sigma = torch.stack((batch, torch.zeros_like(batch)), dim=4)

    with torch.no_grad():
      perturbed_data = sde.numerical_sample(x0s=mu_sigma, ts=t)
      #mean, std = sde.marginal_prob(batch, t)
      #z = torch.randn_like(batch)
      #perturbed_data_analytic = mean + std[:, None, None, None] * z

    perturbed_data = perturbed_data.detach()
    xs = perturbed_data


    if torch.max(torch.abs(xs)) > 1000:
      skip_loss_mask = 0.0
      print("WARNING: distribution of data after forward process anomalous.  Skipping weight update for this batch for stability." )
    else:
      skip_loss_mask = 1.0


    xs.requires_grad_(True)

    #FOR CIM SDE, xs is [...,(mu,sigma)], so we only want the first slice of the last dim
    #xs_img = xs[..., 0]
    #`print("xs:", xs.shape, "xs_img:", xs_img.shape)


    score = score_fn(xs, t)



    nbatch, nchannels, width, height = xs.shape

    vectors = torch.randn(nbatch, nchannels*width*height).to(batch.device)
    vectors = vectors.sign()# / torch.norm(vectors, dim=-1, keepdim=True)

    score = torch.reshape(score, vectors.shape)

    scorev = torch.sum(score*vectors)

    grad2 = torch.autograd.grad(scorev, xs, create_graph=True)[0]
    grad2 = grad2.reshape(vectors.shape)
    loss2 = torch.sum(vectors*grad2, dim=-1)
    
    loss1 = torch.sum(score*vectors, dim=-1) ** 2 * 0.5

    loss = loss1 + loss2

    loss_raw = loss*1.0
    
    
    c = 1e5
    loss = loss.clamp(-c, c)
    #loss = loss * torch.exp(-2.5*(1-t))
    #loss = loss.clamp(-2000, 2000)

    loss_clamped = loss*1.0

    
    if T_sampling == "importance":
      normalization, mask = sde.importance_sampler.get_normalization(T_idx)
      sde.importance_sampler.add(tee=T_idx, ell=loss)
      loss = (loss / normalization.to(batch.device))*mask*skip_loss_mask
    else:
      mask=1.0

    #nans = False 
    #write loss vs. t to disk:
    #os.makedirs("t_sample_dump", exist_ok=True)
    #if not(os.path.exists("loss_vs_t_dump/disable")):
      #data = np.stack((t.cpu(), loss.cpu().detach()), axis=1)
      #np.save(f"loss_vs_t_dump/{uuid.uuid4()}_data", data)
    #np.save(f"t_sample_dump/{uuid.uuid4()}_t", T_idx.cpu().numpy())
    #np.save(f"importance_sampling_distribution", sde.importance_sampler.pt)

    if mask == 0.0 or skip_loss_mask == 0.0:
      os.environ['DISABLE_WEIGHT_UPDATE'] = "true"
    else:
      os.environ['DISABLE_WEIGHT_UPDATE'] = "false"

    
    
    if "training_loss" in os.environ.get('DEBUG'):
      ds='training/losses/loss'
      try:
        with h5py.File("debug_data.h5", 'a') as F:
          F[ds].resize((F[ds].shape[0] + 1), axis=0)
          F[ds+"_raw"].resize((F[ds+"_raw"].shape[0] + 1), axis=0)
          F[ds+"_clamped"].resize((F[ds+"_clamped"].shape[0] + 1), axis=0)
          F[ds][-1] = loss.mean().item()
          F[ds+"_raw"][-1] = loss_raw.mean().item()
          F[ds+"_clamped"][-1] = loss_clamped.mean().item()
      except KeyError:
        with h5py.File("debug_data.h5", "a") as F:
          F.create_dataset(ds,            data=np.array([loss.mean().item()]),         chunks=True, maxshape=(None,))
          F.create_dataset(ds+"_raw",     data=np.array([loss_raw.mean().item()]),     chunks=True, maxshape=(None,))
          F.create_dataset(ds+"_clamped", data=np.array([loss_clamped.mean().item()]), chunks=True, maxshape=(None,))
    
    
    
    
    
    return loss.mean() 






  return loss_fn



def get_smld_loss_fn(vesde, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None, None]
    perturbed_data = noise + batch
    score = model_fn(perturbed_data, labels)
    target = -noise / (sigmas ** 2)[:, None, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn


#Select the loss function we want to override with here:
get_sde_loss_fn = get_sliced_score_matching_loss_fn
#get_sde_loss_fn = get_sde_loss_fn_original
#get_sde_loss_fn = get_dsm_sde_loss_fn

def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if continuous:
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                              continuous=True, likelihood_weighting=likelihood_weighting)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean, likelihood_weighting=likelihood_weighting) 
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch)
      loss.backward()
      optimize_fn(optimizer, model, step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss

  return step_fn
