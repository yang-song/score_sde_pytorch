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
from sde_lib import VPSDE


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  return optimizer

def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def get_score_matching_loss_fn(sde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(sde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    
    #Original DDPM loss from above.
    # I want to make sure that this "works" if I change the loss function 
    labels = torch.randint(0, sde.N, (batch.shape[0],), device=batch.device) 
    ts = labels / (sde.N-1)
    #score_fn = mutils.get_score_fn(sde, model, train=train)
    model_fn = mutils.get_model_fn(model, train=train)

    with torch.no_grad():
    #    numeric=True
        #if not numeric:
      sqrt_alphas_cumprod = sde.sqrt_alphas_cumprod.to(batch.device)
      sqrt_1m_alphas_cumprod = sde.sqrt_1m_alphas_cumprod.to(batch.device)
      noise = torch.randn_like(batch)
      perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                       sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
        #elif numeric:
      #perturbed_data = sde.numeric_sample(x0=batch, Tmax=labels)

    if train:
      xs = perturbed_data.detach()
      xs.requires_grad_(True)
      
      nbatch, nchannel, nwidth, nheight = xs.shape

      vectors = torch.randn(nbatch, nchannel*nwidth*nheight).to(xs.device)
      vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)
      
      score = -model_fn(xs, ts) / 

      score = torch.reshape(score, vectors.shape)  # [32, 3072]

      scorev = torch.sum(score * vectors)  #scalar

      loss1 = torch.sum(score * vectors, dim=-1) ** 2 * 0.5
      grad2 = torch.autograd.grad(scorev, xs, create_graph=True)[0]
      grad2 = torch.reshape(grad2, vectors.shape)
      loss2 = torch.sum(vectors * grad2, dim=-1)

      loss = loss1 + loss2
      
      loss = loss.mean(dim=0)

      print("ATTENTION: Using score matching loss function")
      return loss
    else:
        return torch.tensor([0.])
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
  loss_fn = get_score_matching_loss_fn(sde, train, reduce_mean=reduce_mean)

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
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
    else:
      loss = torch.zeros(1)
    return loss

  return step_fn
