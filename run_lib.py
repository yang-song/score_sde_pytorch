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

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import math
import os
import time
import PIL

import numpy as np
# import tensorflow as tf
# import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
# from models import ddpm, ncsnv2, ncsnpp
from models import unet
from models import cunet
from models import ncsnpp
from models import cncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
# import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint

from ml_downscaling_emulator.utils import cp_model_rotated_pole
import matplotlib.pyplot as plt
import xarray as xr

FLAGS = flags.FLAGS

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)

  image = PIL.Image.open(buf)
  image = torchvision.transforms.ToTensor()(image)#.unsqueeze(0)
  return image

def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # save the config
  config_path = os.path.join(workdir, "config.yml")
  with open(config_path, 'w') as f:
    f.write(config.to_yaml())

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  os.makedirs(sample_dir, exist_ok=True)

  tb_dir = os.path.join(workdir, "tensorboard")
  os.makedirs(tb_dir, exist_ok=True)


  writer = SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  os.makedirs(checkpoint_dir, exist_ok=True)
  os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))

  for step in range(initial_step, num_train_steps + 1):
    try:
      cond_batch, x_batch = train_iter.next()
    except StopIteration:
      train_iter = iter(train_ds)
      cond_batch, x_batch = train_iter.next()

    x_batch = x_batch.to(config.device)
    cond_batch = cond_batch.to(config.device)
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    # batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
    # batch = batch.permute(0, 3, 1, 2)
    x_batch = scaler(x_batch)
    # Execute one training step
    loss = train_step_fn(state, x_batch, cond_batch)
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
      writer.add_scalar("training_loss", loss.cpu().detach(), global_step=step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      # eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
      # TODO not quite a easy to repeatedly cycle through a PyTorch DataLoader compared to a TF dataset
      # eval_batch, _ = next(eval_iter)
      val_set_loss = 0.0
      for eval_cond_batch, eval_x_batch in eval_ds:
        # eval_cond_batch, eval_x_batch = next(iter(eval_ds))
        eval_x_batch = eval_x_batch.to(config.device)
        eval_cond_batch = eval_cond_batch.to(config.device)
        # eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_x_batch = scaler(eval_x_batch)
        eval_loss = eval_step_fn(state, eval_x_batch, eval_cond_batch)

        # Progress
        val_set_loss += eval_loss.item()
        val_set_loss = val_set_loss/len(eval_ds)

      logging.info("step: %d, eval_loss: %.5e" % (step, val_set_loss))
      writer.add_scalar("eval_loss", val_set_loss, global_step=step)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth')
      save_checkpoint(checkpoint_path, state)
      logging.info(f"step: {step}, checkpoint saved to {checkpoint_path}")

      # Generate and save samples
      if config.training.snapshot_sampling:
        logging.info(f"step: {step}, sampling...")
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())

        eval_cond_batch, eval_x_batch = next(iter(eval_ds))
        eval_cond_batch = eval_cond_batch.to(config.device)

        sample, n = sampling_fn(score_model, eval_cond_batch)
        ema.restore(score_model.parameters())
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        os.makedirs(this_sample_dir, exist_ok=True)
        nrow = math.ceil(np.sqrt(sample.shape[0]))

        xr_data = train_ds.dataset.ds
        coords = {"sample_id": np.arange(sample.shape[0]), "grid_longitude": xr_data.coords["grid_longitude"], "grid_latitude": xr_data.coords["grid_latitude"]}
        dims=["sample_id", "grid_latitude", "grid_longitude"]
        ds = xr.Dataset(data_vars={key: xr_data.data_vars[key] for key in ["grid_latitude_bnds", "grid_longitude_bnds", "rotated_latitude_longitude"]}, coords=coords, attrs={})
        ds['pred_pr'] = xr.DataArray(sample.cpu()[:,0].squeeze(1), dims=dims)
        ds['target_pr'] = xr.DataArray(eval_x_batch.cpu()[:,0].squeeze(1), dims=dims)

        fig, axes = plt.subplots(nrow*2, nrow, figsize=(24,24), subplot_kw={'projection': cp_model_rotated_pole})
        if nrow == 1:
          axes = [axes]
        for isample in range(sample.shape[0]):
            ax = axes[(isample // nrow)*2][isample % nrow]
            ax.coastlines()
            ds["pred_pr"].isel(sample_id=isample).plot(ax=ax)
            ax.set_title("Cond generated pr")

            ax = axes[(isample // nrow)*2+1][isample % nrow]
            ax.coastlines()
            ds["target_pr"].isel(sample_id=isample).plot(ax=ax)
            ax.set_title("Target pr")

        with open(os.path.join(this_sample_dir, f"sample.png"), "wb") as fout:
          plt.savefig(fout)

        with open(os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
          np.save(fout, sample.cpu().numpy())

        # writer.add_image("samples", plot_to_image(fig).numpy(), step)

        # with writer.as_default():
        writer.add_image('samples', plot_to_image(fig), global_step=step)
          # tf.summary.image("samples", plot_to_image(fig), step=step)

        # image_grid = make_grid(sample, nrow, padding=2)
        # with tf.io.gfile.GFile(os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
        #   save_image(image_grid, fout)

  writer.flush()
  writer.close()