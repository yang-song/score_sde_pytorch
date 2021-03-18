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

# Lint as: python3
"""Training NCSN++ on CelebAHQ with VE SDE."""

import ml_collections
import torch


def get_config():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  training.batch_size = 8
  training.n_iters = 2400001
  training.snapshot_freq = 50000
  training.log_freq = 50
  training.eval_freq = 100
  training.snapshot_freq_for_preemption = 5000
  training.snapshot_sampling = True
  training.sde = 'vesde'
  training.continuous = True
  training.likelihood_weighting = False
  training.reduce_mean = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'
  sampling.probability_flow = False
  sampling.snr = 0.15
  sampling.n_steps_each = 1
  sampling.noise_removal = True

  # eval
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.batch_size = 1024
  evaluate.num_samples = 50000
  evaluate.begin_ckpt = 1
  evaluate.end_ckpt = 96

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CelebAHQ'
  data.image_size = 1024
  data.centered = False
  data.random_flip = True
  data.uniform_dequantization = False
  data.num_channels = 3
  data.tfrecords_path = '/atlas/u/yangsong/celeba_hq/-r10.tfrecords'

  # model
  config.model = model = ml_collections.ConfigDict()
  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.sigma_max = 1348
  model.num_scales = 2000
  model.ema_rate = 0.9999
  model.sigma_min = 0.01
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 16
  model.ch_mult = (1, 2, 4, 8, 16, 32, 32, 32)
  model.num_res_blocks = 1
  model.attn_resolutions = (16,)
  model.dropout = 0.
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3
  model.embedding_type = 'fourier'

  # optim
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.amsgrad = False
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config
