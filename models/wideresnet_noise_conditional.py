# Code adapted from https://github.com/google-research/google-research/tree/master/flax_models/cifar
# Original copyright statement:
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

"""Wide Resnet Model.

Reference:

Wide Residual Networks, Sergey Zagoruyko, Nikos Komodakis
https://arxiv.org/abs/1605.07146

Initially forked from
github.com/google/flax/blob/master/examples/cifar10/models/wideresnet.py

This implementation mimics the one from
github.com/tensorflow/models/blob/master/research/autoaugment/wrn.py
that is widely used as a benchmark.

It uses identity + zero padding skip connections, with kaiming normal
initialization for convolutional kernels (mode = fan_out, gain=2.0).
The final dense layer uses a uniform distribution U[-scale, scale] where
scale = 1 / sqrt(num_classes) as per the autoaugment implementation.

Using the default initialization instead gives error rates approximately 0.5%
greater on cifar100, most likely because the parameters used in the literature
were finetuned for this particular initialization.

Finally, the autoaugment implementation adds more residual connections between
the groups (instead of just between the blocks as per the original paper and
most implementations). It is possible to safely remove those connections without
degrading the performance, which we do by default to match the original
wideresnet paper. Setting `use_additional_skip_connections` to True will add
them back and then reproduces exactly the model used in autoaugment.
"""

import numpy as np

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Any, Tuple, Optional

_BATCHNORM_MOMENTUM = 0.9
_BATCHNORM_EPSILON = 1e-5

# Kaiming initialization with fan out mode. Should be used to initialize
# convolutional kernels.
conv_kernel_init_fn = jax.nn.initializers.variance_scaling(
  2.0, 'fan_out', 'normal')


def dense_layer_init_fn(key,
                        shape,
                        dtype=jnp.float32):
  """Initializer for the final dense layer.

  Args:
    key: PRNG key to use to sample the weights.
    shape: Shape of the tensor to initialize.
    dtype: Data type of the tensor to initialize.

  Returns:
    The initialized tensor.
  """
  num_units_out = shape[1]
  unif_init_range = 1.0 / (num_units_out) ** (0.5)
  return jax.random.uniform(key, shape, dtype, -1) * unif_init_range


def shake_shake_train(xa,
                      xb,
                      rng=None):
  """Shake-shake regularization in training mode.

  Shake-shake regularization interpolates between inputs A and B
  with *different* random uniform (per-sample) interpolation factors
  for the forward and backward/gradient passes.

  Args:
    xa: Input, branch A.
    xb: Input, branch B.
    rng: PRNG key.

  Returns:
    Mix of input branches.
  """
  if rng is None:
    rng = flax.nn.make_rng()
  gate_forward_key, gate_backward_key = jax.random.split(rng, num=2)
  gate_shape = (len(xa), 1, 1, 1)

  # Draw different interpolation factors (gate) for forward and backward pass.
  gate_forward = jax.random.uniform(
    gate_forward_key, gate_shape, dtype=jnp.float32, minval=0.0, maxval=1.0)
  gate_backward = jax.random.uniform(
    gate_backward_key, gate_shape, dtype=jnp.float32, minval=0.0, maxval=1.0)
  # Compute interpolated x for forward and backward.
  x_forward = xa * gate_forward + xb * (1.0 - gate_forward)
  x_backward = xa * gate_backward + xb * (1.0 - gate_backward)
  # Combine using stop_gradient.
  return x_backward + jax.lax.stop_gradient(x_forward - x_backward)


def shake_shake_eval(xa, xb):
  """Shake-shake regularization in testing mode.

  Args:
    xa: Input, branch A.
    xb: Input, branch B.

  Returns:
    Mix of input branches.
  """
  # Blend between inputs A and B 50%-50%.
  return (xa + xb) * 0.5


def shake_drop_train(x,
                     mask_prob,
                     alpha_min,
                     alpha_max,
                     beta_min,
                     beta_max,
                     rng=None):
  """ShakeDrop training pass.

  See https://arxiv.org/abs/1802.02375

  Args:
    x: Input to apply ShakeDrop to.
    mask_prob: Mask probability.
    alpha_min: Alpha range lower.
    alpha_max: Alpha range upper.
    beta_min: Beta range lower.
    beta_max: Beta range upper.
    rng: PRNG key (if `None`, uses `flax.nn.make_rng`).

  Returns:
    The regularized tensor.
  """
  if rng is None:
    rng = flax.nn.make_rng()
  bern_key, alpha_key, beta_key = jax.random.split(rng, num=3)
  rnd_shape = (len(x), 1, 1, 1)
  # Bernoulli variable b_l in Eqn 6, https://arxiv.org/abs/1802.02375.
  mask = jax.random.bernoulli(bern_key, mask_prob, rnd_shape)
  mask = mask.astype(jnp.float32)

  alpha_values = jax.random.uniform(
    alpha_key,
    rnd_shape,
    dtype=jnp.float32,
    minval=alpha_min,
    maxval=alpha_max)
  beta_values = jax.random.uniform(
    beta_key, rnd_shape, dtype=jnp.float32, minval=beta_min, maxval=beta_max)
  # See Eqn 6 in https://arxiv.org/abs/1802.02375.
  rand_forward = mask + alpha_values - mask * alpha_values
  rand_backward = mask + beta_values - mask * beta_values
  return x * rand_backward + jax.lax.stop_gradient(
    x * rand_forward - x * rand_backward)


def shake_drop_eval(x,
                    mask_prob,
                    alpha_min,
                    alpha_max):
  """ShakeDrop eval pass.

  See https://arxiv.org/abs/1802.02375

  Args:
    x: Input to apply ShakeDrop to.
    mask_prob: Mask probability.
    alpha_min: Alpha range lower.
    alpha_max: Alpha range upper.

  Returns:
    The regularized tensor.
  """
  expected_alpha = (alpha_max + alpha_min) / 2
  # See Eqn 6 in https://arxiv.org/abs/1802.02375.
  return (mask_prob + expected_alpha - mask_prob * expected_alpha) * x


def activation(x,
               train,
               apply_relu=True,
               name=''):
  x = nn.GroupNorm(name=name, epsilon=1e-5, num_groups=min(x.shape[-1] // 4, 32))(x)
  if apply_relu:
    x = jax.nn.relu(x)
  return x


def _output_add(block_x, orig_x):
  """Add two tensors, padding them with zeros or pooling them if necessary.

  Args:
    block_x: Output of a resnet block.
    orig_x: Residual branch to add to the output of the resnet block.

  Returns:
    The sum of blocks_x and orig_x. If necessary, orig_x will be average pooled
      or zero padded so that its shape matches orig_x.
  """
  stride = orig_x.shape[-2] // block_x.shape[-2]
  strides = (stride, stride)
  if block_x.shape[-1] != orig_x.shape[-1]:
    orig_x = nn.avg_pool(orig_x, strides, strides)
    channels_to_add = block_x.shape[-1] - orig_x.shape[-1]
    orig_x = jnp.pad(orig_x, [(0, 0), (0, 0), (0, 0), (0, channels_to_add)])
  return block_x + orig_x


class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""
  embedding_size: int = 256
  scale: float = 1.0

  @nn.compact
  def __call__(self, x):
    W = self.param('W', jax.nn.initializers.normal(stddev=self.scale), (self.embedding_size,))
    W = jax.lax.stop_gradient(W)
    x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class WideResnetBlock(nn.Module):
  """Defines a single WideResnetBlock."""
  channels: int
  strides: Tuple[int] = (1, 1)
  activate_before_residual: bool = False

  @nn.compact
  def __call__(self, x, temb=None, train=True):
    if self.activate_before_residual:
      x = activation(x, train, name='init_bn')
      orig_x = x
    else:
      orig_x = x

    block_x = x
    if not self.activate_before_residual:
      block_x = activation(block_x, train, name='init_bn')

    block_x = nn.Conv(
      self.channels, (3, 3),
      self.strides,
      padding='SAME',
      use_bias=False,
      kernel_init=conv_kernel_init_fn,
      name='conv1')(block_x)

    if temb is not None:
      block_x += nn.Dense(self.channels)(nn.swish(temb))[:, None, None, :]
    block_x = activation(block_x, train=train, name='bn_2')
    block_x = nn.Conv(
      self.channels, (3, 3),
      padding='SAME',
      use_bias=False,
      kernel_init=conv_kernel_init_fn,
      name='conv2')(block_x)

    return _output_add(block_x, orig_x)


class WideResnetGroup(nn.Module):
  """Defines a WideResnetGroup."""
  blocks_per_group: int
  channels: int
  strides: Tuple[int] = (1, 1)
  activate_before_residual: bool = False

  @nn.compact
  def __call__(self, x, temb=None, train=True):
    for i in range(self.blocks_per_group):
      x = WideResnetBlock(self.channels, self.strides if i == 0 else (1, 1),
                          activate_before_residual=self.activate_before_residual and not i,
                          )(x, temb, train)
    return x


class WideResnet(nn.Module):
  """Defines the WideResnet Model."""
  blocks_per_group: int
  channel_multiplier: int
  num_outputs: int

  @nn.compact
  def __call__(self, x, sigmas, train=True):
    # per image standardization
    N = np.prod(x.shape[1:])
    x = (x - jnp.mean(x, axis=(1, 2, 3), keepdims=True)) / jnp.maximum(jnp.std(x, axis=(1, 2, 3), keepdims=True),
                                                                       1. / np.sqrt(N))
    temb = GaussianFourierProjection(embedding_size=128, scale=16)(jnp.log(sigmas))
    temb = nn.Dense(128 * 4)(temb)
    temb = nn.Dense(128 * 4)(nn.swish(temb))

    x = nn.Conv(16, (3, 3), padding='SAME', name='init_conv', kernel_init=conv_kernel_init_fn, use_bias=False)(x)
    x = WideResnetGroup(self.blocks_per_group, 16 * self.channel_multiplier,
                        activate_before_residual=True)(x, temb, train)
    x = WideResnetGroup(self.blocks_per_group, 32 * self.channel_multiplier, (2, 2))(x, temb, train)
    x = WideResnetGroup(self.blocks_per_group, 64 * self.channel_multiplier, (2, 2))(x, temb, train)
    x = activation(x, train=train, name='pre-pool-bn')
    x = nn.avg_pool(x, x.shape[1:3])
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(self.num_outputs, kernel_init=dense_layer_init_fn)(x)
    return x
