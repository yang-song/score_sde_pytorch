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
"""The NCSNv2 model."""

import flax.linen as nn
import functools

from .utils import get_sigmas, register_model
from .layers import (CondRefineBlock, RefineBlock, ResidualBlock, ncsn_conv3x3,
                     ConditionalResidualBlock, get_act)
from .normalization import get_normalization
import ml_collections

CondResidualBlock = ConditionalResidualBlock
conv3x3 = ncsn_conv3x3


def get_network(config):
  if config.data.image_size < 96:
    return functools.partial(NCSNv2, config=config)
  elif 96 <= config.data.image_size <= 128:
    return functools.partial(NCSNv2_128, config=config)
  elif 128 < config.data.image_size <= 256:
    return functools.partial(NCSNv2_256, config=config)
  else:
    raise NotImplementedError(
      f'No network suitable for {config.data.image_size}px implemented yet.')


@register_model(name='ncsnv2_64')
class NCSNv2(nn.Module):
  """NCSNv2 model architecture."""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, labels, train=True):
    # config parsing
    config = self.config
    nf = config.model.nf
    act = get_act(config)
    normalizer = get_normalization(config)
    sigmas = get_sigmas(config)
    interpolation = config.model.interpolation

    if not config.data.centered:
      h = 2 * x - 1.
    else:
      h = x

    h = conv3x3(h, nf, stride=1, bias=True)
    # ResNet backbone
    h = ResidualBlock(nf, resample=None, act=act, normalization=normalizer)(h)
    layer1 = ResidualBlock(nf, resample=None, act=act, normalization=normalizer)(h)
    h = ResidualBlock(2 * nf, resample='down', act=act, normalization=normalizer)(layer1)
    layer2 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer)(h)
    h = ResidualBlock(2 * nf,
                      resample='down',
                      act=act,
                      normalization=normalizer,
                      dilation=2)(layer2)
    layer3 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer, dilation=2)(h)
    h = ResidualBlock(2 * nf,
                      resample='down',
                      act=act,
                      normalization=normalizer,
                      dilation=4)(layer3)
    layer4 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer, dilation=4)(h)
    # U-Net with RefineBlocks
    ref1 = RefineBlock(layer4.shape[1:3],
                       2 * nf,
                       act=act,
                       interpolation=interpolation,
                       start=True)([layer4])
    ref2 = RefineBlock(layer3.shape[1:3],
                       2 * nf,
                       interpolation=interpolation,
                       act=act)([layer3, ref1])
    ref3 = RefineBlock(layer2.shape[1:3],
                       2 * nf,
                       interpolation=interpolation,
                       act=act)([layer2, ref2])
    ref4 = RefineBlock(layer1.shape[1:3],
                       nf,
                       interpolation=interpolation,
                       act=act,
                       end=True)([layer1, ref3])

    h = normalizer()(ref4)
    h = act(h)
    h = conv3x3(h, x.shape[-1])

    # When using the DDPM loss, no need of normalizing the output
    if config.model.scale_by_sigma:
      used_sigmas = sigmas[labels].reshape(
        (x.shape[0], *([1] * len(x.shape[1:]))))
      return h / used_sigmas
    else:
      return h


@register_model(name='ncsn')
class NCSN(nn.Module):
  """NCSNv1 model architecture."""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, labels, train=True):
    # config parsing
    config = self.config
    nf = config.model.nf
    act = get_act(config)
    normalizer = get_normalization(config, conditional=True)
    sigmas = get_sigmas(config)
    interpolation = config.model.interpolation

    if not config.data.centered:
      h = 2 * x - 1.
    else:
      h = x

    h = conv3x3(h, nf, stride=1, bias=True)
    # ResNet backbone
    h = CondResidualBlock(nf, resample=None, act=act, normalization=normalizer)(h, labels)
    layer1 = CondResidualBlock(nf, resample=None, act=act, normalization=normalizer)(h, labels)
    h = CondResidualBlock(2 * nf,
                          resample='down',
                          act=act,
                          normalization=normalizer)(layer1, labels)
    layer2 = CondResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer)(h, labels)
    h = CondResidualBlock(2 * nf,
                          resample='down',
                          act=act,
                          normalization=normalizer,
                          dilation=2)(layer2, labels)
    layer3 = CondResidualBlock(2 * nf,
                               resample=None,
                               act=act,
                               normalization=normalizer,
                               dilation=2)(h, labels)
    h = CondResidualBlock(2 * nf,
                          resample='down',
                          act=act,
                          normalization=normalizer,
                          dilation=4)(layer3, labels)
    layer4 = CondResidualBlock(2 * nf,
                               resample=None,
                               act=act,
                               normalization=normalizer,
                               dilation=4)(h, labels)
    # U-Net with RefineBlocks
    ref1 = CondRefineBlock(layer4.shape[1:3],
                           2 * nf,
                           act=act,
                           normalizer=normalizer,
                           interpolation=interpolation,
                           start=True)([layer4], labels)
    ref2 = CondRefineBlock(layer3.shape[1:3],
                           2 * nf,
                           normalizer=normalizer,
                           interpolation=interpolation,
                           act=act)([layer3, ref1], labels)
    ref3 = CondRefineBlock(layer2.shape[1:3],
                           2 * nf,
                           normalizer=normalizer,
                           interpolation=interpolation,
                           act=act)([layer2, ref2], labels)
    ref4 = CondRefineBlock(layer1.shape[1:3],
                           nf,
                           normalizer=normalizer,
                           interpolation=interpolation,
                           act=act,
                           end=True)([layer1, ref3], labels)

    h = normalizer()(ref4, labels)
    h = act(h)
    h = conv3x3(h, x.shape[-1])

    # When using the DDPM loss, no need of normalizing the output
    if config.model.scale_by_sigma:
      used_sigmas = sigmas[labels].reshape(
        (x.shape[0], *([1] * len(x.shape[1:]))))
      return h / used_sigmas
    else:
      return h


@register_model(name='ncsnv2_128')
class NCSNv2_128(nn.Module):  # pylint: disable=invalid-name
  """NCSNv2 model architecture for 128px images."""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, labels, train=True):
    # config parsing
    config = self.config
    nf = config.model.nf
    act = get_act(config)
    normalizer = get_normalization(config)
    sigmas = get_sigmas(config)
    interpolation = config.model.interpolation

    if not config.data.centered:
      h = 2 * x - 1.
    else:
      h = x

    h = conv3x3(h, nf, stride=1, bias=True)
    # ResNet backbone
    h = ResidualBlock(nf, resample=None, act=act, normalization=normalizer)(h)
    layer1 = ResidualBlock(nf, resample=None, act=act, normalization=normalizer)(h)
    h = ResidualBlock(2 * nf, resample='down', act=act, normalization=normalizer)(layer1)
    layer2 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer)(h)
    h = ResidualBlock(2 * nf, resample='down', act=act, normalization=normalizer)(layer2)
    layer3 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer)(h)
    h = ResidualBlock(4 * nf,
                      resample='down',
                      act=act,
                      normalization=normalizer,
                      dilation=2)(layer3)
    layer4 = ResidualBlock(4 * nf, resample=None, act=act, normalization=normalizer, dilation=2)(h)
    h = ResidualBlock(4 * nf,
                      resample='down',
                      act=act,
                      normalization=normalizer,
                      dilation=4)(layer4)
    layer5 = ResidualBlock(4 * nf, resample=None, act=act, normalization=normalizer, dilation=4)(h)
    # U-Net with RefineBlocks
    ref1 = RefineBlock(layer5.shape[1:3],
                       4 * nf,
                       interpolation=interpolation,
                       act=act,
                       start=True)([layer5])
    ref2 = RefineBlock(layer4.shape[1:3],
                       2 * nf,
                       interpolation=interpolation,
                       act=act)([layer4, ref1])
    ref3 = RefineBlock(layer3.shape[1:3],
                       2 * nf,
                       interpolation=interpolation,
                       act=act)([layer3, ref2])
    ref4 = RefineBlock(layer2.shape[1:3],
                       nf,
                       interpolation=interpolation,
                       act=act)([layer2, ref3])
    ref5 = RefineBlock(layer1.shape[1:3],
                       nf,
                       interpolation=interpolation,
                       act=act,
                       end=True)([layer1, ref4])

    h = normalizer()(ref5)
    h = act(h)
    h = conv3x3(h, x.shape[-1])

    if config.model.scale_by_sigma:
      used_sigmas = sigmas[labels].reshape(
        (x.shape[0], *([1] * len(x.shape[1:]))))
      return h / used_sigmas
    else:
      return h


@register_model(name='ncsnv2_256')
class NCSNv2_256(nn.Module):  # pylint: disable=invalid-name
  """NCSNv2 model architecture for 128px images."""
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, x, labels, train=True):
    # config parsing
    config = self.config
    nf = config.model.nf
    act = get_act(config)
    normalizer = get_normalization(config)
    sigmas = get_sigmas(config)
    interpolation = config.model.interpolation

    if not config.data.centered:
      h = 2 * x - 1.
    else:
      h = x

    h = conv3x3(h, nf, stride=1, bias=True)
    # ResNet backbone
    h = ResidualBlock(nf, resample=None, act=act, normalization=normalizer)(h)
    layer1 = ResidualBlock(nf, resample=None, act=act, normalization=normalizer)(h)
    h = ResidualBlock(2 * nf, resample='down', act=act, normalization=normalizer)(layer1)
    layer2 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer)(h)
    h = ResidualBlock(2 * nf, resample='down', act=act, normalization=normalizer)(layer2)
    layer3 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer)(h)
    h = ResidualBlock(2 * nf, resample='down', act=act, normalization=normalizer)(layer3)
    layer31 = ResidualBlock(2 * nf, resample=None, act=act, normalization=normalizer)(h)
    h = ResidualBlock(4 * nf,
                      resample='down',
                      act=act,
                      normalization=normalizer,
                      dilation=2)(layer31)
    layer4 = ResidualBlock(4 * nf, resample=None, act=act, normalization=normalizer, dilation=2)(h)
    h = ResidualBlock(4 * nf,
                      resample='down',
                      act=act,
                      normalization=normalizer,
                      dilation=4)(layer4)
    layer5 = ResidualBlock(4 * nf, resample=None, act=act, normalization=normalizer, dilation=4)(h)
    # U-Net with RefineBlocks
    ref1 = RefineBlock(layer5.shape[1:3],
                       4 * nf,
                       interpolation=interpolation,
                       act=act,
                       start=True)([layer5])
    ref2 = RefineBlock(layer4.shape[1:3],
                       2 * nf,
                       interpolation=interpolation,
                       act=act)([layer4, ref1])
    ref31 = RefineBlock(layer31.shape[1:3],
                        2 * nf,
                        interpolation=interpolation,
                        act=act)([layer31, ref2])
    ref3 = RefineBlock(layer3.shape[1:3],
                       2 * nf,
                       interpolation=interpolation,
                       act=act)([layer3, ref31])
    ref4 = RefineBlock(layer2.shape[1:3],
                       nf,
                       interpolation=interpolation,
                       act=act)([layer2, ref3])
    ref5 = RefineBlock(layer1.shape[1:3],
                       nf,
                       interpolation=interpolation,
                       act=act,
                       end=True)([layer1, ref4])

    h = normalizer()(ref5)
    h = act(h)
    h = conv3x3(h, x.shape[-1])

    if config.model.scale_by_sigma:
      used_sigmas = sigmas[labels].reshape(
        (x.shape[0], *([1] * len(x.shape[1:]))))
      return h / used_sigmas
    else:
      return h
