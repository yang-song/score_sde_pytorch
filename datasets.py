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
"""Return training and evaluation/test datasets from config files."""
import os
import yaml

# import jax
import tensorflow as tf
import tensorflow_datasets as tfds


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr

class CropT():
  def __init__(self, size):
    self.size = size

  def fit(self, train_ds):
    return self

  def transform(self, ds):
    return ds.isel(grid_longitude=slice(0, self.size),grid_latitude=slice(0, self.size))

class Standardize():
  def __init__(self, variables):
    self.variables = variables

  def fit(self, train_ds):
    self.means = { var:  train_ds[var].mean().values for var in self.variables }
    self.stds = { var:  train_ds[var].std().values for var in self.variables }

    return self

  def transform(self, ds):
    for var in self.variables:
      ds[var] = (ds[var] - self.means[var])/self.stds[var]

    return ds

class UnitRangeT():
  def __init__(self, variables):
    self.variables = variables

  def fit(self, train_ds):
    self.maxs = { var:  train_ds[var].max().values for var in self.variables }

    return self

  def transform(self, ds):
    for var in self.variables:
      ds[var] = ds[var]/self.maxs[var]

    return ds

  def invert(self, ds):
    for var in self.variables:
      ds[var] = ds[var]*self.maxs[var]

    return ds

class ClipT():
  def __init__(self, variables):
    self.variables = variables

  def fit(self, _train_ds):
    return self

  def transform(self, ds):
    # target pr should be all non-negative so transform is no-op
    return ds

  def invert(self, ds):
    for var in self.variables:
      ds[var] = ds[var].clip(min=0.0)

class SqrtT():
  def __init__(self, variables):
    self.variables = variables

  def fit(self, _train_ds):
    return self

  def transform(self, ds):
    for var in self.variables:
      ds[var] = ds[var]**(0.5)

    return ds

  def invert(self, ds):
    for var in self.variables:
      ds[var] = ds[var]**2

    return ds

class ComposeT():
  def __init__(self, transforms):
    self.transforms = transforms

  def fit_transform(self, train_ds):
    for t in self.transforms:
      train_ds = t.fit(train_ds).transform(train_ds)

    return train_ds

  def transform(self, ds):
    for t in self.transforms:
      ds = t.transform(ds)

    return ds

  def invert(self, ds):
    for t in reversed(self.transforms):
      ds = t.invert(ds)

    return ds

class XRDataset(Dataset):
    def __init__(self, ds, variables):
        self.ds = ds
        self.variables = variables

    def __len__(self):
        return len(self.ds.time)

    def __getitem__(self, idx):
        subds = self.ds.isel(time=idx)

        cond = torch.tensor(np.stack([subds[var].values for var in self.variables], axis=0)).float()

        x = torch.tensor(np.stack([subds["target_pr"].values], axis=0)).float()

        return cond, x

def get_variables(config):
  data_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'moose', 'nc-datasets', config.data.dataset_name)
  with open(os.path.join(data_dirpath, 'ds-config.yml'), 'r') as f:
      ds_config = yaml.safe_load(f)

  variables = [ pred_meta["variable"] for pred_meta in ds_config["predictors"] ]
  target_variables = ["target_pr"]

  return variables, target_variables

def get_transform(config):
  variables, target_variables = get_variables(config)
  data_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'moose', 'nc-datasets', config.data.dataset_name)
  xr_data_train = xr.load_dataset(os.path.join(data_dirpath, 'train.nc'))

  transform = ComposeT([
    CropT(config.data.image_size),
    Standardize(variables),
    UnitRangeT(variables)])
  target_transform = ComposeT([
    SqrtT(target_variables),
    ClipT(target_variables),
    UnitRangeT(target_variables),
  ])
  xr_data_train = transform.fit_transform(xr_data_train)
  xr_data_train = target_transform.fit_transform(xr_data_train)

  return transform, target_transform, xr_data_train

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.round(h * ratio, tf.int32)
  w = tf.round(w * ratio, tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(config, uniform_dequantization=False, evaluation=False, split='val'):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
  # if batch_size % jax.device_count() != 0:
  #   raise ValueError(f'Batch sizes ({batch_size} must be divided by'
  #                    f'the number of devices ({jax.device_count()})')

  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1

  # Create dataset builders for each dataset.
  if config.data.dataset == 'CIFAR10':
    dataset_builder = tfds.builder('cifar10')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'SVHN':
    dataset_builder = tfds.builder('svhn_cropped')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'CELEBA':
    dataset_builder = tfds.builder('celeb_a')
    train_split_name = 'train'
    eval_split_name = 'validation'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = central_crop(img, 140)
      img = resize_small(img, config.data.image_size)
      return img

  elif config.data.dataset == 'LSUN':
    dataset_builder = tfds.builder(f'lsun/{config.data.category}')
    train_split_name = 'train'
    eval_split_name = 'validation'

    if config.data.image_size == 128:
      def resize_op(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = resize_small(img, config.data.image_size)
        img = central_crop(img, config.data.image_size)
        return img

    else:
      def resize_op(img):
        img = crop_resize(img, config.data.image_size)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

  elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
    dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
    train_split_name = eval_split_name = 'train'


  elif config.data.dataset == "XR":
    variables, target_variables = get_variables(config)
    transform, target_transform, xr_data_train = get_transform(config)

    data_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'moose', 'nc-datasets', config.data.dataset_name)
    xr_data_eval = xr.load_dataset(os.path.join(data_dirpath, f'{split}.nc')).isel(time=slice(140))
    xr_data_eval = transform.transform(xr_data_eval)
    xr_data_eval = target_transform.transform(xr_data_eval)

    train_dataset = XRDataset(xr_data_train, variables)
    eval_dataset = XRDataset(xr_data_eval, variables)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size)
    print(train_data_loader.dataset.ds["target_pr"].values[0,0,0])
    return train_data_loader, eval_data_loader, None
  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  # Customize preprocess functions for each dataset.
  if config.data.dataset in ['FFHQ', 'CelebAHQ']:
    def preprocess_fn(d):
      sample = tf.io.parse_single_example(d, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
      data = tf.io.decode_raw(sample['data'], tf.uint8)
      data = tf.reshape(data, sample['shape'])
      data = tf.transpose(data, (1, 2, 0))
      img = tf.image.convert_image_dtype(data, tf.float32)
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
      return dict(image=img, label=None)

  else:
    def preprocess_fn(d):
      """Basic preprocessing function scales data to [0, 1) and randomly flips."""
      img = resize_op(d['image'])
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.

      return dict(image=img, label=d.get('label', None))

  def create_dataset(dataset_builder, split):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
        split=split, shuffle_files=True, read_config=read_config)
    else:
      ds = dataset_builder.with_options(dataset_options)
    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  train_ds = create_dataset(dataset_builder, train_split_name)
  eval_ds = create_dataset(dataset_builder, eval_split_name)
  return train_ds, eval_ds, dataset_builder
