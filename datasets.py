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
import logging
import os
import pickle
import yaml

from torch.utils.data import DataLoader
import xarray as xr

from ml_downscaling_emulator.training.dataset import CropT, Standardize, UnitRangeT, ClipT, SqrtT, ComposeT, XRDataset

def get_variables(config):
  data_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'moose', 'nc-datasets', config.data.dataset_name)
  with open(os.path.join(data_dirpath, 'ds-config.yml'), 'r') as f:
      ds_config = yaml.safe_load(f)

  variables = [ pred_meta["variable"] for pred_meta in ds_config["predictors"] ]
  target_variables = ["target_pr"]

  return variables, target_variables

def get_transform(config, transform_dir, evaluation=False):
  variables, target_variables = get_variables(config)
  data_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'moose', 'nc-datasets', config.data.dataset_name)
  xr_data_train = xr.load_dataset(os.path.join(data_dirpath, 'train.nc'))

  input_transform_path = os.path.join(transform_dir, 'input.pickle')
  target_transform_path = os.path.join(transform_dir, 'target.pickle')

  if os.path.exists(input_transform_path):
    with open(input_transform_path, 'rb') as f:
      logging.info("Using stored input transform")
      transform = pickle.load(f)
    with open(target_transform_path, 'rb') as f:
      logging.info("Using stored target transform")
      target_transform = pickle.load(f)
    xr_data_train = transform.transform(xr_data_train)
    xr_data_train = target_transform.transform(xr_data_train)
  else:
    transform = ComposeT([
      CropT(config.data.image_size),
      Standardize(variables),
      UnitRangeT(variables)])
    target_transform = ComposeT([
      SqrtT(target_variables),
      ClipT(target_variables),
      UnitRangeT(target_variables),
    ])
    logging.info("Fitting input transform")
    xr_data_train = transform.fit_transform(xr_data_train)
    logging.info("Fitting target transform")
    xr_data_train = target_transform.fit_transform(xr_data_train)

    if not evaluation:
      with open(input_transform_path, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        logging.info("Storing input transform")
        pickle.dump(transform, f, pickle.HIGHEST_PROTOCOL)
      with open(target_transform_path, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        logging.info("Storing target transform")
        pickle.dump(target_transform, f, pickle.HIGHEST_PROTOCOL)

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


def get_dataset(config, transform_dir, uniform_dequantization=False, evaluation=False, split='val'):
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

  # Create dataset builders for each dataset.
  if config.data.dataset == "XR":
    variables, target_variables = get_variables(config)
    transform, target_transform, xr_data_train = get_transform(config, transform_dir, evaluation=evaluation)

    data_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'moose', 'nc-datasets', config.data.dataset_name)
    xr_data_eval = xr.load_dataset(os.path.join(data_dirpath, f'{split}.nc'))
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
