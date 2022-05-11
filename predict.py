import typer




from enum import Enum
from pathlib import Path
import importlib
import os
import re
import yaml
# import functools
# import itertools
import torch
from torch.utils.data import DataLoader
import xarray as xr

from losses import get_optimizer
from models.ema import ExponentialMovingAverage

# import torch.nn as nn
# import numpy as np
# import tensorflow as tf
# import tensorflow_datasets as tfds
# import tensorflow_gan as tfgan
import tqdm

# from ml_downscaling_emulator.training.dataset import XRDataset

from utils import restore_checkpoint

# from configs.subvp import xarray_cncsnpp_continuous
import models
from models import utils as mutils
# from models import ncsnv2
# from models import ncsnpp
from models import cncsnpp
from models import cunet
# from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
# from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
# from sampling import (ReverseDiffusionPredictor,
#                       LangevinCorrector,
#                       EulerMaruyamaPredictor,
#                       AncestralSamplingPredictor,
#                       NoneCorrector,
#                       NonePredictor,
#                       AnnealedLangevinDynamics)
import datasets

app = typer.Typer()

class SDEOption(str, Enum):
    VESDE = "vesde"
    VPSDE = "vpsde"
    subVPSDE = "subvpsde"

def load_model(config, sde, ckpt_filename):
    if sde == SDEOption.VESDE:
        sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    elif sde == SDEOption.VPSDE:
        sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif sde == SDEOption.subVPSDE:
        sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3

    random_seed = 0 #@param {"type": "integer"}

    sigmas = mutils.get_sigmas(config)
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    score_model = mutils.create_model(config)

    optimizer = get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, optimizer=optimizer,
                 model=score_model, ema=ema)

    state = restore_checkpoint(ckpt_filename, state, config.device)
    ema.copy_to(score_model.parameters())

    # Sampling
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                          config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    return score_model, sampling_fn

def generate_samples(sampling_fn, score_model, config, cond_xr, norm_factors, target_norm_factors):
    cond_batch = torch.stack([torch.Tensor(cond_xr[variable].values/nf) for variable, nf in norm_factors.items()], axis=1).to(config.device)

    samples = (sampling_fn(score_model, cond_batch)[0]*target_norm_factors["target_pr"]).unsqueeze(dim=0).cpu().numpy()
    return samples

def generate_predictions(sampling_fn, score_model, config, cond_xr, norm_factors, target_norm_factors, sample_id):
    print("making predictions", flush=True)
    samples = generate_samples(sampling_fn, score_model, config, cond_xr, norm_factors, target_norm_factors)

    coords = dict(cond_xr.coords)#{key: dict(cond_xr.coords)[key] for key in ["time", "grid_longitude", "grid_latitude"]}
    coords = {**coords, "sample_id": ("sample_id", [sample_id])}

    pred_pr_dims=["sample_id", "time", "grid_latitude", "grid_longitude"]
    pred_pr_attrs = {"grid_mapping": "rotated_latitude_longitude", "standard_name": "pred_pr", "units": "kg m-2 s-1"}
    pred_pr_var = (pred_pr_dims, samples, pred_pr_attrs)
    # data_vars =

    data_vars = {key: cond_xr.data_vars[key] for key in ["rotated_latitude_longitude", "time_bnds", "grid_latitude_bnds", "grid_longitude_bnds", "forecast_period_bnds"]}
    data_vars.update({"pred_pr": pred_pr_var})

    samples_ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs={})

    return samples_ds

def load_config(config_name, sde):
    config_path = os.path.join(os.path.dirname(__file__), "configs", re.sub(r'sde$', '', sde.value.lower()), f"{config_name}.py")

    spec = importlib.util.spec_from_file_location("config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()

@app.command()
def main(output_dirpath: Path, data_dirpath: Path, dataset_split: str = "val", sde: SDEOption = SDEOption.subVPSDE, config_name: str = "xarray_cncsnpp_continuous", checkpoint_id: int = typer.Option(...), batch_size: int = 8, num_samples: int = 3):
    workdir = os.path.join(os.getenv("DERIVED_DATA"), "score-sde", "workdirs", sde.value.lower(), config_name, data_dirpath.name)
    config = load_config(config_name, sde)
    config.training.batch_size = batch_size
    config.eval.batch_size = batch_size
    config.data.dataset_name = data_dirpath.name

    ckpt_filename = os.path.join(workdir, "checkpoints", f"checkpoint_{checkpoint_id}.pth")

    score_model, sampling_fn = load_model(config, sde, ckpt_filename)

    # Data
    # data_dirpath = os.path.join(os.getenv('DERIVED_DATA'), 'nc-datasets', '2.2km-coarsened-2x_london_pr_random')
    xr_data_train = xr.load_dataset(os.path.join(data_dirpath, 'train.nc')).isel(grid_longitude=slice(0, config.data.image_size),grid_latitude=slice(0, config.data.image_size))
    xr_data_eval = xr.load_dataset(os.path.join(data_dirpath, f'{dataset_split}.nc')).isel(grid_longitude=slice(0, config.data.image_size),grid_latitude=slice(0, config.data.image_size)).isel(time=slice(10))

    with open(os.path.join(data_dirpath, 'ds-config.yml'), 'r') as f:
        ds_config = yaml.safe_load(f)

    variables = [ pred_meta["variable"] for pred_meta in ds_config["predictors"] ]

    norm_factors = { variable: xr_data_train[variable].max().values.item() for variable in variables }
    target_norm_factors = {"target_pr": xr_data_train["target_pr"].max().values.item()}

    # eval_dl = DataLoader(XRDataset(xr_data_eval, variables=["pr"]), batch_size=config.training.batch_size)
    _, eval_dl, _ = datasets.get_dataset(config, evaluation=True)

    eval_cond_xr = xr_data_eval.isel(time=slice(0,batch_size))
    for sample_id in range(num_samples):
        typer.echo(f"Sample run {sample_id}...")
        preds = [generate_predictions(sampling_fn, score_model, config, xr_data_eval.isel(time=slice(i, i+config.eval.batch_size)), norm_factors, target_norm_factors, sample_id) for i in range(0, len(xr_data_eval.time), config.eval.batch_size)]

        ds = xr.combine_by_coords(preds, compat='no_conflicts', combine_attrs="drop_conflicts", coords="all", join="inner", data_vars="all")


        output_filepath = output_dirpath/f"predictions-{sample_id}.nc"
        typer.echo(f"Saving samples to {output_filepath}...")
        ds.to_netcdf(output_filepath)


if __name__ == "__main__":
    app()
