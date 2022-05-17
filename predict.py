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

def generate_samples(sampling_fn, score_model, config, cond_batch):
    cond_batch = cond_batch.to(config.device)

    samples = sampling_fn(score_model, cond_batch)[0]
    # drop the feature channel dimension (only have target pr as output)
    samples = samples.squeeze(dim=1)
    # add a dimension for sample_id
    samples = samples.unsqueeze(dim=0)
    # extract numpy array
    samples = samples.cpu().numpy()
    return samples

def generate_predictions(sampling_fn, score_model, config, cond_batch, target_transform, coords, cf_data_vars, sample_id):
    print("making predictions", flush=True)
    samples = generate_samples(sampling_fn, score_model, config, cond_batch)

    coords = {**dict(coords), "sample_id": ("sample_id", [sample_id])}

    pred_pr_dims=["sample_id", "time", "grid_latitude", "grid_longitude"]
    pred_pr_attrs = {"grid_mapping": "rotated_latitude_longitude", "standard_name": "pred_pr", "units": "kg m-2 s-1"}
    pred_pr_var = (pred_pr_dims, samples, pred_pr_attrs)

    data_vars = {**cf_data_vars, "target_pr": pred_pr_var}

    samples_ds = target_transform.invert(xr.Dataset(data_vars=data_vars, coords=coords, attrs={}))
    samples_ds = samples_ds.rename({"target_pr": "pred_pr"})
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
    _, eval_dl, _ = datasets.get_dataset(config, evaluation=True, split=dataset_split)

    xr_data_eval  = eval_dl.dataset.ds

    _, target_transform, _ = datasets.get_transform(config)

    for sample_id in range(num_samples):
        typer.echo(f"Sample run {sample_id}...")
        cf_data_vars = {key: xr_data_eval.data_vars[key] for key in ["rotated_latitude_longitude", "time_bnds", "grid_latitude_bnds", "grid_longitude_bnds", "forecast_period_bnds"]}
        preds = []
        for batch_num, (cond_batch, _) in enumerate(eval_dl):
            typer.echo(f"Working on batch {batch_num}")
            time_idx_start = batch_num*eval_dl.batch_size
            coords = xr_data_eval.isel(time=slice(time_idx_start, time_idx_start+len(cond_batch))).coords

            preds.append(generate_predictions(sampling_fn, score_model, config, cond_batch, target_transform, coords, cf_data_vars, sample_id))

        ds = xr.combine_by_coords(preds, compat='no_conflicts', combine_attrs="drop_conflicts", coords="all", join="inner", data_vars="all")


        output_filepath = output_dirpath/f"predictions-{sample_id}.nc"
        typer.echo(f"Saving samples to {output_filepath}...")
        os.makedirs(output_filepath.parent, exist_ok=True)
        ds.to_netcdf(output_filepath)


if __name__ == "__main__":
    app()
