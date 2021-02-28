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
# pytype: skip-file
"""Various sampling methods."""
import functools

import jax
import jax.numpy as jnp
import jax.random as random
import abc
import flax

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from utils import batch_mul, batch_add

from models import utils as mutils

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, model, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  model=model,
                                  shape=shape,
                                  inverse_scaler=inverse_scaler,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps)
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_sampler(sde=sde,
                                 model=model,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=inverse_scaler,
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each,
                                 probability_flow=config.sampling.probability_flow,
                                 continuous=config.training.continuous,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, rng, x, t):
    """One update of the predictor.

    Args:
      rng: A JAX random state.
      x: A JAX array representing the current state
      t: A JAX array representing the current time step.

    Returns:
      x: A JAX array of the next state.
      x_mean: A JAX array. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, rng, x, t):
    """One update of the corrector.

    Args:
      rng: A JAX random state.
      x: A JAX array representing the current state
      t: A JAX array representing the current time step.

    Returns:
      x: A JAX array of the next state.
      x_mean: A JAX array. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, rng, x, t):
    dt = -1. / self.rsde.N
    z = random.normal(rng, x.shape)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x + drift * dt
    x = x_mean + batch_mul(diffusion, jnp.sqrt(-dt) * z)
    return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, rng, x, t):
    f, G = self.rsde.discretize(x, t)
    z = random.normal(rng, x.shape)
    x_mean = x - f
    x = x_mean + batch_mul(G, z)
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, rng, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = jnp.where(timestep == 0, jnp.zeros(t.shape), sde.discrete_sigmas[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + batch_mul(score, sigma ** 2 - adjacent_sigma ** 2)
    std = jnp.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = random.normal(rng, x.shape)
    x = x_mean + batch_mul(std, noise)
    return x, x_mean

  def vpsde_update_fn(self, rng, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
    beta = sde.discrete_betas[timestep]
    score = self.score_fn(x, t)
    x_mean = batch_mul((x + batch_mul(beta, score)), 1. / jnp.sqrt(1. - beta))
    noise = random.normal(rng, x.shape)
    x = x_mean + batch_mul(jnp.sqrt(beta), noise)
    return x, x_mean

  def update_fn(self, rng, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(rng, x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(rng, x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, rng, x, t):
    return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, rng, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
      alpha = sde.alphas[timestep]
    else:
      alpha = jnp.ones_like(t)

    def loop_body(step, val):
      rng, x, x_mean = val
      grad = score_fn(x, t)
      rng, step_rng = jax.random.split(rng)
      noise = jax.random.normal(step_rng, x.shape)
      grad_norm = jnp.linalg.norm(
        grad.reshape((grad.shape[0], -1)), axis=-1).mean()
      grad_norm = jax.lax.pmean(grad_norm, axis_name='batch')
      noise_norm = jnp.linalg.norm(
        noise.reshape((noise.shape[0], -1)), axis=-1).mean()
      noise_norm = jax.lax.pmean(noise_norm, axis_name='batch')
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + batch_mul(step_size, grad)
      x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x))
    return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, rng, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).astype(jnp.int32)
      alpha = sde.alphas[timestep]
    else:
      alpha = jnp.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    def loop_body(step, val):
      rng, x, x_mean = val
      grad = score_fn(x, t)
      rng, step_rng = jax.random.split(rng)
      noise = jax.random.normal(step_rng, x.shape)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + batch_mul(step_size, grad)
      x = x_mean + batch_mul(noise, jnp.sqrt(step_size * 2))
      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, n_steps, loop_body, (rng, x, x))
    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, rng, x, t):
    return x, x


def shared_predictor_update_fn(rng, state, x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(rng, x, t)


def shared_corrector_update_fn(rng, state, x, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(rng, x, t)


def get_pc_sampler(sde, model, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of a time-dependent score-based model.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.

  Returns:
    A sampling function that takes random states, and a replcated training state and returns samples.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          model=model,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          model=model,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)
  def pc_sampler(rng, state):
    """ The PC sampler funciton.

    Args:
      rng: A JAX random state
      state: A `flax.struct.dataclass` object that represents the training state of a score-based model.
    Returns:
      Samples
    """
    # Initial sample
    rng, step_rng = random.split(rng)
    x = sde.prior_sampling(step_rng, shape)
    timesteps = jnp.linspace(sde.T, eps, sde.N)

    def loop_body(i, val):
      rng, x, x_mean = val
      t = timesteps[i]
      vec_t = jnp.ones(shape[0]) * t
      rng, step_rng = random.split(rng)
      x, x_mean = corrector_update_fn(step_rng, state, x, vec_t)
      rng, step_rng = random.split(rng)
      x, x_mean = predictor_update_fn(step_rng, state, x, vec_t)
      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, sde.N, loop_body, (rng, x, x))
    # Denoising is equivalent to running one predictor step without adding noise.
    return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  return jax.pmap(pc_sampler, axis_name='batch')


def get_ode_sampler(sde, model, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.

  Returns:
    A sampling function that takes random states, and a replicated training state and returns samples.
  """

  @jax.pmap
  def denoise_update_fn(rng, state, x):
    score_fn = get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = jnp.ones((x.shape[0],)) * eps
    _, x = predictor_obj.update_fn(rng, x, vec_eps)
    return x

  @jax.pmap
  def drift_fn(state, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, state.params_ema, state.model_state, train=False, continuous=True)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def ode_sampler(prng, pstate, z=None):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      prng: An array of random state. The leading dimension equals the number of devices.
      pstate: Replicated training state for running on multiple devices.
      z: If present, generate samples from latent code `z`.
    """
    # Initial sample
    rng = flax.jax_utils.unreplicate(prng)
    rng, step_rng = random.split(rng)
    if z is None:
      # If not represent, sample the latent code from the prior distibution of the SDE.
      x = sde.prior_sampling(step_rng, (jax.local_device_count(),) + shape)
    else:
      x = z

    def ode_func(t, x):
      x = from_flattened_numpy(x, (jax.local_device_count(),) + shape)
      vec_t = jnp.ones((x.shape[0], x.shape[1])) * t
      drift = drift_fn(pstate, x, vec_t)
      return to_flattened_numpy(drift)

    # Black-box ODE solver for the probability flow ODE
    solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                   rtol=rtol, atol=atol, method=method)
    nfe = solution.nfev
    x = jnp.asarray(solution.y[:, -1]).reshape((jax.local_device_count(),) + shape)

    # Denoising is equivalent to running one predictor step without adding noise
    if denoise:
      rng, *step_rng = random.split(rng, jax.local_device_count() + 1)
      step_rng = jnp.asarray(step_rng)
      x = denoise_update_fn(step_rng, pstate, x)

    x = inverse_scaler(x)
    return x, nfe

  return ode_sampler
