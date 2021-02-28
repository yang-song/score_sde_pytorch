from models import utils as mutils
import jax.numpy as jnp
import jax
import jax.random as random
from sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
import functools


def get_pc_inpainter(sde, model, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False,
                     denoise=True, eps=1e-5):
  """Create an image inpainting function that uses PC samplers.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for the corrector.
    n_steps: An integer. The number of corrector steps per update of the corrector.
    probability_flow: If `True`, predictor solves the probability flow ODE for sampling.
    continuous: `True` indicates that the score-based model was trained with continuous time.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.

  Returns:
    A pmapped inpainting function.
  """
  # Define predictor & corrector
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

  def get_inpaint_update_fn(update_fn):
    """Modify the update function of predictor & corrector to incorporate data information."""

    def inpaint_update_fn(rng, state, data, mask, x, t):
      rng, step_rng = jax.random.split(rng)
      vec_t = jnp.ones(data.shape[0]) * t
      x, x_mean = update_fn(step_rng, state, x, vec_t)
      masked_data_mean, std = sde.marginal_prob(data, vec_t)
      masked_data = masked_data_mean + jax.random.normal(rng, x.shape) * std[:, None, None, None]
      x = x * (1. - mask) + masked_data * mask
      x_mean = x * (1. - mask) + masked_data_mean * mask
      return x, x_mean

    return inpaint_update_fn

  projector_inpaint_update_fn = get_inpaint_update_fn(predictor_update_fn)
  corrector_inpaint_update_fn = get_inpaint_update_fn(corrector_update_fn)

  def pc_inpainter(rng, state, data, mask):
    """Predictor-Corrector (PC) sampler for image inpainting.

    Args:
      rng: A JAX random state.
      state: A `flax.struct.dataclass` object that contains training state.
      data: A JAX array that represents a mini-batch of images to inpaint.
      mask: A {0, 1} array with the same shape of `data`. Value `1` marks known pixels,
        and value `0` marks pixels that require inpainting.

    Returns:
      Inpainted (complete) images.
    """
    # Initial sample
    rng, step_rng = random.split(rng)
    x = data * mask + sde.prior_sampling(step_rng, data.shape) * (1. - mask)
    timesteps = jnp.linspace(sde.T, eps, sde.N)

    def loop_body(i, val):
      rng, x, x_mean = val
      t = timesteps[i]
      rng, step_rng = random.split(rng)
      x, x_mean = corrector_inpaint_update_fn(step_rng, state, data, mask, x, t)
      rng, step_rng = random.split(rng)
      x, x_mean = projector_inpaint_update_fn(step_rng, state, data, mask, x, t)
      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, sde.N, loop_body, (rng, x, x))
    return inverse_scaler(x_mean if denoise else x)

  return jax.pmap(pc_inpainter, axis_name='batch')


def get_pc_colorizer(sde, model, predictor, corrector, inverse_scaler,
                     snr, n_steps=1, probability_flow=False, continuous=False,
                     denoise=True, eps=1e-5):
  """Create a image colorization function based on Predictor-Corrector (PC) sampling.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score model.
    predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for correctors.
    n_steps: An integer. The number of corrector steps per update of the predictor.
    probability_flow: If `True`, solve the probability flow ODE for sampling with the predictor.
    continuous: `True` indicates that the score-based model was trained with continuous time steps.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The SDE/ODE will start from `eps` to avoid numerical stabilities.

  Returns: A pmapped colorization function.
  """

  # `M` is an orthonormal matrix to decouple image space to a latent space where the gray-scale image
  # occupies a separate channel
  M = jnp.asarray([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 0, 0], [0, 1 / 3, 0]])
  # `invM` is the inverse transformation of `M`
  invM = jnp.asarray([[0, 3, 0], [0, 0, 3], [3, -3, -3]])
  M = jnp.asarray([[5.7735014e-01, -8.1649649e-01, 4.7008697e-08],
                   [5.7735026e-01, 4.0824834e-01, 7.0710671e-01],
                   [5.7735026e-01, 4.0824822e-01, -7.0710683e-01]])
  invM = jnp.linalg.inv(M)

  # Decouple a gray-scale image with `M`
  def decouple(inputs):
    return jnp.einsum('BHWi,ij->BHWj', inputs, M)

  # The inverse function to `decouple`.
  def couple(inputs):
    return jnp.einsum('BHWi,ij->BHWj', inputs, invM)

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

  def get_colorization_update_fn(update_fn):
    "Modify update functions of predictor & corrector to incorporate information of gray-scale images."

    def colorization_update_fn(rng, state, gray_scale_img, x, t):
      mask = get_mask(x)
      rng, step_rng = jax.random.split(rng)
      vec_t = jnp.ones(x.shape[0]) * t
      x, x_mean = update_fn(step_rng, state, x, vec_t)
      masked_data_mean, std = sde.marginal_prob(decouple(gray_scale_img), vec_t)
      masked_data = masked_data_mean + jax.random.normal(rng, x.shape) * std[:, None, None, None]
      x = couple(decouple(x) * (1. - mask) + masked_data * mask)
      x_mean = couple(decouple(x) * (1. - mask) + masked_data_mean * mask)
      return x, x_mean

    return colorization_update_fn

  def get_mask(image):
    mask = jnp.concatenate([jnp.ones_like(image[..., :1]),
                            jnp.zeros_like(image[..., 1:])], axis=-1)
    return mask

  predictor_colorize_update_fn = get_colorization_update_fn(predictor_update_fn)
  corrector_colorize_update_fn = get_colorization_update_fn(corrector_update_fn)

  def pc_colorizer(rng, state, gray_scale_img):
    """Colorize gray-scale images using Predictor-Corrector (PC) sampler.

    Args:
      rng: A JAX random state.
      state: A `flax.struct.dataclass` object that represents the training state.
      gray_scale_img: A minibatch of gray-scale images. Their R,G,B channels have same values.

    Returns:
      Colorized images.
    """
    shape = gray_scale_img.shape
    mask = get_mask(gray_scale_img)
    # Initial sample
    rng, step_rng = random.split(rng)
    x = couple(decouple(gray_scale_img) * mask + \
               decouple(sde.prior_sampling(step_rng, shape) * (1. - mask)))
    timesteps = jnp.linspace(sde.T, eps, sde.N)

    def loop_body(i, val):
      rng, x, x_mean = val
      t = timesteps[i]
      rng, step_rng = random.split(rng)
      x, x_mean = corrector_colorize_update_fn(step_rng, state, gray_scale_img, x, t)
      rng, step_rng = random.split(rng)
      x, x_mean = predictor_colorize_update_fn(step_rng, state, gray_scale_img, x, t)

      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, sde.N, loop_body, (rng, x, x))
    return inverse_scaler(x_mean if denoise else x)

  return jax.pmap(pc_colorizer, axis_name='batch')


def get_pc_conditional_sampler(sde, score_model, classifier, classifier_params, shape,
                               predictor, corrector, inverse_scaler, snr,
                               n_steps=1, probability_flow=False,
                               continuous=False, denoise=True, eps=1e-5):
  """Class-conditional sampling with Predictor-Corrector (PC) samplers.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    score_model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    classifier: A `flax.linen.Module` object that represents the architecture of the noise-dependent classifier.
    classifier_params: A dictionary that contains the weights of the classifier.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for correctors.
    n_steps: An integer. The number of corrector steps per update of the predictor.
    probability_flow: If `True`, solve the probability flow ODE for sampling with the predictor.
    continuous: `True` indicates the score-based model was trained with continuous time.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The SDE/ODE will be integrated to `eps` to avoid numerical issues.

  Returns: A pmapped class-conditional image sampler.
  """
  # A function that gives the logits of the noise-dependent classifier
  logit_fn = mutils.get_logit_fn(classifier, classifier_params)
  # The gradient function of the noise-dependent classifier
  classifier_grad_fn = mutils.get_classifier_grad_fn(logit_fn)

  def conditional_predictor_update_fn(rng, state, x, t, labels):
    """The predictor update function for class-conditional sampling."""
    score_fn = mutils.get_score_fn(sde, score_model, state.params_ema, state.model_state, train=False,
                                   continuous=continuous)

    def total_grad_fn(x, t):
      ve_noise_scale = sde.marginal_prob(x, t)[1]
      return score_fn(x, t) + classifier_grad_fn(x, ve_noise_scale, labels)

    if predictor is None:
      predictor_obj = NonePredictor(sde, total_grad_fn, probability_flow)
    else:
      predictor_obj = predictor(sde, total_grad_fn, probability_flow)
    return predictor_obj.update_fn(rng, x, t)

  def conditional_corrector_update_fn(rng, state, x, t, labels):
    """The corrector update function for class-conditional sampling."""
    score_fn = mutils.get_score_fn(sde, score_model, state.params_ema, state.model_state, train=False,
                                   continuous=continuous)

    def total_grad_fn(x, t):
      ve_noise_scale = sde.marginal_prob(x, t)[1]
      return score_fn(x, t) + classifier_grad_fn(x, ve_noise_scale, labels)

    if corrector is None:
      corrector_obj = NoneCorrector(sde, total_grad_fn, snr, n_steps)
    else:
      corrector_obj = corrector(sde, total_grad_fn, snr, n_steps)
    return corrector_obj.update_fn(rng, x, t)

  def pc_conditional_sampler(rng, score_state, labels):
    """Generate class-conditional samples with Predictor-Corrector (PC) samplers.

    Args:
      rng: A JAX random state.
      score_state: A `flax.struct.dataclass` object that represents the training state
        of the score-based model.
      labels: A JAX array of integers that represent the target label of each sample.

    Returns:
      Class-conditional samples.
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
      x, x_mean = conditional_corrector_update_fn(step_rng, score_state, x, vec_t, labels)
      rng, step_rng = random.split(rng)
      x, x_mean = conditional_predictor_update_fn(step_rng, score_state, x, vec_t, labels)
      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, sde.N, loop_body, (rng, x, x))
    return inverse_scaler(x_mean if denoise else x)

  return jax.pmap(pc_conditional_sampler, axis_name='batch')
