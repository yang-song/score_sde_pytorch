# Score-Based Generative Modeling through Stochastic Differential Equations

This repo contains the official implementation for the paper [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS)

by [Yang Song](https://yang-song.github.io), [Jascha Sohl-Dickstein](http://www.sohldickstein.com/), [Diederik P. Kingma](http://dpkingma.com/), [Abhishek Kumar](http://users.umiacs.umd.edu/~abhishek/), [Stefano Ermon](https://cs.stanford.edu/~ermon/), and [Ben Poole](https://cs.stanford.edu/~poole/)

--------------------

We propose a unified framework that generalizes and improves previous work on score-based generative models through the lens of stochastic differential equations (SDEs). In particular, we can transform data to a simple noise distribution with a continuous-time stochastic process described by an SDE. This SDE can be reversed for sample generation if we know the score of the marginal distributions at each intermediate time step, which can be estimated with score matching. The basic idea is captured in the figure below:

![schematic](assets/schematic.jpg)

Our work enables a better understanding of existing approaches,  new sampling algorithms, exact likelihood computation, uniquely identifiable encoding, latent code manipulation, and brings new conditional generation abilities to the family of score-based generative models.

All combined, we achieved an FID of **2.20** and an Inception score of **9.89** for unconditional generation on CIFAR-10, as well as high-fidelity generation of **1024px** Celeba-HQ images (samples below). In addition, we obtained a likelihood value of **2.99** bits/dim on uniformly dequantized CIFAR-10 images.

![FFHQ samples](assets/ffhq_samples.jpg)

## What does this code do?
Aside from the **NCSN++** and **DDPM++** models in our paper, this codebase also re-implements many previous score-based models all in one place, including **NCSN** from [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600), **NCSNv2** from [Improved Techniques for Training Score-Based Generative Models](https://arxiv.org/abs/2006.09011), and **DDPM** from [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). 

It supports training new models, evaluating the sample quality and likelihoods of existing models. We carefully designed the code to be modular and easily extensible to new SDEs, predictors, or correctors.

## How to run the code

### Dependencies

Run the following to install a subset of necessary python packages for our code
```sh
pip install -r requirements.txt
```

### Usage

Train and evaluate our models through `main.py`.

```sh
main.py:
  --config: Training configuration.
    (default: 'None')
  --eval_folder: The folder name for storing evaluation results
    (default: 'eval')
  --mode: <train|eval>: Running mode: train or eval
  --workdir: Working directory
```

* `config` is the path to the config file. Our prescribed config files are provided in `configs/`. They are formatted according to [`ml_collections`](https://github.com/google/ml_collections) and should be quite self-explanatory.

*  `workdir` is the path that stores all artifacts of one experiment, like checkpoints, samples, and evaluation results.

* `eval_folder` is the name of a subfolder in `workdir` that stores all artifacts of the evaluation process, like meta checkpoints for pre-emption prevention, image samples, and numpy dumps of quantitative results.

* `mode` is either "train" or "eval". When set to "train", it starts the training of a new model, or resumes the training of an old model if its meta-checkpoints (for resuming running after pre-emption in a cloud environment) exist in `workdir` . When set to "eval", it can do an arbitrary combination of the following

  * Evaluate the loss function on the test / validation dataset.

  * Generate a fixed number of samples and compute its Inception score, FID, or KID.

  * Compute the log-likelihood on the training or test dataset.

  These functionalities can be configured through config files, or more conveniently, through the command-line support of the `ml_collections` package. For example, to generate samples and evaluate sample quality, supply the  `--config.eval.enable_sampling` flag; to compute log-likelihoods, supply the `--config.eval.enable_bpd` flag, and specify `--config.eval.dataset=train/test` to indicate whether to compute the likelihoods on the training or test dataset.

## How to extend the code
* **New SDEs**: inherent the `sde_lib.SDE` abstract class and implement all abstract methods. The `discretize()` method is optional and the default is Euler-Maruyama discretization. Existing sampling methods and likelihood computation will automatically work for this new SDE.
* **New predictors**: inherent the `sampling.Predictor` abstract class, implement the `update_fn` abstract method, and register its name with `@register_predictor`. The new predictor can be directly used in `sampling.get_pc_sampler` for Predictor-Corrector sampling, and all other controllable generation methods in `controllable_generation.py`.
* **New correctors**: inherent the `sampling.Corrector` abstract class, implement the `update_fn` abstract method, and register its name with `@register_corrector`. The new corrector can be directly used in `sampling.get_pc_sampler`, and all other controllable generation methods in `controllable_generation.py`.

## Pretrained checkpoints
Link: https://drive.google.com/drive/folders/10pQygNzF7hOOLwP3q8GiNxSnFRpArUxQ?usp=sharing

You may find two checkpoints for some models. The first checkpoint (with a smaller number) is the one that we reported FID scores in Table 3. The second checkpoint (with a larger number) is the one that we reported likelihood values and FIDs of black-box ODE samplers in Table 2. The former corresponds to the smallest FID during the course of training (every 50k iterations). The later is the last checkpoint during training.

Per Google's policy, we cannot release our original CelebA and CelebA-HQ checkpoints. That said, I have re-trained models on FFHQ 1024px, FFHQ 256px and CelebA-HQ 256px with personal resources, and they achieved similar performance to our internal checkpoints. 


## Demonstrations and tutorials
| Link | Description|
|:----:|:-----|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dRR_0gNRmfLtPavX2APzUggBuXyjWW55?usp=sharing)  | Load our pretrained checkpoints and play with sampling, likelihood computation, and controllable synthesis|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SeXMpILhkJPjXUaesvzEhc3Ke6Zl_zxJ?usp=sharing)  | Tutorial of score-based generative models in JAX + FLAX |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing)| Tutorial of score-based generative models in PyTorch |


## References
If you find the code useful for your research, please consider citing
```bib
@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```

This work is built upon some previous papers which might also interest you:

* Song, Yang, and Stefano Ermon. "Generative Modeling by Estimating Gradients of the Data Distribution." *Proceedings of the 33rd Annual Conference on Neural Information Processing Systems*. 2019.
* Song, Yang, and Stefano Ermon. "Improved techniques for training score-based generative models." *Proceedings of the 34th Annual Conference on Neural Information Processing Systems*. 2020.
* Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." *Proceedings of the 34th Annual Conference on Neural Information Processing Systems*. 2020.

