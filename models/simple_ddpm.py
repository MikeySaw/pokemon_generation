import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


### DEBUG, this fucntion should be moved into a "helper.py" file
def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


class DenoiseDiffusion:
    """
    Implement the denoising scheduler mentioned in the paper:
    "Denosing Diffusion Probabilistic Models" by Jonathan Ho, Ajay Jain, Pieter Abbeel
    Args:
        denoising_model: UNet model for predicting noise at each denoising step
        device: "cuda" or "cpu" or "mps"
        n_steps: Number of denoising steps
    """
    def __init__(self, 
                 denoising_model: nn.Module, 
                 device: torch.device, 
                 n_steps: int=1000):
        super().__init__()
        self.denoising_model = denoising_model
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)

        # In the paper, the author mentioned that alpha + beta = 1
        self.alpha = 1. - self.beta
        # calcaulte the cumulative product of alpha, works as the variance of the noise
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # number of denoising steps, in DDPM this would be 1000 steps
        self.n_steps = n_steps
        # for inference sampling
        self.sigma_sqaure = self.beta

    def q_xt_x0(self, 
                x0: torch.Tensor, 
                t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        To Do: Change the formatting into jax typing
        This function works as the q(x_t|x_0) in the paper
        This is the forward pass of the denoising model
        Args:
            x0: Input image, aka Pokemin image in our case
            t: the number of the denoising step
        """
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, 
                 x0: torch.Tensor, 
                 t: torch.Tensor, 
                 eps: Optional[torch.Tensor] = None):
        """
        Sample from the forward pass of the denoising model
        the forward pass will return a distribution q(x_t|x_0)
        this distribution is a Gaussian distribution with mean and variance
        Args:
            x0: Input image, aka Pokemin image in our case
            t: the number of the denoising steps
            eps: the noise to sample from the distribution q(x_t|x_0)
        """
        if eps is None:
            eps = torch.randn_like(x0)

        # get the mean and variance from the forward pass distribution
        mean, var = self.q_xt_x0(x0, t)
        # sample from the distribution, this is the starting point of the denoising process
        return mean + (var ** 0.5) * eps

    def p_sample(self, 
                 xt: torch.Tensor, 
                 t: torch.Tensor):
        """
        Sample from the reverse pass of the denoising model
        the reverse pass will return a distribution p(x_{t-1}|x_t) at each denoising step
        Args:
            xt: the noisy image at the denoising step t
            t: the number of the denoising steps
        """

        # get the predicted noise from the denoising model
        eps_theta = self.denoising_model(xt, t)
        # get the alpha_bar
        alpha_bar = gather(self.alpha_bar, t)
        # get the alpha
        alpha = gather(self.alpha, t)
        # calculate the coefficient for the noise
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # calculate the mean
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # calculate the variance
        var = gather(self.sigma2, t)

        # initialize a random noise for the sampling
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var ** .5) * eps

    def loss(self, 
             x0: torch.Tensor, 
             noise: Optional[torch.Tensor] = None):
        """
        Calculate the loss for the denoising model
        Args:
            x0: Input image, aka Pokemin image in our case
            noise: the noise to sample from the distribution q(x_t|x_0)
        """
        # Get batch size
        batch_size = x0.shape[0]
        # Get a fix denoising step scheduler for each image in the batch
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device)

        # Sample noise if not provided, this should be a Gaussian noise anyway
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample from the forward pass of the denoising model
        xt = self.q_sample(x0, t, eps=noise)
        # Sample from the reverse pass of the denoising model
        eps_theta = self.denoising_model(xt, t)

        # Calculate the "reconstruction" loss
        return F.mse_loss(noise, eps_theta)
