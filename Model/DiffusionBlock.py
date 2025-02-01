import torch.nn as nn
from Model.UNet import *
class DiffusionModel(nn.Module):

    def __init__(self, time_steps,
                 beta_start = 10e-4,
                 beta_end = 0.02,
                 image_dims = (3, 128, 128)):

        super().__init__()
        self.time_steps = time_steps
        self.image_dims = image_dims
        c, h, w = self.image_dims
        self.img_size, self.input_channels = h, c
        self.betas = torch.linspace(beta_start, beta_end, self.time_steps)
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim = -1)
        self.model = UNet(input_channels = 2*c, output_channels = c, time_steps = self.time_steps)

    def add_noise(self, x, ts):
        # 'x' and 'ts' are expected to be batched
        noise = torch.randn_like(x)
        # print(x.shape, noise.shape)
        noised_examples = []
        for i, t in enumerate(ts):
            alpha_hat_t = self.alpha_hats[t]
            noised_examples.append(torch.sqrt(alpha_hat_t)*x[i] + torch.sqrt(1 - alpha_hat_t)*noise[i])
        return torch.stack(noised_examples), noise

    def forward(self, x, t):
        return self.model(x, t)