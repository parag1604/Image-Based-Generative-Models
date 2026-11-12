import math
import os
import argparse

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from utils import CarsDataset, set_seed


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(self.act1(self.norm1(x)))

        time_emb = self.time_mlp(t)
        time_emb = time_emb[..., None, None]
        h = h + time_emb

        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.residual(x)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNet(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        self.conv0 = nn.Conv2d(3, 64, 3, padding=1)

        # Down
        self.down1 = ResBlock(64, 128, time_dim)
        self.down2 = ResBlock(128, 256, time_dim)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bot1 = ResBlock(256, 256, time_dim)

        # Up
        self.up1 = ResBlock(256 + 256, 128, time_dim)
        self.up2 = ResBlock(128 + 128, 64, time_dim)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)

        x1 = self.conv0(x)
        x2 = self.down1(x1, t)
        x3 = self.down2(self.pool(x2), t)

        x_mid = self.bot1(self.pool(x3), t)

        x = self.upsample(x_mid)
        x = torch.cat([x, x3], dim=1)
        x = self.up1(x, t)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, t)

        return self.out(x)


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


@torch.no_grad()
def sample_timestep(
    x, t, betas, model, sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas, posterior_variance):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t[0] == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 


@torch.no_grad()
def sample(model, T, betas, sqrt_one_minus_alphas_cumprod,
           sqrt_recip_alphas, posterior_variance, device):
    """DDPM ancestral sampling: all T reverse steps."""
    img = torch.randn((64, 3, 64, 64), device=device)

    for i in tqdm(reversed(range(T)), total=T):
        t = torch.full((64,), i, device=device, dtype=torch.long)

        img = sample_timestep(
            img, t, betas, model,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas,
            posterior_variance
        )

    return img.clamp(-1, 1)


@torch.no_grad()
def sample_ddim(model, T, alphas_cumprod, device, num_steps=50, eta=0.0):
    batch_size = 64
    img = torch.randn((batch_size, 3, 64, 64), device=device)

    step_size = T // num_steps
    timesteps = list(range(0, T, step_size))[::-1]

    for i in tqdm(range(len(timesteps) - 1), total=len(timesteps) - 1):
        t = torch.full((batch_size,), timesteps[i], device=device, dtype=torch.long)
        t_prev = torch.full((batch_size,), timesteps[i+1], device=device, dtype=torch.long)

        alpha_t = get_index_from_list(alphas_cumprod, t, img.shape)
        alpha_prev = get_index_from_list(alphas_cumprod, t_prev, img.shape)

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

        eps = model(img, t)

        x0 = (img - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t

        sigma = (
            eta
            * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
            * torch.sqrt(1 - alpha_t / alpha_prev)
        )

        noise = torch.randn_like(img) if eta > 0 else 0

        img = (
            torch.sqrt(alpha_prev) * x0
            + torch.sqrt(1 - alpha_prev - sigma**2) * eps
            + sigma * noise
        )

    return img.clamp(-1, 1)


def forward_diffusion_sample(
        x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


def main(**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(kwargs['seed'])
    os.makedirs("models", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    dataset = CarsDataset(kwargs['data_path'])
    dataloader = DataLoader(
        dataset, batch_size=kwargs['batch_size'],
        shuffle=True, num_workers=kwargs['num_workers'])

    # Define beta schedule
    T = kwargs['T']
    betas = linear_beta_schedule(timesteps=T)

    # Pre-calculate different terms
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def generate():
        if kwargs['sampler'] == 'ddpm':
            return sample(
                model, T, betas, sqrt_one_minus_alphas_cumprod,
                sqrt_recip_alphas, posterior_variance, device).cpu()
        return sample_ddim(
            model, T, alphas_cumprod, device,
            num_steps=kwargs['ddim_steps'], eta=0.0).cpu()

    model = UNet().to(device)
    ema = EMA(model, decay=0.9999)
    optimizer = optim.AdamW(
        model.parameters(), lr=kwargs['learning_rate'],
        weight_decay=kwargs['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, step / 5000))

    wandb.init(
        entity=kwargs['wandb_entity'],
        mode=kwargs['wandb_mode'],
        project="Generative-Models",
        name="DDPM",
        config=kwargs,
    )

    real_batch = next(iter(dataloader))
    images = make_grid(real_batch[:64], padding=2, normalize=True).numpy().transpose(1,2,0)
    wandb.log({"Images/Real": wandb.Image(images)}, step=0)

    # Training Loop
    iters = 0
    for epoch in range(kwargs['num_epochs']):
        for data in dataloader:
            t = torch.randint(0, T, (data.shape[0],), device=device).long()
            x_t, noise = forward_diffusion_sample(
                data, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
            noise_pred = model(x_t, t)
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update()

            wandb.log({
                "Loss": loss.item()
            }, step=iters)
            
            if iters % kwargs['log_freq'] == 0:
                print('Iter: %d\tLoss: %.4f' % (iters, loss.item()))
            
            if (iters % kwargs['save_freq'] == 0):
                torch.save({
                    "model": model.state_dict(), "ema": ema.shadow}, f"models/ddpm.pth")
                model.eval()
                with torch.no_grad():
                    samples = generate()
                print("Sampling done. Now applying EMA weights...")
                ema.apply_shadow()
                with torch.no_grad():
                    ema_samples = generate()
                print("EMA Sampling done. Restoring original weights...")
                ema.restore()
                model.train()
                images = make_grid(
                    samples, padding=2, normalize=True).permute(1, 2, 0).numpy()
                ema_images = make_grid(
                    ema_samples, padding=2, normalize=True).permute(1, 2, 0).numpy()
                wandb.log({
                    "Images/Samples": wandb.Image(images),
                    "Images/EMA_Samples": wandb.Image(ema_images)
                }, step=iters)
                save_image(samples, f"samples/ddpm_{iters}.png", normalize=True)
                save_image(ema_samples, f"samples/ddpm_ema_{iters}.png", normalize=True)
            iters += 1
            if kwargs['max_iters'] and iters >= kwargs['max_iters']:
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM on Stanford Cars (64x64)")
    parser.add_argument("--data-path", default="./data/stanford_cars",
                        help="folder containing train/ and test/ (ImageFolder layout)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--T", type=int, default=1000, help="number of diffusion steps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"],
                        help="DDIM (deterministic, few steps) or DDPM ancestral sampling (all T steps)")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--log-freq", type=int, default=100,
                        help="print losses every this many iterations")
    parser.add_argument("--save-freq", type=int, default=1000,
                        help="save weights and sample grids every this many iterations")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="stop after this many iterations (for a quick check)")
    parser.add_argument("--wandb-entity", default=None,
                        help="Weights & Biases team/user (default: your default entity)")
    parser.add_argument("--wandb-mode", default="online",
                        choices=["online", "offline", "disabled"])
    main(**vars(parser.parse_args()))
