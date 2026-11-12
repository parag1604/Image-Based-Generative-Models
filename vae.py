import os
import argparse

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from utils import CarsDataset, set_seed


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        
        # Encoder: 64x64 -> 4x4
        self.encoder = nn.Sequential(
            EncoderBlock(3, 32),    # 32x32
            EncoderBlock(32, 64),   # 16x16
            EncoderBlock(64, 128),  # 8x8
            EncoderBlock(128, 256), # 4x4
        )
        
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        
        # Decoder: Latent -> 64x64
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            DecoderBlock(256, 128), # 8x8
            DecoderBlock(128, 64),  # 16x16
            DecoderBlock(64, 32),   # 32x32
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh() 
        )

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def main(**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(kwargs['seed'])
    os.makedirs("models", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    dataset = CarsDataset(kwargs['data_path'])
    dataloader = DataLoader(
        dataset, batch_size=kwargs['batch_size'],
        shuffle=True, num_workers=kwargs['num_workers'])
    
    model = VAE(latent_dim=kwargs['latent_dim']).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=kwargs['learning_rate'],
        weight_decay=kwargs['weight_decay'])
    
    wandb.init(
        entity=kwargs['wandb_entity'],
        mode=kwargs['wandb_mode'],
        project="Generative-Models",
        name="VAE",
        config=kwargs,
    )

    fixed_z = torch.randn(64, kwargs['latent_dim']).to(device)
    real_batch = next(iter(dataloader))[:64].to(device)
    images = make_grid(real_batch, padding=2, normalize=True).cpu().numpy().transpose(1,2,0)
    wandb.log({"images/input": wandb.Image(images)}, step=0)

    # Training Loop
    iters = 0
    print("Starting Training Loop...")
    for epoch in range(kwargs['num_epochs']):
        for data in dataloader:
            data = data.to(device)
            b_size = data.size(0)
            recon, mu, logvar = model(data)
            recon_loss = torch.abs(recon - data).sum()
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_weight = min(kwargs['kl_weight'], epoch / (kwargs['num_epochs'] / 2))
            loss = (recon_loss + kl_weight * kl_loss) / b_size

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()

            wandb.log({
                "recon_loss": recon_loss.item() / b_size,
                "kl_loss": kl_loss.item() / b_size,
                "total_loss": loss.item()
            }, step=iters)
            
            if iters % kwargs['log_freq'] == 0:
                print('Iter: %d\tLoss_R: %.4f\tLoss_KL: %.4f\tLoss_Total: %.4f'
                    % (iters, recon_loss.item(), kl_loss.item(), loss.item()))
            
            if (iters % kwargs['save_freq'] == 0):
                torch.save(model.state_dict(), f"models/vae.pth")
                model.eval()
                with torch.no_grad():
                    recon = model(real_batch)[0].detach().cpu()
                    sampled = model.decode(fixed_z).detach().cpu()
                model.train()
                recon_imgs = make_grid(
                    recon, padding=2, normalize=True).numpy().transpose(1,2,0)
                sampled_imgs = make_grid(
                    sampled, padding=2, normalize=True).numpy().transpose(1,2,0)
                wandb.log({
                    "images/samples": wandb.Image(sampled_imgs),
                    "images/reconstructions": wandb.Image(recon_imgs)
                }, step=iters)
                save_image(sampled, f"samples/vae_samples_{iters}.png", normalize=True)
                save_image(recon, f"samples/vae_recon_{iters}.png", normalize=True)
            iters += 1
            if kwargs['max_iters'] and iters >= kwargs['max_iters']:
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convolutional VAE on Stanford Cars (64x64)")
    parser.add_argument("--data-path", default="./data/stanford_cars",
                        help="folder containing train/ and test/ (ImageFolder layout)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--kl-weight", type=float, default=1.0, help="final weight of the KL term (annealed from 0 over the first half of training)")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-05)
    parser.add_argument("--log-freq", type=int, default=50,
                        help="print losses every this many iterations")
    parser.add_argument("--save-freq", type=int, default=500,
                        help="save weights and sample grids every this many iterations")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="stop after this many iterations (for a quick check)")
    parser.add_argument("--wandb-entity", default=None,
                        help="Weights & Biases team/user (default: your default entity)")
    parser.add_argument("--wandb-mode", default="online",
                        choices=["online", "offline", "disabled"])
    main(**vars(parser.parse_args()))
