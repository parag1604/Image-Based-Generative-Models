import os
import argparse

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from utils import CarsDataset, set_seed


class Generator(nn.Module):
    def __init__(self, nz, ngf):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main(**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(kwargs['seed'])
    os.makedirs("models", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    dataset = CarsDataset(kwargs['data_path'])
    dataloader = DataLoader(
        dataset, batch_size=kwargs['batch_size'],
        shuffle=True, num_workers=kwargs['num_workers'])

    netD = Discriminator(kwargs['num_feature_maps']).to(device)
    netD.apply(weights_init)
    optimizerD = optim.Adam(
        netD.parameters(), lr=kwargs['learning_rate'],
        betas=(kwargs['beta1'], 0.999))
    
    netG = Generator(kwargs['latent_dim'], kwargs['num_feature_maps']).to(device)
    netG.apply(weights_init)
    optimizerG = optim.Adam(
        netG.parameters(), lr=kwargs['learning_rate'],
        betas=(kwargs['beta1'], 0.999))

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, kwargs['latent_dim'], 1, 1, device=device)
    real_label = 1.
    fake_label = 0.

    wandb.init(
        entity=kwargs['wandb_entity'],
        mode=kwargs['wandb_mode'],
        project="Generative-Models",
        name="DC-GAN",
        config=kwargs,
    )

    real_batch = next(iter(dataloader))
    images = make_grid(real_batch[:64], padding=2, normalize=True).numpy().transpose(1,2,0)
    wandb.log({"Real_Images": wandb.Image(images)}, step=0)

    # Training Loop
    iters = 0
    print("Starting Training Loop...")
    for epoch in range(kwargs['num_epochs']):
        for data in dataloader:
            netD.zero_grad()
            real = data.to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, kwargs['latent_dim'], 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            label.fill_(real_label)
            output = netD(fake).view(-1)
            netG.zero_grad()
            errG = criterion(output, label)
            errG.backward()
            
            D_G_z2 = output.mean().item()
            optimizerG.step()

            wandb.log({
                "Loss_D": errD.item(),
                "Loss_G": errG.item(),
                "D(x)": D_x,
                "D(G(z))_before_update": D_G_z1,
                "D(G(z))_after_update": D_G_z2
            }, step=iters)
            
            if iters % kwargs['log_freq'] == 0:
                print('Iter: %d\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (iters, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            if (iters % kwargs['save_freq'] == 0):
                torch.save(netG.state_dict(), f"models/generator.pth")
                torch.save(netD.state_dict(), f"models/discriminator.pth")
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                images = make_grid(fake, padding=2, normalize=True).numpy().transpose(1,2,0)
                wandb.log({"Generated_Images": wandb.Image(images)}, step=iters)
                save_image(fake, f"samples/gan_{iters}.png", normalize=True)
            iters += 1
            if kwargs['max_iters'] and iters >= kwargs['max_iters']:
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCGAN on Stanford Cars (64x64)")
    parser.add_argument("--data-path", default="./data/stanford_cars",
                        help="folder containing train/ and test/ (ImageFolder layout)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--num-feature-maps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--log-freq", type=int, default=50,
                        help="print losses every this many iterations")
    parser.add_argument("--save-freq", type=int, default=500,
                        help="save weights and sample grids every this many iterations")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="stop after this many iterations (for a quick check)")
    parser.add_argument("--wandb-entity", default=None,
                        help="Weights & Biases team/user (default: your default entity)")
    parser.add_argument("--wandb-mode", default="disabled",
                        choices=["online", "offline", "disabled"])
    main(**vars(parser.parse_args()))
