# WGAN-based MNIST Image Generation

This repository contains a PyTorch implementation of a Wasserstein GAN (WGAN) for generating high-quality handwritten digit images from random latent vectors using the MNIST dataset.

---

## Table of Contents

- [WGAN-based MNIST Image Generation](#wgan-based-mnist-image-generation)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Dependencies](#dependencies)
  - [Dataset](#dataset)
    - [Generator](#generator)
    - [Discriminator](#discriminator)
  - [Training Procedure](#training-procedure)
  - [Results](#results)
  - [Usage](#usage)
  - [Acknowledgments](#acknowledgments)

---

## Overview

This project implements a Wasserstein GAN (WGAN) with gradient penalty (WGAN-GP) to generate high-quality handwritten digit images from latent vectors using the MNIST dataset. WGANs aim to improve training stability by focusing on the Wasserstein distance between real and generated distributions.

---

## Dependencies

Make sure to install the following dependencies to run this project:

```
pip install torch torchvision Pillow matplotlib
```

---

## Dataset

The MNIST dataset is used as the benchmark for generating handwritten digit images. The MNIST dataset consists of 28x28 grayscale images of digits from 0 to 9.

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
data = datasets.MNIST("./data", train=True, download=True, transform=transform)
dataloader = DataLoader(data, batch_size=64, shuffle=True)
```

---

### Generator

The Generator takes a random latent vector as input and produces a 28x28 image.

```
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, channels * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), channels, img_size, img_size)
        return img
```

### Discriminator

The Discriminator evaluates whether the input image is real or generated using a WGAN-based setup.

```
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(channels * img_size * img_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
```

---

## Training Procedure

The GAN is trained using a Wasserstein GAN with gradient penalty (WGAN-GP) to ensure better stability and quality of generated images.

```
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)

        # Train Discriminator
        optimizer_D.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        fake_imgs = generator(z)
        real_validity = discriminator(imgs)
        fake_validity = discriminator(fake_imgs)
        gradient_penalty = compute_gradient_penalty(discriminator, imgs, fake_imgs)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            fake_imgs = generator(z)
            g_loss = -torch.mean(discriminator(fake_imgs))
            g_loss.backward()
            optimizer_G.step()

    print(f"Epoch [{epoch + 1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
```

---

## Results

After training, the model generates high-quality handwritten digits using the Wasserstein distance-based approach.

```
z = torch.randn(5000, latent_dim).to(device)
generated_imgs = generator(z)

# Save Generated Images
os.makedirs("generated_images", exist_ok=True)
for i, img in enumerate(generated_imgs[:25]):
    img = img.cpu().detach().numpy().squeeze()
    img = ((img + 1) / 2) * 255  # Rescale to [0, 255]
    img = Image.fromarray(img.astype(np.uint8))
    img.save(f"generated_images/img_{i + 1}.png")
```

---

## Usage

1. Clone the repository:

```
git clone https://github.com/yourusername/GAN-MNIST.git
cd GAN-MNIST
```

2. Install the dependencies:

```
pip install -r requirements.txt
```

3. Run the training script:

```
python train.py
```

4. Generated images will be saved in the `generated_images/` directory.

---

## Acknowledgments

This project is based on research from Wasserstein GANs (WGANs) and applications such as image generation.

---
