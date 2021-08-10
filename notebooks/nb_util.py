import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms, utils
from IPython.display import display, update_display


# Configure file system
notebook_path = Path(__file__).parent
project_root = notebook_path.parent
data_path = project_root / 'data'

os.chdir(project_root)

# Configure PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# Visualization utilities
to_pil = transforms.ToPILImage()

def display_batch(images, n_cols=4):
    images = images.cpu().detach()
    images = torch.minimum(images, torch.ones(images.shape))
    images = torch.maximum(images, torch.zeros(images.shape))

    fig = plt.figure(figsize=(n_cols * 3, len(images) * 3 // n_cols))
    for i, img in enumerate(images):
        ax = fig.add_subplot(math.ceil(len(images) / n_cols), n_cols, i+1)
        ax.axis('off')
        plt.imshow(to_pil(img), interpolation='nearest', aspect='auto')
    plt.subplots_adjust(hspace=0, wspace=0)

    # Need to explicitly display and get an id if we want to dynamically update it
    display_id = random.randint(0, 100000)
    display(fig, display_id=display_id)

    return fig, display_id


def update_displayed_batch(images, fig, display_id):
    images = images.cpu().detach()
    images = torch.minimum(images, torch.ones(images.shape))
    images = torch.maximum(images, torch.zeros(images.shape))

    for i, img in enumerate(images):
        fig.axes[i].images[0].set_data(to_pil(img))

    update_display(fig, display_id=display_id)


def compare_batches(images, reconstructions, fig=None, display_id=None):
    b, c, h, w = images.shape
    combined_batch = torch.cat((images, reconstructions), axis=1).reshape(2 * b, c, h, w)

    if fig is not None and display_id is not None:
        update_displayed_batch(combined_batch, fig, display_id)
    else:
        return display_batch(combined_batch)


def generate_image_samples(num, model, model_name, latent_shape, batch_size=8, temp=0.75,
                           device=device):
    '''Helper for generating samples from a trained model'''
    assert len(latent_shape) <= 3, 'Latent shape should not include batch size'

    image_path = project_root / 'figures' / 'generated' / model_name
    os.makedirs(image_path, exist_ok=True)

    for i in range(0, num, batch_size):
        iter_size = min(batch_size, num-i)

        # Assume the base distribution is standard normal
        latent_samples = torch.normal(mean=torch.zeros(iter_size, *latent_shape),
                                      std=torch.ones(iter_size, *latent_shape)*temp)
        latent_samples = latent_samples.to(device)
        gen_samples = model(latent_samples)

        for j, samp in enumerate(gen_samples):
            utils.save_image(gen_samples[j], f'{image_path}/img{i+j}.jpg')
            print(f'Saved image {i+j}', end='\r')
