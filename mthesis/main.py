# Import the necessary modules
import argparse
import os
import numpy as np

import torch

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pathlib import Path

from src.Training import training_phase
from utils.utils import sample_image, sample_image_rnn, sample_image_flickr, print_image, print_image_rnn
from models.models import dataset_factory


# ------------------------
# Hyperparameter selection
# ------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default="FLICKR8K", help="Chose the model you want to use among: MNIST, RNN_MNIST, FLICKR8K")

parser.add_argument("--pre_loading", action="store_true", help="Set this flag if you want to use the saved pre-trained models")
parser.add_argument("--start_again", action="store_true", help="Set this flag if you want to start again the training from random")

parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")

parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=(64,64), help="shape of each image dimension")

parser.add_argument("--add_latent_dim", type=int, default=0, help="addditional dimensionalities of the latent space")
parser.add_argument("--add_embedding_dim", type=int, default=0, help="addditional dimensionalities of the embedding space")
parser.add_argument("--add_hidden_dim", type=int, default=0, help="addditional dimensionalities of the hidden space")

parser.add_argument("--sample_interval", type=int, default=200, help="interval between sampling images and saving models")
parser.add_argument('-f')
opt = parser.parse_args()
print(opt)


# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA available")
else:
    device = torch.device("cpu")
    print("CUDA not available")

# Create the necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("data/generated", exist_ok=True)
os.makedirs(f"data/generated/{opt.model}", exist_ok=True)


# ---------------------------------------
# Configure the dataset and the dataloader
# ---------------------------------------
dataset, dataloader, obj = dataset_factory(opt)
img_shape = obj.img_shape

# --------------------------------------
# Initialize generator and discriminator
# --------------------------------------
generator, discriminator = obj.modelling()

generator.to(device)
discriminator.to(device)

# ---------------
# Training phase:
# ---------------
os.makedirs("models", exist_ok=True)

paths = f"models/{opt.model}"
os.makedirs(paths, exist_ok=True)

if isinstance(obj.img_size, tuple):
    generator_name = f"generator{obj.latent_dim}L{obj.embedding_dim}E{obj.hidden_dim}H{obj.channels}x{obj.img_size[0]}x{obj.img_size[1]}"
    discriminator_name = f"discriminator{obj.latent_dim}L{obj.embedding_dim}E{obj.hidden_dim}H{obj.channels}x{obj.img_size[0]}x{obj.img_size[1]}"

if isinstance(obj.img_size, int):
    generator_name = f"generator{obj.latent_dim}L{obj.embedding_dim}E{obj.hidden_dim}H{obj.channels}x{obj.img_size}x{obj.img_size}"
    discriminator_name = f"discriminator{obj.latent_dim}L{obj.embedding_dim}E{obj.hidden_dim}H{obj.channels}x{obj.img_size}x{obj.img_size}"

g_model_file = f"{paths}/{generator_name}.pth"
d_model_file = f"{paths}/{discriminator_name}.pth"
g_model_fileAUX = f"{paths}/auxiliaries/{generator_name}.pth"
d_model_fileAUX = f"{paths}/auxiliaries/{discriminator_name}.pth"

myG = Path(g_model_file)
myD = Path(d_model_file)

myG_AUX = Path(g_model_fileAUX)
myD_AUX = Path(d_model_fileAUX)

if opt.pre_loading and myG.exists() and myD.exists():
    generator.load_state_dict(torch.load(g_model_file, map_location=device))
    discriminator.load_state_dict(torch.load(d_model_file, map_location=device))
    print("Loaded PyTorch Model State for GENERATOR and DISCRIMINATOR from .pth files")
else:
    if not opt.start_again and myG_AUX.exists() and myD_AUX.exists():
        generator.load_state_dict(torch.load(g_model_fileAUX, map_location=device))
        discriminator.load_state_dict(torch.load(d_model_fileAUX, map_location=device))
    training_phase(obj, generator, discriminator, opt, dataloader, dataset)

    # -------------
    # Saving model:
    # -------------
    torch.save(generator.state_dict(), g_model_file)
    print("Saved PyTorch Model State of GENERATOR to '%s'" %g_model_file)

    torch.save(discriminator.state_dict(), d_model_file)
    print("Saved PyTorch Model State of GENERATOR to '%s'" %d_model_file)


# Generating a set of 10 images per class:
if opt.model == "MNIST":
    sample_image("END", opt, obj, generator, dataloader)
    img = mpimg.imread('data/generated/MNIST/imageEND.png')
    imgplot = plt.imshow(img)
    plt.show()

elif opt.model == "RNN_MNIST":
    sample_image_rnn("END", opt, obj, generator, dataloader, dataset)
    img = mpimg.imread('data/generated/RNN_MNIST/imageEND.png')
    imgplot = plt.imshow(img)
    plt.show()

# Generating a set of testing images:
elif opt.model == "FLICKR8K":
    sample_image_flickr("END", opt, obj, generator, dataloader, dataset)
    img = mpimg.imread('data/generated/FLICKR8K/imageEND.png')
    imgplot = plt.imshow(img)
    plt.show()

else:
    # Default loaded model -> RNN_MNIST
    sample_image_rnn("END", opt, obj, generator, dataloader, dataset)
    img = mpimg.imread('data/generated/RNN_MNIST/imageEND.png')
    imgplot = plt.imshow(img)
    plt.show()


