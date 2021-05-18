# Import the necessary modules
import argparse
import os, shutil
import numpy as np

import torch
from torch.utils.data import DataLoader

from torchvision import datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pathlib import Path

from src.Training import rnnMNIST_Dataset, training_phase
from src.CGAN import Generator, Discriminator
from src.RNN_CGAN import RNN_Generator, RNN_Discriminator
from utils.utils import sample_image, sample_image_rnn, print_image, print_image_rnn


# ------------------------
# Hyperparameter selection
# ------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--caption_usage", action="store_true", help="Set this flag if you want to implement caption embeddings")
parser.add_argument("--testing", action="store_true", help="Set this flag if you want to display the testing features")
parser.add_argument("--pre_loading", action="store_true", help="Set this flag if you want to use the saved pre-trained models")
parser.add_argument("--start_again", action="store_true", help="Set this flag if you want to start again the training from random")

parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")

parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

parser.add_argument("--latent_dim", type=int, default=200, help="dimensionality of the latent space")
parser.add_argument("--embedding_dim", type=int, default=128, help="dimensionality of the embedding space")
parser.add_argument("--hidden_dim", type=int, default=256, help="dimensionality of the hidden space")

parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")

parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")

parser.add_argument("--sample_interval", type=int, default=200, help="interval between sampling images and saving models")
parser.add_argument('-f')
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

# Check if we have GPU available
cuda = True if torch.cuda.is_available() else False
if cuda:
    #%matplotlib inline
    print("CUDA available")
else:
    print("CUDA not available")

# Create the necessary directories
os.makedirs("data", exist_ok=True)
if opt.caption_usage:
    os.makedirs("data/generated_rnn", exist_ok=True)
    dataset = rnnMNIST_Dataset
    folder = "data/rnn_mnist"
else:
    os.makedirs("data/generated", exist_ok=True)
    dataset = datasets.MNIST
    folder = "data/mnist"


# -----------------
# Create the datset
# -----------------
dataset = dataset(
    folder,
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    ),
)


# Testing the dataset
if opt.testing:
    index = np.random.randint(len(dataset))
    print(f"Regular caption: {dataset[index][1]}")
    if opt.caption_usage:
        print(f"Encoded caption: {dataset[index][2]}")
    plt.imshow(dataset[index][0][0,:,:])


# ---------------------
# Configure data loader
# ---------------------
os.makedirs(folder, exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)


# --------------------------------------
# Initialize generator and discriminator
# --------------------------------------
if opt.caption_usage:
    vocab_size = len(dataset.encoded_vocab)
    generator = RNN_Generator(vocab_size, opt, img_shape)
    discriminator = RNN_Discriminator(vocab_size, opt, img_shape)
else:
    generator = Generator(opt, img_shape)
    discriminator = Discriminator(opt, img_shape)

if cuda:
    generator.cuda()
    discriminator.cuda()



# ---------------
# Training phase:
# ---------------
os.makedirs("models", exist_ok=True)

if opt.caption_usage:
    g_model_file = "models/generator_RNN.pth"
    d_model_file = "models/discriminator_RNN.pth"
    g_model_fileAUX = "models/AUXgenerator_RNN.pth"
    d_model_fileAUX = "models/AUXdiscriminator_RNN.pth"
else:
    g_model_file = "models/generator.pth"
    d_model_file = "models/discriminator.pth"
    g_model_fileAUX = "models/AUXgenerator.pth"
    d_model_fileAUX = "models/AUXdiscriminator.pth"

myG = Path(g_model_file)
myD = Path(d_model_file)

myG_AUX = Path(g_model_fileAUX)
myD_AUX = Path(d_model_fileAUX)

try:
    if opt.pre_loading and myG.exists() and myD.exists():
        generator.load_state_dict(torch.load(g_model_file))
        discriminator.load_state_dict(torch.load(d_model_file))
        print("Loaded PyTorch Model State for GENERATOR and DISCRIMINATOR from pth files")
    else:
        if not opt.start_again and myG_AUX.exists() and myD_AUX.exists():
            generator.load_state_dict(torch.load(g_model_fileAUX))
            discriminator.load_state_dict(torch.load(d_model_fileAUX))
        training_phase(generator, discriminator, opt, dataloader, dataset)

except RuntimeError:
    source = "models"
    destination = "models/RuntimeErrorRecovered"
    #os.makedirs(destination, exist_ok=True)
    shutil.copytree(source, destination)
    print("The previous models weights are saved in %s folder;\nThe training is restarted!" %destination)
    training_phase(generator, discriminator, opt, dataloader, dataset)

# Testing the results:
if opt.testing:
    if opt.caption_usage:
        sample_image_rnn("END", opt, generator, dataloader, dataset)
    else:
        sample_image("END", opt, generator, dataloader)

    if opt.caption_usage:
        print_image_rnn("this is eight", opt, generator, dataset)
    else:
        print_image(8, opt, generator)



# -------------
# Saving model:
# -------------

torch.save(generator.state_dict(), g_model_file)
print("Saved PyTorch Model State of GENERATOR to '%s'" %g_model_file)

torch.save(discriminator.state_dict(), d_model_file)
print("Saved PyTorch Model State of GENERATOR to '%s'" %d_model_file)



# Generating a set of 10 images per class:
if opt.caption_usage:
    sample_image_rnn("Testing10x10", opt, generator, dataloader, dataset)
    img = mpimg.imread('data/generated_rnn/imageTesting10x10.png')
    imgplot = plt.imshow(img)
    plt.show()

else:
    sample_image("Testing10x10", opt, generator, dataloader)
    img = mpimg.imread('data/generated/imageTesting10x10.png')
    imgplot = plt.imshow(img)
    plt.show()

