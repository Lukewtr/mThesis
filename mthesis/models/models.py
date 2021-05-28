import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from torchvision import datasets
import torchvision.transforms as transforms

from src.Training import rnnMNIST_Dataset
from src.MNIST_CGAN import mnist_Generator, mnist_Discriminator
from src.RNN_MNIST_CGAN import RNNmnist_Generator, RNNmnist_Discriminator


def dataset_factory(parser):
    model_name = parser.model

    if model_name == "MNIST":
        dataset = datasets.MNIST
        folder = "data/mnist"

    elif model_name == "RNN_MNIST":
        dataset = rnnMNIST_Dataset
        folder = "data/rnn_mnist"

    else:
        print(f"{model_name} doesn't exist!")
        print("RNN_MNIST model will be loaded!")

        dataset = rnnMNIST_Dataset
        folder = "data/rnn_mnist"

    os.makedirs(folder, exist_ok=True)

    # Create the datset
    dataset = dataset(
        folder,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(parser.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )

    # Configure data loader
    dataloader = DataLoader(
        dataset,
        batch_size=parser.batch_size,
        shuffle=True,
    )

    if model_name == "MNIST":
        obj = Mnist_Model(parser, dataset)
    elif model_name == "RNN_MNIST":
        obj = RNNmnist_Model(parser, dataset)
    else:
        # Default loaded model -> RNN_MNIST
        obj = RNNmnist_Model(parser, dataset)

    return dataset, dataloader, obj


class Mnist_Model:
    def __init__(self, parser, dataset):
        self.opt = parser

        self.n_classes = 10

        self.latent_dim = 100 + self.opt.add_latent_dim
        self.embedding_dim = 128 + self.opt.add_embedding_dim
        self.hidden_dim = 256 + self.opt.add_hidden_dim
        self.architecture = (self.n_classes, self.latent_dim, self.embedding_dim, self.hidden_dim)

        self.channels = 1
        self.img_size = parser.img_size
        self.img_shape = (self.channels, self.img_size, self.img_size)

    def modelling(self):
        generator = mnist_Generator(self.architecture, self.img_shape)
        discriminator = mnist_Discriminator(self.architecture, self.img_shape)
        model = generator, discriminator
        return model


class RNNmnist_Model:
    def __init__(self, parser, dataset):
        self.opt = parser

        self.n_classes = 10
        self.vocab_size = len(dataset.encoded_vocab)

        self.latent_dim = 200 + self.opt.add_latent_dim
        self.embedding_dim = 128 + self.opt.add_embedding_dim
        self.hidden_dim = 256 + self.opt.add_hidden_dim
        self.architecture = (self.vocab_size, self.latent_dim, self.embedding_dim, self.hidden_dim)

        self.channels = 1
        self.img_size = parser.img_size
        self.img_shape = (self.channels, self.img_size, self.img_size)

    def modelling(self):
        generator = RNNmnist_Generator(self.architecture, self.img_shape)
        discriminator = RNNmnist_Discriminator(self.architecture, self.img_shape)
        model = generator, discriminator
        return model

