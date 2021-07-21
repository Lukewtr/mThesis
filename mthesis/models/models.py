import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from torchvision import datasets
import torchvision.transforms as transforms

from src.FLICKR_CGAN import flickr8k_Dataset #, flickr_Generator, flickr_Discriminator
from src.MNIST_CGAN import mnist_Generator, mnist_Discriminator
from src.RNN_MNIST_CGAN import rnnMNIST_Dataset, RNNmnist_Generator, RNNmnist_Discriminator


def dataset_factory(parser):
    model_name = parser.model
    collate_fn = None

    if model_name == "MNIST":
        dataset = datasets.MNIST
        folder = "data/mnist"

        dict_args = {
            "root": folder,
            "train": True,
            "download": True,
            "transform": transforms.Compose(
                [transforms.Resize(parser.img_size),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])]
            ),
        }

    elif model_name == "RNN_MNIST":
        dataset = rnnMNIST_Dataset
        folder = "data/rnn_mnist"

        dict_args = {
            "root": folder,
            "train": True,
            "download": True,
            "transform": transforms.Compose(
                [transforms.Resize(parser.img_size),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])]
            ),
        }

    elif model_name == "FLICKR8K":
        dataset = flickr8k_Dataset
        folder = "/data/flickr8k"

        dict_args = {
            "root": folder,
            #"ann_file": "data/flickr8k/ann_file/Flickr8k.token.txt",
            "transform": transforms.Compose(
                [transforms.Resize(parser.img_size),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])]
            ),
            "target_transform": torch.tensor,
        }

        #os.makedirs(dict_args["ann_file"], exist_ok=True)
        def collate_fn(data):
            """Creates mini-batch tensors from the list of tuples (image, caption).

            We should build custom collate_fn rather than using default collate_fn, 
            because merging caption (including padding) is not supported in default.
            Args:
                data: list of tuple (image, caption). 
                    - image: torch tensor of shape (3, 256, 256).
                    - caption: torch tensor of shape (?); variable length.
            Returns:
                images: torch tensor of shape (batch_size, 3, 256, 256).
                targets: torch tensor of shape (batch_size, padded_length).
                lengths: list; valid length for each padded caption.
            """
            # Sort a data list by caption length (descending order).
            data.sort(key=lambda x: len(x[1]), reverse=True)
            images, captions = zip(*data)

            # Merge images (from tuple of 3D tensor to 4D tensor).
            images = torch.stack(images, 0)

            # Merge captions (from tuple of 1D tensor to 2D tensor).
            lengths = [len(cap) for cap in captions]
            targets = torch.zeros(len(captions), max(lengths)).long()
            for i, cap in enumerate(captions):
                end = lengths[i]
                # for j in range(end):
                #    targets[i, j] = cap[j]
                targets[i, :end] = cap[:end]
                # targets[i, :end] = torch.from_numpy(np.array(list(map(ord, cap[:end])))).to(torch.long)      
            return images, targets, lengths

    else:
        print(f"{model_name} doesn't exist!")
        print("RNN_MNIST model will be loaded!")

        dataset = rnnMNIST_Dataset
        folder = "data/rnn_mnist"

        dict_args = {
            "root": folder,
            "train": True,
            "download": True,
            "transform": transforms.Compose(
                [transforms.Resize(parser.img_size),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])]
            ),
        }

    os.makedirs(folder, exist_ok=True)

    # Create the datset
    dataset = dataset(**dict_args)

    # Configure data loader
    if collate_fn:
        dataloader = DataLoader(
            dataset,
            batch_size=parser.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    else:
        dataloader = DataLoader(
            dataset,
            batch_size=parser.batch_size,
            shuffle=True,
        )

    if model_name == "MNIST":
        obj = Mnist_Model(parser, dataset)

    elif model_name == "RNN_MNIST":
        obj = RNNmnist_Model(parser, dataset)

    elif model_name == "FLICKR8K":
        obj = Flickr8k_Model(parser, dataset)

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
        self.img_shape = (self.channels, self.img_size[0], self.img_size[1])

    def modelling(self):
        generator = RNNmnist_Generator(self.architecture, self.img_shape)
        discriminator = RNNmnist_Discriminator(self.architecture, self.img_shape)
        model = generator, discriminator
        return model


class Flickr8k_Model:
    def __init__(self, parser, dataset):
        self.opt = parser

        self.vocab_size = len(dataset.vocab)

        self.latent_dim = 200 + self.opt.add_latent_dim
        self.embedding_dim = 128 + self.opt.add_embedding_dim
        self.hidden_dim = 256 + self.opt.add_hidden_dim
        self.architecture = (self.vocab_size, self.latent_dim, self.embedding_dim, self.hidden_dim)

        self.channels = 1
        self.img_size = parser.img_size
        self.img_shape = (self.channels, self.img_size[0], self.img_size[1])

    def modelling(self):
        generator = None
        discriminator = None
        model = generator, discriminator
        return model