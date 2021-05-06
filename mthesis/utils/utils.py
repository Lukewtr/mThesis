# Import the necessary modules
import torch
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


# Sample function:
def sample_image(string, opt, generator, dataloader):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    n_row = 10

    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)

    # Saving images in format "images<n_epoch>_<n_iter>" or "images<string>"
    if isinstance(string, str):
        printed = string

    elif isinstance(string, tuple):
        epoch, iteration = string
        max_epoch = opt.n_epochs

        epoch_s = str(epoch)
        max_epoch_s = str(max_epoch)

        printed = ""
        for _ in range(len(max_epoch_s) - len(epoch_s)):
            printed = printed + "0"
        for char in epoch_s:
            printed = printed + char
        printed = printed + "_"

        max_iter_s = str(len(dataloader))
        iter_s = str(iteration)

        for _ in range(len(max_iter_s) - len(iter_s)):
            printed = printed + "0"
        for char in iter_s:
            printed = printed + char

    else:
        printed = '_CGAN'

    save_image(gen_imgs.data, "data/generated/image%s.png" % printed, nrow=n_row, normalize=True)


def sample_image_rnn(string, opt, generator, dataloader, dataset):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    n_row = 10

    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get captions ranging from 0 to n_classes for n rows
    captions = [dataset.mapping[num] for _ in range(n_row) for num in range(n_row)]

    gen_encoded_captions = []
    for caption in captions:
        gen_encoded_captions.append(tuple([dataset.encoded_vocab[word] for word in caption.split(" ")]))

    generator_input = Variable(
        LongTensor(
            gen_encoded_captions
        )
    )

    gen_imgs = generator(z, generator_input)

    # Saving images in format "images<n_epoch>_<n_iter>" or "images<string>"
    if isinstance(string, str):
        printed = string

    elif isinstance(string, tuple):
        epoch, iteration = string
        max_epoch = opt.n_epochs

        epoch_s = str(epoch)
        max_epoch_s = str(max_epoch)

        printed = ""
        for _ in range(len(max_epoch_s)-len(epoch_s)):
            printed = printed + "0"
        for char in epoch_s:
            printed = printed + char
        printed = printed + "_"

        max_iter_s = str(len(dataloader))
        iter_s = str(iteration)

        for _ in range(len(max_iter_s)-len(iter_s)):
            printed = printed + "0"
        for char in iter_s:
            printed = printed + char

    else:
        printed = '_RNN_CGAN'

    save_image(gen_imgs.data, "data/generated_rnn/image%s.png" % printed, nrow=n_row, normalize=True)


# Generating a single image conditioned on the given label:
def print_image(label, opt, generator):
    if not isinstance(label, int):
        label = 0

    z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
    gen_labels = Variable(
        LongTensor(np.concatenate((np.array((label,)), np.random.randint(0, opt.n_classes, opt.batch_size - 1)))))

    gen_imgs = generator(z, gen_labels)

    plt.imshow(
        gen_imgs[0, 0, :, :]
            .cpu()
            .detach()
            .numpy())


# Generating a single image conditioned on the given caption:
def print_image_rnn(caption, dataset, opt, generator):
    if not isinstance(caption, str):
        caption = "this is zero"

    z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

    gen_labels = np.random.randint(0, opt.n_classes, opt.batch_size - 1)
    gen_captions = [dataset.mapping[key] for key in gen_labels]
    gen_captions = np.concatenate(np.array((caption,)), gen_captions)

    gen_encoded_captions = []
    for caption in gen_captions:
        gen_encoded_captions.append(tuple([dataset.encoded_vocab[word] for word in caption.split(" ")]))

    generator_input = Variable(
        LongTensor(
            gen_encoded_captions
        )
    )

    # Generate a batch of images
    gen_imgs = generator(z, generator_input)

    plt.imshow(
        gen_imgs[0, 0, :, :]
            .cpu()
            .detach()
            .numpy())