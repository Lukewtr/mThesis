# Import the necessary modules
import torch
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import string


FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


# Sample function:
def sample_image(name, opt, obj, generator, dataloader):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    n_row = 10

    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, obj.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)

    # Saving images in format "images<n_epoch>_<n_iter>" or "images<name>"
    if isinstance(name, str):
        printed = name

    elif isinstance(name, tuple):
        epoch, iteration = name
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

    save_image(gen_imgs.data, f"data/generated/MNIST/image{printed}.png", nrow=n_row, normalize=True)


def sample_image_rnn(name, opt, obj, generator, dataloader, dataset):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    n_row = 10

    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, obj.latent_dim))))
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

    # Saving images in format "images<n_epoch>_<n_iter>" or "images<name>"
    if isinstance(name, str):
        printed = name

    elif isinstance(name, tuple):
        epoch, iteration = name
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

    save_image(gen_imgs.data, f"data/generated/RNN_MNIST/image{printed}.png", nrow=n_row, normalize=True)


def sample_image_flickr(name, opt, obj, generator, dataloader, dataset):
    #Saves a grid of generated images
    n_row = 2

    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, obj.latent_dim))))
    # Get captions ranging from 0 to n_classes for n rows
    #captions = [[dataset.captions[np.random.randint(1, len(dataset.vocab))] for _ in range(n_row)] for _ in range(n_row)]
    captions = [[["<start> A men wearing sunglasses sit on a bench <end>"],
                 ["<start> Two men running in a park with blue shirts <end>"]],
                [["<start> A girl playing guitar duirng a concert <end>"],
                 ["<start> Three dogs run in the garden towards a red ball <end>"]]]
    #print("captions:", np.array(captions).shape, len(captions))

    gen_encoded_captions = []
    max_len = 0
    for collect_caption in captions:
        #print("collect_caption:", np.array(collect_caption).shape)
        for [caption] in collect_caption:
            #print("caption:", np.array(caption).shape)
            line = []
            for word in caption.lower().split(" "):
                if word not in string.punctuation:
                    if word.isdigit():
                        for digit in word:
                            line.append(dataset.fword_to_ix(digit.lower()))
                    else:
                        if "<start>" in word: line.append(dataset.fword_to_ix("<start>"))
                        elif "<end>" in word: line.append(dataset.fword_to_ix("<end>"))
                        else: line.append(dataset.fword_to_ix(word))
            max_len = max(max_len, len(line))
            gen_encoded_captions.append(line) # + [0] * (max_len-len(line)))
    #print("gen_encoded_captions:", np.array(gen_encoded_captions).shape, len(gen_encoded_captions))

    pad_encoded_captions = []
    for cap in gen_encoded_captions:
        pad_encoded_captions.append(cap + [0] * (max_len-len(cap)))

    final_encoded_captions = []
    for k in range(n_row):
        sublist = pad_encoded_captions[k*n_row: (k+1)*n_row]
        #print("Sublist:", len(sublist))
        final_encoded_captions.append(sublist)

    #print("final_encoded_captions:", np.array(final_encoded_captions).shape)
    generator_input = Variable(
        LongTensor(
            final_encoded_captions
        )
    )

    #print("generator_input:", generator_input.shape)
    new_generator_input = torch.reshape(generator_input, (n_row**2, generator_input.shape[-1]))
    #print("generator_input:", new_generator_input.shape)

    gen_imgs = generator(z, new_generator_input)

    # Saving images in format "images<n_epoch>_<n_iter>" or "images<name>"
    if isinstance(name, str):
        printed = name

    elif isinstance(name, tuple):
        epoch, iteration = name
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
        printed = '_FLICKR_CGAN'

    save_image(gen_imgs.data, f"data/generated/FLICKR8K/image{printed}.png", nrow=n_row, normalize=True)


# Generating a single image conditioned on the given label:
def print_image(label, opt, obj, generator):
    if not isinstance(label, int):
        label = 0

    z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, obj.latent_dim))))
    gen_labels = Variable(
        LongTensor(np.concatenate((np.array((label,)), np.random.randint(0, obj.n_classes, opt.batch_size - 1)))))

    gen_imgs = generator(z, gen_labels)

    plt.imshow(
        gen_imgs[0, 0, :, :]
            .cpu()
            .detach()
            .numpy())


# Generating a single image conditioned on the given caption:
def print_image_rnn(caption, opt, obj, generator, dataset):
    if not isinstance(caption, str):
        caption = "this is zero"

    z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, obj.latent_dim))))

    gen_labels = np.random.randint(0, obj.n_classes, opt.batch_size - 1)
    gen_captions = [dataset.mapping[key] for key in gen_labels]
    gen_captions = [caption,] + gen_captions

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


# Generating a single image conditioned on the given caption:
def print_image_flickr(caption, opt, obj, generator, dataset):
    """if not isinstance(caption, str):
        caption = "this is zero"

    z = Variable(FloatTensor(np.random.normal(0, 1, (opt.batch_size, obj.latent_dim))))

    gen_labels = np.random.randint(0, obj.n_classes, opt.batch_size - 1)
    gen_captions = [dataset.mapping[key] for key in gen_labels]
    gen_captions = [caption,] + gen_captions

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
            .numpy())"""
    pass
