# Import the necessary modules
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from utils.utils import sample_image, sample_image_rnn, sample_image_flickr
import os


def training_phase(obj, generator, discriminator, opt, dataloader, dataset):
    # Loss functions
    adversarial_loss = torch.nn.MSELoss()
    if torch.cuda.is_available():
        adversarial_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    paths = f"models/{opt.model}"
    generator_name = f"generator{obj.latent_dim}L{obj.embedding_dim}E{obj.hidden_dim}H{obj.channels}x{obj.img_size}x{obj.img_size}"
    discriminator_name = f"discriminator{obj.latent_dim}L{obj.embedding_dim}E{obj.hidden_dim}H{obj.channels}x{obj.img_size}x{obj.img_size}"

    g_model_fileAUX = f"{paths}/auxiliaries/{generator_name}.pth"
    d_model_fileAUX = f"{paths}/auxiliaries/{discriminator_name}.pth"

    os.makedirs(f"{paths}/auxiliaries", exist_ok=True)

    for epoch in range(opt.n_epochs):
        for i, data in enumerate(dataloader):
            # Configure input
            if opt.model == "MNIST":
                (imgs, labels) = data
                batch_size = imgs.shape[0]

                discriminator_input = Variable(labels.type(LongTensor))

                gen_labels = np.random.randint(0, obj.n_classes, batch_size)
                generator_input = Variable(LongTensor(gen_labels))

            elif opt.model == "RNN_MNIST":
                (imgs, captions, encoded_captions) = data
                batch_size = imgs.shape[0]

                discriminator_input = Variable(encoded_captions.type(LongTensor))

                gen_labels = np.random.randint(0, obj.n_classes, batch_size)
                gen_captions = [dataset.mapping[key] for key in gen_labels]

                gen_encoded_captions = []
                for caption in gen_captions:
                    gen_encoded_captions.append(tuple([dataset.encoded_vocab[word] for word in caption.split(" ")]))

                generator_input = Variable(
                    LongTensor(
                        gen_encoded_captions
                    )
                )

            elif opt.model == "FLICKR8K":
                (imgs, encoded_captions, lenghts) = data
                batch_size = imgs.shape[0]

                discriminator_input = Variable(encoded_captions.type(LongTensor))

                gen_idx = np.random.randint(0, len(dataset.captions), batch_size)
                gen_captions = [dataset.captions[j] for j in gen_idx]

                gen_encoded_captions = []
                gen_encoded_words = []
                max_len = 0

                for [caption] in gen_captions:
                    words = caption.lower().split(" ")

                    l = len(words)
                    if l > max_len: max_len = l

                    for j, word in enumerate(words):
                        if "<start>" in word: words[j] = "<start>"
                        if "<end>" in word: words[j] = "<end>"
                    encoded_words = [dataset.fword_to_ix(word) for word in words]
                    gen_encoded_words.append(encoded_words)

                for words in gen_encoded_words:
                    words = words + [0] * (max_len - len(words))
                    gen_encoded_captions.append(words)

                generator_input = Variable(
                    LongTensor(
                        gen_encoded_captions
                    )
                )

            else:
                # Default loaded model -> RNN_MNIST
                (imgs, captions, encoded_captions) = data
                batch_size = imgs.shape[0]

                discriminator_input = Variable(encoded_captions.type(LongTensor))

                gen_labels = np.random.randint(0, obj.n_classes, batch_size)
                gen_captions = [dataset.mapping[key] for key in gen_labels]

                gen_encoded_captions = []
                for caption in gen_captions:
                    gen_encoded_captions.append(tuple([dataset.encoded_vocab[word] for word in caption.split(" ")]))

                generator_input = Variable(
                    LongTensor(
                        gen_encoded_captions
                    )
                )

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            real_imgs = Variable(imgs.type(FloatTensor))

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise and labels/captions as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, obj.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z, generator_input)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, generator_input)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, discriminator_input)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), generator_input)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            printed = (epoch, i)
            if batches_done % opt.sample_interval == 0:
                if opt.model == "MNIST":
                    sample_image(printed, opt, obj, generator, dataloader)
                elif opt.model == "RNN_MNIST":
                    sample_image_rnn(printed, opt, obj, generator, dataloader, dataset)
                elif opt.model == "FLICKR8K":
                    sample_image_flickr(printed, opt, obj, generator, dataloader, dataset)
                else:
                    # Default loaded model -> RNN_MNIST
                    sample_image_rnn(printed, opt, obj, generator, dataloader, dataset)

                torch.save(generator.state_dict(), g_model_fileAUX)
                print("Saved PyTorch Model State of GENERATOR to '%s'" % g_model_fileAUX)

                torch.save(discriminator.state_dict(), d_model_fileAUX)
                print("Saved PyTorch Model State of GENERATOR to '%s'" % d_model_fileAUX)



