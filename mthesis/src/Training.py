# Import the necessary modules
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torch.autograd import Variable
import numpy as np
from utils.utils import sample_image, sample_image_rnn

# Dataset for captions
class rnnMNIST_Dataset(Dataset):

    def __init__(self, root, train, download, transform):
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform
        )

        self.mapping = {0: "this is zero", 1: "this is one", 2: "this is two", 3: "this is three", 4: "this is four",
                        5: "this is five", 6: "this is six", 7: "this is seven", 8: "this is eight",
                        9: "this is nine"}

        self.encoded_vocab = {"this": 0, "is": 1, "zero": 2, "one": 3, "two": 4, "three": 5, "four": 6,
                              "five": 7, "six": 8, "seven": 9, "eight": 10, "nine": 11}

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index: int):
        image, number = self.mnist[index]
        caption = self.mapping[number]
        encoded_caption = torch.tensor([self.encoded_vocab[word] for word in caption.split(" ")])

        return image, caption, encoded_caption


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

    for epoch in range(opt.n_epochs):
        for i, data in enumerate(dataloader):
            if opt.model == "MNIST":
                (imgs, labels) = data

            elif opt.model == "RNN_MNIST":
                (imgs, captions, encoded_captions) = data

            else:
                # Default loaded model -> RNN_MNIST
                (imgs, captions, encoded_captions) = data

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            if opt.model == "MNIST":
                discriminator_input = Variable(labels.type(LongTensor))

            elif opt.model == "RNN_MNIST":
                discriminator_input = Variable(encoded_captions.type(LongTensor))

            else:
                # Default loaded model -> RNN_MNIST
                discriminator_input = Variable(encoded_captions.type(LongTensor))


            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise and labels/captions as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, obj.latent_dim))))

            gen_labels = np.random.randint(0, obj.n_classes, batch_size)

            if opt.model == "MNIST":
                generator_input = Variable(LongTensor(gen_labels))

            elif opt.model == "RNN_MNIST":
                gen_captions = [dataset.mapping[key] for key in gen_labels]

                gen_encoded_captions = []
                for caption in gen_captions:
                    gen_encoded_captions.append(tuple([dataset.encoded_vocab[word] for word in caption.split(" ")]))

                generator_input = Variable(
                    LongTensor(
                        gen_encoded_captions
                    )
                )

            else:
                # Default loaded model -> RNN_MNIST
                gen_captions = [dataset.mapping[key] for key in gen_labels]

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
                else:
                    # Default loaded model -> RNN_MNIST
                    sample_image_rnn(printed, opt, obj, generator, dataloader, dataset)

                torch.save(generator.state_dict(), g_model_fileAUX)
                print("Saved PyTorch Model State of GENERATOR to '%s'" % g_model_fileAUX)

                torch.save(discriminator.state_dict(), d_model_fileAUX)
                print("Saved PyTorch Model State of GENERATOR to '%s'" % d_model_fileAUX)



