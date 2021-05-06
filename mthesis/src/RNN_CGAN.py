# Import the necessary modules
import torch
import torch.nn as nn
import numpy as np

# Generator for RNN_CGAN:
class RNN_Generator(nn.Module):

    def __init__(self, vocab_size: int, parser, image_shape):
        super(RNN_Generator, self).__init__()

        self.opt = parser
        self.embedding_dim = self.opt.embedding_dim
        self.hidden_size = self.opt.hidden_dim
        self.img_shape = image_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.caption_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim)
        self.rnn = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_size, batch_first=True)

        self.model = nn.Sequential(
            *block(self.opt.latent_dim + self.hidden_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, captions):
        captions_embedded = self.rnn(self.caption_emb(captions))[0][:, -1, :]

        # Concatenate caption embedding and noise to produce input
        gen_input = torch.cat((captions_embedded, noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

# Discriminator for RNN_CGAN:
class RNN_Discriminator(nn.Module):
    def __init__(self, vocab_size: int, parser, image_shape):
        super(RNN_Discriminator, self).__init__()

        self.opt = parser
        self.embedding_dim = self.opt.embedding_dim
        self.hidden_size = self.opt.hidden_dim
        self.img_shape = image_shape

        # self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        self.caption_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_dim)
        self.rnn = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_size, batch_first=True)

        self.model = nn.Sequential(
            nn.Linear(self.hidden_size + int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, captions):
        captions_embedded = self.rnn(self.caption_emb(captions))[0][:, -1, :]

        # Concatenate caption embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), captions_embedded), -1)
        validity = self.model(d_in)
        return validity

