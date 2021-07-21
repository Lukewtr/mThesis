# Import the necessary modules
import torch
import torch.nn as nn
import numpy as np
import string
import csv
import os
from urllib import request
from zipfile import ZipFile
from PIL import Image

from torch.utils.data import Dataset


# Dataset for FLICKR8K:
class flickr8k_Dataset(Dataset):
    def __init__(self, root, transform, target_transform):
        self.folder = root

        # Images:
        url_dataset = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
        dataset_name = f'{self.folder}/dataset/Flickr8k_Dataset.zip'
        os.makedirs(f"{self.folder}/dataset", exist_ok=True)
        request.urlretrieve(url_dataset, dataset_name)
        with ZipFile(dataset_name, 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path=f"{root}/dataset")

        # Captions:
        # -Flickr30k:
        with open(f'{self.folder}/ann_file/captions.csv') as csv_file:
            reader = csv.reader(csv_file)
            captions = list(reader)
            csv_file.close()
        self.captions = [cap for [cap] in captions]

        # -Flickr8k:
        self.len_vocab = 6067

        with open(f'{self.folder}/ann_file/Flickr8k.token.txt') as f:
            caption = f.readline()

            imgs_id = []
            caps_id = []

            while caption:
                line = caption.lower().replace("\t", " ").split(" ")
                img_id, n_cap = line[0].split("#")

                if n_cap == "0":
                    imgs_id.append(img_id)

                line = ["<start>"]
                for word in caption.lower().replace("\t", " ").split(" ")[1:-1]:
                    if word.isdigit():
                        for digit in word:
                            line.append(digit)
                    else:
                        if word in string.punctuation:
                            printing = False
                        else:
                            line.append(word)

                line.append("<end>")
                caps_id.append(line)
                caption = f.readline()

            f.close()

        with open(f'{self.folder}/word_to_ix_{self.len_vocab}.csv') as csv_file:
            reader = csv.reader(csv_file)
            word_to_ix = dict(reader)
            word_to_ix.pop("idx")
            csv_file.close()

        with open(f'{self.folder}/ix_to_word_{self.len_vocab}.csv') as csv_file:
            reader = csv.reader(csv_file)
            ix_to_word = dict(reader)
            ix_to_word.pop("idx")
            csv_file.close()

        with open(f'{self.folder}/vocab_{self.len_vocab}.csv') as csv_file:
            reader = csv.DictReader(csv_file)
            vocab = []
            for row in reader:
                vocab.append(row["word"])
            csv_file.close()

        self.word_to_ix = word_to_ix
        self.ix_to_word = ix_to_word
        self.vocab = set(vocab)

        self.imgs_id = imgs_id
        self.caps_id = caps_id

        self.transform = transform

        self.target_transform = target_transform

    def __len__(self):
        return len(self.caps_id)

    def __getitem__(self, index: int):
        img_id = self.imgs_id[index // 5]

        # Images:
        img = Image.open(f"{self.fodler}/dataset/Flicker8k_Dataset/{img_id}").convert('RGB')
        if self.transform:
            img = self.transform(img)

        # Captions:
        target = self.caps_id[index]

        emb_caption = []
        for word in target:
            emb_caption.append(self.word_to_ix[word]) if word in self.vocab else emb_caption.append(self.word_to_ix["UNK"])

        if self.target_transform:
            emb_caption = self.target_transform(emb_caption)

        return img, emb_caption
