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
from pathlib import Path

from torch.utils.data import Dataset


# Dataset for FLICKR8K:
class flickr8k_Dataset(Dataset):
    def __init__(self, root, transform, target_transform):
        self.folder = root

        # Images:
        print("Accessing url database for downloading images:")
        url_dataset = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
        dataset_name = f'{self.folder}/dataset/Flickr8k_Dataset'
        dataset_name_zip = f'{dataset_name}.zip'
        os.makedirs(f"{self.folder}/dataset", exist_ok=True)

        downloaded_dataset = Path(dataset_name_zip)
        if not downloaded_dataset.exists():
            print("...downloading...")
            request.urlretrieve(url_dataset, dataset_name_zip)
        print("Images download COMPLETED!!!")

        downloaded_dataset = Path(dataset_name)
        if not downloaded_dataset.exists():
            with ZipFile(dataset_name_zip, 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                print("...unzipping...")
                zipObj.extractall(path=f"{root}/dataset")
        print("Images AVAILABLE!!!")

        # Captions:
        # -Flickr30k:
        with open(f'{self.folder}/ann_file/captions.csv', encoding='utf8') as csv_file:
            reader = csv.reader(csv_file)
            captions = [st for st in reader]
            csv_file.close()
        self.captions = captions

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

    def fword_to_ix(self, word):
        if word in self.vocab:
            return int(self.word_to_ix[word])
        else:
            return int(self.word_to_ix["UNK"])

    def __len__(self):
        return len(self.caps_id)

    def __getitem__(self, index: int):
        img_id = self.imgs_id[index // 5]

        # Images:
        img = Image.open(f"{self.folder}/dataset/Flickr8k_Dataset/{img_id}").convert('RGB')
        if self.transform:
            img = self.transform(img)

        # Captions:
        target = self.caps_id[index]

        emb_caption = []
        for word in target:
            emb_caption.append(self.fword_to_ix(word))

        if self.target_transform:
            emb_caption = self.target_transform(emb_caption)

        return img, emb_caption
