import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

class TamilCharacterDataLoader:
    def __init__(self, train_dir, test_dir, transform, batch_size=64, val_split=0.2, seed=106):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.transform = transform
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed

    def load_data(self):
        training_set = datasets.ImageFolder(self.train_dir, transform=self.transform)
        trainsize = int(round((1 - self.val_split) * len(training_set)))
        valset_size = len(training_set) - trainsize
        
        trainset, valset = random_split(training_set, [trainsize, valset_size], generator=torch.Generator().manual_seed(self.seed))
        testset = datasets.ImageFolder(self.test_dir, transform=self.transform)
        
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        valloader = DataLoader(valset, batch_size=self.batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=True)
        
        return trainloader, valloader, testloader

    def get_classes(self, testset, unicode_csv):
        import pandas as pd
        df = pd.read_csv(unicode_csv, header=0)
        unicode_list = df["Unicode"].tolist()

        char_list = []
        for element in unicode_list:
            code_list = element.split()
            chars_together = ""
            for code in code_list:
                hex_str = "0x" + code
                char_int = int(hex_str, 16)
                character = chr(char_int)
                chars_together += character
            char_list.append(chars_together)

        classes = []
        for i in range(len(testset.classes)):
            index = int(testset.classes[i])
            char = char_list[index]
            classes.append(char)

        return classes
