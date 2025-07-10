from pathlib import Path
from torch import tensor
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from params import TRAINDIR, TESTDIR
import os, random, shutil, glob
import pathlib

IMAGENET_MEAN = tensor([.485, .456, .406])
IMAGENET_STD = tensor([.229, .224, .225])
SIZE = 128
# SIZE = 256
TESTOMOTE = "/home/kby/mnt/hdd/coffee/PatchCore/coffee/test_omote"
TESTURA = "/home/kby/mnt/hdd/coffee/PatchCore/coffee/test_ura"

transform = transforms.Compose([
    transforms.Resize(SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class CustomImageDataset(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root=root, transform=transform)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, path, target

class Dataset:
    def __init__(self, ):
        # self.cls = cls
        self.train_ds = CustomImageDataset(TRAINDIR, transform)
        self.get_files_path()
        self.test_ds = CustomImageDataset(TESTDIR, transform)

    def get_dataloaders(self):
        return DataLoader(self.train_ds), DataLoader(self.test_ds)


    def get_files_path(self):
        root_dir = pathlib.Path(TESTDIR)
        shutil.rmtree(TESTDIR+"/omote/good")
        shutil.rmtree(TESTDIR+"/ura/good")
        for subdir, dirs, files in os.walk(TESTOMOTE + "/good"):
            pathlib.Path(TESTDIR+"/omote/good").mkdir()
            random_paths = random.sample(files, 65)
            for _, fi in enumerate(random_paths):
                f_name = os.path.split(fi)[1]
                shutil.copy(TESTOMOTE+"/good/"+f_name, TESTDIR+"/omote/good/"+f_name)

        for subdir, dirs, files in os.walk(TESTURA + "/good"):
            pathlib.Path(TESTDIR+"/ura/good").mkdir()
            random_paths = random.sample(files, 65)
            for _, fi in enumerate(random_paths):
                f_name = os.path.split(fi)[1]
                shutil.copy(TESTURA+"/good/"+f_name, TESTDIR+"/ura/good/"+f_name)

    def make_dir(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)