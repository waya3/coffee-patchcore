from pathlib import Path
from torch import tensor
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

DATASETS_PATH = Path("/home/kby/mnt/hdd/coffee/PatchCore")
IMAGENET_MEAN = tensor([.485, .456, .406])
IMAGENET_STD = tensor([.229, .224, .225])
SIZE = 224

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
    def __init__(self, cls):    #cls=coffeeDir
        self.cls = cls
        self.train_ds = CustomImageDataset(DATASETS_PATH / cls / "train", transform)
        self.test_ds = CustomImageDataset(DATASETS_PATH / cls / "test", transform)

    def get_dataloaders(self):
        return DataLoader(self.train_ds), DataLoader(self.test_ds)


