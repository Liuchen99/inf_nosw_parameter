from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from timm.data.transforms_factory import create_transform

from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CIFAR10(datasets.CIFAR10):
    def __init__(self, opt):
        self.opt=opt
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if opt["train"]:
            if opt["large"]: 
                transform = create_transform(
                input_size=(3, 224, 224),
                is_training=opt["train"],   # is training, true or false
                interpolation="bicubic", 
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                crop_pct=0.95,
                crop_mode="center"
                )
            else:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
        else:
            if opt["large"]:
                transform = create_transform(
                    input_size=(3, 224, 224),
                    is_training=opt["train"],   # is training, true or false
                    interpolation="bicubic",
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    crop_pct=0.95,
                    crop_mode="center"
                )
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
        print(opt["train"])
        super().__init__(root=opt["dataroot"], train=opt["train"], transform=transform, download=opt["download"])
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        
        return {"img": img, "target": target}


@DATASET_REGISTRY.register()
class CIFAR100(datasets.CIFAR100):
    def __init__(self, opt):
        self.opt=opt
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        if opt["train"]:
            if opt["large"]: 
                transform = create_transform(
                input_size=(3, 224, 224),
                is_training=opt["train"],   # is training, true or false
                interpolation="bicubic", 
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                crop_pct=0.95,
                crop_mode="center"
                )
            else:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
        else:
            if opt["large"]:
                transform = create_transform(
                    input_size=(3, 224, 224),
                    is_training=opt["train"],   # is training, true or false
                    interpolation="bicubic",
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    crop_pct=0.95,
                    crop_mode="center"
                )
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
        super().__init__(root=opt["dataroot"], train=opt["train"], transform=transform, download=opt["download"])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return {"img": img, "target": target}
    

@DATASET_REGISTRY.register()
class ImageNet(datasets.ImageFolder):
    def __init__(self, opt):
        self.opt = opt

        transform = create_transform(
            input_size=(3, 224, 224),
            is_training=opt["train"],   # is training, true or false
            interpolation="bicubic", 
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            crop_pct=0.95,
            crop_mode="center"
        )

        super().__init__(root=opt["dataroot"], transform=transform)

    def create_transform(is_training):
        return 

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"img": sample, "target": target}