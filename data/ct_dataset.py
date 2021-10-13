import os.path

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset

from PIL import Image
import random
import numpy as np
from torchvision import transforms
import numpy as np
import torchvision.transforms.functional as F
from torch import Tensor
import math

class RandomCropIfNecessary():
    def __init__(self, size, n_downsampling):
        assert size%8==0
        self.size = size
        self.w = self.h = int(size/2)
        self.n_downsampling = n_downsampling

    def __call__(self, x):
        if x.shape[1] + x.shape[2] > self.size:
            if x.shape[1] < x.shape[2]:
                h = min(x.shape[1], self.h)
                w = self.size-h
            else:
                w = min(x.shape[2], self.w)
                h = self.size-w
            ind_h = random.randint(0, max(0, x.shape[1]-h))
            ind_w = random.randint(0, max(0, x.shape[2]-w))
            x = x[:, ind_h:ind_h + h, ind_w: ind_w + w]

        padding_h = x.shape[1]%(2**self.n_downsampling)
        padding_w = x.shape[2]%(2**self.n_downsampling)
        x = F.pad(x, (0,0,padding_w, padding_h), padding_mode='reflect')
        return x

class PadTo():
    def __init__(self, height: int, width: int):
        assert width%4 == 0 and height%4 == 0
        self.width = width
        self.height = height

    def __call__(self, x: Tensor) -> Tensor:
        h = x.shape[-2]
        w = x.shape[-1]
        pad = (math.floor((self.width-w)/2.), math.floor((self.height-h)/2.), math.ceil((self.width-w)/2.), math.ceil((self.height-h)/2.))
        assert min(pad)>=0, 'The encoutered image with shape %s is bigger than the configured maximum dimension'%str(x.shape)
        x = F.pad(x, pad, fill=0, padding_mode='constant')
        return x

class CTDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_dir = os.path.join(self.opt.dataroot, 'ct', self.opt.phase)
        self.B_dir = os.path.join(self.opt.dataroot, 'mri', self.opt.phase)
        self.A_paths = sorted(make_dataset(self.A_dir, opt.max_dataset_size)) 
        self.B_paths = sorted(make_dataset(self.B_dir, opt.max_dataset_size)) 
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        transformations = [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: self.center(x, opt.mean_norm, opt.std_norm)),
            # transforms.RandomCrop((148,100)),
            RandomCropIfNecessary(1200, opt.n_downsampling) if opt.uniform_size is None else PadTo(*opt.uniform_size),
            # transforms.Pad((1,0,0,0), padding_mode='reflect')
        ]
        if(opt.isTrain):
            transformations += [transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()]
        self.transformations = transforms.Compose(transformations)

    def normalize(self, x):
        x_min = x.amin()
        x_max = x.amax()
        x = (x - x_min) / (x_max - x_min) * 2. -1.
        return x

    def center(self, x, mean, std):
        return (x - mean) / std

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            S1 (tensor)       -- an image in the input domain
            S2 (tensor)       -- an image in the input domain
            S3 (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- concatenated image paths
            B_paths (str)    -- image paths
        """

        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B % self.B_size]

        A_img = np.array(Image.open(A_path), dtype=np.float32)
        B_img = np.array(Image.open(B_path), dtype=np.float32)

        if self.opt.serial_batches:
            AB_img = np.concatenate([A_img[:,:,None],B_img[:,:,None]], axis=2)
            AB_img = self.transformations(AB_img)
            A_img = AB_img[0:1]
            B_img = AB_img[1:2]
        else:
            A_img = self.transformations(A_img)
            B_img = self.transformations(B_img)

        return {'A': A_img, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

