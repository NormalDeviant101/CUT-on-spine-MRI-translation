import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset

from PIL import Image
import random
import numpy as np
from torchvision import transforms
import numpy as np
import torch

class UnalignedDataset(BaseDataset):
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
        self.dir_S1 = os.path.join(opt.dataroot, opt.phase + 'A' , 'S1')  # create a path '/path/to/data/trainA/S1'
        self.dir_S2 = os.path.join(opt.dataroot, opt.phase + 'A' , 'S2')  # create a path '/path/to/data//trainA/S2'
        self.dir_S3 = os.path.join(opt.dataroot, opt.phase + 'A' , 'S3')  # create a path '/path/to/data/trainA/S3'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B', 'T')  # create a path '/path/to/data/trainB/T'

        if opt.use_mask:
            self.dir_mask_S1 = os.path.join(opt.dataroot, opt.phase + 'A', 'mask_S1')  # create a path '/path/to/data/trainA/mask_S1'
            self.dir_mask_T = os.path.join(opt.dataroot, opt.phase + 'B', 'mask_T')  # create a path '/path/to/data/trainB/mask_T'

        if opt.phase == "test" and (not os.path.exists(self.dir_S1) or not os.path.exists(self.dir_S2) or not os.path.exists(self.dir_S3))\
           and os.path.exists(os.path.join(opt.dataroot, "valS1")) and os.path.exists(os.path.join(opt.dataroot, "valS2")) and os.path.exists(os.path.join(opt.dataroot, "valS3")):
           #self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_S1 = os.path.join(opt.dataroot, "valS1")
            self.dir_S2 = os.path.join(opt.dataroot, "valS2")
            self.dir_S3 = os.path.join(opt.dataroot, "valS3")
            self.dir_B = os.path.join(opt.dataroot, "valB")

            if opt.use_mask:
                self.dir_mask_S1 = os.path.join(opt.dataroot, "valmask_S1")
                self.dir_mask_T = os.path.join(opt.dataroot, "valmask_T")

        #self.A_path = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA/S1'
        self.S1_paths = sorted(make_dataset(self.dir_S1, opt.max_dataset_size))   # load images from '/path/to/data/trainA/S1'
        self.S2_paths = sorted(make_dataset(self.dir_S2, opt.max_dataset_size))  # load images from '/path/to/data/trainA/S2'
        self.S3_paths = sorted(make_dataset(self.dir_S3, opt.max_dataset_size))  # load images from '/path/to/data/trainA/S3'
        self.S1_size = len(self.S1_paths)  # get the size of dataset S1
        self.S2_size = len(self.S2_paths)  # get the size of dataset S2
        self.S3_size = len(self.S3_paths)  # get the size of dataset S3
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.B_size = len(self.B_paths)  # get the size of dataset B

        if opt.phase == "train" and opt.use_mask:
            self.mask_S1_paths = sorted(make_dataset(self.dir_mask_S1, opt.max_dataset_size))  # load images from '/path/to/data/trainA/mask_S1'
            self.mask_T_paths = sorted(make_dataset(self.dir_mask_T, opt.max_dataset_size))  # load images from '/path/to/data/trainA/mask_S1'
            self.mask_S1_size = len(self.mask_S1_paths)  # get the size of dataset masks
            self.mask_T_size = len(self.mask_T_paths)  # get the size of dataset masks

        self.transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: self.normalize(x)),
            transforms.Pad(2, padding_mode='reflect')
        ])


    def normalize(self, x):
        x_min = x.amin()
        x_max = x.amax()
        x = (x - x_min) / x_max * 2 -1
        return x

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

        S1_path = self.S1_paths[index % self.S1_size]  # make sure index is within then range
        S2_path = self.S2_paths[index % self.S2_size]  # make sure index is within then range
        S3_path = self.S3_paths[index % self.S3_size]  # make sure index is within then range

        B_path = self.B_paths[index % self.B_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        img_array_S1 = np.array(Image.open(S1_path), dtype=np.float32)
        img_array_S2 = np.array(Image.open(S2_path), dtype=np.float32)
        img_array_S3 = np.array(Image.open(S3_path), dtype=np.float32)
        img_array_B = np.array(Image.open(B_path), dtype=np.float32)

        img_array_S123 = np.stack((img_array_S1, img_array_S2, img_array_S3), axis=2)

        A = self.transformations(img_array_S123)
        B = self.transformations(img_array_B[..., np.newaxis])


        if self.opt.phase == "train" and self.opt.use_mask:
            mask_S1_path = self.mask_S1_paths[index % self.mask_S1_size]  # make sure index is within then range
            mask_T_path = self.mask_T_paths[index % self.mask_T_size]  # make sure index is within then range
            img_array_mask_S1 = np.array(Image.open(mask_S1_path), dtype=np.float32)
            img_array_mask_T = np.array(Image.open(mask_T_path), dtype=np.float32)
            mask_S1 = torch.from_numpy(img_array_mask_S1).unsqueeze(0)
            mask_T = torch.from_numpy(img_array_mask_T).unsqueeze(0)

            mask_S1 = transforms.F.pad(mask_S1, 2, padding_mode='reflect')
            mask_T = transforms.F.pad(mask_T, 2, padding_mode='reflect')

            return {'mask_T': mask_T, 'mask_S1': mask_S1, 'A': A, 'B': B, 'S1_paths': S1_path, 'S2_paths': S2_path,
                    'S3_paths': S3_path, 'mask_T_paths': mask_T_path, 'mask_S1_paths': mask_S1_path, 'B_paths': B_path}

        return {'A': A, 'B': B, 'S1_paths': S1_path, 'S2_paths': S2_path, 'S3_paths': S3_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.S1_size, self.S2_size, self.S3_size, self.B_size)

