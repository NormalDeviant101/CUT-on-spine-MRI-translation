import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset

from PIL import Image
import random
import util.util as util
import numpy as np
from torchvision import transforms
import numpy as np
import torch
from skimage.color import rgb2gray
from matplotlib import cm
import SimpleITK as sitk
import imageio
import matplotlib.image as mpimg
#Script Fix from Github for transform functions
transform = transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))
])

print("Started~!")

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
        #self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A', 'A_img')  # create a path '/path/to/data/trainA/S1'
        self.dir_S1 = os.path.join(opt.dataroot, opt.phase + 'A' , 'S1')  # create a path '/path/to/data/trainA/S1'
        self.dir_S2 = os.path.join(opt.dataroot, opt.phase + 'A' , 'S2')  # create a path '/path/to/data//trainA/S2'
        self.dir_S3 = os.path.join(opt.dataroot, opt.phase + 'A' , 'S3')  # create a path '/path/to/data/trainA/S3'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B', 'T')  # create a path '/path/to/data/trainB/T'

        if opt.phase == "test" and not os.path.exists(self.dir_S1) and not os.path.exists(self.dir_S2) and not os.path.exists(self.dir_S3)\
           and os.path.exists(os.path.join(opt.dataroot, "valS1")) and os.path.exists(os.path.join(opt.dataroot, "valS2")) and os.path.exists(os.path.join(opt.dataroot, "valS3")):
           #self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_S1 = os.path.join(opt.dataroot, "valS1")
            self.dir_S2 = os.path.join(opt.dataroot, "valS2")
            self.dir_S3 = os.path.join(opt.dataroot, "valS3")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        #self.A_path = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA/S1'
        self.S1_paths = sorted(make_dataset(self.dir_S1, opt.max_dataset_size))   # load images from '/path/to/data/trainA/S1'
        self.S2_paths = sorted(make_dataset(self.dir_S2, opt.max_dataset_size))  # load images from '/path/to/data/trainA/S2'
        self.S3_paths = sorted(make_dataset(self.dir_S3, opt.max_dataset_size))  # load images from '/path/to/data/trainA/S3'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.S1_size = len(self.S1_paths)  # get the size of dataset S1
        self.S2_size = len(self.S2_paths)  # get the size of dataset S2
        self.S3_size = len(self.S3_paths)  # get the size of dataset S3
        self.B_size = len(self.B_paths)  # get the size of dataset B

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

        #sitkimage = sitk.ReadImage(S1_)
        #numpyImage = sitk.GetArrayFromImage(sitkimage)
        #A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        S1_path = self.S1_paths[index % self.S1_size]  # make sure index is within then range
        S2_path = self.S2_paths[index % self.S2_size]  # make sure index is within then range
        S3_path = self.S3_paths[index % self.S3_size]  # make sure index is within then range
        B_path = self.B_paths [index % self.B_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        #sitkimage_S1 = sitk.ReadImage(S1_path)
        #sitkimage_S2 = sitk.ReadImage(S2_path)
        #sitkimage_S3 = sitk.ReadImage(S3_path)
        #sitkimage_B = sitk.ReadImage(B_path)

        #img_array_S1 = mpimg.imread(S1_path)
        #img_array_S2 = mpimg.imread(S2_path)
        #img_array_S3 = mpimg.imread(S3_path)
        #img_array_B = mpimg.imread(B_path)

        #img_array_S1 = np.array(Image.open(S1_path).convert('LA'))[:, :, 0]  # [:, :, np.newaxis] #shape (320, 320, 1)
        #img_array_S2 = np.array(Image.open(S2_path).convert('LA'))[:, :, 0]  # [:, :, np.newaxis] #shape (320, 320, 1)
        #img_array_S3 = np.array(Image.open(S3_path).convert('LA'))[:, :, 0]  # [:, :, np.newaxis] #shape (320, 320, 1)
        #img_array_B = np.array(Image.open(B_path).convert('LA'))[:, :, 0]  # [:, :, np.newaxis] #shape (320, 320, 1)

        img_array_S1 = np.array(Image.open(S1_path))#[:, :, np.newaxis] #shape (320, 320, 1)
        img_array_S2 = np.array(Image.open(S2_path))#[:, :, np.newaxis] #shape (320, 320, 1)
        img_array_S3 = np.array(Image.open(S3_path))#[:, :, np.newaxis] #shape (320, 320, 1)
        img_array_B = np.array(Image.open(B_path))#[:, :, np.newaxis] #shape (320, 320, 1)

        img_array_S1 = img_array_S1[:, :, np.newaxis]  # [:, :, np.newaxis] #shape (320, 320, 1)
        img_array_S2 = img_array_S2[:, :, np.newaxis]  # [:, :, np.newaxis] #shape (320, 320, 1)
        img_array_S3 = img_array_S3[:, :, np.newaxis]  # [:, :, np.newaxis] #shape (320, 320, 1)
        img_array_B = img_array_B[:, :, np.newaxis]  # [:, :, np.newaxis] #shape (320, 320, 1)

        self.img_array_S1 = img_array_S1 #shape (320, 320, 1)
        self.img_array_S2 = img_array_S2 #shape (320, 320, 1)
        self.img_array_S3 = img_array_S3 #shape (320, 320, 1)
        self.img_array_B = img_array_B #shape (320, 320, 1)
        #print('the S1 shape is_', np.shape(img_array_S1))
        #print('the S2 shape is_', np.shape(img_array_S2 ))
        #print('the S3 shape is_', np.shape(img_array_S3))
        #print('the B shape is_', np.shape(img_array_B))
        #A_img = Image.open(A_path).convert('RGB')
        #B_img = Image.open(B_path)
        #A_img = np.concatenate((self.img_array_S1, self.img_array_S2, self.img_array_S3),axis=-1)
        #B_img = np.concatenate((self.img_array_B, self.img_array_B, self.img_array_B), axis=-1)

        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
	    # print('current_epoch', self.current_epoch)
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)

        # transform np array to PIL image
        #A_img = Image.fromarray(np.uint8(A_img))
        img_array_S1_RGB = np.concatenate((self.img_array_S1,self.img_array_S1, self.img_array_S1),axis=-1) # shape (320, 320, 3)
        img_array_S2_RGB = np.concatenate((self.img_array_S2,self.img_array_S2,self.img_array_S2),axis=-1) # shape (320, 320, 3)
        img_array_S3_RGB = np.concatenate((self.img_array_S3, self.img_array_S3, self.img_array_S3),axis=-1) # shape (320, 320, 3)
        #print('the fucking 2D shape is_', np.shape(img_array_S1))
        #print('the fucking 3D shape is_', np.shape(img_array_S1_RGB))
        #S1 - S3 transformation to Tensors
        S1 = transform(Image.fromarray(np.uint8(img_array_S1_RGB * 255)))  # shape (320, 320, 3)
        #S1 = S1.permute(1,2,0)
        #print('--------------- S1 Tensor.shape --------------')
        #print(S1.shape[0])
        #print(S1.shape[1])
        #print(S1.shape[2])

        S2 = transform(Image.fromarray(np.uint8(img_array_S2_RGB * 255)))  # shape (320, 320, 3)
        #S2 = S2.permute(1, 2, 0)
        #print('--------------- S2 Tensor.shape --------------')
        #print(S2.shape[0])
        #print(S2.shape[1])
        #print(S2.shape[2])

        S3 = transform(Image.fromarray(np.uint8(img_array_S3_RGB * 255)))  # shape (320, 320, 3)
        #S3 = S3.permute(1, 2, 0)
        #print('--------------- S3 Tensor.shape --------------')
        #print(S3.shape[0])
        #print(S3.shape[1])
        #print(S3.shape[2])

        #
        A = np.concatenate((self.img_array_S1, self.img_array_S2, self.img_array_S3), axis=-1) # shape (320, 320, 3)
        A = transform(Image.fromarray(np.uint8(A*255))) #shape (320, 320, 3)
        #A = A.permute(1, 2, 0)
        #print('--------------- A Tensor.shape --------------')
        #print(A.shape[0])
        #print(A.shape[1])
        #print(A.shape[2])
        #
        B = np.concatenate((self.img_array_B, self.img_array_B, self.img_array_B), axis=-1)  #shape (320, 320, 3)
        B = transform(Image.fromarray(np.uint8(B*255))) #shape (320, 320, 3)
        #B = B.permute(1,2,0)
        #print('--------------- B Tensor.shape --------------')
        #print(B.shape[0])
        #print(B.shape[1])
        #print(B.shape[2])



        return {'S1': S1, 'S2': S2, 'S3': S3, 'A': A, 'B': B, 'S1_paths': S1_path, 'S2_paths': S2_path, 'S3_paths': S3_path,'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.S1_size, self.S2_size, self.S3_size, self.B_size)

