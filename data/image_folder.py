"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path

import numpy as np
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images= []
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path= os.path.join(root,fname) # There should be a folder named S1 under the Root folder
                images.append(path)
 
    return images[:min(max_dataset_size, len(images))]

def normalize_array(array):
    array = array / 127.5 - 1
 #   array = array / 100
    return array

#def default_loader(path):
    #return Image.open(path).convert('RGB')

def default_loader(path):
    images_array = []
    for image in path[1]:
		    image = np.array(Image.open(path))
		    image = normalize_array(image)
		    images_array.append(image)
    return images_array

'''
def default_loader(path):
	#read image array S1
	image_array = []
	for image_name in path:
		if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
		    if nr_of_channels == 1:  # Gray scale image -> MR image
		        image = np.array(Image.open(os.path.join(path, image_name)))
		        image = image[:, :, np.newaxis]
		    else:                   # RGB image -> 3 channels
		        image = np.array(Image.open(os.path.join(image_path, image_name)))
		    image = normalize_array(image)
		    image_array.append(image)
	#read image array S2

	#read image array S3
	images = np.concatenate((self.S1_train, self.S2_train, self.S3_train), axis=-1)
    return Image.open(path).convert('RGB')
'''

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
            imgs_S1 = make_dataset(root) # read img path list s1, the list should be [img_paths, len(imgs)]
            imgs_S2 = make_dataset(root) # read img path list s2, the list should be [img_paths, len(imgs)]
            imgs_S3 = make_dataset(root) # read img path list s3, the list should be [img_paths, len(imgs)]
            imgs_T = make_dataset(root)  # read img path list T, the list should be [img_paths, len(imgs)]
            if len(imgs_S1) == 0: raise(RuntimeError("Found 0 images in: " + root + "\n" "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

            self.root = root
            self.imgs_S1 = imgs_S1
            self.imgs_S2 = imgs_S2
            self.imgs_S3 = imgs_S3
            self.imgs_T = imgs_T
            self.transform = transform
            self.return_paths = return_paths
            self.loader = loader

    def __getitem__(self, index):
            path_S1 = self.imgs_S1[index] #read path from img path list with random index
            path_S2 = self.imgs_S2[index] #read path from img path list with random index
            path_S3 = self.imgs_S3[index] #read path from img path list with random index
            path_T = self.imgs_T[index]  # read path from img path list with random index

            img_array_S1 = self.loader(path_S1) #load img with Default img loader for img array
            img_array_S2 = self.loader(path_S2) #load img with Default img loader for img array
            img_array_S3 = self.loader(path_S3) #load img with Default img loader for img array
            img_array_T = self.loader(path_T)  # load img with Default img loader for img array

            self.img_array_S1 = img_array_S1
            self.img_array_S2 = img_array_S2
            self.img_array_S3 = img_array_S3
            self.img_array_T = img_array_T

            if self.transform is not None:
                img_S1 = self.transform(img_array_S1)
                img_S2 = self.transform(img_array_S2)
                img_S3 = self.transform(img_array_S3)
                img_T = self.transform(img_array_T)
                # transform concatenated img arrays to tensor
            if self.return_paths:
                return img_S1, img_S2, img_S3, img_T,path_S1,path_S2,path_S3
            else:
                return img_S1, img_S2, img_S3, img_T

    def __len__(self):
        return len(self.imgs)

