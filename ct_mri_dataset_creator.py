import numpy as np
from PIL import Image
import nibabel as nib
import os
from tqdm import tqdm

def is_nifty_file(filename: str):
    return filename.endswith('ct.nii.gz')

def make_dataset(dir, max_dataset_size=float("inf")):
    ct_scans = []
    mri_scans = []
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_nifty_file(fname):
                fname: str
                path = os.path.join(root, fname)
                ct_scans.append(path)
                folder = os.path.basename(path)
                if os.path.exists(path[:-9]+'dixon.nii.gz'):
                    mri_scans.append(path[:-9]+'dixon.nii.gz')
                else:
                    mri_scans.append(path[:-9]+'t1.nii.gz')
    return ct_scans[:min(max_dataset_size, len(ct_scans))], mri_scans[:min(max_dataset_size, len(mri_scans))]

ct, mri = make_dataset('/media/data_4T/william/CT_2_MRI_raw')
counter = 0
if not os.path.exists('/media/data_4T/william/CT_2_MRI/ct'):
    os.makedirs('/media/data_4T/william/CT_2_MRI/ct')
    os.makedirs('/media/data_4T/william/CT_2_MRI/mri')
if not os.path.exists('/media/data_4T/william/CT_2_MRI/ct/train'):
    os.makedirs('/media/data_4T/william/CT_2_MRI/ct/test')
    os.makedirs('/media/data_4T/william/CT_2_MRI/ct/train')
    os.makedirs('/media/data_4T/william/CT_2_MRI/mri/test')
    os.makedirs('/media/data_4T/william/CT_2_MRI/mri/train')
for i in tqdm(range(len(ct))):
    ct_image = nib.load(ct[i]).get_fdata()
    ct_image = (((ct_image - ct_image.min()) / (ct_image.max() - ct_image.min())) * 255).astype(np.uint8)
    mri_image = nib.load(mri[i]).get_fdata()
    mri_image = (((mri_image - mri_image.min()) / (mri_image.max() - mri_image.min())) * 255).astype(np.uint8)
    for z in range(ct_image.shape[2]):
        if np.min(ct_image[:,:,z].shape) > 100:
            counter+=1
            if counter<=774:
                Image.fromarray(ct_image[:,:,z]).save('/media/data_4T/william/CT_2_MRI/ct/test/{}.png'.format(1000*i + z))
                Image.fromarray(mri_image[:,:,z]).save('/media/data_4T/william/CT_2_MRI/mri/test/{}.png'.format(1000*i + z))
            else:
                Image.fromarray(ct_image[:,:,z]).save('/media/data_4T/william/CT_2_MRI/ct/train/{}.png'.format(1000*i + z))
                Image.fromarray(mri_image[:,:,z]).save('/media/data_4T/william/CT_2_MRI/mri/train/{}.png'.format(1000*i + z))
        else:
            break
        
print(counter)
print('Done')