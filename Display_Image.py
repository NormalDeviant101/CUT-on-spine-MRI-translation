import os, sys
import glob
import dominate
import ntpath
from dominate import document
from dominate.tags import *
from PIL import Image
from resizeimage import resizeimage

#Filling path to the image folder.
IMAGE_FOLDER_PATH = '/home/ana/CUT/CUT_TEST/test_6/images'
path_folder_rA = 'real_A'
path_folder_fB = 'fake_B'
path_folder_rB = 'real_B'

print("VALIDATING THE IMAGE FOLDER STATUS")
print(os.path.exists(IMAGE_FOLDER_PATH))

def read_image_to_list_rA(): # modal = real_A, real_B, etc..
    image_list = []
    path = os.path.join(IMAGE_FOLDER_PATH, path_folder_rA)
    path_list = glob.glob(os.path.join(path, '*'))
    if os.path.exists(path):
        for filename in path_list:
            try:
                short_path = ntpath.basename(filename)
                name = os.path.splitext(short_path)[0]
                if int(float(name)) >= 900:
                    print(name+ ' is appended.')
                    image_list.append(filename)
                else:
                    pass
            except ValueError:
                print(name,' is invalid')
    else:
        print('Real_A Image folder path Invalid')
    return image_list

def read_image_to_list_fB(): # modal = real_A, real_B, etc..
    image_list = []
    path = os.path.join(IMAGE_FOLDER_PATH, path_folder_fB)
    path_list = glob.glob(os.path.join(path, '*'))
    if os.path.exists(path):
        for filename in path_list:
            try:
                short_path = ntpath.basename(filename)
                name = os.path.splitext(short_path)[0]
                if int(float(name)) >= 900:
                    image_list.append(filename)
                    print(name + ' is appended.')
                else:
                    pass
            except ValueError:
                print(name,' is invalid')
    else:
        print('Fake_B Image folder path Invalid')
    return image_list

def read_image_to_list_rB(): # modal = real_A, real_B, etc..
    image_list = []
    path = os.path.join(IMAGE_FOLDER_PATH, path_folder_rB)
    path_list = glob.glob(os.path.join(path, '*'))
    if os.path.exists(path):
        for filename in path_list:
            try:
                short_path = ntpath.basename(filename)
                name = os.path.splitext(short_path)[0]
                if int(float(name)) >= 900:
                    image_list.append(filename)
                    print(name + ' is appended.')
                else:
                    pass
            except ValueError:
                print(name,' is invalid')
    else:
        print('Real_B Image folder path Invalid')
    return image_list

def create_page():#creating html page using dominate module
    doc = dominate.document(title='Image Gallery')
    print("Executing image generation of html file")

    with doc.head:
    	link(rel='stylesheet', href='style.css')

    #photos_real_A = glob.glob('/home/william/CUT/results/experiment_name/test_latest/images/real_A/*.jpg')
    #photos_fake_B = glob.glob('/home/william/CUT/results/experiment_name/test_latest/images/fake_B/*.jpg')
    #photos_real_B = glob.glob('/home/william/CUT/results/experiment_name/test_latest/images/real_B/*.jpg')

    photos_real_A = read_image_to_list_rA()
    photos_fake_B = read_image_to_list_fB()
    photos_real_B = read_image_to_list_rB()

    with doc:
        with div(cls='container_rA').add(h1('real_A')):
            for path in photos_real_A:
                div(img(src=path), _class='photo_real_A')
        with div(cls='container_fB').add(h1('fake_B')):
            for path in photos_fake_B:
                div(img(src=path), _class='photo_fake_B')
        with div(cls='container_rB').add(h1('real_B')):
            for path in photos_real_B:
                div(img(src=path), _class='photo_real_B')

    with open('Image_Display.html', 'w') as file:
        file.write(doc.render())

#print(read_image_to_list_rA())
#print(read_image_to_list_fB())
#print(read_image_to_list_rA())
create_page()

print("Finished image generation of html file")
