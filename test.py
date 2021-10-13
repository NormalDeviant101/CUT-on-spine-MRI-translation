"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util.visualizer import Visualizer
from util import html
import util.util as util
import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.use_mask = False
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers 
            #model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 1 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, width=opt.display_winsize, mean=opt.mean_norm, std=opt.std_norm)
    webpage.save()  # save the HTML
















print("____________________________________________________________________________________________________")
print("__                                                                                                __")
print("__                          DEBUGGING PART OF THE SCRIPT STARTED WATCH OUT!                       __")
print("__                                                                                                __")
print("____________________________________________________________________________________________________")


print('--------------------------------- Print Image Path of model -----------------------------------------')
img_path = model.get_image_paths()
short_path = os.path.basename(img_path).strip("']")
print(img_path)
print(type(img_path))
print(short_path)
name = os.path.splitext(short_path)[0]
print(name)
#print(img_path_save)
print('--------------------------------- Print Image Path of model -----------------------------------------')
##img_ordered_dict = visuals
#print(type(img_ordered_dict))
#for img_dict in img_ordered_dict.items():
    #print(type(img_dict))
    #print(img_dict)
    #im_data = img_dict.values()
    #im = util.tensor2im(im_data)
    #print(np.shape(im))

#print(type(img_dict))
#print(img_dict.keys())
#print(np.shape(im))
#for label in img_dict:
    #print(label)



print('--------------------------------- Print Dimension of html Real_A size --------------------------------')
print(type(visuals['real_A']))
print(np.shape(visuals['real_A']))

print('--------------------------------- Print Dimension of html fake_B size --------------------------------')
print(type(visuals['fake_B']))
print(np.shape(visuals['fake_B']))

print('--------------------------------- Print Dimension of html Real_B size --------------------------------')
print(type(visuals['real_B']))
print(np.shape(visuals['real_B']))

print('--------------------------------- Print image directory of the webpage -------------------------------')
#image_dir = webpage.get_image_dir()
#print(type(image_dir))
#print(type(dataset))

#for data in enumerate(dataset):
    #print(img_path)
    #print(type(img_path))
    #short_path = ntpath.basename(img_path[2:42])
    #rint(short_path)
    #name = os.path.splitext(short_path)[0]
    #print(name)

print('--------------------------------- Print dataset information of the test -------------------------------')
print(type(dataset))
print("____________________________________________________________________________________________________")
print("__                                                                                                __")
print("__                          DEBUGGING PART 2         !!!!!!!!!!                                   __")
print("__                                                                                                __")
print("____________________________________________________________________________________________________")

print('--------------------------------- Print Names of the images ----------------------------------------')
print(type(name))
print(len(name))
print(name)
print('--------------------------------- Print real_A and real_B and Fake_B information -------------------')
print(type(model.get_real_A()))
print(np.shape(model.get_real_A()))
print('--------------------------------- Print Visual information -----------------------------------------')

print(type(visuals))
print(len(visuals))
for key, value in visuals.items():
    print(key, ' with shape ', np.shape(value))
for label, im_data in visuals.items():
    im_1 = util.tensor2im(im_data)
    print(np.shape(im_1[:,:,0, np.newaxis]))

print('--------------------------------- Print Image Path Information 1------------------------------------')
print(type(model.get_image_paths()))
print(len(model.get_image_paths()))
print(model.get_image_paths_save())
print('--------------------------------- Print Image Path Information 2------------------------------------')
img_dir, short_path = Visualizer.save_image_path_checker(webpage, model.get_image_paths_save())
print(type(img_dir))
print(img_dir)

