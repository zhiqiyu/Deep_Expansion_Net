"""Build a SR-ready dataset from UCMerced Land Use data"""

import argparse
import random
import os
import glob

from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--scale', default=2, help="The scale to resize original images")
parser.add_argument('--data_dir', default='../../Datasets/UCMerced_LandUse/Images/', help="Directory that contain the original UCMerced dataset")
parser.add_argument('--output_dir', default='data/x2/', help="Where to write the new data")


class UCMercedBuilder:
    '''
    Build a super-resolution task oriented dataset from UCMerced Land Use dataset. 
    Use the down-scaled images and original    
    '''
    def __init__(self, data_dir, out_dir, valid_ratio=0.1, test_ratio=0.5, resample=Image.BICUBIC, scale=2):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.scale = scale
        self.resample = resample
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

    def run(self):
        # get the folders (classes) of the original data
        folder_list = glob.glob(self.data_dir + '*')

        # create training and testing folders in the output directory
        os.mkdir(os.path.join(self.out_dir, 'training'))
        os.mkdir(os.path.join(self.out_dir, 'validation'))
        os.mkdir(os.path.join(self.out_dir, 'test'))

        with tqdm(total=len(folder_list)) as pbar:
            for folder in folder_list:
                # get the paths of images in the folder
                img_list = glob.glob(folder + '/*.tif')

                for img in img_list:
                    # read image as PIL.Image.Image object
                    im = Image.open(img)

                    # get the original image size, and filename
                    filename = im.filename
                    width, height = im.size

                    # check if size matches (256, 256), if not, resize it to the size
                    if width != 256 or height != 256:
                        im = im.resize((256, 256), resample=Image.BICUBIC)

                    # compute the up-scaled size
                    width_scaled, height_scaled = int(256/self.scale), int(256/self.scale)

                    # resize the image 
                    shrinked_img = im.resize((width_scaled, height_scaled), resample=self.resample)

                    # randomly decide if this image should be put into test set, validation set or training set
                    rand_num = random.uniform(0,1)
                    if rand_num < self.valid_ratio:
                        which_set = 1
                    elif rand_num < self.valid_ratio+self.test_ratio:
                        which_set = 2
                    else:
                        which_set = 0
                    
                    # place the resized image and original image into one folder in the output directory
                    filename = os.path.basename(filename)             # base name for the image file
                    foldername = filename[:filename.find('.')]           # get rid of the file extension (.tif), and use it as the folder name

                    # decide whether training or test set this sample belongs to and save them to the designated folder
                    if which_set == 0: 
                        out_folder = os.path.join(self.out_dir, 'training', foldername)
                    elif which_set == 1: 
                        out_folder = os.path.join(self.out_dir, 'validation', foldername)
                    else:
                        out_folder = os.path.join(self.out_dir, 'test', foldername)

                    os.mkdir(out_folder)                                                 # make a folder in the training folder
                    im.save(os.path.join(out_folder, 'original.tif'))                    # save the original image as ground truth 
                    shrinked_img.save(os.path.join(out_folder, 'input.tif'), **im.info)  # save the scaled image as input 

                pbar.update()


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    builder = UCMercedBuilder(args.data_dir, args.output_dir, scale=int(args.scale))
    builder.run()
