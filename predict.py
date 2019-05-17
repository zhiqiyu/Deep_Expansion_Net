'''Feed forward the model to get output'''

import argparse
import os
import glob
import math
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from model.x2net import x2Net
from model.x3net import x3Net
from model.x4net import x4Net

import utils


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/x2/test', help="Directory containing the test test dataset")
parser.add_argument('--model', default='x2net', help="The model to use.")
parser.add_argument('--model_dir', default='experiments/x2net/', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")

def postprocess(output):
    output[output < 0] = 0.
    output[output > 1] = 1.
    return output

if __name__ == "__main__":
    
    # Load the parameters
    args = parser.parse_args()

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    params.device = torch.device("cuda:2" if params.cuda else "cpu")

    # Define the model
    model_name = args.model
    if model_name == 'x2net':
        model = x2Net().to(params.device)
    elif model_name == 'x3net':
        model = x3Net().to(params.device)
    elif model_name == 'x4net':
        model = x4Net().to(params.device)
    else:
        print('not implemented')
        exit()

    if params.cuda:
        model = torch.nn.DataParallel(model, device_ids=[2,3])

    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # load images
    imgs = glob.glob(os.path.join(args.data_dir, '*/input.tif'))
    for img in imgs:
        img_file = Image.open(img).convert('YCbCr')

        # apply transform on the image
        input_img = transforms.ToTensor()(img_file).unsqueeze(0).to(params.device)

        # feed through the network
        out = model(input_img)
        output = out.cpu().detach()
        
        # delete the batch dimension
        output = torch.squeeze(output, 0)
        output = postprocess(output)

        out_img = transforms.ToPILImage(mode='YCbCr')(output).convert('RGB')

        # save image
        out_img.save(os.path.join(os.path.dirname(img), 'output.tif'))





        
    

    