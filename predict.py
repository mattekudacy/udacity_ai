import numpy as np
import argparse
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import json
from PIL import Image
import futility
import fmodel

parser = argparse.ArgumentParser(description='Parser for predict.py')
parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type=str)
parser.add_argument('--dir', action="store", dest="data_dir", default="./flowers/")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type=str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = parser.parse_args()

def main():
    model = fmodel.load_checkpoint(args.checkpoint)
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    probabilities = fmodel.predict(args.input, model, args.top_k, args.gpu)
    probability = np.array(probabilities[0][0])
    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    
    for i in range(args.top_k):
        print("{} with a probability of {}".format(labels[i], probability[i]))
    print("Finished Predicting!")

if __name__ == "__main__":
    main()
