import cv2, numpy as np, argparse, os
from segmentation import *

argParser = argparse.ArgumentParser(description='apply shape extraction algorithm to all images in input directory and save the binary masks in the outpur directory. The parameters are the blurSize and ellipseSize.')
argparser = argparse.add_argument('-i', type=isDirectory, required=True)
argparser = argparse.add_argument('-m', type=isDirectory, required=True)
argparser = argparse.add_argument('-o', type=isDirectory, required=True)
argparser = argparse.add_argument('-e', type=int, required=True)
argparser = argparse.add_argument('-b', type=int, required=True)

def isDirectory(str):
    if(os.path.isdir(str)):
        return str
    raise argparse.ArgumentTypeError("{0} is not a directory.".format(str))

args = argParser.parse_args()

inDir = args.i
maskDir = args.m
outDir = args.o

blurSize = args.b
ellipseSize = args.e

def newWindow(name, width, height):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)


def extractShape(im_filename, mask_filename, out_filename):
    im = cv2.imread(im_filename)
    gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    mask = gray.copy()
    mask, _, _, _ = thresholdSegmentation(gray, blurSize, ellipseSize)
    cv2.imwrite("{0}.png".format(mask_filename), mask)

    #inverse the mask
    mask = cv2.bitwise_not(mask)
    if(len(im.shape) >2 and im.shape[2] == 3):
        mask = np.stack((mask,)*3, -1)

    fragment = cv2.subtract(im, mask)
    cv2.imwrite("{0}.png".format(out_filename), fragment)

print "extracting fragment shapes from {0} directory to {1}.".format(inDir, outDir)

for file in os.listdir(args.inDir):
    print(file)
    if(os.path.isfile("{0}{1}".format(args.inDir,file))):
        f, ext = os.path.splitext(file)
        testImage("{0}{1}".format(args.dir, file),
                  "{0}{1}".format(outDir, f))


        
        
