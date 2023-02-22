import cv2, cv2.aruco as aruco, numpy as np
import argparse, os, re
from utils.fragment import Fragment
import utils.segmentation as segmentation
argParser = argparse.ArgumentParser(description='process a full directory of images, extract their fragment shapes and align them')

def typeDir(str):
    if(not os.path.isdir(str)):
        raise argparse.ArgumentTypeError("{0} is not a directory.".format(str))
    return str
argParser.add_argument('file', type=str)
args = argParser.parse_args()

im = cv2.imread(args.file)
ss = segmentation.detectArucoSetSquare(im)
if(ss is not None):
    ar_mask = segmentation.createArucoSetSquareMask(ss, im.shape[:2], 1.)

def newWindow(name, width, height):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)

newWindow("mask", 600,400)
cv2.imshow("mask", ar_mask)
cv2.waitKey(0)

