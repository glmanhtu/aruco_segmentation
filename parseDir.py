#!/usr/bin/python3
import cv2, cv2.aruco as aruco, numpy as np
import argparse, os, re, shutil
from tinydb import TinyDB, Query
import data.dataconfig as dataconfig
from utils.fragment import Fragment
import utils.segmentation as segmentation
import utils.registration as registration

def typeDir(str):
    if(not os.path.isdir(str)):
        raise argparse.ArgumentTypeError("{0} is not a directory.".format(str))
    return str

argParser = argparse.ArgumentParser(description='process a full directory of images, extract their fragment shapes and align them')

argParser.add_argument('-i', '--inputDir', type=typeDir, required=True)
argParser.add_argument('-o', '--outputDir', type=typeDir, required=True)
argParser.add_argument('--object', type=str, required=True)
args = argParser.parse_args()

#shape extraction parameters
#variables globales en dur pour le moment à améliorer
shape_arucoMarkers = [1,0,2]#identifiants des marqueurs qui constituent l'équerre
shape_arucoDict = aruco.DICT_4X4_50#dictionnaire aruco utilisé
shape_ellipseSize = 60#taille de l'ellipse pour la segmentation
shape_blurSize = 11#taille du flou pour la segmentation
shape_object = args.object

RVregistration_scalingRatio = 0.1

print("input dir : " + args.inputDir + " output dir : " + args.outputDir)
with open("./log_processDirectory.txt", 'w') as logfile:
    logfile.write("processDirectory    inputDir = {}  outputDir = {}\n".format(args.inputDir, args.outputDir))
    logfile.write("creating subdirectories in outputDir\n")
    os.makedirs(args.outputDir+dataconfig.FRAGMENT_DIRECTORY, exist_ok=True)
    os.makedirs(args.outputDir+dataconfig.RESULT_VISUALIZATION, exist_ok=True)

    #regex : a number (at least one digit) + eventually a letter + R or V + .JPG or -COL.JPG
    #regex_filename = re.compile(r'([0-9]+[a-zA-Z]?)(R|V)(-COL)?\.JPG')
    #regex_filename = re.compile(r'([0-9]+(_[a-zA-Z])?(+)_(r|v)_(IR|CL)\.JPG')
    regex_filename = re.compile(r'([a-zA-Z0-9_+]+)_(r|v)_(IR|CL)\.JPG')

    logfile.write("Parsing files in inputDir :\n")
    fragments = {}
    for file in os.listdir(args.inputDir):
        print(file)
        logfile.write("    "+file+"\n")
        #if it is a file
        if(os.path.isfile(args.inputDir+file)):
            m = re.fullmatch(regex_filename, file)
            if(m):
                name = m.group(1)
                recto = m.group(2) == "r"
                color = m.group(3) == "CL"
                #print("\tname : ", m.group(1), "\n\trecto : ", m.group(2), "\n\tcolor : ", m.group(3))
                print("\tname : ", name, "\n\trecto : ", recto, "\n\tcolor : ", color) 

            else:
                logfile.write("ERROR : file {} does not match the file pattern.\n".format(file))
                print("ERROR : unrecognized file name " + file)
