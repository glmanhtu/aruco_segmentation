#!/usr/bin/python3

import argparse
import cv2, numpy as np

argParser = argparse.ArgumentParser()
argparser.add_argument('name', type=str)

#chercher le fragment dans la BD

#si on le trouve afficher :
# - les 4 photos originales
# - les 4 segmentations
# - les alignements R/V
# - l'alignement IR/COL
