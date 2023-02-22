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
    #regex_filename = re.compile(r'([0-9]+(_[a-zA-Z])?)_(r|v)_(IR|CL)\.JPG')
    regex_filename = re.compile(r'([a-zA-Z0-9_+]+)_(r|v)_(IR|CL)\.(jpg|JPG|PNG|png)')

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

                if(name not in fragments):
                    logfile.write("    new fragment {}\n".format(name))
                    print("new fragment : "+name)
                    f = Fragment()
                    f.name = name
                    fragments[name] = f
                    f.fragDir = dataconfig.FRAGMENT_DIRECTORY+name+"/"
                    os.makedirs(args.outputDir+dataconfig.FRAGMENT_DIRECTORY+name, exist_ok=True)
                f = fragments[name]
                if(recto):
                    if(color):
                        if(f.COLR_file is None):
                            f.COLR_file = file
                        else:
                            logfile.write("    ERROR : fragment {} duplicate COLR (old : {}  current : {})\n".format(f.name, f.COLR_file, file))
                    else:
                        if(f.IRR_file is None):
                            f.IRR_file = file
                        else:
                            logfile.write("    ERROR : fragment {} duplicate IRR (old : {}  current : {})\n".format(f.name, f.IRR_file, file))
                else:
                    if(color):
                        if(f.COLV_file is None):
                            f.COLV_file = file
                        else:
                            logfile.write("    ERROR : fragment {} duplicate COLV (old : {}  current : {})\n".format(f.name, f.COLV_file, file))
                    else:
                        if(f.IRV_file is None):
                            f.IRV_file = file
                        else:
                            logfile.write("    ERROR : fragment {} duplicate IRV (old : {}  current : {})\n".format(f.name, f.IRV_file, file))
                        f.IRV_file = file
                shutil.copy(args.inputDir+file, args.outputDir+dataconfig.FRAGMENT_DIRECTORY+name+"/"+file)
            else:
                logfile.write("ERROR : file {} does not match the file pattern.\n".format(file))
                print("ERROR : unrecognized file name " + file)

    

    logfile.write("inputDir parsing is over\n")
    logfile.write("list of parsed fragments :\n")
    for k,f in fragments.items():
        logfile.write("    "+f.toString()+"\n")

    logfile.write("Extracting shapes :\n")
    cpt = 0
    for k,f in fragments.items():
        cpt += 1
        logfile.write("    fragment {}:\n".format(f.name))
        print("fragment {}/{}".format(cpt, len(fragments)))
        fragDir = args.outputDir+dataconfig.FRAGMENT_DIRECTORY+f.name+"/"

        try:
            ##### shape extraction #####
            if(f.IRR_file):
                logfile.write("        extracting IRR shape from : {}\n".format(f.IRR_file))
                im = cv2.imread(fragDir+f.IRR_file)
                object = cv2.cvtColor(cv2.imread(args.object), cv2.COLOR_BGR2GRAY)
                shapeExt, ppc = segmentation.extractShape(im, True, 15, shape_arucoMarkers, shape_arucoDict, object, shape_ellipseSize, shape_blurSize)
                f.IRR_pixelsPerCentimeter = ppc
                shapeVisu = segmentation.createMaskVisualization(im, shapeExt)
                cv2.imwrite(fragDir+dataconfig.FRAGMENT_SHAPE_MASK_IRR, shapeExt)
                f.IRR_shapeMask = dataconfig.FRAGMENT_SHAPE_MASK_IRR
                cv2.imwrite(args.outputDir+dataconfig.RESULT_VISUALIZATION+f.name+"IRR.png", shapeVisu)
                mask = np.zeros_like(im)
                mask[:,:,0] = shapeExt
                mask[:,:,1] = shapeExt
                mask[:,:,2] = shapeExt
                cv2.imwrite(args.outputDir+f.fragDir+f.name+"_IRR.png", cv2.subtract(im, cv2.bitwise_not(mask)))

            if(f.IRV_file):
                logfile.write("        extracting IRV shape from : {}\n".format(f.IRV_file))
                im = cv2.imread(fragDir+f.IRV_file)
                object = cv2.cvtColor(cv2.imread(args.object), cv2.COLOR_BGR2GRAY)
                shapeExt, ppc = segmentation.extractShape(im, True, 15, shape_arucoMarkers, shape_arucoDict, object, shape_ellipseSize, shape_blurSize)
                f.IRV_pixelsPerCentimeter = ppc
                shapeVisu = segmentation.createMaskVisualization(im, shapeExt)
                cv2.imwrite(fragDir+dataconfig.FRAGMENT_SHAPE_MASK_IRV, shapeExt)
                f.IRV_shapeMask = dataconfig.FRAGMENT_SHAPE_MASK_IRV
                cv2.imwrite(args.outputDir+dataconfig.RESULT_VISUALIZATION+f.name+"IRV.png", shapeVisu)
                mask = np.zeros_like(im)
                mask[:,:,0] = shapeExt
                mask[:,:,1] = shapeExt
                mask[:,:,2] = shapeExt
                cv2.imwrite(args.outputDir+f.fragDir+f.name+"_IRV.png", cv2.subtract(im, cv2.bitwise_not(mask)))

            if(f.COLR_file):
                logfile.write("        extracting COLR shape from : {}\n".format(f.COLR_file))
                im = cv2.imread(fragDir+f.COLR_file)
                object = cv2.cvtColor(cv2.imread(args.object), cv2.COLOR_BGR2GRAY)
                shapeExt, ppc = segmentation.extractShape(im, True, 15, shape_arucoMarkers, shape_arucoDict, object, shape_ellipseSize, shape_blurSize)
                f.COLR_pixelsPerCentimeter = ppc
                shapeVisu = segmentation.createMaskVisualization(im, shapeExt)
                cv2.imwrite(fragDir+dataconfig.FRAGMENT_SHAPE_MASK_COLR, shapeExt)
                f.COLR_shapeMask = dataconfig.FRAGMENT_SHAPE_MASK_COLR
                cv2.imwrite(args.outputDir+dataconfig.RESULT_VISUALIZATION+f.name+"COLR.png", shapeVisu)
                mask = np.zeros_like(im)
                mask[:,:,0] = shapeExt
                mask[:,:,1] = shapeExt
                mask[:,:,2] = shapeExt
                cv2.imwrite(args.outputDir+f.fragDir+f.name+"_COLR.png", cv2.subtract(im, cv2.bitwise_not(mask)))

            if(f.COLV_file):
                logfile.write("        extracting COLV shape from : {}\n".format(f.COLV_file))
                im = cv2.imread(fragDir+f.COLV_file)
                object = cv2.cvtColor(cv2.imread(args.object), cv2.COLOR_BGR2GRAY)
                shapeExt, ppc = segmentation.extractShape(im, True, 15, shape_arucoMarkers, shape_arucoDict, object, shape_ellipseSize, shape_blurSize)
                f.COLV_pixelsPerCentimeter = ppc
                shapeVisu = segmentation.createMaskVisualization(im, shapeExt)
                cv2.imwrite(fragDir+dataconfig.FRAGMENT_SHAPE_MASK_COLV, shapeExt)
                f.COLV_shapeMask = dataconfig.FRAGMENT_SHAPE_MASK_COLV
                cv2.imwrite(args.outputDir+dataconfig.RESULT_VISUALIZATION+f.name+"COLV.png", shapeVisu)
                mask = np.zeros_like(im)
                mask[:,:,0] = shapeExt
                mask[:,:,1] = shapeExt
                mask[:,:,2] = shapeExt
                cv2.imwrite(args.outputDir+f.fragDir+f.name+"_COLV.png", cv2.subtract(im, cv2.bitwise_not(mask)))
        except segmentation.MarkerNotFoundException:
            logfile.write(" Couldn't detect markers!!!!")
        
    logfile.write("Shape extraction is over.\n")
    logfile.write("list of extracted fragments :\n")
    for k,f in fragments.items():
        logfile.write("    "+f.toString()+"\n")

    #recto/verso registration
    cpt = 0
    for k,f in fragments.items():
        cpt += 1
        print("fragment {}/{}".format(cpt, len(fragments)))
        fragDir = args.outputDir+dataconfig.FRAGMENT_DIRECTORY+f.name+"/"
        if(f.IRR_file is not None and f.IRV_file is not None):
            recto, rectoMask, verso, versoMask = registration.loadRectoVerso(fragDir+f.IRR_file,
                                                                fragDir+dataconfig.FRAGMENT_SHAPE_MASK_IRR,
                                                                fragDir+f.IRV_file,
                                                                fragDir+dataconfig.FRAGMENT_SHAPE_MASK_IRV,
                                                                applyMask=True)
            _, rectoSubsample, _, versoSubsample = registration.loadRectoVerso(fragDir+f.IRR_file,
                                                                  fragDir+dataconfig.FRAGMENT_SHAPE_MASK_IRR,
                                                                  fragDir+f.IRV_file,
                                                                  fragDir+dataconfig.FRAGMENT_SHAPE_MASK_IRV,
                                                                  applyMask=True,
                                                                  scalingRatio=RVregistration_scalingRatio)            

            transformation = registration.PSOregistration(rectoSubsample, versoSubsample,
                                             f.IRV_pixelsPerCentimeter/f.IRR_pixelsPerCentimeter,
                                                          debug=False)
            f.IRRV_transformation = transformation
            result = registration.visualizeRegistration(rectoMask, versoMask, *transformation, translationScaling=1/RVregistration_scalingRatio)
            cv2.imwrite(args.outputDir+dataconfig.RESULT_VISUALIZATION+f.name+"IR_registration.png", result)

        if(f.COLR_file is not None and f.COLV_file is not None):
            recto, rectoMask, verso, versoMask = registration.loadRectoVerso(fragDir+f.COLR_file,
                                                                fragDir+dataconfig.FRAGMENT_SHAPE_MASK_COLR,
                                                                fragDir+f.COLV_file,
                                                                fragDir+dataconfig.FRAGMENT_SHAPE_MASK_COLV,
                                                                applyMask=True)
            _, rectoSubsample, _, versoSubsample = registration.loadRectoVerso(fragDir+f.COLR_file,
                                                                  fragDir+dataconfig.FRAGMENT_SHAPE_MASK_COLR,
                                                                  fragDir+f.COLV_file,
                                                                  fragDir+dataconfig.FRAGMENT_SHAPE_MASK_COLV,
                                                                  applyMask=True,
                                                                  scalingRatio=RVregistration_scalingRatio)            

            transformation = registration.PSOregistration(rectoSubsample, versoSubsample,
                                             f.COLV_pixelsPerCentimeter/f.COLR_pixelsPerCentimeter,
                                                          debug=False)
            f.COLRV_transformation = transformation

            result = registration.visualizeRegistration(rectoMask, versoMask, *transformation, translationScaling=1/RVregistration_scalingRatio)
            cv2.imwrite(args.outputDir+dataconfig.RESULT_VISUALIZATION+f.name+"COL_registration.png", result)
    #TODO ajouter les autres étapes du pipeline : alignement recto verso  (alignement IR/COLOR ?)

    db = TinyDB(args.outputDir+"frags.json")
    db.purge_tables()
    for k, f in fragments.items():
        f.saveToTinyDB(db, processState=Fragment.PROCESS_STATE_DEFAULT)
