import numpy as np, cv2, sys, argparse
import threading, time
from segmentation import thresholdSegmentation
from segmentation import segmentationDiff
print(cv2.__version__)

def newWindow(name, width, height):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)


def process():
    global mask
    global blur
    global thresh
    global ALCM
    global diff

    mask, blur, thresh, ALCM = thresholdSegmentation(gray, blurSize, ellipseSize, args.steps, thresholdSize, thresholdOffset)
    if(args.groundTruth):
        diff, _, _, _, _, _ = segmentationDiff(groundTruth, mask)
        
def displayDst():
    if(args.groundTruth):
        cv2.imshow("dst", cv2.addWeighted(diff, maskOpacity, im, 1-maskOpacity, 0))
    else:
        height, width = mask.shape[:2]
        maskRed = np.zeros((height, width, 3), mask.dtype)
        maskRed[:,:,2] = mask.copy()
        cv2.imshow("dst", cv2.addWeighted(maskRed, maskOpacity, im, 1-maskOpacity, 0))

    cv2.imshow("src", gray)

    
newWindow("mask", 800, 600)
newWindow("dst", 800, 600)
newWindow("src", 800,600)

# vars
blurSize = 15
ellipseSize = 60
maskOpacity = 0.5

argParser = argparse.ArgumentParser(description='thresholding segmentation approach')
argParser.add_argument('filename', type=str)
argParser.add_argument('--steps', action='store_true')
argParser.add_argument('-b', '--blurSize', type=int, default=blurSize)
argParser.add_argument('-e', '--ellipseSize', type=int, default=ellipseSize)
argParser.add_argument('--groundTruth', type=str)

args = argParser.parse_args()

filename = args.filename
print "Opening file :", filename

im = cv2.imread(filename)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

mask = np.zeros(gray.shape, gray.dtype)
diff = im.copy()
if(args.steps):
    blur = im.copy()
    thresh = im.copy()
    ALCM = im.copy()
#diff = np.zeros(gray.shape, gray.dtype)
dst = np.zeros(gray.shape, gray.dtype)

if(args.groundTruth):
    groundTruth = cv2.cvtColor(cv2.imread(args.groundTruth), cv2.COLOR_BGR2GRAY)

overlay = np.zeros(gray.shape, gray.dtype)
h, w = gray.shape[:2]
cv2.rectangle(overlay, (45, 5), (570, 130), (0,0,0), -1)
cv2.putText(overlay, "Processing", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255,255,255), 3)

if(args.steps):
    newWindow("blur", 800, 600)
    newWindow("threshold", 800, 600)
    newWindow("ALCM", 800, 600)
    # newWindow("ALCM1", 800,600)
    # newWindow("ALCM2", 800,600)
    # newWindow("ALCM3", 800,600)
    # newWindow("ALCM4", 800,600)
    # newWindow("ALCMTest", 800, 600)

blurSize = args.blurSize
thresholdSize = 25
thresholdOffset = 2
ellipseSize = args.ellipseSize
    
def blurSizeChanged(val):
    global blurSize
    blurSize = val
    if(~blurSize%2):
        blurSize +=1
    global changed
    changed = True

def thresholdSizeChanged(val):
    global thresholdSize
    thresholdSize = val
    if(~thresholdSize%2):
        thresholdSize += 1
    global changed
    changed = True

def thresholdOffsetChanged(val):
    global thresholdOffset
    thresholdOffset = val
    global changed
    changed = True
    
def ellipseSizeChanged(val):
    global ellipseSize
    ellipseSize = val
    global changed
    changed = True
    
def opacityChanged(val):
    global maskOpacity
    maskOpacity = val/100.


cv2.createTrackbar("blur size", "mask", blurSize, 150, blurSizeChanged)
cv2.createTrackbar("threshold size", "mask", thresholdSize, 150, thresholdSizeChanged)
cv2.createTrackbar("threshold offset", "mask", thresholdOffset, 200, thresholdOffsetChanged)

cv2.createTrackbar("ellipse size", "mask", ellipseSize, 200, ellipseSizeChanged)

cv2.createTrackbar("opacity", "dst", int(maskOpacity*100), 100, opacityChanged)

changed = True
processing = False

def backgroundProcess():
    global changed
    global processing
    while(True):
        if(changed):
            changed = False
            processing = True
            process()
            processing = False
        time.sleep(0.05)
        

processThread = threading.Thread(name='process', target=backgroundProcess)
processThread.setDaemon(True)
processThread.start()

#process()
while(cv2.waitKey(50) == -1):
    if(processing):
        cv2.imshow("mask", cv2.addWeighted(mask, 0.8, overlay, 1, 0))
    else:
        cv2.imshow("mask", mask)

    displayDst()
    if(args.steps):
        cv2.imshow("blur", blur)
        cv2.imshow("threshold", thresh)
        cv2.imshow("ALCM", ALCM)

    
