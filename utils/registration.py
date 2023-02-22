import cv2, numpy as np, sys, math
from pyswarm import pso

from utils.segmentation import loadSegmentationMask

def registrationError(im1, im2):
    h, w = im1.shape[:2]
    res = np.zeros((h, w, 3), np.uint8)

    res[:,:,0] = cv2.bitwise_and(cv2.bitwise_not(im1), im2)
    res[:,:,1] = cv2.bitwise_and(im1, im2)
    res[:,:,2] = cv2.bitwise_and(im1, cv2.bitwise_not(im2))

    im1Count = cv2.countNonZero(im1)
    im2Count = cv2.countNonZero(im2)
    surplus = cv2.countNonZero(res[:,:,0])
    common = cv2.countNonZero(res[:,:,1])
    missing = cv2.countNonZero(res[:,:,2])

    assert im1Count == common+missing
    assert im2Count == common+surplus

    if(common == 0):
        common += 1
    error = (surplus+missing)/float(common)
    
    return (res, error)

def dist(x, y):
    return math.sqrt((x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1]))

def registrationError2(cnt1, cnt2):
    sumd = 0
    for p in cnt1:
        mind = dist(p[0], cnt2[0][0])
        for q in cnt2:
            d = dist(p[0], q[0])
            if(d < mind):
                mind = d
        sumd += mind
    sumd /= len(cnt1)

    return sumd

#src mask
#apply mask to src
#

def extractFragment(fragment, mask):
    m=mask
    if(len(fragment.shape) == 3 and fragment.shape[2] == 3):
        m = np.zeros(fragment.shape, fragment.dtype)
        m[:,:,0] = mask
        m[:,:,1] = mask
        m[:,:,2] = mask
    return cv2.subtract(fragment, cv2.bitwise_not(m))

def cropFragment(mask, fragment=None):
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(mask))
    if(fragment is not None):
        return mask[y:(y+h), x:(x+w)], fragment[y:(y+h), x:(x+w), :]
    else:
        return mask[y:(y+h), x:(x+w)], None    

def loadRectoVerso(rectoFile, rectoMaskFile, versoFile, versoMaskFile, scalingSize=(0,0), scalingRatio=1, addTransformationPadding=False, applyMask=True, flipVerso=True):
    #load recto/verso image and it's respective mask, extract the fragment from the image using the mask, croping the both fragment and mask using a bounding rect
    rectoMask = loadSegmentationMask(rectoMaskFile)

    if(rectoFile is not None):
        recto = cv2.imread(rectoFile)
        if(applyMask is True):
            recto = extractFragment(recto, rectoMask)
    else:
        recto = None
    rectoMask, recto = cropFragment(rectoMask, recto)

    versoMask = loadSegmentationMask(versoMaskFile)
    if(versoFile is not None):
        verso = cv2.imread(versoFile)
        if(applyMask is True):
            verso = extractFragment(verso, versoMask)
    else:
        verso = None
    versoMask, verso = cropFragment(versoMask, verso)


    rectoMask = cv2.resize(rectoMask, scalingSize, fx = scalingRatio, fy = scalingRatio, interpolation=cv2.INTER_NEAREST)
    rectoH, rectoW = rectoMask.shape[:2]

    versoMask = cv2.resize(versoMask, scalingSize, fx = scalingRatio, fy = scalingRatio, interpolation=cv2.INTER_NEAREST)
    versoH, versoW = versoMask.shape[:2]

    if(addTransformationPadding is True):
        diagVerso = math.sqrt(versoH*versoH+versoW*versoW)
        H = int(max(rectoH, diagVerso))
        W = int(max(rectoW, diagVerso))
    else:
        H = max(rectoH, versoH)
        W = max(rectoW, versoW)

    top = int((H-rectoH)/2)
    bottom = H-rectoH-top
    left = int((W-rectoW)/2)
    right = W-rectoW-left
    if(recto is not None):
        recto = cv2.resize(recto, scalingSize, fx = scalingRatio, fy = scalingRatio, interpolation=cv2.INTER_NEAREST)
        recto = cv2.copyMakeBorder(recto, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
    rectoMask = cv2.copyMakeBorder(rectoMask, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)

    top = int((H-versoH)/2)
    bottom = H-versoH-top
    left = int((W-versoW)/2)
    right = W-versoW-left
    if(verso is not None):
        verso = cv2.resize(verso, scalingSize, fx = scalingRatio, fy = scalingRatio, interpolation=cv2.INTER_NEAREST)
        verso = cv2.copyMakeBorder(verso, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
    versoMask = cv2.copyMakeBorder(versoMask, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)

    if(flipVerso is True):
        if(verso is not None):
            verso = cv2.flip(verso, 0)
        versoMask = cv2.flip(versoMask, 0)
    
    return recto, rectoMask, verso, versoMask    

def transformImage(im, scaling=1, theta=0, translation=(0,0)):
    """
    apply a transformation to an image.
    scaling and rotation is applied first (rotation is done around the image's center)
    then translation
    """
    h, w = im.shape[:2]
    thetaRad = theta * math.pi/180.
    cosTheta = math.cos(thetaRad)
    sinTheta = math.sin(thetaRad)
    rotMat = cv2.getRotationMatrix2D((w/2, h/2), theta, scaling)
    rotMat[0,2] += translation[0]
    rotMat[1,2] += translation[1]
    rotation = cv2.warpAffine(im,
                              rotMat,
                              (int(w), int(h)))
    _, res = cv2.threshold(rotation, 0,255, cv2.THRESH_BINARY)
    return res

def transformContour(cnt, h, w, scaling=1, theta=0, translation=(0,0)):
    """
    apply a transformation to a contour
    scaling and rotation is applied first (rotation is done around the contour's image's center)
    then translation
    """
    thetaRad = theta * math.pi/180.
    cosTheta = math.cos(thetaRad)
    sinTheta = math.sin(thetaRad)
    rotMat = cv2.getRotationMatrix2D((w/2, h/2), theta, scaling)
    rotMat[0,2] += translation[0]
    rotMat[1,2] += translation[1]
    res = cv2.transform(cnt, rotMat)
    return res

def PSOregistration(recto, verso, versoScaling=1, swarmsize=500, omega=0.5, maxiter=30, phip=0.5, phig=0.5, debug=False):

    def fitness(x):
        tverso = transformImage(verso, versoScaling, x[0], (x[1], x[2]))
        _, error = registrationError(recto, tverso)
        return error

    rh, rw = recto.shape[:2]
    lowerBounds = [-360,-rw/4,-rh/4]
    upperBounds = [360,rw/4,rh/4]
    print(swarmsize)
    xopt, fopt = pso(fitness, lowerBounds, upperBounds, swarmsize=swarmsize, omega=omega, maxiter=maxiter, phip=phip, phig=phig, debug=debug)
    
    return versoScaling, xopt[0], (xopt[1], xopt[2])


def visualizeRegistration(recto, verso, scaling, rotation, translation, translationScaling=1):
    t = (translation[0]*translationScaling, translation[1]*translationScaling)
    tverso = transformImage(verso, scaling, rotation, t)

    res, _ = registrationError(recto, tverso)
    return res


