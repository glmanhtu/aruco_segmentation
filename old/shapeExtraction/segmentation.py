import cv2, numpy as np
from steerableFilterALCM import steerableFilterALCM

def reconnectContours(contours, a, b):
    b = a/2
    m1 = steerableFilterALCM(contours, a, b, 0)
    ret, t1 = cv2.threshold(m1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    m2 = steerableFilterALCM(contours, a, b, 45)
    ret, t2 = cv2.threshold(m2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    m3 = steerableFilterALCM(contours, a, b, -45)
    ret, t3 = cv2.threshold(m3, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    m4 = steerableFilterALCM(contours, a, b, 90)
    ret, t4 = cv2.threshold(m4, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # cv2.imshow("ALCM1", t1)
    # cv2.imshow("ALCM2", t2)
    # cv2.imshow("ALCM3", t3)
    # cv2.imshow("ALCM4", t4)
    # cv2.imwrite("ALCM1.png", t1)
    # cv2.imwrite("ALCM2.png", t2)
    # cv2.imwrite("ALCM3.png", t3)
    # cv2.imwrite("ALCM4.png", t4)
    return cv2.add(t1, cv2.add(t2, cv2.add(t3, t4)))

def thresholdSegmentation(im, blurSize, ellipseSize, debug=False, threshSize=25, threshOffset=2):
    blur = im.copy()
    blur = cv2.medianBlur(blur, blurSize)
    blur = cv2.GaussianBlur(blur, (blurSize, blurSize), 0, 0)
    
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,threshSize,threshOffset)

    #contour reconstruction
    a = ellipseSize
    b = a/2
    ALCM = reconnectContours(thresh, a, b)
    
    ALCM_bak = ALCM.copy()
    _, contours, _ = cv2.findContours(ALCM, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(im.shape, im.dtype)
    frag = [max(contours, key = cv2.contourArea)]
    cv2.drawContours(mask, frag, -1, 255, -1)

    # if(debug):
    #     cv2.imshow("blur", blur)
    #     cv2.imshow("threshold", thresh)
    #     cv2.imshow("ALCM", ALCM_bak)
        # cv2.imwrite("input.png", im)
        # cv2.imwrite("blur.png", blur)
        # cv2.imwrite("thresh.png", thresh)
        # cv2.imwrite("ALCM.png", ALCM_bak)
        # cv2.imwrite("output.png", mask)

    return mask, blur, thresh, ALCM_bak

def loadSegmentationMask(filename):
    ret = cv2.imread(filename)
    ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
    _, ret = cv2.threshold(ret, 0, 255, cv2.THRESH_BINARY)
    return ret

def segmentationDiff(ref, seg):
    h, w = ref.shape[:2]
    res = np.zeros((h, w, 3), np.uint8)

    res[:,:,0] = cv2.bitwise_and(cv2.bitwise_not(ref), seg)
    res[:,:,1] = cv2.bitwise_and(ref, seg)
    res[:,:,2] = cv2.bitwise_and(ref, cv2.bitwise_not(seg))

    refCount = cv2.countNonZero(ref)
    segCount = cv2.countNonZero(seg)
    surplus = cv2.countNonZero(res[:,:,0])
    common = cv2.countNonZero(res[:,:,1])
    missing = cv2.countNonZero(res[:,:,2])

    assert refCount == common+missing
    assert segCount == common+surplus

    return (res, refCount, segCount, common, surplus, missing)
