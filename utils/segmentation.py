import cv2, numpy as np
import cv2.aruco as aruco
import utils.steerableFilterALCM as ALCM
from operator import add
from math import sqrt


class MarkerNotFoundException(Exception):
    pass


def reconnectContours(contours, a, b):
    b = a / 2
    m1 = ALCM.steerableFilterALCM(contours, a, b, 0)
    ret, t1 = cv2.threshold(m1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m2 = ALCM.steerableFilterALCM(contours, a, b, 45)
    ret, t2 = cv2.threshold(m2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m3 = ALCM.steerableFilterALCM(contours, a, b, -45)
    ret, t3 = cv2.threshold(m3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m4 = ALCM.steerableFilterALCM(contours, a, b, 90)
    ret, t4 = cv2.threshold(m4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.add(t1, cv2.add(t2, cv2.add(t3, t4)))


def thresholdSegmentation(im, blurSize, ellipseSize, threshSize=25, threshOffset=2, mask=None):
    blur = im.copy()
    blur = cv2.medianBlur(blur, blurSize)
    blur = cv2.GaussianBlur(blur, (blurSize, blurSize), 0, 0)

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, threshSize,
                                   threshOffset)

    # masking
    if (mask is not None):
        thresh = cv2.subtract(thresh, cv2.bitwise_not(mask))

    # contour reconstruction
    a = ellipseSize
    b = a / 2
    ALCM = reconnectContours(thresh, a, b)

    ALCM_bak = ALCM.copy()
    contours, _ = cv2.findContours(ALCM, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    res = np.zeros(im.shape, im.dtype)
    frag = [max(contours, key=cv2.contourArea)]
    cv2.drawContours(res, frag, -1, 255, -1)

    return res, blur, thresh, ALCM_bak


def check_markers_are_found(ids, marker_ids):
    if ids is None:
        return False
    return sum([x in ids for x in marker_ids]) == len(marker_ids)


def detectArucoSetSquare(im, outerLength=15., markerIds=[1, 0, 2], dictionary=cv2.aruco.DICT_4X4_50):
    """
    detect the set square in the image im and returns its points.
    outerLength is the length in centimeter between the outer edges of the markers
    markersIds defines which markers from the dictionary are at which position. If the set square is in a L position we have markersIds = [lowerRight lowerLeft(corner) upperRight]
    +---+
    | 2 |
    +---+
    |   |
    |   |
    +---+----+---+
    | 1 |    | 0 |
    +---+----+---+
    dictionary is the id of the aruco dictionary used (default cv2.aruco.DICT_4X4_50)
    
    points are returned in the following order :
    9---10
    |   |
    8---11
    |   |
    |   |
    5---6----1---2
    |   |    |   |
    4---7----0---3
    """

    if len(markerIds) != 3:
        raise ValueError('markersIds must contain 3 ids', markerIds)

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
    parameters = aruco.DetectorParameters_create()

    # Adjust the corner refinement method
    # parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

    # detection of the markers in the images
    c, ids, rejected = aruco.detectMarkers(im, dictionary, parameters=parameters)

    # Add padding seems to improve performance of aruco.detectMarkers
    border_padding = 50
    im = cv2.copyMakeBorder(im.copy(), border_padding, border_padding, border_padding, border_padding,
                            cv2.BORDER_CONSTANT, value=[255, 255, 255])
    found_markers = check_markers_are_found(ids, markerIds)
    if not found_markers:
        for t in range(40, 250, 10):

            _, thresh = cv2.threshold(im, t, 255, cv2.THRESH_BINARY)
            cv2.imwrite("test.png", thresh)
            print("Trying to threshold the image ... (threshold : {})".format(t))
            c, ids, rejected = aruco.detectMarkers(thresh, dictionary, parameters=parameters)
            found_markers = check_markers_are_found(ids, markerIds)

            if not found_markers:
                print(f"Current marks: {ids}")
            else:
                break

        if not found_markers:
            raise MarkerNotFoundException()

    # flatten ids list ([[a][b]...] -> [a b ...])
    ids = list(ids.flatten())

    # convert markersIds to indices of these markers in the detection result array
    markerIds = [ids.index(x) for x in markerIds]
    pts = []
    for mId in markerIds:
        for p in c[mId][0]:
            pts.append(p - border_padding)

    Ax = (pts[2][0] + pts[3][0] - pts[4][0] - pts[5][0]) / (2 * outerLength)
    Ay = (pts[2][1] + pts[3][1] - pts[4][1] - pts[5][1]) / (2 * outerLength)
    Bx = (pts[9][0] + pts[10][0] - pts[4][0] - pts[7][0]) / (2 * outerLength)
    By = (pts[9][1] + pts[10][1] - pts[4][1] - pts[7][1]) / (2 * outerLength)

    return pts, (Ax, Ay), (Bx, By)


def createArucoSetSquareMask(setSquare, size, margin):
    """
    creates and returns an image mask of the set square :
        - setSquare : the tuple returned by detectArucoSetSquare
        - size : the size of the image mask to return
        - margin : the margin to add around the setSquare in centimeters (if set to 0, the mask will stop at the edges of the markers) 
    """

    pts, A, B = setSquare
    mask = np.zeros(size, np.uint8)
    poly = []
    for (p, ASign, BSign) in [(2, 1, 1),
                              (3, 1, -1),
                              (4, -1, -1),
                              (9, -1, 1),
                              (10, 1, 1),
                              (6, 1, 1)]:
        poly.append(list(map(add, list(map(add, pts[p], margin * ASign * np.array(A))), margin * BSign * np.array(B))))
    cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
    return mask


def createObjectMask(im, object, requiredMatches, mask=None):
    orb = cv2.ORB_create()

    # keypoints and descriptors computation
    kp1, dsc1 = orb.detectAndCompute(im, None)  # mask)
    kp2, dsc2 = orb.detectAndCompute(object, None)

    # print(dsc1)
    # print(dsc2)
    # matcher declaration
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # matching descriptors from the two images
    matches = bf.match(dsc1, dsc2)

    # sorting matches according to their "distance"
    sortMatches = sorted(matches, key=lambda x: x.distance)

    # find the homography between matches
    srcPts = np.float32([kp1[m.queryIdx].pt for m in sortMatches])
    dstPts = np.float32([kp2[m.trainIdx].pt for m in sortMatches])

    mat, goodPoints = cv2.findHomography(dstPts, srcPts, method=cv2.RANSAC)
    goodMatches = len([x for x in goodPoints if x == 1])
    # print "goodMatches ", goodMatches, " requiredMatches ", requiredMatches
    print(goodMatches < requiredMatches, goodMatches, requiredMatches)
    if (goodMatches < requiredMatches):
        return None

    # if the object is in the image, we create a white mask and warp it at the matching object's position in im, thus masking the object in im
    objMask = np.ones(object.shape, object.dtype) * 255
    res = cv2.warpPerspective(objMask, mat, tuple(reversed(im.shape[:2])))
    return res


def loadSegmentationMask(filename):
    ret = cv2.imread(filename)
    ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
    _, ret = cv2.threshold(ret, 0, 255, cv2.THRESH_BINARY)
    return ret


def segmentationDiff(ref, seg):
    h, w = ref.shape[:2]
    res = np.zeros((h, w, 3), np.uint8)

    res[:, :, 0] = cv2.bitwise_and(cv2.bitwise_not(ref), seg)
    res[:, :, 1] = cv2.bitwise_and(ref, seg)
    res[:, :, 2] = cv2.bitwise_and(ref, cv2.bitwise_not(seg))

    refCount = cv2.countNonZero(ref)
    segCount = cv2.countNonZero(seg)
    surplus = cv2.countNonZero(res[:, :, 0])
    common = cv2.countNonZero(res[:, :, 1])
    missing = cv2.countNonZero(res[:, :, 2])

    assert refCount == common + missing
    assert segCount == common + surplus

    return (res, refCount, segCount, common, surplus, missing)


def extractShape(im, setSquare=True, outerLength=15., markers=[1, 0, 2], arucoDict=aruco.DICT_4X4_50, object=None,
                 ellipseSize=11, blurSize=60):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # create a mask
    mask = np.ones(gray.shape, gray.dtype)

    # if we need to remove an aruco setSquare
    if (setSquare):
        ss = detectArucoSetSquare(gray, outerLength, markers, arucoDict)
        _, A, B = ss
        Anorm = sqrt(A[0] * A[0] + A[1] * A[1])
        Bnorm = sqrt(B[0] * B[0] + B[1] * B[1])
        print("Anorm : {}\nBnorm : {}".format(Anorm, Bnorm))
        if (ss is not None):
            aruco_mask = createArucoSetSquareMask(ss, gray.shape[:2], margin=1.)
            mask = cv2.subtract(mask, aruco_mask)

    # if we need to remove an object
    if (object is not None):
        # look for the object in the image
        obj_mask = createObjectMask(gray, object, 25, mask)
        if (obj_mask is not None):
            print(mask.shape, obj_mask.shape)
            mask = cv2.subtract(mask, obj_mask)

    result = im.copy()
    result, _, _, _ = thresholdSegmentation(gray, blurSize, ellipseSize, mask=mask)

    return result, (Anorm + Bnorm) / 2


def createMaskVisualization(im, mask, maskOpacity=0.2, imOpacity=0.8):
    if (len(im.shape) == 2 or im.shape[2] == 1):
        im = cv2.cvtColor(im, cv2.GRAY2BGR)
    mask_red = np.zeros(im.shape, im.dtype)
    mask_red[:, :, 2] = mask
    return cv2.addWeighted(im, imOpacity, mask_red, maskOpacity, 0)


def crop_image(image, pixel_value=0):
    # Remove the zeros padding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    crop_rows_gray = gray[~np.all(gray == pixel_value, axis=1), :]

    crop_rows = image[~np.all(gray == pixel_value, axis=1), :]
    cropped_image = crop_rows[:, ~np.all(crop_rows_gray == pixel_value, axis=0)]

    black_pixels = np.where(
        (cropped_image[:, :, 0] == 0) &
        (cropped_image[:, :, 1] == 0) &
        (cropped_image[:, :, 2] == 0)
    )

    # set those pixels to white
    cropped_image[black_pixels] = [255, 255, 255]

    return cropped_image
