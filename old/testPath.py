import cv2, numpy as np

class Fragment:

    nextId = 0
    
    def __init__(self):
        self.id = nextId

        self.name = None
        self.IRR_file = None
        self.IRV_file = None
        self.COLR_file = None
        self.COLV_file = None

    def loadIROriginalR(self):
        if(self.IRR_file is None):
            return None
        else:
            return cv2.imread(self.IRR_file)

    def loadIROriginalV(self):
        print "not implemented yet"

    def loadColorOriginalR(self):
        print "not implemented yet"

    def loadColorOriginalV(self):
        print "not implemented yet"
               
