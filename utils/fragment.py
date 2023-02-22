import cv2, numpy as np
from tinydb import TinyDB, Query
import utils.registration as registration
import os

class Fragment:

    PROCESS_STATE_VALID = 'valid'
    PROCESS_STATE_DEFAULT = 'unclassified'
    PROCESS_STATE_REJECTED = 'rejected'
    
    def __init__(self):
        self.id = None
        self.name = None
        self.IRR_file = None
        self.IRR_pixelsPerCentimeter = None
        self.IRR_shapeMask = False
        self.IRV_file = None
        self.IRV_pixelsPerCentimeter = None
        self.IRV_shapeMask = False
        self.COLR_file = None
        self.COLR_pixelsPerCentimeter = None
        self.COLR_shapeMask = False
        self.COLV_file = None
        self.COLV_pixelsPerCentimeter = None
        self.COLV_shapeMask = False

        self.IRRV_transformation = None
        self.COLRV_transformation = None        

        self.processState = None

        self.databaseDir = None
        self.fragDir = None
        self.resultDir = None

    def loadFromTinyDB(self, dict):
        self.__dict__ = dict
        
    def loadOriginal_IRR(self):
        if(self.fragDir is not None and self.IRR_file is not None):
            return cv2.imread(os.path.join(self.databaseDir,self.fragDir,self.IRR_file))
        else:
            return None
        
    def loadShapeMask_IRR(self):
        if(self.fragDir is not None and self.IRR_shapeMask is not None):
            return cv2.imread(os.path.join(self.databaseDir,self.fragDir,self.IRR_shapeMask))
        else:
            return None

    def loadOriginal_IRV(self):
        if(self.fragDir is not None and self.IRV_file is not None):
            return cv2.imread(self.databaseDir+self.fragDir+self.IRV_file)
        else:
            return None
                              
    def loadShapeMask_IRV(self):
        if(self.fragDir is not None and self.IRV_shapeMask is not None):
            return cv2.imread(self.databaseDir+self.fragDir+self.IRV_shapeMask)
        else:
            return None

    def loadOriginal_COLR(self):
        if(self.fragDir is not None and self.COLR_file is not None):
            return cv2.imread(self.databaseDir+self.fragDir+self.COLR_file)
        else:
            return None
                              
    def loadShapeMask_COLR(self):
        if(self.fragDir is not None and self.COLR_shapeMask is not None):
            return cv2.imread(self.databaseDir+self.fragDir+self.COLR_shapeMask)
        else:
            return None

    def loadOriginal_COLV(self):
        if(self.fragDir is not None and self.COLV_file is not None):
            return cv2.imread(self.databaseDir+self.fragDir+self.COLV_file)
        else:
            return None
                              
    def loadShapeMask_COLV(self):
        if(self.fragDir is not None and self.COLV_shapeMask is not None):
            return cv2.imread(self.databaseDir+self.fragDir+self.COLV_shapeMask)
        else:
            return None

    # def getFragment_IRR(self, crop=True):
    #     if(self.fragDir is not None and self.IRR_file is not None and self.IRR_shapeMask is True):
    #         fragment = registration.extractFragment(self.loadIROriginalR(), self.loadIRR_shapeMask()

    def saveToTinyDB(self, db, processState=None):
        q = Query()
        if(self.id is None):
            same_name = db.search(q.name == self.name)
            if(len(same_name) > 0):
                self.id = same_name[0]['id']
            self.id = len(db)
        if(processState is not None):
            self.processState = processState

        db.upsert(self.__dict__, q.id == self.id)

    def toString(self):
        return str(self.__dict__)#"id : {0}, name : {1}, IRR file : {2}, IRV file : {3}, COLR file : {4}, COLV file : {5}".format(self.id, self.name, self.IRR_file, self.IRV_file, self.COLR_file, self.COLV_file)
