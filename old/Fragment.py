import cv2, numpy as np
from tinydb import TinyDB, Query

class Fragment:
    
    def __init__(self):
        self.id = None
        self.name = None
        self.IRR_file = None
        self.IRV_file = None
        self.COLR_file = None
        self.COLV_file = None

    def loadFromTinyDB(self, dict):
        self.__dict__ = dict
        
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

    def saveToTinyDB(self, db):
        if(self.id is None):
            self.id = len(db)
            print self.id
        q = Query()
        db.upsert(self.__dict__, q.id == self.id)

    def toString(self):
        return "id : {0}, name : {1}, IRR file : {2}, IRV file : {3}, COLR file : {4}, COLV file : {5}".format(self.id, self.name, self.IRR_file, self.IRV_file, self.COLR_file, self.COLV_file)
