import numpy as np
from matplotlib import pyplot as plt

# ---------- This part is for the use functions for TOC curve
def AreaUnderCurve(listx, listy):
    listx = np.array(listx)
    listy = np.array(listy)
    Llistx = np.delete(listx, listx.shape[1] - 1)
    Hlistx = np.delete(listx, 0)
    Llisty = np.delete(listy, listy.shape[1] - 1)
    Hlisty = np.delete(listy, 0)
    Areas = np.sum(np.int64([(Hlistx - Llistx) * (Hlisty + Llisty)]), axis=1) / 2.0
    return Areas

# ---------- This part is the class for TOC curve ----------
class TOC:
    def __init__(self, booleanArray=None, indexArray=None, thresholds=None, maskArray=None, weightsArray=None):
        self.booleanArray = booleanArray
        self.indexArray = indexArray
        self.thresholds = np.array(thresholds).flatten()
        if (maskArray is None):
            self.maskArray = np.ones_like(self.booleanArray)
        else:
            self.maskArray = maskArray
        if (weightsArray is None):
            self.weightsArray = np.ones_like(self.booleanArray)
        else:
            self.weightsArray = weightsArray
        # TOCX and TOCY
        self.TOCX = np.array([[]])
        self.TOCY = np.array([[]])
        self.AUC = 0
        # calculate the correctCornerY and totalNum
        if(booleanArray is not None):
            self.presenceInY = np.sum(self.booleanArray * self.weightsArray)
            self.totalNum = np.sum(self.weightsArray)
        else:
            self.presenceInY = 0
            self.totalNum = 0
        self.thresholdLabel = np.array([])

    def calculate_TOC(self):
        # There is no origin point when the minimum threshold is the same as index minimum
        self.TOCX = []
        self.TOCY = []
        for threshold in self.thresholds:
            # tell whether the threshold is ascending or descending
            if (self.thresholds[-1] < self.thresholds[0]):
                BoolIndexArray = (self.indexArray>= threshold).astype(
                    int)  # set value 1 for pixel which is higher than threshold, other pixels are assigned 0
            else:
                BoolIndexArray = (self.indexArray <= threshold).astype(
                    int)  # set value 1 for pixel which is higher than threshold, other pixels are assigned 0
            self.TOCX.append((BoolIndexArray * self.maskArray).sum())  # Hits and False Alarms
            self.TOCY.append((BoolIndexArray * self.maskArray * self.booleanArray).sum())  # Hits

        # add original zero
        if(self.TOCX[0]!=0):
            self.TOCX = [0]+self.TOCX
            self.TOCY = [0]+self.TOCY
            self.thresholdLabel = np.append(np.array(['origin']), self.thresholds)
        else:
            self.thresholdLabel = np.array(self.thresholds, dtype=np.str_)


        # when customize the threshold, the upper bond is not achieved, so we need to add it to the threshold labels
        if (self.TOCX[-1] != self.totalNum):
            self.TOCX.append(self.totalNum)
            self.TOCY.append(self.presenceInY)
            self.thresholdLabel = np.append(self.thresholdLabel, np.array(['end']))
        self.TOCY = np.array([self.TOCY])
        self.TOCX = np.array([self.TOCX])

        # self.correctCornerY = correctCorner(self.TOCX, self.TOCY, self.presenceInY)
    def format_coordinates(self):
        self.TOCX = np.array(self.TOCX).flatten()
        self.TOCX = self.TOCX.reshape((1, self.TOCX.shape[0]))
        self.TOCY = np.array(self.TOCY).flatten()
        self.TOCY = self.TOCY.reshape((1, self.TOCY.shape[0]))

    def calculate_AUC(self):
        self.format_coordinates()
        presenceInY = self.TOCY[0,-1]
        totalNum = self.TOCX[0, -1]
        Area01 = AreaUnderCurve(self.TOCX,self.TOCY)
        Area02 = AreaUnderCurve(np.array([[0, presenceInY]]),np.array([[0, presenceInY]]))
        Area03 = AreaUnderCurve(np.array([[0, totalNum]]),np.array([[0, presenceInY]]))*2
        AUC = (Area01-Area02)/(Area03-2*Area02)
        self.AUC = float(AUC)
