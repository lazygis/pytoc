import numpy as np
from matplotlib import pyplot as plt
from matplotlib import transforms
from matplotlib import ticker as mticker
from matplotlib import colors as matpltColors

# ---------- This part is for the painting class
class TOC_paint:
    def __init__(self, TOC_list, index_names = None, boolUniform = False, boolCorrectCorner = False):
        # To make one TOC read as a list
        if type(TOC_list) != list:
            self.TOC_list = [TOC_list]
        else:
            self.TOC_list = TOC_list
        # if there is no name updated
        if index_names is None:
            self.index_names = ["TOC{0}".format(i+1) for i in range(len(self.TOC_list))]
        else:
            self.index_names = index_names
        first_TOC = self.TOC_list[0]
        self.totalNum = first_TOC.TOCX[0, -1]
        self.presenceInY = first_TOC.TOCY[0, -1]
        self.TOCNum = len(self.TOC_list)
        self.boolUniform = boolUniform
        self.boolCorrectCorner = boolCorrectCorner
        self.fig = plt.figure()

        cmap = matpltColors.ListedColormap('#e0e0e0')
        plt.tripcolor([self.totalNum - self.presenceInY, self.totalNum, self.totalNum], [0, 0, self.presenceInY], [0, 1, 2], np.array([1, 1, 1]), edgecolor="k", lw=0,
                      cmap=cmap)
        plt.tripcolor([0, 0, self.presenceInY], [0, self.presenceInY, self.presenceInY], [0, 1, 2], np.array([1, 1, 1]), edgecolor="k", lw=0,
                      cmap=cmap)

        # make the coordinates square
        plt.axis('square')
        # plt.axis('auto')
        plt.axis([0, self.totalNum, 0, self.presenceInY])
        plt.gca().set_aspect(1 / plt.gca().get_data_ratio())
    def paint(self):
        if (self.boolCorrectCorner):
            # CorrectCornerText1 = plt.gca().text(0, self.presenceInY * 1.01, 'The ', color="black", fontsize=8)
            # CorrectCornerText1.draw(plt.gca().figure.canvas.get_renderer())
            # ex = CorrectCornerText1.get_window_extent()
            # t = transforms.offset_copy(
            #     CorrectCornerText1.get_transform(), x=ex.width, units='dots')
            # CorrectCornerText2 = plt.gca().text(0, self.presenceInY * 1.01, '★ ', color="red", fontsize=8, transform=t)
            # CorrectCornerText2.draw(plt.gca().figure.canvas.get_renderer())
            # ex = CorrectCornerText2.get_window_extent()
            # t = transforms.offset_copy(
            #     CorrectCornerText2.get_transform(), x=ex.width, units='dots')
            # plt.gca().text(0, self.presenceInY * 1.01, 'marks where False Alarms equals Misses.', color="black", fontsize=8,
            #                transform=t)
            plt.text(0, self.presenceInY * 1.01, 'The red star marks where False Alarms equals Misses.', color="black", fontsize=8)
            #     marks where Misses equals False Alarms.
        if (self.boolUniform):
            l2 = plt.plot([0, self.totalNum], [0, self.presenceInY], ':', color="violet", label='Uniform')
        for i in range(self.TOCNum):
            self.__addOne(i, self.index_names[i])
        plt.show()
    ## This function is private to add cures to the painter
    def __addOne(self, index, Name, marker=None):
        Xlist = self.TOC_list[index].TOCX
        Ylist = self.TOC_list[index].TOCY
        if(marker==None):
            plt.plot(Xlist[0,:], Ylist[0,:], label=Name, picker=2, clip_on=False)
        else:
            plt.plot(Xlist[0,:], Ylist[0,:], marker+'-', label=Name, picker=2, clip_on=False)
        handles, labels = plt.gca().get_legend_handles_labels()
        numCurve = len(handles)
        # change the order of maximum and minimum
        # in handles the order will always uniform, x1, x2
        if(labels[0]=='Uniform'):
            order = list(range(1,numCurve))
            order.extend([0])
        else:
            order = list(range(0, numCurve))
        # put the maximum line first, uniform and minimum at last
        # order.insert(0,0)
        # order.extend([1,2])
        plt.gca().legend([handles[idx] for idx in order], [labels[idx] for idx in order],loc='center left', bbox_to_anchor=(1, 0.5))


def drawTOC(TOC_list):
    ## if the input is only one TOC object, not list.
    if type(TOC_list)!=list:
        TOC_list = [TOC_list]
    first_TOC = TOC_list[0]
    totalNum = first_TOC.TOCX[0,-1]
    presenceInY = first_TOC.TOCX[0,-1]
    TOCNum = len(TOC_list)
    fig, ax = plt.plot()


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
    def __init__(self, booleanArray, indexArray, thresholds, maskArray=None, weightsArray=None):
        self.booleanArray = np.array(booleanArray).flatten()
        self.indexArray = np.array(indexArray).flatten()
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
        self.calculate_AUC()

        # self.correctCornerY = correctCorner(self.TOCX, self.TOCY, self.presenceInY)
    def format_coordinates(self):
        self.TOCX = np.array(self.TOCX).flatten()
        self.TOCX = self.TOCX.reshape((1, self.TOCX.shape[0]))
        self.TOCY = np.array(self.TOCY).flatten()
        self.TOCY = self.TOCY.reshape((1, self.TOCY.shape[0]))

    def read_TOC_coor(self, TOCX, TOCY, thresholdLable=np.array([])):
        self.TOCX = TOCX
        self.TOCY = TOCY
        self.thresholdLabel = thresholdLable
        self.format_coordinates()
        self.presenceInY = self.TOCY[0, -1]
        self.totalNum = self.TOCX[0, -1]

    def calculate_AUC(self):
        self.format_coordinates()
        presenceInY = self.TOCY[0,-1]
        totalNum = self.TOCX[0, -1]
        Area01 = AreaUnderCurve(self.TOCX,self.TOCY)
        Area02 = AreaUnderCurve(np.array([[0, presenceInY]]),np.array([[0, presenceInY]]))
        Area03 = AreaUnderCurve(np.array([[0, totalNum]]),np.array([[0, presenceInY]]))*2
        AUC = (Area01-Area02)/(Area03-2*Area02)
        self.AUC = float(AUC)
