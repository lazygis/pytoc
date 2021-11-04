import numpy as np
from matplotlib import pyplot as plt
# from matplotlib import transforms
# from matplotlib import ticker as mticker
from matplotlib import colors as matpltColors


# ---------- This part is for the painting class
class TOC_painter:
    def __init__(self, TOC_list=[], index_names=[], color_list=[], marker_list=[], line_list=[], boolUniform=False, boolCorrectCorner=False):
        # To make one TOC read as a list
        if type(TOC_list) != list:
            self.TOC_list = [TOC_list]
        else:
            self.TOC_list = TOC_list
        if type(color_list) != list:
            self.color_list = [color_list]
        else:
            self.color_list = color_list
        if type(marker_list) != list:
            self.marker_list = [marker_list]
        else:
            self.marker_list = marker_list
        if type(line_list) != list:
            self.line_list = [line_list]
        else:
            self.line_list = line_list
        # to make one index name read as a list
        if type(index_names) != list:
            self.index_names = [index_names]
        else:
            self.index_names = index_names
        # if there is no name updated
        self.totalNum = 0
        self.presenceInY = 0
        self.TOCNum=0
        self.curve_list = []
        if(len(self.TOC_list)>0):
            if len(self.index_names) == 0:
                self.index_names = ["TOC{0}".format(i + 1) for i in range(len(self.TOC_list))]
            if len(self.color_list) == 0:
                self.color_list = ["" for _ in range(len(self.TOC_list))]
            if len(self.marker_list) == 0:
                self.marker_list = ["" for _ in range(len(self.TOC_list))]
            if len(self.line_list) == 0:
                self.line_list = ["-" for _ in range(len(self.TOC_list))]
            for i in range(len(self.TOC_list)):
                item_dic = {'TOCX':self.TOC_list[i].TOCX,
                            'TOCY':self.TOC_list[i].TOCY,
                            'threshold':self.TOC_list[i].thresholdLabel,
                            'color':self.color_list[i],
                            'marker':self.marker_list[i],
                            'line':self.line_list[i],
                            'name':self.index_names[i]}
                self.curve_list.append(item_dic)
            self.__update_status()

        self.boolUniform = boolUniform
        self.boolCorrectCorner = boolCorrectCorner
        self.fig, self.ax = plt.subplots()

    def __update_status(self):
        self.TOCNum = len(self.curve_list)
        if(self.TOCNum>0):
            self.totalNum = self.curve_list[0]['TOCX'][0,-1]
            self.presenceInY = self.curve_list[0]['TOCY'][0,-1]


    ## This function is private to add cures to the painter
    def __addOne(self, index):
        Xlist = self.curve_list[index]['TOCX']
        Ylist = self.curve_list[index]['TOCY']
        color = self.curve_list[index]['color']
        marker = self.curve_list[index]['marker']
        line = self.curve_list[index]['line']
        Name = self.curve_list[index]['name']
        symbol = marker+line
        if (len(color)==0):
            plt.plot(Xlist[0, :], Ylist[0, :], symbol, label=Name, clip_on=False)
        else:
            plt.plot(Xlist[0, :], Ylist[0, :], symbol, color=color, label=Name, clip_on=False)
        handles, labels = plt.gca().get_legend_handles_labels()
        numCurve = len(handles)
        # change the order of maximum and minimum
        # in handles the order will always uniform, x1, x2
        if (labels[0] == 'Uniform'):
            order = list(range(1, numCurve))
            order.extend([0])
        else:
            order = list(range(0, numCurve))
        plt.gca().legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='center left',
                         bbox_to_anchor=(1, 0.5))
    def __format_coordinates(self, TOCX, TOCY):
        TOCX = np.array(TOCX).flatten()
        TOCX = TOCX.reshape((1, TOCX.shape[0]))
        TOCY = np.array(TOCY).flatten()
        TOCY = TOCY.reshape((1, TOCY.shape[0]))
        return TOCX, TOCY

    def add_correct_corner(self, index):
        TOCX = self.curve_list[index]['TOCX']
        TOCY = self.curve_list[index]['TOCY']
        presenceInY = TOCY[0,-1]
        bool_presence = (TOCX <= presenceInY).astype(int)
        corner_index = bool_presence.sum() - 1
        if (corner_index == TOCY.shape[1] - 1):
            y_res = TOCX[0, -1]
        else:
            y_res = (TOCY[0, corner_index + 1] - TOCY[0, corner_index]) * 1.0 / (
                        TOCX[0, corner_index + 1] - TOCX[0, corner_index]) * (presenceInY - TOCX[0, corner_index]) + \
                    TOCY[0, corner_index] * 1.0
        plt.plot(presenceInY, y_res, 'r*')
    def add_all_correct_corner(self):
        for i in range(self.TOCNum):
            self.add_correct_corner(i)


    def add_TOC(self, TOC, index_name=None, color='', marker='', line='-'):
        self.__update_status()
        if(index_name is None):
            index_list = [j['name'] for j in self.curve_list]
            for i in range(1,self.TOCNum+2):
                if('TOC'+str(i) not in index_list):
                    index_name = 'TOC'+str(i)
                    break
        item_dic = {'TOCX': TOC.TOCX,
                    'TOCY': TOC.TOCY,
                    'threshold': TOC.thresholdLabel,
                    'color': color,
                    'marker': marker,
                    'line': line,
                    'name': index_name}
        self.curve_list.append(item_dic)
        self.__update_status()



    def add_TOC_coor(self, TOCX, TOCY, threshold=[], index_name=None, color='', marker='', line='-'):
        self.__update_status()
        if (index_name is None):
            index_list = [j['name'] for j in self.curve_list]
            for i in range(1, self.TOCNum + 2):
                if ('TOC' + str(i) not in index_list):
                    index_name = 'TOC' + str(i)
                    break
        TOCX, TOCY = self.__format_coordinates(TOCX, TOCY)
        item_dic = {'TOCX': TOCX,
                    'TOCY': TOCY,
                    'threshold': threshold,
                    'color': color,
                    'marker': marker,
                    'line': line,
                    'name': index_name}
        self.curve_list.append(item_dic)
        self.__update_status()


    def paint(self):
        ## draw grey areas
        self.__update_status()
        cmap = matpltColors.ListedColormap('#e0e0e0')
        plt.tripcolor([self.totalNum - self.presenceInY, self.totalNum, self.totalNum], [0, 0, self.presenceInY],
                      [0, 1, 2], np.array([1, 1, 1]), edgecolor="k", lw=0,
                      cmap=cmap)
        plt.tripcolor([0, 0, self.presenceInY], [0, self.presenceInY, self.presenceInY], [0, 1, 2], np.array([1, 1, 1]),
                      edgecolor="k", lw=0,
                      cmap=cmap)

        # make the coordinates square
        plt.axis('square')
        # plt.axis('auto')
        plt.axis([0, self.totalNum, 0, self.presenceInY])
        plt.gca().set_aspect(1 / plt.gca().get_data_ratio())
        if (self.boolCorrectCorner):
            plt.text(0, self.presenceInY * 1.01, 'The red star marks where False Alarms equals Misses.', color="black",
                     fontsize=8)
        if (self.boolUniform):
            l2 = plt.plot([0, self.totalNum], [0, self.presenceInY], ':', color="violet", label='Uniform')
        ## draw curves
        for i in range(self.TOCNum):
            self.__addOne(i)
        plt.show()


# def drawTOC(TOC_list):
#     ## if the input is only one TOC object, not list.
#     if type(TOC_list) != list:
#         TOC_list = [TOC_list]
#     first_TOC = TOC_list[0]
#     totalNum = first_TOC.TOCX[0, -1]
#     presenceInY = first_TOC.TOCX[0, -1]
#     TOCNum = len(TOC_list)
#     fig, ax = plt.plot()


# ---------- This part is for the use functions for TOC curve



# ---------- This part is the class for TOC curve ----------
class TOC:
    def __init__(self, booleanArray, indexArray, thresholds, maskArray=None, weightsArray=None):
        ## define boolean and index variables and thresholds
        self.booleanArray = np.array(booleanArray).flatten()
        self.indexArray = np.array(indexArray).flatten()
        self.thresholds = np.array(thresholds).flatten()
        ## if Users don't assign any mask, the class will assign it as a matrix with all 1
        if (maskArray is None):
            self.maskArray = np.ones_like(self.booleanArray).flatten()
        else:
            self.maskArray = np.array(maskArray).flatten()
        ## if Users don't assign any mask, the class will assign it as a matrix with all 1
        if (weightsArray is None):
            self.weightsArray = np.ones_like(self.booleanArray).flatten()
        else:
            self.weightsArray = np.array(weightsArray).flatten()
        # initialize TOCX and TOCY and AUC
        # TOCX and TOCY
        self.TOCX = np.array([[]])
        self.TOCY = np.array([[]])
        self.AUC = 0
        self.correctCornerY = 0
        ##init threshold labels
        self.thresholdLabel = np.array([])
        # calculate the correctCornerY and totalNum
        self.presenceInY = np.sum(self.booleanArray * self.weightsArray)
        self.totalNum = np.sum(self.weightsArray)
        ## calculate TOCX, TOCY, threshold labels, Y value of correctCorner, and AUC
        self.__calculate_TOC()
    def summary(self):
        print('The size of extent:', self.totalNum)
        print('Abundance:',self.presenceInY)
        print('AUC:',self.AUC)
        print('The coordinate of point below top left corner:','({0},{1})'.format(self.presenceInY,self.correctCornerY))

    def __calculate_TOC(self):
        # There is no origin point when the minimum threshold is the same as index minimum
        self.TOCX = []
        self.TOCY = []
        for threshold in self.thresholds:
            # tell whether the threshold is ascending or descending
            if (self.thresholds[-1] < self.thresholds[0]):
                BoolIndexArray = (self.indexArray >= threshold).astype(
                    int)  # set value 1 for pixel which is higher than threshold, other pixels are assigned 0
            else:
                BoolIndexArray = (self.indexArray <= threshold).astype(
                    int)  # set value 1 for pixel which is higher than threshold, other pixels are assigned 0
            self.TOCX.append((BoolIndexArray * self.maskArray).sum())  # Hits and False Alarms
            self.TOCY.append((BoolIndexArray * self.maskArray * self.booleanArray).sum())  # Hits

        # add original zero
        if (self.TOCX[0] != 0):
            self.TOCX = [0] + self.TOCX
            self.TOCY = [0] + self.TOCY
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
        self.__calculate_AUC()
        self.__format_coordinates()
        self.correctCornerY = self.__correctCorner(self.TOCX, self.TOCY, self.presenceInY)

    def __format_coordinates(self):
        self.TOCX = np.array(self.TOCX).flatten()
        self.TOCX = self.TOCX.reshape((1, self.TOCX.shape[0]))
        self.TOCY = np.array(self.TOCY).flatten()
        self.TOCY = self.TOCY.reshape((1, self.TOCY.shape[0]))

    def __correctCorner(self, TOCX, TOCY, presenceInY):
        bool_presence = (TOCX<=presenceInY).astype(int)
        corner_index = bool_presence.sum()-1
        if(corner_index == TOCY.shape[1]-1):
            y_res = TOCX[0, -1]
        else:
            y_res = (TOCY[0,corner_index+1]-TOCY[0,corner_index])*1.0/(TOCX[0,corner_index+1]-TOCX[0,corner_index])*(presenceInY-TOCX[0,corner_index])+TOCY[0,corner_index]*1.0
        return y_res


    def __calculate_AUC(self):
        self.__format_coordinates()
        presenceInY = self.TOCY[0, -1]
        totalNum = self.TOCX[0, -1]
        Area01 = self.__AreaUnderCurve(self.TOCX, self.TOCY)
        Area02 = self.__AreaUnderCurve(np.array([[0, presenceInY]]), np.array([[0, presenceInY]]))
        Area03 = self.__AreaUnderCurve(np.array([[0, totalNum]]), np.array([[0, presenceInY]])) * 2
        AUC = (Area01 - Area02) / (Area03 - 2 * Area02)
        self.AUC = float(AUC)

    def __AreaUnderCurve(self, listx, listy):
        listx = np.array(listx)
        listy = np.array(listy)
        Llistx = np.delete(listx, listx.shape[1] - 1)
        Hlistx = np.delete(listx, 0)
        Llisty = np.delete(listy, listy.shape[1] - 1)
        Hlisty = np.delete(listy, 0)
        Areas = np.sum(np.int64([(Hlistx - Llistx) * (Hlisty + Llisty)]), axis=1) / 2.0
        return Areas
