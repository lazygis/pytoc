import sys
from osgeo import gdal
import numpy as np
import xlrd
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from matplotlib import colors as matpltColors
from matplotlib import transforms
from PyQt5.QtWidgets import QTableWidgetItem


# ---------- This part is the class for TOC curve ----------

class TOC_Multi_Generator:
    def __init__(self, booleanArray, continuousArray, maskArray, thresholds):
        self.booleanArray = booleanArray
        self.continuousArray = continuousArray
        self.maskArray = maskArray
        self.thresholds = thresholds
        self.thresholdLabel = thresholds
        self.booleanMaskArray=self.booleanArray*self.maskArray
        self.presenceInY = np.sum(self.booleanMaskArray)
        self.correctCornerY = 0
        self.totalNum = np.sum(self.maskArray)
        self.matrixShape=np.shape(self.booleanArray)
        # TOCX and TOCY are list
        self.TOCX=0
        self.TOCY=0
        self.AUC=0


    def calculate_TOC(self):
        # There is no origin point when the minimum threshold is the same as index minimum
        self.TOCX=[0]
        self.TOCY=[0]
        for threshold in self.thresholds:
            # tell whether the threshold is ascending or descending
            if(self.thresholds[-1]<self.thresholds[0]):
                BoolDisArray = (self.continuousArray >= threshold).astype(
                    int)  # set value 1 for pixel which is higher than threshold, other pixels are assigned 0
            else:
                BoolDisArray = (self.continuousArray <= threshold).astype(
                    int)  # set value 1 for pixel which is higher than threshold, other pixels are assigned 0


            ResultArray = BoolDisArray * self.booleanMaskArray  # get the result array
            self.TOCX.append((BoolDisArray * self.maskArray).sum())  # Hits and False Alarms
            self.TOCY.append(ResultArray.sum())  # Hits

        # delete redundant zero
        # if(self.TOCX[1]==0):
        #     self.TOCX = self.TOCX[1:]
        #     self.TOCY = self.TOCY[1:]
        # else:
        #     self.thresholdLabel = np.append(np.array(['origin']), self.thresholds)

        self.thresholdLabel = np.append(np.array(['origin']), self.thresholds)

        # when customize the threshold, the upper bond is not achieved, so we need to add it to the threshold labels
        if(self.TOCX[-1]!=self.totalNum):
            self.TOCX.append(self.totalNum)
            self.TOCY.append(self.presenceInY)
            self.thresholdLabel = np.append(self.thresholdLabel, np.array(['end']))
        self.TOCY = np.array([self.TOCY])
        self.TOCX = np.array([self.TOCX])

        self.correctCornerY = correctCorner(self.TOCX,self.TOCY,self.presenceInY)
    # calculate categorical values for TOC
    def calculate_TOC_C(self):
        self.TOCX = [0]
        self.TOCY = [0]
    #     InputMaskArray = InputArray * MaskArray
    #     cate_sequence = self.lineEdit_03_cate_custom.text()
    #     countdown_init = cate_sequence.strip('][').split(',')
    #     countdown = [float(i) for i in countdown_init]
    #     distancemin = countdown[0]
    #     distancemax = countdown[-1]
        y_num = {}
        x_num = {}
        className = np.unique(self.continuousArray)
        if (0 in className):
            if (np.sum((1 - self.continuousArray.astype(bool)) * self.maskArray) == 0):
                className = np.delete(className, list(className).index(0))
    #
        for i in className:
            number = np.sum(self.booleanArray * (1 - (self.continuousArray - i).astype(bool).astype(int)) * self.maskArray)
            numberX = np.sum((1 - (self.continuousArray - i).astype(bool).astype(int)) * self.maskArray)
            y_num[i] = number
            x_num[i] = numberX
        sumY = 0
        sumX = 0
        for item in self.thresholds:
            sumY = sumY + y_num[item]
            sumX = sumX + x_num[item]
            self.TOCX.append(sumX)
            self.TOCY.append(sumY)
        self.thresholdLabel = np.append(np.array(['origin']), self.thresholds)
        self.presenceInY = self.TOCX[-1]
        self.totalNum = self.TOCY[-1]
        self.TOCY = np.array([self.TOCY])
        self.TOCX = np.array([self.TOCX])

    def calculate_AUC(self):
        Area01 = areaUnderCurve(self.TOCX,self.TOCY)
        Area02 = areaUnderCurve(np.array([[0, self.presenceInY]]),np.array([[0, self.presenceInY]]))
        Area03 = areaUnderCurve(np.array([[0, self.totalNum]]),np.array([[0, self.presenceInY]]))*2
        AUC = (Area01-Area02)/(Area03-2*Area02)
        self.AUC = float(AUC)




# ---------- This part is painter class----------
#  digits rule: 1. maximum three significant digits; 2. the same decimal places
class labelTranslator:
    def __init__(self, digits,decimalPlaces):
        self.digits = digits
        self.decimalPlaces = decimalPlaces
    def specialDigits(self,temp,position):
        if (temp == 0):
            return '0'
        else:
            return '%.1f' % (temp / 10 ** self.digits)

    def threeDigits(self, temp, position):
        if(temp==0):
            return '0'
        elif(self.decimalPlaces==0):
            return '%.0f' % (temp / 10 ** self.digits)
        elif (self.decimalPlaces == 1):
            return '%.1f' % (temp / 10 ** self.digits)
        elif (self.decimalPlaces == 2):
            return '%.2f' % (temp / 10 ** self.digits)
        elif (self.decimalPlaces == 3):
            return '%.3f' % (temp / 10 ** self.digits)
class painter_Generator:
    def __init__(self, Xlist, Ylist, unit):
        self.Xlist = Xlist
        self.Ylist = Ylist
        self.totalNum = Xlist[0][-1]
        self.presenceInY = Ylist[0][-1]
        self.unit = unit
        self.TOCNum = len(Xlist)
        self.clickIndex = 0
        self.labelList = []
        self.fig = 0

    def axisRange(self,Xmax):
        numDigit = len(str(int(Xmax))) - 1
        resultRange = np.arange(0, Xmax, Xmax / 5)
        # if ((Xmax - resultRange[-1]) < 10 ** numDigit / 5.0):
        #     resultRange = np.delete(resultRange, -1)
        resultRange = np.append(resultRange, Xmax)
        return resultRange


    def paintInit(self, boolUniform,boolCorrectcorner):
        Xmax = self.totalNum
        Ymax = self.presenceInY
        plt.close()

        self.fig = plt.figure()

        #  maximum, minimum and uniform line
        # l3 = plt.plot([0, Ymax, Xmax], [0, Ymax, Ymax], 'r--', color="blue",
        #               label='Maximum')
        if(boolCorrectcorner):
            CorrectCornerText1 = plt.gca().text(0, Ymax * 1.01, 'The ', color="black", fontsize=8)
            CorrectCornerText1.draw(plt.gca().figure.canvas.get_renderer())
            ex = CorrectCornerText1.get_window_extent()
            t = transforms.offset_copy(
                CorrectCornerText1.get_transform(), x=ex.width, units='dots')
            CorrectCornerText2=plt.gca().text(0, Ymax * 1.01, '★ ', color="red", fontsize=8, transform = t)
            CorrectCornerText2.draw(plt.gca().figure.canvas.get_renderer())
            ex = CorrectCornerText2.get_window_extent()
            t = transforms.offset_copy(
                CorrectCornerText2.get_transform(), x=ex.width, units='dots')
            plt.gca().text(0, Ymax * 1.01, 'marks where False Alarms equals Misses.', color="black", fontsize=8, transform = t)
        #     marks where Misses equals False Alarms.
        if(boolUniform):
            l2 = plt.plot([0, Xmax], [0, Ymax], 'r:', color="violet", label='Uniform')

        # l4 = plt.plot([0, Xmax - Ymax, Xmax], [0, 0, Ymax], 'r--',
        #               color="purple", label='Minimum')
        #  Grey area in outer TOC area
        cmap = matpltColors.ListedColormap('#e0e0e0')
        plt.tripcolor([Xmax-Ymax, Xmax, Xmax], [0, 0, Ymax], [0, 1, 2], np.array([1, 1, 1]), edgecolor="k", lw=0, cmap=cmap)
        plt.tripcolor([0, 0, Ymax], [0, Ymax, Ymax], [0, 1, 2], np.array([1, 1, 1]), edgecolor="k", lw=0,
                      cmap=cmap)

        # make the coordinates square
        plt.axis('square')
        # plt.axis('auto')
        plt.axis([0, Xmax, 0, Ymax])
        self.ticksOption()

        # plt.axis([0, Xmax, 0, Ymax])
        plt.gca().set_aspect(1 / plt.gca().get_data_ratio())
    def show(self):
        plt.show()

    def paintOne(self, index, Name, marker):
        plt.plot(self.Xlist[index], self.Ylist[index], marker+'-', label=Name, picker=2, clip_on=False)
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
    def correctCorner(self,index):
        correctCornerY = correctCorner(np.array([self.Xlist[index]]), np.array([self.Ylist[index]]),self.presenceInY)
        plt.plot(self.presenceInY, correctCornerY, 'r*')

    def correctCornerLabel(self, index):
        correctCornerY = correctCorner(np.array([self.Xlist[index]]), np.array([self.Ylist[index]]), self.presenceInY)
        plt.text(self.presenceInY, correctCornerY, str((self.presenceInY, round(correctCornerY,2))),color="red")



    # set up for a ticks labels
    def ticksOption(self):
        # ticks for x and y axis
        Xmax = self.totalNum
        Ymax = self.presenceInY
        plt.xticks(self.axisRange(Xmax))
        plt.yticks(self.axisRange(Ymax))

        reference = {0:'',3:'thousand ',6:'million ', 9:'billion ', 12:'trillion '}


        Digits_X_3 = (int(len(str(int(Xmax))) - 1) // 3) * 3
        X_decimal = decimalPlace(Xmax, Digits_X_3)
        plt.xlabel('Hits+False Alarms' + ' (' + reference[Digits_X_3] + self.unit + ')')
        LTranslator = labelTranslator(Digits_X_3,X_decimal)
        plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(LTranslator.threeDigits))
        # plt.text(Xmax * 1.02, Ymax * 0.000, 'E' + str(Digits_X_3), fontsize=10)

        Digits_Y_3 = (int(len(str(int(Ymax))) - 1) // 3) * 3
        Y_decimal = decimalPlace(Ymax, Digits_Y_3)
        plt.ylabel('Hits' + ' (' + reference[Digits_Y_3] + self.unit + ')')
        LTranslator = labelTranslator(Digits_Y_3,Y_decimal)
        plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(LTranslator.threeDigits))
        # plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(LTranslator.specialDigits))
        # plt.text(Xmax * 0.00, Ymax * 1.02, 'E' + str(Digits_Y_3), fontsize=10)
    def setLabelList(self, labelList):
        self.labelList = labelList
    # show labels (can change properties later)
    def showLabel(self,list_label):
        for item in list_label:
            plt.text(float(item[0]), float(item[1]), item[2])
    def ClickReact(self, thresholdTable, indexCombo, thresholdDigits):
        fig=self.fig
        self.clickIndex = indexCombo.currentIndex()
        x_for_label = self.Xlist[self.clickIndex]
        y_for_label = self.Ylist[self.clickIndex]
        label_dis = self.labelList[self.clickIndex]

        def onpick(event):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            # points = tuple(zip(xdata[ind], ydata[ind]))
            index_label = -1
            # print(ind)
            # print(xdata.shape)☺
            if(np.array_equal(x_for_label,xdata)):
                index_label=np.array([ind])[0,0]
            # print(index_label)
            # print(x_for_label)

            # if(xdata==x_for_label):
            #     index_label = ind[0]
            #     print(index_label)
            if(index_label>=0):
                # print(ind)
                #
                # print(index_label)
                # print('hello')
                threshold_digits = int(thresholdDigits.text())
                row = thresholdTable.rowCount()
                formatThreshold='{0:.'+str(threshold_digits)+'f}'
                thresholdTable.insertRow(thresholdTable.rowCount())
                thresholdTable.setItem(row, 0, QTableWidgetItem(str(xdata[ind][0])))
                thresholdTable.setItem(row, 1, QTableWidgetItem(str(ydata[ind][0])))
                if (label_dis[index_label]=='origin' or label_dis[index_label]=='end'):
                    thresholdTable.setItem(row, 2, QTableWidgetItem(label_dis[index_label]))
                else:
                    thresholdTable.setItem(row, 2, QTableWidgetItem(formatThreshold.format((float(label_dis[index_label])))))







        fig.canvas.mpl_connect('pick_event', onpick)







# ---------- This part is the relevant functions for painter----------
def decimalPlace(maximum, tenPower):
    resultRange = np.arange(0, maximum, maximum / 5)
    decimalDigits = 0
    for i in resultRange:
        str1="{:.3g}".format(i/(10 ** tenPower))
        if('.' in str1):
            if (len(str1.split('.')[1]) > decimalDigits):
                decimalDigits = len(str1.split('.')[1])
    return decimalDigits

def readCoordinates(filename):
        x1 = []
        y1 = []
        label = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                # x1.append(int(float(line.strip().split(' ')[0])))
                # y1.append(int(float(line.strip().split(' ')[1])))
                x1.append((float(line.strip().split(' ')[0])))
                y1.append((float(line.strip().split(' ')[1])))
                label.append((line.strip().split(' ')[2]))
        X = np.array(x1)
        Y = np.array(y1)
        return X, Y, label
def readMultipleCoordinates(filenameList):
    Xlist = list()
    Ylist = list()
    labels = list()
    for item in filenameList:
        X, Y, Z = readCoordinates(item)
        Xlist.append(X)
        Ylist.append(Y)
        labels.append(Z)
    return Xlist, Ylist, labels


# ---------- This part is the relevant functions for TOC class ----------

# This function is to generate the customized thresholds
# sequenceOption: 0 is descending, 1 is ascending
def threshold_generate_custom(min, max, interval, sequenceOption):
    if(sequenceOption == 0):
        countdown = np.arange(max, min, -interval)
        if(min!=countdown[-1]):
            countdown = np.append(countdown,min)
    if (sequenceOption == 1):
        countdown = np.arange(min, max, interval)
        if(max!=countdown[-1]):
            countdown = np.append(countdown,max)
    return countdown

# This function is to generate the unique thresholds
# sequenceOption: 0 is descending, 1 is ascending
def threshold_generate_uniq(indexArray, sequenceOption):
    One_row_array = indexArray.flatten()
    One_row_array_sorted = np.sort(One_row_array)
    # get the uniq value
    One_row_array_uniq = np.diff(One_row_array_sorted)
    One_row_array_uniq = np.r_[1, One_row_array_uniq]
    res = One_row_array_sorted[One_row_array_uniq != 0]
    if(sequenceOption == 0):
        return res[::-1]
    if(sequenceOption == 1):
        return res


# use the matrix to speed up the calculation
def areaUnderCurve(listx, listy):  # compute the area under the curve
    if(listx[0,-1]>65536):
        listx=np.int64(listx)
        listy=np.int64(listy)
        Llistx = np.delete(listx, listx.shape[1] - 1)
        Hlistx = np.delete(listx, 0)
        Llisty = np.delete(listy, listy.shape[1] - 1)
        Hlisty = np.delete(listy, 0)
        Areas = np.sum(np.int64([(Hlistx - Llistx) * (Hlisty + Llisty)]), axis=1) / 2.0
    else:
        Llistx = np.delete(listx, listx.shape[1] - 1)
        Hlistx = np.delete(listx, 0)
        Llisty = np.delete(listy, listy.shape[1] - 1)
        Hlisty = np.delete(listy, 0)
        Areas = np.sum(np.array([(Hlistx - Llistx) * (Hlisty + Llisty)]), axis=1) / 2.0


    # Areas=0
    # for i in range(listx.shape[1] - 1):
    #     Areas = (listy[0,i] + listy[0,i + 1]) * (listx[0,i + 1] - listx[0,i]) / 2.0 + Areas
    return Areas

def correctCorner(TOCX,TOCY,presenceInY):
    Boolx1 = (TOCX <= presenceInY).astype(int)
    index1 = Boolx1.sum() - 1
    # print((presenceInY - TOCX[0, index1]))
    if (index1 == TOCY.shape[1] - 1):
        y_res = TOCY[0, -1]
    else:
        y_res = (TOCY[0, index1 + 1] - TOCY[0, index1]) * 1.0 / (TOCX[0, index1 + 1] - TOCX[0, index1]) * (presenceInY - TOCX[0, index1]) + TOCY[0, index1]*1.0
    # problem located
    return y_res

def calculate_category_threshold(booleanArray,continuousArray,maskArray,type_index):
    y_num = {}
    x_num = {}
    slope = {}
    className = np.unique(continuousArray)
    if (0 in className):
        if (np.sum((1 - continuousArray.astype(bool)) * maskArray) == 0):
            className = np.delete(className, list(className).index(0))
    if (type_index == 0):
        for i in className:
            number = np.sum(booleanArray * (1 - (continuousArray - i).astype(bool).astype(int)) * maskArray)
            numberX = np.sum((1 - (continuousArray - i).astype(bool).astype(int)) * maskArray)
            y_num[i] = number
            x_num[i] = numberX
            slope[i] = number * 1.0 / numberX
        opt_category = sorted(slope.items(), key=lambda d: d[1], reverse=True)
        opt_countdown = []
        for item in opt_category:
            opt_countdown.append(item[0])
        return opt_countdown
    elif (type_index == 1):
        return(className.tolist()[::-1])
    elif (type_index == 2):
        return(className.tolist())
    else:
        pass

# ---------- This part is to store the function to read the source and return numpy array ----------

def readRaster(file_address):
    InputRaster = gdal.Open(file_address)
    inputArray = InputRaster.ReadAsArray()
    return inputArray
def readTiff(file_address):
    InputTiff = gdal.Open(file_address)
    inputArray = InputTiff.ReadAsArray()
    return inputArray

def readExcel(file_address):
    wb = xlrd.open_workbook(file_address)
    sheet1 = wb.sheet_by_index(0)
    col1 = sheet1.col_values(0)
    col2 = sheet1.col_values(1)
    if(len(sheet1.row_values(0))>2):
        classification = sheet1.col_values(2)
        IndexArray = np.array(col1)
        BooleanArray = np.array(col2)
        MaskArray = np.ones((1, len(col1)))
        return BooleanArray, IndexArray, MaskArray, classification
    else:
        IndexArray = np.array(col1)
        BooleanArray = np.array(col2)
        MaskArray = np.ones((1, len(col1)))
        return BooleanArray, IndexArray, MaskArray, None

def readTxt(file_address):
    txtArray = np.loadtxt(file_address)
    IndexArray = txtArray[:, 0]
    BooleanArray = txtArray[:, 1]
    MaskArray = np.ones((1, txtArray.shape[0]))
    return BooleanArray, IndexArray, MaskArray


# default array filled with ones. use when mask is not accessible.
def default_one_array(sameSizeArray):
    return np.ones(np.shape(sameSizeArray))


if (__name__ == "__main__"):
    # booleanRaster = "../source/1971_Built_Gain.rst"
    # continuousRaster = "../source/Distance_1971_Built.rst"
    # maskRaster = "../source/Nonbuilt_1971.rst"
    booleanRaster = "../source/S1_new.rst"
    continuousRaster = "../source/S2_MNDWI.rst"
    maskRaster = "../source/QC_Mask2.rst"
    BArray = readRaster(booleanRaster)
    CArray = readRaster(continuousRaster)
    MArray = readRaster(maskRaster)

    excelFile = "../source/testdata.xlsx"
    BArray_E, CArray_E, MArray_E =readExcel(excelFile)

    txtFile = "../source/test.txt"
    BArray_T, CArray_T, MArray_T = readTxt(txtFile)


    num1 = 1014
    TestArray = np.arange(0,num1+1,1)
    testMask = default_one_array(TestArray)
    # TOCGen=TOC_Multi_Generator(TestArray,TestArray,testMask,0)
    TOCGen = TOC_Multi_Generator(BArray, CArray, MArray, 0)
    TOCGen.thresholds = threshold_generate_uniq(TOCGen.continuousArray,1)
    # TOCGen.thresholds = threshold_generate_custom(30,98,1,1)
    TOCGen.calculate_TOC()
    xList = TOCGen.TOCX
    yList = TOCGen.TOCY
    # integrate multiple numpy array together
    testList = np.vstack([xList,yList])
    paintL = painter_Generator(xList,yList,'pixels')
    paintL.paintInit()
    paintL.paintOne(0,'distance','^')
    paintL.show()
#
#     # print(areaUnderCurve(xList,yList))

