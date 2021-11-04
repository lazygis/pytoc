# PyTOC
A Python library for generating Total Operating Characteristic (TOC) Curves.

**The main features of this library are:**

- It calculates all properties of TOC at once.
- It draws multiple TOC curves in one diagram.
## Table of Contents
- [Quick start](#quick-start)
- [Usage](#usage)


## Quick start
- Install
```angular2html
pip install pytoc
```
When you install the **pytoc** package, it will also install the dependencies -- Numpy and matplotlib.
## Usage
There are two classes in the library -- TOC and TOC_painter.
### TOC
TOC is a class to calculate properties of a TOC curve and save them in the class. We can import it use codes
```angular2html
from pytoc import TOC
```
The format of inputs can be list or Numpy array.

The inputs are:
- booleanArray: a reference array or list. 0 means absence and 1 mean presence.
- indexArray: an array or list which contain index values.
- thresholds: an array or list of thresholds.
- maskArray: an array or list to show the study area. 0 means non study area and 1 means study area. If users don't assign a mask, all elements will be considered in the calculation.
- weightsArray: an array or list to set weights to each element. If users don't assign weights, all elements have the weight of 1.

For example, we have data like this
```
booleanArray = [0,1,0,1,0]
indexArray = [0,1,2,3,4]
thresholds = [0,1,2,3,4]
maskArray = [1,1,1,0,1]
weightsArray = [1,2,3,2,1]
```
If we initialize the function
```angular2html
TOC_1 = TOC(booleanArray, indexArray, thresholds)
```
It means all elements are put into the calculation and they all have the same weights.

If we initialize the function
```angular2html
TOC_1 = TOC(booleanArray, indexArray, thresholds, maskArray=maskArray)
```
It means we only use the first, second, third, and fifth element to calculate TOC and they have the same weights, whcih are 1.

