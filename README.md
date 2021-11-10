# PyTOC
A Python library for generating Total Operating Characteristic (TOC) Curves.

**The main features of this library are:**

- It calculates all properties of TOC at once.
- It draws multiple TOC curves in one diagram.
## Table of Contents
- [Quick start](#quick-start)
- [Usage](#usage)
- [Example](#example)


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
It means all elements are put into the calculation and they all have the same weights， which are 1.

If we initialize the function
```angular2html
TOC_1 = TOC(booleanArray, indexArray, thresholds, maskArray=maskArray)
```
It means we only use the first, second, third, and fifth element to calculate TOC and they have the same weights, which are 1.

If we initialize the function
```angular2html
TOC_1 = TOC(booleanArray, indexArray, thresholds, weightsArray=weighsArray)
```
It means we use all elements for calulation and set different weights to elements.

After Initializing TOC class, we can use summary function to get the properties of the TOC. It will show the size of extent, Abundance, AUC, and the coordinate of point below the top left corner.
```angular2html
TOC_1.summary()
```
Also, we can just call those properties like:
```angular2html
print("The size of extent:", TOC_1.totalNum)
print("Abundance:", TOC_1.presenceInY)
print("AUC:", TOC_1.AUC)
print("The vertical coordinate of point below the top left corner:", TOC_1.correctCornerY)
```
### TOC_painter
The other important class is TOC_painter. This class is to draw TOC curves. It supports to display multiple TOC curves  in one diagream. The initialization is
```angular2html
painter = TOC_painter(TOC_list=[TOC_1], index_names=['distance'], color_list=['r'], marker_list=['^'], line_list=['-'], boolUniform=False, boolCorrectCorner=False)
painter.paint()
```
Also, If there is only one TOC curve to show, we can just initilize it like:
```angular2html
painter = TOC_painter(TOC_list=TOC_1, index_name='distance', color_list='r', marker_list='^', line_list='-',  boolUniform=False, boolCorrectCorner=False)
painter.paint()
```
First, let us go through all the parameters in the initialization function.
- TOC_list: The TOC curve you want to show. It should be a list of TOC objects.
- index_names: Names of TOC curves. They will be shown in the legend.
- color_list: A list of colors corresponding to TOC curves. The color can be one character like 'r' (red), 'g'(green), 'b'(blue); a color word, like "aqua", "green"; or hexadecimal color notation, like "#1f77b4". The details are in the [link](https://matplotlib.org/stable/gallery/color/named_colors.html).
- marker_list: A list of marker types corresponding to TOC curves. Markers can be "^" (triangle_up), "v" (triangle_down).All possible markers are in the [link](https://matplotlib.org/stable/api/markers_api.html). 
- line_list: A list of line types corresponding to TOC curves. It can be '-' (solid line), '--' (dashed line). All possible line types are in the [link](https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle)
- boolUniform: If true, a uniform TOC curve will be put in the diagram.
- boolCorrectCorner: If True, there is a explanation text shown on the top of axes. "The red star marks where False Alarms equals Misses." It tells readers about the meaning of red stars in the diagram.

If you don't want to set the properties when you initialize the painter, you can only initialize TOC_list, like
```angular2html
painter = TOC_painter(TOC_list=[TOC_1, TOC_2])
painter.paint()
```
The default settings for those parameters will be:
- index names will be "TOC1", "TOC2"...
- color list will be set randomly.
- marker will be non-marker.
- line will be solid line.

You can even initialize the object TOC_painter without any parameters, like
```angular2html
painter = TOC_painter()
```
and add TOC objects or coordinates before using paint() function

There are two functions in the TOC_painter to add TOC curves to the painter. There are

**add_TOC** and **add_TOC_coor**

1. add_TOC is the function to add object TOC to the painter. 
```angular2html
painter.add_TOC(TOC_1, index_name=None, color='', marker='', line='-')
```
If you don't want set any parameter and use the default settings, you can also just
```angular2html
painter.add_TOC(TOC_1)
```
2. add_TOC_coor is the function to add TOC coordinates to the painter.

It is designed to avoid unnessary generation of object TOC. Users can read TOC coordinates from files and show them use TOC_painter.
```angular2html
painter.add_TOC_coor(x_coor, y_coor, index_name=None, color='', marker='', line='-')
```
x_coor and y_coor can be lists or arrays.If you don't want set any parameter and use the default settings, you can also just
```angular2html
painter.add_TOC_coor(x_coor, y_coor)
```

#### add_correct_corner() and add_all_correct_corner()
TOC_painter also provides functions to draw the point where the misses equals false alarms on the TOC curves. There are two functions to use.
- add_correct_corner(): The function can show the correct corner for one TOC curve. The parameter in the function is the index of the TOC curve.
```angular2html
painter = TOC_painter([TOC_1, TOC_2])
## show the correct corner for TOC_2, because 0 is for TOC_1 and 1 is for TOC_2
painter.add_correct_corner(1)
painter.paint()
```
- add_all_correct_corner(): The function can show all correct corners in the painter.
```angular2html
painter = TOC_painter([TOC_1, TOC_2])
## show correct corners for all TOC curves
painter.add_all_correct_corner()
painter.paint()
```

#### Some tricks
Becasue the painter is on the basis of Python library matplotlib, users can customize the diagram using matplotlib functions before painter.paint(). There are some tricks.
1. Set x and y title
```angular2html
# init painter
painter = TOC_painter(TOC_1)
plt.xlabel('Hits+False alarms (square km)')
plt.ylabel('Hits (square km)')
painter.paint()
```

2. Change ticks of x and y axes
```angular2html
painter = TOC_painter(TOC_1)
plt.xticks([0,1000,2000,3000,4000])
plt.yticks([0, 100, 200, 300, 400])
painter.paint()
```

## Example

There is an example to read raster data to generate and display a TOC curve. In this case, we want to analyze the relationship between new disturbance and the distance to the existing disturbance.
- Read Data: there is several python libaries to read raster data, like gdal, rasterio. Users can use what they prefer. The important step here is to convert the input data to data in the format of numpy.array
```angular2html
import matplotlib.pyplot as plt
import rasterio
from pytoc import TOC, TOC_painter

## read raster file
src_label = rasterio.open('data/1971_Built_Gain.rst')
src_index = rasterio.open('data/Distance_1971_Built.rst')
src_mask = rasterio.open('data/Nonbuilt_1971.rst')
## convert inputs to numpy arrays
label_array = np.array(src_label.read(1))
index_array = np.array(src_index.read(1))
mask_array = np.array(src_mask.read(1))
```
- Create TOC object: we want to every values in the index array as thresholds. so, we use np.unique to get unique values.
```angular2html
TOC_1 = TOC(label_array,index_array,np.unique(index_array),mask_array)
```
- Show TOC curves
```angular2html
## initialize TOC_painter class with only one TOC object
new_paint = TOC_painter(TOC_1)
## show explanation text for correct corners and draw a uniform curve
new_paint.boolCorrectCorner=True
new_paint.boolUniform=True
## add all correct corners to curves
new_paint.add_all_correct_corner()
## set text on horizontal and vertical axes
plt.xlabel('Hits+False alarms (square km)')
plt.ylabel('Hits (square km)')
## set horizontal and vertical ticks
plt.xticks([0,1000,2000,3000,4096])
plt.yticks([0,90,180,270,360,460])
## paint the diagram
new_paint.paint()
```