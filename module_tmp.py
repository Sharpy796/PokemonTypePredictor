#Version: v0.1
#Date Last Updated: 12-20-2023

#%% STANDARDS   -DO NOT include this block in a new module
'''
Unless otherwise required, use the following guidelines
* Style:
    - Sort all alphabatically
    - Write the code in aesthetically-pleasing style
    - Names should be self-explanatory
    - Add brief comments
    - Use relative path
    - Use generic coding instead of manually-entered constant values

* Performance and Safety:
    - Avoid if-block in a loop-block
    - Avoid declarations in a loop-block
    - Initialize an array if size is known

    - Use immutable types
    - Use deep-copy
    - Use [None for i in Sequence] instead of [None]*len(Sequence)

'''

#%% MODULE BEGINS
module_name = '<***>'

'''
Version: <***>

Description:
    <***>

Authors:
    Christian Bankovic
    Ethan Cochran
    Davidson Rock

Date Created     :  3/28/26
Date Last Updated:  3/28/26

Doc:
    <***>

Notes:
    <***>
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
   import os
   #os.chdir("./../..")
#

# Not sure if we need alll this, but I have this now
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from torchvision.transforms import ToTensor
# import scipy
# from scipy.stats import zscore
# from time import sleep
# from tqdm import tqdm
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score, f1_score, log_loss
import matplotlib.pyplot as plt
# import matplotlib as mpl
# from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import pandas as pd
import numpy as np
# import kagglehub
# from kagglehub import KaggleDatasetAdapter
# import math
import cv2
from PIL import Image
from collections import Counter
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


#custom imports


#other imports
from   copy       import deepcopy as dpcpy

'''
from   matplotlib import pyplot as plt
import mne
import numpy  as np 
import os
import pandas as pd
import seaborn as sns
'''
#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CSV_FILEPATH = "pokemon_images/pokedex.csv"

# These are for color categorizing purposes.
primary_colors = { # bgr
    'Green':[0,225,0],
    'Red':[0,0,255],
    'Blue':[255,0,0],
    'White':[255,255,255],
    'Brown':[0,60,135],
    'Orange':[0,121,255],
    'Yellow':[0,255,255],
    'Purple':[255,0,180],
    'Pink':[255, 0, 255],
    'Gray':[125,125,125],
    'Black':[0,0,0]
}

# These are for chart sorting/styling purposes.
COLORS = ['Red','Yellow','Green','Blue','Purple','Pink','Brown','White','Gray','Black']
COLORS_CHART = ['Red','Yellow','Green','Blue','Purple','Pink','Brown','Gainsboro','Gray','Black']
TYPES1 = ['Normal','Fire','Water','Grass','Electric','Ice','Fighting','Poison','Ground','Flying','Psychic','Bug','Rock','Ghost','Dragon','Dark','Steel','Fairy']
TYPES2 = ['Normal','Fire','Water','Grass','Electric','Ice','Fighting','Poison','Ground','Flying','Psychic','Bug','Rock','Ghost','Dragon','Dark','Steel','Fairy','None']
TYPES1_COLORS_CHART = ['wheat','darkorange','skyblue','limegreen','yellow','paleturquoise','darkred','mediumorchid','khaki','thistle','deeppink','yellowgreen','goldenrod','mediumpurple','mediumblue','black','lightsteelblue','pink']
TYPES2_COLORS_CHART = ['wheat','darkorange','skyblue','limegreen','yellow','paleturquoise','darkred','mediumorchid','khaki','thistle','deeppink','yellowgreen','goldenrod','mediumpurple','mediumblue','black','lightsteelblue','pink','dimgray']

#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here



#Class definitions Start Here



#Function definitions Start Here
def main():
    pass
#

def visualizeData(df, save_name):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
    df_visualizer = df.drop(['id','name','pokedex_id',], axis=1)
    df_visualizer = df_visualizer.sort_values('type1', key=lambda s: s.apply(TYPES1.index), ignore_index=True)
    axs[0][0].set_title("type1")
    axs[0][0].bar(TYPES1,df_visualizer['type1'].value_counts(sort=False).to_numpy(),color=TYPES1_COLORS_CHART)
    df_visualizer = df_visualizer.sort_values('primary_color', key=lambda s: s.apply(COLORS.index), ignore_index=True)
    axs[1][0].set_title("primary_color")
    axs[1][0].bar(COLORS,df_visualizer['primary_color'].value_counts(sort=False).to_numpy(),color=COLORS_CHART)
    df_visualizer_nona = df_visualizer.dropna().sort_values('type2', key=lambda s: s.apply(TYPES1.index), ignore_index=True)
    axs[1][1].set_title("type2 removing mono-type pokemon ('None')")
    axs[1][1].bar(TYPES1,df_visualizer_nona['type2'].value_counts(sort=False).to_numpy(),color=TYPES1_COLORS_CHART)
    df_visualizer = df_visualizer.fillna("None").sort_values('type2', key=lambda s: s.apply(TYPES2.index), ignore_index=True)
    axs[0][1].set_title("type2")
    axs[0][1].bar(TYPES2,df_visualizer['type2'].value_counts(sort=False).to_numpy(),color=TYPES2_COLORS_CHART)
    for i,sax in enumerate(axs):
        for j,ax in enumerate(sax):
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
    plt.tight_layout()
    if (save_name != None):
        plt.savefig(save_name)
    #
#

def assign_spritepaths(df):
    root = "pokemon_images/sprites/"
    for i,entry in df.iterrows():
        front = root+f"{entry['id']:04d}"+'-'+entry['name']+'-'+str(entry['pokedex_id'])+'/front/normal'
        images = os.listdir(front)
        if len(images) > 0:
            front += "/"+images[len(images)-1]
        else:
            front = "NONE"
        df.at[i,'spritepath_front'] = front
    #
#
def inRange(val,min,max):
    return (min <= val < max)
#
def hsvToColor(hsv_tuple):
    h,s,v = int(hsv_tuple[0])*2,int(hsv_tuple[1])/255.0,int(hsv_tuple[2])/255
    color = "None"
    if (v<.15): color = "Black"
    elif (s<.20):
        if (v>.70): color = "White"
        else: color = "Gray"
    elif inRange(h,0,20): 
        if (s<.70 and v<.40) or (s<.65 and v<.65): color = "Brown"
        else: color = "Red"
    elif inRange(h,20,40):
        if (s<.60): color = "White"
        elif (v<.50): color = "Brown"
        else: color = "Orange"
    elif inRange(h,40,65):
        if (s<.50): color = "White"
        else: color = "Yellow"
    elif inRange(h,65,175): color = "Green"
    elif inRange(h,175,260): color = "Blue"
    elif inRange(h,260,300): color = "Purple"
    elif inRange(h,300,340): color = "Pink"
    else: color = "Red"
    # print(h,color)
    return color
#
def rgbToColor(rgb_tuple): # TODO: Broken classifier.
    close = lambda x,y : abs(x-y) <= 30
    thresh_bw = 70
    color = "None"
    # eg. rgb_tuple = (2,44,300)
    r = rgb_tuple[0]
    g = rgb_tuple[1]
    b = rgb_tuple[2]
    if (close(r,g) and close(g,b) and close(r,b)):
        if (r<thresh_bw and g<thresh_bw and b<thresh_bw): color = "Black"
        elif (r>255-thresh_bw and g>255-thresh_bw and b>255-thresh_bw): color = "White"
        else: color = "Gray"
    else:
        colors = {
            "Red": (255,0,0),
            "Green": (0,255,0),
            "Green": (167,211,0),
            "Green": (80, 97, 41),
            "Green": (105, 140, 90),
            "Blue": (0,0,255),
            "Blue": (169, 131, 235),
            "Blue": (174, 233, 184),
            "Blue": (0,255,255),
            "Blue": (84, 168, 152),
            "Yellow": (255,255,0),
            "Yellow": (255, 214, 109),
            "Pink": (255,0,200),
            "Pink": (225, 97, 124),
            "Orange": (200,100,25),
            "Orange": (241, 98, 69),
            "Purple": (175,0,255),
            "Purple": (140, 72, 139),
            "Purple": (75, 40, 65),
            "Purple": (207, 95, 180),
            "Brown": (116, 74, 50),
            "Brown": (115, 75, 0),
            "Brown": (207, 131, 92),
            "Brown": (207, 156, 124),
            "Brown": (166, 98, 91),
            "Brown": (58, 41, 1),
        }
        manhattan = lambda x,y : abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2]) 
        distances = {k: manhattan(v, rgb_tuple) for k, v in colors.items()}
        color = min(distances, key=distances.get)
    return color
#

# def convertImgToHexCode(imgPath):
#     #Load image and get list of pixels
#     pic = Image.open(imgPath).convert("RGB")
#     pixelList = list(pic.getdata())

#     #Convert individual pixels to hexcode
#     hexcodes = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in pixelList]

#     #Count hexcode occurences
#     hexCodeCount = Counter(hexcodes)

#     #Sort frequency
#     sortedHexes = hexCodeCount.most_common()

#     #Return hexcode Counts
#     return sortedHexes
# #
def isPureWhite(hsv_tuple):
    return (hsv_tuple[0] == 0 and hsv_tuple[1] == 0 and hsv_tuple[2] == 255)
#
def convertImgToColors(imgPath):
    #Load image and get list of pixels
    img = cv2.imread(imgPath)
    pixelList = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imwrite('test.png', pixelList)
    
    #Convert individual pixels to colors -- REMOVE PURE WHITE BACKGROUND PIXELS
    colors = []
    for row in pixelList:
        for pixel in row:
            if not isPureWhite(pixel):
                colors.append(hsvToColor(pixel))

    #Count color occurences
    colorCount = Counter(colors)

    #Sort frequency
    sortedColors = colorCount.most_common()

    # Return color Counts
    return sortedColors
#
def discretizePixels(imgPath):
    img = cv2.imread(imgPath)
    pixelList = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixelListDiscrete = []
    for row in range(0,len(pixelList)):
        pixelListDiscrete.append([])
        for pixel in pixelList[row]:
            pixelListDiscrete[row].append(primary_colors[hsvToColor(pixel)])
    return pixelListDiscrete
#
def storeColorCounts(df, row, colorCounts):
    first = True
    for color, count in colorCounts:
        # Compare with primary_color roughly to test accuracy
        # if first:
        #     df.at[row,"best_color"] = color
        #     first = False
        # #
        df.at[row,color] = count
    #
#
# def populateHexCounts(df):
#     for i,entry in df.iterrows():
#         storeColorCounts(df, i, convertImgToHexCode(df.at[i,'spritepath_front']))
#     #
# #
def createDiscreteImage(img,name):
    discrete = np.array(discretizePixels(img))
    cv2.imwrite('discrete_images/'+name+'.png',discrete)
    # print("made image")
#
def populateColorCounts(df):
    for i,entry in df.iterrows():
        storeColorCounts(df, i, convertImgToColors(df.at[i,'spritepath_front']))
        # createDiscreteImage(df.at[i,'spritepath_front'],f"{entry['pokedex_id']:04d}"+'-'+df.at[i,'name'])
    #
#

#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here
df = pd.read_csv(CSV_FILEPATH)

# print(df['primary_color'].unique())

### Preprocessing
pd.set_option('display.max_rows', 10) # Default df display()
# pd.set_option('display.max_rows', 1000) # Modified df display()

# Remove entries we don't want to use 
df = df[df['image_fn'] != '[]']
df = df[df['gigantamax'] == False]
df = df[df['mega_evolution'] == False]
# Reset the index
df.reset_index(inplace=True, drop=True)
# Remove columns we aren't using
df = df.drop(['shape','legendary','mega_evolution','alolan_form','galarian_form','gigantamax','image_fn'], axis=1)
# Find the filepaths to each sprite we want to use
assign_spritepaths(df)

# Keep an original copy in case we want names of pokemon
df_original = df.copy()
### Visualize Data
# visualizeData(df_original)
# visualizeData(df_original, "data.png") # this is for if we want to save the graphs to a file

# Remove columns we won't be training on, Fix columns we will be using
df = df.drop(['id','name','pokedex_id','primary_color'], axis=1) # we won't need 'primary_color' later
populateColorCounts(df)
df = df.drop(['spritepath_front'],axis=1) # we don't need this anymore
df['type2'] = df['type2'].fillna("None") # do we need to do this??
df = df.fillna(0)

display(df)

# Export to CSV
# df.to_csv("pokedex_extracted.csv",index=False) # Keeping this in case we need to run this again

#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    
    #TEST Code
    main()