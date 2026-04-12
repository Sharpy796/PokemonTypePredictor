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
#import cv2 as cv
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
primary_colors = [
    'Green',
    'Red',
    'Blue',
    'White',
    'Brown',
    'Yellow',
    'Purple',
    'Pink',
    'Gray',
    'Black'
]

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
            front += "/"+images[0]
        else:
            front = "NONE"
        df.at[i,'spritepath_front'] = front
    #
#
def rgbToColor(rgb_tuple): # TODO: Broken classifier.
    # eg. rgb_tuple = (2,44,300)

    colors = {
        "red": (255,0,0),
        "green": (0,255,0),
        "blue": (0,0,255),
        "white": (255,255,255),
        # "grey": (125,125,125),
        "black": (0,0,0),
        "yellow": (255,255,0),
        "pink": (255,0,255),
        "purple": (125,0,255),
        "orange": (255,125,0),
        # "brown": (150, 75, 0),
    }

    manhattan = lambda x,y : abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2]) 
    distances = {k: manhattan(v, rgb_tuple) for k, v in colors.items()}
    color = min(distances, key=distances.get)
    return color
#

def convertImgToHexCode(imgPath):
    #Load image and get list of pixels
    pic = Image.open(imgPath).convert("RGB")
    pixelList = list(pic.getdata())

    #Convert individual pixels to hexcode
    hexcodes = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in pixelList]

    #Count hexcode occurences
    hexCodeCount = Counter(hexcodes)

    #Sort frequency
    sortedHexes = hexCodeCount.most_common()

    #Return hexcode Counts
    return sortedHexes
#
def convertImgToColors(imgPath):
    #Load image and get list of pixels
    pic = Image.open(imgPath).convert("RGB")
    pixelList = list(pic.getdata())

    #Convert individual pixels to colors -- REMOVE PURE WHITE BACKGROUND PIXELS
    colors = [rgbToColor((r,g,b)) for r,g,b in pixelList if not (r==255 and g==255 and b==255)]

    #Count color occurences
    colorCount = Counter(colors)

    #Sort frequency
    sortedColors = colorCount.most_common()

    #Return color Counts
    return sortedColors
#
def storeColorCounts(df, row, colorCounts):
    first = True
    for color, count in colorCounts:
        if first:
            df.at[row,"primary_color"] = color
            first = False
        df.at[row,color] = count
    #
#
def populateHexCounts(df):
    for i,entry in df.iterrows():
        storeColorCounts(df, i, convertImgToHexCode(df.at[i,'spritepath_front']))
    #
#
def populateColorCounts(df):
    for i,entry in df.iterrows():
        storeColorCounts(df, i, convertImgToColors(df.at[i,'spritepath_front']))
    #
#

#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here
df = pd.read_csv(CSV_FILEPATH)

# print(df['primary_color'].unique())

### Preprocessing
# pd.set_option('display.max_rows', 10) # Default df display()
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
df['type2'] = df['type2'].fillna("None") # do we need to do this??
df = df.fillna(0)

# convertImgToColors('pokemon_images/sprites/0000-Bulbasaur-1/front/normal/1-gen3_e-frame2.png')
display(df)

# Export to CSV
# df.to_csv("pokedex_extracted.csv",index=False) # TODO: Run this once our classifier is fixed.

#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    
    #TEST Code
    main()