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
    'Red':[0,0,255],
    'Orange':[0,121,255],
    'Yellow':[0,255,255],
    'Green':[0,225,0],
    'Cyan':[255,255,0],
    'Blue':[255,0,0],
    'Purple':[255,0,180],
    'Pink':[255, 0, 255],
    'White':[255,255,255],
    'Gray':[125,125,125],
    'Black':[0,0,0],
    'Brown':[0,60,135],
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

def visualizeDataPreprocessed(df, save_name):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
    df_visualizer = df.drop(['id','name','pokedex_id',], axis=1)
    df_visualizer = df_visualizer.sort_values('type1', key=lambda s: s.apply(TYPES1.index), ignore_index=True)
    axs[0][0].set_title("type1")
    axs[0][0].bar(TYPES1,df_visualizer['type1'].value_counts(sort=False).to_numpy(),color=TYPES1_COLORS_CHART)
    df_visualizer = df_visualizer.sort_values('primary_color', key=lambda s: s.apply(COLORS.index), ignore_index=True)
    axs[1][0].set_title("primary_color")
    axs[1][0].bar(COLORS,df_visualizer['primary_color'].value_counts(sort=False).to_numpy(),color=COLORS_CHART)
    df_visualizer_nona = df_visualizer[df_visualizer.type2 != 'None'].sort_values('type2', key=lambda s: s.apply(TYPES1.index), ignore_index=True)
    axs[1][1].set_title("type2 removing mono-type pokemon ('None')")
    axs[1][1].bar(TYPES1,df_visualizer_nona['type2'].value_counts(sort=False).to_numpy(),color=TYPES1_COLORS_CHART)
    df_visualizer = df_visualizer.sort_values('type2', key=lambda s: s.apply(TYPES2.index), ignore_index=True)
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
def visualizeDataExtracted(df, save_name):
    pass # TODO: Maybe graph this bc it'd be cool to see.
#
def assign_spritepaths(df):
    root = "pokemon_images/sprites/"
    for i,entry in df.iterrows():
        front = root+f"{entry['id']:04d}"+'-'+entry['name']+'-'+str(entry['pokedex_id'])+'/front/normal'
        images = os.listdir(front)
        df.at[i,'spritepath'] = front + "/"+images[len(images)-1]
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
        if inRange(h,0,10) and s<.5 and v>.8: color = "Pink"
        elif (s<.70 and v<.40) or (s<.65 and v<.65): color = "Brown"
        else: color = "Red"
    elif inRange(h,20,40):
        if (s<.60): color = "White"
        elif (v<.50): color = "Brown"
        else: color = "Orange"
    elif inRange(h,40,65):
        if (s<.50): color = "White"
        else: color = "Yellow"
    elif inRange(h,65,155): color = "Green"
    elif inRange(h,155,200): color = "Cyan"
    elif inRange(h,200,260): color = "Blue"
    elif inRange(h,260,300): color = "Purple"
    elif inRange(h,300,340): color = "Pink"
    elif inRange(h,340,352):
        if (s<.65): color = "Pink"
        else: color = "Red"
    else:
        if s<.5 and v>.8: color = "Pink"
        else: color = "Red"
    return color
#
def isPureWhite(hsv_tuple):
    return (hsv_tuple[0] == 0 and hsv_tuple[1] == 0 and hsv_tuple[2] == 255)
#
def convertImgToColors(imgPath):
    #Load image and get list of pixels
    img = cv2.imread(imgPath)
    pixelList = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #Convert individual pixels to colors
    colors = []
    for row in pixelList:
        for pixel in row:
            if not isPureWhite(pixel): # bg is pure white, we don't want to include the bg
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
        df.at[row,color] = count
    #
#
def createDiscreteImage(from_path,to_path):
    cv2.imwrite(to_path,np.array(discretizePixels(from_path)))
#
def populateColorCounts(df,make_discrete=False):
    for i,entry in df.iterrows():
        storeColorCounts(df, i, convertImgToColors(df.at[i,'spritepath']))
        if (make_discrete):
            createDiscreteImage(df.at[i,'spritepath'],f"pokemon_images/sprites_discrete/{entry['pokedex_id']:04d}"+'-'+df.at[i,'name']+'.png')
        #
    #
#
def addProminentColor(df, visualize_data=False):
    df.insert(0, 'primary_color', df.apply('idxmax', axis=1))
#
def viewTypeColorAverages(df, visualize_data=False):
    dft1 = df.drop('type2',axis=1).rename(columns={'type1': 'type'})
    dft2 = df.drop('type1',axis=1).rename(columns={'type2': 'type'})
    dft = pd.concat([dft1,dft2])
    dft = dft.groupby(['type']).mean().apply(np.floor).sort_values('type', key=lambda s: s.apply(TYPES2.index))
    addProminentColor(dft)
    if visualize_data:
        visualizeDataExtracted(dft.reset_index(), 'data_extracted.png')
    display(dft)
#

#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here
df = pd.read_csv(CSV_FILEPATH)
# pd.set_option('display.max_rows', 10) # Default df display()
pd.set_option('display.max_rows', 20) # Modified df display()
# pd.set_option('display.max_rows', 1000) # Modified df display()

### Preprocessing
# Remove entries we don't want to use 
df = df[df['image_fn'] != '[]']
df = df[df['gigantamax'] == False]
df = df[df['mega_evolution'] == False]
# Reset the index
df.reset_index(inplace=True, drop=True)
# Remove columns we aren't using
df = df.drop(['shape','legendary','mega_evolution','alolan_form','galarian_form','gigantamax','image_fn'], axis=1)
# Fix up bad values
df['type2'] = df['type2'].fillna("None") # do we need to do this??
# Find the filepaths to each sprite we want to use
assign_spritepaths(df)

### Visualize Data
df_processed = df.copy() # Keep a copy in case we want names of pokemon
# visualizeDataPreprocessed(df_processed, "assets/data.png") # this is for if we want to save the graphs to a file
df_processed = df_processed.drop(['id','name','pokedex_id','primary_color'], axis=1) # we don't need these anymore
# df_processed.to_csv("pokedex_processed.csv",index=False) # Keeping this in case we need to run this again
# display(df_processed)

### Feature Extraction - Remove columns we won't be training on, Fix columns we will be using
populateColorCounts(df, True)
df = df.drop(['id','name','pokedex_id','primary_color','spritepath'], axis=1) # we don't need these anymore
df = df.fillna(0)
df = df.reindex(['type1','type2']+list(primary_colors.keys()), axis=1) # Sort columns

df_extracted = df.copy()
# df_extracted.to_csv("pokedex_extracted.csv",index=False) # Keeping this in case we need to run this again
# viewTypeColorAverages(df_extracted) # data of average colors for each type
# display(df_extracted)

# Show color palette
# createDiscreteImage('assets/hsv.png','assets/hsv_discrete_2.png')
# createDiscreteImage('assets/color-tester.png','assets/color-tester-discrete.png')

display(df)

#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    
    #TEST Code
    main()