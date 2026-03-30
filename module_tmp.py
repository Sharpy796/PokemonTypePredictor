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

def assign_spritepaths(df):
    df_copy = df.copy()
    df_filepaths = pd.DataFrame(columns=['spritepath_front','spritepath_back'])
    root = "pokemon_images/sprites/"
    new_row = {}
    for i,entry in df_copy.iterrows():
        new_row = {'spritepath_front': root+f"{entry['id']:04d}"+'-'+entry['name']+'-'+str(entry['pokedex_id'])+'/front/normal',
                   'spritepath_back': root+f"{entry['id']:04d}"+'-'+entry['name']+'-'+str(entry['pokedex_id'])+'/back/normal',}
        df_filepaths.loc[len(df_filepaths)] = new_row
    #
    df_copy = pd.concat([df_copy, df_filepaths], sort=False, axis=1)
    return df_copy
#

#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here
df = pd.read_csv(CSV_FILEPATH)

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
df = assign_spritepaths(df)
# df['type2'] = df['type2'].fillna("None")

# Keep an original copy in case we want names of pokemon
df_original = df.copy()
# Remove columns we won't be training on
df = df.drop(['id','name','pokedex_id','primary_color'], axis=1) # we won't need 'primary_color' later
# Export to CSV
df.to_csv("pokedex_processed.csv",index=False)

# Visualize Data
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
df_visualizer = df_original.drop(['id','name','pokedex_id',], axis=1)
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

#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    
    #TEST Code
    main()