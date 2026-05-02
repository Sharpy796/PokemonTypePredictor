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
import torch
import torch_directml
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import Dataset
# from torch import nn
from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from torchvision.transforms import ToTensor
# import scipy
# from scipy.stats import zscore
# from time import sleep
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score, f1_score, log_loss
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
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
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



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
# Type indices
TYPES1_TO_IDX = {t: i for i, t in enumerate(TYPES1)}
IDX_TO_TYPES1 = {i: t for i, t in enumerate(TYPES1)}
TYPES2_TO_IDX = {t: i for i, t in enumerate(TYPES2)}
IDX_TO_TYPES2 = {i: t for i, t in enumerate(TYPES2)}
# Come up with a type combo map so we can use them numerically in our data
PRIMARY_TYPE_TO_IDX = {}
TYPES_COMBO_TO_IDX,seen = {},set()
for type1 in TYPES1:
    for type2 in TYPES2:
        if type2 == "None": key = type1
        else:
            if type1 == type2: continue
            key = f"{type1}/{type2}"
            reverse = f"{type2}/{type1}"
            if reverse in seen:
                TYPES_COMBO_TO_IDX[key] = TYPES_COMBO_TO_IDX[reverse]
                continue
        if key not in seen:
            TYPES_COMBO_TO_IDX[key] = len(seen)
            seen.add(key)
TYPES1_COLORS_CHART = ['wheat','darkorange','skyblue','limegreen','yellow','paleturquoise','darkred','mediumorchid','khaki','thistle','deeppink','yellowgreen','goldenrod','mediumpurple','mediumblue','black','lightsteelblue','pink']
TYPES2_COLORS_CHART = ['wheat','darkorange','skyblue','limegreen','yellow','paleturquoise','darkred','mediumorchid','khaki','thistle','deeppink','yellowgreen','goldenrod','mediumpurple','mediumblue','black','lightsteelblue','pink','dimgray']

#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here



#Class definitions Start Here

class PrimaryTypeDataset(Dataset):
    def __init__(self, csv_file, train, train_val=.9, transform=None):
        self.__parsecsv__(csv_file)
        # Shuffle the data around so we don't get any bias in the order that they are listed
        self.df = self.df.sample(frac=1,random_state=796).reset_index(drop=True)

        # Set aside a section of our data for training or testing
        if train:
            self.df = self.df.iloc[1:int(self.df.shape[0]*train_val)+1].reset_index(drop=True)
        else:
            self.df = self.df.iloc[int(self.df.shape[0]*train_val):self.df.shape[0]].reset_index(drop=True)

        # Create a tensor of our color data
        df_numeric = self.df.drop('type1', axis=1)
        self.data_numeric = torch.tensor(df_numeric.values, dtype=torch.float32)
        df_target = self.df['type1'].map(TYPES1_TO_IDX).values
        self.data_targets = torch.tensor(df_target, dtype=torch.long)

        self.transform = transform
    
    def __parsecsv__(self, csv_file):
        self.df_original = pd.read_csv(csv_file)
        self.df = self.df_original.copy()
        self.df = self.df.drop(['type2'],axis=1)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (
            self.data_numeric[idx],
            self.data_targets[idx]
        )
    
    def __len__(self):
        return len(self.df)
    
    def getdf(self):
        return self.df
    
    def getdf_original(self):
        return self.df_original

class DualTypeDataset(Dataset):
    def __init__(self, csv_file, train, train_val=.9, transform=None):
        self.__parsecsv__(csv_file)
        # Shuffle the data around so we don't get any bias in the order that they are listed
        self.df = self.df.sample(frac=1,random_state=796).reset_index(drop=True)

        # Set aside a section of our data for training or testing
        if train:
            self.df = self.df.iloc[1:int(self.df.shape[0]*train_val)+1].reset_index(drop=True)
        else:
            self.df = self.df.iloc[int(self.df.shape[0]*train_val):self.df.shape[0]].reset_index(drop=True)

        # Create a tensor of our color data
        df_numeric = self.df.drop(['type1','type2'], axis=1)
        self.data_numeric = torch.tensor(df_numeric.values, dtype=torch.float32)
        df_target1 = self.df['type1'].map(TYPES1_TO_IDX).values
        df_target2 = self.df['type2'].map(TYPES2_TO_IDX).values
        self.data_target1 = torch.tensor(df_target1, dtype=torch.long)
        self.data_target2 = torch.tensor(df_target2, dtype=torch.long)

        self.transform = transform
    
    def __parsecsv__(self, csv_file):
        self.df_original = pd.read_csv(csv_file)
        self.df = self.df_original.copy()
        self.df['type2'] = self.df['type2'].fillna('None')
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (
            self.data_numeric[idx],
            (self.data_target1[idx],self.data_target2[idx])
        )
    
    def __len__(self):
        return len(self.df)
    
    def getdf(self):
        return self.df
    
    def getdf_original(self):
        return self.df_original

# TODO: Create the neural network
class PokemonTypePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_classes_2=0):
        super(PokemonTypePredictor,self).__init__()
        weight1 = 1.5
        weight2 = 2
        weight3 = 2
        
        self.l1 = nn.Linear(input_size,int(hidden_size*weight1)) # first layer
        self.act1 = nn.Sigmoid() # activation function
        self.l2 = nn.Linear(int(hidden_size*weight1),int(hidden_size*weight2)) # second layer
        self.act2 = nn.Sigmoid() # activation function
        self.l3 = nn.Linear(int(hidden_size*weight2),int(hidden_size*weight3)) # third layer
        self.act3 = nn.Sigmoid() # activation function
        self.l4_1 = nn.Linear(int(hidden_size*weight3),num_classes) # fourth layer
        if num_classes_2 > 0:
            self.l4_2 = nn.Linear(int(hidden_size*weight3),num_classes_2) # fourth layer
    
    def forward(self, numeric):
        x = numeric
        x = self.l1(x)
        x = self.act1(x)
        x = self.l2(x)
        x = self.act2(x)
        x = self.l3(x)
        x = self.act3(x)
        y1 = self.l4_1(x)
        try:
            y2 = self.l4_2(x)
            return y1,y2
        except:
            return y1


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
def create_confusion_matrix(y_true,y_pred,table,title="Pokemon Type Confusion Matrix", ax=None):
    cm = confusion_matrix(y_true=y_true,y_pred=y_pred,labels=list(table.values()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=list(table.keys()))
    oldcolors = mpl.colormaps['hsv']
    newcolors = oldcolors(np.linspace(0,.70, 128))
    val = 200
    grey = np.array([val/256, val/256, val/256, 1])
    newcolors[:1, :] = grey
    cmap = ListedColormap(newcolors)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    disp.plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title(title)

#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# #Main code start here
# # df = pd.read_csv(CSV_FILEPATH)
# # pd.set_option('display.max_rows', 10) # Default df display()
# # pd.set_option('display.max_rows', 20) # Modified df display()
# # pd.set_option('display.max_rows', 1000) # Modified df display()

# Hyperparameters
table1 = TYPES1_TO_IDX
table2 = TYPES2_TO_IDX
input_size = len(primary_colors) # column length (amount of colors)
hidden_size = 100 # number of nodes in hidden layer
num_classes = len(table1) # number of classes, Normal, Fire/Normal, Water/Normal, etc
num_classes_2 = len(table2) # number of classes, Normal, Fire/Normal, Water/Normal, etc
num_epochs = 100 # number of times we go through the entire dataset
batch_size = 64 # number of samples in one forward/backward pass
learning_rate = 0.001 # learning rate
train_val = .7

# Datasets & Dataloaders
# train_dataset = PrimaryTypeDataset("pokedex_extracted.csv",train=True,train_val=train_val)
# test_dataset = PrimaryTypeDataset("pokedex_extracted.csv",train=False,train_val=train_val)
train_dataset = DualTypeDataset("pokedex_extracted.csv",train=True,train_val=train_val)
test_dataset = DualTypeDataset("pokedex_extracted.csv",train=False,train_val=train_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# TODO: Figure out the "verification" part

# Model
model = PokemonTypePredictor(input_size, hidden_size, num_classes, num_classes_2=num_classes_2).to(device)
criterion_1 = nn.CrossEntropyLoss()
criterion_2 = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%% TRAINING
model.train()
n_total_steps = len(train_loader)
loss_values = []
i = 0
loss = 0
alpha = 0.4
beta = 1-alpha
pbar = tqdm(range(num_epochs),desc="Training",unit="epochs",postfix={"loss":loss},colour="red",ascii=True)
for epoch in pbar:
    running_loss = 0.0
    for i, (numeric,target) in enumerate(train_loader):

        # --- CRITICAL STEP: Move all input tensors to the GPU ---
        numeric = numeric.to(device)
        if (isinstance(target, (list, tuple))):
            target1 = target[0].to(device)
            target2 = target[1].to(device)
        else:
            target = target.to(device)
        
        outputs = model(numeric)
        if (isinstance(target, (list, tuple))):
            loss1 = criterion_1(outputs[0],target1)
            loss2 = criterion_2(outputs[1],target2)
            loss = loss1*alpha + loss2*beta
        else:
            loss = criterion_1(outputs[0],target)
        # record loss for graphing loss function
        running_loss += loss.item()
        loss_values.append(running_loss / len(train_dataset))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % (2) == 0:
            pbar.set_postfix({"step":f'{(i+1):02d}/{n_total_steps}',"loss":f'{loss.item():.4}'})

print("Finished training.")

plt.plot(loss_values)

#%% TESTING
with torch.no_grad(): # we don't need gradients in the testing phase
    y_true1, y_true2 = [], []
    y_pred1, y_pred2 = [], []
    cm = None
    for numeric,target in test_loader:
    # for numeric,targets in test_loader:

        # --- CRITICAL STEP: Move all input tensors to the GPU ---
        numeric = numeric.to(device)
        if (isinstance(target, (list, tuple))):
            target1 = target[0].to(device)
            target2 = target[1].to(device)
        else:
            target1 = target.to(device)

        outputs = model(numeric)

        if (isinstance(target, (list, tuple))):
            _, predictions_1 = torch.max(outputs[0],1) # 1 is the dimension
            _, predictions_2 = torch.max(outputs[1],1) # 1 is the dimension
            y_true2 = np.concatenate([y_true2, target2.cpu().numpy()])
            y_pred2 = np.concatenate([y_pred2, predictions_2.cpu().numpy()])
        else:
            _, predictions_1 = torch.max(outputs[0],1) # 1 is the dimension

        y_true1 = np.concatenate([y_true1, target1.cpu().numpy()])
        y_pred1 = np.concatenate([y_pred1, predictions_1.cpu().numpy()])
    
    # Calculate accuracy
    n_correct_1 = (y_true1 == y_pred1).sum().item()
    n_samples_1 = y_true1.shape[0]
    avg = 'macro'
    print("=== Type 1 ===")
    print(f'Accuracy:\t{accuracy_score(y_true1,y_pred1)*100:.2f}% ({n_correct_1}/{n_samples_1})')
    print(f'Recall:\t\t{recall_score(y_true1,y_pred1,average=avg)*100:.2f}%')
    print(f'Precision:\t{precision_score(y_true1,y_pred1,average=avg)*100:.2f}%')
    print(f'F1 Score:\t{f1_score(y_true1,y_pred1,average=avg)*100:.2f}%')

    if len(y_true2)>0:
        n_correct_2 = (y_true2 == y_pred2).sum().item()
        n_samples_2 = y_true2.shape[0]
        print("=== Type 2 ===")
        print(f'Accuracy:\t{accuracy_score(y_true2,y_pred2)*100:.2f}% ({n_correct_2}/{n_samples_2})')
        print(f'Recall:\t\t{recall_score(y_true2,y_pred2,average=avg)*100:.2f}%')
        print(f'Precision:\t{precision_score(y_true2,y_pred2,average=avg)*100:.2f}%')
        print(f'F1 Score:\t{f1_score(y_true2,y_pred2,average=avg)*100:.2f}%')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        create_confusion_matrix(y_true1, y_pred1, table1, "Primary Type Confusion Matrix",  ax=ax1)
        create_confusion_matrix(y_true2, y_pred2, table2, "Secondary Type Confusion Matrix", ax=ax2)
    else:
        create_confusion_matrix(y_true1, y_pred1, table1, "Primary Type Confusion Matrix")
    plt.tight_layout()
    plt.show()

#%% Processing
# ### Preprocessing
# # Remove entries we don't want to use 
# df = df[df['image_fn'] != '[]']
# df = df[df['gigantamax'] == False]
# df = df[df['mega_evolution'] == False]
# # Reset the index
# df.reset_index(inplace=True, drop=True)
# # Remove columns we aren't using
# df = df.drop(['shape','legendary','mega_evolution','alolan_form','galarian_form','gigantamax','image_fn'], axis=1)
# # Fix up bad values
# df['type2'] = df['type2'].fillna("None") # do we need to do this??
# # Find the filepaths to each sprite we want to use
# assign_spritepaths(df)

# ### Visualize Data
# df_processed = df.copy() # Keep a copy in case we want names of pokemon
# # visualizeDataPreprocessed(df_processed, "assets/data.png") # this is for if we want to save the graphs to a file
# df_processed = df_processed.drop(['id','name','pokedex_id','primary_color'], axis=1) # we don't need these anymore
# # df_processed.to_csv("pokedex_processed.csv",index=False) # Keeping this in case we need to run this again
# # display(df_processed)

# ### Feature Extraction - Remove columns we won't be training on, Fix columns we will be using
# populateColorCounts(df, False) # change this to True if you want to save edited sprites to a file
# df = df.drop(['id','name','pokedex_id','primary_color','spritepath'], axis=1) # we don't need these anymore
# df = df.fillna(0)
# df = df.reindex(['type1','type2']+list(primary_colors.keys()), axis=1) # Sort columns

# df_extracted = df.copy()
# # df_extracted.to_csv("pokedex_extracted.csv",index=False) # Keeping this in case we need to run this again
# # viewTypeColorAverages(df_extracted) # data of average colors for each type
# # display(df_extracted)

# # Show color palette
# # createDiscreteImage('assets/hsv.png','assets/hsv_discrete_2.png')
# # createDiscreteImage('assets/color-tester.png','assets/color-tester-discrete.png')

# # display(df)

#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    
    #TEST Code
    main()