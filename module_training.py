#Version: v0.1
#Date Last Updated: 12-20-2023

#%% MODULE BEGINS
module_name = 'training'

'''
Version: 1.0.0

Description:
    Trains a neural network on processed
    pokedex data to predict a pokemon's
    type based purely on its sprite colors.

Authors:
    Christian Bankovic
    Ethan Cochran
    Davidson Rock

Date Created     :  3/28/26
Date Last Updated:  5/06/26

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

#custom imports


#other imports
from copy import deepcopy as dpcpy
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score, f1_score, log_loss
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
from warnings import simplefilter, filterwarnings
from sklearn.exceptions import UndefinedMetricWarning
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
filterwarnings("ignore", category=UndefinedMetricWarning)

#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CSV_FILEPATH = "training_data/pokedex_extracted.csv"

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

TYPES1 = ['Normal','Fire','Water','Grass','Electric','Ice','Fighting','Poison','Ground','Flying','Psychic','Bug','Rock','Ghost','Dragon','Dark','Steel','Fairy']
TYPES2 = ['Normal','Fire','Water','Grass','Electric','Ice','Fighting','Poison','Ground','Flying','Psychic','Bug','Rock','Ghost','Dragon','Dark','Steel','Fairy','None']
# Type indices
TYPES1_TO_IDX = {t: i for i, t in enumerate(TYPES1)}
IDX_TO_TYPES1 = {i: t for i, t in enumerate(TYPES1)}
TYPES2_TO_IDX = {t: i for i, t in enumerate(TYPES2)}
IDX_TO_TYPES2 = {i: t for i, t in enumerate(TYPES2)}

# Hyperparameters
table1 = TYPES1_TO_IDX
table2 = TYPES2_TO_IDX
input_size = len(primary_colors) # column length (amount of colors)
hidden_size = 100 # number of nodes in hidden layer
num_classes = len(table1) # number of classes, Normal, Fire/Normal, Water/Normal, etc
num_classes_2 = len(table2) # number of classes, Normal, Fire/Normal, Water/Normal, etc
num_epochs = 100 # number of times we go through the entire dataset
batch_size = 64 # number of samples in one forward/backward pass
learning_rate = 0.001/2 # learning rate
train_val = .7
dropout_p = 0.2

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
            self.save_csv_filepath = 'training_data/pokedex_training.csv'
        else:
            self.df = self.df.iloc[int(self.df.shape[0]*train_val):self.df.shape[0]].reset_index(drop=True)
            self.save_csv_filepath = 'training_data/pokedex_testing.csv'
        self.df.to_csv(self.save_csv_filepath)

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
            self.save_csv_filepath = 'training_data/pokedex_training.csv'
        else:
            self.df = self.df.iloc[int(self.df.shape[0]*train_val):self.df.shape[0]].reset_index(drop=True)
            self.save_csv_filepath = 'training_data/pokedex_testing.csv'
        self.df.to_csv(self.save_csv_filepath)

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

class PokemonTypePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_classes_2=0,dropout_p=0.3):
        super(PokemonTypePredictor,self).__init__()
        weight1 = 1.5
        weight2 = 2
        weight3 = 2
        weight3_5 = 4
        
        self.l1 = nn.Linear(input_size,int(hidden_size*weight1)) # first layer
        self.act1 = nn.ReLU() # activation function
        self.drop1 = nn.Dropout(p=dropout_p)
        self.l2 = nn.Linear(int(hidden_size*weight1),int(hidden_size*weight2)) # second layer
        self.act2 = nn.ReLU() # activation function
        self.drop2 = nn.Dropout(p=dropout_p)
        self.l3 = nn.Linear(int(hidden_size*weight2),int(hidden_size*weight3)) # third layer
        self.act3 = nn.ReLU() # activation function
        self.drop3 = nn.Dropout(p=dropout_p)
        # new stuff
        self.l3_2 = nn.Linear(int(hidden_size*weight3),int(hidden_size*weight3_5)) # another layer
        self.act3_2 = nn.ReLU() # activation function
        self.drop3_2 = nn.Dropout(p=dropout_p)
        # new stuff
        self.l4_1 = nn.Linear(int(hidden_size*weight3_5),num_classes) # fourth layer
        if num_classes_2 > 0:
            self.l4_2 = nn.Linear(int(hidden_size*weight3_5),num_classes_2) # fourth layer
    
    def forward(self, numeric):
        x = numeric
        x = self.drop1(self.act1(self.l1(x)))
        x = self.drop2(self.act2(self.l2(x)))
        x = self.drop3(self.act3(self.l3(x)))
        x = self.drop3_2(self.act3_2(self.l3_2(x)))
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

def create_confusion_matrix(y_true,y_pred,table,title="Pokemon Type Confusion Matrix", ax=None):
    cm = confusion_matrix(y_true=y_true,y_pred=y_pred,labels=list(table.values()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=list(table.keys()))
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
#

#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Datasets & Dataloaders
# TODO: Add weights for various types to improve accuracy
train_dataset = DualTypeDataset(CSV_FILEPATH,train=True,train_val=train_val,)
test_dataset = DualTypeDataset(CSV_FILEPATH,train=False,train_val=train_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Model
model = PokemonTypePredictor(input_size, hidden_size, num_classes, num_classes_2=num_classes_2,dropout_p=dropout_p).to(device)
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
best_loss = float('inf')
patience, patience_counter = 10, 0
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
    if running_loss < best_loss:
        best_loss = running_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break


print("Finished training.")

plt.plot(loss_values)

#%% TESTING
model.eval()
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

#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    
    #TEST Code
    main()