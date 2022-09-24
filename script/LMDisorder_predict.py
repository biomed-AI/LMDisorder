import os
import math
from re import I
import torch
import os, datetime, argparse, re
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, accuracy_score,precision_recall_curve, average_precision_score
from scipy.stats import pearsonr
import csv
import logging
import os
import sys
import MODEL


## parameters
BATCH_SIZE = 1
FEATURE_DIM = 1024
HIDDEN_DIM = 128
NUM_ENCODER_LAYERS = 2
NUM_HEADS = 4
AUGMENT_EPS = 0.05
DROPOUT = 0.3

model_bucket = ['model.pkl']

# set_csv('../example/demo.fasta')
def set_csv(fasta_file):
    with open(fasta_file,'r') as reader:
        data_fasta = reader.readlines()
        name = []
        sequence = []
        for item in data_fasta:
            if item[0] == ">":
                name.append(item[1:])
            else:
                sequence.append(item)
    csv_writer = csv.writer(open('../example/demo.csv','w')) 
    csv_writer.writerow(["name","sequence"])
    for i in range(len(name)):
        csv_writer.writerow([name[i],sequence[i]])

def load_features(sequence_name):
    return np.load("../example/ProtTrans/"+f"{sequence_name}.npy")

class ProDataset(Dataset):
    
    def __init__(self, dataframe):
        self.names = dataframe['name'].values
        self.sequences = dataframe['sequence'].values

    def __getitem__(self, index):
        sequence_name = self.names[index].replace("\n","")
        sequence = self.sequences[index]
        node_features = load_features(sequence_name)
        
        return sequence_name, sequence, node_features

    def __len__(self):
        return len(self.sequences)

def evaluate(model, data_loader,device):
    model.eval()
    valid_pred = []

    for data in data_loader:
        with torch.no_grad():
            """
            node_features: [b,L,d]
            feature_mask: [b,L]
            label: [b,L]
            """
            sequence_names, sequence_eva, node_features = data

            if torch.cuda.is_available() and device == "gpu":
                features = Variable(node_features.cuda())
                # feature_mask = Variable(feature_mask.cuda())
            else:
                features = Variable(node_features)
                # feature_mask = Variable(feature_mask)
            y_pred = model(features, None) # y_pred:[b,L]
            y_pred = y_pred.sigmoid()
            np.save(f'../example/result/{sequence_names[0]}',y_pred)

def LMDisorder(test_dataframe,device,Path_Model):
    data_loader = DataLoader(dataset=ProDataset(test_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=0,drop_last=False)
    for model_name in model_bucket:
        print(model_name)
        Model = MODEL.LMDisorder(feature_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM, num_encoder_layers=NUM_ENCODER_LAYERS, num_heads=NUM_HEADS, augment_eps=0.05, dropout=DROPOUT)
        if torch.cuda.is_available() and device == 'gpu':
            Model.cuda()
        
        Model.load_state_dict(torch.load(Path_Model))

        evaluate(Model, data_loader,device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type = str, help = "Input fasta file")
    parser.add_argument("--device", type = str, help = "Use GPU for feature extraction and LMDisorder prediction")
    parser.add_argument("--model_path", type = str, help = "The path of model")

    args = parser.parse_args()
   
    set_csv(args.fasta)
    device = args.device
    Path_Model = args.model_path

    dataframe = pd.read_csv("../example/demo.csv", sep=',')
    LMDisorder(dataframe,device,Path_Model)
       