# -*- coding: utf-8 -*-
"""
@author: Mathias
"""

#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd 
import subprocess
import itertools

#Silence Tf logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from sklearn.metrics import roc_auc_score

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


seed=15
peptide_models = ['GILGFVFTL', 'RAKFKQLL', 'KLGGALQAK', 'AVFDRKSDAK', 'ELAGIGILTV',
                  'NLVPMVATV', 'IVTDFSVIK', 'LLWNGPMAV', 'CINGVCWTV', 'GLCTLVAML',
                  'SPRWYFYYL', 'ATDALMTGF', 'DATYQRTRALVR', 'KSKRTPMGF', 'YLQPRTFLL',
                  'HPVTKYIM', 'RFPLTFGWCF', 'GPRLGVRAT', 'CTELKLSDY', 'RLRAEAQVK',
                  'RLPGVLPRA', 'SLFNTVATLY', 'RPPIFIRRL', 'FEDLRLLSF', 'VLFGLGFAI',
                  'FEDLRVLSF']
##################
###USE SRC PATH###
##################

#


import keras_utils
import time, random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", help="Specify the directory where the GitHub repository has been downloaded to")
parser.add_argument("-i", "--infile", help="Specify input file with peptide and all six CDR sequences")
parser.add_argument("-o", "--output_dir",  help="Specify the output folder, where temporary files are saved as well as the prediction file")
parser.add_argument("--output_file", default = "nettcr_predictions.csv", help="Specify the name of the output file")
parser.add_argument("-a", "--alpha", default = 10, help="Determines how much the final predictions takes similarity to known binders into account via TCRbase.\nThe final prediction score is given by pred = CNN_pred*TCRbase_pred^alpha. An alpha of 0 disables TCRbase scaling")
parser.add_argument("-t", "--threshold", default = 100, help="Used to filter away shown predictions with a percentile rank above the specified threshold. Only affects which predictions are shown, as all predictions are still saved to the output file.")
args = parser.parse_args()

infile = str(args.infile)
alpha = int(args.alpha)
threshold = int(args.threshold)
input_dir = str(args.dir)
output_dir = str(args.output_dir)
output_file = str(args.output_file)

blf_path = "{}/tbcr_align/data/BLOSUM50".format(input_dir)
blqij_path = "{}/tbcr_align/data/blosum62.qij".format(input_dir)

sys.path.append("{}/src".format(input_dir))
sys.path.append("{}/models".format(input_dir))

### Model parameters ###
train_parts = {0, 1, 2, 3, 4} #Partitions
encoding = keras_utils.blosum50_20aa #Encoding for amino acid sequences

#Padding to certain length
a1_max = 7
a2_max = 8
a3_max = 22
b1_max = 6
b2_max = 7
b3_max = 23
pep_max = 12


# Set random seed
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
tf.random.set_seed(seed) # Tensorflow random seed

### Input/Output ###
#Check number of columns in input
with open(infile, 'r') as csv:
     line = csv.readline()
     n_columns = len(line.split(","))
     
     if n_columns < 7:
         print("WARNING: Columns are missing in the input file\nMake sure that the input file contains the following columns in the correct order (comma-separated & without headers): 'peptide','CDR1α', 'CDR2α', 'CDR3α', 'CDR1β', 'CDR2β', 'CDR3β'\n")
         sys.exit()

# Read in data
full_data = pd.read_csv(infile, index_col = False, header = None)

#Naming of columns. The first 7 columns are expected to be the input to the model. If 8 columns are present, the 8th column is assumed to be the target label (binder). All extra columns are given arbitrary names
if n_columns == 7:
    full_data.columns = ["peptide", "A1", "A2", "A3", "B1", "B2", "B3"]   
elif n_columns == 8:
    full_data.columns = ["peptide", "A1", "A2", "A3", "B1", "B2", "B3", "binder"] 
else:
    full_data.columns = ["peptide", "A1", "A2", "A3", "B1", "B2", "B3", "binder"] + full_data.columns.tolist()[8:]   

def check_input(x):
    """Ensures that the input only consists of standard amino acids"""
    aa_set = set(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])
    flag = set(x).issubset(aa_set)
    return flag

#Loop over input to check input requirements in terms of accepted characters
i = 0
for feature in ["peptide", "A1", "A2", "A3", "B1", "B2", "B3"]:
    full_data[feature] = full_data[feature].str.upper()
    if i == 0:
        accepted_idx = full_data[feature].apply(check_input)
        i += 1
    else:
        accepted_idx = accepted_idx * full_data[feature].apply(check_input) 
        
#Limit observations to those with correct input, and print skipped observations
if full_data[~accepted_idx].shape[0] != 0:
    print("The following observations were discard from predictions, due to unconventional amino acids:\n")
    print(full_data[~accepted_idx], end = "\n\n")
full_data = full_data[accepted_idx]
    
#Check that sequences are not longer than the maximum allowed length
for feature, max_len in zip(["peptide", "A1", "A2", "A3", "B1", "B2", "B3"],[pep_max, a1_max, a2_max, a3_max, b1_max, b2_max, b3_max]): 
    if max_len < np.max(full_data[feature].apply(len)):
        print("WARNING: A {} sequence in the input data was found to be longer than the maxmimum allowed length for that feature (Max = {} amino acids)".format(feature, max_len))
        sys.exit()
        
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
def get_similarity_df():
    """Calculates the BLOSUM62 kernel similarity between the peptides in the input data,
    and the peptides, which the models were trained on. The maximum similarity to train,
    as well as the most similar peptide, is then returned per peptide"""
    
    #Run kernel similarity
    sim_df = subprocess.run("{}/src/pairlistscore_kernel {}/peplist -blf {} -blqij {}".format(input_dir, output_dir, blf_path, blqij_path), shell = True, capture_output = True, check = False, text = True)

    #Prepare for DataFrame
    sim_df = sim_df.stdout.split("\n")
    sim_df = [x for x in sim_df if x.startswith("Sim")]
    sim_df = [x.split("Sim ")[1] for x in sim_df]   
    peptide_1 = []
    peptide_2 = []
    similarity = []
    
    for line in sim_df:
        line = line.split(" ")
        peptide_1.append(line[0])
        peptide_2.append(line[1])
        similarity.append(line[2])
        
    sim_df = pd.DataFrame({"Peptide_1": peptide_1,
                           "Peptide_2": peptide_2,
                           "Similarity": similarity})
    
    sim_df["Similarity"] = sim_df["Similarity"].apply(float)
    
    similarity_summary = pd.DataFrame({"Peptide": sim_df.Peptide_1.unique()})
    
    #Prepare max similarity calculation
    max_similarity = []
    best_hit = []
    for pep in sim_df.Peptide_1.unique():
        max_similarity.append(np.max(sim_df[sim_df.Peptide_1 == pep]["Similarity"]))
        best_hit_id = np.argmax(sim_df[sim_df.Peptide_1 == pep]["Similarity"])
        best_hit.append(sim_df.Peptide_2.unique()[best_hit_id])
    similarity_summary["max_similarity_to_train"] = max_similarity
    similarity_summary["most_similar_peptide"] = best_hit
    similarity_summary.set_index("Peptide", inplace = True)
    
    #Print out DataFrame to stdout
    print("\n{}\n".format(similarity_summary))

    return similarity_summary

def get_percentile_rank(x, control):
    """Calculates the percentile rank by comparing prediction scores to
    negative controls for the given peptide"""
    rank = round((control>x).mean()*100, 3)
    return rank

def get_tcrbase_prediction(df, pep):
    """Runs TCRbase on the input data for a given peptide, and returns the prediction scores"""
    weights = "113113"
    weights_sum = np.sum([int(x) for x in weights])
    
    train_data = pd.read_csv('{}/data/nettcr_2_2_limited_dataset.csv'.format(input_dir))
    test_data = df.copy(deep = True)
    
    train_data["index"] = train_data.index
    test_data["index"] = test_data.index
        
    tcrbase = '{}/tbcr_align/src/tbcr_align'.format(input_dir)
    blqij = '{}/tbcr_align/data/blosum62.qij'.format(input_dir)

    if not os.path.exists("{}/{}".format(output_dir,pep)):
        os.makedirs("{}/{}".format(output_dir,pep))
        
    train_df = train_data[train_data.peptide == pep]
    train_df = train_df[train_df.binder == 1]
    
    test_df = test_data[test_data.peptide == pep].copy(deep = True)
        
    train_df[["index", "A1", "A2", "A3", "B1", "B2", "B3"]].to_csv("{}/{}/pos_db".format(output_dir,pep), header = None, index = False, sep = " ")
    test_df[["index", "A1", "A2", "A3", "B1", "B2", "B3"]].to_csv("{}/{}/test".format(output_dir,pep), header = None, index = False, sep = " ")
    
    pos_db = "{}/{}/pos_db".format(output_dir,pep)
    test = "{}/{}/test".format(output_dir,pep)
    subprocess.run('{} -db {} -w {} -blqij {} {} > {}/{}/prediction'.format(tcrbase, pos_db, ",".join(weights), blqij, test, output_dir, pep), shell = True, text = True, check = True)

    infile = open('{}/{}/prediction'.format(output_dir,pep), mode = "r")
    
    all_lines = [x for x in infile.readlines() if x.startswith("#") is False]
    
    infile.close()
    
    #Clean up temporary files
    os.remove('{}/{}/prediction'.format(output_dir,pep))
    os.remove('{}/{}/pos_db'.format(output_dir,pep))
    os.remove('{}/{}/test'.format(output_dir,pep))

    all_idx = [int(float(x.split(" ")[2])) for x in all_lines]
    pep_df = pd.DataFrame({"index" : all_idx,
                           "peptide": pep})
    
    all_predictions = [round(float(x.split(" ")[-2])/weights_sum, 5) for x in all_lines]
    pep_df["prediction"] = all_predictions

    #pep_df = pep_df.set_index("index").sort_index()
    #pep_df.to_csv('{}/{}/TCRbase_prediction.csv'.format(output_dir, pep), index = False)
    
    tcrbase_prediction = pep_df.prediction.values
    
    return tcrbase_prediction

def my_numpy_function(y_true, y_pred):
    """Implementation of AUC 0.1 metric for Tensorflow"""
    try:
        auc = roc_auc_score(y_true, y_pred, max_fpr = 0.1)
    except ValueError:
        #Case when only one class exists
        auc = np.array([float(0)])
    return auc
    
def auc_01(y_true, y_pred):
    """Converts function to optimised tensorflow numpy function"""
    auc_01 = tf.numpy_function(my_numpy_function, [y_true, y_pred], tf.float64)
    return auc_01
     
def make_tf_ds(df, encoding = keras_utils.blosum50_20aa):
    """Encodes amino acid sequences using a BLOSUM50 matrix with a normalization factor of 5.
    Sequences are right-padded with [-1x20] for each AA missing, compared to the maximum embedding 
    length for that given feature
    
    Additionally, the input is prepared for predictions, by loading the data into a list of numpy arrays"""
    
    encoded_pep = keras_utils.enc_list_bl_max_len(df.peptide, encoding, pep_max)/5
    encoded_a1 = keras_utils.enc_list_bl_max_len(df.A1, encoding, a1_max)/5
    encoded_a2 = keras_utils.enc_list_bl_max_len(df.A2, encoding, a2_max)/5
    encoded_a3 = keras_utils.enc_list_bl_max_len(df.A3, encoding, a3_max)/5
    encoded_b1 = keras_utils.enc_list_bl_max_len(df.B1, encoding, b1_max)/5
    encoded_b2 = keras_utils.enc_list_bl_max_len(df.B2, encoding, b2_max)/5
    encoded_b3 = keras_utils.enc_list_bl_max_len(df.B3, encoding, b3_max)/5
    tf_ds = [np.float32(encoded_pep),
             np.float32(encoded_a1), np.float32(encoded_a2), np.float32(encoded_a3), 
             np.float32(encoded_b1), np.float32(encoded_b2), np.float32(encoded_b3)]


    return tf_ds
    
### Get peptides in input file ###
pep_list = list(full_data.peptide.value_counts(ascending=False).index)

###Calculating similarity to train###
pep_1_list = []
pep_2_list = []
for pep_1 in pep_list:
    for pep_2 in peptide_models:
        pep_1_list.append(pep_1)
        pep_2_list.append(pep_2)        
sim_df = pd.DataFrame({"Peptide_1": pep_1_list,
                       "Peptide_2": pep_2_list})
sim_df.to_csv("{}/peplist".format(output_dir), sep = " ", header = False, index = False)

#Allow for creation of file
time.sleep(0.1)

#Get similarity to training peptides
print("The maximum similarity of each peptide to the peptides in the training data is reported here\nPredictions for peptides with low similarity to the training (<95%) are particularly volatile\n")
similarity_df = get_similarity_df()

#Prepare output DataFrame (test predictions)
full_pred_df = pd.DataFrame()

#Necessary to load the model with the custom metric
dependencies = {
    'auc_01': auc_01
}

#Predictions
for pep in pep_list:
    time_start = time.time()
    pred_df = full_data[full_data.peptide == pep].copy(deep = True)
    test_tensor = make_tf_ds(pred_df, encoding = encoding)
    
    #Flag for scaling with TCRbase
    scale_prediction = False
    
    #Used for announcing that a model does not exist for the given peptide
    print_flag = 0
    
    print("Making predictions for {}".format(pep), end = "")
    if alpha != 0:
        if pep in peptide_models:
            scale_prediction = True
    avg_prediction = 0
    for t in train_parts:
        x_test = test_tensor[0:7]
        
        for v in train_parts:
            if v!=t:      
                if pep in peptide_models:
                    # Load the TFLite model and allocate tensors.
                    interpreter = tf.lite.Interpreter(model_path = "{}/models/nettcr_2_2_pretrained".format(input_dir)+'/'+pep+'/checkpoint/'+'t.'+str(t)+'.v.'+str(v)+".tflite")
                else:
                    print_flag += 1
                    # Load pan-specific TFLite model and allocate tensors.
                    interpreter = tf.lite.Interpreter(model_path = "{}/models/nettcr_2_2_pan".format(input_dir)+'/checkpoint/'+'t.'+str(t)+'.v.'+str(v)+".tflite")
                    if print_flag == 1:
                        print(". WARNING: A model for {} does not exist. Using pan-specific model instead ".format(pep), end = "")
                
                # Get input and output tensors for the model.
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                #Fix Output dimensions
                output_shape = output_details[0]['shape']
                interpreter.resize_tensor_input(output_details[0]["index"], [x_test[0].shape[0], output_details[0]["shape"][1]])
                
                #Fix Input dimensions
                for i in range(len(input_details)):
                    interpreter.resize_tensor_input(input_details[i]["index"], [x_test[0].shape[0], input_details[i]["shape"][1], input_details[i]["shape"][2]])
                
                #Prepare tensors
                interpreter.allocate_tensors()
                
                data_dict = {"pep": x_test[0],
                             "a1": x_test[1],
                             "a2": x_test[2],
                             "a3": x_test[3],
                             "b1": x_test[4],
                             "b2": x_test[5],
                             "b3": x_test[6]}
                
                #Assign input data
                for i in range(len(input_details)):   
                    #Set input data for a given feature based on the name of the input in "input_details"
                    interpreter.set_tensor(input_details[i]['index'], data_dict[input_details[i]["name"].split(":")[0].split("_")[-1]])
                
                #Prepare the model for predictions
                interpreter.invoke()

                #Predict on input tensor
                avg_prediction += interpreter.get_tensor(output_details[0]['index'])
    
                #Clears the session for the next model
                tf.keras.backend.clear_session()
    
    #Averaging the predictions between all models
    avg_prediction = avg_prediction/20
    
    #Flatten list of predictions
    avg_prediction = list(itertools.chain(*avg_prediction))
    
    #Run TCRbase if alpha is not set to 0, and a positive database for the peptide exists
    if scale_prediction:
        tcrbase_prediction = get_tcrbase_prediction(pred_df, pep)
        pred_df['prediction'] = avg_prediction * tcrbase_prediction ** alpha  
    else:
        pred_df['prediction'] = avg_prediction
    
    #Calculate percentile rank (if possible)
    if pep in peptide_models:
        control_df = pd.read_csv('{}/models/nettcr_2_2_pretrained/negative_controls/{}.csv'.format(input_dir, pep))
        control_predictions = np.float64(control_df.prediction)
        if alpha != 0:
            tcrbase_control_df = pd.read_csv('{}/models/TCRbase/negative_controls/{}.csv'.format(input_dir, pep))
            tcrbase_control_predictions = np.float64(tcrbase_control_df.prediction)
            control_predictions = control_predictions * tcrbase_control_predictions ** alpha
        pred_df["percentile_rank"] = pred_df.prediction.apply(get_percentile_rank, control = control_predictions)
        
    else:
        pred_df["percentile_rank"] = np.nan
    
    full_pred_df = pd.concat([full_pred_df, pred_df])
    print("- Predictions took "+str(round(time.time()-time_start, 3))+" seconds\n")
    
#Save predictions in the same order as the input
full_pred_df.sort_index(inplace = True)
full_pred_df.to_csv(output_dir + '/' + output_file, index=False, sep = ',')

#Print prediction to stdout
print("\n \nBelow is a table represention of binding predictions between T-Cell receptors and peptides. \n \n")
print(full_pred_df[(full_pred_df.percentile_rank.isna()) | (full_pred_df.percentile_rank < threshold)])