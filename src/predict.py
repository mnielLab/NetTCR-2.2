# -*- coding: utf-8 -*-
#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_data", help = "Dataset to perform the predictions on", type = str)
parser.add_argument("--outdir", help = "Folder where the models are stored. The prediction file is saved to this location", type = str)
parser.add_argument("--model_name", help = "Prefix for the saved models. This prefix is also used for the prediction file", type = str)
parser.add_argument("--model_type", help = "Type of NetTCR 2.2 model", choices = ["pan", "peptide", "pretrained"], type = str)
parser.add_argument("--seed", default = 15, type = int)
parser.add_argument("--inter_threads", default=1)    

args = parser.parse_args()

### Model prediction parameters ###
test_data = str(args.test_data)
outdir = str(args.outdir)
model_name = str(args.model_name)
model_type = str(args.model_type)
seed = int(args.seed)

#Import remaining modules
import tensorflow as tf

from sklearn.metrics import roc_auc_score

import os
import numpy as np
import pandas as pd 

#Imports the util module and network architectures for NetTCR
import keras_utils
import random
import time

tf.config.threading.set_inter_op_parallelism_threads(
    int(args.inter_threads)
)

encoding = keras_utils.blosum50_20aa #Encoding for amino acid sequences

# Set random seed
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
tf.random.set_seed(seed) # Tensorflow random seed

# Read in data
test_df = pd.read_csv(test_data)

#From https://github.com/mnielLab/NetTCR-2.1
assert "A1" and "A2" and "A3" in test_df.columns, "Make sure the input files contains all the CDRs"
assert "B1" and "B2" and "B3" in test_df.columns, "Make sure the input files contains all the CDRs"
assert "peptide" in test_df.columns, "Couldn't find peptide in the input data"

pep_list = list(test_df.peptide.value_counts(ascending=False).index)

encoding = keras_utils.blosum50_20aa #Encoding for amino acid sequences

#Padding to certain length
a1_max = 7
a2_max = 8
a3_max = 22
b1_max = 6
b2_max = 7
b3_max = 23
pep_max = 12

def make_tf_ds(df, encoding):
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

def my_numpy_function(y_true, y_pred):
    """Implementation of AUC 0.1 metric for Tensorflow"""
    try:
        auc = roc_auc_score(y_true, y_pred, max_fpr = 0.1)
    except ValueError:
        auc = np.array([float(0)])
    return auc

#Custom metric for AUC 0.1
def auc_01(y_true, y_pred):
    """Converts function to optimised tensorflow numpy function"""
    auc_01 = tf.numpy_function(my_numpy_function, [y_true, y_pred], tf.float64)
    return auc_01

#Creation of output directory
if not os.path.exists(outdir):
    os.makedirs(outdir)
    
#Necessary to load the model with the custom metric
dependencies = {
    'auc_01': auc_01
}
     
 
#Prepare output dataframe (test predictions)
full_pred_df = pd.DataFrame()
if model_type != "pan":
    for pep in pep_list:
        print(pep)
        time_start = time.time()
        x_test_df = test_df[test_df.peptide == pep].copy(deep = True)
        test_tensor = make_tf_ds(x_test_df, encoding = encoding)
        x_test = test_tensor[0:7]
        avg_prediction = 0
                     
        # Load the TFLite model and allocate tensors.
        try:
            interpreter = tf.lite.Interpreter(model_path = outdir+'/'+pep+'/checkpoint/{}.tflite'.format(model_name))
            
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
            
            
            #Ready the model
            interpreter.invoke()
        
            #Make prediction
            avg_prediction  += interpreter.get_tensor(output_details[0]['index'])
        
            #Clears the session for the next model
            tf.keras.backend.clear_session()
            
            #Add prediction to test data
            x_test_df['prediction'] = avg_prediction
            full_pred_df = pd.concat([full_pred_df, x_test_df])
            
            #Report time spent for prediction
            print(str(round(time.time()-time_start, 3))+" seconds")
            
        except ValueError:
            print("A model for {} does not exist. Skipping predictions for this peptide".format(pep))
        
    #Save prediction
    full_pred_df.to_csv(outdir + '/{}_prediction.csv'.format(model_name), index=False)

else:
    time_start = time.time()
    x_test_df = test_df
    test_tensor = make_tf_ds(x_test_df, encoding = encoding)
    x_test = test_tensor[0:7]
    avg_prediction = 0
                 
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path = outdir+'/checkpoint/{}.tflite'.format(model_name))
    
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
    
    
    #Ready the model
    interpreter.invoke()

    #Make prediction
    avg_prediction  += interpreter.get_tensor(output_details[0]['index'])

    #Clears the session for the next model
    tf.keras.backend.clear_session()
    
    #Add prediction to test data
    x_test_df['prediction'] = avg_prediction
    print(str(round(time.time()-time_start, 3))+" seconds")
    x_test_df.to_csv(outdir + '/{}_prediction.csv'.format(model_name), index=False)
