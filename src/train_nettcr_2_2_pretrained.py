# -*- coding: utf-8 -*-
#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", help = "Dataset used for training", type = str)
parser.add_argument("--val_data", help = "Dataset used for validation", type = str)
parser.add_argument("--outdir", help = "Folder to save the model in", type = str)
parser.add_argument("--model_name", default = "pretrained_model", help = "Prefix of the saved model", type = str)
parser.add_argument("--dropout_rate", "-dr", default = 0.6, help = "Fraction of concatenated max-pooling values set to 0. Used for preventing overfitting", type = float)
parser.add_argument("--learning_rate", "-lr", default = 0.001, type = float)
parser.add_argument("--patience", "-p", default = 100, type = int)
parser.add_argument("--batch_size", "-bs", default = 64, type = int)
parser.add_argument("--epochs", "-e", default = 200, type = int)
parser.add_argument("--verbose", default = 2, choices = [0,1,2], type = int)
parser.add_argument("--seed", default = 15, type = int)
parser.add_argument("--inter_threads", default = 1, type = int)
    
args = parser.parse_args()

### Model training parameters ###
train_data = str(args.train_data)
val_data = str(args.val_data)
outdir = str(args.outdir)
model_name = str(args.model_name)
patience = int(args.patience) #Patience for Early Stopping
dropout_rate = float(args.dropout_rate) #Dropout Rate
lr = float(args.learning_rate)
batch_size = int(args.batch_size) #Default batch size. Might be changed slightly due to the adjust_batch_size function
EPOCHS = int(args.epochs) #Number of epochs in the training
verbose = int(args.verbose) #Determines how often metrics are reported during training
seed = int(args.seed)

assert 1> dropout_rate >= 0, "Dropout rate must be lower than 1, with 0 being the lowest possible value. A dropout rate of 0 turns the dropout off"

#Import other modules
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import roc_auc_score

import os
import numpy as np
import pandas as pd 

#Imports the util module and network architectures for NetTCR
import keras_utils
import matplotlib.pyplot as plt
import seaborn as sns
import random

from nettcr_archs import CNN_CDR123_global_max_two_step_pre_training

#Set style to seaborn
sns.set()

tf.config.threading.set_inter_op_parallelism_threads(
    int(args.inter_threads)
)

encoding = keras_utils.blosum50_20aa #Encoding for amino acid sequences

# Set random seed
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
tf.random.set_seed(seed) # Tensorflow random seed

### Input/Output ###
# Read in data
train_df = pd.read_csv(train_data)
val_df = pd.read_csv(val_data)


#From https://github.com/mnielLab/NetTCR-2.1
for df in [train_df, val_df]:
    assert "A1" and "A2" and "A3" in df.columns, "Make sure the input files contains all the CDRs"
    assert "B1" and "B2" and "B3" in df.columns, "Make sure the input files contains all the CDRs"
    assert "peptide" in df.columns, "Couldn't find peptide in the input data"
    assert "binder" in df.columns, "Couldn't find target labels in the input data, which is required for training"

#Sample weights (for first round of training)
weight_dict = np.log2(train_df.shape[0]/(train_df.peptide.value_counts()))/np.log2(len(train_df.peptide.unique()))
#Normalize, so that loss is comparable
weight_dict = weight_dict*(train_df.shape[0]/np.sum(weight_dict*train_df.peptide.value_counts()))
train_df["sample_weight"] = train_df["peptide"].map(weight_dict)
val_df["sample_weight"] = val_df["peptide"].map(weight_dict)

#List of peptide
pep_list = list(train_df[train_df.binder==1].peptide.value_counts(ascending=False).index)

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
    targets = df.binder.values
    sample_weights = df.sample_weight
    tf_ds = [np.float32(encoded_pep),
             np.float32(encoded_a1), np.float32(encoded_a2), np.float32(encoded_a3), 
             np.float32(encoded_b1), np.float32(encoded_b2), np.float32(encoded_b3),
             targets,
             sample_weights]

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
for pep in pep_list:
    if not os.path.exists(outdir+"/"+pep):
        os.makedirs(outdir+"/"+pep)
    
dependencies = {
    'auc_01': auc_01
}


#####################################
### First round of training (pan) ###
#####################################

pep_train = train_df
pep_val = val_df

# Prepare plotting
fig, ax = plt.subplots(figsize=(15, 10))

#Training data
train_tensor = make_tf_ds(pep_train, encoding = encoding)
x_train = train_tensor[0:7]
targets_train = train_tensor[7]
weights_train = train_tensor[8]

#Validation data
valid_tensor = make_tf_ds(pep_val, encoding = encoding)
x_valid = valid_tensor[0:7]
targets_valid = valid_tensor[7]
weights_valid = valid_tensor[8]

#Load model architecture
model = CNN_CDR123_global_max_two_step_pre_training(dropout_rate = dropout_rate, seed = seed)

#Freeze certain layers
for layer in model.layers:
    #Layer for first training
    if layer.name.startswith('first'):
        layer.trainable = True
    #Layer for second training
    if layer.name.startswith('second'):
        layer.trainable = False

#For saving model checkpoints for pre-training?
ModelCheckpoint = keras.callbacks.ModelCheckpoint(
        filepath = outdir+'/checkpoint/{}.h5'.format(model_name),
        monitor = "val_auc_01",
        mode = "max",
        save_best_only = True)

#Setting up the EarlyStopping function
EarlyStopping = keras.callbacks.EarlyStopping(
    monitor = "val_auc_01",
    mode = "max",
    patience = patience)

#Callbacks to include for the model
callbacks_list = [EarlyStopping,
                  ModelCheckpoint
    ]

#Optimizers, loss functions, and additional metrics to track
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = [auc_01, "AUC"],
              weighted_metrics = [])

#Announce Training
print("Training pan-specific CNN block", end = "\n")

#Model training
history = model.fit(x = {"pep": x_train[0],
                         "a1": x_train[1],
                         "a2": x_train[2],
                         "a3": x_train[3],
                         "b1": x_train[4],
                         "b2": x_train[5],
                         "b3": x_train[6]},
          y = targets_train,
          batch_size = batch_size,
          epochs = EPOCHS,
          verbose = verbose,
          sample_weight = weights_train,
          validation_data = ({"pep": x_valid[0],
                              "a1": x_valid[1],
                              "a2": x_valid[2],
                              "a3": x_valid[3],
                              "b1": x_valid[4],
                              "b2": x_valid[5],
                              "b3": x_valid[6]}, 
                             targets_valid,
                             weights_valid),
          validation_batch_size = batch_size,
          shuffle = True,
          callbacks=callbacks_list
          )            

#Record training and validation loss            
valid_loss = history.history["val_loss"]
train_loss = history.history["loss"]

#Plotting the losses
ax.plot(train_loss, label='train')
ax.plot(valid_loss, label='validation')
ax.set_ylabel("Loss")
ax.set_xlabel("Epoch")
ax.legend()

#Load the best model
model = keras.models.load_model(outdir+'/checkpoint/{}.h5'.format(model_name), custom_objects=dependencies)

#Converting the model to a TFlite model (much quicker for predictions)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()   

#Save the model.
with open(outdir+'/checkpoint/{}.tflite'.format(model_name), 'wb') as f:
  f.write(tflite_model)

#Optional - Remove standard model to save space. Uncheck to enable this deletion
#os.remove(outdir+'/'+pep+'/checkpoint/model.h5')

#Save training/validation loss plot
plt.tight_layout()
plt.show()
fig.savefig(outdir+'/{}_learning_curves.png'.format(model_name), dpi=100)


#Clears the session for the next model
tf.keras.backend.clear_session()

##########################################
### Second round of training (peptide) ###
##########################################

#Sample weights for peptide-specific training
train_df["sample_weight"] = 1
val_df["sample_weight"] = 1

#Minimum fraction filled for last batch
batch_threshold = 0.5

for pep in pep_list:
    pep_train = train_df[train_df.peptide == pep]
    pep_val = val_df[val_df.peptide == pep]
    
    # Prepare plotting
    fig, ax = plt.subplots(figsize=(15, 10))
    
    #Training data
    train_tensor = make_tf_ds(pep_train, encoding = encoding)
    x_train = train_tensor[0:7]
    targets_train = train_tensor[7]
    weights_train = train_tensor[8]
    train_batch_size = keras_utils.adjust_batch_size(x_train[0].shape[0], batch_size, batch_threshold)
    
    #Validation data
    valid_tensor = make_tf_ds(pep_val, encoding = encoding)
    x_valid = valid_tensor[0:7]
    targets_valid = valid_tensor[7]
    weights_valid = valid_tensor[8]
    valid_batch_size = keras_utils.adjust_batch_size(x_valid[0].shape[0], batch_size, batch_threshold)
    
    #Pre-trained model
    model = keras.models.load_model(outdir+'/checkpoint/{}.h5'.format(model_name), custom_objects = dependencies)
    
    #Freeze certain layers
    for layer in model.layers:
        #Layer for first training
        if layer.name.startswith('first'):
            layer.trainable = False
        #Layer for second training
        if layer.name.startswith('second'):
            layer.trainable = True
    
    #For saving model checkpoints for pre-training?
    ModelCheckpoint = keras.callbacks.ModelCheckpoint(
            filepath = outdir+'/'+pep+'/checkpoint/{}.h5'.format(model_name),
            monitor = "val_auc_01",
            mode = "max",
            save_best_only = True)
    
    #Setting up the EarlyStopping function
    EarlyStopping = keras.callbacks.EarlyStopping(
        monitor = "val_auc_01",
        mode = "max",
        patience = patience)
    
    #Callbacks to include for the model
    callbacks_list = [EarlyStopping,
                      ModelCheckpoint
        ]
    
    #Optimizers, loss functions, and additional metrics to track
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
                  loss = tf.keras.losses.BinaryCrossentropy(),
                  metrics = [auc_01, "AUC"],
                  weighted_metrics = [])
    
    #Announce Training
    print("Training {} model".format(pep), end = "\n")
    
    #Model training
    history = model.fit(x = {"pep": x_train[0],
                             "a1": x_train[1],
                             "a2": x_train[2],
                             "a3": x_train[3],
                             "b1": x_train[4],
                             "b2": x_train[5],
                             "b3": x_train[6]},
              y = targets_train,
              batch_size = train_batch_size,
              epochs = EPOCHS,
              verbose = verbose,
              sample_weight = weights_train,
              validation_data = ({"pep": x_valid[0],
                                  "a1": x_valid[1],
                                  "a2": x_valid[2],
                                  "a3": x_valid[3],
                                  "b1": x_valid[4],
                                  "b2": x_valid[5],
                                  "b3": x_valid[6]}, 
                                 targets_valid,
                                 weights_valid),
              validation_batch_size = valid_batch_size,
              shuffle = True,
              callbacks=callbacks_list
              )            
    
    #Record training and validation loss            
    valid_loss = history.history["val_loss"]
    train_loss = history.history["loss"]
    
    #Plotting the losses
    ax.plot(train_loss, label='train')
    ax.plot(valid_loss, label='validation')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.legend()

    #Load the best model
    model = keras.models.load_model(outdir+'/'+pep+'/checkpoint/{}.h5'.format(model_name), custom_objects=dependencies)
    
    #Converting the model to a TFlite model (much quicker for predictions)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()   
    
    #Save the model.
    with open(outdir+'/'+pep+'/checkpoint/{}.tflite'.format(model_name), 'wb') as f:
      f.write(tflite_model)
    
    #Optional - Remove standard model to save space. Uncheck to enable this deletion
    #os.remove(outdir+'/'+pep+'/checkpoint/model.h5')
    
    #Save training/validation loss plot
    plt.tight_layout()
    plt.show()
    fig.savefig(outdir+'/'+pep+'/{}_learning_curves.png'.format(model_name), dpi=100)