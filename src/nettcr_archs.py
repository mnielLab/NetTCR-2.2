# -*- coding: utf-8 -*-

from tensorflow import keras
from tensorflow.keras import layers

#These networks are based on NetTCR 2.1 by Alessandro Montemurro
def CNN_CDR123_global_max(dropout_rate, seed, embed_dim = 20):
    a1_max = 7
    a2_max = 8
    a3_max = 22
    b1_max = 6
    b2_max = 7
    b3_max = 23
    pep_max = 12
    #embed_dim = 20
    conv_activation = "relu"
    dense_activation = "sigmoid"
    
    pep = keras.Input(shape = (pep_max, embed_dim), name ="pep")
    a1 = keras.Input(shape = (a1_max, embed_dim), name ="a1")
    a2 = keras.Input(shape = (a2_max, embed_dim), name ="a2")
    a3 = keras.Input(shape = (a3_max, embed_dim), name ="a3")
    b1 = keras.Input(shape = (b1_max, embed_dim), name ="b1")
    b2 = keras.Input(shape = (b2_max, embed_dim), name ="b2")
    b3 = keras.Input(shape = (b3_max, embed_dim), name ="b3")
    
    pep_1_CNN = layers.Conv1D(filters = 16, kernel_size = 1, padding = "same", activation = conv_activation)(pep)
    pep_3_CNN = layers.Conv1D(filters = 16, kernel_size = 3, padding = "same", activation = conv_activation)(pep)
    pep_5_CNN = layers.Conv1D(filters = 16, kernel_size = 5, padding = "same", activation = conv_activation)(pep)
    pep_7_CNN = layers.Conv1D(filters = 16, kernel_size = 7, padding = "same", activation = conv_activation)(pep)
    pep_9_CNN = layers.Conv1D(filters = 16, kernel_size = 9, padding = "same", activation = conv_activation)(pep)
    
    a1_1_CNN = layers.Conv1D(filters = 16, kernel_size = 1, padding = "same", activation = conv_activation)(a1)
    a1_3_CNN = layers.Conv1D(filters = 16, kernel_size = 3, padding = "same", activation = conv_activation)(a1)
    a1_5_CNN = layers.Conv1D(filters = 16, kernel_size = 5, padding = "same", activation = conv_activation)(a1)
    a1_7_CNN = layers.Conv1D(filters = 16, kernel_size = 7, padding = "same", activation = conv_activation)(a1)
    a1_9_CNN = layers.Conv1D(filters = 16, kernel_size = 9, padding = "same", activation = conv_activation)(a1)
    
    a2_1_CNN = layers.Conv1D(filters = 16, kernel_size = 1, padding = "same", activation = conv_activation)(a2)
    a2_3_CNN = layers.Conv1D(filters = 16, kernel_size = 3, padding = "same", activation = conv_activation)(a2)
    a2_5_CNN = layers.Conv1D(filters = 16, kernel_size = 5, padding = "same", activation = conv_activation)(a2)
    a2_7_CNN = layers.Conv1D(filters = 16, kernel_size = 7, padding = "same", activation = conv_activation)(a2)
    a2_9_CNN = layers.Conv1D(filters = 16, kernel_size = 9, padding = "same", activation = conv_activation)(a2)
    
    a3_1_CNN = layers.Conv1D(filters = 16, kernel_size = 1, padding = "same", activation = conv_activation)(a3)
    a3_3_CNN = layers.Conv1D(filters = 16, kernel_size = 3, padding = "same", activation = conv_activation)(a3)
    a3_5_CNN = layers.Conv1D(filters = 16, kernel_size = 5, padding = "same", activation = conv_activation)(a3)
    a3_7_CNN = layers.Conv1D(filters = 16, kernel_size = 7, padding = "same", activation = conv_activation)(a3)
    a3_9_CNN = layers.Conv1D(filters = 16, kernel_size = 9, padding = "same", activation = conv_activation)(a3)
    
    b1_1_CNN = layers.Conv1D(filters = 16, kernel_size = 1, padding = "same", activation = conv_activation)(b1)
    b1_3_CNN = layers.Conv1D(filters = 16, kernel_size = 3, padding = "same", activation = conv_activation)(b1)
    b1_5_CNN = layers.Conv1D(filters = 16, kernel_size = 5, padding = "same", activation = conv_activation)(b1)
    b1_7_CNN = layers.Conv1D(filters = 16, kernel_size = 7, padding = "same", activation = conv_activation)(b1)
    b1_9_CNN = layers.Conv1D(filters = 16, kernel_size = 9, padding = "same", activation = conv_activation)(b1)
    
    b2_1_CNN = layers.Conv1D(filters = 16, kernel_size = 1, padding = "same", activation = conv_activation)(b2)
    b2_3_CNN = layers.Conv1D(filters = 16, kernel_size = 3, padding = "same", activation = conv_activation)(b2)
    b2_5_CNN = layers.Conv1D(filters = 16, kernel_size = 5, padding = "same", activation = conv_activation)(b2)
    b2_7_CNN = layers.Conv1D(filters = 16, kernel_size = 7, padding = "same", activation = conv_activation)(b2)
    b2_9_CNN = layers.Conv1D(filters = 16, kernel_size = 9, padding = "same", activation = conv_activation)(b2)
    
    b3_1_CNN = layers.Conv1D(filters = 16, kernel_size = 1, padding = "same", activation = conv_activation)(b3)
    b3_3_CNN = layers.Conv1D(filters = 16, kernel_size = 3, padding = "same", activation = conv_activation)(b3)
    b3_5_CNN = layers.Conv1D(filters = 16, kernel_size = 5, padding = "same", activation = conv_activation)(b3)
    b3_7_CNN = layers.Conv1D(filters = 16, kernel_size = 7, padding = "same", activation = conv_activation)(b3)
    b3_9_CNN = layers.Conv1D(filters = 16, kernel_size = 9, padding = "same", activation = conv_activation)(b3) 
    
    pep_1_pool = layers.GlobalMaxPooling1D()(pep_1_CNN)
    pep_3_pool = layers.GlobalMaxPooling1D()(pep_3_CNN)
    pep_5_pool = layers.GlobalMaxPooling1D()(pep_5_CNN)
    pep_7_pool = layers.GlobalMaxPooling1D()(pep_7_CNN)
    pep_9_pool = layers.GlobalMaxPooling1D()(pep_9_CNN)
    
    a1_1_pool = layers.GlobalMaxPooling1D()(a1_1_CNN)
    a1_3_pool = layers.GlobalMaxPooling1D()(a1_3_CNN)
    a1_5_pool = layers.GlobalMaxPooling1D()(a1_5_CNN)
    a1_7_pool = layers.GlobalMaxPooling1D()(a1_7_CNN)
    a1_9_pool = layers.GlobalMaxPooling1D()(a1_9_CNN)
    
    a2_1_pool = layers.GlobalMaxPooling1D()(a2_1_CNN)
    a2_3_pool = layers.GlobalMaxPooling1D()(a2_3_CNN)
    a2_5_pool = layers.GlobalMaxPooling1D()(a2_5_CNN)
    a2_7_pool = layers.GlobalMaxPooling1D()(a2_7_CNN)
    a2_9_pool = layers.GlobalMaxPooling1D()(a2_9_CNN)
    
    a3_1_pool = layers.GlobalMaxPooling1D()(a3_1_CNN)
    a3_3_pool = layers.GlobalMaxPooling1D()(a3_3_CNN)
    a3_5_pool = layers.GlobalMaxPooling1D()(a3_5_CNN)
    a3_7_pool = layers.GlobalMaxPooling1D()(a3_7_CNN)
    a3_9_pool = layers.GlobalMaxPooling1D()(a3_9_CNN)
    
    b1_1_pool = layers.GlobalMaxPooling1D()(b1_1_CNN)
    b1_3_pool = layers.GlobalMaxPooling1D()(b1_3_CNN)
    b1_5_pool = layers.GlobalMaxPooling1D()(b1_5_CNN)
    b1_7_pool = layers.GlobalMaxPooling1D()(b1_7_CNN)
    b1_9_pool = layers.GlobalMaxPooling1D()(b1_9_CNN)
    
    b2_1_pool = layers.GlobalMaxPooling1D()(b2_1_CNN)
    b2_3_pool = layers.GlobalMaxPooling1D()(b2_3_CNN)
    b2_5_pool = layers.GlobalMaxPooling1D()(b2_5_CNN)
    b2_7_pool = layers.GlobalMaxPooling1D()(b2_7_CNN)
    b2_9_pool = layers.GlobalMaxPooling1D()(b2_9_CNN)
    
    b3_1_pool = layers.GlobalMaxPooling1D()(b3_1_CNN)
    b3_3_pool = layers.GlobalMaxPooling1D()(b3_3_CNN)
    b3_5_pool = layers.GlobalMaxPooling1D()(b3_5_CNN)
    b3_7_pool = layers.GlobalMaxPooling1D()(b3_7_CNN)
    b3_9_pool = layers.GlobalMaxPooling1D()(b3_9_CNN)
    
    cat = layers.Concatenate()([pep_1_pool, pep_3_pool, pep_5_pool, pep_7_pool, pep_9_pool,
                                a1_1_pool, a1_3_pool, a1_5_pool, a1_7_pool, a1_9_pool,
                                a2_1_pool, a2_3_pool, a2_5_pool, a2_7_pool, a2_9_pool,
                                a3_1_pool, a3_3_pool, a3_5_pool, a3_7_pool, a3_9_pool,
                                b1_1_pool, b1_3_pool, b1_5_pool, b1_7_pool, b1_9_pool,
                                b2_1_pool, b2_3_pool, b2_5_pool, b2_7_pool, b2_9_pool,
                                b3_1_pool, b3_3_pool, b3_5_pool, b3_7_pool, b3_9_pool])
    
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)
    dense = layers.Dense(64, activation = dense_activation)(cat_dropout)
    out = layers.Dense(1,activation = "sigmoid")(dense)
    
    model = keras.Model(inputs = [pep, a1, a2, a3, b1, b2, b3],
                        outputs = out)
    
    return model

def CNN_CDR123_global_max_two_step_pre_training(dropout_rate, seed, embed_dim = 20, nr_of_filters_1 = 32, nr_of_filters_2 = 16):
    a1_max = 7
    a2_max = 8
    a3_max = 22
    b1_max = 6
    b2_max = 7
    b3_max = 23
    pep_max = 12
    #embed_dim = 20
    conv_activation = "relu"
    dense_activation = "sigmoid"
    
    pep = keras.Input(shape = (pep_max, embed_dim), name ="pep")
    a1 = keras.Input(shape = (a1_max, embed_dim), name ="a1")
    a2 = keras.Input(shape = (a2_max, embed_dim), name ="a2")
    a3 = keras.Input(shape = (a3_max, embed_dim), name ="a3")
    b1 = keras.Input(shape = (b1_max, embed_dim), name ="b1")
    b2 = keras.Input(shape = (b2_max, embed_dim), name ="b2")
    b3 = keras.Input(shape = (b3_max, embed_dim), name ="b3")
    
    #Trained in a pan-specific setup
    pep_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_pep_1_conv")(pep)
    pep_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_pep_3_conv")(pep)
    pep_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_pep_5_conv")(pep)
    pep_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_pep_7_conv")(pep)
    pep_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_pep_9_conv")(pep)
    
    a1_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a1_1_conv")(a1)
    a1_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a1_3_conv")(a1)
    a1_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a1_5_conv")(a1)
    a1_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a1_7_conv")(a1)
    a1_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a1_9_conv")(a1)
    
    a2_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a2_1_conv")(a2)
    a2_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a2_3_conv")(a2)
    a2_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a2_5_conv")(a2)
    a2_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a2_7_conv")(a2)
    a2_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a2_9_conv")(a2)
    
    a3_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a3_1_conv")(a3)
    a3_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a3_3_conv")(a3)
    a3_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a3_5_conv")(a3)
    a3_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a3_7_conv")(a3)
    a3_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a3_9_conv")(a3)
    
    b1_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b1_1_conv")(b1)
    b1_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b1_3_conv")(b1)
    b1_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b1_5_conv")(b1)
    b1_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b1_7_conv")(b1)
    b1_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b1_9_conv")(b1)
    
    b2_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b2_1_conv")(b2)
    b2_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b2_3_conv")(b2)
    b2_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b2_5_conv")(b2)
    b2_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b2_7_conv")(b2)
    b2_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b2_9_conv")(b2)
    
    b3_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b3_1_conv")(b3)
    b3_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b3_3_conv")(b3)
    b3_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b3_5_conv")(b3)
    b3_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b3_7_conv")(b3)
    b3_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b3_9_conv")(b3) 
    
    pep_1_pool = layers.GlobalMaxPooling1D(name = "first_pep_1_pool")(pep_1_CNN)
    pep_3_pool = layers.GlobalMaxPooling1D(name = "first_pep_3_pool")(pep_3_CNN)
    pep_5_pool = layers.GlobalMaxPooling1D(name = "first_pep_5_pool")(pep_5_CNN)
    pep_7_pool = layers.GlobalMaxPooling1D(name = "first_pep_7_pool")(pep_7_CNN)
    pep_9_pool = layers.GlobalMaxPooling1D(name = "first_pep_9_pool")(pep_9_CNN)
    
    a1_1_pool = layers.GlobalMaxPooling1D(name = "first_a1_1_pool")(a1_1_CNN)
    a1_3_pool = layers.GlobalMaxPooling1D(name = "first_a1_3_pool")(a1_3_CNN)
    a1_5_pool = layers.GlobalMaxPooling1D(name = "first_a1_5_pool")(a1_5_CNN)
    a1_7_pool = layers.GlobalMaxPooling1D(name = "first_a1_7_pool")(a1_7_CNN)
    a1_9_pool = layers.GlobalMaxPooling1D(name = "first_a1_9_pool")(a1_9_CNN)
    
    a2_1_pool = layers.GlobalMaxPooling1D(name = "first_a2_1_pool")(a2_1_CNN)
    a2_3_pool = layers.GlobalMaxPooling1D(name = "first_a2_3_pool")(a2_3_CNN)
    a2_5_pool = layers.GlobalMaxPooling1D(name = "first_a2_5_pool")(a2_5_CNN)
    a2_7_pool = layers.GlobalMaxPooling1D(name = "first_a2_7_pool")(a2_7_CNN)
    a2_9_pool = layers.GlobalMaxPooling1D(name = "first_a2_9_pool")(a2_9_CNN)
    
    a3_1_pool = layers.GlobalMaxPooling1D(name = "first_a3_1_pool")(a3_1_CNN)
    a3_3_pool = layers.GlobalMaxPooling1D(name = "first_a3_3_pool")(a3_3_CNN)
    a3_5_pool = layers.GlobalMaxPooling1D(name = "first_a3_5_pool")(a3_5_CNN)
    a3_7_pool = layers.GlobalMaxPooling1D(name = "first_a3_7_pool")(a3_7_CNN)
    a3_9_pool = layers.GlobalMaxPooling1D(name = "first_a3_9_pool")(a3_9_CNN)
    
    b1_1_pool = layers.GlobalMaxPooling1D(name = "first_b1_1_pool")(b1_1_CNN)
    b1_3_pool = layers.GlobalMaxPooling1D(name = "first_b1_3_pool")(b1_3_CNN)
    b1_5_pool = layers.GlobalMaxPooling1D(name = "first_b1_5_pool")(b1_5_CNN)
    b1_7_pool = layers.GlobalMaxPooling1D(name = "first_b1_7_pool")(b1_7_CNN)
    b1_9_pool = layers.GlobalMaxPooling1D(name = "first_b1_9_pool")(b1_9_CNN)
    
    b2_1_pool = layers.GlobalMaxPooling1D(name = "first_b2_1_pool")(b2_1_CNN)
    b2_3_pool = layers.GlobalMaxPooling1D(name = "first_b2_3_pool")(b2_3_CNN)
    b2_5_pool = layers.GlobalMaxPooling1D(name = "first_b2_5_pool")(b2_5_CNN)
    b2_7_pool = layers.GlobalMaxPooling1D(name = "first_b2_7_pool")(b2_7_CNN)
    b2_9_pool = layers.GlobalMaxPooling1D(name = "first_b2_9_pool")(b2_9_CNN)
    
    b3_1_pool = layers.GlobalMaxPooling1D(name = "first_b3_1_pool")(b3_1_CNN)
    b3_3_pool = layers.GlobalMaxPooling1D(name = "first_b3_3_pool")(b3_3_CNN)
    b3_5_pool = layers.GlobalMaxPooling1D(name = "first_b3_5_pool")(b3_5_CNN)
    b3_7_pool = layers.GlobalMaxPooling1D(name = "first_b3_7_pool")(b3_7_CNN)
    b3_9_pool = layers.GlobalMaxPooling1D(name = "first_b3_9_pool")(b3_9_CNN)
    
    #Trained in a peptide-specific setup (second round of training)
    second_a1_1_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 1, padding = "same", activation = conv_activation, name = "second_a1_1_conv")(a1)
    second_a1_3_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 3, padding = "same", activation = conv_activation, name = "second_a1_3_conv")(a1)
    second_a1_5_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 5, padding = "same", activation = conv_activation, name = "second_a1_5_conv")(a1)
    second_a1_7_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 7, padding = "same", activation = conv_activation, name = "second_a1_7_conv")(a1)
    second_a1_9_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 9, padding = "same", activation = conv_activation, name = "second_a1_9_conv")(a1)
    
    second_a2_1_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 1, padding = "same", activation = conv_activation, name = "second_a2_1_conv")(a2)
    second_a2_3_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 3, padding = "same", activation = conv_activation, name = "second_a2_3_conv")(a2)
    second_a2_5_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 5, padding = "same", activation = conv_activation, name = "second_a2_5_conv")(a2)
    second_a2_7_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 7, padding = "same", activation = conv_activation, name = "second_a2_7_conv")(a2)
    second_a2_9_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 9, padding = "same", activation = conv_activation, name = "second_a2_9_conv")(a2)
    
    second_a3_1_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 1, padding = "same", activation = conv_activation, name = "second_a3_1_conv")(a3)
    second_a3_3_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 3, padding = "same", activation = conv_activation, name = "second_a3_3_conv")(a3)
    second_a3_5_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 5, padding = "same", activation = conv_activation, name = "second_a3_5_conv")(a3)
    second_a3_7_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 7, padding = "same", activation = conv_activation, name = "second_a3_7_conv")(a3)
    second_a3_9_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 9, padding = "same", activation = conv_activation, name = "second_a3_9_conv")(a3)
    
    second_b1_1_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 1, padding = "same", activation = conv_activation, name = "second_b1_1_conv")(b1)
    second_b1_3_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 3, padding = "same", activation = conv_activation, name = "second_b1_3_conv")(b1)
    second_b1_5_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 5, padding = "same", activation = conv_activation, name = "second_b1_5_conv")(b1)
    second_b1_7_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 7, padding = "same", activation = conv_activation, name = "second_b1_7_conv")(b1)
    second_b1_9_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 9, padding = "same", activation = conv_activation, name = "second_b1_9_conv")(b1)
    
    second_b2_1_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 1, padding = "same", activation = conv_activation, name = "second_b2_1_conv")(b2)
    second_b2_3_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 3, padding = "same", activation = conv_activation, name = "second_b2_3_conv")(b2)
    second_b2_5_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 5, padding = "same", activation = conv_activation, name = "second_b2_5_conv")(b2)
    second_b2_7_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 7, padding = "same", activation = conv_activation, name = "second_b2_7_conv")(b2)
    second_b2_9_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 9, padding = "same", activation = conv_activation, name = "second_b2_9_conv")(b2)
    
    second_b3_1_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 1, padding = "same", activation = conv_activation, name = "second_b3_1_conv")(b3)
    second_b3_3_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 3, padding = "same", activation = conv_activation, name = "second_b3_3_conv")(b3)
    second_b3_5_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 5, padding = "same", activation = conv_activation, name = "second_b3_5_conv")(b3)
    second_b3_7_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 7, padding = "same", activation = conv_activation, name = "second_b3_7_conv")(b3)
    second_b3_9_CNN = layers.Conv1D(filters = nr_of_filters_2, kernel_size = 9, padding = "same", activation = conv_activation, name = "second_b3_9_conv")(b3) 
    
    second_a1_1_pool = layers.GlobalMaxPooling1D(name = "second_a1_1_pool")(second_a1_1_CNN)
    second_a1_3_pool = layers.GlobalMaxPooling1D(name = "second_a1_3_pool")(second_a1_3_CNN)
    second_a1_5_pool = layers.GlobalMaxPooling1D(name = "second_a1_5_pool")(second_a1_5_CNN)
    second_a1_7_pool = layers.GlobalMaxPooling1D(name = "second_a1_7_pool")(second_a1_7_CNN)
    second_a1_9_pool = layers.GlobalMaxPooling1D(name = "second_a1_9_pool")(second_a1_9_CNN)
    
    second_a2_1_pool = layers.GlobalMaxPooling1D(name = "second_a2_1_pool")(second_a2_1_CNN)
    second_a2_3_pool = layers.GlobalMaxPooling1D(name = "second_a2_3_pool")(second_a2_3_CNN)
    second_a2_5_pool = layers.GlobalMaxPooling1D(name = "second_a2_5_pool")(second_a2_5_CNN)
    second_a2_7_pool = layers.GlobalMaxPooling1D(name = "second_a2_7_pool")(second_a2_7_CNN)
    second_a2_9_pool = layers.GlobalMaxPooling1D(name = "second_a2_9_pool")(second_a2_9_CNN)
    
    second_a3_1_pool = layers.GlobalMaxPooling1D(name = "second_a3_1_pool")(second_a3_1_CNN)
    second_a3_3_pool = layers.GlobalMaxPooling1D(name = "second_a3_3_pool")(second_a3_3_CNN)
    second_a3_5_pool = layers.GlobalMaxPooling1D(name = "second_a3_5_pool")(second_a3_5_CNN)
    second_a3_7_pool = layers.GlobalMaxPooling1D(name = "second_a3_7_pool")(second_a3_7_CNN)
    second_a3_9_pool = layers.GlobalMaxPooling1D(name = "second_a3_9_pool")(second_a3_9_CNN)
    
    second_b1_1_pool = layers.GlobalMaxPooling1D(name = "second_b1_1_pool")(second_b1_1_CNN)
    second_b1_3_pool = layers.GlobalMaxPooling1D(name = "second_b1_3_pool")(second_b1_3_CNN)
    second_b1_5_pool = layers.GlobalMaxPooling1D(name = "second_b1_5_pool")(second_b1_5_CNN)
    second_b1_7_pool = layers.GlobalMaxPooling1D(name = "second_b1_7_pool")(second_b1_7_CNN)
    second_b1_9_pool = layers.GlobalMaxPooling1D(name = "second_b1_9_pool")(second_b1_9_CNN)
    
    second_b2_1_pool = layers.GlobalMaxPooling1D(name = "second_b2_1_pool")(second_b2_1_CNN)
    second_b2_3_pool = layers.GlobalMaxPooling1D(name = "second_b2_3_pool")(second_b2_3_CNN)
    second_b2_5_pool = layers.GlobalMaxPooling1D(name = "second_b2_5_pool")(second_b2_5_CNN)
    second_b2_7_pool = layers.GlobalMaxPooling1D(name = "second_b2_7_pool")(second_b2_7_CNN)
    second_b2_9_pool = layers.GlobalMaxPooling1D(name = "second_b2_9_pool")(second_b2_9_CNN)
    
    second_b3_1_pool = layers.GlobalMaxPooling1D(name = "second_b3_1_pool")(second_b3_1_CNN)
    second_b3_3_pool = layers.GlobalMaxPooling1D(name = "second_b3_3_pool")(second_b3_3_CNN)
    second_b3_5_pool = layers.GlobalMaxPooling1D(name = "second_b3_5_pool")(second_b3_5_CNN)
    second_b3_7_pool = layers.GlobalMaxPooling1D(name = "second_b3_7_pool")(second_b3_7_CNN)
    second_b3_9_pool = layers.GlobalMaxPooling1D(name = "second_b3_9_pool")(second_b3_9_CNN)
    
    cat = layers.Concatenate(name = "first_cat")([pep_1_pool, pep_3_pool, pep_5_pool, pep_7_pool, pep_9_pool,
                                a1_1_pool, a1_3_pool, a1_5_pool, a1_7_pool, a1_9_pool,
                                a2_1_pool, a2_3_pool, a2_5_pool, a2_7_pool, a2_9_pool,
                                a3_1_pool, a3_3_pool, a3_5_pool, a3_7_pool, a3_9_pool,
                                b1_1_pool, b1_3_pool, b1_5_pool, b1_7_pool, b1_9_pool,
                                b2_1_pool, b2_3_pool, b2_5_pool, b2_7_pool, b2_9_pool,
                                b3_1_pool, b3_3_pool, b3_5_pool, b3_7_pool, b3_9_pool])
    
    second_cat = layers.Concatenate(name = "second_cat")([second_a1_1_pool, second_a1_3_pool, second_a1_5_pool, second_a1_7_pool, second_a1_9_pool,
                                       second_a2_1_pool, second_a2_3_pool, second_a2_5_pool, second_a2_7_pool, second_a2_9_pool,
                                       second_a3_1_pool, second_a3_3_pool, second_a3_5_pool, second_a3_7_pool, second_a3_9_pool,
                                       second_b1_1_pool, second_b1_3_pool, second_b1_5_pool, second_b1_7_pool, second_b1_9_pool,
                                       second_b2_1_pool, second_b2_3_pool, second_b2_5_pool, second_b2_7_pool, second_b2_9_pool,
                                       second_b3_1_pool, second_b3_3_pool, second_b3_5_pool, second_b3_7_pool, second_b3_9_pool])
    
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)
    second_cat_dropout = layers.Dropout(dropout_rate, seed = seed)(second_cat)
    
    dense = layers.Dense(64, activation = dense_activation, name = "first_dense")(cat_dropout)
    second_dense = layers.Dense(64, activation = dense_activation, name = "second_dense")(second_cat_dropout)
    
    dense_cat = layers.Concatenate(name = "final_cat")([dense, second_dense])
    final_dense = layers.Dense(32, activation = dense_activation, name = "final_dense")(dense_cat)
    
    out = layers.Dense(1,activation = "sigmoid", name = "output_layer")(final_dense)
    
    model = keras.Model(inputs = [pep, a1, a2, a3, b1, b2, b3],
                        outputs = out)
    
    return model