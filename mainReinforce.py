# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 19:39:49 2019

@author: Tiago
"""
import os
import tensorflow as tf
from keras import backend as K
from reinforce import Reinforcement
from keras.models import Sequential
from model import Model 
from prediction import Predictor
from predictSMILES import *
from utils import *


#from tensorflow.python.client import device_lib 
#print(device_lib.list_local_devices())

config_file = 'configReinforce.json' 
property_identifier = 'kor' # It can be 'kor', 'sas', 'logP', or 'jak2'

os.environ["CUDA_VISIBLE_DEVICES"]="0"
session = tf.compat.v1.Session()
K.set_session(session)
 
def main():
    
    """
    Main routine
    """
    # load configuration file
    configReinforce,exp_time=load_config(config_file)
    
    # Load generator
    generator_model = Sequential()
    generator_model = Model(configReinforce)
    generator_model.model.load_weights(configReinforce.model_name_unbiased)

    
    if property_identifier != 'sas': # To compute SA score it's not necessary to have a Predictor model
        # Load the predictor model
        predictor = Predictor(configReinforce,property_identifier)
    else:
        predictor = None
  
    # Create reinforcement learning object
    RL_obj = Reinforcement(generator_model, predictor,configReinforce,property_identifier)
    
    #   SMILES generation with unbiased model 
    smiles_original, prediction_original = RL_obj.test_generator(configReinforce.n_to_generate,0,True)
    
    #  Training Generator with RL    
    RL_obj.policy_gradient()
    
    # SMILES generation after 25 training iterations 
    smiles_epoch25,prediction_epoch25 = RL_obj.test_generator(configReinforce.n_to_generate,25, False)
   
    plot_evolution(prediction_original,prediction_epoch25)
    
    # To directly compare the original and biased models several times
    for k in range(10):
        print("BIASED GENERATION: " + str(k))
        RL_obj.compare_models(configReinforce.n_to_generate,True)

if __name__ == '__main__':
    main()