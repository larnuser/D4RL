# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:22:10 2019

@author: Tiago
"""
import csv
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sascorer_calculator import SAscore
from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from rdkit.Chem import Crippen
import time
from bunch import Bunch
from tqdm import tqdm
from predictSMILES import *

def load_config(config_file):
    """
    This function loads the configuration file in .json format.
    ----------
    config_file: name of the configuration file
    
    Returns
    -------
    Configuration file
    """
    with open(config_file, 'r') as config_file:
        config_dict = json.load(config_file)
        config = Bunch(config_dict)
        exp_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    return config, exp_time;

def reading_csv(config,property_identifier):
    """
    This function loads the labels (pIC50) of the property specified by the identifier.
    ----------
    config: configuration file
    property_identifier: Identifier of the property to optimize.
    
    Returns
    -------
    raw_labels: Returns the labels in a numpy array. 
    """
    if property_identifier == "jak2":
        file_path = config.file_path_jak2
    elif property_identifier == "logP":
        file_path = config.file_path_logP
    elif property_identifier == "kor":
        file_path = config.file_path_kor
        
    raw_labels = []
        
    with open(file_path, 'r') as csvFile:
        reader = csv.reader(csvFile)
        
        it = iter(reader)   
        for row in it:
            if property_identifier == "jak2" or property_identifier == "kor":
                try:
                    raw_labels.append(float(row[1]))
#                    raw_labels.append(row[0])
                except:
                    pass
            elif property_identifier == "logP":
                raw_labels.append(float(row[2]))

    return raw_labels


def smilesDict(tokens):
    """
    This function extracts the dictionary that makes the correspondence between
    each charachter and an integer (Tokenization).
    ----------
    tokens: Set of characters
    
    Returns
    -------
    tokenDict: Returns the dictionary that maps characters to integers
    """
    tokenDict = dict((token, i) for i, token in enumerate(tokens))
    return tokenDict

def pad_seq(smiles,tokens,paddSize):
    """
    This function performs the padding of each SMILE.
    ----------
    smiles: Set of SMILES strings with different sizes;
    tokens: Set of characters;
    paddSize: Integer that specifies the maximum size of the padding    
    
    Returns
    -------
    newSmiles: Returns the padded smiles, all with the same size.
    """
    maxSmile= max(smiles, key=len)
    maxLength = 0
    
    if paddSize != 0:
        maxLength = paddSize
    else:
        maxLength = len(maxSmile) 

    for i in range(0,len(smiles)):
        if len(smiles[i]) < maxLength:
            smiles[i] = smiles[i] + tokens[-1]*(maxLength - len(smiles[i]))

    return smiles,maxLength
             
def smiles2idx(smiles,tokenDict):  
    """
    This function transforms each SMILES character to the correspondent integer,
    according the token dictionary.
    ----------
    smiles: Set of SMILES strings with different sizes;
    tokenDict: Dictionary that maps the characters to integers;    
    
    Returns
    -------
    newSmiles: Returns the transformed smiles, with the characters replaced by 
    the numbers. 
    """         
    newSmiles =  np.zeros((len(smiles), len(smiles[0])))
    for i in range(0,len(smiles)):
        try:
            for j in range(0,len(smiles[i])):          
                newSmiles[i,j] = tokenDict[smiles[i][j]]
        except:
            pass
    return newSmiles


def scalarization(reward_kor,reward_sas,scalarMode,weights,pred_range):
    """
    This function computes a linear scalarization of the two objectives to 
    obtain a unique reward.
    ----------
    reward_kor: Reward obtained from kor property;
    reward_sas: Reward obtained from sas property;
    scalarMode: Type of scalarization;
    weights: List with the weights for two properties: weights[0]: kor, weights[1]: sas
    
    Returns
    -------
    Returns the scalarized reward
    """
    if scalarMode == 'linear' or scalarMode == 'linear_adaptative':
    
        w_kor = weights[0]
        w_sas = weights[1]
        
        return w_kor*reward_kor + w_sas*reward_sas
    
    elif scalarMode == 'chebyshev':
        dist_sas = 0 
        dist_kor = 0
        
        w_kor = weights[0]
        w_sas = weights[1]
        
        max_kor = pred_range[0]
        min_kor = pred_range[1]
        min_sas = -pred_range[2]
        max_sas = -pred_range[3]
        
        rescaled_rew_kor = (reward_kor - min_kor)/(max_kor-min_kor)
        
        if rescaled_rew_kor < 0:
            rescaled_rew_kor = 0
        elif rescaled_rew_kor > 1:
            rescaled_rew_kor = 1
            
        rescaled_rew_sas = (reward_sas - min_sas)/(max_sas-min_sas)
        
        if rescaled_rew_sas < 0:
            rescaled_rew_sas = 0
        elif rescaled_rew_sas > 1:
            rescaled_rew_sas = 1
        
        dist_sas = abs(rescaled_rew_sas - max_sas)*w_sas
        dist_kor = abs(rescaled_rew_kor - max_kor)*w_kor
        
        if dist_sas > dist_kor:
            return dist_sas
        else:
            return dist_kor

def canonical_smiles(smiles,sanitize=True, throw_warning=False):
    """
    Takes list of generated SMILES strings and returns the list of valid SMILES.
    Parameters
    ----------
    smiles: list
        list of SMILES strings to validate
    sanitize: bool (default True)
        parameter specifying whether to sanitize SMILES or not.
            For definition of sanitized SMILES check
            http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol
    throw_warning: bool (default False)
        parameter specifying whether warnings will be thrown if a SMILES is
        invalid
    Returns
    -------
    new_smiles: list of valid SMILES (if it is valid and has <65 characters)
    and NaNs if SMILES string is invalid
    valid: number of valid smiles, regardless of the its size
        
    """
    new_smiles = []
    valid = 0
    for sm in smiles:
        try:
            mol = Chem.MolFromSmiles(sm[0], sanitize=sanitize)
            s = Chem.MolToSmiles(mol)
            
            if len(s) <= 60:
                new_smiles.append(s)       
            else:
                new_smiles.append('')
            valid = valid + 1 
        except:
            if throw_warning:
                warnings.warn(sm + ' can not be canonized: invalid '
                                   'SMILES string!', UserWarning)
            new_smiles.append('')
    return new_smiles,valid

def smiles2mol(smiles_list):
    """
    Function that converts a list of SMILES strings to a list of RDKit molecules 
    Parameters
    ----------
    smiles: List of SMILES strings
    ----------
    Returns list of molecules objects 
    """
    mol_list = []
    if isinstance(smiles_list, str):
        mol = Chem.MolFromSmiles(smiles_list, sanitize=True)
        mol_list.append(mol)
    else:
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            mol_list.append(mol)
    return mol_list

def plot_hist(prediction, n_to_generate,valid,property_identifier):
    """
    Function that plots the predictions's distribution of the generated SMILES 
    strings
    Parameters
    ----------
    prediction: list with the desired property predictions.
    n_to_generate: number of generated SMILES.
    valid: number of valid smiles, regardless of the its size.
    property_identifier: String identifying the property 
    """
    prediction = np.array(prediction)
    x_label = ''
    plot_title = '' 
    
    print("Proportion of valid SMILES:", valid/n_to_generate)
    
    if property_identifier == "jak2" or property_identifier == "kor":
        print("Max of pIC50: ", np.max(prediction))
        print("Mean of pIC50: ", np.mean(prediction))
        print("Min of pIC50: ", np.min(prediction))
        x_label = "Predicted pIC50"
        plot_title = "Distribution of predicted pIC50 for generated molecules"
    elif property_identifier == "sas":
        print("Max SA score: ", np.max(prediction))
        print("Mean SA score: ", np.mean(prediction))
        print("Min SA score: ", np.min(prediction))
        x_label = "Calculated SA score"
        plot_title = "Distribution of SA score for generated molecules"
        
    elif property_identifier == "logP":
        percentage_in_threshold = np.sum((prediction >= 0.0) & 
                                     (prediction <= 5.0))/len(prediction)
        print("Percentage of predictions within drug-like region:", percentage_in_threshold)
        print("Average of log_P: ", np.mean(prediction))
        print("Median of log_P: ", np.median(prediction))
        plt.axvline(x=0.0)
        plt.axvline(x=5.0)
        x_label = "Predicted LogP"
        plot_title = "Distribution of predicted LogP for generated molecules"
        
#    sns.set(font_scale=1)
    ax = sns.kdeplot(prediction, shade=True,color = 'b')
    ax.set(xlabel=x_label,
           title=plot_title)
    plt.show()
    
def plot_hist_both(prediction_unb,prediction_b, n_to_generate,valid_unb,valid_b,property_identifier):
    """
    Function that plots the predictions's distribution of the generated SMILES 
    strings, obtained by the unbiased and biased generators.
    Parameters
    ----------
    prediction_unb: list with the desired property predictions of unbiased 
                    generator.
    prediction_b: list with the desired property predictions of biased 
                    generator.
    n_to_generate: number of generated SMILES.
    valid_unb: number of valid smiles of the unbiased generator, regardless of 
            the its size.
    valid_b: number of valid smiles of the biased generator, regardless of 
            the its size.
    property_identifier: String identifying the property 
    """
    prediction_unb = np.array(prediction_unb)
    prediction_b = np.array(prediction_b)
    
    legend_unb = ''
    legend_b = '' 
    label = ''
    plot_title = ''
    
    print("\nProportion of valid SMILES (UNB,B):", valid_unb/n_to_generate,valid_b/n_to_generate )
    if property_identifier == 'jak2' or property_identifier == "kor":
        legend_unb = 'Unbiased pIC50 values'
        legend_b = 'Biased pIC50 values'
        print("Max of pIC50: (UNB,B)", np.max(prediction_unb),np.max(prediction_b))
        print("Mean of pIC50: (UNB,B)", np.mean(prediction_unb),np.mean(prediction_b))
        print("Min of pIC50: (UNB,B)", np.min(prediction_unb),np.min(prediction_b))
    
        label = 'Predicted pIC50'
        plot_title = 'Distribution of predicted pIC50 for generated molecules'
        
    elif property_identifier == "sas":
        legend_unb = 'Unbiased SA score values'
        legend_b = 'Biased SA score values'
        print("Max of SA score: (UNB,B)", np.max(prediction_unb),np.max(prediction_b))
        print("Mean of SA score: (UNB,B)", np.mean(prediction_unb),np.mean(prediction_b))
        print("Min of SA score: (UNB,B)", np.min(prediction_unb),np.min(prediction_b))
    
        label = 'Predicted SA score'
        plot_title = 'Distribution of SA score values for generated molecules'   
    elif property_identifier == 'logP':
        legend_unb = 'Unbiased logP values'
        legend_b = 'Biased logP values'
        
        percentage_in_threshold_unb = np.sum((prediction_unb >= 0.0) & 
                                            (prediction_unb <= 5.0))/len(prediction_unb)
        percentage_in_threshold_b = np.sum((prediction_b >= 0.0) & 
                                 (prediction_b <= 5.0))/len(prediction_b)
        print("% of predictions within drug-like region (UNB,B):", 
          percentage_in_threshold_unb,percentage_in_threshold_b)
        print("Average of log_P: (UNB,B)", np.mean(prediction_unb),np.mean(prediction_b))
        print("Median of log_P: (UNB,B)", np.median(prediction_unb),np.median(prediction_b))
    
        label = 'Predicted logP'
        plot_title = 'Distribution of predicted LogP for generated molecules'
        plt.axvline(x=0.0)
        plt.axvline(x=5.0)
        
    v1 = pd.Series(prediction_unb, name=legend_unb)
    v2 = pd.Series(prediction_b, name=legend_b)
   
    
    ax = sns.kdeplot(v1, shade=True,color='b')
    sns.kdeplot(v2, shade=True,color='r')

    ax.set(xlabel=label, 
           title=plot_title)

    plt.show()
    
def denormalization(predictions,labels): 
    """
    This function performs the denormalization step.
    ----------
    predictions: list with the desired property predictions.
    labels: list with the real values of the desired property
    
    Returns
    -------
    predictions: Returns the denormalized predictions.
    """
    for l in range(len(predictions)):
        
        q1 = np.percentile(labels,5)
        q3 = np.percentile(labels,95)
       
        for c in range(len(predictions[0])):
            predictions[l,c] = (q3 - q1) * predictions[l,c] + q1
#            predictions[l,c] = predictions[l,c] * sd_train + m_train
          
    return predictions

def get_reward(predictor, smile,property_identifier):
    """
    This function takes the predictor model and the SMILES string and returns 
    a numerical reward.
    ----------
    predictor: object of the predictive model that accepts a trajectory
        and returns a numerical prediction of desired property for the given 
        trajectory
    smile: SMILES string of the generated molecule
    property_identifier: String identifying the property 
    Returns
    -------
    Outputs the reward value for the predicted property of the input SMILES 
    """
    
    list_ss = [smile] 

    if property_identifier == 'sas':
        list_ss[0] = Chem.MolFromSmiles(smile)
        reward_list = SAscore(list_ss)
        reward = reward_list[0] 
    else:
    
        pred = predictor.predict(list_ss)
    
        reward = np.exp(pred/4 - 1)
#    reward = np.exp(pred/10) - 1
    return reward
  
#    if (pred >= 1) and (pred <= 4):
#        return 0.5
#    else:
#        return -0.01
    

def get_reward_MO(predictor, smile):
    """
    This function takes the predictor model and the SMILES string and returns 
    a numerical reward.
    ----------
    predictor: object of the predictive model that accepts a trajectory
        and returns a numerical prediction of desired property for the given 
        trajectory
    smile: SMILES string of the generated molecule
    
    Returns
    -------
    Outputs the reward value for the predicted property of the input SMILES 
    """
    
    list_ss = [smile] 

    # SAScore prediction
    list_ss[0] = Chem.MolFromSmiles(smile)
    sas_list = SAscore(list_ss)
    sas_smiles = sas_list[0] 
    reward_sas = np.exp(-sas_smiles + 3)

    # pIC50 for kor prediction
    list_ss = [smile] 
    pred = predictor.predict(list_ss)

    reward_kor = np.exp(pred/4 - 1)
#    reward = np.exp(pred/10) - 1
    return reward_kor,reward_sas

def moving_average(previous_values, new_value, ma_window_size=10): 
    """
    This function performs a simple moving average between the last 9 elements
    and the last one obtained.
    ----------
    previous_values: list with previous values 
    new_value: new value to append, to compute the average with the last ten 
               elements
    
    Returns
    -------
    Outputs the average of the last 10 elements 
    """
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma

def plot_training_progress(training_rewards,training_losses):
    """
    This function plots the progress of the training performance
    ----------
    training_rewards: list with previous reward values
    training_losses: list with previous loss values
    """
    plt.plot(training_rewards)
    plt.xlabel('Training iterations')
    plt.ylabel('Average rewards')
    plt.show()
    plt.plot(training_losses)
    plt.xlabel('Training iterations')
    plt.ylabel('Average losses')
    plt.show()
    
def plot_individual_rewds(rew_sas,rew_kor):
    """
    Multi-Objective Scenario.
    This function plots the progress of the training performance for both 
    properties we want to optimize.
    ----------
    rew_sas: list with previous reward values regarding sas
    rew_kor: list with previous reward values regarding kor
    """
    plt.plot(rew_sas)
    plt.xlabel('Training iterations')
    plt.ylabel('Average rewards sas')
    plt.show()
    plt.plot(rew_kor)
    plt.xlabel('Training iterations')
    plt.ylabel('Average losses kor')
    plt.show()
    
def plot_evolution(pred_original,pred_iter25):
    """
    This function plots the comparison of the property's distribution between
    the original model and after 25 iterations
    ----------
    pred_original: list with predictions from the original Generator
    pred_iter25: list with predictions from Generator after 25 iterations
    """
    pred_original = np.array(pred_original)
#    pred_iter1 = np.array(pred_iter1)
    pred_iter25 = np.array(pred_iter25)
 
    legend_0 = "Original Distribution" 
#    legend_1 = "Iteration 1" 
    legend_25 = "Iteration 25" 

    label = 'Predicted pIC50 for KOR'
    plot_title = 'Distribution of predicted pIC50 for generated molecules'
        
    v1 = pd.Series(pred_original, name=legend_0)
#    v2 = pd.Series(pred_iter1, name=legend_1)
    v3 = pd.Series(pred_iter25, name=legend_25)
 
    ax = sns.kdeplot(v1, shade=True,color='g')
#    sns.kdeplot(v2, shade=True,color='g')
    sns.kdeplot(v3, shade=True,color='r')

    ax.set(xlabel=label, 
           title=plot_title)

    plt.show()
def remove_padding(trajectory):    
    """
    Function that removes the padding character
    Parameters
    ----------
    trajectory: Generated molecule padded with "A"
    Returns
    -------
    SMILE string without the padding character 
    """
    if 'A' in trajectory:
        
        firstA = trajectory.find('A')
        trajectory = trajectory[0:firstA]
    return trajectory

def generate2file(predictor,generator,configReinforce,n2generate,original_model):
    """
    Function that generates new SMILES strings and predicts some properties. The
    SMILES and predictions are saved to files.
    Parameters
    ----------
    predictor: Predictor model
    generator: Generator model
    configReinforce: Configuration file
    n2generate: Number of SMILES strings to generate
    original_model: Boolean that indicates if we use the Original or the Biased
    Generator
    Returns
    -------
    Saves the file with the newly generated SMILES 
    """    
    if original_model:
        model_type = 'original'
    else:
        model_type = 'biased'
    
    generated = []
    pbar = tqdm(range(n2generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        predictSMILES = PredictSMILES(generator,None,False,0,configReinforce)
        generated.append(predictSMILES.sample())

    sanitized,valid = canonical_smiles(generated,sanitize=True, throw_warning=False)
    unique_smiles = list(np.unique(sanitized))[1:]
    
    mol_list= smiles2mol(unique_smiles)
    prediction_sas = SAscore(mol_list)
    

    if predictor != None:
        prediction_prop = predictor.predict(unique_smiles)
        with open("Generated/generated_prop_"+model_type+".smi", 'w') as f:
#                f.write("SMILES, Property, SA_score, LogP\n" ) 
            for i,cl in enumerate(unique_smiles):
                data = str(unique_smiles[i]) + "," +  str(prediction_prop[i]) + "," + str(prediction_sas[i]) 
                f.write("%s\n" % data)    
    else:
        with open("Generated/generated_sas"+model_type+".smi", 'w') as f:
#                f.write("SMILES, SA_score, LogP\n" )
            for i,cl in enumerate(unique_smiles):
                data = str(unique_smiles[i]) + "," + str(prediction_sas[i]) 
                f.write("%s\n" % data)  
            
def countRepeated(smile):
    """
    This function that counts the number of repeated characters and attributes
    a penalty. Also, this function counts the number of different characters 
    and gives a bonus.  
    Parameters
    ----------
    smiles: SMILES string to be analysed
    Returns
    -------
    bonus, penalty: floats that penalize the repetition and gives a bonus to 
    diversity in each SMILES
    """
    
    n_different = len(list(set(smile)))
    bonus = 0
    if n_different < 6:
        bonus = 0
    elif n_different >= 6 and n_different <= 11:
        bonus = 0.3
    elif n_different > 11:
        bonus = 0.5
        
    
    repetitions = []
    repeated = ''
    for i,symbol in enumerate(smile):
        if i > 0:
            if symbol == smile[i-1]:
                repeated = repeated + symbol
            else:
                
                if len(repeated) > 4:
                    repetitions.append(repeated)

                repeated = ''
    penalty = 0
    
    for rep in repetitions:
        penalty = penalty + 0.1*len(rep)

    return penalty,bonus

def compute_thresh(rewards):
    """
    This function verifies the evolution of the reward and based on the
    evolution we assign different values to the threshold to decide which
    Generator will be used.
    Parameters
    ----------
    reward: List with the last three rewards values
    Returns
    -------
    threshold: float indicating threshold value to decide which generator
    """
    reward_t_2 = rewards[0]
    reward_t_1 = rewards[1]
    reward_t = rewards[2]
    q_t_1 = reward_t_2/reward_t_1
    q_t = reward_t_1/reward_t
    
    threshold = 0
    if q_t_1 < 1 and q_t < 1:
        threshold = 0.05
    elif q_t_1 > 1 and q_t > 1:
        threshold = 0.3
    else:
        threshold = 0.2
        
    return threshold

def compute_weights(rew_kor,rew_sas):
    """
    This function verifies the evolution of the reward and based on the
    evolution we assign different values to the weights to perform the
    scalarization step.
    Parameters
    ----------
    rew_kor: List with the last three rewards values regarding kor objective
    rew_sas: List with the last three rewards values regarding sas objective
    Returns
    -------
    weights: List with weights to each objective.
    """
    weights = []
    
    reward_kor_t_2 = rew_kor[0]
    reward_kor_t_1 = rew_kor[1]
    reward_kor_t = rew_kor[2]
    
    reward_sas_t_2 = rew_sas[0]
    reward_sas_t_1 = rew_sas[1]
    reward_sas_t = rew_sas[2]
    
    q_kor_t_1 = reward_kor_t_2/reward_kor_t_1
    q_kor_t = reward_kor_t_1/reward_kor_t
    
    q_sas_t_1 = reward_sas_t_2/reward_sas_t_1
    q_sas_t = reward_sas_t_1/reward_sas_t
    
    w_kor = 0
    w_sas = 0
    if q_kor_t_1 > 1 and q_kor_t > 1:
        w_kor = 0.7
    elif q_kor_t > 1:
        w_kor = 0.6
    else:
        w_kor = 0.5
        
    w_sas = 1 - w_kor
    
    weights.append(w_kor)
    weights.append(w_sas)
    
    return weights
    