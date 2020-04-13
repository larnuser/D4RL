# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 00:39:28 2019

@author: Tiago
"""
from rdkit import Chem
import tensorflow as tf
from prediction import *
import numpy as np
from Smiles_to_tokens import SmilesToTokens
from predictSMILES import *
from sascorer_calculator import SAscore
from tqdm import tqdm
from utils import * 
from tqdm import trange

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.global_variables_initializer()) # initializes weights randomly 
            
       
class Reinforcement(object):
    def __init__(self, generator, predictor, configReinforce,property_identifier):  
        """
        Constructor for the Reinforcement object.
        Parameters
        ----------
        generator: generative model that produces string of characters 
            (trajectories)
        predictor: object of any predictive model type
            predictor accepts a trajectory and returns a numerical
            prediction of desired property for the given trajectory
        configReinforce: bunch
            Parameters to use in the predictive model and get_reward function 
        property_identifier: string
            It indicates what property we want to optimize
        Returns
        -------
        object Reinforcement used for implementation of Reinforcement Learning 
        model to bias the Generator
        """

        super(Reinforcement, self).__init__()
        self.generator_unbiased = generator
        self.generator_biased = generator
        self.generator = generator
        self.configReinforce = configReinforce
        self.generator_unbiased.model.load_weights(self.configReinforce.model_name_unbiased)
        self.generator_biased.model.load_weights(self.configReinforce.model_name_unbiased)
        token_table = SmilesToTokens()
        self.table = token_table.table
        self.predictor = predictor
        self.get_reward = get_reward
        self.property_identifier = property_identifier 
        self.all_rewards = []
        self.all_losses = []
        self.threshold_greedy = 0.1
        
    def policy_gradient(self, n_batch= 7, gamma=0.98):    
            """
            Implementation of the policy gradient algorithm.
    
            Parameters:
            -----------
    
            i,j: int
                indexes of the number of iterations and number of policies, 
                respectively, to load the models properly, i.e, it's necessary 
                to load the original model just when i=0 and j=0, after that 
                it is loaded the updated one 
            n_batch: int (default 2)
                number of trajectories to sample per batch.    
            gamma: float (default 0.97)
                factor by which rewards will be discounted within one trajectory.
                Usually this number will be somewhat close to 1.0.

            Returns
            -------
            total_reward: float
                value of the reward averaged through n_batch sampled trajectories
    
            rl_loss: float
                value for the policy_gradient loss averaged through n_batch sampled
                trajectories
             """
#            opt = tf.train.AdamOptimizer(learning_rate=0.0001)
#            sess.run(tf.initialize_all_variables())
            training_rewards = []
            training_losses = []    
            for i in range(self.configReinforce.n_iterations):
                for j in trange(self.configReinforce.n_policy, desc='Policy gradient progress'):
                    
                    self.opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        #            opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,name='Adam')
                    # loss scalar in tensor format
                    self.loss = tf.zeros(dtype=tf.float32, shape=1) 
                
                    cur_reward = 0
                    
                    # Necessary object to transform new generated smiles
                    token_table = SmilesToTokens()
            
                    for _ in range(n_batch):
            
                        # Sampling new trajectory
                        reward = 0
                       
                        while reward == 0:
                            predictSMILES =  PredictSMILES(self.generator_unbiased,self.generator_biased,True,self.threshold_greedy,self.configReinforce) # generate new trajectory
                            trajectory = predictSMILES.sample() 
           
                            try:                     
                                s = trajectory[0] # because predictSMILES returns a list of smiles strings
                                if 'A' in s: # A is the padding character
                                    s = remove_padding(trajectory[0])
                                    
                                print("Validation of: ", s) 
            
                                mol = Chem.MolFromSmiles(s)
             
                                trajectory = 'G' + Chem.MolToSmiles(mol) + 'E'
                                reward = self.get_reward(self.predictor,trajectory[1:-1],self.property_identifier)
                                
                                print(reward)
                               
                            except:
                                reward = 0
                                print("\nInvalid SMILES!")
            
                        # Converting string of characters to one-hot enconding
                        trajectory_input,_ = token_table.one_hot_encode(token_table.tokenize(trajectory))
                        discounted_reward = reward
                        cur_reward += reward
                 
                        # "Following" the trajectory and accumulating the loss
                        for p in range(1,len(trajectory_input[0,:,])):
                            
                            output = self.generator_biased.model.predict(trajectory_input[:,0:p,:])[0][-1]
                            c = tf.compat.v1.math.log_softmax(self.generator_biased.model.output[0,0,:])
                            idx = np.nonzero(trajectory_input[0,p,:])
                            l = c[np.asscalar(idx[0])]
        #                    l = losses.categorical_crossentropy(-trajectory_input[0,p,:],self.generator.model.output[0,0,:])
                            self.loss = tf.math.subtract(self.loss,tf.math.multiply(l,tf.constant(discounted_reward,dtype="float32")))
                            discounted_reward = discounted_reward * gamma
                                            
                    # Doing backward pass and parameters update
                    self.loss = tf.math.divide(self.loss,tf.constant(n_batch,dtype="float32"))
        
                    cur_loss = sess.run(self.loss,feed_dict={self.generator_biased.model.input: trajectory_input})                    
        
                    # Compute the gradients for a list of variables.
        #            grads_and_vars = opt.compute_gradients(self.loss, self.generator_biased.model.trainable_weights[0:-2])
                    self.grads_and_vars = self.opt.compute_gradients(self.loss, self.generator_biased.model.trainable_weights)
                    # Ask the optimizer to apply the calculated gradients.
                    sess.run(self.opt.apply_gradients(self.grads_and_vars),feed_dict={self.generator_biased.model.input: trajectory_input})
                       
                    cur_reward = cur_reward / n_batch
               
                    # serialize model to JSON
                    model_json = self.generator_biased.model.to_json()
                    with open(self.configReinforce.model_name_biased + ".json", "w") as json_file:
                        json_file.write(model_json)
                    # serialize weights to HDF5
                    self.generator_biased.model.save_weights(self.configReinforce.model_name_biased + ".h5")
                    print("Updated model saved to disk")
                    
                    self.all_rewards.append(cur_reward)
                    
                    if len(self.all_rewards) > 2:
                        self.threshold_greedy = compute_thresh(self.all_rewards[-3:])
 
                    self.all_rewards.append(moving_average(self.all_rewards, cur_reward)) 
                    self.all_losses.append(moving_average(self.all_losses, cur_loss))
    
                plot_training_progress(self.all_rewards,self.all_losses)
        
        
    def test_generator(self, n_to_generate,iteration, original_model):
        """
        Function to generate molecules with the specified generator model. 

        Parameters:
        -----------

        n_to_generate: Integer that indicates the number of molecules to 
                    generate
        iteration: Integer that indicates the current iteration. It will be 
                   used to build the filename of the generated molecules                       
        original_model: Boolean that specifies generator model. If it is 
                        'True' we load the original model, otherwise, we 
                        load the fine-tuned model 

        Returns
        -------
        The plot containing the distribuiton of the property we want to 
        optimize. It saves one file containing the generated SMILES strings.
        """
        
        if original_model:
             self.generator.model.load_weights(self.configReinforce.model_name_unbiased)
             print("....................................")
             print("original model load_weights is DONE!")
        else:
             self.generator.model.load_weights(self.configReinforce.model_name_biased + ".h5")
             print("....................................")
             print("updated model load_weights is DONE!")
    
        
        generated = []
        pbar = tqdm(range(n_to_generate))
        for i in pbar:
            pbar.set_description("Generating molecules...")
            predictSMILES = PredictSMILES(self.generator,None,False,self.threshold_greedy,self.configReinforce)
            generated.append(predictSMILES.sample())
    
        sanitized,valid = canonical_smiles(generated,sanitize=True, throw_warning=False)# validar 
        unique_smiles = list(np.unique(sanitized))[1:]
        
#        rep = []
#        for smi in unique_smiles:
#            if smi in data_smiles:
#                rep.append(smi)
#        
#        percentage_valid = (valid/len(sanitized))*100
#        percentage_unique = (1 - (len(rep)/len(unique_smiles)))*100        
                
        if self.property_identifier != 'sas':
            prediction = self.predictor.predict(unique_smiles)
        else:
            mol_list = smiles2mol(unique_smiles)
            prediction = SAscore(mol_list)
                                                           
        plot_hist(prediction,n_to_generate,valid,self.property_identifier)
            
        with open(self.configReinforce.file_path_generated + '_' + str(len(prediction)) + '_iter'+str(iteration)+".smi", 'w') as f:
            for i,cl in enumerate(unique_smiles):
                data = str(unique_smiles[i]) + " ," +  str(prediction[i])
                f.write("%s\n" % data)  
                
        return generated, prediction
            

    def compare_models(self, n_to_generate,individual_plot):
        """
        Function to generate molecules with the both models

        Parameters:
        -----------
        n_to_generate: Integer that indicates the number of molecules to 
                    generate
                    
        individual_plot: Boolean that indicates if we want to represent the 
                         property distribution of the pre-trained model.

        Returns
        -------
        The plot that contains the distribuitons of the property we want to 
        optimize originated by the original and fine-tuned models. Besides 
        this, it saves a "generated.smi" file containing the valid generated 
        SMILES and the respective property value in "data\" folder
        """

        self.generator.model.load_weights(self.configReinforce.model_name_unbiased)
        print("\n --------- Original model LOADED! ---------")
        
        generated_unb = []
        pbar = tqdm(range(n_to_generate))
        for i in pbar:
            pbar.set_description("Generating molecules...")
            predictSMILES = PredictSMILES(self.generator,None,False,self.threshold_greedy,self.configReinforce)
            generated_unb.append(predictSMILES.sample())
    
        sanitized_unb,valid_unb = canonical_smiles(generated_unb, sanitize=False, throw_warning=False) # validar 
        unique_smiles_unb = list(np.unique(sanitized_unb))[1:]
        
        if self.property_identifier != 'sas':
            prediction_unb = self.predictor.predict(unique_smiles_unb)
        else:
            prediction_unb = SAscore(unique_smiles_unb)
        
        if individual_plot:
            plot_hist(prediction_unb,n_to_generate,valid_unb,self.property_identifier)
            
        
        # Load Biased Generator Model 
        self.generator.model.load_weights(self.configReinforce.model_name_biased + ".h5")
        print("\n --------- Updated model LOADED! ---------")
        
        generated_b = []
        pbar = tqdm(range(n_to_generate))
        for i in pbar:
            pbar.set_description("Generating molecules...")
            predictSMILES = PredictSMILES(self.generator,None,False,self.threshold_greedy,self.configReinforce)
            generated_b.append(predictSMILES.sample())
    
        sanitized_b,valid_b = canonical_smiles(generated_b, sanitize=False, throw_warning=False) # validar 
        unique_smiles_b = list(np.unique(sanitized_b))[1:]
        
        if self.property_identifier != 'sas':
            prediction_b = self.predictor.predict(unique_smiles_b)
        else:
            prediction_b = SAscore(unique_smiles_b)
        
        plot_hist_both(prediction_unb,prediction_b,n_to_generate,valid_unb,valid_b,self.property_identifier)

#        with open(self.configReinforce.file_path_generated, 'w') as f:
#            for i,cl in enumerate(unique_smiles_b):
#                data = str(unique_smiles_b[i]) + "," +  str(prediction_b[i])
#                f.write("%s\n" % data)    
#            