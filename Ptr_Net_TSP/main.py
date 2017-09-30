#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import DataGenerator
from actor import Actor
from config import get_config, print_config
from tsp_with_ortools import Solver


### Model: Critic (state value function approximator) = slim mean RNN (parametric baseline ***) w/o moving baseline
###        Encoder = RNN
###        Decoder init_state = Encoder last_state                        
###        Decoder inputs = Encoder outputs
###        Decoder Glimpse = on Attention_g (mask - current)
###        No Residual connections


def main():
    # Get running configuration
    config, _ = get_config()
    print_config()

    # Build tensorflow graph from config
    print("Building graph...")
    actor = Actor(config)

    # Saver to save & restore all the variables.
    variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)  

    print("Starting session...")
    with tf.Session() as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Restore variables from disk.
        if config.restore_model==True:
            saver.restore(sess, "save/"+config.restore_from+"/actor.ckpt")
            print("Model restored.")
    
        # Initialize data generator
        solver = Solver(actor.max_length) ###### ######
        training_set = DataGenerator(solver)

        # Training mode
        if not config.inference_mode:

            # Summary writer
            writer = tf.summary.FileWriter(config.log_dir, sess.graph)

            print("Starting training...")
            for i in tqdm(range(config.nb_epoch)):
                # Get feed dict
                input_batch = training_set.train_batch(actor.batch_size, actor.max_length, actor.input_dimension)
                feed = {actor.input_: input_batch}

                # Forward pass & train step
                summary, train_step1, train_step2 = sess.run([actor.merged, actor.train_step1, actor.train_step2], feed_dict=feed)

                if i % 100 == 0:
                    writer.add_summary(summary,i)

                # Save the variables to disk
                if i % max(1,int(config.nb_epoch/5)) == 0 and i!=0 :
                    save_path = saver.save(sess,"save/"+config.save_to+"/tmp.ckpt", global_step=i)
                    print("\n Model saved in file: %s" % save_path)
        
            print("Training COMPLETED !")
            saver.save(sess,"save/"+config.save_to+"/actor.ckpt")


        # Inference mode
        else:

            targets=[]
            predictions=[]
            predictions_NN=[]

            for __ in tqdm(range(1000)): # num of examples

                # Get feed_dict (single input)
                seed_ = 1+__
                input_batch, or_sequence = training_set.test_batch(actor.batch_size, actor.max_length, actor.input_dimension, seed=seed_) # seed=0 means None
                feed = {actor.input_: input_batch}
                
                # Solve instance (OR tools)
                opt_seq, opt_length = training_set.solve_instance(or_sequence)
                targets.append(opt_length/100)
                #print('\n Optimal length:',opt_length/100)

                # Solve instance (NN heuristic)
                NN_seq, NN_length = training_set.solve_NN_policy(or_sequence)
                predictions_NN.append(NN_length/100)
                #print(' NN prediction:',NN_length/100)
                
                # Solve instance (Ptr-Net)
                best_lengths = []
                best_trips = []
                for ___ in range(1): ###############################
                    # Sample solutions
                    permutation, circuit_length = sess.run([actor.positions, actor.distances], feed_dict=feed)
                    # Find best solution
                    j = np.argmin(circuit_length)
                    best_lengths.append(circuit_length[j])
                    best_trips.append(input_batch[j][permutation[j][:-1]])
                    # New input = Shuffle or Predictions
                    input_batch = training_set.shuffle_batch(input_batch)   
                    feed = {actor.input_: input_batch}
                best_lengths = np.array(best_lengths)
                best_trips = np.array(best_trips)
                round_ = np.argmin(best_lengths)
                #print('round',round_)
                predictions.append(best_lengths[round_])
                #print(' Ptr_Net prediction:',best_lengths[round_])

                # plot solution
                #training_set.visualize_2D_trip(opt_seq/100-0.5)
                #training_set.visualize_2D_trip(best_trips[round_])

            delta = np.asarray(predictions_NN)/np.asarray(targets)
            delta_ = np.asarray(predictions)/np.asarray(targets)
            print('\n NN Predictions: \n',np.mean(predictions_NN))
            print('\n Ptr-Net Predictions: \n',np.mean(predictions))
            print('\n Targets: \n', np.mean(targets))
            print('\n Average deviation (NN prediction): \n', np.mean(delta))
            print('\n Average deviation (best prediction): \n', np.mean(delta_))

            n, bins, patches = plt.hist(delta, 50, facecolor='r', alpha=0.75)
            n, bins, patches = plt.hist(delta_, 50, facecolor='b', alpha=0.75)

            plt.xlabel('Prediction/target')
            plt.ylabel('Counts')
            plt.title('Comparison to Google OR tools')
            plt.axis([0.9, 1.4, 0, 500])
            plt.grid(True)
            plt.show()





if __name__ == "__main__":
    main()
