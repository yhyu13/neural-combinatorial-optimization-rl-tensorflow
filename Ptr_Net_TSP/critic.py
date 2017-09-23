import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, DropoutWrapper
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class Critic(object):


    def __init__(self, config):
        self.config=config

        # Data config
        self.batch_size = config.batch_size # batch size
        self.max_length = config.max_length # input sequence length (number of cities)
        self.input_dimension = config.input_dimension # dimension of a city (coordinates)

        # Network config
        self.input_embed = config.input_embed # dimension of embedding space
        self.num_neurons = config.hidden_dim # dimension of hidden states (LSTM cell)
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer

        # Baseline setup
        self.init_baseline = self.max_length/2 # good initial baseline for TSP

        # Training config
        self.is_training = not config.inference_mode


    def predict_rewards(self,input):

        with tf.variable_scope("encoder"):

            with tf.variable_scope("embedding"):
                # Embed input sequence
                W_embed =tf.get_variable("weights", [1,self.input_dimension, self.input_embed], initializer=self.initializer)
                embedded_input = tf.nn.conv1d(input, W_embed, 1, "VALID", name="embedded_input")
                # Batch Normalization
                embedded_input = tf.layers.batch_normalization(embedded_input, axis=2, training=self.is_training, name='layer_norm', reuse=None)
                
            with tf.variable_scope("dynamic_rnn"):
                # Encode input sequence
                cell1 = LSTMCell(self.num_neurons, initializer=self.initializer)  # cell = DropoutWrapper(cell, output_keep_prob=dropout) or MultiRNNCell([cell] * num_layers)
                # Return the output activations [Batch size, Sequence Length, Num_neurons] and last hidden state as tensors.
                encoder_output, encoder_state = tf.nn.dynamic_rnn(cell1, embedded_input, dtype=tf.float32)
                frame = tf.reduce_mean(encoder_output, 1) # [Batch size, Sequence Length, Num_neurons] to [Batch size, Num_neurons]

            with tf.variable_scope("ffn"):
                # ffn 1
                h0 = tf.layers.dense(frame, self.num_neurons, activation=tf.nn.relu, kernel_initializer=self.initializer)
                # ffn 2
                w1 =tf.get_variable("w1", [self.num_neurons, 1], initializer=self.initializer)
                b1 = tf.Variable(self.init_baseline, name="b1")
                self.predictions = tf.squeeze(tf.matmul(h0, w1)+b1)