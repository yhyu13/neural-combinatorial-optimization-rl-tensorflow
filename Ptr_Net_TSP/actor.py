import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, DropoutWrapper
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import DataGenerator
from decoder import Pointer_decoder
from critic import Critic
from config import get_config, print_config



# Tensor summaries for TensorBoard visualization
def variable_summaries(name,var, with_max_min=False):
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    if with_max_min == True:
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))



class Actor(object):


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

        # Training config (actor)
        self.global_step = tf.Variable(0, trainable=False, name="global_step") # global step
        self.lr1_start = config.lr1_start # initial learning rate
        self.lr1_decay_rate= config.lr1_decay_rate # learning rate decay rate
        self.lr1_decay_step= config.lr1_decay_step # learning rate decay step
        self.is_training = not config.inference_mode

        # Training config (critic)
        self.global_step2 = tf.Variable(0, trainable=False, name="global_step2") # global step
        self.lr2_start = config.lr1_start # initial learning rate
        self.lr2_decay_rate= config.lr1_decay_rate # learning rate decay rate
        self.lr2_decay_step= config.lr1_decay_step # learning rate decay step

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_ = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.input_dimension], name="input_coordinates")

        self.build_permutation()
        self.build_critic()
        self.build_reward()
        self.build_optim()
        self.merged = tf.summary.merge_all()


    def build_permutation(self):

        with tf.variable_scope("encoder"):

            with tf.variable_scope("embedding"):
                # Embed input sequence
                W_embed =tf.get_variable("weights", [1,self.input_dimension, self.input_embed], initializer=self.initializer)
                embedded_input = tf.nn.conv1d(self.input_, W_embed, 1, "VALID", name="embedded_input")
                # Batch Normalization
                embedded_input = tf.layers.batch_normalization(embedded_input, axis=2, training=self.is_training, name='layer_norm', reuse=None)
                
            with tf.variable_scope("dynamic_rnn"):
                # Encode input sequence
                cell1 = LSTMCell(self.num_neurons, initializer=self.initializer)  # cell = DropoutWrapper(cell, output_keep_prob=dropout) or MultiRNNCell([cell] * num_layers)
                # Return the output activations [Batch size, Sequence Length, Num_neurons] and last hidden state as tensors.
                encoder_output, encoder_state = tf.nn.dynamic_rnn(cell1, embedded_input, dtype=tf.float32)

        with tf.variable_scope('decoder'):
            # Ptr-net returns permutations (self.positions), with their log-probability for backprop
            self.ptr = Pointer_decoder(encoder_output, self.config)
            self.positions, self.log_softmax = self.ptr.loop_decode(encoder_state)
            variable_summaries('log_softmax',self.log_softmax, with_max_min = True)


    def build_critic(self):

        with tf.variable_scope("critic"):
            # Critic predicts reward (parametric baseline for REINFORCE)
            self.critic = Critic(self.config)
            self.critic.predict_rewards(self.input_)
            variable_summaries('predictions',self.critic.predictions, with_max_min = True)

    def build_reward(self):

        with tf.name_scope('permutations'):

            # Reorder input % tour
            self.permutations = tf.stack([tf.tile(tf.expand_dims(tf.range(self.batch_size,dtype=tf.int32),1),[1,self.max_length+1]),self.positions],2)
            self.ordered_input_ = tf.gather_nd(self.input_,self.permutations)
            self.ordered_input_ = tf.transpose(self.ordered_input_,[2,1,0]) # [batch size, seq length +1, features] to [features, seq length +1, batch_size]

            # Ordered coordinates
            ordered_x_ = self.ordered_input_[0] # [seq length +1, batch_size]
            delta_x2 = tf.transpose(tf.square(ordered_x_[1:]-ordered_x_[:-1]),[1,0]) # [batch_size, seq length]        delta_x**2
            ordered_y_ = self.ordered_input_[1] # [seq length +1, batch_size]
            delta_y2 = tf.transpose(tf.square(ordered_y_[1:]-ordered_y_[:-1]),[1,0]) # [batch_size, seq length]        delta_y**2

        with tf.name_scope('environment'):

            # Get tour length (euclidean distance)
            inter_city_distances = tf.sqrt(delta_x2+delta_y2) # sqrt(delta_x**2 + delta_y**2) this is the euclidean distance between each city: depot --> ... ---> depot      [batch_size, seq length]
            self.distances = tf.reduce_sum(inter_city_distances, axis=1) # [batch_size]

            # Define reward
            self.reward = tf.cast(self.distances,tf.float32)
            variable_summaries('reward',self.reward, with_max_min = True)


    def build_optim(self):
        # Update moving_mean and moving_variance for batch normalization layers
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            with tf.name_scope('reinforce'):
                # Actor learning rate
                self.lr1 = tf.train.exponential_decay(self.lr1_start, self.global_step, self.lr1_decay_step,self.lr1_decay_rate, staircase=False, name="learning_rate1")
                # Optimizer
                self.opt1 = tf.train.AdamOptimizer(learning_rate=self.lr1,beta1=0.9,beta2=0.99, epsilon=0.0000001)
                # Discounted reward
                self.reward_baseline = tf.stop_gradient(self.reward - self.critic.predictions) # [Batch size, 1] 
                variable_summaries('reward_baseline',self.reward_baseline, with_max_min = True)
                # Loss
                self.loss1 = tf.reduce_mean(self.reward_baseline*self.log_softmax,0)
                tf.summary.scalar('loss1', self.loss1)
                # Minimize step
                gvs = self.opt1.compute_gradients(self.loss1)
                capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs if grad is not None] # L2 clip
                self.train_step1 = self.opt1.apply_gradients(capped_gvs, global_step=self.global_step)

            with tf.name_scope('state_value'):
                # Critic learning rate
                self.lr2 = tf.train.exponential_decay(self.lr2_start, self.global_step2, self.lr2_decay_step,self.lr2_decay_rate, staircase=False, name="learning_rate1")
                # Optimizer
                self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr2,beta1=0.9,beta2=0.99, epsilon=0.0000001)
                # Loss
                weights_ = 1.0 #weights_ = tf.exp(self.log_softmax-tf.reduce_max(self.log_softmax)) # probs / max_prob
                self.loss2 = tf.losses.mean_squared_error(self.reward, self.critic.predictions, weights = weights_)
                tf.summary.scalar('loss2', self.loss1)
                # Minimize step
                gvs2 = self.opt2.compute_gradients(self.loss2)
                capped_gvs2 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs2 if grad is not None] # L2 clip
                self.train_step2 = self.opt1.apply_gradients(capped_gvs2, global_step=self.global_step2)




if __name__ == "__main__":
    # get config
    config, _ = get_config()

    # Build Model and Reward from config
    actor = Actor(config)

    print("Starting training...")
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print_config()

        solver = [] #Solver(actor.max_length)
        training_set = DataGenerator(solver)

        nb_epoch=2
        for i in tqdm(range(nb_epoch)): # epoch i

            # Get feed_dict
            input_batch = training_set.train_batch(actor.batch_size, actor.max_length, actor.input_dimension)
            feed = {actor.input_: input_batch}

            # Forward pass
            permutation, distances, reward, lp, loss1, predictions, _ = sess.run([actor.positions, actor.distances, actor.reward, actor.log_softmax, actor.loss1, actor.critic.predictions, actor.loss2], feed_dict=feed)
            #print(' Permutation \n',permutation)
            print(' Reward \n',reward)
            #print(' Predictions \n',predictions)
            print(' LP \n',lp)
            print(' LP discount \n', lp-np.max(lp))
            print(' Probs rel \n',np.exp(lp-np.max(lp))) # = probs / max_prob
            #print(' Loss1 \n',loss1)


        variables_names = [v.name for v in tf.global_variables() if 'Adam' not in v.name]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: ", k, "Shape: ", v.shape)