import tensorflow as tf
import os
import pickle
from Utils.utils import *


w_init = lambda:tf.random_normal_initializer(stddev=0.02)

class Model(object):

    def __init__(self, config):
        self.config = config
        self.net_shape = config['net_shape']
        self.net_input_dim = config['net_input_dim']
        self.is_init = config['is_init']
        self.pretrain_params_path = config['pretrain_params_path']

        self.num_net_layers = len(self.net_shape)

        if self.is_init:
            if os.path.isfile(self.pretrain_params_path):
                with open(self.pretrain_params_path, 'rb') as handle:
                    self.W_init, self.b_init = pickle.load(handle)


    def forward_net(self, x, drop_prob, modal, reuse=False):

        with tf.variable_scope(modal+'_encoder', reuse=reuse) as scope:
            cur_input = x
            print(cur_input.get_shape())

            # ============encoder===========
            struct = self.net_shape
            for i in range(self.num_net_layers):
                name = modal+'_encoder_' + str(i)
                if self.is_init:
                    cur_input = tf.layers.dense(cur_input, units=struct[i],
                                                kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                                bias_initializer=tf.constant_initializer(self.b_init[name]))
                else:
                    cur_input = tf.layers.dense(cur_input, units=struct[i], kernel_initializer=w_init())
                if i < self.num_net_layers - 1:
                    #cur_input = lrelu(cur_input)
                    cur_input = tf.nn.sigmoid(cur_input)
                    cur_input = tf.layers.dropout(cur_input, drop_prob)
                print(cur_input.get_shape())

            net_H = cur_input

            # ====================decoder=============
            struct.reverse()
            cur_input = net_H
            for i in range(self.num_net_layers - 1):
                name = modal+'_decoder_' + str(i)
                if self.is_init:
                    cur_input = tf.layers.dense(cur_input, units=struct[i + 1],
                                                kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                                bias_initializer=tf.constant_initializer(self.b_init[name]))
                else:
                    cur_input = tf.layers.dense(cur_input, units=struct[i + 1], kernel_initializer=w_init())
                #cur_input = lrelu(cur_input)
                cur_input = tf.nn.sigmoid(cur_input)
                cur_input = tf.layers.dropout(cur_input, drop_prob)
                print(cur_input.get_shape())

            name = modal+'_decoder_' + str(self.num_net_layers - 1)
            if self.is_init:
                cur_input = tf.layers.dense(cur_input, units=self.net_input_dim,
                                            kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                            bias_initializer=tf.constant_initializer(self.b_init[name]))
            else:
                cur_input = tf.layers.dense(cur_input, units=self.net_input_dim, kernel_initializer=w_init())
            cur_input = tf.nn.sigmoid(cur_input)
            x_recon = cur_input
            print(cur_input.get_shape())

            self.net_shape.reverse()

        return net_H, x_recon
