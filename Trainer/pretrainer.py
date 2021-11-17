from Utils.utils import *
from Model.SingleAE import SingleAE
import pickle


class PreTrainer(object):

    def __init__(self, config):
        self.config = config
        self.net_input_dim = config['net_input_dim']
        self.net_shape = config['net_shape']
        self.pretrain_params_path = config['pretrain_params_path']

        self.W_init = {}
        self.b_init = {}


    def pretrain(self, data, modal):

        shape = [self.net_input_dim] + self.net_shape

        for i in range(len(shape) - 1):
            print(shape[i], shape[i+1])


            activation_fun1 = lrelu
            activation_fun2 = lrelu
            #activation_fun1 = tf.nn.sigmoid
            #activation_fun2 = tf.nn.sigmoid
            if i == 0:
                activation_fun2 = tf.nn.sigmoid
            if i == len(shape) - 2:
                activation_fun1 = None


            SAE = SingleAE([shape[i], shape[i + 1]],
                           {"iters": 50000, "batch_size": 256, "lr": 1e-3, "dropout": 0.8}, data,
                           i, modal, activation_fun1, activation_fun2)
            SAE.doTrain()
            W1, b1, W2, b2 = SAE.getWb()

            name = modal + "_encoder_" + str(i)
            self.W_init[name] = W1
            self.b_init[name] = b1
            name = modal + "_decoder_" + str(len(shape) - i - 2)
            self.W_init[name] = W2
            self.b_init[name] = b2

            data = SAE.getH()

        with open(self.pretrain_params_path, 'wb') as handle:
            pickle.dump([self.W_init, self.b_init], handle, protocol=pickle.HIGHEST_PROTOCOL)


