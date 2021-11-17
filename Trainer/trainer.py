import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from Model.SingleAE import SingleAE
import pickle


class Trainer(object):

    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.data_type = config['data_type']
        self.dataset = config['dataset']
        self.num_nets = config['num_nets']
        self.net_input_dim = config['net_input_dim']
        self.net_shape = config['net_shape']
        self.drop_prob = config['drop_prob']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.alpha = config['alpha']
        self.learning_rate = config['learning_rate']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.model_path = config['model_path']


        self.x = []
        self.w = []
        for i in range(0, self.num_nets):
            temp_x = tf.placeholder(tf.float32, [None, self.net_input_dim])
            temp_w = tf.placeholder(tf.float32, [None, None])
            self.x.append(temp_x)
            self.w.append(temp_w)   

        self.optimizer, self.loss, self.recon_loss, self.first_order_loss, self.cross_modal_loss = self._build_training_graph()
        self.nets_H, self.H = self._build_eval_graph()

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _build_training_graph(self):
        nets_H = []
        nets_recon = []
        for i in range(0, self.num_nets):
            net_H, net_recon = self.model.forward_net(self.x[i], drop_prob=self.drop_prob, modal=self.data_type+'_net_'+str(i), reuse=False)
            nets_H.append(net_H)
            nets_recon.append(net_recon)

        #================high-order proximity=====================
        recon_loss = 0.0
        for i in range(0, self.num_nets):
            recon_loss_temp = tf.reduce_mean(tf.reduce_sum(tf.square(self.x[i] - nets_recon[i]), 1))
            recon_loss += recon_loss_temp
        
        #===============cross modality proximity==================
        cross_modal_loss = 0.0
        for i in range(0, self.num_nets-1):
            for j in range(i+1, self.num_nets):
                cross_modal_loss_temp = tf.reduce_mean(tf.reduce_sum(tf.square(nets_H[i] - nets_H[j]), 1), keep_dims=False)
                cross_modal_loss += cross_modal_loss_temp    
        
        #=============== first-order proximity================
        first_order_loss = 0.0
        for i in range(0, self.num_nets):
            D = tf.diag(tf.reduce_sum(self.w[i],1))
            L = D - self.w[i] ## L is laplation-matriX
            first_order_loss_temp = 2*tf.trace(tf.matmul(tf.matmul(tf.transpose(nets_H[i]),L),nets_H[i]))
            first_order_loss += first_order_loss_temp

        #==========================================================
        loss = recon_loss * self.beta + first_order_loss * self.gamma + cross_modal_loss * self.alpha
        
        vars_net = []
        for i in range(0, self.num_nets):
            vars_net_temp = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.data_type+'_net_'+str(i)+'_encoder')
            vars_net.append(vars_net_temp)
        print('vars_net:')
        print(vars_net)


        opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=vars_net)

        return opt, loss, recon_loss, first_order_loss, cross_modal_loss

    def _build_eval_graph(self):
        H = None
        nets_H = []
        for i in range(0, self.num_nets):
            net_H, _ = self.model.forward_net(self.x[i], drop_prob=0.0, modal=self.data_type+'_net_'+str(i), reuse=True)
            nets_H.append(net_H)
            if H is None:
                H = net_H
            else:
                H = tf.concat([tf.nn.l2_normalize(H, dim=1), tf.nn.l2_normalize(net_H, dim=1)], axis=1)

        return nets_H, H



    def train(self, graphs):

        for epoch in range(self.num_epochs):

            order = np.arange(graphs[0].num_nodes)
            np.random.shuffle(order)

            index = 0
            cost = 0.0
            cost_recon = 0.0
            cost_first = 0.0
            cost_cross = 0.0
            cnt = 0
            while True:
                if index > graphs[0].num_nodes:
                    break
                if index + self.batch_size < graphs[0].num_nodes:
                    mini_batch = []
                    for i in range(0, self.num_nets):
                        mini_batch_temp = graphs[i].sample_by_idx(order[index:index + self.batch_size])
                        mini_batch.append(mini_batch_temp)
                else:
                    mini_batch = []
                    for i in range(0, self.num_nets):
                        mini_batch_temp = graphs[i].sample_by_idx(order[index:])
                        mini_batch.append(mini_batch_temp)
                index += self.batch_size

                data_dict = {}
                for i in range(0, self.num_nets):
                    data_dict[self.x[i]] = mini_batch[i].X
                    data_dict[self.w[i]] = mini_batch[i].W
                loss, recon_loss, first_order_loss, cross_modal_loss, _ = self.sess.run([self.loss, self.recon_loss, self.first_order_loss, self.cross_modal_loss, self.optimizer], feed_dict=data_dict)

                cost += loss
                cost_recon += recon_loss
                cost_first += first_order_loss
                cost_cross += cross_modal_loss
                cnt += 1

                if graphs[0].is_epoch_end:
                    break
            cost /= cnt
            cost_recon /= cnt
            cost_first /= cnt
            cost_cross /= cnt

            if epoch % 50 == 0:

                emb = None
                while True:
                    mini_batch = []
                    for i in range(0, self.num_nets):
                        mini_batch_temp = graphs[i].sample(self.batch_size, do_shuffle=False)
                        mini_batch.append(mini_batch_temp)

                    data_dict = {}
                    for i in range(0, self.num_nets):
                        data_dict[self.x[i]] = mini_batch[i].X
                        data_dict[self.w[i]] = mini_batch[i].W
                    temp_emb = self.sess.run(self.H, feed_dict=data_dict)
                    if emb is None:
                        emb = temp_emb
                    else:
                        emb = np.vstack((emb, temp_emb))

                    if graphs[0].is_epoch_end:
                        break
              
                print('Epoch-{}, loss: {:.4f}, recon_loss: {:.4f}, first_order_loss: {:.4f}, cross_modal_loss: {:.4f}'.format(epoch, cost, cost_recon, cost_first, cost_cross))

        self.save_model()
        np.savetxt("./Result/"+self.dataset+"/"+self.data_type+"_emb_"+str(self.net_shape[-1]*self.num_nets)+".txt", emb, fmt="%f", delimiter="\t")
        print(self.data_type+' node embedding file saved')
        


    def save_model(self):
        self.saver.save(self.sess, self.model_path)

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
