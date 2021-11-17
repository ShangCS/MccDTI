import numpy as np
import linecache
from utils import *

class Dataset(object):

    def __init__(self, config):
        self.graph_file = config['graph_file']
        self.node_file = config['node_file']
        self.weight_file = config['weight_file']
        self.net_index = config['net_index']

        self.W, self.X, self.num_edges = self._load_data()

        self.num_nodes = self.W.shape[0]   
        print('net: {}, nodes: {}, edes: {}. '.format(self.net_index, self.num_nodes, self.num_edges))


        self._order = np.arange(self.num_nodes)
        self._index_in_epoch = 0
        self.is_epoch_end = False

    def _load_data(self):
        lines = linecache.getlines(self.node_file)
        lines = [line.rstrip('\n') for line in lines]

        #===========load node============
        node_map = {}
        for idx, line in enumerate(lines):
            line = line.split(' ')
            node_map[line[1]] = idx
        num_nodes = len(node_map)


        #==========load graph========
        W = np.zeros((num_nodes, num_nodes))
        count = 0
        lines = linecache.getlines(self.graph_file)
        lines = [line.rstrip('\n') for line in lines]
        for line in lines:
            line = line.split(' ')
            idx1 = node_map[line[0]]
            idx2 = node_map[line[1]]
            weight = 1.0
            if len(line) == 3:
                weight = float(line[2])
            W[idx2, idx1] = float(weight)
            W[idx1, idx2] = float(weight)
            count += 1

        #=========load weight=========
        X = np.zeros((num_nodes, num_nodes))
        lines = linecache.getlines(self.weight_file)
        lines = [line.rstrip('\n') for line in lines]
        for line in lines:
            line = line.split(' ')
            idx1 = node_map[line[0]]
            idx2 = node_map[line[1]]
            weight = 1.0
            if len(line) == 3:
                weight = float(line[2])
            X[idx2, idx1] = float(weight)
            X[idx1, idx2] = float(weight)

        return W, X, count


    def sample(self, batch_size, do_shuffle=True):
        if self.is_epoch_end:
            if do_shuffle:
                np.random.shuffle(self._order)
            else:
                self._order = np.sort(self._order)
            self.is_epoch_end = False
            self._index_in_epoch = 0

        mini_batch = Dotdict()
        end_index = min(self.num_nodes, self._index_in_epoch + batch_size)
        cur_index = self._order[self._index_in_epoch:end_index]
        mini_batch.X = self.X[cur_index]
        mini_batch.W = self.W[cur_index][:, cur_index]

        if end_index == self.num_nodes:
            end_index = 0
            self.is_epoch_end = True
        self._index_in_epoch = end_index

        return mini_batch

    def sample_by_idx(self, idx):
        mini_batch = Dotdict()
        mini_batch.X = self.X[idx]
        mini_batch.W = self.W[idx][:, idx]

        return mini_batch

