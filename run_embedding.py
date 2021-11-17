import numpy as np
import linecache
from Utils.dataset import Dataset
from Model.model import Model
from Trainer.trainer import Trainer
from Trainer.pretrainer import PreTrainer
from Utils import gpu_info
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random


if __name__=='__main__':

    gpus_to_use, free_memory = gpu_info.get_free_gpu()
    print('GPU info: ', gpus_to_use, free_memory)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use

    random.seed(9001)

    data_type = ['drug', 'protein']

    #DTINet_data
    dataset = 'DTINet_data'
    drug_string_nets = ['Drugs','drug_se','drug_drug','drug_disease']
    protein_string_nets = ['Proteins','protein_disease','protein_protein']
    drug_net_shape = [500, 150]
    protein_net_shape = [800, 300]

    #deepDTnet_data
    #dataset = 'deepDTnet_data'
    #drug_string_nets = ['drugdrug','Sim_drugDisease','Sim_drugsideEffect','drugsim1network','drugsim2network','drugsim3network','drugsim4network','drugsim5network','drugsim6network']
    #protein_string_nets = ['proteinprotein','Sim_proteinDisease','proteinsim1network','proteinsim2network','proteinsim3network','proteinsim4network']
    #drug_net_shape = [500, 100]
    #protein_net_shape = [800, 150]

    for t in data_type:
        if t == 'drug':
            string_nets = drug_string_nets
            net_shape = drug_net_shape
        else:
            string_nets = protein_string_nets
            net_shape = protein_net_shape
        net_shape_len = len(net_shape)
        num_nets = len(string_nets)
        graphs = []
        for i in range(0, num_nets):
            dataset_config ={'graph_file': './Database/'+dataset+'/'+t+'/'+string_nets[i]+'.txt',
                             'weight_file': './Database/'+dataset+'/'+t+'/'+string_nets[i]+'_G.txt',
                             'node_file': './Database/'+dataset+'/'+t+'/'+t+'_node_list.txt',
                             'net_index': t+'_net_'+str(i)}
            graph = Dataset(dataset_config)
            graphs.append(graph)
        num_nodes = graphs[0].num_nodes

        pretrain_config = {
            'net_shape': net_shape,
            'net_input_dim': num_nodes,
            'pretrain_params_path': './Log/'+dataset+'/'+t+'_pretrain_params.pkl'}

        model_config = {
            'net_shape': net_shape,
            'net_input_dim': num_nodes,
            'is_init': True,
            'pretrain_params_path': './Log/'+dataset+'/'+t+'_pretrain_params.pkl'}

        trainer_config = {
            'data_type': t,
            'dataset': dataset,
            'net_shape': net_shape,
            'num_nets': num_nets,
            'net_input_dim': num_nodes,
            'drop_prob': 0.2,
            'learning_rate': 1e-5,
            'batch_size': 100,
            'num_epochs': 500,
            'beta': 100,
            'alpha': 1,
            'gamma': 0.1,
            'model_path': './Log/'+dataset+'/'+t+'_model.pkl',}

        #'''
        pretrainer = PreTrainer(pretrain_config)
        for i in range(0, num_nets):
            pretrainer.pretrain(graphs[i].X, t+'_net_'+str(i))
        #'''
        
        model = Model(model_config)
        trainer = Trainer(model, trainer_config)
        trainer.train(graphs)

