from dataset import Dataset, ground_truth, safe_mkdir
from model import Model, make_model_from_dataset, make_save_path, make_model_param, make_train_param

import os

import tensorflow as tf
import pretty_midi as pm
import numpy as np
from datetime import datetime
import sys


# TO INSPECT: ['burg_agitato']

note_range = [21,109]
note_min = note_range[0]
note_max = note_range[1]
max_len = 60 #3 seconds files only
timestep_type = 'note_long'

n_hiddens = [ 256] #number of features in hidden layer
learning_rates = [0.001, 0.01]


train_param = make_train_param()
train_param['epochs']=100
train_param['batch_size']=5
train_param['display_per_epoch']=1000000
train_param['save_step']=1
train_param['max_to_keep']=1
train_param['summarize']=False

print "Computation start : "+str(datetime.now())

# files_to_exclude = ['data/Piano-midi.de/train/burg_quelle.mid',
# 'data/Piano-midi.de/train/bach_847.mid',
# 'data/Piano-midi.de/train/chpn-p8.mid',
# 'data/Piano-midi.de/train/bach_846.mid',
# 'data/Piano-midi.de/train/debussy_cc_4.mid',
# 'data/Piano-midi.de/train/chpn_op27_2.mid',
# 'data/Piano-midi.de/train/schumm-6.mid',
# 'data/Piano-midi.de/valid/mendel_op19_1.mid']
files_to_exclude = []

data = Dataset()
data.load_data('data/test_dataset/',note_range=note_range,
    timestep_type=timestep_type,max_len=max_len,exclude=files_to_exclude,with_onsets=True)
# data.transpose_all()


base_path = 'test'

for n_hidden in n_hiddens:
    for learning_rate in learning_rates:

        model_param = make_model_param()
        model_param['n_hidden']=n_hidden
        model_param['learning_rate']=learning_rate
        # model_param['loss_type']='combined_cw'
        model_param['cell_type']='LSTM'
        # model_param['non_binary']=3

        save_path = make_save_path(base_path,model_param)
        log_path = os.path.join("ckpt",save_path)
        safe_mkdir(log_path)
        # f= open(os.path.join(log_path,"log.txt"), 'w')
        # sys.stdout = f

        print "________________________________________"
        print "Hidden nodes = "+str(n_hidden)+", Learning rate = "+str(learning_rate)
        print "________________________________________"
        print "."

        model = make_model_from_dataset(data,model_param)
        model.train(data,save_path=save_path,train_param=train_param)
        tf.reset_default_graph()
        print "."
        print "."
        print "."

#print "Computation end : "+str(datetime.now())
