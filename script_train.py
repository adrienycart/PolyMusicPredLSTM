from dataset import Dataset, ground_truth, safe_mkdir
from model import Model, make_model_from_dataset, make_save_path, make_model_param, make_train_param

import os

import tensorflow as tf
import pretty_midi as pm
import numpy as np
from datetime import datetime
import sys
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('data_path',type=str,help="folder containing the split dataset")
parser.add_argument('save_path',type=str,help='folder where the models will be saved (inside folder ckpt/)')
parser.add_argument("step", type=str, choices=["note_long","note_short", "time_long","time_short", "event"], help='timestep type used')
parser.add_argument('--augment','-a',type=str,choices=["C","all","none"],help='type of data augmentation done (default all)',default='all')
parser.add_argument("--diagRNN", help="Use diagonal RNN units", action="store_true")
parser.add_argument('--loss','-l',type=str,choices=["H","S"],help='type of loss to use (default H)',default='H')
parser.add_argument('--w_tr',type=float,help='value for w_tr (default 1)',default=1)
parser.add_argument('--w_ss',type=float,help='value for w_ss (default 1)',default=1)
parser.add_argument('--alpha',type=float,help='value for alpha (default 0)',default=0)

parser.add_argument("--max_len",type=int,help="test on the first max_len seconds of each text file. Default is 60s",
                    default=60)
parser.add_argument("--learning_rate",'-lr',type=float,help='learning rate (default 0.01)',default=0.01)
parser.add_argument("--n_hidden",'-n',type=int,help='number of hidden nodes (default 256)',default=256)
parser.add_argument("--max_epochs",'-e',type=int,help='max number of epochs (default 500)',default=500)
parser.add_argument("--early_stop",'-s',type=int,help='patience for early stopping, in number of epochs (default 15)',default=15)
parser.add_argument("--grad_clip",'-g',type=float,help='gradient clipping (not used if unspecified)')

args = parser.parse_args()



note_range = [21,109]
note_min = note_range[0]
note_max = note_range[1]
max_len = args.max_len
timestep_type = args.step

n_hidden = args.n_hidden
learning_rate = args.learning_rate


train_param = make_train_param()
train_param['epochs']=args.max_epochs
train_param['batch_size']=50
train_param['display_per_epoch']=5
train_param['save_step']=1
train_param['max_to_keep']=1
train_param['summarize']=True
train_param['early_stop_epochs']=args.early_stop


model_param = make_model_param()
model_param['n_hidden']=n_hidden
model_param['learning_rate']=learning_rate
if args.loss_type == 'H':
    model_param['loss_type']='XE'
elif args.loss_type == 'S':
    model_param['loss_type']='combined'
    model_param['weights_tr_ss'] = [args.w_tr,args.w_ss]
    model_param['alpha'] = args.alpha
model_param['grad_clip'] = args.grad_clip

if args.diagRNN:
    model_param['cell_type']='diagLSTM'
else:
    model_param['cell_type']='LSTM'


print "Computation start : "+str(datetime.now())


files_to_exclude = []


if timestep_type == 'note_long' and args.loss_type == 'S':
    files_to_exclude += ['burg_quelle.mid',
    'bach_847.mid',
    'chpn-p8.mid',
    'bach_846.mid',
    'debussy_cc_4.mid',
    'chpn_op27_2.mid',
    'mendel_op19_1.mid']

data = Dataset()
data.load_data(args.data_path,note_range=note_range,
    timestep_type=timestep_type,max_len=max_len,exclude=files_to_exclude)
if args.augment == 'all':
    data.transpose_all()
elif args.augment == 'C':
    data.transpose_C()
elif args.augment == 'none':
    pass

save_path = args.save_path


ckpt_path = os.path.join("ckpt",save_path)
safe_mkdir(ckpt_path)


print "________________________________________"
print "Hidden nodes = "+str(n_hidden)+", Learning rate = "+str(learning_rate)
print "________________________________________"
print "."

model = make_model_from_dataset(data,model_param)
model.train(data,save_path=save_path,train_param=train_param)
tf.reset_default_graph()
