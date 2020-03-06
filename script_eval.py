from dataset import Dataset
from model import make_model_from_dataset, make_model_param
from utils import safe_mkdir
from eval_utils import get_best_eval_metrics, get_outputs

import os

import tensorflow as tf
import pretty_midi as pm
import numpy as np
from datetime import datetime
import sys
import argparse
import cPickle as pickle


parser = argparse.ArgumentParser()

parser.add_argument('data_path',type=str,help="folder containing the split dataset")
parser.add_argument('load_path',type=str,help='folder where the models will be saved (inside folder ckpt/)')
parser.add_argument("step", type=str, choices=["note_long","note_short", "time_long","time_short", "event"], help='timestep type used')
parser.add_argument('--save_path','-s',type=str,help='location to save the results')

parser.add_argument('--transpose_C',action='store_true',help='use if the model was trained with --augment C')
parser.add_argument("--diagRNN", help="Use diagonal RNN units", action="store_true")

parser.add_argument("--max_len",type=int,help="test on the first max_len seconds of each text file. Default is 60s",
                    default=60)
parser.add_argument("--n_hidden",'-n',type=int,help='number of hidden nodes (default 256)',default=256)

args = parser.parse_args()



note_range = [21,109]
note_min = note_range[0]
note_max = note_range[1]
max_len = args.max_len
timestep_type = args.step

n_hidden = args.n_hidden

model_param = make_model_param()
model_param['n_hidden']=n_hidden


if args.diagRNN:
    model_param['cell_type']='diagLSTM'
else:
    model_param['cell_type']='LSTM'


print "Computation start : "+str(datetime.now())


files_to_exclude = []


data = Dataset()
data.load_data(args.data_path,note_range=note_range,
    timestep_type=timestep_type,max_len=max_len,exclude=files_to_exclude)

save_path = args.save_path


ckpt_path = os.path.join("ckpt",save_path)
safe_mkdir(ckpt_path)



model = make_model_from_dataset(data,model_param)
result,res_dict = get_best_eval_metrics(data,model,save_path,verbose=True, with_dict=True)

if args.save_path is not None:
    safe_mkdir(args.save_path)
    pickle.dump(result, open(os.path.join(args.save_path,'results.p'), "wb"))
    pickle.dump(res_dict, open(os.path.join(args.save_path,'res_dict.p'), "wb"))
