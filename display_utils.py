from dataset import Dataset, ground_truth, safe_mkdir
from model import Model, make_model_from_dataset
from pianoroll import Pianoroll, get_quant_piano_roll
from utils import get_chord_counter, get_chord_counter_by_position,my_get_end_time

import os

import tensorflow as tf
import pretty_midi as pm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def print_measures(F,pre,rec):
    print "F : "+str(F)+", pre : "+str(pre)+", rec : "+str(rec)


def load_piano_roll(midi_file,crop,fs=100):
    midi = pm.PrettyMIDI(midi_file)
    piano_roll = midi.get_piano_roll(fs)
    #Binarize and crop the piano_roll
    piano_roll = np.not_equal(piano_roll,0).astype(int)
    if crop:
        piano_roll = piano_roll[crop[0]:crop[1]+1,:]
    return piano_roll

def load_quant_piano_roll(midi_file,crop,fs=4,max_len=None):
    midi = pm.PrettyMIDI(midi_file)
    piano_roll = get_quant_piano_roll(midi,fs,max_len)
    #Binarize and crop the piano_roll
    piano_roll = np.not_equal(piano_roll,0).astype(int)
    if crop:
        piano_roll = piano_roll[crop[0]:crop[1]+1,:]
    return piano_roll



def compare_piano_rolls(piano_roll1,piano_roll2,crop=None,title="",time_grid=False,show=False):
    if crop:
        labels = list(range(crop[0],crop[1]))
    else :
        labels = list(range(0,128))
    labels = [pm.note_number_to_name(x) for x in labels]



    fig, [ax1,ax2] = plt.subplots(2,1)
    ax1.imshow(piano_roll1,aspect='auto',origin='lower')
    ax1.set_yticks([x+0.5 for x in list(range(len(labels)))])
    ax1.set_yticklabels(labels,fontsize=5)
    if time_grid:
        ax1.set_xticks([x+0.5 for x in list(range(piano_roll1.shape[1]))])
        ax1.grid(True,axis='both',color='black')
    else:
        ax1.grid(True,axis='y',color='black')
    plt.title(title)

    ax2.imshow(piano_roll2,aspect='auto',origin='lower')
    ax2.set_yticks([x+0.5 for x in list(range(len(labels)))])
    ax2.set_yticklabels(labels,fontsize=5)
    if time_grid:
        ax2.set_xticks([x+0.5 for x in list(range(piano_roll2.shape[1]))])
        ax2.grid(True,axis='both',color='black')
    else:
        ax2.grid(True,axis='y',color='black')
    if show:
        plt.show()



def display_prediction(piano_roll,model,save_path,n_model=None,sigmoid=False,save=False,full=False,time_grid=False):

    roll, length = piano_roll.get_roll()
    roll = np.asarray([roll])
    length = [length]
    note_range = piano_roll.note_range

    # print piano_roll.shape
    # print np.transpose(piano_roll,[0,2,1]).shape
    pred = model.run_prediction(roll,length,save_path,n_model,sigmoid)
    pred = pred[0]
    target = ground_truth(roll)
    target = target[0]

    midi_name = piano_roll.name
    title = os.path.dirname(save_path)+", "+midi_name

    # if crop:
    #     labels = list(range(crop[0],crop[1]+1))
    # else :
    #     labels = list(range(0,128))
    # labels = [pm.note_number_to_name(x) for x in labels]

    # plt.figure()
    #
    # plt.subplot(211)
    # plt.imshow(target,aspect='auto',origin='lower')
    # plt.yticks([x+0.5 for x in list(range(len(labels)))] , labels,fontsize=5)
    # ax = plt.gca()
    # ax.grid(True,axis='y',color='black')
    # plt.title(os.path.basename(save_path)+", "+midi_name)
    #
    # plt.subplot(212)
    # plt.imshow(pred,aspect='auto',origin='lower')
    # plt.yticks([x+0.5 for x in list(range(len(labels)))] , labels,fontsize=5)
    # ax = plt.gca()
    # ax.grid(True,axis='y',color='black')

    compare_piano_rolls(target,pred,note_range,title,time_grid)

    if save :
        if sigmoid:
            fig_save_path = os.path.join("./fig/",save_path,midi_name+'_sigmoid.png')
        else:
            fig_save_path = os.path.join("./fig/",save_path,midi_name+'.png')
        safe_mkdir(fig_save_path)
        plt.savefig(fig_save_path)
    else :
            #TODO : find a way to fisplay full screen on MacOSX
        plt.show()

def make_hist(count):
    labels, values = zip(*count)
    indexes = np.arange(len(labels))
    width = 1
    plt.bar(indexes, values, width)
    plt.xticks(indexes , labels,rotation=90,fontsize=5)

def hist_chords(folder):
    train_path = os.path.join(folder,"train/")
    valid_path = os.path.join(folder,"valid/")
    test_path = os.path.join(folder,"test/")

    count_train = get_chord_counter(train_path)
    count_valid = get_chord_counter(valid_path)
    count_test = get_chord_counter(test_path)


    plt.figure()
    plt.subplot(131)
    make_hist(count_train)
    plt.title("Train")

    plt.subplot(132)
    make_hist(count_valid)
    plt.title("Valid")

    plt.subplot(133)
    make_hist(count_test)
    plt.title("Test")
    plt.show()

def hist_chords_by_position(subfolder):

    counters = get_chord_counter_by_position(subfolder)


    plt.figure()
    plt.subplot(131)
    make_hist(counters[0])
    plt.title("1st position")

    plt.subplot(132)
    make_hist(counters[1])
    plt.title("2nd position")

    plt.subplot(133)
    make_hist(counters[2])
    plt.title("3rd position")
    plt.show()

def hist_length(folder):
    midi_list = []
    for subfolder in ['train','valid','test']:
        subfolder_path = os.path.join(folder,subfolder)
        midi_list += [[os.path.join(subfolder_path,x) for x in os.listdir(subfolder_path) if x.endswith('.mid')]]

    len_list = []
    for sub_list in midi_list:
        sub_len_list = []
        for midi_name in sub_list:
            midi = pm.PrettyMIDI(midi_name)
            piano_roll = midi.get_piano_roll()
            length = piano_roll.shape[1]/100
            sub_len_list += [length]
        len_list += sub_len_list
        print sum(sub_len_list)

    plt.hist(len_list, bins=25, normed=True,cumulative=True)
    plt.show()

def print_cumul_length(folder):
    folder_list = [os.path.join(folder,x) for x in os.listdir(folder) if os.path.isdir(os.path.join(folder,x)) ]
    for subfolder in folder_list :
        midi_list = [os.path.join(subfolder,x) for x in os.listdir(subfolder) if x.endswith('.mid')]
        length = 0
        for midi_file in midi_list:
            midi = pm.PrettyMIDI(midi_file)
            length += my_get_end_time(midi)
        print os.path.split(subfolder)[1]
        print length/60.0

def hist_notes(folder):
    midi_list = []
    for subfolder in ['train','valid','test']:
        subfolder_path = os.path.join(folder,subfolder)
        midi_list += [os.path.join(subfolder_path,x) for x in os.listdir(subfolder_path) if x.endswith('.mid')]

    count = np.zeros([88])
    for midi_name in midi_list:
        piano_roll = Pianoroll()
        piano_roll.make_from_file(midi_name,4,max_len=60,note_range=[21,109],quant=True)
        count += np.sum(piano_roll.roll,1)

    count = np.cumsum(count)
    count = count/float(count[-1])
    x = range(21,109)
    plt.bar(x,count)
    plt.show()



#print_cumul_length('data/Piano-midi-sorted')
#hist_notes('data/Piano-midi.de')
