import os
import numpy as np
import pretty_midi as pm
from random import shuffle, sample
import cPickle as pickle
from datetime import datetime
import copy
from pianoroll import Pianoroll, PianorollMIREX
from utils import check_corrupt
from tqdm import tqdm

class Dataset(object):
    """Classe representing the dataset."""

    def __init__(self):
        self.train = []
        self.test = []
        self.valid = []

        self.note_range = [0,128]
        self.max_len = 0

    def walkdir(self,folder):
        for fn in os.listdir(folder):
            if fn.endswith('.mid') and not fn.startswith('.'):
                yield fn


    def load_data_one(self,folder,subset,timestep_type,max_len=None,note_range=[0,128],length_of_chunks=None,key_method='main',exclude=[],with_onsets=False):
        dataset = []
        subfolder = os.path.join(folder,subset)

        #Set up progress bar
        filecounter = 0
        for filepath in self.walkdir(subfolder):
            filecounter += 1
        print "Now loading: "+subset.upper()
        pbar = tqdm(self.walkdir(subfolder), total=filecounter, unit="files")
        for fn in pbar:
            pbar.set_postfix(file=fn[:10], refresh=False)

            filename = os.path.join(subfolder,fn)
            if os.path.split(filename)[1] not in exclude:
                # print filename
                midi_data = pm.PrettyMIDI(filename)
                if length_of_chunks == None:
                    piano_roll = Pianoroll()
                    if max_len == None:
                        piano_roll.make_from_pm(midi_data,timestep_type,None,note_range,key_method,with_onsets)
                    else:
                        piano_roll.make_from_pm(midi_data,timestep_type,[0,max_len],note_range,key_method,with_onsets)
                    piano_roll.name = os.path.splitext(os.path.basename(filename))[0]
                    dataset += [piano_roll]
                else :
                    if max_len == None:
                        end_file = midi_data.get_piano_roll().shape[1]/100.0
                    else :
                        end_file = max_len
                    begin = 0
                    end = 0
                    i = 0
                    pr_list = []
                    while end < end_file:
                        end = min(end_file,end+length_of_chunks)
                        piano_roll = Pianoroll()
                        piano_roll.make_from_pm(midi_data,timestep_type,[begin,end],note_range,key_method,with_onsets)
                        piano_roll.name = os.path.splitext(os.path.basename(filename))[0]+"_"+str(i)
                        pr_list += [piano_roll]
                        begin = end
                        i += 1
                    dataset += pr_list

        if subset in ["train","valid","test"]:
            setattr(self,subset,dataset)
        return dataset

    def load_data(self,folder,timestep_type,max_len=None,note_range=[0,128],length_of_chunks=None,key_method='main',exclude=[],with_onsets=False):
        self.note_range = note_range
        for subset in ["train","valid","test"]:
            self.load_data_one(folder,subset,timestep_type,max_len,note_range,length_of_chunks,key_method,exclude,with_onsets)
        self.zero_pad()
        print "Dataset loaded ! "+str(datetime.now())

    def load_data_custom(self,folder,train,valid,test,timestep_type,max_len=None,note_range=[0,128],length_of_chunks=None,with_onsets=False):
        self.note_range = note_range

        for subset in train:
            self.train += self.load_data_one(folder,subset,timestep_type,max_len,note_range,length_of_chunks,with_onsets=with_onsets)
        for subset in valid:
            self.valid += self.load_data_one(folder,subset,timestep_type,max_len,note_range,length_of_chunks,with_onsets=with_onsets)
        for subset in test:
            self.test += self.load_data_one(folder,subset,timestep_type,max_len,note_range,length_of_chunks,with_onsets=with_onsets)

        self.zero_pad()
        print "Dataset loaded ! "+str(datetime.now())

    def get_n_files(self,subset):
        return len(getattr(self,subset))
    def get_n_notes(self):
        return self.note_range[1]-self.note_range[0]
    def get_len_files(self):
        return self.max_len

    def get_dataset(self,subset,with_names=False,with_key_masks=False,with_weights=False):
        #Outputs an array containing all the piano-rolls (3D-tensor)
        #and the list of the actual lengths of the piano-rolls
        pr_list = getattr(self,subset)
        n_files = len(pr_list)
        len_file = pr_list[0].roll.shape[1]
        n_notes = self.get_n_notes()

        dataset = np.zeros([n_files,n_notes,len_file])
        lengths = np.zeros([n_files],dtype=int)
        if with_names:
            names = []
        if with_key_masks:
            key_masks = np.zeros([n_files,n_notes,len_file-1])
            key_lists = []
        if with_weights:
            w_trss = np.zeros([n_files,len_file-1])
            w_k = np.zeros([n_files,len_file-1])


        for i, piano_roll in enumerate(pr_list):
            roll = piano_roll.roll
            dataset[i] = roll
            lengths[i] = piano_roll.length
            if with_names:
                names += [piano_roll.name]
            if with_key_masks:
                key_masks[i] = piano_roll.get_key_profile_matrix()
                key_list = [time for (key,time) in piano_roll.key_list]+[piano_roll.length]
                key_lists += [key_list]
            if with_weights:
                w_trss[i,piano_roll.length-1] = piano_roll.get_weights_tr_ss()
                w_k[i,piano_roll.length-1] = piano_roll.get_weights_key()


        output = [dataset, lengths]
        if with_names:
            output += [names]
        if with_key_masks:
            output += [key_masks]
            #Zero-pad the key_lists
            max_len = max(map(len,key_lists))
            key_lists_array = np.zeros([n_files,max_len])
            for i,key_list in enumerate(key_lists):
                key_lists_array[i,:len(key_list)]=key_list
                key_lists_array[i,len(key_list):]=key_list[-1]
            output += [key_lists_array]
        if with_weights:
            output += [w_trss]
            output += [w_k]
        return output

    def get_dataset_chunks(self,subset,len_chunk):
        #Outputs an array containing all the pieces cut in chunks (4D-tensor)
        #and a list of lists for the lengths
        pr_list = getattr(self,subset)
        n_files = len(pr_list)
        len_file = pr_list[0].roll.shape[1]
        n_notes = self.get_n_notes()
        n_chunks = int(np.ceil(float(len_file)/len_chunk))

        dataset = np.zeros([n_files,n_chunks,n_notes,len_chunk])
        lengths = np.zeros([n_files,n_chunks])
        i = 0
        while i<n_files:
            piano_roll = pr_list[i]
            chunks, chunks_len = piano_roll.cut(len_chunk)
            dataset[i] = chunks
            lengths[i] = chunks_len
            i += 1

        return dataset, lengths

    def get_dataset_chunks_no_pad(self,subset,len_chunk):
        #Outputs an array containing all the pieces cut in chunks (3D-tensor)
        #and a list for the lengths
        pr_list = getattr(self,subset)
        n_files = len(pr_list)
        len_file = pr_list[0].roll.shape[1]
        n_notes = self.get_n_notes()

        dataset = []
        lengths = []

        i = 0
        while i<n_files:
            piano_roll = pr_list[i]
            chunks, chunks_len = piano_roll.cut(len_chunk,keep_padding=False)
            dataset += list(chunks)
            lengths += list(chunks_len)
            i += 1

        return np.asarray(dataset), np.asarray(lengths)


    def shuffle_one(self,subset):
        data = getattr(self,subset)
        shuffle(data)


    def __max_len(self,dataset):
        if dataset == []:
            return 0
        else :
            return max(map(lambda x: x.length, dataset))

    def zero_pad(self,max_len=None):
        if max_len is None:
            max_train = self.__max_len(self.train)
            max_valid = self.__max_len(self.valid)
            max_test = self.__max_len(self.test)
            max_len = max([max_train,max_valid,max_test])

        self.max_len = max_len

        for subset in ["train","valid","test"]:
            self.zero_pad_one(subset,max_len)


    def zero_pad_one(self,subset,max_len):
        #Zero-padding the dataset
        dataset = getattr(self,subset)
        for piano_roll in dataset:
            piano_roll.zero_pad(max_len)


    def transpose_all_one(self,subset):
        data = getattr(self,subset)
        tr_range = [-7,5]
        new_data = []
        for piano_roll in data:
            for j in range(*tr_range):
                tr_piano_roll = piano_roll.transpose(j)
                new_data += [tr_piano_roll]
        setattr(self,subset,new_data)

    def transpose_all(self):
        print "Transposing train set in every key..."
        #You only augment the training dataset
        for subset in ["train"]: #,"valid","test"]:
            self.transpose_all_one(subset)

    def transpose_C_one(self,subset):
        data = getattr(self,subset)
        new_data = []
        for piano_roll in data:
            key = piano_roll.key
            if key <= 7:
                #Transpose down
                tr_piano_roll = piano_roll.transpose(-key)
            else:
                #Transpose up
                tr_piano_roll = piano_roll.transpose(12-key)
            new_data += [tr_piano_roll]
        setattr(self,subset,new_data)

    def transpose_C(self):
        print "Transposing all subsets in C..."
        #You have to transpose all the subsets
        for subset in ["train","valid","test"]:
            self.transpose_C_one(subset)

    def timestretch_all(self):
        print "Timestretching..."
        #You only timestretch the training dataset
        data = self.train
        new_data = []
        for piano_roll in data:
            stretch_piano_roll = piano_roll.timestretch()
            new_data += [stretch_piano_roll]
            new_data += [piano_roll]
        self.train = new_data
        self.zero_pad()


def ground_truth(data):
    return data[:,:,1:]


def safe_mkdir(dir,clean=False):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if clean and not os.listdir(dir) == [] :
        old_path = os.path.join(dir,"old")
        safe_mkdir(old_path)
        for fn in os.listdir(dir):
            full_path = os.path.join(dir,fn)
            if not os.path.isdir(full_path):
                os.rename(full_path,os.path.join(old_path,fn))
