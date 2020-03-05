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



#
# folder = "data/Piano-midi.de/"
# for subfolder in ["train","valid","test"]:
#     subfolder = os.path.join(folder,subfolder)
#     for fn in os.listdir(subfolder):
#         if fn.endswith('.mid') and not fn.startswith('.'):
#             filename = os.path.join(subfolder,fn)
#             midi_data = pm.PrettyMIDI(filename)
#             time_signatures = midi_data.time_signature_changes
#             if not (time_signatures[0].denominator == 4 or (time_signatures[0].numerator == 4 and time_signatures[0].numerator == 2)):
#                 print filename
#                 print midi_data.time_signature_changes
#                 print midi_data.get_end_time()


# data=Dataset()
# data.load_data("data/test_dataset","note_long",60,[21,109])
# data2=Dataset()
# data2.load_data("data/test_dataset","event",60,[21,109])


# data = Dataset()
# # data.load_data('data/dummy_midi_data_test/',
# #         fs=4,max_len=None,note_range=[60,65],quant=True,length_of_chunks=1.25)
# data.load_data_custom('data/Piano-midi-sorted',train=['albeniz','borodin'],valid=['clementi'],test=['grieg'],
#         fs=4,max_len=15,note_range=[21,109],quant=True,length_of_chunks=None)
# print data.train
# for pr in data.train:
#     print pr.name
#     print pr.length
# for pr in data2.train:
#     print pr.name
#     print pr.length



#split_files("data/dummy_midi_data_poly_upwards/")
#unsplit_files("dummy_midi_data_poly/")

# liste = get_chord_counter('dummy_midi_data_poly/train')
# nums = [ x[1] for x in liste]
# print max(nums)
# print min(nums)
# print max(nums)-min(nums)


# for fn in os.listdir("data/Piano-midi.de"):
#     path = os.path.join("data/Piano-midi.de",fn)
#     if os.path.isdir(path):
#         split_files(path)
#
# safe_mkdir("data/Piano-midi.de/train")
# safe_mkdir("data/Piano-midi.de/valid")
# safe_mkdir("data/Piano-midi.de/test")
#
# for fn in os.listdir("data/Piano-midi.de"):
#     folder = os.path.join("data/Piano-midi.de",fn)
#     if os.path.isdir(folder):
#         train_path = os.path.join(folder,"train/")
#         valid_path = os.path.join(folder,"valid/")
#         test_path = os.path.join(folder,"test/")
#
#         move_files(os.listdir(train_path),train_path,"data/Piano-midi.de/train")
#         move_files(os.listdir(valid_path),valid_path,"data/Piano-midi.de/valid")
#         move_files(os.listdir(test_path),test_path,"data/Piano-midi.de/test")


class DatasetMIREX(Dataset):

    def __init__(self,rand_transp=True):
        self.train = []
        self.test = []
        self.valid = []

        self.train_idx = []

        self.note_range = [0,128]
        self.max_len = None
        self.rand_transp=rand_transp



    def walkdir(self,folder):
        folder_list = os.listdir(folder)
        # folder_list = sorted(folder_list)
        for fn in folder_list:
            if fn.endswith('.csv') and not fn.startswith('.'):
                yield fn

    def load_data(self,prime_folder,cont_folder,note_range=[0,128],test=False):

        self.note_range=note_range

        dataset = []

        #Set up progress bar
        filecounter = 0
        for filepath in self.walkdir(prime_folder):
            filecounter += 1
        pbar = tqdm(self.walkdir(prime_folder), total=filecounter, unit="files")
        for filename in pbar:
            pbar.set_postfix(file=filename[:10], refresh=False)

            prime_filepath = os.path.join(prime_folder,filename)
            cont_filepath = os.path.join(cont_folder,filename)

            pr = PianorollMIREX()
            pr.make_from_file(prime_filepath,cont_filepath,note_range)
            dataset += [pr]

        if test:
            self.test=dataset

        else:
            n_data = len(dataset)
            n_valid = int(0.10*n_data)
            n_test = int(0.10*n_data)

            ptr = 0
            self.valid = dataset[ptr:ptr+n_valid]
            ptr += n_valid
            self.test = dataset[ptr:ptr+n_test]
            ptr += n_test
            self.train = dataset[ptr:]

        self.zero_pad()

        self.update_idx_list()



    def check_data(self):
        removed = {}
        for subset in ['train','valid','test']:
            to_keep = []
            removed[subset] = []
            for i, pianoroll in enumerate(getattr(self,subset)):
                pr = pianoroll.roll[:,:pianoroll.length]

                data_extended = np.pad(pr,[[0,0],[1,1]],'constant')
                diff = data_extended[:,1:] - data_extended[:,:-1]
                steady = np.where(np.sum(np.abs(diff),axis=0)==0)[0]

                try:
                    if steady[-1] == pr.shape[1]:
                        steady = steady[:-1]
                except IndexError:
                    assert steady.size == 0
                    print 'Removing',pianoroll.name
                    removed[subset] += [i]
                else:
                    try:
                        if steady[0] == 0:
                            steady = steady[1:]
                    except IndexError:
                        assert steady.size == 0
                        print 'Removing',pianoroll.name
                        removed[subset] += [i]
                    else:
                        if steady.size == 0:
                            print 'Removing',pianoroll.name
                            removed[subset] += [i]
                        else:
                            to_keep += [pianoroll]
            setattr(self,subset,to_keep)

        self.update_idx_list()
        return removed

    def remove_data(self,remove_dict):
        for subset in ['train','valid','test']:
            data_list = getattr(self,subset)

            # Remove from last to avoid indexing problems
            for i in remove_dict[subset][::-1]:
                del data_list[i]

        self.update_idx_list()


    def update_idx_list(self):
        self.train_idx = list(range(len(self.train)))
        self.valid_idx = list(range(len(self.valid)))
        self.test_idx = list(range(len(self.test)))

    def shuffle_one(self,subset):
        shuffle(getattr(self,subset+'_idx'))

    def transpose_all_one(self,subset,tr_range=[-4,4]):
        data = getattr(self,subset)
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
        self.update_idx_list()

    def get_dataset(self,subset,with_names=False,with_key_masks=False,with_weights=False):
        #Outputs an array containing all the piano-rolls (3D-tensor)
        #and the list of the actual lengths of the piano-rolls
        pr_list = getattr(self,subset)
        idx = getattr(self,subset+'_idx')
        n_files = len(pr_list)
        len_file = self.max_len
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


        for i,j in enumerate(idx):
            piano_roll = pr_list[j]
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

    def get_dataset_generator(self,subset,batch_size):
        seq_buff = []
        len_buff = []
        pr_list = getattr(self,subset)
        files_left = list(range(len(pr_list)))

        n_notes = self.note_range[1]-self.note_range[0]
        if self.max_len is None:
            self.set_max_len()

        while files_left != [] or len(seq_buff)>=batch_size:
            if len(seq_buff)<batch_size:
                file_index = files_left.pop()
                piano_roll = pr_list[file_index]
                if self.rand_transp:
                    transp = np.random.randint(-7,6)
                    piano_roll = piano_roll.transpose(transp)
                roll= piano_roll.roll
                length = piano_roll.length
                seq_buff.append(roll)
                len_buff.append(length)

            else:

                output_roll = np.zeros([batch_size,n_notes,self.max_len])
                for i,seq in enumerate(seq_buff[:batch_size]):
                    output_roll[i,:,:seq.shape[1]]=seq
                output = (output_roll[:,:,:],output_roll[:,:,1:],np.array(len_buff[:batch_size]))

                del seq_buff[:batch_size]
                del len_buff[:batch_size]
                yield output


class DatasetMIREXClassif(Dataset):

    def __init__(self,rand_transp=True):
        self.train = []
        self.test = []
        self.valid = []

        self.train_idx = []

        self.note_range = [0,128]
        self.max_len = None
        self.rand_transp=rand_transp



    def walkdir(self,folder,first_n_files):
        folder_list = os.listdir(folder)
        # folder_list = sorted(folder_list)
        if first_n_files is not None:
            folder_list = folder_list[:first_n_files]
        for fn in folder_list:
            if fn.endswith('.csv') and not fn.startswith('.'):
                yield fn

    def load_data(self,prime_folder,cont_folder_real,cont_folder_fake,note_range=[0,128],test=False,first_n_files=None):

        self.note_range= note_range
        dataset = []

        #Set up progress bar
        filecounter = 0
        for filepath in self.walkdir(prime_folder,first_n_files):
            filecounter += 1
        pbar = tqdm(self.walkdir(prime_folder,first_n_files), total=filecounter, unit="files")
        for filename in pbar:
            pbar.set_postfix(file=filename[:10], refresh=False)

            prime_filepath = os.path.join(prime_folder,filename)
            cont_filepath_real = os.path.join(cont_folder_real,filename)
            cont_filepath_fake = os.path.join(cont_folder_fake,filename)

            pr_real = PianorollMIREX()
            pr_real.make_from_file(prime_filepath,cont_filepath_real,note_range=note_range)
            pr_fake = PianorollMIREX()
            pr_fake.make_from_file(prime_filepath,cont_filepath_fake,note_range=note_range)
            dataset += [[pr_real,pr_fake]]

        if test:
            self.test=dataset

        else:
            n_data = len(dataset)
            n_valid = int(0.10*n_data)
            n_test = int(0.10*n_data)

            ptr = 0
            self.valid = dataset[ptr:ptr+n_valid]
            ptr += n_valid
            self.test = dataset[ptr:ptr+n_test]
            ptr += n_test
            self.train = dataset[ptr:]

        self.zero_pad()

        self.update_idx_list()



    def check_data(self):
        for subset in ['train','valid','test']:
            to_keep = []
            for i, pianoroll_real_fake in enumerate(getattr(self,subset)):
                is_OK = True
                for pianoroll in pianoroll_real_fake:
                    pr = pianoroll.roll[:,:pianoroll.length]

                    data_extended = np.pad(pr,[[0,0],[1,1]],'constant')
                    diff = data_extended[:,1:] - data_extended[:,:-1]
                    steady = np.where(np.sum(np.abs(diff),axis=0)==0)[0]

                    try:
                        if steady[-1] == pr.shape[1]:
                            steady = steady[:-1]
                    except IndexError:
                        assert steady.size == 0
                        is_OK = False

                    else:
                        try:
                            if steady[0] == 0:
                                steady = steady[1:]
                        except IndexError:
                            assert steady.size == 0
                            is_OK = False
                        else:
                            if steady.size == 0:
                                is_OK = False
                if is_OK:
                    to_keep += [pianoroll_real_fake]
                else:
                    print 'Removing',pianoroll_real_fake[0].name
            setattr(self,subset,to_keep)

        self.update_idx_list()
        return


    def update_idx_list(self):
        self.train_idx = list(range(len(self.train)))
        self.valid_idx = list(range(len(self.valid)))
        self.test_idx = list(range(len(self.test)))

    def shuffle_one(self,subset):
        shuffle(getattr(self,subset+'_idx'))


    def __max_len(self,dataset):
        if dataset == []:
            return 0
        else :
            return max(map(lambda x: max(x[0].length,x[1].length), dataset))

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
            piano_roll[0].zero_pad(max_len)
            piano_roll[1].zero_pad(max_len)

    def transpose_all_one(self,subset,tr_range=[-4,4]):
        data = getattr(self,subset)
        new_data = []
        for piano_roll in data:
            for j in range(*tr_range):
                tr_piano_roll = [piano_roll[0].transpose(j),piano_roll[1].transpose(j)]
                new_data += [tr_piano_roll]
        setattr(self,subset,new_data)

    def transpose_all(self):
        print "Transposing train set in every key..."
        #You only augment the training dataset
        for subset in ["train"]: #,"valid","test"]:
            self.transpose_all_one(subset)
        self.update_idx_list()


    def get_dataset(self,subset,with_names=False,with_key_masks=False):
        #Outputs an array containing all the piano-rolls (3D-tensor)
        #and the list of the actual lengths of the piano-rolls
        pr_list = getattr(self,subset)
        idx = getattr(self,subset+'_idx')
        n_files = len(pr_list)
        len_file = self.max_len
        n_notes = self.get_n_notes()

        dataset_real = np.zeros([n_files,n_notes,len_file])
        dataset_fake = np.zeros([n_files,n_notes,len_file])
        lengths_real = np.zeros([n_files],dtype=int)
        lengths_fake = np.zeros([n_files],dtype=int)
        if with_names:
            names = []
        if with_key_masks:
            key_masks = np.zeros([n_files,n_notes,len_file-1])
            key_lists = []


        for i,j in enumerate(idx):
            piano_roll_real = pr_list[j][0]
            roll_real = piano_roll_real.roll
            dataset_real[i] = roll_real
            lengths_real[i] = piano_roll_real.length
            piano_roll_fake = pr_list[j][1]
            roll_fake = piano_roll_fake.roll
            dataset_fake[i] = roll_fake
            lengths_fake[i] = piano_roll_fake.length
            if with_names:
                names += [piano_roll_real.name]
            if with_key_masks:
                max_length = max(piano_roll_real.length, piano_roll_fake.length)
                key_masks[i] = piano_roll_real.get_key_profile_matrix(length=max_length-1)
                key_list = [time for (key,time) in piano_roll_real.key_list]+[max_length]
                key_lists += [key_list]



        output = [dataset_real,dataset_fake, lengths_real, lengths_fake]
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

        return output


#
# prime_folder = 'data/PPDD-Sep2018_sym_poly_small/prime_csv'
# cont_folder_real = 'data/PPDD-Sep2018_sym_poly_small/cont_true_csv'
# cont_folder_fake = 'data/PPDD-Sep2018_sym_poly_small/cont_foil_csv'
#
# data = DatasetMIREXClassif()
# data.load_data(prime_folder,cont_folder_real,cont_folder_fake)
# data.check_data()
# for x,y,length in data.get_dataset_generator('train',15):
#     print x.shape, y.shape


# data.transpose_all()
# print 'transposed'
# dataset_real,dataset_fake,lengths_real,lengths_fake,key_masks,key_lists = data.get_dataset('train',with_key_masks=True)
# print dataset_real.shape,dataset_fake.shape,key_masks.shape
# dataset,lengths = data.get_dataset('train')
# print 'data'
# data.shuffle_one('train')
# print 'shuffled'
# print dataset.shape,key_masks.shape



# Removing ef8dab0b-26d8-478b-a848-1f9065d959f6.csv
# Removing cb7063ef-97af-43a9-8a20-900bf42a950d.csv
# Removing 77cc6150-9edf-4362-b4c6-b226ba750cd5.csv
# Removing 3f70320b-aac6-4af0-9c34-7bbcf6133a2b.csv
# Removing c8a56605-e771-4afe-b40c-0d3638bbd0e1.csv
# Removing 398e16c7-ceca-4234-a96a-01f6cf793bd6.csv
# Removing 703e8f95-904a-4ae6-9f61-25076b61efe4.csv
# Removing 166af1d3-701c-49c9-aa32-e1291bf58e59.csv
# Removing 1bd7f32c-1d4d-405d-a1e2-7820efd640b2.csv
# Removing 4ee3b5f0-4ced-427d-87d1-b364889d723e.csv

# Removing 1bd7f32c-1d4d-405d-a1e2-7820efd640b2.csv
# Removing 4ee3b5f0-4ced-427d-87d1-b364889d723e.csv
# Removing 77cc6150-9edf-4362-b4c6-b226ba750cd5.csv
# Removing 703e8f95-904a-4ae6-9f61-25076b61efe4.csv
# Removing c8a56605-e771-4afe-b40c-0d3638bbd0e1.csv
# Removing ef8dab0b-26d8-478b-a848-1f9065d959f6.csv
# Removing 166af1d3-701c-49c9-aa32-e1291bf58e59.csv
# Removing 398e16c7-ceca-4234-a96a-01f6cf793bd6.csv
# Removing 3f70320b-aac6-4af0-9c34-7bbcf6133a2b.csv
# Removing cb7063ef-97af-43a9-8a20-900bf42a950d.csv
