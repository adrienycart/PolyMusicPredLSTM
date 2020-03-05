# -*- coding: utf-8 -*-

import pretty_midi as pm
import numpy as np
import os
import copy

class Pianoroll(object):
    """Classe representing a piano-roll."""

    def __init__(self):
        self.roll = []
        self.name = ""
        self.length = 0
        self.end_time = 0
        self.note_range=[0,128]
        self.timestep_type = None
        self.with_onsets = False
        self.key = 0
        self.key_list = []
        self.key_list_times = []
        self.key_profiles_list = []
        self.corresp = []
        self.corresp_time_long = []
        self.corresp_note_short = []
        self.corresp_note_long = []
        self.corresp_event = []

    def make_from_file(self,filename,timestep_type,section=None,note_range=[0,128],key_method='main',with_onsets=False):
        midi_data = pm.PrettyMIDI(filename)
        self.make_from_pm(midi_data,timestep_type,section,note_range,key_method,with_onsets)
        self.name = os.path.splitext(os.path.basename(filename))[0]
        return

    def make_from_pm(self,data,timestep_type,section=None,note_range=[0,128],key_method='main',with_onsets=False):
        #Set all velocities to 1
        # for instr in data.instruments:
        #     for note in instr.notes:
        #         note.velocity=100

        self.timestep_type = timestep_type
        self.with_onsets = with_onsets

        ####Set timestep positions
        total_duration = data.get_piano_roll().shape[1]/100.0
        end_time = min(section[1],total_duration) if section is not None else total_duration
        self.end_time = end_time
        # Time long
        self.corresp_time_long = np.arange(0,end_time,0.18)
        # Note short
        end_tick = data.time_to_tick(end_time)
        PPQ = float(data.resolution)
        end_note = end_tick/PPQ
        note_steps = np.arange(0,end_note,1.0/12)
        tick_steps = np.round(note_steps*PPQ).astype(int)
        corresp = np.zeros_like(tick_steps,dtype=float)
        for i,tick in enumerate(tick_steps):
            corresp[i]=data.tick_to_time(int(tick))
        self.corresp_note_short = corresp
        # Note long
        self.corresp_note_long = corresp[0::3]
        # Event
        steps = np.unique(data.get_onsets())
        #Remove onsets that are within 50ms of each other (keep first one only)
        diff = steps[1:] - steps[:-1]
        close = diff<0.05
        while np.any(close):
            to_keep = np.where(np.logical_not(close))
            steps = steps[to_keep[0]+1]
            diff = steps[1:] - steps[:-1]
            close = diff<0.05
        index = np.argmin(np.abs(steps - self.end_time))
        steps = steps[:index+1]
        self.corresp_event = steps
        if section is not None and section[0]!=0:
            for table in 'corresp_event','corresp_note_long','corresp_note_short','corresp_time_long':
                corresp = getattr(self,table)
                index = np.argmin(np.abs(corresp - section[0]))
                setattr(self,table,corresp[index:])

        ##### Get the roll matrix
        if self.timestep_type == 'time_short':
            piano_roll = data.get_piano_roll(100)
            if not section == None :
                min_time_step = int(round(section[0]*100))
                max_time_step = int(round(section[1]*100))
                self.roll = piano_roll[:,min_time_step:max_time_step]
            else:
                self.roll = piano_roll
            self.binarize()
            if with_onsets:
                for instr in data.instruments:
                    for note in instr.notes:
                        onset_idx = int(note.start*100)
                        self.roll[note.pitch,onset_idx] = 2
        elif self.timestep_type == '40ms':
            piano_roll = data.get_piano_roll(25)
            if not section == None :
                min_time_step = int(round(section[0]*25))
                max_time_step = int(round(section[1]*25))
                self.roll = piano_roll[:,min_time_step:max_time_step]
            else:
                self.roll = piano_roll
            self.binarize()
            if with_onsets:
                for instr in data.instruments:
                    for note in instr.notes:
                        onset_idx = int(note.start*25)
                        piano_roll[note.pitch,onset_idx] = 2
        else:
            if self.timestep_type == 'time_long':
                self.corresp = self.corresp_time_long
                on_tol = 0.75
                off_tol = 0.4
            elif self.timestep_type == 'note_long':
                self.corresp = self.corresp_note_long
                on_tol = 0.5
                off_tol = 0.35
            elif self.timestep_type == 'note_short':
                self.corresp = self.corresp_note_short
                on_tol = 0.6
                off_tol = 0.55
            elif self.timestep_type == 'event':
                self.corresp = self.corresp_event
                on_tol = 0.35
                off_tol = 0.45
            else:
                raise ValueError('Timestep type not understood!: '+str(self.timestep_type))
            self.roll = get_piano_roll(data,self.corresp, on_tol=on_tol,off_tol=off_tol,with_onsets=with_onsets)

        #### Set key data
        self.set_key(data,section,key_method)
        self.set_key_list(data,section)
        self.set_key_profile_list(data,section)

        self.length = self.roll.shape[1]
        self.crop(note_range)
        # self.binarize()

        return



    def make_name(self):
        #TODO : add fields "composer", "name of the piece"
        #to be able to sort the pieces easily
        return

    def set_key(self,data,section,method='main'):
        key_sigs = data.key_signature_changes
        if section is None:
            end = data.get_piano_roll().shape[1]/100.0
            section = [0,end]

        if method == 'main':
            #Choose the tonality most represented in section
            prev_key = 0
            keys_section = []
            times_section = []
            for key_sig in key_sigs:
                key = key_sig.key_number
                time = key_sig.time
                if time < section[0]:
                    prev_key = key
                elif time==section[0]:
                    keys_section +=[key]
                    times_section += [time]
                else: #time > section[0]
                    if keys_section == [] and times_section==[]:
                        keys_section +=[prev_key]
                        times_section += [section[0]]

                    if time <= section[1]:
                        keys_section +=[key]
                        times_section += [min(time,section[1])]
                    #if time > section[1], do nothing


            times_section += [section[1]]
            times_section = np.array(times_section)
            diff = times_section[1:] - times_section[:-1]

            cumul_by_key = np.zeros([12,1])
            for key, dur in zip(keys_section,diff):
                cumul_by_key[key] += dur
            self.key = np.argmax(cumul_by_key)

        elif method=='first':
            #Choose first tonality in section
            start = section[0]
            key = 0
            for key_sig in key_sigs:
                time = key_sig.time
                if time <= start:
                    key = key_sig.key_number
                else:
                    break
            self.key = key

    def set_key_list(self,data,section):
        if section is None:
            section = [0,self.end_time]

        key_sigs = data.key_signature_changes

        prev_key = 0
        keys_section = []
        times_section = []

        for key_sig in key_sigs:
            key = key_sig.key_number
            time = key_sig.time
            if time < section[0]:
                prev_key = key
            elif time==section[0]:
                keys_section +=[key]
                times_section += [time]
            else: #time > section[0]
                if keys_section == [] and times_section==[]:
                    keys_section +=[prev_key]
                    times_section += [section[0]]
                if time <= section[1]:
                    keys_section +=[key]
                    times_section += [min(time,section[1])]
                #if time > section[1], do nothing

        self.key_list_times = zip(keys_section,times_section)

        key_list = []

        for key, time in zip(keys_section,times_section):
            if self.timestep_type == "time_short":
                new_time = int(round(time*float(100)))
            elif self.timestep_type == "40ms":
                new_time = int(round(time*float(25)))
            else:
                if self.timestep_type == "event" or self.timestep_type == "time_long" :
                    new_time = np.argmin(np.abs(self.corresp-time))
                else:
                    if self.timestep_type == "note_long":
                        fs=4
                    elif self.timestep_type == "note_short":
                        fs=12
                    time_quant = data.time_to_tick(time)/float(data.resolution)
                    new_time = int(round(time_quant*float(fs)))

            key_list += [(key,new_time)]


        self.key_list = key_list

    def set_key_profile_list(self,data,section):

        key_profiles=[]
        note_range = self.note_range
        roll = (data.get_piano_roll()>0).astype(int)

        if section is None:
            section = [0,self.end_time]

        if self.key_list_times == []:
            key_list = [(0,0)]
        else:
            key_list = self.key_list_times

        times = [x[1] for x in key_list]
        times += [section[1]]

        for time1,time2 in zip(times[:-1],times[1:]):
            idx1 = int(round(time1*100))
            idx2 = int(round(time2*100))

            # Cap values to 1, in case we use the value 2 for onsets
            key_profile = np.sum(np.minimum(roll[:,idx1:idx2],1),axis=1)/float(idx2-idx1)
            key_profiles += [key_profile]


        self.key_profiles_list = key_profiles


    def binarize(self):
        roll = self.roll
        self.roll = np.not_equal(roll,np.zeros(roll.shape)).astype(int)
        return

    def crop(self,note_range):
        if self.note_range != note_range:
            old_note_range = self.note_range

            min1 = old_note_range[0]
            max1 = old_note_range[1]
            min2 = note_range[0]
            max2 = note_range[1]

            key_profiles_cropped = []
            roll = self.roll

            #Crop roll
            if min1<min2:
                new_roll = roll[min2-min1:,:]
            else:
                new_roll = np.append(np.zeros([min1-min2,roll.shape[1]]),roll,0)

            if max1<=max2:
                new_roll = np.append(new_roll,np.zeros([max2-max1,roll.shape[1]]),0)
            else:
                new_roll = new_roll[:-(max1-max2),:]

            #Crop key profiles
            for k in self.key_profiles_list:
                if min1<min2:
                    new_k = k[min2-min1:]
                else:
                    new_k = np.append(np.zeros([min1-min2]),k,0)

                if max1<=max2:
                    new_k = np.append(new_k,np.zeros([max2-max1]),0)
                else:
                    new_k = new_k[:-(max1-max2)]
                key_profiles_cropped+=[new_k]

            self.roll = new_roll
            self.key_profiles_list = key_profiles_cropped
            self.note_range = note_range
        return

    def zero_pad(self,length):
        #Makes the piano-roll of given length
        #Cuts if longer, zero-pads if shorter
        #DO NOT change self.length !!

        roll = self.roll
        if self.length >= length:
            roll_padded = roll[:,0:length]
        else :
            roll_padded = np.pad(roll,pad_width=((0,0),(0,length-roll.shape[1])),mode='constant')
        self.roll = roll_padded
        return


    def cut(self,len_chunk,keep_padding=True):
        #Returns the roll cut in chunks of len_chunk elements, as well as
        #the list of lengths of the chunks
        #The last element is zero-padded to have len_chunk elements

        roll = self.roll
        if keep_padding:
            size = roll.shape[1]
        else:
            size = self.length
        N_notes = roll.shape[0]
        n_chunks = int(np.ceil(float(size)/len_chunk))

        roll_cut = np.zeros([n_chunks,N_notes,len_chunk])
        lengths = np.zeros([n_chunks])

        j = 0
        n = 0
        length = self.length
        while j < size:
            lengths[n] = min(length,len_chunk)
            length = max(0, length-len_chunk)
            if j + len_chunk < size:
                roll_cut[n]= roll[:,j:j+len_chunk]
                j += len_chunk
                n += 1
            else : #Finishing clause : zero-pad the remaining
                roll_cut[n,:,:]= np.pad(roll[:,j:size],pad_width=((0,0),(0,len_chunk-(size-j))),mode='constant')
                j += len_chunk
        return roll_cut, lengths


    def copy_section(self,section):
        data = copy.deepcopy(self)
        assert section[0] < section[1]
        if self.timestep_type == "time_short":
            begin_index = int(round(section[0]*100))
            end_index = int(round(section[1]*100))
        elif self.timestep_type == "40ms":
            begin_index = int(round(section[0]*25))
            end_index = int(round(section[1]*25))
        else:
            begin_index = np.argmin(np.abs(section[0],self.corresp))
            begin_index = np.argmin(np.abs(section[1],self.corresp))
        data.roll = self.roll[:,begin_index:end_index]
        data.length = end_index-begin_index
        data.end_time = None
        # Shorten all corresp tables
        for table in 'corresp_event','corresp_note_long','corresp_note_short','corresp_time_long':
            corresp = getattr(self,table)
            index0 = np.argmin(np.abs(corresp - section[0]))
            index1 = np.argmin(np.abs(corresp - section[1]))
            setattr(data,table,corresp[index0:index1])

        #Make new key_list
        prev_key = 0
        new_key_list = []
        for key,time in self.key_list:
            key = key_sig.key_number
            time = key_sig.time
            if time < begin_index:
                prev_key = key
            elif time==begin_index:
                new_key_list += [[key,time]]
            else: #time > begin_index
                if keys_section == [] and times_section==[]:
                    new_key_list += [[prev_key,begin_index]]
                if time < end_index:
                    new_key_list += [[key,time]]
                #if time >= section[1], do nothing
        data.key_list = new_key_list

        # TODO: Adapt data.key based on new key_list (if needed later on)

        return data

    def transpose(self,diff):
        #Returns a copy of self, transposed of diff semitones
        #diff can be positive or negative
        pr_trans = copy.deepcopy(self)
        roll = self.roll
        if diff<0:
            pr_trans.roll = np.append(roll[-diff:,:],np.zeros([-diff,roll.shape[1]]),0)
            new_profile_list = []
            for profile in self.key_profiles_list:
                new_profile_list += [np.append(profile[-diff:],np.zeros([-diff]))]
            pr_trans.key_profiles_list = new_profile_list
        elif diff>0:
            pr_trans.roll = np.append(np.zeros([diff,roll.shape[1]]),roll[:-diff,:],0)
            new_profile_list = []
            for profile in self.key_profiles_list:
                new_profile_list += [np.append(np.zeros([diff]),profile[:-diff])]
            pr_trans.key_profiles_list = new_profile_list
        #if diff == 0 : do nothing

        pr_trans.key = (self.key+diff)%12
        pr_trans.key_list = [((key+diff)%12,time) for (key,time) in self.key_list]

        return pr_trans


    def timestretch(self):
        pr_stretch = copy.deepcopy(self)
        roll = self.roll
        length = roll.shape[1]
        #duplicate each column by multiplying by a clever matrix
        a = np.zeros([length,2*length])
        i,j = np.indices(a.shape)
        a[i==j//2]=1
        pr_stretch.roll = np.matmul(roll,a)
        pr_stretch.length = 2*self.length
        pr_stretch.key_list = [(key,time*2) for (key,time) in self.key_list]

        # TODO: modify the corresp tables accordingly (if needed later on)

        return pr_stretch


    def get_roll(self):
        return self.roll, self.length

    def get_gt(self):
        return self.roll[:,1:]

    def get_key_profile_matrix(self,length=None):
        key_list = self.key_list

        roll = self.roll[:,1:]
        shape = roll.shape
        if length is None:
            length = min(roll.shape[1],self.length+1) #Allow 1 more timesteps just in case

        if key_list == []:
            key_list = [(0,0)]

        times = [max(0,x[1]-1) for x in key_list] # -1 because the targets are shifted
        times += [length]
        key_profile_matrix = np.zeros(shape)
        for time1,time2,key_profile in zip(times[:-1],times[1:],self.key_profiles_list):
            key_profile_repeat = np.tile(key_profile,(time2-time1,1)).transpose()
            key_profile_matrix[:,time1:time2]=key_profile_repeat

        return key_profile_matrix

    def get_key_profile_matrix_octave(self):

        #Octave-wrapping
        key_profile = self.get_key_profile_matrix()
        octave_profile = np.zeros([12,key_profile.shape[1]],dtype=float)
        i = np.arange(key_profile.shape[0])

        n_active = np.ones_like(key_profile)
        octave_active =  np.zeros_like(octave_profile)

        for pitch in range(12):
            octave_profile[pitch,:] = np.sum(key_profile[i%12==pitch,:],axis=0)
            octave_active[pitch,:] = np.sum(n_active[i%12==pitch,:],axis=0)
        octave_profile = octave_profile#/octave_active
        print octave_active[:,0]

        #
        # #Normalisation by key segment: max = 1
        # length = min(key_profile.shape[1],self.length+1) #Allow 1 more timesteps just in case
        # key_list = self.key_list
        # if key_list == []:
        #     key_list = [(0,0)]
        # times = [max(0,x[1]-1) for x in key_list] # -1 because the targets are shifted
        # times += [length]
        # for time1,time2 in zip(times[:-1],times[1:]):
        #     octave_profile[:,time1:time2] = octave_profile[:,time1:time2]/float(np.max(octave_profile[:,time1:time2]))

        return octave_profile



    def get_key_mask_matrix(self,mode='major'):
        #Returns a matrix of given shape with 1 if the given pitch is in the scale at the given time
        key_list = self.key_list
        shape = self.roll.shape
        note_range = self.note_range
        key_list_len = len(key_list)
        assert key_list[0][1] == 0
        assert note_range[1]-note_range[0] == shape[0]
        mask_matrix = np.zeros(shape)
        for i, (key,start) in enumerate(key_list):
            #Do not generate mask for longer than shape of matrix
            if start < shape[1]:
                if i == key_list_len-1:
                    end = shape[1]
                else:
                    end = key_list[i+1][1]
                key_mask = scale_template(key,note_range,mode)
                key_indexes = [x - note_range[0] for x in key_mask]
                mask_matrix[key_indexes,start:end]= 1
        return mask_matrix


    def get_weights_ss_tr(self):
        pr = self.roll
        data_extended = np.pad(pr,[[0,0],[1,1]],'constant')
        diff = np.abs(data_extended[:,1:] - data_extended[:,:-1])
        sum_diff = np.sum(diff,axis=0)[1:-1]
        steady_mask = (sum_diff==0)
        trans_mask = np.logical_not(steady_mask)
        n_steady = np.sum(steady_mask.astype(int))
        n_trans = np.sum(trans_mask.astype(int))
        count_trans = sum_diff[trans_mask]

        weights = np.ones([pr.shape[1]-1],dtype=float)


        weights[steady_mask] = 1.0/n_steady
        weights[trans_mask] = 1.0/(n_trans*count_trans)


        return weights

    def get_weights_key(self):

        pr = self.roll
        data_extended = np.pad(pr,[[0,0],[1,1]],'constant')
        diff = np.abs(data_extended[:,1:] - data_extended[:,:-1])
        sum_diff = np.sum(diff,axis=0)[1:-1]
        steady_mask = (sum_diff==0)
        trans_mask = np.logical_not(steady_mask)
        n_steady = np.sum(steady_mask.astype(int))
        n_trans = np.sum(trans_mask.astype(int))
        count_trans = sum_diff[trans_mask]

        key_matrix = (self.get_key_profile_matrix()>0.05).astype(int)
        sum_key = np.sum(key_matrix,axis=0)

        weights = np.ones([pr.shape[1]-1],dtype=float)

        weights = weights/sum_key

        weights[steady_mask] = 1.0/(n_steady)
        weights[trans_mask] = 1.0/(n_trans)

        print n_steady, n_trans

        return weights





def get_piano_roll(midi_data,steps,on_tol=0,off_tol=0,with_onsets=False):
    #on_tol and off_tol are in fraction of a step

    output = np.zeros([128,len(steps)])
    end_time = steps[-1]

    for instr in midi_data.instruments:
        for note in instr.notes:
            if note.start < end_time: #Only consider notes withing steps range
                if False:#note.start > 30 and 50+21<note.pitch<60+21:
                    verbose=True
                else:
                    verbose=False



                #### ONSET
                start_pos = np.argmin(np.abs(steps-note.start))

                if verbose:

                    print '-----------------'
                    print 'ONSET',note.pitch
                    print note.start,steps[start_pos]
                    print steps[start_pos]-note.start, (steps[start_pos+1] - steps[start_pos])*on_tol
                    print midi_data.time_to_tick(note.start),midi_data.time_to_tick(steps[start_pos]),midi_data.time_to_tick(note.start)-midi_data.time_to_tick(steps[start_pos])


                if note.start <= steps[start_pos] :
                    # If note starts before closest step, we consider it active at that step
                    pass
                else:
                    # If note starts after closest step, we consider it active at the next step,
                    # unless it starts within tolerance threshold
                    # tolerance allows slight imprecision, both for onsets and offset

                    try: # steps[start_pos+1]  might not exist!
                        is_within_thresh = note.start-steps[start_pos] <= (steps[start_pos+1] - steps[start_pos])*on_tol
                    except: #If it doesnt, do not use threshold
                        is_within_thresh=False

                    if verbose:
                        print is_within_thresh

                    if is_within_thresh:
                        pass
                    else:
                        start_pos += 1

                #### OFFSET
                end_pos = np.argmin(np.abs(steps-note.end))

                if verbose:
                    print '-----------------'
                    print 'OFFSET',note.pitch
                    print note.end,steps[end_pos]
                    print steps[end_pos]-note.end, (steps[end_pos+1] - steps[end_pos])*on_tol
                    print midi_data.time_to_tick(note.end),midi_data.time_to_tick(steps[end_pos]),midi_data.time_to_tick(note.end)-midi_data.time_to_tick(steps[end_pos])


                if note.end <= steps[end_pos] :
                    # If note ends before closest step, we consider it ends at that step
                    pass
                else:
                    # If note ends after closest step, we consider it active at the next step,
                    # unless it starts within tolerance threshold
                    # tolerance allows slight imprecision, both for onsets and offset

                    try: # steps[end_pos+1]  might not exist!
                        is_within_thresh = note.end - steps[end_pos] <= (steps[end_pos+1] - steps[end_pos])*off_tol
                    except: #If it doesnt, do not use threshold
                        is_within_thresh=False

                    if verbose:
                        print is_within_thresh

                    if is_within_thresh:
                        pass
                    else:
                        end_pos += 1

                if verbose:
                    print start_pos,end_pos,end_pos-start_pos
                output[note.pitch,start_pos:end_pos]=1
                if with_onsets:
                    output[note.pitch,start_pos]=2

    return output


def get_piano_roll_tick_tol(midi_data,steps,on_tol=0,off_tol=0):
    output = np.zeros([128,len(steps)])

    for instr in midi_data.instruments:
        for note in instr.notes:
            #### ONSET
            start_pos = np.argmin(np.abs(steps-note.start))
            if midi_data.time_to_tick(note.start)<=midi_data.time_to_tick(steps[start_pos])+on_tol:
                # If note starts before closest step, we consider it active at that step
                # tolerance allows slight imprecision, both for onsets and offsets
                pass
            else:
                # If note starts after closest step, we consider it active at the next step
                start_pos += 1

            #### OFFSET
            end_pos = np.argmin(np.abs(steps-note.end))
            if midi_data.time_to_tick(note.end)<=midi_data.time_to_tick(steps[end_pos])+off_tol:
                # If note ends before closest step, we consider it ends at that step
                # tolerance allows slight imprecision, both for onsets and offsets
                pass
            else:
                # If note ends after closest step, we consider it active at the next step
                end_pos += 1

            output[note.pitch,start_pos:end_pos]=1

    return output



def get_quant_piano_roll(midi_data,fs=4,section=None):
    # DEPRECATED

    data = copy.deepcopy(midi_data)

    PPQ = float(data.resolution)

    for instr in data.instruments:
        for note in instr.notes:
            note.start = data.time_to_tick(note.start)/PPQ
            note.end = data.time_to_tick(note.end)/PPQ


    # quant_piano_roll = data.get_piano_roll(fs)

    # PROPER WAY OF IMPORTING PIANO-ROLLS !
    length = data.get_piano_roll().shape[1]/100.0
    quant_piano_roll = data.get_piano_roll(times=np.arange(0,length,1/float(fs)))
    quant_piano_roll = (quant_piano_roll>=7).astype(int)


    if not section == None:
        begin = section[0]
        end = section[1]
        assert begin < end

        begin_index = int(round(midi_data.time_to_tick(begin)/PPQ*fs))
        end_index = int(round(midi_data.time_to_tick(end)/PPQ*fs))
        quant_piano_roll = quant_piano_roll[:,begin_index:end_index]

    return quant_piano_roll

def get_event_roll(midi_data,section=None):
    # DEPRECATED

    steps = np.unique(midi_data.get_onsets())

    #Remove onsets that are within 50ms of each other (keep first one only)
    diff = steps[1:] - steps[:-1]
    close = diff<0.05
    while np.any(close):
        to_keep = np.where(np.logical_not(close))
        steps = steps[to_keep[0]+1]
        diff = steps[1:] - steps[:-1]
        close = diff<0.05


    for s1,s2 in zip(steps[:-1],steps[1:]):
        if s2-s1 < 0.05:
            print s1, s2, s2-s1
            print round(s1*20)/20, round(s2*20)/20

    pr = midi_data.get_piano_roll(fs=500,times=steps)

    pr = (pr>=7).astype(int)

    if not section is None:
        #Select the relevant portion of the pianoroll
        begin = section[0]
        end = section[1]
        assert begin < end
        begin_index = np.argmin(np.abs(begin-steps))
        end_index = np.argmin(np.abs(end-steps))
        pr = pr[:,begin_index:end_index]

    return pr, steps

def get_event_roll_OLD(midi_data,section=None):
    # DEPRECATED

    roll = get_quant_piano_roll(midi_data,fs=12,section=section)
    data_extended = np.pad(roll,[[0,0],[1,1]],'constant')
    diff = data_extended[:,1:] - data_extended[:,:-1]
    trans_mask = np.logical_or(diff==1, diff==-1)
    transitions= np.where(trans_mask)


    transitions_unique = np.unique(transitions[1])
    #We drop the last offset if it corresponds to an added zero, and add index zero if not already in
    if transitions_unique[-1]==roll.shape[1]:
        transitions_unique = transitions_unique[:-1]
    if transitions_unique[0]!=0:
        transitions_unique = np.concatenate(([0],transitions_unique),axis=0)

    event_roll = roll[:,transitions_unique]
    return event_roll,transitions_unique



def scale_template(scale,note_range=[21,109],mode='major'):
    #Returns a 88*1 matrix with True if the corresponding pitch is in the given scale
    #If scale is minor, natural, melodic and harmonic minor are accepted.

    scale = int(scale)
    note_min, note_max = note_range
    key = scale%12

    # is_major = scale//12==0
    #Treat everything as in C
    note_min_t = note_min-key
    note_max_t = note_max-key
    octave_max = note_max_t//12
    octave_min = note_min_t//12

    if mode == 'major':
        scale = [0,2,4,5,7,9,11]
    elif mode == 'minor':
        key = key - 3
        scale = [0,2,3,5,7,8,10]
    elif mode == 'harmonic_minor':
        key = key - 3
        scale = [0,2,3,5,7,8,11]
    elif mode == 'melodic_minor':
        key = key - 3
        scale = [0,2,3,5,7,9,11]
    current_scale = [x+key for x in scale]
    single_notes = []

    for i in range(octave_min,octave_max+1):
        to_add =  [12*i+x for x in scale if 12*i+x>= note_min_t and 12*i+x< note_max_t]
        single_notes = single_notes + to_add
    #Transpose back to the correct key
    output = [x + key for x in single_notes]
    return output


def align_matrix(input_matrix,corresp,input_fs,section=None,method='avg'):
    #Makes a quantised input
    #The original input has to be downsampled: the method argument
    #specifies the downsampling method.

    n_notes = input_matrix.shape[0]
    end_sec = min(input_matrix.shape[1]/float(input_fs),corresp[-1])
    (n_steps,_),_ = get_closest(end_sec,corresp)

    aligned_input = np.zeros([n_notes,n_steps])



    def fill_value(sub_input,i):
        #Computes the value of the note-based input, and puts it in the matrix
        #sub_input is the portion of the input corresponding to the current sixteenth note
        #i is the index of the current sixteenth note


        if method=='avg':
            #Take the mean of the values for the current sixteenth note
            value = np.mean(sub_input,axis=1)
        elif method=='step':
            #Take the mean of first quarter of the values for the current sixteenth note
            #Focus on the attacks
            step = max(int(round(0.25*sub_input.shape[1])),1)
            value = np.mean(sub_input[:,:step],axis=1)
        elif method=='exp':
            #Take the values multiplied by an exponentially-decaying window.
            #Accounts for the exponentially-decaying nature of piano notes
            def exp_window(length,end_value=0.05):
                a = np.arange(0,length)
                b = pow(0.1,1.0/length)
                return np.power(np.full(a.shape,b),a)
            window = exp_window(sub_input.shape[1])
            sub_input_window = sub_input*window[np.newaxis,:]
            value = np.sum(sub_input_window,axis=1)
        elif method=='max':
            #If a note is active in the considered sixteenth-note time step,
            #(actually: active more than 5% and more than 3 samples, to account for imprecisions of the alignment)
            #then it is active for the whole time step.
            #Used to convert binary inputs from time-based to note-based time steps.

            value_mean = np.mean(sub_input,axis=1)
            value_sum = np.sum(sub_input,axis=1)
            value = (np.logical_and(value_mean>0.05,value_sum>=3)).astype(int)
        elif method=='quant':
            #If a note is active more than half of the sixteenth note time step,
            #it is active for the whole time step.
            #Used to quantise binary inputs (ie align onsets and offsets to the closest sixteenth note)
            value = np.mean(sub_input,axis=1)
            value = (value>0.5).astype(int)
        else:
            raise ValueError("Method type not understood: "+str(method))


        aligned_input[:,i]=value

    for i in range(aligned_input.shape[1]-1):
        begin = corresp[i]
        end = corresp[i+1]
        begin_index = int(round(begin*input_fs)) #input_fs is the sampling frequency of the input
        end_index = max(int(round(end*input_fs)),int(round(begin*input_fs))+1) #We want to select at least 1 frame of the input
        sub_input = input_matrix[:,begin_index:end_index]

        if sub_input.shape[1]==0:
            #Used for debugging
            print("error making align input")
            print(begin, end,end-begin)
            print(begin_index, end_index)
            print(begin*input_fs,end*input_fs)
            print(sub_input.shape)
            print(input_matrix.shape)

        fill_value(sub_input,i)

    last_begin = corresp[-1]
    last_begin_index = int(round(last_begin*input_fs))
    last_sub_input = input_matrix[:,last_begin_index:]

    #Prevents some warnings when the corresp file is not perfect
    if not last_sub_input.shape[1]==0:
        fill_value(sub_input,-1)

    if not section==None:
        #Select only the relevant portion of the input
        begin = section[0]
        end = section[1]
        assert begin < end
        [begin_index, begin_val],[index2, val2] = get_closest(begin,corresp)
        [end_index, end_val],[index2, val2] = get_closest(end,corresp)

        aligned_input = aligned_input[:,begin_index:end_index]

    return aligned_input

def convert_note_to_time(pianoroll,corresp,output_fs,max_len=None):
    #Converts a pianoroll from note-based to time-based time steps,
    #using the corresp table.

    fs=output_fs

    #Make sure corresp is not longer than pianoroll
    roll_end = pianoroll.shape[1]
    if roll_end < len(corresp):
        corresp = corresp[:roll_end]


    #Set length of resulting piano-roll
    if max_len==None:
        length = corresp[-1]
        n_steps = corresp.shape[0]
    else:
        length = min(max_len, corresp[-1])
        [n_steps,val], _  = get_closest(max_len,list(corresp))
    n_notes = pianoroll.shape[0]
    n_times = int(round(length*fs))

    time_roll = np.zeros([n_notes,n_times])

    for i in range(n_steps-1):
        time1 = corresp[i]
        time2 = corresp[i+1]

        index1 = int(round(time1*fs))
        index2 = int(round(time2*fs))

        active = pianoroll[:,i:i+1] #do this to keep the shape [88,1] instead of [88]

        time_roll[:,index1:index2]=np.repeat(active,index2-index1,axis=1)

    last_time = corresp[n_steps-1]
    last_index = int(round(last_time*fs))
    last_active = np.transpose([pianoroll[:,n_steps-1]],[1,0])

    time_roll[:,last_index:]=np.repeat(last_active,max(n_times-last_index,0),axis=1)

    return time_roll

def get_closest(e,l):
    #Get index of closest element in list l from value e
    #l has to be ordered
    #first output is the closest [index, value]
    #second output is the second closest [index, value]

    if 'numpy.ndarray' in str(type(l)):
        l=list(l)

    default_val = l[-1]
    val2 = next((x for x in l if x>=e),default_val)
    index2 = l.index(val2)
    if index2==0:
        index1 = index2+1
    else:
        index1 = index2-1
    val1 = l[index1]


    if abs(val2-e) < abs(e-val1):
        return [index2, val2], [index1, val1]
    else:
        return [index1, val1],[index2, val2]
#
# import matplotlib.pyplot as plt
# #
# # np.seterr(all='raise')
# # l = []
# folder = "data/Piano-midi.de/"
# for subfolder in ["test"]:#,"valid","test"]:
#     subfolder = os.path.join(folder,subfolder)
#     for fn in os.listdir(subfolder):
#         if fn.endswith('.mid') and not fn.startswith('.'):
#             filename = os.path.join(subfolder,fn)
#             print filename
#
#             pr = Pianoroll()
#             pr.make_from_file(filename,"note_long",note_range=[21,109],section=[0,15])
#             weights = pr.get_weights_key()
#             matrix = np.concatenate([weights[None,:]/max(weights),pr.roll[:,1:]],axis=0)
#             # print pr.roll[:,1:].shape, matrix.shape
#             plt.imshow(matrix,aspect="auto",origin='lower')
#             plt.show()

#             key_profiles1 = pr.key_profiles_list
#
#             pr = Pianoroll()
#             pr.make_from_file(filename,"note_short",note_range=[21,109])
#             key_profiles2 = pr.key_profiles_list
#
#             for k1, k2 in zip(key_profiles1,key_profiles2):
#                 print k1.shape
#                 print np.all(np.equal(k1>0.05,k2>0.05))




#             data = pm.PrettyMIDI(filename)
#             end_time = 60
#             end_tick = data.time_to_tick(end_time)
#             PPQ = float(data.resolution)
#             print PPQ
#             end_note = end_tick/PPQ
#             note_steps = np.arange(0,end_note,0.25)
#             tick_steps = np.round(note_steps*PPQ).astype(int)
#             corresp = np.zeros_like(tick_steps,dtype=float)
#             for i,tick in enumerate(tick_steps):
#                 corresp[i]=data.tick_to_time(int(tick))
#
#             roll1 = get_piano_roll(data,corresp,on_tol=0.50,off_tol=0.35)
#             # roll1 = get_piano_roll_tick_tol(data,corresp,on_tol=60,off_tol=42)
#             pr = Pianoroll()
#             pr.make_from_file(filename,'time_short',[0,60],[21,109])
#
#             fig, (ax1,ax2) = plt.subplots(2,1,figsize=(12,6.5))
#             ax1.imshow(pr.roll,origin='lower',aspect='auto')
#             for time in corresp:
#                 ax1.plot([time*100]*2,[0,87], linewidth=0.2, color='lightgrey')
#             ax2.imshow(roll1,origin='lower',aspect='auto')
#             ax2.set_xticks([x+0.5 for x in list(range(roll1.shape[1]))])
#             ax2.grid(True,axis='both',color='lightgrey',linewidth=0.2 )
#             plt.show()


# import matplotlib.pyplot as plt
#
# filename = "data/Piano-midi.de/test/chpn-p7.mid"
# data = pm.PrettyMIDI(filename)
# # end_time = 15
# # end_tick = data.time_to_tick(end_time)
# # PPQ = float(data.resolution)
# # end_note = end_tick/PPQ
# # note_steps = np.arange(0,end_note,0.25)
# # tick_steps = np.round(note_steps*PPQ).astype(int)
# # corresp = np.zeros_like(tick_steps,dtype=float)
# # for i,tick in enumerate(tick_steps):
# #     corresp[i]=data.tick_to_time(int(tick))
# # roll1 = get_piano_roll(data,corresp,on_tol=10,off_tol=20)
# pr = Pianoroll()
# pr.make_from_file(filename,'event',[0,15],[21,109])
# print pr.corresp
#
# plt.imshow(pr.roll,origin='lower',aspect='auto')
# plt.show()

#
# fig, (ax1,ax2) = plt.subplots(2,1)
# ax1.imshow(pr.roll,origin='lower',aspect='auto')
# for time in corresp:
#     ax1.plot([time*100]*2,[0,87], linewidth=1, color='lightgrey')
# ax2.imshow(roll1,origin='lower',aspect='auto')
# ax2.set_xticks([x+0.5 for x in list(range(roll1.shape[1]))])
# ax2.grid(True,axis='both',color='lightgrey')
# plt.show()

# import matplotlib.pyplot as plt
# subfolder = 'data/Piano-midi.de/test'
# for fn in os.listdir(subfolder):
#     if fn.endswith('.mid') and not fn.startswith('.'):
#         print fn
#         filename = os.path.join(subfolder,fn)
#         pr1 = Pianoroll()
#         pr1.make_from_file(filename,'event',[0,60],[21,109])
#
#         pr2 = Pianoroll()
#         pr2.make_from_file(filename,'note_long',[0,60],[21,109])
#
#         fig, (ax1,ax2) = plt.subplots(2,1)
#         ax1.bar(list(range(88)),pr1.get_key_profile_matrix()[:,0]>0.05,88*[1])
#         ax2.bar(list(range(88)),pr2.get_key_profile_matrix()[:,0]>0.05,88*[1])
#         plt.show()
#
#         fig, (ax1,ax2) = plt.subplots(2,1)
#         ax1.plot(pr1.get_key_profile_matrix()[:,0])
#         ax2.plot(pr2.get_key_profile_matrix()[:,0])
#         plt.show()





# pr = Pianoroll()
# pr.roll = np.array([[1,1,2,2,3,3],[1,1,2,2,3,3],[1,1,2,2,3,3]])
# pr.length = 6
# print pr.cut(4,keep_padding=True)
# print pr.cut(8,keep_padding=False)


class PianorollMIREX(Pianoroll):
    def __init__(self):
        self.roll = []
        self.name = ""
        self.length = 0
        self.end_time = 0
        self.note_range=[0,128]
        self.timestep_type = None
        self.key=0
        self.key_list = [(0,0)]
        self.key_list_times = [(0,0)]
        self.key_profiles_list = []

    def make_from_file(self,prime_path,cont_path,note_range=[0,128]):
        prime_matrix = np.loadtxt(prime_path,delimiter=',')
        cont_matrix = np.loadtxt(cont_path,delimiter=',')

        self.roll = self.build_piano_roll(prime_matrix,cont_matrix,note_range)
        self.length = self.roll.shape[1]
        self.set_key_profile_list()
        self.name = os.path.basename(prime_path)

    def build_piano_roll(self,prime_csv,cont_csv,note_range=[0,128]):

        length =  cont_csv[-1,0]-  prime_csv[0,0]
        begin = prime_csv[0,0]
        n_steps = int(round(length*12))
        pr = np.zeros([128,n_steps])
        for matrix in [prime_csv,cont_csv]:
            for row in matrix:
                start = row[0]-begin
                pitch = int(row[1])
                dur = row[3]
                end = start+dur
                start_step, end_step = int(round(start*12)),int(round(end*12))


                pr[pitch,start_step:end_step] = 1
        pr = pr[note_range[0]:note_range[1],:]

        return pr

    def set_key_profile_list(self):
        roll = self.roll
        key_profile = np.sum(roll,axis=1)/float(roll.shape[1])
        self.key_profiles_list += [key_profile]


def check_correct_onsets(roll):

    for i in range(roll.shape[1]-1):
        frame_1 = roll[:,i]
        frame_2  = roll[:,i+1]


        if np.all(np.logical_not(np.logical_and(frame_1==0,frame_2==1))):
            pass
        else:
            # incorrect= np.where(np.logical_not(np.logical_or.reduce((diff==2,diff==1,diff==0))))
            # for idx in zip(incorrect[0],incorrect[1]):
            #     print idx, diff[idx]
            print i
            import matplotlib.pyplot as plt
            plt.imshow(pr.roll,aspect='auto',origin='lower')
            plt.show()

# folder = 'data/Piano-midi.de/train'
# for fn in os.listdir(folder):
#     if fn.endswith('.mid') and not fn.startswith('.'):
#         filename = os.path.join(folder,fn)
#         print filename
#         pr = Pianoroll()
#         pr.make_from_file(filename,'note_long',note_range=[21,109],with_onsets=True)
#         check_correct_onsets(pr.roll)
#         print pr.get_key_profile_matrix().shape


# class PianorollMIREX_Classif(PianorollMIREX):
#     def __init__(self):
#         self.roll = []
#         self.roll_fake = []
#         self.name = ""
#         self.length = 0
#         self.end_time = 0
#         self.note_range=[0,128]
#         self.timestep_type = None
#         self.key=0
#         self.key_list = [(0,0)]
#         self.key_list_times = [(0,0)]
#         self.key_profiles_list = []
#
#     def make_from_file(self,prime_path,cont_path_real,cont_path_fake):
#         prime_matrix = np.loadtxt(prime_path,delimiter=',')
#         cont_matrix_real = np.loadtxt(cont_path_real,delimiter=',')
#         cont_matrix_fake = np.loadtxt(cont_path_fake,delimiter=',')
#
#         self.roll = super(PianorollMIREX_Classif,self).build_piano_roll(prime_matrix,cont_matrix_real)
#         self.roll_fake = super(PianorollMIREX_Classif,self).build_piano_roll(prime_matrix,cont_matrix_fake)
#         self.length = max(self.roll_fake.shape[1],self.roll.shape[1])
#         self.set_key_profile_list()
#         self.name = os.path.basename(prime_path)
#
#     def set_key_profile_list(self):
#         roll = self.roll
#         key_profile = np.sum(roll,axis=1)/float(roll.shape[1])
#         self.key_profiles_list += [key_profile]


# pr = PianorollMIREX_Classif()
# pr.make_from_file('data/PPDD-Sep2018_sym_poly_small/prime_csv/232996a6-8154-4e2c-ad68-f528e749dcba.csv','data/PPDD-Sep2018_sym_poly_small/cont_true_csv/232996a6-8154-4e2c-ad68-f528e749dcba.csv','data/PPDD-Sep2018_sym_poly_small/cont_foil_csv/232996a6-8154-4e2c-ad68-f528e749dcba.csv')
# print pr.roll.shape, pr.roll_fake.shape, pr.length, pr.length_fake
# print np.where(pr.key_profiles_list[0])
# pr = pr.transpose(-1)
# print np.where(pr.key_profiles_list[0])
