import numpy as np

def tests_key_XE(data,model,save_path):
    sess, saver = model.load(save_path,None)
    dataset, lengths, key_masks,key_lists = data.get_dataset('valid',with_key_masks=True)

    inputs = model._transpose_data(dataset)
    targets = model._transpose_data(dataset[:,:,1:])
    k_masks = model._transpose_data(key_masks)

    x = model.inputs
    seq_len = model.seq_lens
    y = model.labels
    thresh = model.thresh
    thresh_key = model.thresh_key
    k_mask = model.key_masks
    k_lists = model.key_lists
    key_XE = model.cross_entropy_key

    out = {}
    for threshold in [0.01,0.02,0.05,0.10]:
        feed_dict = {x:inputs,
                     y:targets,
                     seq_len:lengths,
                     thresh_key:threshold,
                     k_mask:k_masks,
                     k_lists:key_lists}
        result = sess.run(key_XE,feed_dict)
        out[threshold] = result
        print "Thresh:", threshold
        print "result:", result
    return out


def get_best_thresh(dataset,lengths,key_masks,key_lists,model,save_path,verbose=False,sess=None,saver=None,pred_values=None,logits=False,trim_outputs=True):


    if sess==None and saver==None and pred_values is None:
        sess, saver = model.load(save_path,None)

    if pred_values is None:
        metrics_function = lambda thresh: model.compute_eval_metrics_pred(dataset,lengths,key_masks,key_lists,threshold=thresh,save_path=save_path,sess=sess,saver=saver)
    else:
        metrics_function = lambda thresh: model.compute_eval_metrics_from_outputs(dataset,pred_values,lengths,key_masks,key_lists,thresh,logits=logits,trim_outputs=trim_outputs)

    F_list1 = []
    thresh_list1 = np.arange(0,1,0.1)

    for thresh in thresh_list1:
        result = metrics_function(thresh)
        F_list1 += [result[0]]

    max_value1 = max(F_list1)
    max_index1 = F_list1.index(max_value1)
    max_thresh1 = thresh_list1[max_index1]

    F_list2 = []
    thresh_list2 = np.arange(max(0,max_thresh1-0.09),min(1,max_thresh1+0.095),0.01)
    for thresh in thresh_list2:
        result = metrics_function(thresh)
        F_list2 += [result[0]]

    max_value2 = max(F_list2)
    max_index2 = F_list2.index(max_value2)
    max_thresh2 = thresh_list2[max_index2]

    if verbose:
        model.print_params()
        print "Best F0 : "+str(max_value2)
        print "Best thresh : "+str(max_thresh2)

    return max_thresh2, max_value2

def get_best_eval_metrics(data,model,save_path,chunks=None,verbose=False,with_dict=False,pred_values=None,expected_measures=False,logits=False,trim_outputs=True):

    if pred_values is None:
        sess, saver = model.load(save_path,None)
    else:
        #Not needed in this case
        sess, saver = None, None

    dataset, lengths, outputs, key_masks, key_lists = get_dataset(data,'valid',chunks,pred_values,trim_outputs)

    thresh,_ = get_best_thresh(dataset,lengths,key_masks,key_lists,model,save_path,verbose,sess,saver,pred_values=outputs,logits=logits,trim_outputs=trim_outputs)
    # thresh = 0.5

    dataset, lengths, outputs, key_masks, key_lists  = get_dataset(data,'test',chunks,pred_values,trim_outputs)

    if pred_values is None:
        results = model.compute_eval_metrics_pred(dataset,lengths,key_masks,key_lists,threshold=thresh,save_path=save_path,sess=sess,saver=saver)
    else:
        results = model.compute_eval_metrics_from_outputs(dataset,outputs,lengths,key_masks,key_lists,thresh,expected_measures,logits=logits,trim_outputs=trim_outputs)

    if verbose :
        output = ''
        for string,value in zip(["F: ",", P: ",", R: ",", XE: ",", XE_tr: ",", XE_st: ",", XE_l: ",", XE_a: ",", XE_k: ",", S: "],results):
            output += string
            output += str(value)
        print output

    if with_dict:
        res_dict = {}
        for pr in data.test:
            name = pr.name
            dataset,lengths = pr.get_roll()
            dataset, lengths = np.array([dataset]), np.array([lengths])
            key_masks = np.array([pr.get_key_profile_matrix()])
            key_lists = np.array([[time for (key,time) in pr.key_list]+[pr.length]])
            if pred_values is None:
                result = model.compute_eval_metrics_pred(dataset,lengths,key_masks,key_lists,threshold=thresh,save_path=save_path,sess=sess,saver=saver)
            else:
                pred_padded = np.zeros([1,dataset.shape[1],dataset.shape[2]])
                length = pred_values['test'][name].shape[0]
                pred_padded[0,:,:length] = np.transpose(pred_values['test'][name])
                pred_value = np.array([np.transpose(pred_values['test'][name])])
                result = model.compute_eval_metrics_from_outputs(dataset,pred_value,lengths,key_masks,key_lists,thresh,expected_measures,logits=logits,trim_outputs=trim_outputs)
            res_dict[name]=result

        return results, res_dict

    else:
        return results

def get_outputs(data,model,save_path,sigmoid=True):

    outputs = {}

    sess, saver = model.load(save_path,None)

    for subset in ['valid','test']:
        outputs[subset]={}
        for piano_roll in getattr(data,subset):
            roll, length = piano_roll.get_roll()
            roll = np.asarray([roll])
            length = [length]
            note_range = piano_roll.note_range
            name = piano_roll.name

            pred = model.run_prediction(roll,length,save_path,sigmoid=sigmoid,sess=sess,saver=saver)
            pred = pred[0]

            outputs[subset][name]=pred

    return outputs

def get_dataset(data,subset,chunks=None,pred_values=None,trim_outputs=True):

    if pred_values is None:
        outputs = None
        if chunks :
            dataset, lengths = data.get_dataset_chunks_no_pad(subset,chunks)
        else :
            dataset, lengths, key_masks, key_lists = data.get_dataset(subset,with_key_masks=True)
    else:
        assert chunks is None

        dataset, lengths, names, key_masks, key_lists = data.get_dataset(subset,with_names=True,with_key_masks=True)
        outputs_dict = pred_values[subset]
        if trim_outputs:
            outputs = np.zeros_like(dataset)
        else:
            outputs = np.zeros([dataset.shape[0],dataset.shape[1],dataset.shape[2]-1])
        for i,name in enumerate(names):
            try:
                output = outputs_dict[name]
                length = output.shape[0]
                # print output.shape, outputs[i,:,:].shape, outputs[i,:length,:].shape, dataset.shape
                outputs[i,:,:] = np.transpose(output,[1,0])
            except KeyError:
                print "Key not found", name
    return dataset, lengths, outputs, key_masks, key_lists

def get_notes_intervals(data,fs):
    #Returns the list of note events from a piano-roll

    data_extended = np.pad(data,((0,0),(1,1)),'constant')
    diff = data_extended[:,1:] - data_extended[:,:-1]

    #Onset: when a new note activates (doesn't count repeated notes)
    onsets= np.where(diff==1)
    #Onset: when a new note deactivates (doesn't count repeated notes)
    offsets= np.where(diff==-1)

    assert onsets[0].shape == offsets[0].shape
    assert onsets[1].shape == offsets[1].shape

    pitches = []
    intervals = []
    for [pitch1,onset], [pitch2,offset] in zip(zip(onsets[0],onsets[1]),zip(offsets[0],offsets[1])):
        # print pitch1, pitch2
        # print onset, offset
        assert pitch1 == pitch2
        pitches += [pitch1+1]
        if fs is None:
            intervals += [[onset, offset]]
        else:
            intervals += [[onset/float(fs), offset/float(fs)]]
        # print pitches
        # print intervals
    return np.array(pitches), np.array(intervals)
