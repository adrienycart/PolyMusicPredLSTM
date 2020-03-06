from dataset import Dataset, ground_truth, safe_mkdir
import os
import tensorflow as tf
import numpy as np
import scipy as scp
from datetime import datetime


class ModLSTMCell(tf.contrib.rnn.RNNCell):
    """Modified LSTM Cell (directly copied from https://github.com/ycemsubakan/diagonal_rnns) """

    def __init__(self, num_units, initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32), wform = 'diagonal'):
        self._num_units = num_units
        self.init = initializer
        self.wform = wform

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"

            c, h = state
            init = self.init
            self.L1 = inputs.get_shape().as_list()[1]

            mats, biases = self.get_params_parallel()
            if self.wform == 'full' or self.wform == 'diag_to_full':

                res = tf.matmul(tf.concat([h,inputs],axis=1),mats)
                res_wbiases = tf.nn.bias_add(res, biases)

                i,j,f,o = tf.split(res_wbiases,num_or_size_splits=4,axis=1)
            elif self.wform == 'diagonal':
                h_concat = tf.concat([h,h,h,h],axis=1)

                W_res = tf.multiply(h_concat,mats[0])

                U_res = tf.matmul(inputs,mats[1])

                res = tf.add(W_res,U_res)
                res_wbiases = tf.nn.bias_add(res, biases)

                i,j,f,o = tf.split(res_wbiases,num_or_size_splits=4,axis=1)

            elif self.wform == 'constant':

                h_concat = tf.concat([h,h,h,h],axis=1)

                U_res = tf.matmul(inputs,mats)

                res = tf.add(h_concat,U_res)
                res_wbiases = tf.nn.bias_add(res, biases)

                i,j,f,o = tf.split(res_wbiases,num_or_size_splits=4,axis=1)


            new_c = (c * tf.nn.sigmoid(f) + tf.nn.sigmoid(i)*tf.nn.tanh(j))

            new_h = tf.nn.tanh(new_c) * tf.nn.sigmoid(o)
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        return new_h, new_state


    def get_params_parallel(self):
        if self.wform == 'full':
            mats = tf.get_variable("mats",
                    shape = [self._num_units+self.L1,self._num_units*4],
                    initializer = self.init )
            biases = tf.get_variable("biases",
                    shape = [self._num_units*4],
                    initializer = self.init )
        elif self.wform == 'diagonal':
            Ws = tf.get_variable("Ws",
                    shape = [1,self._num_units*4],
                    initializer = self.init )
            Umats = tf.get_variable("Umats",
                    shape = [self.L1,self._num_units*4],
                    initializer = self.init )
            biases = tf.get_variable("biases",
                    shape = [self._num_units*4],
                    initializer = self.init )
            mats = [Ws, Umats]
        elif self.wform == 'constant':
            mats = tf.get_variable("mats",
                    shape = [self.L1,self._num_units*4],
                    initializer = self.init )
            biases = tf.get_variable("biases",
                    shape = [self._num_units*4],
                    initializer = self.init )
        elif self.wform == 'diag_to_full':
            #get the current variable scope
            var_scope = tf.get_variable_scope().name.replace('rnn2/','')

            #first filtering
            vars_to_use = [var for var in self.init if var_scope in var[0]]

            #next, assign the variables
            for var in vars_to_use:
                if '/Ws' in var[0]:
                    Ws = np.split(var[1], indices_or_sections = 4, axis = 1)
                    diag_Ws = np.concatenate([np.diag(w.squeeze()) for w in Ws], axis = 1)

                elif '/Umats' in var[0]:
                    Us = var[1]

                elif '/biases' in var[0]:
                    biases_np = var[1]

            mats_np = np.concatenate([diag_Ws, Us], axis = 0)
            mats_init = tf.constant_initializer(mats_np)

            mats = tf.get_variable("mats",
                    shape = [self._num_units+self.L1,self._num_units*4],
                    initializer = mats_init )

            biases_init = tf.constant_initializer(biases_np)
            biases = tf.get_variable("biases",
                    shape = [self._num_units*4],
                    initializer = biases_init )




        return mats, biases



class Model:

    def __init__(self, model_param):
        tf.reset_default_graph()

        #Unpack parameters
        for key,value in model_param.iteritems():
            setattr(self,key,value)

        self._inputs = None
        self._seq_lens = None
        self._labels = None
        self._key_masks = None
        self._key_masks_w = None
        self._weight_mat_trss = None
        self._weight_mat_k = None
        self._key_lists = None
        self._thresh = None
        self._thresh_key = None
        self._thresh_active = None

        self._prediction = None
        self._pred_sigm = None
        self._pred_thresh = None
        self._cross_entropy = None
        self._cross_entropy_list = None
        self._cross_entropy_transition = None
        self._cross_entropy_transition_list = None
        self._cross_entropy_length = None
        self._cross_entropy_length_list = None
        self._cross_entropy_steady = None
        self._cross_entropy_steady_list = None
        self._cross_entropy_active = None
        self._cross_entropy_key = None
        self._cross_entropy_key_list = None

        self._combined_metric = None
        self._combined_metric_list = None
        self._combined_metric_norm = None
        self._combined_metric_norm_list = None
        self._combined_metric_cw = None
        self._combined_metric_cw_list = None

        self._loss = None
        self._optimize = None
        self._tp = None
        self._fp = None
        self._fn = None
        self._precision = None
        self._recall = None
        self._f_measure = None
        self._enqueue_op = None


        self._fake_inputs = None
        self._fake_seq_lens = None
        self._fake_labels = None
        self._classif_loss = None
        self._classif_logits = None
        self._classif_accuracy = None


        #Call to create the graph
        self.cross_entropy


    def _transpose_data(self, data):
        return np.transpose(data,[0,2,1])

    def print_params(self):
        print "Learning rate : ",self.learning_rate
        print "Hidden nodes : ",self.n_hidden
        if not type(self.n_hidden)==int:
            print "Activation function : ",self.activ
        if self.chunks:
            print "Chunks : ",self.chunks
        if self.memory:
            print "With memory"



    @property
    def tp(self):
        if self._tp is None:
            with tf.device(self.device_name):
                pred = self.pred_thresh

                y = self.labels

                if self.non_binary:
                    bool_matrix = tf.logical_and(tf.greater_equal(pred,1),tf.greater_equal(y,1))
                else:
                    bool_matrix = tf.logical_and(tf.equal(pred,1),tf.equal(y,1))
                reduced = tf.reduce_sum(tf.cast(bool_matrix,tf.float32),[1,2])
                self._tp = reduced
        return self._tp

    @property
    def fp(self):
        if self._fp is None:
            with tf.device(self.device_name):
                pred = self.pred_thresh

                y = self.labels
                if self.non_binary:
                    bool_matrix = tf.logical_and(tf.greater_equal(pred,1),tf.equal(y,0))
                else:
                    bool_matrix = tf.logical_and(tf.equal(pred,1),tf.equal(y,0))
                reduced = tf.reduce_sum(tf.cast(bool_matrix,tf.float32),[1,2])
                self._fp = reduced
        return self._fp

    @property
    def fn(self):
        if self._fn is None:
            with tf.device(self.device_name):
                pred = self.pred_thresh

                y = self.labels
                if self.non_binary:
                    bool_matrix = tf.logical_and(tf.equal(pred,0),tf.greater_equal(y,0))
                else:
                    bool_matrix = tf.logical_and(tf.equal(pred,0),tf.equal(y,1))
                reduced = tf.reduce_sum(tf.cast(bool_matrix,tf.float32),[1,2])
                self._fn = reduced
        return self._fn

    @property
    def precision(self):
        #Returns a vector of length len(dataset), mean has to be computed afterwards
        if self._precision is None:
            with tf.device(self.device_name):
                TP = self.tp
                FP = self.fp
                self._precision = tf.truediv(TP,tf.add(tf.add(TP,FP),1e-6))
        return self._precision


    @property
    def recall(self):
        #Returns a vector of length len(dataset), mean has to be computed afterwards
        if self._recall is None:
            with tf.device(self.device_name):
                TP = self.tp
                FN = self.fn
                self._recall = tf.truediv(TP,tf.add(tf.add(TP,FN),1e-6))
        return self._recall

    @property
    def f_measure(self):
        #Returns a vector of length len(dataset), mean has to be computed afterwards
        if self._f_measure is None:
            with tf.device(self.device_name):
                prec = self.precision
                rec = self.recall
                self._f_measure = tf.truediv(tf.scalar_mul(2,tf.multiply(prec,rec)),tf.add(tf.add(prec,rec),1e-6))
        return self._f_measure


    @property
    def inputs(self):
        if self._inputs is None:
            n_notes = self.n_notes
            n_steps = self.n_steps
            suffix = self.suffix

            x = tf.placeholder("float", [None,n_steps,n_notes],name="x"+suffix)

            self._inputs = x
        return self._inputs

    @property
    def seq_lens(self):
        if self._seq_lens is None:
            suffix = self.suffix
            seq_len = tf.placeholder("int32",[None], name="seq_len"+suffix)

            self._seq_lens = seq_len
        return self._seq_lens

    @property
    def key_lists(self):
        if self._key_lists is None:
            suffix = self.suffix
            #1st dim: batch size, 2nd dim: max list length
            key_lists = tf.placeholder("int32",[None,None], name="key_lists"+suffix)

            self._key_lists = key_lists
        return self._key_lists

    @property
    def key_masks(self):
        if self._key_masks is None:
            suffix = self.suffix
            n_steps = self.n_steps
            n_notes = self.n_notes
            key_masks = tf.placeholder('float',[None,n_steps-1,n_notes],name="key_masks"+suffix)

            self._key_masks = key_masks
        return self._key_masks

    @property
    def key_masks_w(self):
        if self._key_masks is None:
            suffix = self.suffix
            y = self.labels
            n_steps = self.n_steps
            key_masks = tf.placeholder('float',[None,n_steps-1,12],name="key_masks_w"+suffix)

            self._key_masks_w = key_masks
        return self._key_masks_w

    @property
    def weight_mat_trss(self):
        if self._weight_mat_trss is None:
            suffix = self.suffix
            n_steps = self.n_steps
            n_notes = self.n_notes
            weight_mat_trss = tf.placeholder('float',[None,n_steps-1,n_notes],name="weight_mat_trss"+suffix)

            self._weight_mat_trss = weight_mat_trss
        return self._weight_mat_trss

    @property
    def weight_mat_k(self):
        if self._weight_mat_k is None:
            suffix = self.suffix
            n_steps = self.n_steps
            n_notes = self.n_notes
            weight_mat_k = tf.placeholder('float',[None,n_steps-1,n_notes],name="weight_mat_k"+suffix)

            self._weight_mat_k= weight_mat_k
        return self._weight_mat_k



    @property
    def prediction(self):
        if self._prediction is None:
            with tf.device(self.device_name):
                chunks = self.chunks
                memory = self.memory
                n_notes = self.n_notes
                activ = self.activ
                n_classes = n_notes
                if chunks:
                    n_steps = chunks
                else :
                    n_steps = self.n_steps
                n_hidden = self.n_hidden
                suffix = self.suffix


                x = self.inputs
                seq_len = self.seq_lens
                dropout = tf.placeholder_with_default(1.0, shape=(), name="dropout"+suffix)


                if self.non_binary:

                    n_outputs = self.non_binary
                    x_expanded = tf.one_hot(tf.cast(x,tf.int32),depth=n_outputs,dtype=tf.float32)
                    x_flat = tf.reshape(x,[-1,n_steps,n_classes*n_outputs])

                    W = tf.Variable(tf.truncated_normal([n_hidden,n_classes*n_outputs]),name="W"+suffix)
                    b = tf.Variable(tf.truncated_normal([n_classes*n_outputs]),name="b"+suffix)

                    if self.cell_type == "LSTM":
                        cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple=True,forget_bias = 1.0)
                    elif self.cell_type == "diagLSTM":
                        cell = ModLSTMCell(n_hidden,tf.truncated_normal_initializer())

                    outputs, _ = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32,time_major=False)#,sequence_length=seq_len)


                    outputs = tf.reshape(outputs,[-1,n_hidden])
                    pred = tf.matmul(outputs,W) + b
                    pred = tf.reshape(pred,[-1,n_steps,n_notes,n_outputs])
                    #drop last prediction of each sequence (you don't have ground truth for this one)
                    pred = pred[:,:n_steps-1,:,:]


                else:

                    W = tf.Variable(tf.truncated_normal([n_hidden,n_classes]),name="W"+suffix)
                    b = tf.Variable(tf.truncated_normal([n_classes]),name="b"+suffix)

                    if self.cell_type == "LSTM":
                        cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple=True,forget_bias = 1.0)
                    elif self.cell_type == "diagLSTM":
                        cell = ModLSTMCell(n_hidden,tf.truncated_normal_initializer())

                    outputs, _ = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32,time_major=False)#,sequence_length=seq_len)


                    outputs = tf.reshape(outputs,[-1,n_hidden])
                    pred = tf.matmul(outputs,W) + b
                    pred = tf.reshape(pred,[-1,n_steps,n_notes])
                    #drop last prediction of each sequence (you don't have ground truth for this one)
                    pred = pred[:,:n_steps-1,:]


                self._prediction = pred
        return self._prediction


    @property
    def pred_sigm(self):
        if self._pred_sigm is None:
            with tf.device(self.device_name):
                pred = self.prediction
                if self.non_binary:
                    pred = tf.nn.softmax(pred)
                else:
                    pred = tf.sigmoid(pred)
                self._pred_sigm = pred
        return self._pred_sigm

    @property
    def thresh(self):
        if self._thresh is None:
            suffix = self.suffix
            thresh = tf.placeholder_with_default(0.5,shape=[],name="thresh"+suffix)
            self._thresh = thresh
        return self._thresh

    @property
    def pred_thresh(self):
        if self._pred_thresh is None:
            with tf.device(self.device_name):
                if self.non_binary:
                    pred = tf.argmax(self.prediction,axis=3)
                else:
                    thresh = self.thresh
                    pred = self.pred_sigm
                    pred = tf.greater(pred,thresh)
                    pred = tf.cast(pred,tf.int8)
                self._pred_thresh = pred
        return self._pred_thresh



    @property
    def labels(self):
        if self._labels is None:
            n_notes = self.n_notes
            n_steps = self.n_steps
            suffix = self.suffix
            chunks = self.chunks

            if chunks:
                y = tf.placeholder("float", [None,chunks-1,n_notes],name="y"+suffix)
            else :
                y = tf.placeholder("float", [None,n_steps-1,n_notes],name="y"+suffix)

            self._labels = y
        return self._labels


    @property
    def cross_entropy(self):
        #Mean cross entropy
        if self._cross_entropy is None:
            with tf.device(self.device_name):
                cross_entropy = self.cross_entropy_list
                cross_entropy = tf.reduce_mean(cross_entropy)
                self._cross_entropy = cross_entropy
        return self._cross_entropy


    @property
    def cross_entropy_list(self):
        #Cross entropy as a vector of length batch_size
        if self._cross_entropy_list is None:
            with tf.device(self.device_name):
                n_notes = self.n_notes
                n_steps = self.n_steps
                suffix = self.suffix
                y = self.labels

                if self.non_binary:
                    cross_entropy_list = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction, labels=tf.cast(y,tf.int32)),axis=[1,2])
                else:
                    cross_entropy_list = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=y),axis=[1,2])
                self._cross_entropy_list = cross_entropy_list
        return self._cross_entropy_list


    def split_steady(self, x,*args):
        #x is one single input, same for all tensors in args
        data_extended = tf.pad(x,[[1,1],[0,0]],'constant')
        diff = data_extended[1:,:] - data_extended[:-1,:]
        # diff = tf.split(diff,tf.shape(diff)[0])
        trans_mask = tf.logical_or(tf.equal(diff,1), tf.equal(diff,-1))
        steady = tf.where(tf.logical_not(trans_mask))
        # lol = transitions[:,1,:]

        steady_unique, _ ,count_steady = tf.unique_with_counts(steady[:,0])
        steady_unique = tf.where(tf.equal(count_steady,self.n_notes))[:,0]
        steady_unique_steps = tf.gather(steady_unique,tf.where(tf.logical_and(tf.logical_not(tf.equal(steady_unique,0)),tf.logical_not(tf.equal(steady_unique,tf.cast(tf.shape(x)[0],tf.int64)))))[:,0])
        # steady_unique = steady_unique[:-1]
        # steady_unique_steps = tf.Print(steady_unique_steps,[seq_len,tf.shape(x)[0]])
        out = []
        for tensor in args:
            out += [tf.gather(tensor,tf.add(steady_unique_steps,-1))]
        out += [count_steady,tf.add(steady_unique_steps,-1)]

        return out


    def split_trans(self, x,*args):
        #x is one single input, same for all tensors in args
        data_extended = tf.pad(x,[[1,1],[0,0]],'constant')
        diff = data_extended[1:,:] - data_extended[:-1,:]
        # diff = tf.split(diff,tf.shape(diff)[0])
        trans_mask = tf.logical_or(tf.equal(diff,1), tf.equal(diff,-1))
        transitions= tf.where(trans_mask)

        transitions_unique, _ ,count_trans = tf.unique_with_counts(transitions[:,0])
        #We drop the first onset only if it is 0
        idx_to_keep = tf.where(tf.logical_and(tf.logical_not(tf.equal(transitions_unique,0)),tf.logical_not(tf.equal(transitions_unique,tf.cast(tf.shape(x)[0],tf.int64)))))[:,0]
        transitions_unique_trim = tf.gather(transitions_unique,idx_to_keep)
        count_trans_trim = tf.gather(count_trans,idx_to_keep)

        out = []

        # pred_trans = tf.gather(pred,tf.add(transitions_unique,-1))
        # y_trans = tf.gather(y,tf.add(transitions_unique,-1))
        for tensor in args:
            out += [tf.gather(tensor,tf.add(transitions_unique_trim,-1))]
        out += [count_trans_trim,tf.add(transitions_unique_trim,-1)]

        #return pred_trans, y_trans, count_trans
        return out

    def get_cross_entropy_transition(self,xs,pred,ys,seq_lens):

        def compute_one(elems):
            x = elems[0]
            y = elems[1]
            pred = elems[2]
            seq_len = elems[3]

            x = x[:seq_len,:]
            y = y[:seq_len-1,:]
            pred = pred[:seq_len-1,:]

            y, pred, count, _ = self.split_trans(x,y,pred)


            cross_entropy_trans = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y)
            cross_entropy_trans = tf.reduce_mean(cross_entropy_trans,axis=1)
            if self.normalise_XEtr:
                cross_entropy_trans = tf.reduce_mean(tf.div(cross_entropy_trans,tf.cast(count,tf.float32)))
            else:
                cross_entropy_trans = tf.reduce_mean(cross_entropy_trans)

            #It is necessary that the output has the same dimensions as input (even if not used)
            return cross_entropy_trans, tf.cast(tf.shape(pred),tf.float32), 0.0,0


        XEs = tf.map_fn(compute_one,[xs,ys,pred,seq_lens],dtype=(tf.float32,tf.float32,tf.float32,tf.int32))
        cross_entropy_trans = XEs[0]
        # cross_entropy_trans = tf.Print(cross_entropy_trans,[tf.where(tf.is_nan(cross_entropy_trans)),XEs[1]],message="trans",summarize=1000000)
        # pred_test, y_test , count_test, _  = self.split_trans(xs[0],ys[0],pred[0])
        # test1 = tf.identity([pred_test,y_test],name='test1')
        return cross_entropy_trans

    @property
    def cross_entropy_transition_list(self):
        if self._cross_entropy_transition_list is None:
            with tf.device(self.device_name):

                cross_entropy_trans = self.get_cross_entropy_transition(self.inputs,self.prediction,self.labels,self.seq_lens)

                self._cross_entropy_transition_list = cross_entropy_trans
        return self._cross_entropy_transition_list

    @property
    def cross_entropy_transition(self):
        if self._cross_entropy_transition is None:
            with tf.device(self.device_name):

                XEs = self.cross_entropy_transition_list
                XEs = tf.gather(XEs,tf.where(tf.logical_not(tf.is_nan(XEs))))
                cross_entropy_trans = tf.reduce_mean(XEs)

                self._cross_entropy_transition = cross_entropy_trans
        return self._cross_entropy_transition


    def get_cross_entropy_steady(self,xs,pred,ys,seq_lens):
        def compute_one(elems):
            x = elems[0]
            y = elems[1]
            pred = elems[2]
            seq_len = elems[3]

            x = x[:seq_len,:]
            y = y[:seq_len-1,:]
            pred = pred[:seq_len-1,:]

            y, pred, _ ,_ = self.split_steady(x,y,pred)

            cross_entropy_steady = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y)
            cross_entropy_steady = tf.reduce_mean(cross_entropy_steady)
            # cross_entropy_steady = tf.Print(tf.reduce_mean(cross_entropy_steady),[tf.shape(cross_entropy_steady),tf.reduce_sum(cross_entropy_steady),tf.reduce_mean(cross_entropy_steady)],message="steady")
            # output = tf.cond(tf.equal(tf.reduce_sum(cross_entropy_steady),0),
            #         fn1 = lambda: tf.Print(0.0,[0],message='steady zero'),
            #         fn2 = lambda: tf.Print(cross_entropy_steady,[tf.shape(cross_entropy_steady),tf.reduce_sum(cross_entropy_steady),tf.reduce_mean(cross_entropy_steady)],message="steady"))


            #It is necessary that the output has the same dimensions as input (even if not used)
            return cross_entropy_steady, tf.cast(tf.shape(pred),tf.float32), 0.0, 0


        XEs = tf.map_fn(compute_one,[xs,ys,pred,seq_lens],dtype=(tf.float32,tf.float32,tf.float32,tf.int32))
        cross_entropy_steady = XEs[0]
        # cross_entropy_steady = tf.Print(cross_entropy_steady,[tf.where(tf.is_nan(cross_entropy_steady)),XEs[1]],message="steady",summarize=1000000)
        return cross_entropy_steady


    @property
    def cross_entropy_steady_list(self):
        if self._cross_entropy_steady_list is None:
            with tf.device(self.device_name):

                cross_entropy_steady = self.get_cross_entropy_steady(self.inputs,self.prediction,self.labels,self.seq_lens)

                self._cross_entropy_steady_list = cross_entropy_steady
        return self._cross_entropy_steady_list

    @property
    def cross_entropy_steady(self):
        if self._cross_entropy_steady is None:
            with tf.device(self.device_name):


                XEs = self.cross_entropy_steady_list
                XEs_no_nan = tf.gather(XEs,tf.where(tf.logical_not(tf.is_nan(XEs))))
                cross_entropy_steady = tf.reduce_mean(XEs_no_nan)


                self._cross_entropy_steady = cross_entropy_steady
        return self._cross_entropy_steady

    @property
    def cross_entropy_length_list(self):
        if self._cross_entropy_length_list is None:
            with tf.device(self.device_name):
                n_notes = self.n_notes
                n_steps = self.n_steps
                suffix = self.suffix
                y = self.labels
                seq_len = self.seq_lens

                mask = tf.sequence_mask(seq_len-1,maxlen=n_steps-1)
                mask = tf.expand_dims(mask,-1)
                mask = tf.tile(mask,[1,1,n_notes])

                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=y)
                cross_entropy_masked = cross_entropy*tf.cast(mask,tf.float32)

                cross_entropy_length = tf.reduce_mean(cross_entropy_masked*n_steps,axis=[1,2])/tf.cast(seq_len,tf.float32)

                self._cross_entropy_length_list = cross_entropy_length
        return self._cross_entropy_length_list

    @property
    def cross_entropy_length(self):
        if self._cross_entropy_length is None:
            with tf.device(self.device_name):
                cross_entropy_length = self.cross_entropy_length_list
                cross_entropy_length = tf.reduce_mean(cross_entropy_length)

                self._cross_entropy_length = cross_entropy_length
        return self._cross_entropy_length

    @property
    def cross_entropy_active(self):
        if self._cross_entropy_active is None:
            with tf.device(self.device_name):

                x = self.inputs
                y = self.labels
                pred = self.prediction

                cross_entropy_masked = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y)*tf.cast(y,tf.float32)
                cross_entropy_active = tf.reduce_sum(cross_entropy_masked,axis=[1,2])
                sum_active = tf.reduce_sum(y,axis=[1,2])
                cross_entropy_active = tf.reduce_mean(cross_entropy_active/tf.cast(sum_active,tf.float32))

                self._cross_entropy_active = cross_entropy_active
        return self._cross_entropy_active


    @property
    def thresh_key(self):
        if self._thresh_key is None:
            with tf.device(self.device_name):
                thresh_key = tf.placeholder_with_default(0.05, shape=(), name="thresh_key"+self.suffix)
                self._thresh_key = thresh_key
        return self._thresh_key

    @property
    def thresh_active(self):
        if self._thresh_active is None:
            with tf.device(self.device_name):
                thresh_active = tf.placeholder_with_default(0.05, shape=(), name="thresh_active"+self.suffix)
                self._thresh_active = thresh_active
        return self._thresh_active


    def get_cross_entropy_key(self,x,pred,y,seq_lens):
        ### Out of all these variants, we only end up using:
        ##  key_XE, key_XE_tr and key_XE_ss (indices 0, 2 and 3)


        def octave_wrap(tensor):
            #Tensor has to be of shape [None, n_steps,n_notes]
            tensor_pad = tf.pad(tensor,((0,0),(0,0),(0,12-(n_notes%12))))
            tensor_reshape = tf.reshape(tensor_pad,[tf.shape(tensor_pad)[0],n_steps-1,12,-1])
            tensor_wrap = tf.reduce_sum(tensor_reshape,axis=[-1])
            return tensor_wrap

        def normalise(tensor):
            sum_tensor = tf.add(tf.reduce_sum(tensor,axis=[-1]),1e-7)
            sum_tensor = tf.tile(tf.expand_dims(sum_tensor,[-1]),[1,1,tf.shape(tensor)[-1]])
            tensor_normalised = tf.div(tensor,sum_tensor)
            return tensor_normalised

        def logit(tensor):
            return tf.log(tensor+1e-7) - tf.log(1-tensor+1e-7)

        def average_one(elems):
            tensor,active_mask,key_mask,key_list = elems
            def compute(x,a,k,l,out):
                n_steps = l[0]
                x_split = tf.cast(x[:n_steps,:],tf.float32)
                key_split = tf.cast(k[:n_steps,:],tf.float32)
                active_split = tf.cast(a[:n_steps,:],tf.float32)

                x_avg = logit(tf.div(tf.reduce_sum(tf.sigmoid(x_split)*active_split,axis=[0]),tf.reduce_sum(active_split,axis=[0])+1e-7))
                # x_avg_norm = normalise(x_avg)
                key_avg = tf.reduce_mean(key_split,axis=[0])
                # out = tf.Print(tf.concat([out,tf.expand_dims(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_avg,labels=key_avg),axis=[0])],axis=0),["x_avg",tf.where(tf.is_nan(x_avg)),tf.reduce_sum(tf.sigmoid(x_split),axis=[0])[41],(tf.reduce_sum(active_split,axis=[0])+1e-7)[41],tf.shape(x_split)])
                out = tf.concat([out,tf.expand_dims(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_avg,labels=key_avg),axis=[0])],axis=0)
                x = x[n_steps:]
                k = k[n_steps:]
                a = a[n_steps:]
                l = l[1:]
                return x,a,k,l,out

            key_list = tf.concat([key_list,key_list[-1:]],axis=0)
            split_list = key_list[1:]-key_list[:-1]
            n_notes=self.n_notes

            _,_,_,_,out = tf.while_loop(lambda x,a,k,l,out:l[0]>0,
                                compute,
                                [tensor,active_mask,key_mask,split_list,tf.zeros([0,self.n_notes],dtype=tf.float32)],
                                shape_invariants=[tf.TensorShape([None,n_notes]),
                                                  tf.TensorShape([None,n_notes]),
                                                  tf.TensorShape([None,n_notes]),
                                                  tf.TensorShape([None]),
                                                  tf.TensorShape([None,n_notes])])

            split_list_non_zero = tf.cast(tf.boolean_mask(split_list,tf.greater(split_list,0)),tf.float32)
            out_w_avg = tf.reduce_sum(out*tf.tile(tf.expand_dims(split_list_non_zero,[-1]),[1,self.n_notes]))/(tf.reduce_sum(split_list_non_zero)*self.n_notes)
            # out = tf.Print(out_w_avg,["out",tf.where(tf.is_nan(out)),out_w_avg,tf.reduce_sum(out*tf.tile(tf.expand_dims(split_list_non_zero,[-1]),[1,self.n_notes])),split_list,split_list_non_zero],summarize=100000)
            return out_w_avg, 0.0, 0.0, 0.0

        def transitions_one(elems):
            tensor,x,active_mask,key_mask,seq_len = elems

            x = x[:seq_len,:]
            tensor = tensor[:seq_len-1,:]
            active_mask = active_mask[:seq_len-1,:]
            key_mask = key_mask[:seq_len-1,:]

            tensor,active_mask,key_mask,_,_  = self.split_trans(x,tensor,active_mask,key_mask)
            XE = tf.nn.sigmoid_cross_entropy_with_logits(logits=tensor, labels=key_mask)*active_mask
            XE = tf.reduce_sum(XE)/tf.reduce_sum(active_mask)
            return XE, 0.0,0.0,0.0,0

        def transitions_one_norm(elems):
            tensor,x,seq_len,active_mask,key_mask  = elems

            x = x[:seq_len,:]
            tensor = tensor[:seq_len-1,:]
            active_mask = active_mask[:seq_len-1,:]
            key_mask = key_mask[:seq_len-1,:]

            tensor_split,active_mask_split,key_mask_split,_, _  = self.split_trans(x,tensor,active_mask,key_mask)
            XE = tf.nn.sigmoid_cross_entropy_with_logits(logits=tensor_split, labels=key_mask_split)*active_mask_split
            norm_factor = tf.expand_dims(tf.reduce_sum(key_mask_split,axis=[1]),axis=1)
            XE = XE/norm_factor
            # XE = tf.where(tf.is_nan(XE), tf.zeros_like(XE), XE)
            XE= tf.reduce_sum(XE)/tf.reduce_sum(active_mask_split)
            # XE = tf.Print(XE,[tf.where(tf.equal(norm_factor,0)),tf.where(tf.equal(tf.reduce_sum(key_mask_split,axis=1),0)),tf.where(tf.equal(tf.reduce_sum(active_mask_split,axis=1),0)),tf.shape(norm_factor)],message="trans",summarize=1000)

            return XE, 0.0,0.0,0.0,0.0

        def steady_one(elems):
            tensor,x,active_mask,key_mask,seq_len = elems

            x = x[:seq_len,:]
            tensor = tensor[:seq_len-1,:]
            active_mask = active_mask[:seq_len-1,:]
            key_mask = key_mask[:seq_len-1,:]

            tensor,active_mask,key_mask,_,_  = self.split_steady(x,tensor,active_mask,key_mask)
            XE = tf.nn.sigmoid_cross_entropy_with_logits(logits=tensor, labels=key_mask)*active_mask
            XE = tf.reduce_sum(XE)/tf.reduce_sum(active_mask)

            return XE, 0.0,0.0,0.0,0

        def steady_one_norm(elems):
            tensor,x,seq_len,active_mask,key_mask = elems

            x = x[:seq_len,:]
            tensor = tensor[:seq_len-1,:]
            active_mask = active_mask[:seq_len-1,:]
            key_mask = key_mask[:seq_len-1,:]

            tensor_split,active_mask_split,key_mask_split,_,_  = self.split_steady(x,tensor,active_mask,key_mask)
            XE = tf.nn.sigmoid_cross_entropy_with_logits(logits=tensor_split, labels=key_mask_split)*active_mask_split
            norm_factor = tf.expand_dims(tf.reduce_sum(key_mask_split,axis=[1]),axis=1)
            XE = XE/norm_factor
            # XE = tf.where(tf.is_nan(XE), tf.zeros_like(XE), XE)
            XE= tf.reduce_sum(XE)/tf.reduce_sum(active_mask_split)
            # XE = tf.Print(XE,[tf.where(tf.equal(tf.reduce_sum(key_mask_split,axis=1),0)),tf.where(tf.equal(tf.reduce_sum(active_mask_split,axis=1),0)),tf.shape(norm_factor)],message="steady",summarize=1000)

            return XE, 0.0,0.0,0.0,0



        n_notes = self.n_notes
        n_steps = self.n_steps
        key_masks = self.key_masks
        thresh_key = self.thresh_key
        thresh_active = self.thresh_active

        key_lists = self.key_lists

        key_masks = tf.cast(tf.greater(key_masks,thresh_key),tf.float32)

        label_mask = tf.cast(tf.abs(1-y),tf.float32)
        active_mask = tf.cast(tf.greater(pred,thresh_active),tf.float32)
        # label_mask = label_mask*tf.cast(tf.abs(1-x[:,:-1,:]),tf.float32)
        length_mask = tf.sequence_mask(seq_lens-1,maxlen=n_steps-1)
        length_mask = tf.expand_dims(length_mask,-1)
        length_mask = tf.cast(tf.tile(length_mask,[1,1,n_notes]),tf.float32)
        XE_mask = label_mask*length_mask
        prop_mask = label_mask*length_mask*active_mask
        pred_masked = pred*XE_mask


        # #Octave_wrapped predictions
        # pred_w = octave_wrap(pred_masked)
        # mask_w = octave_wrap(mask)
        # key_masks_w = octave_wrap(key_masks)

        #Normalise
        # pred_norm = normalise(pred_masked)
        # pred_w_norm = normalise(pred_w)
        # key_masks_w = normalise(key_masks_w)


        key_XE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=key_masks)*XE_mask,axis=[1,2])/tf.reduce_sum(XE_mask,axis=[1,2])


        #Average cross-entropy by key segment
        output = tf.map_fn(average_one,(pred_masked,XE_mask,key_masks,key_lists),dtype=(tf.float32,tf.float32,tf.float32,tf.int32))
        key_XE_avg = output[0]

        #Key cross_entropy on transitions
        output = tf.map_fn(transitions_one,(pred_masked,x,XE_mask,key_masks,seq_lens),dtype=(tf.float32,tf.int32,tf.float32,tf.float32,tf.int32))
        key_XE_tr = output[0]

        #Binary evaluation of false positives
        #--> For each false positive, we want to check whether it is in-key or out-of-key
        binary_false_positives = prop_mask
        in_key_false_positives = key_masks * binary_false_positives
        in_key_prop = tf.reduce_sum(in_key_false_positives,axis=[1,2])/tf.reduce_sum(binary_false_positives,axis=[1,2])


        #Key cross_entropy on steady state
        output = tf.map_fn(steady_one,(pred_masked,x,XE_mask,key_masks,seq_lens),dtype=(tf.float32,tf.int32,tf.float32,tf.float32,tf.int32))
        key_XE_ss = output[0]


        #NORMALISED XE_k
        norm_factor = tf.expand_dims(tf.reduce_sum(key_masks,axis=[2]),axis=2)
        key_XE_n = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=key_masks)*XE_mask/norm_factor
        key_XE_n = tf.where(tf.is_nan(key_XE_n), tf.zeros_like(key_XE_n), key_XE_n)
        key_XE_n = tf.reduce_sum(key_XE_n,axis=[1,2])/tf.reduce_sum(XE_mask,axis=[1,2])

        #NORMALISED XE_k,tr
        output = tf.map_fn(transitions_one_norm,(pred_masked,x,seq_lens,XE_mask,key_masks),dtype=(tf.float32,tf.int32,tf.float32,tf.float32,tf.float32))
        key_XE_tr_n = output[0]


        #NORMALISED XE_k,ss
        output = tf.map_fn(steady_one_norm,(pred_masked,x,seq_lens,XE_mask,key_masks),dtype=(tf.float32,tf.int32,tf.float32,tf.float32,tf.float32))
        key_XE_ss_n = output[0]

        ## We only use: key_XE, key_XE_tr and key_XE_ss
        cross_entropy_key_masked = [key_XE,key_XE_avg,key_XE_tr,key_XE_ss,in_key_prop,key_XE_n,key_XE_tr_n,key_XE_ss_n]
        return cross_entropy_key_masked


    @property
    def cross_entropy_key_list(self):
        if self._cross_entropy_key_list is None:
            with tf.device(self.device_name):

                cross_entropy_key_masked = self.get_cross_entropy_key(self.inputs,self.prediction,self.labels,self.seq_lens)

                self._cross_entropy_key_list = cross_entropy_key_masked
        return self._cross_entropy_key_list

    @property
    def cross_entropy_key(self):
        if self._cross_entropy_key is None:
            with tf.device(self.device_name):
                ### Out of all these variants, we only end up using:
                ##  key_XE, key_XE_tr and key_XE_ss (indices 0, 2 and 3)

                key_XE,key_XE_avg,key_XE_tr,key_XE_ss,in_key_prop,key_XE_n,key_XE_tr_n,key_XE_ss_n = self.cross_entropy_key_list

                key_XE = tf.reduce_mean(key_XE)

                #Average cross-entropy by key segment
                key_XE_avg = tf.reduce_mean(tf.gather(key_XE_avg,tf.where(tf.logical_not(tf.is_nan(key_XE_avg)))))

                #Key cross_entropy on transitions
                key_XE_tr = tf.reduce_mean(tf.gather(key_XE_tr,tf.where(tf.logical_not(tf.is_nan(key_XE_tr)))))

                #Binary evaluation of false positives
                #--> For each false positive, we want to check whether it is in-key or out-of-key
                in_key_prop = tf.reduce_mean(in_key_prop)

                #Key cross_entropy on steady state
                key_XE_ss = tf.reduce_mean(tf.gather(key_XE_ss,tf.where(tf.logical_not(tf.is_nan(key_XE_ss)))))
                # key_XE_ss = tf.where(tf.is_nan(key_XE_ss), tf.zeros_like(key_XE_ss), key_XE_ss)


                #NORMALISED XE_k
                key_XE_n = tf.reduce_mean(key_XE_n)

                #NORMALISED XE_k,tr
                key_XE_tr_n = tf.reduce_mean(tf.gather(key_XE_tr_n,tf.where(tf.logical_not(tf.is_nan(key_XE_tr_n)))))

                #NORMALISED XE_k,ss
                key_XE_ss_n = tf.reduce_mean(tf.gather(key_XE_ss_n,tf.where(tf.logical_not(tf.is_nan(key_XE_ss_n)))))


                cross_entropy_key_masked = [key_XE,key_XE_avg,key_XE_tr,key_XE_ss,in_key_prop,key_XE_n,key_XE_tr_n,key_XE_ss_n]

                self._cross_entropy_key = cross_entropy_key_masked
        return self._cross_entropy_key



    @property
    def combined_metric_list(self):
        if self._combined_metric_list is None:
            with tf.device(self.device_name):
                XE_tr = self.cross_entropy_transition_list
                XE_ss = self.cross_entropy_steady_list
                XE_k = self.cross_entropy_key_list
                XE_ktr = XE_k[2]
                XE_kss = XE_k[3]

                w_tr_ss = self.weights_tr_ss
                w_tr = w_tr_ss[0]
                w_ss = w_tr_ss[1]

                alpha = self.alpha
                assert -1<=alpha<=1

                # no_nan_mask = tf.logical_or(tf.is_nan(XE_tr),tf.is_nan(XE_tr))
                # no_nan_mask = tf.logical_or(no_nan_mask,tf.is_nan(XE_ktr))
                # no_nan_mask = tf.logical_or(no_nan_mask,tf.is_nan(XE_kss))
                # no_nan_mask = tf.logical_not(no_nan_mask)
                #
                # XE_tr = tf.gather(XE_tr,tf.where(no_nan_mask))
                # XE_ss = tf.gather(XE_ss,tf.where(no_nan_mask))
                # XE_ktr = tf.gather(XE_ktr,tf.where(no_nan_mask))
                # XE_kss = tf.gather(XE_kss,tf.where(no_nan_mask))

                combined_metric = tf.sqrt(tf.pow((w_tr*XE_tr+w_ss*XE_ss),1+alpha)*tf.pow((w_tr*XE_ktr+w_ss*XE_kss),1-alpha))


                self._combined_metric_list = combined_metric
        return self._combined_metric_list

    @property
    def combined_metric(self):
        if self._combined_metric is None:
            with tf.device(self.device_name):
                combined_metric = self.combined_metric_list

                combined_metric = tf.gather(combined_metric,tf.where(tf.logical_not(tf.is_nan(combined_metric))))
                combined_metric = tf.reduce_mean(combined_metric)


                self._combined_metric = combined_metric
        return self._combined_metric

    @property
    def combined_metric_norm_list(self):
        if self._combined_metric_norm_list is None:
            with tf.device(self.device_name):
                XE_tr = self.cross_entropy_transition_list
                XE_ss = self.cross_entropy_steady_list
                XE_k = self.cross_entropy_key_list
                XE_ktr = XE_k[6]
                XE_kss = XE_k[7]

                w_tr_ss = self.weights_tr_ss
                w_tr = w_tr_ss[0]
                w_ss = w_tr_ss[1]



                combined_metric_norm = tf.sqrt((w_tr*XE_tr+w_ss*XE_ss)*(w_tr*XE_ktr+w_ss*XE_kss))


                self._combined_metric_norm_list = combined_metric_norm
        return self._combined_metric_norm_list

    @property
    def combined_metric_norm(self):
        if self._combined_metric_norm is None:
            with tf.device(self.device_name):
                combined_metric = self.combined_metric_norm_list

                combined_metric = tf.gather(combined_metric,tf.where(tf.logical_not(tf.is_nan(combined_metric))))
                combined_metric = tf.reduce_mean(combined_metric)


                self._combined_metric_norm = combined_metric
        return self._combined_metric_norm

    @property
    def combined_metric_cw_list(self):
        if self._combined_metric_cw_list is None:
            with tf.device(self.device_name):
                w_tr_ss = self.weights_tr_ss
                w_tr = w_tr_ss[0]
                w_ss = w_tr_ss[1]


                def compute_one(elems):
                    x = elems[0]
                    XE = elems[1]
                    XE_k = elems[2]
                    XE_mask = elems[3]
                    seq_len = elems[4]

                    x = x[:seq_len,:]
                    XE = XE[:seq_len-1,:]
                    XE_mask = XE_mask[:seq_len-1,:]
                    XE_k = XE_k[:seq_len-1,:]

                    _,indices_trans = self.split_trans(x)
                    _,indices_steady= self.split_steady(x)

                    trans_mask = tf.sparse_to_dense(indices_trans,output_shape=tf.cast(tf.shape(XE)[0:1],tf.int64),sparse_values=tf.cast(w_tr,tf.float32))
                    steady_mask = tf.sparse_to_dense(indices_steady,output_shape=tf.cast(tf.shape(XE)[0:1],tf.int64),sparse_values=tf.cast(w_ss,tf.float32))

                    trans_mask = tf.tile(tf.expand_dims(trans_mask,-1),[1,self.n_notes])
                    steady_mask = tf.tile(tf.expand_dims(steady_mask,-1),[1,self.n_notes])

                    XE_weighted = tf.reduce_mean((XE*trans_mask + XE*steady_mask)*self.n_steps)/tf.cast(seq_len,tf.float32)
                    XE_k_weighted = tf.reduce_sum((XE_k*trans_mask + XE_k*steady_mask)*XE_mask)/tf.reduce_sum(XE_mask)

                    combined = tf.sqrt(XE_weighted*XE_k_weighted)


                    #It is necessary that the output has the same dimensions as input (even if not used)
                    return combined, 0.0, 0.0, 0.0, 0



                n_notes = self.n_notes
                n_steps = self.n_steps
                key_masks = self.key_masks
                thresh_key = self.thresh_key

                x = self.inputs
                y = self.labels
                seq_lens = self.seq_lens
                pred = self.prediction


                # Preparing Masks
                key_masks = tf.cast(tf.greater(key_masks,thresh_key),tf.float32)

                label_mask = tf.cast(tf.abs(1-y),tf.float32)
                length_mask = tf.sequence_mask(seq_lens-1,maxlen=n_steps-1)
                length_mask = tf.expand_dims(length_mask,-1)
                length_mask = tf.cast(tf.tile(length_mask,[1,1,n_notes]),tf.float32)
                XE_mask = label_mask*length_mask

                # Getting XE matrices
                XE_matrix = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=y)
                XE_matrix = XE_matrix*tf.cast(length_mask,tf.float32)

                key_XE_matrix = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=key_masks)




                # Applying weights, and averaging
                combined_cw = tf.map_fn(compute_one,[x,XE_matrix,key_XE_matrix,XE_mask,seq_lens],dtype=(tf.float32,tf.float32,tf.float32,tf.float32,tf.int32))
                combined_cw = combined_cw[0]



                self._combined_metric_cw_list = combined_cw
        return self._combined_metric_cw_list

    @property
    def combined_metric_cw(self):
        if self._combined_metric_cw is None:
            with tf.device(self.device_name):
                combined_metric = self.combined_metric_cw_list

                combined_metric = tf.gather(combined_metric,tf.where(tf.logical_not(tf.is_nan(combined_metric))))
                combined_metric = tf.reduce_mean(combined_metric)


                self._combined_metric_cw = combined_metric
        return self._combined_metric_cw

    @property
    def loss(self):
        if self._loss is None:
            with tf.device(self.device_name):

                XE = self.cross_entropy

                if self.loss_type == 'XE':
                    loss = XE #+ 0.1*cross_entropy_trans
                elif self.loss_type == 'combined':
                    loss = self.combined_metric


                elif self.loss_type == 'combined_norm':
                    loss = self.combined_metric_norm

                elif self.loss_type == 'combined_cw':
                    loss = self.combined_metric_cw

                elif self.loss_type == "XEtr_XEss":
                    XE_tr = self.cross_entropy_transition_list
                    XE_ss = self.cross_entropy_steady_list

                    w_tr_ss = self.weights_tr_ss
                    w_tr = w_tr_ss[0]
                    w_ss = w_tr_ss[1]

                    no_nan_mask = tf.logical_or(tf.is_nan(XE_tr),tf.is_nan(XE_tr))
                    no_nan_mask = tf.logical_not(no_nan_mask)

                    XE_tr = tf.gather(XE_tr,tf.where(no_nan_mask))
                    XE_ss = tf.gather(XE_ss,tf.where(no_nan_mask))

                    loss = tf.reduce_mean(w_tr*XE_tr+w_ss*XE_ss)
                else:
                    raise ValueError("loss_type value not understood: "+str(self.loss_type) )



                if self.classif_metric_type is not None:
                    classif_loss = self.classif_loss

                    loss = loss + classif_loss


            self._loss = loss
        return self._loss


    @property
    def optimize(self):
        if self._optimize is None:
            with tf.device(self.device_name):


                loss = self.loss

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                gvs = optimizer.compute_gradients(loss)
                if self.grad_clip is not None:
                    capped_gvs = [(tf.clip_by_value(grad, -float(self.grad_clip), float(self.grad_clip)), var) for grad, var in gvs]
                    train_op = optimizer.apply_gradients(capped_gvs)
                else:
                    train_op = optimizer.apply_gradients(gvs)
                self._optimize = train_op
        return self._optimize

    def _run_by_batch(self,sess,op,feed_dict,batch_size,mean=True):
        ### WARNING: this does not work with any op argument

        suffix = self.suffix
        x = self.inputs
        try:
            y = self.labels
            seq_len = self.seq_lens
        except KeyError:
            n_steps = x.get_shape()[1]
            n_notes = x.get_shape()[2]
            y = tf.placeholder("float", [None,n_steps-1,n_notes],name="y"+suffix)

        if y in feed_dict:
            dataset = feed_dict[x]
            target = feed_dict[y]
            len_list = feed_dict[seq_len]
        else:
            dataset = feed_dict[x]

        no_of_batches = int(np.ceil(float(len(dataset))/batch_size))
        #crosses = np.zeros([dataset.shape[0]])
        #results = np.empty(dataset.shape)
        results = []
        ptr = 0
        for j in range(no_of_batches):
            if y in feed_dict:
                batch_x = dataset[ptr:ptr+batch_size]
                batch_y = target[ptr:ptr+batch_size]
                batch_len_list = len_list[ptr:ptr+batch_size]
                feed_dict={x: batch_x, y: batch_y,seq_len: batch_len_list}
            else :
                batch_x = dataset[ptr:ptr+batch_size]
                feed_dict={x: batch_x}
            ptr += batch_size
            result_batch = sess.run(op, feed_dict=feed_dict)
            results = np.append(results,result_batch)
        if mean:
            return np.mean(results)
        else :
            return results


    def extract_data(self,dataset,subset,with_keys=False):
        chunks = self.chunks

        if chunks:

            data_raw, lengths = dataset.get_dataset_chunks_no_pad(subset,chunks)
        else :

            if not with_keys:
                data_raw, lengths = dataset.get_dataset(subset)
            else:
                data_raw, lengths, names, key_masks, key_lists = dataset.get_dataset(subset,with_names=True,with_key_masks=True)


        data = self._transpose_data(data_raw)
        target = self._transpose_data(ground_truth(data_raw))

        output = [data,target,lengths]

        if with_keys:
            k_masks = self._transpose_data(key_masks)
            output += [names,k_masks,key_lists]

        return output

    def initialize_training(self,save_path,train_param,sess=None):
        #Unpack values
        for key,val in train_param.items():
            exec(key + '=val')


        optimizer = self.optimize #Create the optimizer variable so they can be initialised
        loss = self.loss
        cross_entropy = self.cross_entropy


        ckpt_save_path = os.path.join("./ckpt/",save_path)
        summ_save_path = os.path.join("./summ/",save_path)
        safe_mkdir(ckpt_save_path)

        init = tf.global_variables_initializer()
        if not sess:
            init = tf.global_variables_initializer()
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            sess.run(init)

        else:
            variables_to_initialize = sess.run(tf.report_uninitialized_variables())
            var_list = []
            for var in tf.global_variables():
                for var_to_init in variables_to_initialize:
                    if var_to_init in var.name:
                        var_list += [var]
            init = tf.variables_initializer(var_list)
            sess.run(init)

        if summarize:
            precision = self.precision
            recall = self.recall
            f_measure = self.f_measure

            safe_mkdir(summ_save_path,clean=True)
            tf.summary.scalar('loss epoch',loss,collections=['epoch'])
            tf.summary.scalar('cross entropy epoch',cross_entropy,collections=['epoch'])
            tf.summary.scalar('precision epoch',tf.reduce_mean(precision),collections=['epoch'])
            tf.summary.scalar('recall epoch',tf.reduce_mean(recall),collections=['epoch'])
            tf.summary.scalar('f_measure epoch',tf.reduce_mean(f_measure),collections=['epoch'])

            tf.summary.scalar('loss bath',loss,collections=['batch'])
            tf.summary.scalar('cross entropy batch',cross_entropy,collections=['batch'])
            tf.summary.scalar('precision batch',tf.reduce_mean(precision),collections=['batch'])
            tf.summary.scalar('recall batch',tf.reduce_mean(recall),collections=['batch'])
            tf.summary.scalar('f_measure batch',tf.reduce_mean(f_measure),collections=['batch'])

            summary_epoch = tf.summary.merge_all('epoch')
            summary_batch = tf.summary.merge_all('batch')
            train_writer = tf.summary.FileWriter(summ_save_path,
                                      sess.graph)


        if early_stop:
            saver = [tf.train.Saver(max_to_keep=max_to_keep), tf.train.Saver(max_to_keep=1)]
        else:
            saver = tf.train.Saver(max_to_keep=max_to_keep)

        outputs = [sess, saver, ckpt_save_path]
        if summarize:
            outputs += [train_writer, summary_batch, summary_epoch]

        return outputs


    def perform_training(self,data,save_path,train_param,sess,saver,train_writer,ckpt_save_path,summary_batch, summary_epoch,n_batch=0,n_epoch=0):
        #Unpack values
        for key,val in train_param.items():
            exec(key + '=val')


        chunks = self.chunks


        optimizer = self.optimize
        cross_entropy = self.cross_entropy
        cross_entropy_list= self.cross_entropy_list
        precision = self.precision
        recall = self.recall
        f_measure = self.f_measure
        suffix = self.suffix




        x = self.inputs
        y = self.labels
        seq_len = self.seq_lens

        drop = tf.get_default_graph().get_tensor_by_name("dropout"+suffix+":0")

        if self.loss_type in ["combined",'combined_norm','combined_cw',"XEtr_XEss"] or self.classif_metric_type in ["combined",'combined_norm',"XEtr_XEss"]:
            k_m = self.key_masks
            k_l = self.key_lists

        if self.classif_metric_type is not None:
            fake_x = self.fake_inputs
            fake_y = self.fake_labels
            fake_seq_len = self.fake_seq_lens

        print 'Starting computations : '+str(datetime.now())


        print "Total number of parameters:", getTotalNumParameters()
        if early_stop:
            best_cross = 100000
            epoch_since_best = 0
            saver_best = saver[1]
            saver = saver[0]

        i = n_epoch
        while i < n_epoch+epochs and epoch_since_best<early_stop_epochs:
            # print 1
            start_epoch = datetime.now()
            ptr = 0


            if self.loss_type == "XE":
                training_data, training_target, training_lengths = self.extract_data(data,'train')
                valid_data, valid_target, valid_lengths = self.extract_data(data,'valid')
            elif self.loss_type in ["combined",'combined_norm','combined_cw',"XEtr_XEss"]:
                training_data, training_target, training_lengths, training_names, training_k_masks, training_k_lists = self.extract_data(data,'train',with_keys=True)
                valid_data, valid_target, valid_lengths, valid_names, valid_k_masks, valid_k_lists = self.extract_data(data,'valid',with_keys=True)


            # n_files = training_data.shape[0]
            n_files = len(data.train)
            no_of_batches = int(np.ceil(float(n_files)/batch_size))

            display_step = max(int(round(float(no_of_batches)/display_per_epoch)),1)



            for j in range(no_of_batches):
                # print "batch",j

                batch_x = training_data[ptr:ptr+batch_size]
                batch_y = training_target[ptr:ptr+batch_size]
                batch_lens = training_lengths[ptr:ptr+batch_size]


                train_dict = {x: batch_x, y: batch_y, seq_len: batch_lens, drop: dropout}

                if self.loss_type in ["combined",'combined_norm','combined_cw',"XEtr_XEss"]:
                    batch_names = training_names[ptr:ptr+batch_size]
                    # print batch_names
                    # print batch_lens
                    batch_k_masks = training_k_masks[ptr:ptr+batch_size]
                    batch_k_lists = training_k_lists[ptr:ptr+batch_size]
                    train_dict.update({k_m:batch_k_masks,k_l:batch_k_lists})

                ptr += batch_size

                # print 'ready to run'
                sess.run(optimizer, feed_dict=train_dict)
                # print 'optim ran'

                if j%display_step == 0 :
                    # print batch_names
                    if self.loss_type == "XE":
                        cross_batch = sess.run(self.loss, feed_dict=train_dict)
                    else:
                        cross_batch = sess.run([self.loss,self.cross_entropy], feed_dict=train_dict)
                    print_str = "Batch "+str(j)+ ", Cross entropy = "+str(cross_batch)

                    print print_str
                    if summarize:
                        summary_b = sess.run(summary_batch,feed_dict=train_dict)
                        train_writer.add_summary(summary_b,global_step=n_batch)
                n_batch += 1


            valid_dict = {x: valid_data, y: valid_target, seq_len: valid_lengths}
            if self.loss_type in ["combined",'combined_norm','combined_cw',"XEtr_XEss"]:
                valid_dict.update({k_m:valid_k_masks,k_l:valid_k_lists})


            if self.loss_type == "XE":
                cross = self._run_by_batch(sess,cross_entropy_list,valid_dict,batch_size)
            else:
                cross = sess.run([self.loss,self.cross_entropy], feed_dict=valid_dict)
            print_str = "Epoch: " + str(i) + ", Cross Entropy = " + str(cross)

            if summarize:
                #summary = self._run_by_batch(sess,summary_op,{x: valid_data, y: valid_target},batch_size)
                summary_e = sess.run(summary_epoch,feed_dict=valid_dict)
                train_writer.add_summary(summary_e, global_step=i)
            print "_________________"
            print print_str
                          # "{:.5f}".format(cross)


            end_epoch = datetime.now()
            print "Computation time =", str(end_epoch-start_epoch)

            if self.loss_type != 'XE':
                cross= cross[0]
            #Check if cross is NaN, if so, stop computations
            if cross != cross :
                break



            # Save the variables to disk.
            if early_stop:
                if cross<best_cross:
                    saved = saver_best.save(sess, os.path.join(ckpt_save_path,"best_model.ckpt"),global_step=i)
                    best_cross = cross
                    epoch_since_best = 0
                else:
                    saved = saver.save(sess, os.path.join(ckpt_save_path,"model.ckpt"),global_step=i)
                    epoch_since_best += 1

            else:
                if i%save_step == 0 or i == epochs-1:
                    saved = saver.save(sess, os.path.join(ckpt_save_path,"model.ckpt"),global_step=i)
                else :
                    saved = saver.save(sess, os.path.join(ckpt_save_path,"model.ckpt"))
            print("Model saved in file: %s" % saved)

            i += 1
            # Shuffle the dataset before next epochs
            if not (chunks and memory):
                data.shuffle_one('train')
            print "_________________"

        return n_batch, n_epoch+epochs


    def train(self, data, save_path, train_param,sess=None,n_batch=0,n_epoch=0):


        if not train_param['summarize']:
            sess, saver, ckpt_save_path= self.initialize_training(save_path,train_param,sess=sess)
            train_writer, summary_batch, summary_epoch = None, None, None
        else:
            sess, saver, ckpt_save_path, train_writer, summary_batch, summary_epoch = self.initialize_training(save_path,train_param,sess=sess)

        n_batch,n_epoch  = self.perform_training(data,save_path,train_param,sess,saver,train_writer,ckpt_save_path,summary_batch, summary_epoch,n_batch=n_batch,n_epoch=n_epoch)

        print("Optimization finished ! "+str(datetime.now()))

        return n_batch, n_epoch

    def load(self,save_path,n_model):

        #x = tf.get_default_graph().get_tensor_by_name("x"+suffix+":0")

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        saver = tf.train.Saver()
        if n_model==None:
            # path = tf.train.latest_checkpoint(os.path.join("./ckpt/",save_path))
            folder = os.path.join("./ckpt/",save_path)
            path = None
            for file in os.listdir(folder):
                if 'best_model' in file and '.meta' in file:
                    path = os.path.join(folder,file.replace('.meta',''))
            if path is None:
                print "BEST MODEL NOT FOUND!"
                path = tf.train.latest_checkpoint(os.path.join("./ckpt/",save_path))
        else:
            path = os.path.join("./ckpt/",save_path,"model.ckpt-"+str(n_model))

        print "Loading "+path

        saver.restore(sess, path)
        return sess, saver

    def resume_training(self,load_path,data,save_path,train_param,n_model=None,n_batch=0,n_epoch=0):
        sess, saver = self.load(load_path,n_model)

        n_batch,n_epoch = self.train(data,save_path,train_param,sess=sess,n_batch=n_batch,n_epoch=n_epoch)
        return n_batch,n_epoch


    def run_prediction(self,dataset,len_list, save_path,n_model=None,sigmoid=False,sess=None,saver=None):

        if sess==None and saver==None:
            sess, saver = self.load(save_path,n_model)

        suffix = self.suffix
        if sigmoid:
            pred = self.pred_sigm
        else:
            pred = self.prediction
        x = self.inputs
        seq_len = self.seq_lens

        dataset = self._transpose_data(dataset)


        notes_pred = sess.run(pred, feed_dict = {x: dataset, seq_len: len_list} )
        if self.non_binary:
            output = np.transpose(notes_pred,[0,2,1,3])
        else:
            output = np.transpose(notes_pred,[0,2,1])
        return output


    def run_cross_entropy(self,dataset,len_list, save_path,n_model=None,batch_size=50,mean=True):
        sess, saver = self.load(save_path,n_model)
        cross_entropy = self.cross_entropy_list

        suffix = self.suffix
        x = self.inputs
        seq_len = self.seq_lens
        y = self.labels

        target = ground_truth(dataset)
        dataset = self._transpose_data(dataset)
        target = self._transpose_data(target)
#        print type(target)

        cross = self._run_by_batch(sess,cross_entropy,{x: dataset,y: target,seq_len: len_list},batch_size,mean=mean)
        return cross



    def compute_eval_metrics_pred(self,dataset,len_list,key_masks,key_lists,threshold,save_path,batch_size=1,n_model=None,sess=None,saver=None,key_thresh=0.05,active_thresh=0.05):

        # preds = self.run_prediction(dataset,len_list, save_path,n_model,sigmoid=True)
        # idx = preds[:,:,:] > threshold
        # preds_thresh = idx.astype(int)


        if sess==None and saver==None:
            sess, saver = self.load(save_path,n_model)

        cross = self.cross_entropy
        cross_trans = self.cross_entropy_transition
        cross_steady = self.cross_entropy_steady
        cross_len = self.cross_entropy_length
        cross_active = self.cross_entropy_active
        cross_key = self.cross_entropy_key
        combined = self.combined_metric
        combined_norm = self.combined_metric_norm
        k_thresh = self.thresh_key
        a_thresh = self.thresh_active


        data = self._transpose_data(dataset)
        targets = self._transpose_data(ground_truth(dataset))
        k_masks = self._transpose_data(key_masks)

        prec = tf.reduce_mean(self.precision)
        rec = tf.reduce_mean(self.recall)
        F0 = tf.reduce_mean(self.f_measure)

        suffix = self.suffix
        x = self.inputs
        seq_len = self.seq_lens
        y = self.labels
        thresh = self.thresh
        k_mask = self.key_masks
        k_lists = self.key_lists


        cross, cross_trans, cross_steady, cross_len, cross_active, cross_key , combined, combined_norm,precision, recall, F_measure = sess.run([cross, cross_trans, cross_steady, cross_len, cross_active, cross_key, combined, combined_norm, prec, rec, F0], feed_dict = {x: data, seq_len: len_list, y: targets, thresh: threshold,k_mask:k_masks,k_lists:key_lists,k_thresh:key_thresh, a_thresh:active_thresh} )

        return F_measure, precision, recall, cross, cross_trans, cross_steady, cross_len, cross_active,cross_key, combined, combined_norm


    def compute_eval_metrics_from_outputs(self,inputs,outputs,len_list,key_masks,key_lists,threshold,expected_measures=False,logits=False,trim_outputs=True):
        cross = self.cross_entropy
        cross_trans = self.cross_entropy_transition
        cross_steady = self.cross_entropy_steady
        cross_len = self.cross_entropy_length
        cross_active = self.cross_entropy_active
        cross_key = self.cross_entropy_key
        combined = self.combined_metric
        combined_norm = self.combined_metric_norm

        prec = tf.reduce_mean(self.precision)
        rec = tf.reduce_mean(self.recall)
        F0 = tf.reduce_mean(self.f_measure)


        pred = self.prediction
        pred_thresh = self.pred_thresh

        suffix = self.suffix
        x = self.inputs
        seq_len = self.seq_lens
        y = self.labels
        thresh = tf.get_default_graph().get_tensor_by_name("thresh"+suffix+":0")
        k_mask = self.key_masks
        k_lists = self.key_lists

        data = self._transpose_data(inputs)
        if trim_outputs:
            outputs = self._transpose_data(outputs[:,:,1:])
        else:
            outputs = self._transpose_data(outputs)
        targets = self._transpose_data(ground_truth(inputs))
        k_masks = self._transpose_data(key_masks)

        if logits:
            outputs = scp.special.logit(outputs)
            outputs = np.nan_to_num(outputs)

        outputs_thresh = (scp.special.expit(outputs)>threshold).astype(int)

        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # sess.run(init)

        cross, cross_trans, cross_steady, cross_len, cross_active, cross_key, combined, combined_norm = sess.run([cross, cross_trans, cross_steady, cross_len, cross_active,cross_key, combined, combined_norm], feed_dict = {x: data, seq_len: len_list, y: targets, thresh: threshold, pred:outputs,k_mask:k_masks,k_lists:key_lists } )
        precision, recall, F_measure = sess.run([prec, rec, F0], feed_dict = {pred_thresh: outputs_thresh, y: targets} )

        if expected_measures:
            precs = []
            recs = []
            fs = []
            for i in range(50):
                sample_pred = sample(outputs)
                sample_p, sample_r, sample_f = sess.run([prec, rec, F0], feed_dict = {pred_thresh: sample_pred, y: targets} )
                precs += [sample_p]
                recs += [sample_r]
                fs += [sample_f]
            exp_p = sum(precs)/len(precs)
            exp_r = sum(recs)/len(recs)
            exp_f = sum(fs)/len(fs)

            return F_measure, precision, recall, cross, cross_trans, cross_steady, cross_len, cross_active, cross_key, combined, combined_norm,  exp_f, exp_p, exp_r

        else:
            return F_measure, precision, recall, cross, cross_trans, cross_steady, cross_len, cross_active, cross_key, combined, combined_norm

def sample(probs):
    #Takes in a vector of probabilities, and returns a random vector of 0s and 1s sampled from the input vector
    return np.floor(probs + np.random.uniform(0, 1,probs.shape))

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)



def getTotalNumParameters():
    '''
    Returns the total number of parameters contained in all trainable variables
    :return: Number of parameters (int)
    '''
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        print variable.name
        print variable.get_shape()
    return total_parameters


def make_model_from_dataset(dataset,model_param):

    n_notes = dataset.get_n_notes()
    n_steps = dataset.get_len_files()

    model_param['n_notes']=n_notes
    model_param['n_steps']=n_steps

    return Model(model_param)

def make_save_path(base_path,model_param,rep=None):

    n_hidden = model_param['n_hidden']
    learning_rate = model_param['learning_rate']

    if type(n_hidden) == int:
        hidden_learning_path= str(n_hidden)+"_"+str(learning_rate)
    else:
        n_hidden_string = '-'.join([str(n) for n in n_hidden])
        hidden_learning_path= n_hidden_string+"_"+str(learning_rate)

    if model_param['chunks']:
        chunk_path = "_ch"+str(model_param['chunks'])
    else:
        chunk_path = ""

    if model_param['learning_rate_pre'] and model_param['epochs_pre']:
        pretrain_path = "_lrpre"+str(model_param['learning_rate_pre'])+"_ep"+str(model_param['epochs_pre'])
    else:
        pretrain_path=""

    if rep!=None:
        rep_path = "_"+str(rep)
    else:
        rep_path=""

    return os.path.join(base_path,hidden_learning_path+chunk_path+pretrain_path+rep_path+"/")


def make_model_param():
    model_param = {}

    model_param['n_hidden']=128
    model_param['learning_rate']=0.01
    model_param['n_notes']=88
    model_param['n_steps']=300
    model_param['batch_size']=50

    model_param['chunks']=None
    model_param['non_binary']=False
    model_param['grad_clip'] = None
    model_param['cell_type'] = 'LSTM'
    model_param['loss_type'] = 'XE'
    model_param['weights_tr_ss'] = [1,1]
    model_param['alpha'] = 0
    model_param['normalise_XEtr'] = True
    model_param['activ']='sigm'
    model_param['device_name']="/gpu:0"
    model_param['suffix']=""

    model_param['learning_rate_pre']=None
    model_param['epochs_pre']=None

    return model_param

def make_train_param():
    train_param = {}

    train_param['epochs']=20
    train_param['batch_size']=50
    train_param['dropout']=1.0

    train_param['display_per_epoch']=10,
    train_param['save_step']=1
    train_param['max_to_keep']=5
    train_param['summarize']=True
    train_param['early_stop']=True
    train_param['early_stop_epochs']=15
    train_param['dataset_generator']=False
    return train_param
