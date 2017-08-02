import tensorflow as tf
import numpy as np

def fc_layer(input_, output_size, activation = None, bias = True, scope=None):
    '''
    fully convlolution layer
    Args :
        input_  - 2D tensor
            general shape : [batch, input_size]
        output_size - int
            shape of output 2D tensor
        activation - activation function
            defaults to be None
        scope - string
            defaults to be None then scope becomes "fc"
    '''
    with tf.variable_scope(scope or "fc"):
        w = tf.get_variable(name="weight", shape = [input_.get_shape().as_list()[1], output_size], initializer=tf.contrib.layers.xavier_initializer())
        output_ = tf.matmul(input_, w)
        if bias:
            b = tf.get_variable(name="bias", shape = [output_size], initializer=tf.constant_initializer(0.001))
            output_+=b
        return output_ if activation is None else activation(output_)

class MANNCell(tf.contrib.rnn.RNNCell):
    def __init__(self, mem_size=128, mem_dim=40, controller_dim=200, batch_size=100, n_reads=4, output_dim = 28, name=None):
        self.mem_size=mem_size
        self.mem_dim=mem_dim
        self.n_h=controller_dim
        self.n_reads=n_reads
        self.batch_size=batch_size
        self.n_o=output_dim
        if name is None:
            name="MANN"
        self.name=name
        self.gamma=0.8
        self.lstm = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple=True)
        
    @property
    def state_size(self):
        return 
    
    @property
    def output_size(self):
        return self.n_h
    
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]        

    def __call__(self, x, state=None, scope=None):
        '''
        Args:
            x - 2D tensor [batch_size, input_dim]
            state - dict
                M - 3D tensor [batch_size, mem_size, mem_dim]
                    memory cell
                c - 2D tensor [batch_size, n_h]
                    cell_state
                h - 2D tensor [batch_size, n_h]
                    hidden_state
                r - 2D tensor [batch_size, n_reads * mem_dim]
                    read vector
                wr - 3D tensor [batch_size, n_reads, mem_size]
                    read_weight vector what to read 
                wu - 2D tensor [batch_size, mem_size]
                    usage vector
        '''
        self.input_dim = x.get_shape().as_list()[1]
        if state == None:
            state = self.initial_state()
        
        M_prev = state['M'] # [batch_size, mem_size, mem_dim]
        c_prev = state['c'] # [batch_size, n_h]
        h_prev = state['h'] # [batch_size, n_h]
        r_prev = state['r'] # [batch_size, nreads*mem_dim]
        wr_prev = state['wr'] # [batch_size, n_reads, mem_size]
        wu_prev = state['wu'] # [batch_size, mem_size]
        
        with tf.variable_scope(self.name):
            # LSTM
            _, (h_t, c_t) = self.lstm(tf.concat(values=(x, r_prev), axis=1), (h_prev, c_prev))
            # key vector
            k_t = tf.reshape(fc_layer(h_t, self.n_reads*self.mem_dim, activation=tf.tanh, scope='key'), shape=[-1, self.n_reads, self.mem_dim]) # [batch_size, n_reads, mem_dim]
            # write add vector
            a_t = tf.reshape(fc_layer(h_t, self.n_reads*self.mem_dim, activation=tf.tanh, scope='add'), shape=[-1, self.n_reads, self.mem_dim]) # [batch_size, n_reads, mem_dim]
            # sigma value how much interpolate
            sigma_t = fc_layer(h_t, self.n_reads, activation = tf.nn.sigmoid, scope = 'sigma') # [batch_size, n_reads]
        
        sigma_t_tile = tf.tile(tf.reshape(sigma_t, [-1, self.n_reads, 1]), [1, 1, self.mem_size]) # [batch_size, n_reads, mem_size]

        def least_used_checker(tensor, n=1):
            '''
            tensor - weight used vector 
            For every row of tensor, least used element becomes 1 otherwise 0
            
            Args:
                tensor - 2D tensor [row, col]
                    all positive 
                n - int
                    defaults to be 1
            Return :
                2D tensor same shape with tensor, tf.float32
            '''
            _, col = tensor.get_shape().as_list()
            tensor_3d = tf.tile(tf.expand_dims(tensor, axis=1), [1, n, 1]) # [batch, n_reads, mem_size]
            n_small, _= tf.nn.top_k(-tensor, n, sorted=True) # [batch, n_reads]
            n_small_tile = tf.tile(tf.expand_dims(-n_small, 2), [1, 1, col]) # [batch, n_reads, mem_size]
            return tf.cast(tf.equal(n_small_tile, tensor_3d), tf.float32)
        '''
        wu_prev - [batch_size, mem_size]
            n_reads < mem_size
            usage track vector
        wlu_prev_3d - [batch_size, n_reads, mem_size]
            least used weight 
            only 1 of mem_size should be 1, otherwise 0
            least n_reads elements become 1
        wlu_prev_2d - [batch_size, mem_size]
            least n_reads elements become 1 according to wu_prev 
        '''
        wlu_prev_3d = least_used_checker(wu_prev, self.n_reads) # [batch, n_reads, mem_size]
        wlu_prev_2d = tf.reduce_sum(wlu_prev_3d, axis=1) # [batch, mem_size]
        '''
        Memory write 
            ww_t - sigma*wr_prev + (1-sigma)*wlu_prev
                wr_prev - [batch_size, n_reads, mem_size]
                sigma_t_tile - [batch_size, n_reads, mem_size]
                wlu_prev_3d - [batch_size, nreads, mem_size]
        '''
        ww_t = sigma_t_tile*wr_prev + (1.0-sigma_t_tile)*wlu_prev_3d # [batch_size, n_reads, mem_size]
        '''
        Memory
            M_t - [batch_size, mem_size, mem_dim]
            M_t = M_prev(delete least used index) + ww_t*a_t
                ww_t - [batch_size, n_reads, mem_size]
                transpose(ww_t) - [batch_size, mem_size, n_reads]
                a_t - [batch_size, n_reads, mem_dim]
        '''
        M_t = tf.tile(tf.reshape(1.0-wlu_prev_2d, [-1, self.mem_size, 1]), [1, 1, self.mem_dim])*M_prev
        with tf.variable_scope("M_t"):
            M_t+=tf.matmul(tf.transpose(ww_t, perm=[0,2,1]), a_t)
        
        '''
        Memory read
            k_t - [batch_size, n_reads, mem_dim]
            M_t - [batch_size, mem_size, mem_dim]
                =>K_t - [batch_size, n_reads, Mstate] 
                =>K_t - [batch_size*n_reads, Mstate]
        '''
        K_t = tf.matmul(k_t, tf.transpose(M_t, perm=[0, 2,1,])) # [batch_size, n_reads, mem_size]
        K_t = tf.reshape(K_t, [self.batch_size*self.n_reads, self.mem_size]) # [batch_size*n_reads, mem_size]
        wr_t = tf.reshape(tf.nn.softmax(K_t, dim=-1), [self.batch_size, self.n_reads, self.mem_size]) # [batch_size, n_reads, mem_size]
        '''
        wr_t - [batch_size, n_reads, mem_size]
        M_prev - [batch_size, mem_size, mem_dim]
            =>
            r_t - [batch_size, n_reads, mem_dim]
            =>
            r_t - [batch_size, n_reads*mem_dim]
        '''
        r_t = tf.reshape(tf.matmul(wr_t, M_prev), [self.batch_size,-1])
        '''
        Update usage weight
            wu_t = gamma*wu_prev+wr_t+ww_r
        '''
        wu_t = self.gamma * wu_prev + tf.reduce_sum(wr_t, axis=1) + tf.reduce_sum(ww_t, axis=1) #[batch_size, mem_size]
        
        new_state = {'M' : M_t, 'c' : c_t, 'h': h_t, 'r': r_t, 'wr': wr_t, 'wu' : wu_t}
        with tf.variable_scope(self.name):
            o_t = fc_layer(h_t, self.n_o, scope="ho")
        return o_t, new_state

    def initial_state(self):
        '''Generate initial state when state is None'''
        def dummy_one_hot(shape):
            '''dummy one_hot vectors on last dim'''
            dummy = np.zeros(shape, dtype = np.float32)
            dummy[...,0] = 1
            return dummy
        
        with tf.variable_scope("init_state"):
            M_0 = tf.Variable(1e-6*np.ones([self.batch_size, self.mem_size, self.mem_dim]), name= 'memory', dtype=tf.float32, trainable=False) # memory
            c_0 = tf.Variable(np.zeros([self.batch_size, self.n_h]), name='memory_cell_state', dtype=tf.float32, trainable=False) # cell_state 
            h_0 = tf.Variable(np.zeros([self.batch_size, self.n_h]), name='hidden_state', dtype=tf.float32, trainable=False) # hiden_state
            r_0 = tf.Variable(np.zeros([self.batch_size, self.n_reads * self.mem_dim]), name='read_vector', dtype=tf.float32, trainable=False) # Read vector
            wr_0 = tf.Variable(dummy_one_hot([self.batch_size, self.n_reads, self.mem_size]), name='wr', dtype=tf.float32, trainable=False) # read_weight 
            wu_0 = tf.Variable(dummy_one_hot([self.batch_size, self.mem_size]), name='wu', dtype=tf.float32, trainable=False) # usage_weight
        
        state = {'M' : M_0, 'c' : c_0, 'h' : h_0, 'r' : r_0, 'wr' : wr_0, 'wu' : wu_0} 
        return state
