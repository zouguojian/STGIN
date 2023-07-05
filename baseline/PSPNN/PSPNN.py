# -- coding: utf-8 --
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
'''
noted that , for PSPNN, we do not use weather or other additional data,
we just use the PSPNN model to extract the spatio-temporal correlation of input traffic data,
and to predict long-term traffic speed.
if you have weather or other data, you can added it to model, thanks.
'''
class PspnnClass(object):
    def __init__(self, batch_size, predict_time,layer_num=1, nodes=64, placeholders=None):
        '''

        :param batch_size:
        :param layer_num:
        :param nodes:
        :param is_training:
        '''
        self.batch_size=batch_size
        self.layer_num=layer_num
        self.nodes=nodes
        self.placeholders=placeholders
        self.predict_time=predict_time
        self.h=3
        self.w=3
        self.position_size=108
        self.decoder()

    def cnn(self,X):
        '''
        :param x: shape is [batch size * input length, site num, features]
        :return: shape is [batch size, height, channel]
        '''

        filter1=tf.get_variable("filter1", [self.h,self.w,1,32], initializer=tf.truncated_normal_initializer())
        X=tf.nn.conv2d(input=X,filter=filter1,strides=[1,1,1,1],padding='SAME')
        X=tf.nn.relu(X)

        filter3 = tf.get_variable("filter3", [self.h,self.w,32,128], initializer=tf.truncated_normal_initializer())
        X = tf.nn.conv2d(input=X, filter=filter3, strides=[1, 1, 1, 1], padding='SAME')
        X=tf.nn.relu(X)

        '''shape is  : [batch size, site num, features, channel]'''
        print('X shape is : ',X.shape)
        return X

    def decoder(self):
        '''
        :return:  shape is [batch size, time size, hidden size]
        '''

        def cell():
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)  # single lstm unit
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=1-self.placeholders['dropout'])

            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.nodes)  # single lstm unit
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=1-self.placeholders['dropout'])

            return cell_fw, cell_bw

        cell_fw, cell_bw=cell()

        self.df_mlstm=tf.nn.rnn_cell.MultiRNNCell([cell_fw for _ in range(self.layer_num)])
        self.db_mlstm = tf.nn.rnn_cell.MultiRNNCell([cell_bw for _ in range(self.layer_num)])

    def decoding(self,  x):
        '''
        :param encoder_hs:
        :return:  shape is [batch size, prediction size]
        '''

        shape = x.get_shape().as_list()
        x=self.cnn(X=tf.reshape(x, shape=[shape[0], shape[1], shape[2], shape[3]]))
        shape = x.get_shape().as_list()

        x=tf.transpose(x,perm=[0, 2, 1, 3])
        x=tf.reshape(x,shape=[-1, shape[1], shape[3]]) # [batch * site num, input leangth, dim]
        print(x.shape)

        with tf.variable_scope('encoder_lstm'):

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.df_mlstm, self.db_mlstm, x,
                                                                        dtype=tf.float32)  # [2, batch_size, seq_length, output_size]
            outputs = tf.concat(outputs, axis=2)
            print(outputs.shape)

            results_speed = tf.layers.dense(inputs=outputs[:,-1,:], units=64, activation=tf.nn.relu, name='layer_spped_1')
            results = tf.layers.dense(inputs=results_speed, units=self.predict_time, name='layer_speed_2')

        return tf.reshape(results, shape=[-1, shape[2], self.predict_time])