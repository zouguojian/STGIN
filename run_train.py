# -- coding: utf-8 --
'''
the shape of sparsetensor is a tuuple, like this
(array([[  0, 297],
       [  0, 296],
       [  0, 295],
       ...,
       [161,   2],
       [161,   1],
       [161,   0]], dtype=int32), array([0.00323625, 0.00485437, 0.00323625, ..., 0.00646204, 0.00161551,
       0.00161551], dtype=float32), (162, 300))
axis=0: is nonzero values, x-axis represents Row, y-axis represents Column.
axis=1: corresponding the nonzero value.
axis=2: represents the sparse matrix shape.
'''

from __future__ import division
from __future__ import print_function
from models.utils import *
from models.models import GCN
from models.hyparameter import parameter
from models.embedding import embedding
from models.bridge import BridgeTrans
from models.st_block import ST_Block
from models.inits import *
from models.data_load import *

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
tf.random.set_random_seed(seed=22)
np.random.seed(22)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


class Model(object):
    def __init__(self, para, mean, std):
        self.para = para
        self.mean = mean
        self.std = std
        self.num_heads = self.para.num_heads
        self.input_len = self.para.input_length
        self.output_len = self.para.output_length
        self.total_len = self.input_len + self.output_len
        self.features = self.para.features
        self.batch_size = self.para.batch_size
        self.epochs = self.para.epoch
        self.site_num = self.para.site_num
        self.emb_size = self.para.emb_size
        self.is_training = self.para.is_training
        self.learning_rate = self.para.learning_rate
        self.model_name = self.para.model_name
        self.granularity = self.para.granularity
        self.decay_epoch=self.para.decay_epoch
        self.adj = preprocess_adj(self.adjecent())
        self.num_train = 23967

        # define gcn model
        if self.para.model_name == 'gcn_cheby':
            self.support = chebyshev_polynomials(self.adj, self.para.max_degree)
            self.num_supports = 1 + self.para.max_degree
            self.model_func = GCN
        else:
            self.support = [self.adj]
            self.num_supports = 1
            self.model_func = GCN

        # define placeholders
        self.placeholders = {
            'position': tf.placeholder(tf.int32, shape=(1, self.site_num), name='input_position'),
            'day_of_week': tf.placeholder(tf.int32, shape=(None, self.site_num), name='input_day_of_week'),
            'minute_of_day': tf.placeholder(tf.int32, shape=(None, self.site_num), name='input_minute_of_day'),
            'indices_i': tf.placeholder(dtype=tf.int64, shape=[None, None], name='input_indices'),
            'values_i': tf.placeholder(dtype=tf.float32, shape=[None], name='input_values'),
            'dense_shape_i': tf.placeholder(dtype=tf.int64, shape=[None], name='input_dense_shape'),
            'features': tf.placeholder(tf.float32, shape=[None, self.input_len, self.site_num, self.features], name='input_s'),
            'labels': tf.placeholder(tf.float32, shape=[None, self.site_num, self.total_len], name='labels_s'),
            'features_all': tf.placeholder(tf.float32, shape=[None, self.total_len, self.site_num, self.features], name='input_all_s'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='input_dropout'),
            'num_features_nonzero': tf.placeholder(tf.int32, name='input_zero'),  # helper variable for sparse dropout
            'is_training': tf.placeholder(shape=(), dtype=tf.bool)
        }
        self.supports = [tf.SparseTensor(indices=self.placeholders['indices_i'],
                                         values=self.placeholders['values_i'],
                                         dense_shape=self.placeholders['dense_shape_i']) for _ in range(self.num_supports)]
        self.embeddings()
        self.model()

    def adjecent(self):
        '''
        :return: adj matrix
        '''
        data = pd.read_csv(filepath_or_buffer=self.para.file_adj)
        adj = np.zeros(shape=[self.para.site_num, self.para.site_num])
        for line in data[['src_FID', 'nbr_FID']].values:
            adj[line[0]][line[1]] = 1
        return adj

    def embeddings(self):
        '''
        :return:
        '''
        p_emd = embedding(self.placeholders['position'], vocab_size=self.para.site_num, num_units=self.emb_size,scale=False, scope="position_embed")
        p_emd = tf.reshape(p_emd, shape=[1, self.site_num, self.emb_size])
        self.p_emd = tf.expand_dims(p_emd, axis=0)

        w_emb = embedding(self.placeholders['day_of_week'], vocab_size=7, num_units=self.emb_size, scale=False, scope="day_embed")
        self.w_emd = tf.reshape(w_emb, shape=[-1, self.total_len, self.site_num, self.emb_size])

        m_emb = embedding(self.placeholders['minute_of_day'], vocab_size=24 * 60 //self.granularity, num_units=self.emb_size,scale=False, scope="minute_embed")
        self.m_emd = tf.reshape(m_emb, shape=[-1, self.total_len, self.site_num, self.emb_size])

    def model(self):
        '''
        :param batch_size: 64
        :param encoder_layer:
        :param decoder_layer:
        :param encoder_nodes:
        :param prediction_size:
        :param is_training: True
        :return:
        '''
        with tf.variable_scope(name_or_scope='encoder'):
            '''
            return, the gcn output --- for example, inputs.shape is :  (32, 3, 162, 32)
            axis=0: bath size
            axis=1: input data time size
            axis=2: numbers of the nodes
            axis=3: output feature size
            '''
            timestamp = [self.w_emd, self.m_emd]
            position = self.p_emd

            global_step = tf.Variable(0, trainable=False)
            bn_momentum = tf.train.exponential_decay(0.5, global_step,
                                                     decay_steps=self.decay_epoch * self.num_train // self.batch_size,
                                                     decay_rate=0.5, staircase=True)
            bn_decay = tf.minimum(0.99, 1 - bn_momentum)

            X_All = FC(self.placeholders['features_all'], units=[self.emb_size, self.emb_size], activations=[tf.nn.relu, None],
                bn=True, bn_decay=bn_decay, is_training=self.placeholders['is_training'])

            if self.model_name == 'STGIN_1':
                speed = FC(self.placeholders['features'], units=[self.emb_size, self.emb_size], activations=[tf.nn.relu, None],
                            bn=True, bn_decay=bn_decay, is_training=self.placeholders['is_training'])
            else:
                speed = tf.transpose(self.placeholders['features'],perm=[0, 2, 1, 3])
                speed = tf.reshape(speed, [-1, self.input_len, self.features])
                speed3 = tf.layers.conv1d(inputs=speed,
                                         filters=self.emb_size,
                                         kernel_size=3,
                                         padding='SAME',
                                         kernel_initializer=tf.truncated_normal_initializer(),
                                         name='conv_1')
                speed2 = tf.layers.conv1d(inputs=tf.reverse(speed,axis=[1]),
                                         filters=self.emb_size,
                                         kernel_size=3,
                                         padding='SAME',
                                         kernel_initializer=tf.truncated_normal_initializer(),
                                         name='conv_2')
                speed1 = tf.layers.conv1d(inputs=speed,
                                         filters=self.emb_size,
                                         kernel_size=1,
                                         padding='SAME',
                                         kernel_initializer=tf.truncated_normal_initializer(),
                                         name='conv_3')
                speed2 = tf.reverse(speed2, axis=[1])
                speed2 = tf.multiply(speed2, tf.nn.sigmoid(speed2))
                speed3 = tf.multiply(speed3, tf.nn.sigmoid(speed3))
                speed = tf.add_n([speed1, speed2, speed3])
                speed = tf.reshape(speed, [-1, self.site_num, self.input_len, self.emb_size])
                speed = tf.transpose(speed, perm=[0, 2, 1, 3])

            STE = STEmbedding(position, timestamp, 0, self.emb_size, True, bn_decay, self.placeholders['is_training'])
            st_block = ST_Block(hp=self.para, placeholders=self.placeholders, input_length=self.input_len,
                                    model_func=self.model_func)
            if self.para.model_name == 'STGIN_2':
                encoder_outs = st_block.spatiotemporal_(bn=True,
                                                        bn_decay=bn_decay,
                                                        is_training=self.placeholders['is_training'],
                                                        speed=speed,
                                                        STE=STE[:, :self.input_len],
                                                        supports=self.supports,
                                                        speed_all=X_All)
            else:
                encoder_outs = st_block.spatiotemporal(bn=True,
                                                       bn_decay=bn_decay,
                                                       is_training=self.placeholders['is_training'],
                                                       speed=speed,
                                                       STE=STE[:, :self.input_len],
                                                       supports=self.supports,
                                                       speed_all=X_All, adj=self.adj)
            print('encoder encoder_outs shape is : ', encoder_outs.shape)

        with tf.variable_scope(name_or_scope='bridge'):
            X = encoder_outs
            X = BridgeTrans(X, X + STE[:, :self.input_len], STE[:, self.input_len:] + X_All[:,self.input_len:], self.num_heads, self.emb_size // self.num_heads, True, bn_decay, self.placeholders['is_training'])
            print('bridge bridge_outs shape is : ', X.shape)
        # X = st_block.dynamic_decoding(hiddens=encoder_outs, STE=STE[:, self.input_len:])

        pre = FC(
                X, units=[self.emb_size, 1], activations=[None, None],
                bn=True, bn_decay=bn_decay, is_training=self.placeholders['is_training'],
                use_bias=True, drop=0.1)
        pre = pre * (self.std) + self.mean
        self.pre = tf.transpose(tf.squeeze(pre, axis=-1), [0, 2, 1], name='output_y')
        print('prediction values shape is : ', self.pre.shape)
        observed = self.placeholders['labels'][:,:,self.input_len:]
        predicted = self.pre

        learning_rate = tf.train.exponential_decay(
            self.learning_rate, global_step,
            decay_steps=self.decay_epoch * self.num_train // self.batch_size,
            decay_rate=0.7, staircase=True)
        learning_rate = tf.maximum(learning_rate, 1e-5)
        self.loss = mae_los(predicted, observed)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)

    def test(self):
        '''
        :param batch_size: usually use 1
        :param encoder_layer:
        :param decoder_layer:
        :param encoder_nodes:
        :param prediction_size:
        :param is_training: False
        :return:
        '''
        model_file = tf.train.latest_checkpoint('weights/')
        self.saver.restore(self.sess, model_file)

    def initialize_session(self,session):
        self.sess = session
        self.saver = tf.train.Saver()

    def run_epoch(self, trainX, trainDoW, trainM, trainL, trainXAll, valX, valDoW, valM, valL, valXAll):
        '''
        from now on,the model begin to training, until the epoch to 100
        '''
        max_mae = 100
        shape = trainX.shape
        num_batch = math.floor(shape[0] / self.batch_size)
        self.num_train=shape[0]
        self.sess.run(tf.global_variables_initializer())
        start_time = datetime.datetime.now()
        iteration=0
        for epoch in range(self.epochs):
            # shuffle
            permutation = np.random.permutation(shape[0])
            trainX = trainX[permutation]
            trainDoW = trainDoW[permutation]
            trainM = trainM[permutation]
            trainL = trainL[permutation]
            trainXAll = trainXAll[permutation]
            for batch_idx in range(num_batch):
                iteration+=1
                start_idx = batch_idx * self.batch_size
                end_idx = min(shape[0], (batch_idx + 1) * self.batch_size)
                xs = np.expand_dims(trainX[start_idx : end_idx], axis=-1)
                day_of_week = np.reshape(trainDoW[start_idx : end_idx], [-1, self.site_num])
                minute_of_day = np.reshape(trainM[start_idx : end_idx], [-1, self.site_num])
                labels = trainL[start_idx : end_idx]
                xs_all = np.expand_dims(trainXAll[start_idx : end_idx], axis=-1)
                feed_dict = construct_feed_dict(xs=xs,
                                                xs_all=xs_all,
                                                labels=labels,
                                                day_of_week=day_of_week,
                                                minute_of_day=minute_of_day,
                                                adj=self.adj,
                                                placeholders=self.placeholders,
                                                sites=self.site_num)
                feed_dict.update({self.placeholders['dropout']: self.para.dropout})

                loss, _ = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)
                # print("after %d steps,the training average loss value is : %.6f" % (batch_idx, loss))

                if iteration % 100 == 0:
                    end_time = datetime.datetime.now()
                    total_time = end_time - start_time
                    print("Total running times is : %f" % total_time.total_seconds())

            print('validation')
            mae = self.evaluate(valX, valDoW, valM, valL, valXAll)  # validate processing
            if max_mae > mae:
                print("in the %dth epoch, the validate average loss value is : %.3f" % (epoch + 1, mae))
                max_mae = mae
                self.saver.save(self.sess, save_path=self.para.save_path)

    def evaluate(self, testX, testDoW, testM, testL, testXAll):
        '''
        :param para:
        :param pre_model:
        :return:
        '''
        labels_list, pres_list = list(), list()
        if not self.is_training:
            # model_file = tf.train.latest_checkpoint(self.para.save_path)
            saver = tf.train.import_meta_graph(self.para.save_path + '.meta')
            # saver.restore(sess, args.model_file)
            print('the model weights has been loaded:')
            saver.restore(self.sess, self.para.save_path)

        parameters = 0
        for variable in tf.trainable_variables():
            parameters += np.product([x.value for x in variable.get_shape()])
        print('trainable parameters: {:,}'.format(parameters))

        textX_shape = testX.shape
        total_batch = math.floor(textX_shape[0] / self.batch_size)
        start_time = datetime.datetime.now()
        for b_idx in range(total_batch):
            start_idx = b_idx * self.batch_size
            end_idx = min(textX_shape[0], (b_idx + 1) * self.batch_size)
            xs = np.expand_dims(testX[start_idx: end_idx], axis=-1)
            day_of_week = np.reshape(testDoW[start_idx: end_idx], [-1, self.site_num])
            minute_of_day = np.reshape(testM[start_idx: end_idx], [-1, self.site_num])
            labels = testL[start_idx: end_idx]
            xs_all = np.expand_dims(testXAll[start_idx: end_idx], axis=-1)
            feed_dict = construct_feed_dict(xs=xs,
                                            xs_all=xs_all,
                                            labels=labels,
                                            day_of_week=day_of_week,
                                            minute_of_day=minute_of_day,
                                            adj=self.adj,
                                            placeholders=self.placeholders,
                                            sites=self.site_num, is_traning=False)
            feed_dict.update({self.placeholders['dropout']: 0.0})
            pre= self.sess.run((self.pre), feed_dict=feed_dict)

            labels_list.append(labels[:,:,self.input_len:])
            pres_list.append(pre)

        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print("Total running times is : %f" % total_time.total_seconds())

        labels_list = np.concatenate(labels_list, axis=0)
        pres_list = np.concatenate(pres_list, axis=0)
        np.savez_compressed('data/STGIN-' + 'YINCHUAN', **{'prediction': pres_list, 'truth': labels_list})

        print('                MAE\t\tRMSE\t\tMAPE')
        if not self.is_training:
            for i in range(self.para.output_length):
                mae, rmse, mape = metric(pres_list[:,:,i], labels_list[:,:,i])
                print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' % (i + 1, mae, rmse, mape * 100))
        mae, rmse, mape = metric(pres_list, labels_list)  # 产生预测指标
        print('average:         %.3f\t\t%.3f\t\t%.3f%%' %(mae, rmse, mape * 100))

        return mae


def main(argv=None):
    '''
    :param argv:
    :return:
    '''
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = InteractiveSession(config=config)
    print('#......................................beginning........................................#')
    para = parameter(argparse.ArgumentParser())
    para = para.get_para()

    print('Please input a number : 1 or 0. (1 and 0 represents the training or testing, respectively).')
    val = input('please input the number : ')

    if int(val) == 1:
        para.is_training = True
    else:
        para.batch_size = 1
        para.is_training = False

    trainX, trainDoW, trainM, trainL, trainXAll, valX, valDoW, valM, valL, valXAll, testX, testDoW, testM, testL, testXAll, mean, std = loadData(para)
    print('trainX: %s\ttrainY: %s' % (trainX.shape, trainL.shape))
    print('valX:   %s\t\tvalY:   %s' % (valX.shape, valL.shape))
    print('testX:  %s\t\ttestY:  %s' % (testX.shape, testL.shape))
    print('data loaded!')

    pre_model = Model(para, mean, std)
    pre_model.initialize_session(session)
    if int(val) == 1:
        pre_model.run_epoch(trainX, trainDoW, trainM, trainL, trainXAll, valX, valDoW, valM, valL, valXAll)
    else:
        pre_model.evaluate(testX, testDoW, testM, testL, testXAll)

    print('#...................................finished............................................#')


if __name__ == '__main__':
    main()