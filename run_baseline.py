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
from models.hyparameter import parameter
from models.embedding import embedding
from models.data_load import *
from baseline.PSPNN.PSPNN import PspnnClass
from baseline.FIRNNs.FI_LSTM import FirnnClass
from models.inits import *

tf.reset_default_graph()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logs_path = "board"

os.environ['CUDA_VISIBLE_DEVICES']='4'
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

tf.random.set_random_seed(seed=22)
np.random.seed(22)


class Model(object):
    def __init__(self, para, mean, std):
        self.para = para
        self.mean = mean
        self.std = std
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
        self.num_train = 23967

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

        self.embeddings()
        self.model()

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

        # encoder
        print('#................................in the encoder step....................................#')
        if self.para.model_name=='PSPNN':
            features = tf.reshape(self.placeholders['features'], shape=[self.batch_size,
                                                                         self.input_len,
                                                                         self.site_num,
                                                                         self.features])
            # this step use to encoding the input series data
            '''
            lstm, return --- for example ,output shape is :(32, 3, 162, 128)
            axis=0: bath size
            axis=1: input data time size
            axis=2: numbers of the nodes
            axis=3: output feature size
            '''
            encoder_init = PspnnClass(self.batch_size ,
                                        predict_time=self.output_len,
                                        layer_num=self.para.hidden_layer,
                                        nodes=self.para.hidden_size,
                                        placeholders=self.placeholders)
            # this step to presict the polutant concentration
            self.pre = encoder_init.decoding(features)

        elif self.para.model_name=='FI-RNNs':
            timestamp = [self.w_emd, self.m_emd]
            position = self.p_emd
            STE = STEmbedding(position, timestamp, 0, self.emb_size, False, 0.99, self.is_training)
            Q_STE = STE[:, :self.input_len]

            features = tf.reshape(self.placeholders['features'], shape=[self.batch_size,
                                                                         self.input_len,
                                                                         self.site_num,
                                                                         self.features])
            # this step use to encoding the input series data
            '''
            lstm, return --- for example ,output shape is :(32, 3, 162, 128)
            axis=0: bath size
            axis=1: input data time size
            axis=2: numbers of the nodes
            axis=3: output feature size
            '''
            encoder_init = FirnnClass(self.batch_size * self.site_num,
                                        predict_time=self.output_len,
                                        layer_num=self.para.hidden_layer,
                                        nodes=self.para.hidden_size,
                                        placeholders=self.placeholders)
            inputs = tf.transpose(features, perm=[0, 2, 1, 3])
            inputs = tf.reshape(inputs, shape=[-1, self.input_len, self.features])
            Q_STE = tf.transpose(Q_STE, perm=[0, 2, 1, 3])
            Q_STE = tf.reshape(Q_STE, shape=[-1, self.input_len, self.emb_size])
            h_states= encoder_init.encoding(inputs, STE=Q_STE)

            # this step to presict the polutant concentration
            self.pre=encoder_init.decoding_(h_states, self.para.site_num)

        self.pre = self.pre * (self.std) + self.mean
        print('prediction values shape is : ', self.pre.shape)

        self.loss = mae_los(self.pre, self.placeholders['labels'][:,:,self.input_len:])
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        print('#...............................in the training step.....................................#')

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
                                                adj=None,
                                                placeholders=self.placeholders,
                                                sites=self.site_num)
                feed_dict.update({self.placeholders['dropout']: self.para.dropout})
                loss, _ = self.sess.run((self.loss, self.train_op), feed_dict=feed_dict)

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
                                            adj=None,
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
        # np.savez_compressed('data/STGIN-' + 'YINCHUAN', **{'prediction': pres_list, 'truth': labels_list})

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