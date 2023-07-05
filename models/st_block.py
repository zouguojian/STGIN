# -- coding: utf-8 --
from models.inits import *
from models.lstm import LstmClass, GRUClass
from models import tf_utils
from models.utils import *
from models.bridge import BridgeTransformer

class MultiHeadGATLayer(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 **kwargs):

        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.attn_kernel_initializer = attn_kernel_initializer

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.attn_kernel_regularizer = attn_kernel_regularizer
        self.activity_regularizer = activity_regularizer

        self.kernels = []
        self.biases = []
        self.atten_kernels = []

        super(MultiHeadGATLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        for head in range(self.attn_heads):
            kernel = self.add_weight(shape=(self.in_dim, self.out_dim),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     name='kernel_{}'.format(head))
            self.kernels.append(kernel)

            if self.use_bias:
                bias = self.add_weight(shape=(self.out_dim,),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            atten_kernel = self.add_weight(shape=(2 * self.out_dim, 1),
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           name='kernel_{}'.format(head))
            self.atten_kernels.append(atten_kernel)

        self.built = True

    def call(self, inputs, training):
        X = inputs[0]
        A = inputs[1]

        N = X.shape[1]
        dim = X.shape[2]

        outputs = []
        for head in range(self.attn_heads):

            kernel = self.kernels[head]

            features = tf.matmul(X, kernel)

            concat_features = tf.concat([tf.reshape(tf.tile(features, [1, 1, N]), [-1, N * N, dim]), tf.tile(features, [1, N, 1])], axis=-1)

            concat_features = tf.transpose(concat_features, [1, 0, 2 ]) # [N*N, -1, 2 * dim]

            atten_kernel = self.atten_kernels[head]

            dense = tf.matmul(concat_features, atten_kernel) # [N*N, -1, 1]

            dense = tf.keras.layers.LeakyReLU(alpha=0.2)(dense)

            attention = tf.reshape(dense, [N, N, -1]) # [N, N, -1]

            # zero_vec = -9e15 * tf.ones_like(dense)
            # attention = tf.where(A > 0, dense, zero_vec)

            dense = tf.keras.activations.softmax(attention, axis=-1)
            dense = tf.transpose(dense, [2, 0, 1])

            dropout_attn = tf.keras.layers.Dropout(self.dropout_rate)(dense, training=training)
            dropout_feat = tf.keras.layers.Dropout(self.dropout_rate)(features, training=training)

            node_features = tf.matmul(dropout_attn, dropout_feat)

            if self.use_bias:
                node_features = tf.add(node_features, self.biases[head])

            outputs.append(node_features)

        if self.attn_heads_reduction == 'concat':
            output = tf.concat(outputs, axis=-1)
        else:
            output = tf.reduce_mean(tf.stack(outputs), axis=-1)

        if self.activation is not None:
            output = self.activation(output)

        return output

def fusionGate(x, y):
    '''
    :param x: [-1, len, site, dim]
    :param y: [-1, len, site, dim]
    :return: [-1, len, site, dim]
    '''
    z = tf.nn.sigmoid(tf.multiply(x, y))
    h = tf.add(tf.multiply(z, x), tf.multiply(1 - z, y))
    return h

class ST_Block():
    def __init__(self, hp=None, placeholders=None, input_length=12, model_func=None):
        '''
        :param hp:
        '''
        self.para = hp
        self.batch_size = self.para.batch_size
        self.emb_size = self.para.emb_size
        self.site_num = self.para.site_num
        self.is_training = self.para.is_training
        self.dropout = self.para.dropout
        self.hidden_size = self.para.hidden_size
        self.hidden_layer =self.para.hidden_layer
        self.features = self.para.features
        self.placeholders = placeholders
        self.input_length = input_length
        self.num_heads = self.para.num_heads
        self.model_func = model_func

    def FC(self, x, units, activations, bn, bn_decay, is_training, use_bias=True):
        if isinstance(units, int):
            units = [units]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            activations = list(activations)
        assert type(units) == list
        for num_unit, activation in zip(units, activations):
            x = tf_utils.conv2d(
                x, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
                padding='VALID', use_bias=use_bias, activation=activation,
                bn=bn, bn_decay=bn_decay, is_training=is_training)
        return x

    def gatedFusion(self, HS, HT, D, bn, bn_decay, is_training):
        '''
        gated fusion
        HS:     [batch_size, num_step, N, D]
        HT:     [batch_size, num_step, N, D]
        D:      output dims
        return: [batch_size, num_step, N, D]
        '''
        XS = self.FC(
            HS, units=D, activations=None,
            bn=bn, bn_decay=bn_decay,
            is_training=is_training, use_bias=False)
        XT = self.FC(
            HT, units=D, activations=None,
            bn=bn, bn_decay=bn_decay,
            is_training=is_training, use_bias=True)
        z = tf.nn.sigmoid(tf.add(XS, XT))
        H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
        H = self.FC(
            H, units=[D, D], activations=[tf.nn.relu, None],
            bn=bn, bn_decay=bn_decay, is_training=is_training)
        return H

    def STEmbedding(self, SE, TE, T, D, bn, bn_decay, is_training):
        '''
        spatio-temporal embedding
        SE:     [N, D]
        TE:     [batch_size, P + Q, 2] (dayofweek, timeofday)
        T:      num of time steps in one day
        D:      output dims
        retrun: [batch_size, P + Q, N, D]
        '''
        # spatial embedding
        SE = self.FC(
            SE, units=[D, D], activations=[tf.nn.relu, None],
            bn=bn, bn_decay=bn_decay, is_training=is_training)
        # temporal embedding
        TE = tf.add_n(TE)
        # TE = tf.concat((TE), axis=-1)
        TE = self.FC(
            TE, units=[D, D], activations=[tf.nn.relu, None],
            bn=bn, bn_decay=bn_decay, is_training=is_training)
        return tf.add(SE, TE)

    def spatiotemporal(self, bn, bn_decay, is_training, speed=None, STE=None, supports=None,mask=True, speed_all=None, adj =None):
        X = speed
        X_ALL=speed_all
        for _ in range(self.para.num_blocks):
            HT = temporalAttention(X + X_ALL[:,:self.input_length] + STE, STE, self.num_heads, self.emb_size // self.num_heads, bn, bn_decay, is_training, mask=mask)

        XL = tf.transpose(X + X_ALL[:,:self.input_length], perm=[0, 2, 1, 3])
        XL = tf.reshape(XL, shape=[-1, self.input_length, self.emb_size])
        lstm_init = LstmClass(batch_size=self.batch_size * self.site_num,
                                layer_num=self.hidden_layer,
                                nodes=self.hidden_size,
                                placeholders=self.placeholders)
        XL, _ = lstm_init.encoding(XL)
        XL = tf.reshape(XL, shape=[self.batch_size, self.site_num, self.input_length, self.emb_size])
        XL = tf.transpose(XL, perm=[0, 2, 1, 3])
        HT = fusionGate(HT, XL)

        # GATs
        # HS = tf.reshape(X, shape=[-1, self.site_num, self.emb_size])
        # GAT =MultiHeadGATLayer(in_dim=self.emb_size,out_dim=self.emb_size)
        # HS = GAT([HS, adj], self.is_training)
        # HS = tf.reshape(HS, shape=[-1, self.input_length, self.site_num, self.emb_size])

        for _ in range(self.para.num_blocks):
            HS = spatialAttention(X + X_ALL[:,self.input_length:], STE, self.num_heads, self.emb_size // self.num_heads, bn, bn_decay, is_training)
        XS = tf.reshape(X + X_ALL[:,self.input_length:] + STE, shape=[-1, self.site_num, self.emb_size * 1])
        gcn = self.model_func(self.placeholders,
                                input_dim=self.emb_size * 1,
                                para=self.para,
                                supports=supports)
        XS = gcn.predict(XS)
        XS = tf.reshape(XS, shape=[-1, self.input_length, self.site_num, self.emb_size])
        HS = fusionGate(HS, XS)

        H = fusionGate(HS, HT)
        # H = gatedFusion(HS, HT, self.emb_size, bn, bn_decay, is_training)
        X = tf.add(X, H)
        return X

    def dynamic_decoding(self, hiddens=None, STE=None):
        X = hiddens
        y=[]
        X = tf.transpose(X, perm=[0, 2, 1, 3])
        X = tf.reshape(X, shape=[-1, self.input_length, self.emb_size])
        T = BridgeTransformer(self.para)
        temp = X[:,-1:]
        for time_step in range(self.para.output_length):
            temp = T.encoder(hiddens = X,
                            hidden = temp)
            y.append(temp)
        X = tf.reshape(tf.concat(y, axis=1), shape=[-1, self.para.site_num, self.para.output_length, self.emb_size])
        X = tf.transpose(X, perm=[0, 2, 1, 3])
        return X

    def spatiotemporal_(self, bn, bn_decay, is_training, speed=None, STE=None, supports=None, speed_all=None):
        X = speed
        for _ in range(self.para.num_blocks):
            X = STAttBlock(X + speed_all[:,:self.input_length], STE, self.para.num_heads, self.para.emb_size // self.para.num_heads, bn=bn, bn_decay=bn_decay, is_training=is_training)
        return X

def spatialAttention(X, STE, K, d, bn, bn_decay, is_training):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    D = K * d
    # [batch_size, num_step, N, K * d]
    query = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # [K * batch_size, num_step, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    # [K * batch_size, num_step, N, N]
    attention = tf.matmul(query, key, transpose_b=True)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis=-1)
    # [batch_size, num_step, N, D]
    X = tf.matmul(attention, value)
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X


def temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask=True):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    D = K * d
    query = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    key = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    value = FC(
        X, units=D, activations=tf.nn.relu,
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # [K * batch_size, num_step, N, d]
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    # query: [K * batch_size, N, num_step, d]
    # key:   [K * batch_size, N, d, num_step]
    # value: [K * batch_size, N, num_step, d]
    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))
    # [K * batch_size, N, num_step, num_step]
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    # mask attention score
    if mask:
        batch_size = tf.shape(X)[0]
        num_step = X.get_shape()[1].value
        N = X.get_shape()[2].value
        mask = tf.ones(shape=(num_step, num_step))
        mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
        mask = tf.expand_dims(tf.expand_dims(mask, axis=0), axis=0)
        mask = tf.tile(mask, multiples=(K * batch_size, N, 1, 1))
        mask = tf.cast(mask, dtype=tf.bool)
        attention = tf.compat.v2.where(
            condition=mask, x=attention, y=-2 ** 15 + 1)
    # softmax
    attention = tf.nn.softmax(attention, axis=-1)
    # [batch_size, num_step, N, D]
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return X

def gatedFusion(HS, HT, D, bn, bn_decay, is_training):
    '''
    gated fusion
    HS:     [batch_size, num_step, N, D]
    HT:     [batch_size, num_step, N, D]
    D:      output dims
    return: [batch_size, num_step, N, D]
    '''
    XS = FC(
        HS, units=D, activations=None,
        bn=bn, bn_decay=bn_decay,
        is_training=is_training, use_bias=False)
    XT = FC(
        HT, units=D, activations=None,
        bn=bn, bn_decay=bn_decay,
        is_training=is_training, use_bias=True)
    z = tf.nn.sigmoid(tf.add(XS, XT))
    H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
    H = FC(
        H, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    return H

def STAttBlock(X, STE, K, d, bn, bn_decay, is_training, mask=True):
    HS = spatialAttention(X, STE, K, d, bn, bn_decay, is_training)
    HT = temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask=mask)
    H = gatedFusion(HS, HT, K * d, bn, bn_decay, is_training)
    return tf.add(X, H)