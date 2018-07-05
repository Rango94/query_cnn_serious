import tensorflow as tf

NUM_CHANNELS=1

WORD_LEN=200
FEATURE_SHAPE=[12,3665]

CONV1_DEEP=1
CONV1_WIDE=WORD_LEN
CONV1_HIGH=1

CONV2_DEEP=1
CONV2_WIDE=WORD_LEN
CONV2_HIGH=2

CONV3_DEEP=1
CONV3_WIDE=WORD_LEN
CONV3_HIGH=3

CONV4_DEEP=1
CONV4_WIDE=WORD_LEN
CONV4_HIGH=4

FC_SIZ=24
NUM_LABELS=21





def inference(input_tensor,train,regularizer):
    #embedding层，目前定义的输入数据shape是[batch_size,12,3665]
    #需要将其映射为[batch_size,12,200]
    #由于cnn的特性，我们需要将它reshape成[batch_size,12,200,1]
    with tf.variable_scope('layer0-embedding'):
        input_tensor = tf.reshape(input_tensor, [-1, FEATURE_SHAPE[1]])
        embedding_weights = tf.get_variable('weight', [FEATURE_SHAPE[1], WORD_LEN],
                                            initializer=tf.random_uniform_initializer(maxval=0.5 / WORD_LEN,
                                                                                      minval=-0.5 / WORD_LEN))
        embedding_baises = tf.get_variable('bais', [WORD_LEN], initializer=tf.constant_initializer(0.0))
        embedding = tf.matmul(input_tensor, embedding_weights) + embedding_baises
        embedding = tf.reshape(embedding, [-1, 12, WORD_LEN, NUM_CHANNELS])

    with tf.variable_scope('layer1-conv1'):
        conv1_weights=tf.get_variable('wegiht',[CONV1_WIDE,CONV1_HIGH,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable('bais',[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv1=tf.nn.conv2d(embedding,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.variable_scope('layer1-conv2'):
        conv2_weights=tf.get_variable('wegiht',[CONV1_WIDE,CONV1_HIGH,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases=tf.get_variable('bais',[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv2=tf.nn.conv2d(embedding,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    with tf.variable_scope('layer1-conv3'):
        conv3_weights=tf.get_variable('wegiht',[CONV1_WIDE,CONV1_HIGH,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases=tf.get_variable('bais',[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv3=tf.nn.conv2d(embedding,conv3_weights,strides=[1,1,1,1],padding='SAME')
        relu3=tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))

    with tf.variable_scope('layer1-conv4'):
        conv4_weights=tf.get_variable('wegiht',[CONV1_WIDE,CONV1_HIGH,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases=tf.get_variable('bais',[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv4=tf.nn.conv2d(embedding,conv4_weights,strides=[1,1,1,1],padding='SAME')
        relu4=tf.nn.relu(tf.nn.bias_add(conv4,conv4_biases))

    with tf.name_scope('layer2-pool1'):
        conv1_shape=relu1.get_shape().as_list()
        pool1=tf.nn.max_pool(relu1,ksize=[1,conv1_shape[0],conv1_shape[1],1],strides=[1,2,2,1],padding='SAME')

    with tf.name_scope('layer2-pool2'):
        conv2_shape = relu1.get_shape().as_list()
        pool2=tf.nn.max_pool(relu2,ksize=[1,conv2_shape[0],conv2_shape[1],1],strides=[1,2,2,1],padding='SAME')

    with tf.name_scope('layer2-pool3'):
        conv3_shape = relu1.get_shape().as_list()
        pool3=tf.nn.max_pool(relu3,ksize=[1,conv3_shape[0],conv3_shape[1],1],strides=[1,2,2,1],padding='SAME')

    with tf.name_scope('layer2-pool4'):
        conv4_shape = relu1.get_shape().as_list()
        pool4=tf.nn.max_pool(relu4,ksize=[1,conv4_shape[0],conv4_shape[1],1],strides=[1,2,2,1],padding='SAME')



    pool_shape=pool1.get_shape().as_list()
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped1=tf.reshape(pool1,[pool_shape[0],nodes])

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped2 = tf.reshape(pool2, [pool_shape[0], nodes])

    pool_shape = pool3.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped3 = tf.reshape(pool3, [pool_shape[0], nodes])

    pool_shape = pool4.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped4 = tf.reshape(pool4, [pool_shape[0], nodes])

    reshaped=tf.concat([reshaped1,reshaped2],1)
    reshaped=tf.concat([reshaped,reshaped3],1)
    reshaped=tf.concat([reshaped,reshaped4],1)

    with tf.variable_scope('layer5,fc1'):
        fc1_weights=tf.get_variable('weight',[nodes,FC_SIZ],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases=tf.get_variable('bias',[FC_SIZ],initializer=tf.constant_initializer(0.1))
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train:fc1=tf.nn.dropout(fc1,0.5)

    with tf.variable_scope('layer6,fc2'):
        fc2_weights=tf.get_variable('weight',[FC_SIZ,NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases=tf.get_variable('bias',[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit=tf.matmul(fc1,fc2_weights)+fc2_biases

    return logit








