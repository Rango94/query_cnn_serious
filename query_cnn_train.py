import os
import query_cnn_infernece
import tensorflow as tf
from data_helper import data_helper


MOVING_AVERAGE_DECAY=0.99

BATCH_SIZE=100



LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99

MODEL_SAVE_PATH='/model/'
MODEL_NAME='model'
REGULARIZER_RATE=0.0001

def train(dh):
    x=tf.placeholder(tf.float32, [None, query_cnn_infernece.FEATURE_SHAPE, query_cnn_infernece.FEATURE_SHAPE, query_cnn_infernece.NUM_CHANNELS], name='x-input')
    y_=tf.placeholder(tf.float32, [query_cnn_infernece.NUM_LABELS], name='y-input')
    Regularzer=tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)
    y=query_cnn_infernece.inference(x, True, Regularzer)
    global_step=tf.Variable(0,trainable=False)

    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())

    cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,dh.train_file_num*5000/BATCH_SIZE,LEARNING_RATE_DECAY)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='train')

    saver=tf.train.Saver()

    TRAINING_STEPS = 30000

    dh.set_batch_size(BATCH_SIZE)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(TRAINING_STEPS):
            xs,ys=dh.next_batch()
            xs=tf.reshape(xs, [BATCH_SIZE, query_cnn_infernece.FEATURE_SHAPE, query_cnn_infernece.FEATURE_SHAPE, query_cnn_infernece.NUM_CHANNELS])
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%1000==0:
                print('After %d training steps, loss on training batch is %g.' %(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main(argv=None):
    dh=data_helper()
    train(dh)
    #1

tf.app.run()







