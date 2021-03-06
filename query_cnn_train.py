import os
import query_cnn_infernece
import tensorflow as tf
from data_helper import data_helper


MOVING_AVERAGE_DECAY=0.99

BATCH_SIZE=100



LEARNING_RATE_BASE=1.0
LEARNING_RATE_DECAY=0.99

MODEL_SAVE_PATH='/model/'
MODEL_NAME='model'
REGULARIZER_RATE=0.001

def train(dh):
    x=tf.placeholder(tf.float32, [None, query_cnn_infernece.FEATURE_SHAPE[0], query_cnn_infernece.FEATURE_SHAPE[1]], name='x-input')
    y_=tf.placeholder(tf.float32, [None,query_cnn_infernece.NUM_LABELS], name='y-input')
    Regularzer=tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)
    y=query_cnn_infernece.inference(x, True, Regularzer)
    global_step=tf.Variable(0,trainable=False)

    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())

    cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=y,labels=y_)
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,dh.train_file_num*5000/BATCH_SIZE,LEARNING_RATE_DECAY)
    train_step=tf.train.AdadeltaOptimizer(learning_rate).minimize(loss,global_step=global_step)

    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='train')

    # saver=tf.train.Saver()

    TRAINING_STEPS = 30000

    dh.set_batch_size(BATCH_SIZE)


    X_test,Y_test=dh.get_test_data()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(TRAINING_STEPS):
            xs,ys=dh.next_batch()
            # print(sess.run(y,feed_dict={x:xs,y_:ys}))
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%10==0:
                print('After %d training steps, loss on training batch is %g.' %(step,loss_value))
                # saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
                y_out=sess.run(y,feed_dict={x:X_test,y_:Y_test}).tolist()
                # print(y_out[0],y_out[10],y_out[50])
                y_out_r=Y_test.tolist()
                r=0
                tt=0
                for idx,i in enumerate(y_out):
                    if i.index(max(i))==y_out_r[idx].index(max(y_out_r[idx])):
                        r+=1
                    tt+=1
                print(r,tt,r/tt)
def main(argv=None):
    dh=data_helper()
    train(dh)
    #1

tf.app.run()







