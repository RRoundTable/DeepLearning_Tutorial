import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# Hyperparameter
growth_k = 24
nb_block = 2 # how many (dense block + Transition Layer) ?
init_learning_rate = 1e-4
epsilon = 1e-4 # AdamOptimizer epsilon
dropout_rate = 0.2

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

# Label & batch_size
batch_size = 64
class_num=10
iteration = 782
# batch_size * iteration = data_set_number

test_iteration = 10

total_epochs = 300

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    """
    :param input: input
    :param filter: filter 개수, channel
    :param kernel: kernel size
    :return: conv
    """
    with tf.name_scope(layer_name):
        network=tf.layers.conv2d(inputs=input,use_bias=False,filters=filter,kernel_size=kernel, strides=stride,padding="SAME")
        return network

def Gloval_Average_Pooing(x, stride=1):
    """
       width = np.shape(x)[1]
       height = np.shape(x)[2]
       pool_size = [width, height] # 한번에 pooling한다
       return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter 1*1*dim
       It is global average pooling without tflearn
       """
    return global_avg_pool(x, name="Global_avg_pooling")


def Batch_Normalization(x,training,scope):
    """Stores the default arguments for the given set of list_ops."""
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):

        """
        condition에 따라 다른 함수 실행
        training == True:  accumulate the statistics of the moments into moving_mean and moving_variance 
        training == False : use the values of the moving_mean and the moving_variance.
        """
        return tf.cond(training, lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda :batch_norm(inputs=x,is_training=training,reuse=True))


def Drop_out(x, rate,training):
    return tf.layers.dropout(inputs=x,rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Avergae_pooling(x,pool_size=[2,2],stride=2,padding="VALID"):
    return tf.layers.average_pooling2d(inputs=x,pool_size=pool_size,padding=padding,strides=stride)


def Max_pooling(x,pool_size=[2,2],stride=2,padding="VALID"):
    return tf.layers.max_pooling2d(inputs=x,pool_size=pool_size,strides=stride,padding=padding)

def Concatenation(layers):
    """
    axis 3 : (batch, channels, height, width)
    """
    return tf.concat(layers, axis=3)

def Linear(x):
    return tf.layers.dense(x,units=class_num, name="linear")

def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 1000

    for it in range(test_iteration):
        test_batch_x=test_x[test_pre_index:test_pre_index+add]
        test_batch_y=test_y[test_pre_index:test_pre_index+add]
        test_pre_index=test_pre_index+add

        test_feed_dict={
            x:test_batch_x,
            label:test_batch_y,
            learning_rate :epoch_learning_rate,
            training_flag:False
        }

        loss_, acc_= sess.run([cost,accuracy], feed_dict=test_feed_dict)

        test_loss+=loss_/10.0
        test_acc+=acc_/10.0
    """
    tf.Summary : tensorborad에서 시각화 할때 필요하다
    """
    summary=tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                              tf.Summary.Value(tag="test_Accuracy", simple_value=test_acc)])



class DenseNet:
    def __init__(self,x,nb_blocks,filters,training):
        self.filters=filters
        self.nb_block=nb_block
        self.training=training
        self.model=self.densenet(x)


    def transition_layer(self,x,scope):
        """
        1*1 conv
        2*2 average pool, stride 2

        :return: transition layer
        """
        with tf.name_scope(scope):

            x=Batch_Normalization(x,training=self.training,scope=scope+"b1")
            x=Relu(x)
            x=conv_layer(x,filter=self.filters,kernel=[1,1],stride=1,layer_name=scope+"conv1")
            x=Drop_out(x,rate=dropout_rate,training=self.training)
            x=Avergae_pooling(x,pool_size=[2,2],stride=2)


        return x

    def bottle_neck(self, inputs, layer_name):
        """
        the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)
        """
        H=Batch_Normalization(inputs,self.training,scope=layer_name+"b1")
        H=Relu(H)
        H=conv_layer(H,filter=self.filters, kernel=[1,1],layer_name=layer_name+"conv1")
        H=Drop_out(H,rate=dropout_rate,training=self.training)

        H=Batch_Normalization(H,self.training,scope=layer_name+"b2")
        H=Relu(H)
        H=conv_layer(H,filter=self.filters,kernel=[3,3],layer_name=layer_name+"conv2") # reduce the number of feature-maps (차원이 아니라 크기를 의미한다)
        H=Drop_out(H, rate=dropout_rate,training=self.training)

        return H

    def dense_block(self,inputs,nb_layer,layer_name):
        """
        x` = H`([x0, x1, . . . , x`−1])
        """
        with tf.name_scope(layer_name):

            layer_concat=[] # input의 변화 : 점진적으로 증가한다
            x=self.bottle_neck(inputs=inputs, layer_name=layer_name+"bottleN"+str(0))
            layer_concat.append(x)

            for i in range(nb_layer-1):
                x=Concatenation(layer_concat)
                x=self.bottle_neck(x,layer_name=layer_name+"bottleN"+str(i+1))
                layer_concat.append(x)

        x=Concatenation(layer_concat)

        return x

    def densenet(self, input_x):
        x=conv_layer(input_x,filter=self.filters, kernel=[7,7], stride=2,layer_name="conv0")
        x=Max_pooling(x,pool_size=[3,3],stride=2)

        for i in range(self.nb_block): # dense block 만들기

            x=self.dense_block(x,nb_layer=4,layer_name="dense"+str(i))
            x=self.transition_layer(x,scope="trans"+str(i))

        x=self.dense_block(x,nb_layer=32,layer_name="dense_final")

        # 100 layer
        x=Batch_Normalization(x,training=self.training,scope="linear_batch")
        x=Relu(x)
        x=Gloval_Average_Pooing(x)
        x=flatten(x)
        x=Linear(x)

        return x


x = tf.placeholder(tf.float32, shape=[None, 784])
batch_images = tf.reshape(x, [-1, 28, 28, 1])

label = tf.placeholder(tf.float32, shape=[None, 10])

training_flag = tf.placeholder(tf.bool)


learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = DenseNet(x=batch_images, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
train = optimizer.minimize(cost)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)

    global_step = 0
    epoch_learning_rate = init_learning_rate
    for epoch in range(total_epochs):
        if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10

        total_batch = int(mnist.train.num_examples / batch_size)

        for step in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag : True
            }

            _, loss = sess.run([train, cost], feed_dict=train_feed_dict)

            if step % 100 == 0:
                global_step += 100
                train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=train_feed_dict)
                # accuracy.eval(feed_dict=feed_dict)
                print("Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)
                writer.add_summary(train_summary, global_step=epoch)

            test_feed_dict = {
                x: mnist.test.images,
                label: mnist.test.labels,
                learning_rate: epoch_learning_rate,
                training_flag : False
            }

        accuracy_rates = sess.run(accuracy, feed_dict=test_feed_dict)
        print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy =', accuracy_rates)
        # writer.add_summary(test_summary, global_step=epoch)

    saver.save(sess=sess, save_path='./model/dense.ckpt')