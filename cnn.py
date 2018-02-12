# -*- coding: utf-8 -*-
# file: adjust_light_contrast.py
# author: Wang Kang
# time: 12/26/2017 15:57 PM
# ----------------------------------------------------------------
import time
import tensorflow as tf
from datetime import timedelta
import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_train(train_path, weight_size, hight_size, classes):
    """
    load训练数据
    :param train_path: 训练数据路径
    :param image_size: 图片需要resize的尺寸
    :param classes: 图片类别（数组）
    :return: 图片、标签
    """
    images = []
    labels = []
    print('Reading training images......')
    for fld in classes:  # 假设数据目录的每个类都有一个单独的文件夹，每个文件夹都以类命名
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (weight_size, hight_size), cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(classes))  # 相当于做了一次one-hot
            label[index] = 1.0  # 相当于做了一次one-hot
            labels.append(label)  # 相当于做了一次one-hot

    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def load_test(test_path, weight_size, hight_size):
    path = os.path.join(test_path, '*g')
    files = sorted(glob.glob(path))
    X_test = []
    X_test_id = []
    print("Reading test images")
    for fl in files:
        flbase = os.path.basename(fl)
        img = cv2.imread(fl)
        img = cv2.resize(img, (weight_size, hight_size), cv2.INTER_LINEAR)
        X_test.append(img)
        X_test_id.append(flbase)

    X_test = np.array(X_test, dtype=np.uint8)
    X_test = X_test.astype('float32')
    return X_test, X_test_id


def read_train_sets(train_path, weight_size, hight_size, classes, validation_size):
    images, labels = load_train(train_path, weight_size, hight_size, classes)
    images, labels = shuffle(images, labels, )

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]

    return train_images, train_labels, validation_images, validation_labels


def read_test_set(test_path, weight_size, hight_size, ):
    images, ids = load_test(test_path, weight_size, hight_size)
    return images, ids


img_size_flat = 224 * 224 * 3  # 图片flatten为1D向量

classes = ['dog', 'cat']
num_classes = len(classes)  # 分类数量
batch_size = None
early_stopping = None  # # 在终止训练前，等待验证集失败后的等待时间

train_path = 'F:/catdog/new_test/'
test_path = 'F:/catdog/test/'
checkpoint_dir = "models/"

# ## Load Data
train_images, train_labels, validation_images, validation_labels = read_train_sets(train_path, 224, 224, classes,
                                                                                   validation_size=0.01)

# print(validation_images.shape,train_images.shape,validation_labels.shape,train_labels.shape)
test_images, test_ids = read_test_set(test_path, weight_size=224, hight_size=224)
num_examples = train_images.shape[0]

print("Size of:")
print("Training_set:\t{}".format(num_examples))  # 训练集样本数
print("Test_set:\t{}".format(len(test_images)))  # 测试集样本数
print("Validation_set:\t{}".format(validation_images.shape[0]))  # 验证集样本数

# -----------------------数据预处理结束------------------------
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat],
                       name='x')  # img_size_flat = img_size * img_size * num_channels # EL输入

    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')  # label的真实值

with tf.name_scope('input_reshape'):
    x_image = tf.reshape(x, [-1, 224, 224, 3])  # cnn的输入
    tf.summary.image('input', x_image, 10)

y_true_label = tf.argmax(y_true, 1)


def weight_variable(shape, name="weights"):  # 初始化权重
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)  # 不为标准正态分布\截断的正态分布噪声
    return tf.Variable(initial, name=name)


def bias_variable(shape, name="biases"):  # 初始化偏置项 b
    initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial, name=name)


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def conv2d(input, w):  # 卷积项，w为卷积核
    tf.summary.histogram('weight&bias', w)
    return tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME')


def pool_max(input):  # 最大池化项
    return tf.nn.max_pool(input,
                          ksize=[1, 2, 2, 1],  # 池化层尺寸
                          strides=[1, 2, 2, 1],  # 步长尺寸
                          padding='SAME',
                          )


def fc(input, w, b):  # 全连接层
    return tf.matmul(input, w) + b



# conv1
with tf.name_scope('conv1_1') as scope:
    kernel = weight_variable([3, 3, 3, 64])  # 重点关注第一层卷积。卷积核为3*3,输入channels = 1(输入通道数),卷集核个数（输出通道数）为64。
    biases = bias_variable([64])
    variable_summaries(kernel)
    variable_summaries(biases)
    output_conv1_1 = tf.nn.relu(conv2d(x_image, kernel) + biases, name=scope)
    tf.summary.histogram('output_conv1_1', output_conv1_1)

with tf.name_scope('conv1_2') as scope:
    kernel = weight_variable([3, 3, 64, 64])
    biases = bias_variable([64])
    variable_summaries(kernel)
    variable_summaries(biases)
    output_conv1_2 = tf.nn.relu(conv2d(output_conv1_1, kernel) + biases, name=scope)
    tf.summary.histogram('output_conv1_2', output_conv1_2)

pool1 = pool_max(output_conv1_2)

# conv2
with tf.name_scope('conv2_1') as scope:
    kernel = weight_variable([3, 3, 64, 128])
    biases = bias_variable([128])
    variable_summaries(kernel)
    variable_summaries(biases)
    output_conv2_1 = tf.nn.relu(conv2d(pool1, kernel) + biases, name=scope)
    tf.summary.histogram('output_conv2_1', output_conv2_1)

with tf.name_scope('conv2_2') as scope:
    kernel = weight_variable([3, 3, 128, 128])
    biases = bias_variable([128])
    variable_summaries(kernel)
    variable_summaries(biases)
    output_conv2_2 = tf.nn.relu(conv2d(output_conv2_1, kernel) + biases, name=scope)
    tf.summary.histogram('output_conv2_2', output_conv2_2)

pool2 = pool_max(output_conv2_2)

# conv3
with tf.name_scope('conv3_1') as scope:
    kernel = weight_variable([3, 3, 128, 256])
    biases = bias_variable([256])
    variable_summaries(kernel)
    variable_summaries(biases)
    output_conv3_1 = tf.nn.relu(conv2d(pool2, kernel) + biases, name=scope)
    tf.summary.histogram('output_conv3_1', output_conv3_1)

with tf.name_scope('conv3_2') as scope:
    kernel = weight_variable([3, 3, 256, 256])
    biases = bias_variable([256])
    variable_summaries(kernel)
    variable_summaries(biases)
    output_conv3_2 = tf.nn.relu(conv2d(output_conv3_1, kernel) + biases, name=scope)
    tf.summary.histogram('output_conv3_2', output_conv3_2)

with tf.name_scope('conv3_3') as scope:
    kernel = weight_variable([3, 3, 256, 256])
    biases = bias_variable([256])
    variable_summaries(kernel)
    variable_summaries(biases)
    output_conv3_3 = tf.nn.relu(conv2d(output_conv3_2, kernel) + biases, name=scope)
    tf.summary.histogram('output_conv3_3', output_conv3_3)

pool3 = pool_max(output_conv3_3)

# conv4
with tf.name_scope('conv4_1') as scope:
    kernel = weight_variable([3, 3, 256, 512])
    biases = bias_variable([512])
    variable_summaries(kernel)
    variable_summaries(biases)
    output_conv4_1 = tf.nn.relu(conv2d(pool3, kernel) + biases, name=scope)
    tf.summary.histogram('output_conv4_1', output_conv4_1)

with tf.name_scope('conv4_2') as scope:
    kernel = weight_variable([3, 3, 512, 512])
    biases = bias_variable([512])
    variable_summaries(kernel)
    variable_summaries(biases)
    output_conv4_2 = tf.nn.relu(conv2d(output_conv4_1, kernel) + biases, name=scope)

with tf.name_scope('conv4_3') as scope:
    kernel = weight_variable([3, 3, 512, 512])
    biases = bias_variable([512])
    variable_summaries(kernel)
    variable_summaries(biases)
    output_conv4_3 = tf.nn.relu(conv2d(output_conv4_2, kernel) + biases, name=scope)

pool4 = pool_max(output_conv4_3)

# conv5
with tf.name_scope('conv5_1') as scope:
    kernel = weight_variable([3, 3, 512, 512])
    biases = bias_variable([512])
    variable_summaries(kernel)
    variable_summaries(biases)
    output_conv5_1 = tf.nn.relu(conv2d(pool4, kernel) + biases, name=scope)

with tf.name_scope('conv5_2') as scope:
    kernel = weight_variable([3, 3, 512, 512])
    biases = bias_variable([512])
    variable_summaries(kernel)
    variable_summaries(biases)
    output_conv5_2 = tf.nn.relu(conv2d(output_conv5_1, kernel) + biases, name=scope)

with tf.name_scope('conv5_3') as scope:
    kernel = weight_variable([3, 3, 512, 512])
    biases = bias_variable([512])
    variable_summaries(kernel)
    variable_summaries(biases)
    output_conv5_3 = tf.nn.relu(conv2d(output_conv5_2, kernel) + biases, name=scope)

pool5 = pool_max(output_conv5_3)

# fc6
with tf.name_scope('fc6') as scope:
    shape = int(np.prod(pool5.get_shape()[1:]))
    pool5_flat = tf.reshape(pool5, [-1, shape])
    kernel = weight_variable([shape, 1024])
    biases = bias_variable([1024])
    variable_summaries(kernel)
    variable_summaries(biases)
    output_fc6 = tf.nn.relu(fc(pool5_flat, kernel, biases), name=scope)

# drop操作
keep_prob = tf.placeholder(tf.float32)
output_fc6_drop = tf.nn.dropout(output_fc6, keep_prob)

# fc7
with tf.name_scope('fc7') as scope:
    kernel = weight_variable([1024, 1024])
    biases = bias_variable([1024])
    variable_summaries(kernel)
    variable_summaries(biases)
    output_fc7 = tf.nn.relu(fc(output_fc6_drop, kernel, biases), name=scope)

# drop操作
output_fc7_drop = tf.nn.dropout(output_fc7, keep_prob)

# fc8
with tf.name_scope('fc8') as scope:
    kernel = weight_variable([1024, 2])  # 重点关注最后一个全连接层，输入为1024个通道数、输出为4个（4分类）
    biases = bias_variable([2])
    variable_summaries(kernel)
    variable_summaries(biases)
    output_fc8 = tf.nn.relu(fc(output_fc7_drop, kernel, biases), name=scope)

new_output_fc8 = tf.nn.softmax(output_fc8)
y_pred_label = tf.argmax(new_output_fc8, 1)

with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_fc8, labels=y_true))

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(y_pred_label, y_true_label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

session = tf.Session()
session.run(tf.global_variables_initializer())


# 执行许多优化迭代的功能，以便逐步改进网络层的变量。在每次迭代中，从训练集中选择了一批新的数据，然后TensorFlow使用这些训练样例来执行优化器。每一个epoch后就打印一次。
def optimize(epoch, batch_size):
    start_time = time.time()
    for i in range(epoch):
        all_batch_train_acc = 0
        all_batch_train_loss = 0
        all_batch_val_acc = 0
        all_batch_val_loss = 0
        for j in range(0, int(num_examples / batch_size)):
            start = j * batch_size
            end = start + batch_size
            session.run(optimizer,
                        feed_dict={x_image: train_images[start:end], y_true: train_labels[start:end], keep_prob: 0.5})

            # 每训练一组batch_size数据后，输出 batch_acc\val等信息
            batch_train_acc = session.run(accuracy,
                                          feed_dict={x_image: train_images[start:end], y_true: train_labels[start:end], keep_prob: 1.0})
            batch_train_loss = session.run(cost, feed_dict={x_image: train_images[start:end],
                                                            y_true: train_labels[start:end], keep_prob: 1.0})
            batch_val_acc = session.run(accuracy,
                                        feed_dict={x_image: validation_images, y_true: validation_labels, keep_prob: 1.0})
            batch_val_loss = session.run(cost,
                                         feed_dict={x_image: validation_images, y_true: validation_labels, keep_prob: 1.0})

            batch_output_string = "batch %s : Train_acc:%.3f, Train_loss:%.3f, Val_acc:%.3f, Val_loss:%.3f"
            print(batch_output_string % (j + 1, batch_train_acc, batch_train_loss, batch_val_acc,
                                         batch_val_loss))  # 输出每batch_size个样本后的train_acc、train_loss、val_acc、val_loss

            all_batch_train_acc += batch_train_acc
            all_batch_train_loss += batch_train_loss
            all_batch_val_acc += batch_val_acc
            all_batch_val_loss += batch_val_loss
        '''
        # 每一轮epoch后输出 acc等（数据来源于这一轮epoch的最后一个batch_szie）
        acc = session.run(accuracy, feed_dict={x_image: train_images[start:end], y_true: train_labels[start:end]})
        train_loss = session.run(cost, feed_dict={x_image: train_images[start:end], y_true: train_labels[start:end]})

        val_acc = session.run(accuracy, feed_dict={x_image: validation_images[:5], y_true: validation_labels[:5]})
        val_loss = session.run(cost, feed_dict={x_image: validation_images[:5], y_true: validation_labels[:5]})
        '''
        # print(all_batch_train_acc, all_batch_train_loss, all_batch_val_acc, all_batch_val_loss)
        # 每一轮epoch后输出平均acc等（数据来源于这一轮epoch的最后一个batch_szie
        avg_train_acc = all_batch_train_acc / int(num_examples / batch_size)
        avg_train_loss = all_batch_train_loss / int(num_examples / batch_size)
        avg_val_acc = all_batch_val_acc / int(num_examples / batch_size)
        avg_val_loss = all_batch_val_loss / int(num_examples / batch_size)
        output_string = "Epoch %s >>> Train_acc:%.3f, Train_loss:%.3f, Val_acc:%.3f, Val_loss:%.3f <<<"
        print(output_string % (i + 1, avg_train_acc, avg_train_loss, avg_val_acc, avg_val_loss))
        print('\r')

        '''
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
            if patience == early_stopping:
                break
        '''
    end_time = time.time()
    time_dif = end_time - start_time
    # print("保存到：", save_path)
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))

    writer = tf.summary.FileWriter(r'C:\Users\Administrator\tf', tf.get_default_graph())
    writer.close()
optimize(epoch=2, batch_size=4)
session.close()
