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


img_size_flat = 256 * 256 * 3  # 图片flatten为1D向量

classes = ['dog', 'cat']
num_classes = len(classes)  # 分类数量
batch_size = None
early_stopping = None  # # 在终止训练前，等待验证集失败后的等待时间

train_path = 'F:/catdog/new_test/'
test_path = 'F:/catdog/test/'
checkpoint_dir = "models/"

# ## Load Data
train_images, train_labels, validation_images, validation_labels = read_train_sets(train_path, 256, 256, classes, validation_size=0.005)

# print(validation_images.shape,train_images.shape,validation_labels.shape,train_labels.shape)
test_images, test_ids = read_test_set(test_path, weight_size=256, hight_size=256)
num_examples = train_images.shape[0]

print("Size of:")
print("Training_set:\t{}".format(num_examples))  # 训练集样本数
print("Test_set:\t{}".format(len(test_images)))  # 测试集样本数
print("Validation_set:\t{}".format(validation_images.shape[0]))  # 验证集样本数

# -----------------------数据预处理结束------------------------

x = tf.placeholder(tf.float32, shape=[None, img_size_flat],
                   name='x')  # img_size_flat = img_size * img_size * num_channels # EL输入
x_image = tf.reshape(x, [-1, 256, 256, 3])  # cnn的输入
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')  # label的真实值
y_true_label = tf.argmax(y_true, 1)


def new_weights(shape):  # 权重
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def new_biases(length):  # 偏差
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1],
                         padding='SAME')  # 为卷积创建一个TensorFlow操作。注意，所有维度上的步幅都设置为1，填充设置为"SAME"也就是输入图像用0填充，所以输出的大小是一样的。
    layer += biases  # 把bias偏差加到卷积的结果中。每个过滤通道都添加了一个bias值。
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights


def flatten_layer(layer):  # 卷积层产生一个带有4维的输出张量。我们将在卷积层之后添加全连通的层，因此我们需要将4D张量减少为2D，这可以作为全连通层的输入。
    layer_shape = layer.get_shape()  # layer_shape == [num_images, img_height, img_width, num_channels]
    num_features = layer_shape[1:4].num_elements()  # features数量: img_height * img_width * num_channels
    layer_flat = tf.reshape(layer, [-1, num_features])  # Reshape the layer to [num_images, num_features].
    return layer_flat, num_features


def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):  # 输入是input = layer_flat, num_inputs = num_features
    weights = new_weights(shape=[num_inputs, num_outputs])  # Create new weights and biases.
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=3, filter_size=3,
                                            num_filters=64, use_pooling=False)

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=64,
                                            filter_size=3, num_filters=64, use_pooling=True)

layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2, num_input_channels=64,
                                            filter_size=3, num_filters=128, use_pooling=False)

layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3, num_input_channels=128,
                                            filter_size=3, num_filters=128, use_pooling=True)

layer_conv5, weights_conv5 = new_conv_layer(input=layer_conv4, num_input_channels=128,
                                            filter_size=3, num_filters=256, use_pooling=False)

layer_conv6, weights_conv6 = new_conv_layer(input=layer_conv5, num_input_channels=256,
                                            filter_size=3, num_filters=256, use_pooling=True)

layer_conv7, weights_conv7 = new_conv_layer(input=layer_conv6, num_input_channels=256,
                                            filter_size=3, num_filters=512, use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv7)

layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=512, use_relu=True)

layer_fc1_drop = tf.nn.dropout(layer_fc1, keep_prob=0.5)

layer_fc2 = new_fc_layer(input=layer_fc1_drop, num_inputs=512, num_outputs=2, use_relu=True)

y_pred_label = tf.argmax(layer_fc2, 1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_label, y_true_label)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
            session.run(optimizer, feed_dict={x_image: train_images[start:end], y_true: train_labels[start:end]})

            # 每训练一组batch_size数据后，输出 batch_acc\val等信息
            batch_train_acc = session.run(accuracy,
                                          feed_dict={x_image: train_images[start:end], y_true: train_labels[start:end]})
            batch_train_loss = session.run(cost, feed_dict={x_image: train_images[start:end],
                                                            y_true: train_labels[start:end]})
            batch_val_acc = session.run(accuracy,
                                        feed_dict={x_image: validation_images, y_true: validation_labels})
            batch_val_loss = session.run(cost,
                                         feed_dict={x_image: validation_images, y_true: validation_labels})

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
    saver = tf.train.Saver()
    save_path = saver.save(session, 'F:/model/model.ckpt')
    print("保存到：", save_path)
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))


optimize(epoch=20, batch_size=2)
session.close()
