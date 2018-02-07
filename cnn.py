# -*- coding: utf-8 -*-
# file: adjust_light_contrast.py
# author: Wang Kang
# time: 12/26/2017 15:57 PM
# ----------------------------------------------------------------
import time
import tensorflow as tf
import dataset
from datetime import timedelta

# 设置超参数

# Convolutional Layer 1.
filter_size1 = 3  # 卷积核尺寸
num_filters1 = 32  # 卷积核输出通道数（卷积核个数）

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

# Fully-connected layer.
fc_size = 128  # 全连接层卷积核输出通道数
num_channels = 3  # 图片通道数（做CNN输入使用）
img_size = 128  # 图片尺寸
img_size_flat = img_size * img_size * num_channels  # 图片flatten为1D向量
img_shape = (img_size, img_size)

classes = ['dog', 'cat']
num_classes = len(classes)  # 分类数量

# batch size
batch_size = 8  # 一次训练8个样本数
validation_size = .16  # 验证集比例
early_stopping = None  # # 在终止训练前，等待验证集失败后的等待时间

train_path = 'F:/catdog/train/'
test_path = 'F:/catdog/test/'
checkpoint_dir = "models/"

# ## Load Data
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images, test_ids = dataset.read_test_set(test_path, img_size)

print("Size of:")
print("Training_set:\t{}".format(len(data.train.labels)))  # 训练集样本数
print("Test_set:\t{}".format(len(test_images)))  # 测试集样本数
print("Validation_set:\t{}".format(len(data.valid.labels)))  # 验证集样本数

images, cls_true = data.train.images, data.train.cls  # 训练集上的图片和标签

# -----------------------数据预处理结束------------------------

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')  # img_size_flat = img_size * img_size * num_channels # EL输入
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])  # cnn的输入
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')  # label的真实值
y_true_cls = tf.argmax(y_true, 1)

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



# Convolutional Layer 1
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1,
                                            num_filters=num_filters1, use_pooling=True)

# Convolutional Layers 2 and 3
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1,
                                            filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)
layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2, num_input_channels=num_filters2,
                                            filter_size=filter_size3, num_filters=num_filters3, use_pooling=True)

# Flatten Layer
layer_flat, num_features = flatten_layer(layer_conv3)

# Fully-Connected Layer 1
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)

# Fully-Connected Layer 2
layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=True)

# y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(layer_fc2, 1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true))

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TensorFlow Run
session = tf.Session()
session.run(tf.global_variables_initializer())
train_batch_size = batch_size  # 样本数


def print_progress(epoch, feed_dict_train, train_loss, feed_dict_validate, val_loss):
    """
    每轮epoch后打印出acc、val_acc、val_loss
    :param epoch: epoch 轮数
    :param feed_dict_train: 格式为[num_train,mg_size * img_size * num_channels]
    :param feed_dict_validate: 格式为[num_train,mg_size * img_size * num_channels]
    :param val_loss:
    :return:epoch、train_acc、val_acc、val_loss
    """
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Train_acc:{1:>6.2%}, Train_loss:{3:.3f}, Val_acc:{2:>6.2%}, Val_loss:{3:.3f}"
    print(msg.format(epoch + 1, acc, train_loss, val_acc, val_loss))


# 执行许多优化迭代的功能，以便逐步改进网络层的变量。在每次迭代中，从训练集中选择了一批新的数据，然后TensorFlow使用这些训练样例来执行优化器。每一个epoch后就打印一次。
def new_optimize(epoch, batch_size):
    start_time = time.time()
    for i in range(epoch):
        for j in range(int(data.train.num_examples/batch_size)):

            x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
            x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)
            x_batch = x_batch.reshape(train_batch_size, img_size_flat)  # train_batch_size:样本数
            x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)
            feed_dict_train = {x: x_batch, y_true: y_true_batch}
            feed_dict_validate = {x: x_valid_batch, y_true: y_valid_batch}

            session.run(optimizer, feed_dict=feed_dict_train)

        val_loss = session.run(cost, feed_dict=feed_dict_validate)
        train_loss = session.run(cost, feed_dict=feed_dict_train)
        print_progress(i, feed_dict_train, train_loss, feed_dict_validate, val_loss)  # 每个epoch的最后一个batch_size 的feed_dict_train, feed_dict_validate值。

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
            if patience == early_stopping:
                break
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))


new_optimize(epoch=30, batch_size=2)
session.close()


