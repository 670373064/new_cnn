# coding: utf-8

# # TensorFlow Convolutional Neural Network for Image Classification
import time
import math
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import dataset
import cv2

from sklearn.metrics import confusion_matrix
from datetime import timedelta

# ## 配置和超参数设置

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
fc_size = 128  # Number of neurons in fully-connected layer.  全连接层卷积核输出通道数

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3  # 图片通道数（做CNN输入使用）

img_size = 128  # 图片尺寸
img_size_flat = img_size * img_size * num_channels  # 图片flatten为1D向量
img_shape = (img_size, img_size)

# class info
classes = ['dog', 'cat']
num_classes = len(classes)  # 分类数量

# batch size
batch_size = 32  # 一次训练32个样本数

# validation split
validation_size = .16  # 验证集比例

# 在终止训练前，等待验证集失败后的等待时间
early_stopping = None  # use None if you don't want to implement early stopping

train_path = 'E:/catdog/train/'
test_path = 'E:/catdog/test/'
checkpoint_dir = "models/"

# ## Load Data
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images, test_ids = dataset.read_test_set(test_path, img_size)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))  # 训练集样本数
print("- Test-set:\t\t{}".format(len(test_images)))  # 测试集样本数
print("- Validation-set:\t{}".format(len(data.valid.labels)))  # 验证集样本数


# ### Helper-function for plotting images

# 从训练集上得到一些随机的图片和标签

images, cls_true = data.train.images, data.train.cls
# -----------------------数据预处理结束-------------------------


# ## TensorFlow Graph

# ### Helper-functions for creating new variables
# 在给定的形状中创建新的TensorFlow变量并以随机值初始化它们的函数。注意，初始化实际上并没有完成，它仅仅是在TensorFlow图中定义了。

def new_weights(shape):  # 权重
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):  # 偏差
    return tf.Variable(tf.constant(0.05, shape=[length]))


# ### Helper-function for creating a new Convolutional Layer

# 以下函数在TensorFlow的计算图中创建了一个新的卷积层。这里什么都没有计算，我们只是把数学公式添加到TensorFlow图中。
#
# 假设输入是一个4D的张量，有以下维度:
# 1. Image number.
# 2. Y-axis of each image.
# 3. X-axis of each image.
# 4. Channels of each image.

# 请注意，输入通道可能是彩色通道，或者如果输入是从先前的卷积层产生的，则也可能是过滤通道（即上一个卷积核的输出核数量）。
#
# 输出是另一个4D的张量，有以下维度:
# 1. Image number, same as input.
# 2. Y-axis of each image. If 2x2 pooling is used, then the height and width of the input images is divided by 2.
# 3. X-axis of each image. Ditto.
# 4. Channels produced by the convolutional filters.

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    # 为卷积创建一个TensorFlow操作。注意，所有维度上的步幅都设置为1，填充设置为"SAME"也就是输入图像用0填充，所以输出的大小是一样的。
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    # 把bias偏差加到卷积的结果中。每个过滤通道都添加了一个bias值。
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights

# ### Helper-function for flattening a layer
# 卷积层产生一个带有4维的输出张量。我们将在卷积层之后添加全连通的层，因此我们需要将4D张量减少为2D，这可以作为全连通层的输入。

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    # layer_shape == [num_images, img_height, img_width, num_channels]
    # features数量: img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])  # Reshape the layer to [num_images, num_features].

    # The shape of the flattened layer is now: [num_images, img_height * img_width * num_channels]
    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

# 这个函数在TensorFlow的计算图中创建了一个新的全连通层。这里什么都没有计算，我们只是把数学公式添加到TensorFlow图中。
# It is assumed that the input is a 2-dim tensor of shape `[num_images, num_inputs]`. The output is a 2-dim tensor of shape `[num_images, num_outputs]`.

def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

# Placeholde占位符变量作为TensorFlow计算图的输入，我们每次执行图表时都可能会改变它。我们将此称为占位符变量，并在下面进一步演示。
# 首先，我们为输入图像定义占位符变量。这允许我们改变输入到TensorFlow图形的图像。这就是一个所谓的张量，也就是说它是一个多维的向量或矩阵。
# 数据类型被设置为float32，shape被设为[None，img_size_flat]，在这里None表示张量可以容纳任意数量的图像，每个图像都是长度为img_size_flat的向量。

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

# 卷积层希望x被编码成一个4D的张量，so we have to reshape it so its shape is instead `[num_images, img_height, img_width, num_channels]`.
#  Note that `img_height == img_width == img_size` 。 `num_images` 可以通过设为-1为第一个维度的大小表示输入的不确定数量. So the reshape operation is:

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

# 接下来，我们有一个占位符变量，用于与占位符变量x中输入的图像相关联的真实标签。这个占位符变量的形状是[None, num_classes]，这意味着它可以容纳任意数量的标签，每个标签都是长度为`num_classes`的向量。
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# ### Convolutional Layer 1

layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)

# ### Convolutional Layers 2 and 3

# Create the second and third convolutional layers, which take as input the output from the first and second convolutional layer respectively. The number of input channels corresponds to the number of filters in the previous convolutional layer.
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)
layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2, num_input_channels=num_filters2, filter_size=filter_size3, num_filters=num_filters3, use_pooling=True)

# ### Flatten Layer
layer_flat, num_features = flatten_layer(layer_conv3)

# ### Fully-Connected Layer 1
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)

# ### Fully-Connected Layer 2
layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)

# ### Predicted Class
y_pred = tf.nn.softmax(layer_fc2)

# The class-number is the index of the largest element.
y_pred_cls = tf.argmax(y_pred, dimension=1)

# ### Cost-function to be optimized
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)

# 现在我们已经计算了每个图像分类的交叉熵，因此我们可以测量模型在每个图像上的表现。但是为了使用交叉熵来指导模型变量的优化，我们需要一个标量值，因此我们只需要对所有图像分类的交叉熵取平均值。
cost = tf.reduce_mean(cross_entropy)

# ### Optimization Method
# 注意，在这一点上没有执行优化。实际上，什么都没有计算，我们只是将优化器对象添加到TensorFlow图中，以便以后执行。
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)


# 我们需要更多的性能度量来显示用户的进展。这是一个布尔值的向量，是否预测的类等于每个图像的真类。
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

# 通过先将布尔值的向量转换为浮点数来计算分类的精度，从而使False变为0，True变为1，然后计算这些数字的平均值。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# ## TensorFlow Run
# ### Create TensorFlow session
# 一旦创建了TensorFlow图，我们就必须创建一个TensorFlow会话，用于执行该图表。
session = tf.Session()

# ### Initialize variables
# 在我们开始优化它们之前，必须初始化权重和偏差的变量。
session.run(tf.initialize_all_variables())

# ### Helper-function to perform optimization iterations
# 使用整个大型数据集来计算模型的梯度需要很长时间。因此，我们只在优化器的每次迭代中使用少量的图像。
train_batch_size = batch_size


def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    # 计算训练集的准确性。
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


# 执行许多优化迭代的功能，以便逐步改进网络层的变量。在每次迭代中，从训练集中选择了一批新的数据，然后TensorFlow使用这些训练样例来执行优化器。每100次迭代就打印一次。
# Counter for total number of iterations performed so far.
total_iterations = 0


def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # 在下面打印时间使用的时间。
    start_time = time.time()

    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations, total_iterations + num_iterations):
        # Get a batch of training examples.x_batch ：now holds a batch of images and  y_true_batch ：这些图像的真正标签
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        # Convert shape from [num examples, rows, columns, depth] to [num examples, flattened image shape]
        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        # Put the batch into a dict with the proper names for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        feed_dict_validate = {x: x_valid_batch, y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status at end of each epoch (defined as full pass through training dataset).
        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples / batch_size))

            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)

            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))


# ### Helper-function for showing the performance

# 用于在测试集上打印分类准确率的功能。
def print_validation_accuracy(show_example_errors=False,
                              show_confusion_matrix=False):
    num_test = len(data.valid.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0
    while i < num_test:
        j = min(i + batch_size, num_test)
        images = data.valid.images[i:j, :].reshape(batch_size, img_size_flat)
        labels = data.valid.labels[i:j, :]
        feed_dict = {x: images,y_true: labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j

    cls_true = np.array(data.valid.cls)
    cls_pred = np.array([classes[x] for x in cls_pred])
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

optimize(num_iterations=1)
print_validation_accuracy()

# ## Performance after 100 optimization iterations
#
# After 100 optimization iterations, the model should have significantly improved its classification accuracy.


optimize(num_iterations=99)  # We already performed 1 iteration above.

print_validation_accuracy(show_example_errors=True)

# ## Performance after 1000 optimization iterations


optimize(num_iterations=900)  # We performed 100 iterations above.

print_validation_accuracy(show_example_errors=True)

# ## Performance after 10,000 optimization iterations


optimize(num_iterations=9000)  # We performed 1000 iterations above.

print_validation_accuracy(show_example_errors=True, show_confusion_matrix=True)

# ## Test on Sample Image

test_cat = cv2.imread('cat.jpg')
test_cat = cv2.resize(test_cat, (img_size, img_size), cv2.INTER_LINEAR) / 255

preview_cat = plt.imshow(test_cat.reshape(img_size, img_size, num_channels))

test_dog = cv2.imread('lucy.jpg')
test_dog = cv2.resize(test_dog, (img_size, img_size), cv2.INTER_LINEAR) / 255

preview_dog = plt.imshow(test_dog.reshape(img_size, img_size, num_channels))


def sample_prediction(test_im):
    feed_dict_test = {
        x: test_im.reshape(1, img_size_flat),
        y_true: np.array([[1, 0]])
    }

    test_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    return classes[test_pred[0]]


print("Predicted class for test_cat: {}".format(sample_prediction(test_cat)))
print("Predicted class for test_dog: {}".format(sample_prediction(test_dog)))

session.close()
