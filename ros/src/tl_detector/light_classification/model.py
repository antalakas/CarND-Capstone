import os
from collections import namedtuple
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import flatten

ConvLayer = namedtuple("ConvLayer", ["kernel_size", "num_filter", "pool_size", "pool_stride"])
FCLayer = namedtuple("FCLayer", ["output_size", "activation", "dropout"])

class Model(object):

    def __init__(self):
        self.num_classes = 3
        self.save_dir = os.path.join(os.path.dirname(__file__), "checkpoint")
        self.x = tf.placeholder(tf.float32, (None, 600, 800, 3), name="input_image")
        self.y = tf.placeholder(tf.int32, (None))
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.learning_rate = 2e-3
        self.scale = 1e-4

        self.conv_layers = [
            ConvLayer(5, 4, 2, 2),
            ConvLayer(5, 5, 2, 2),
            ConvLayer(5, 6, 2, 2),
            ConvLayer(5, 7, 2, 2),
            ConvLayer(5, 7, 2, 2)
        ]
        self.fc_layers = [
            FCLayer(32, tf.nn.relu, True),
            FCLayer(16, tf.nn.relu, True),
            FCLayer(self.num_classes, None, False)
        ]

        self.build_inference()
        self.build_loss()
        self.build_prediction()
        self.build_metric()
        self.build_optimize()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_inference(self):
        regularizer = tf.contrib.layers.l2_regularizer(self.scale)
        conv_input = self.x
        for i, layer in enumerate(self.conv_layers):
            conv = tf.layers.conv2d(
                conv_input,
                layer.num_filter,
                layer.kernel_size,
                kernel_initializer=xavier_initializer(),
                kernel_regularizer=regularizer
            )
            pool = tf.layers.max_pooling2d(
                conv,
                layer.pool_size,
                layer.pool_stride
            )
            conv_input = tf.nn.relu(pool)
        fc_input = flatten(conv_input)
        for layer in self.fc_layers:
            fc_output = tf.layers.dense(
                fc_input,
                layer.output_size,
                activation=layer.activation,
                kernel_initializer=xavier_initializer(),
                kernel_regularizer=regularizer
            )
            if layer.dropout:
                fc_input = tf.nn.dropout(fc_output, self.keep_prob)
            else:
                fc_input = fc_output
        self.logits = fc_input
        print(self.logits)


    def build_loss(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.y
        )
        self.loss = tf.reduce_mean(cross_entropy)

    def build_prediction(self):
        self.prediction = tf.argmax(self.logits, axis=1, name="output")

    def build_metric(self):
        correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.y)
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32)
        )

    def build_optimize(self):
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.decay_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, 100, 0.96)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

    def train(self, data, labels):
        _, l = self.sess.run(
            [self.optimize,
             self.loss],
            feed_dict={
                self.x: data,
                self.y: labels,
                self.keep_prob: 0.5
            }
        )
        return l

    def evaluate(self, data, labels):
        acc = self.sess.run(
            self.accuracy,
            feed_dict={
                self.x: data,
                self.y:labels,
                self.keep_prob: 1.0
            }
        )
        return acc

    def save(self, filename):
        file_path = os.path.join(self.save_dir, filename)
        self.saver.save(self.sess, file_path)

    def load(self, filename):
        file_path = os.path.join(self.save_dir, filename)
        self.saver.restore(self.sess, file_path)
