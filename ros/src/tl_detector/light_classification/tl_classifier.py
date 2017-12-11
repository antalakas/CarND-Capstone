from styx_msgs.msg import TrafficLight
import tensorflow as tf

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return TrafficLight.UNKNOWN

def make_model():
    inputs = tf.placeholder(tf.float32, shape=[None, None], name="inputs")
    keep_prob = tf.placeholder(tf.float32, name="dropout")

    # Convolutional
    conv1 = tf.layers.conv2d(
        filters=16,
        kernel_size=5,
        padding="SAME"
    )
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=2,
        strides=2
    )
    out1 = tf.nn.relu(pool1)

    conv2 = tf.layers.conv2d(
        filters=32,
        kernel_size=3,
        padding="SAME"
    )
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=2,
        strides=2
    )
    out = tf.nn.relu(pool2)

    # Global average pooling
    pooled = tf.reduce_mean(out, axis=[1, 2])

    # Fully Connected
    fc1 = tf.layers.dense(
        pooled,
        200,
        activation=tf.nn.relu
    )

    dropped1 = tf.nn.dropout(fc1, keep_prob=keep_porb)

    fc2 = tf.layers.dense(
        dropped1,
        100,
        activation=tf.nn.relu
    )

    logits = tf.lauers.dense(
        fc2,
        output_size
    )

    predictions = tf.argmax(logits, name="predictions")

    return inputs, keep_prob, logits, predictions
