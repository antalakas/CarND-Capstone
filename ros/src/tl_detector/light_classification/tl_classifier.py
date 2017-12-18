from styx_msgs.msg import TrafficLight
import glob
import tensorflow as tf
import numpy as np
import os

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.sess=tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25)))
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.saver = tf.train.import_meta_graph(os.path.join(data_dir, 'model.ckpt.meta'))
        self.saver.restore(self.sess, os.path.join(data_dir, "model.ckpt"))

        self.graph = tf.get_default_graph()
        self.x = self.graph.get_tensor_by_name("input_image:0")
        self.keep_prob = self.graph.get_tensor_by_name("keep_prob:0")
        self.output_value = self.graph.get_tensor_by_name("output:0")

        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        predictions = self.sess.run(self.output_value, feed_dict = {self.x: [image], self.keep_prob: 1.0})
        predictions = predictions[0]
        if predictions == 0:
            return TrafficLight.RED
        elif predictions == 2:
            return TrafficLight.GREEN
        return TrafficLight.UNKNOWN
