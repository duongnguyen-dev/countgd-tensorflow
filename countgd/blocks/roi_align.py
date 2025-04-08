import tensorflow as tf
from countgd.blocks.helpers import log2_graph

class RoiAlign(tf.keras.layers.Layer):
    def __init__(self, pool_shape):
        super().__init__()
        self.pool_shape = pool_shape
    
    def call(self, inputs):
        boxes = inputs[0]
        feature_maps = inputs[-1]

        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1

        # Normalized coordinates        
        roi_level = log2_graph(tf.sqrt(h * w) / 224.0)
        roi_level = tf.cast()

        # Loop through levels and apply ROI pooling to each feature maps
        pooled = []
        box_to_level = []

        for i, level in enumerate(feature_maps):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.minimum()