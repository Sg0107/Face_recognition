import tensorflow as tf
from tensorflow.keras.layers import Layer

class l1dist(Layer):
    def __init__(self,**kwargs):
        super().__init__()
    def call(self,input_embedding,validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)