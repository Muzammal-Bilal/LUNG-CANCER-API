from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization
import tensorflow as tf

class ViTAttention(Layer):
    def __init__(self, num_heads=8, embed_dim=128, **kwargs):
        super(ViTAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.att = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.layer_norm = LayerNormalization()

    def build(self, input_shape):
        super(ViTAttention, self).build(input_shape)

    def call(self, inputs, training=False):
        seq_len = inputs.shape[1]
        inputs_reshaped = tf.reshape(inputs, (-1, seq_len, self.embed_dim))
        att_output = self.att(inputs_reshaped, inputs_reshaped)
        output = self.layer_norm(inputs_reshaped + att_output)
        return output
