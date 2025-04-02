import tensorflow as tf
from countgd.blocks.layers.linear_embedding import LinearEmbedding

def linear_embedding(patch):
    patch_part = LinearEmbedding(num_patches=1024, embed_dim=128)
    return tf.shape(patch_part(patch))

def test_linear_embedding():
    arr = tf.zeros((1, 1024, 48))
    assert linear_embedding(arr).numpy().tolist() == [1, 1024, 128]