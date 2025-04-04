import tensorflow as tf
from countgd.blocks.layers.w_mha import WindowAttention

def window_attention(arr):
    attn = WindowAttention(
        dim=128,
        window_size=(7, 7),
        num_heads=4, 
        qkv_bias=True,
        qk_scale=None, 
        attn_drop=0.0, 
        proj_drop=0.0
    )
    return tf.shape(attn(arr))

def test_window_attention():
    arr = tf.zeros((64, 49, 128))
    assert window_attention(arr).numpy().tolist() == [64, 49, 128]
