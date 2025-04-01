import tensorflow as tf
from countgd.blocks.layers.w_mha import WindowAttention

def window_attention(arr):
    attn = WindowAttention(
        dim=96,
        window_size=(4, 4),
        num_heads=8, 
        qkv_bias=True,
        qk_scale=None, 
        attn_drop=0.0, 
        proj_drop=0.0
    )
    return tf.shape(attn(arr))

def test_window_attention():
    arr = tf.zeros((1, 196, 16, 96))
    assert window_attention(arr).numpy().tolist() == [1, 196, 16, 96]
