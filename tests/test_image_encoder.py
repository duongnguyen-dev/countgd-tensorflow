import tensorflow as tf
from countgd.blocks.image_encoder import ImageEncoderBlock

def img_encoder(inputs):
    image_encoder = ImageEncoderBlock(
        input_resolution=(224, 224), 
        num_heads=[4,8,16,32], 
        embed_dim=128,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0., 
        drop_path_rate=0.2,
        crop_size=7,
        patch_size=4,
        depths = [2, 2, 18, 2]
    )
    res = image_encoder(inputs)
    return tf.shape(res[0]), tf.shape(res[1])

def test_img_encoder():
    inputs = (
        tf.zeros([1, 224, 224, 3]),
        tf.zeros([1, 3, 4])
    )
    feat_shape, ex_shape = img_encoder(inputs)
    assert feat_shape.numpy().tolist() == [1, 28, 28, 256] and ex_shape.numpy().tolist() == [1, 3, 7, 7, 256]