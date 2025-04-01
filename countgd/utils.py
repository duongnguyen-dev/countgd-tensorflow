import tensorflow as tf
from countgd.blocks.layers.patch_partition import PatchParition
from countgd.blocks.layers.linear_embedding import LinearEmbedding
from countgd.blocks.swint import SwinTransformerBlock

def create_swintransformer(input_shape=(224, 224, 3), window_size=4, embed_dim=96, num_heads=8):
    num_patch_x = input_shape[0] // window_size
    num_patch_y = input_shape[1] // window_size
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Patch extractor
    patches = PatchParition(window_size)(inputs)
    patches_embed = LinearEmbedding(num_patch_x * num_patch_y, embed_dim)(patches)

    # first Swin Transformer block
    out_stage_1 = SwinTransformerBlock(
        dim=embed_dim,
        input_resolution=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0
    )(patches_embed)
    # second Swin Transformer block
    out_stage_1 = SwinTransformerBlock(
        dim=embed_dim,
        input_resolution=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=1
    )(out_stage_1)
    
    model = tf.keras.Model(inputs=inputs, outputs=out_stage_1)
    return model