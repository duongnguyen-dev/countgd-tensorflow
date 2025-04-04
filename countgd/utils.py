import tensorflow as tf
from countgd.config import SwintBConfig
from countgd.blocks.layers.patch_partition import PatchParition
from countgd.blocks.layers.patch_merging import PatchMerging
from countgd.blocks.layers.linear_embedding import LinearEmbedding
from countgd.blocks.swint import SwinTransformerBlock

def create_swintransformerB(
        input_shape=SwintBConfig.INPUT_SHAPE, 
        embed_dim=SwintBConfig.EMBED_DIM, 
        num_heads=SwintBConfig.NUM_HEADS,
        depths=SwintBConfig.DEPTHS,
        patch_size=SwintBConfig.PATCH_SIZE,
        window_size=SwintBConfig.WINDOW_SIZE,
        mlp_ratio=SwintBConfig.MLP_RATIO,
        qk_scale=SwintBConfig.QK_SCALE,
        qkv_bias=SwintBConfig.QKV_BIAS,
        proj_drop=SwintBConfig.PROJ_DROP,
        attn_drop=SwintBConfig.ATTN_DROP, 
        drop_path=SwintBConfig.DROP_PATH
    ):
    num_patch_x = input_shape[0] // patch_size
    num_patch_y = input_shape[1] // patch_size

    inputs = tf.keras.layers.Input(shape=input_shape)
    # Patch extractor
    patches = PatchParition(patch_size)(inputs)

    # Stage 1
    patches_embed = LinearEmbedding(num_patch_x * num_patch_y, embed_dim)(patches)

    out_stage_1 = patches_embed
    for i in range(depths[0]):
        out_stage_1 = SwinTransformerBlock(
            embed_dim=embed_dim,
            input_resolution=(num_patch_x, num_patch_y),
            num_heads=num_heads[0],
            window_size=window_size,
            shift_size=1 if i % 2 == 0 else 0,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=proj_drop,
            attn_drop_rate=attn_drop,
            drop_path_rate=drop_path
        )(out_stage_1)
    
    # Stage 2
    representation = PatchMerging((num_patch_x, num_patch_y), channels=embed_dim)(out_stage_1)
    num_patch_x //= 2
    num_patch_y //= 2
    embed_dim *= 2

    out_stage_2 = representation

    for i in range(depths[1]):
        out_stage_2 = SwinTransformerBlock(
            embed_dim=embed_dim,
            input_resolution=(num_patch_x, num_patch_y),
            num_heads=num_heads[1],
            window_size=window_size,
            shift_size= 1 if i % 2 == 0 else 0,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=proj_drop,
            attn_drop_rate=attn_drop,
            drop_path_rate=drop_path
        )(out_stage_2)

    # Stage 3
    representation = PatchMerging((num_patch_x, num_patch_y), channels=embed_dim)(out_stage_2)
    num_patch_x //= 2
    num_patch_y //= 2
    embed_dim *= 2

    out_stage_3 = representation
    for i in range(depths[2]):
        out_stage_3 = SwinTransformerBlock(
            embed_dim=embed_dim,
            input_resolution=(num_patch_x, num_patch_y),
            num_heads=num_heads[2],
            window_size=window_size,
            shift_size= 1 if i % 2 == 0 else 0,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=proj_drop,
            attn_drop_rate=attn_drop,
            drop_path_rate=drop_path
        )(out_stage_3)

    # Stage 4
    representation = PatchMerging((num_patch_x, num_patch_y), channels=embed_dim)(out_stage_3)
    num_patch_x //= 2
    num_patch_y //= 2
    embed_dim *= 2

    out_stage_4 = representation
    for i in range(depths[3]):
        out_stage_4 = SwinTransformerBlock(
            embed_dim=embed_dim,
            input_resolution=(num_patch_x, num_patch_y),
            num_heads=num_heads[3],
            window_size=window_size,
            shift_size=1 if i % 2 == 0 else 0,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=proj_drop,
            attn_drop_rate=attn_drop,
            drop_path_rate=drop_path
        )(out_stage_4)
    
    model = tf.keras.Model(inputs=inputs, outputs=(out_stage_2, out_stage_3, out_stage_4))

    return model