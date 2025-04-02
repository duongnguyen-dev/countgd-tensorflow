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
        window_size=SwintBConfig.WINDOW_SIZE,
        mlp_ratio=SwintBConfig.MLP_RATIO,
        qk_scale=SwintBConfig.QK_SCALE,
        qkv_bias=SwintBConfig.QKV_BIAS,
        proj_drop=SwintBConfig.PROJ_DROP,
        attn_drop=SwintBConfig.ATTN_DROP, 
        drop_path=SwintBConfig.DROP_PATH
    ):
    num_patch_x = input_shape[0] // window_size
    num_patch_y = input_shape[1] // window_size

    inputs = tf.keras.layers.Input(shape=input_shape)
    # Patch extractor
    patches = PatchParition(window_size)(inputs)

    # Stage 1
    patches_embed = LinearEmbedding(num_patch_x * num_patch_y, embed_dim)(patches)

    out_stage_1 = SwinTransformerBlock(
        dim=embed_dim,
        input_resolution=(num_patch_x, num_patch_y),
        num_heads=num_heads[0],
        window_size=window_size,
        shift_size=0,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        proj_drop=proj_drop,
        attn_drop=attn_drop,
        drop_path=drop_path
    )(patches_embed)
    out_stage_1 = SwinTransformerBlock(
        dim=embed_dim,
        input_resolution=(num_patch_x, num_patch_y),
        num_heads=num_heads[0],
        window_size=window_size,
        shift_size=1,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        proj_drop=proj_drop,
        attn_drop=attn_drop,
        drop_path=drop_path
    )(out_stage_1)
    
    # Stage 2
    representation = PatchMerging((num_patch_x, num_patch_y), channels=embed_dim)(out_stage_1)
    num_patch_x = int(tf.sqrt(tf.cast(tf.shape(representation)[1], tf.float32)).numpy())
    num_patch_y = int(tf.sqrt(tf.cast(tf.shape(representation)[1], tf.float32)).numpy())
    embed_dim = tf.shape(representation)[-1].numpy().item()

    out_stage_2 = SwinTransformerBlock(
        dim=embed_dim,
        input_resolution=(num_patch_x, num_patch_y),
        num_heads=num_heads[1],
        window_size=window_size,
        shift_size=0,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        proj_drop=proj_drop,
        attn_drop=attn_drop,
        drop_path=drop_path
    )(representation)
    out_stage_2 = SwinTransformerBlock(
        dim=embed_dim,
        input_resolution=(num_patch_x, num_patch_y),
        num_heads=num_heads[1],
        window_size=window_size,
        shift_size=1,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        proj_drop=proj_drop,
        attn_drop=attn_drop,
        drop_path=drop_path
    )(out_stage_2)

    # Stage 3
    representation = PatchMerging((num_patch_x, num_patch_y), channels=embed_dim)(out_stage_2)
    num_patch_x = int(tf.sqrt(tf.cast(tf.shape(representation)[1], tf.float32)).numpy())
    num_patch_y = int(tf.sqrt(tf.cast(tf.shape(representation)[1], tf.float32)).numpy())
    embed_dim = tf.shape(representation)[-1].numpy().item()

    out_stage_3 = SwinTransformerBlock(
        dim=embed_dim,
        input_resolution=(num_patch_x, num_patch_y),
        num_heads=num_heads[2],
        window_size=window_size,
        shift_size=0,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        proj_drop=proj_drop,
        attn_drop=attn_drop,
        drop_path=drop_path
    )(representation)
    out_stage_3 = SwinTransformerBlock(
        dim=embed_dim,
        input_resolution=(num_patch_x, num_patch_y),
        num_heads=num_heads[2],
        window_size=window_size,
        shift_size=1,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        proj_drop=proj_drop,
        attn_drop=attn_drop,
        drop_path=drop_path
    )(out_stage_3)

    # Stage 4
    representation = PatchMerging((num_patch_x, num_patch_y), channels=embed_dim)(out_stage_3)
    num_patch_x = int(tf.sqrt(tf.cast(tf.shape(representation)[1], tf.float32)).numpy())
    num_patch_y = int(tf.sqrt(tf.cast(tf.shape(representation)[1], tf.float32)).numpy())
    embed_dim = tf.shape(representation)[-1].numpy().item()
    
    out_stage_4 = SwinTransformerBlock(
        dim=embed_dim,
        input_resolution=(num_patch_x, num_patch_y),
        num_heads=num_heads[3],
        window_size=window_size,
        shift_size=0,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        proj_drop=proj_drop,
        attn_drop=attn_drop,
        drop_path=drop_path
    )(representation)
    out_stage_4 = SwinTransformerBlock(
        dim=embed_dim,
        input_resolution=(num_patch_x, num_patch_y),
        num_heads=num_heads[3],
        window_size=window_size,
        shift_size=1,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        proj_drop=proj_drop,
        attn_drop=attn_drop,
        drop_path=drop_path
    )(out_stage_4)

    model = tf.keras.Model(inputs=inputs, outputs=out_stage_4)
    
    return model