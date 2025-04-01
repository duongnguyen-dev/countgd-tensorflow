import tensorflow as tf
from countgd.blocks.layers.w_mha import WindowAttention
from countgd.blocks.layers.drop_path import DropPath
from countgd.blocks.layers.linear_embedding import LinearEmbedding
from countgd.blocks.layers.mlp import MLP
from countgd.blocks.layers.patch_merging import PatchMerging
from countgd.blocks.layers.patch_partition import PatchParition

class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self, 
        dim, 
        input_resolution, 
        num_heads, 
        window_size=7,
        shift_size=0,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        proj_drop=0.,
        attn_drop=0., 
        drop_path=0.
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=self.num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else tf.identity
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = MLP