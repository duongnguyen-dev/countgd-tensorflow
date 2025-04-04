import tensorflow as tf
import numpy as np
from countgd.blocks.layers.w_mha import WindowAttention
from countgd.blocks.layers.drop_path import DropPath
from countgd.blocks.layers.patch_partition import PatchParition
from countgd.blocks.layers.linear_embedding import LinearEmbedding
from countgd.blocks.layers.patch_merging import PatchMerging
from countgd.blocks.layers.w_mha import WindowAttention
from countgd.blocks.layers.mlp import MLP
from countgd.blocks.helpers import *

class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self, 
        input_resolution, 
        num_heads, 
        embed_dim=128,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0., 
        drop_path_rate=0.2
    ):
        super().__init__()
        
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.embed_dim = embed_dim
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
            embed_dim, 
            window_size=(self.window_size, self.window_size), 
            num_heads=self.num_heads,
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_drop=attn_drop_rate, 
            proj_drop=drop_rate
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else tf.identity
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(mlp_hidden_dim, embed_dim, dropout_rate=drop_rate)

        if self.shift_size > 0:
            # Calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1])
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
        
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            img_mask = tf.constant(img_mask)
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size*self.window_size])
            attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
            self.attn_mask = tf.where(attn_mask==0, -100., 0.)
        else:
            self.attn_mask = None
    
    def call(self, x):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, N, embed_dim).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, N, D = x.shape
        H, W = self.input_resolution
        assert N == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, [-1, H, W, D]) # B x H x W x C

        # Pad feature maps to multiples window size
        pad_b = (self.window_size - H%self.window_size)%self.window_size
        pad_r = (self.window_size - W%self.window_size)%self.window_size
        paddings = tf.constant([
            [0, 0], 
            [0, pad_b],
            [0, pad_r],
            [0, 0]
        ])

        x = tf.pad(x, paddings)
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=(1, 2))
            attn_mask = self.attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size) # nW*B, window_size, window_size, D
        x_windows = tf.reshape(x_windows, [-1, self.window_size * self.window_size, D])  # nW*B, window_size*window_size, D
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask) # nW*B, window_size*window_size, D

        # Merge windows
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, D])
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=(1, 2))
        else:
            x = shifted_x
        x = tf.reshape(x, [-1, H * W, D])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
        