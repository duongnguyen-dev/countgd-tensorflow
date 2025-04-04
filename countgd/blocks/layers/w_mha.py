import tensorflow as tf

class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, 
                 dim, 
                 window_size, 
                 num_heads,
                 qkv_bias=True, 
                 qk_scale=None, 
                 attn_drop=0.,
                 proj_drop=0.
                 ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
    
        initializer = tf.keras.initializers.TruncatedNormal(
            mean = 0., stddev = .02
        )
        table_shape = ((2 * self.window_size[0] - 1) * (2 * self.window_size[1] -1), num_heads) # (2*Wh-1, 2*Ww-1, nH)
        self.relative_position_bias_table = tf.Variable(initializer(shape=table_shape)) # (2*Wh-1, 2*Ww-1, nH)

        coords_h = tf.range(self.window_size[0]) # (Wh, )
        coords_w = tf.range(self.window_size[1]) # (Ww, )
        meshed_coords = tf.meshgrid(coords_h, coords_w) # (Wh, Ww)
        coords = tf.stack(meshed_coords) # (2, Wh, Ww)
        coords_flatten = tf.reshape(coords, [2, -1]) # (2, Wh, Ww) => (2, Wh*Ww) 
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # (2, Wh*Ww) => (2, Wh*Ww, Wh*Ww)
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0]) # => (Wh*Ww, Wh*Ww, 2)
        relative_coords = relative_coords + [self.window_size[0] - 1, self.window_size[1] - 1] # (Wh*Ww, Wh*Ww, 2)
        relative_coords = relative_coords * [2 * self.window_size[1] - 1, 1] # (Wh*Ww, Wh*Ww, 2)
        self.relative_position_index = tf.math.reduce_sum(relative_coords, -1) # (Wh*Ww, Wh*Ww)

        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=qkv_bias, kernel_initializer=initializer)
        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(dim, kernel_initializer=initializer)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, x, mask=None):
        L, N, C = x.shape # L = nW*B, N = window_size*window_size, D
        qkv = self.qkv(x) # (L, N, dim * 3)
        print(qkv.shape)
        qkv = tf.reshape(qkv, [-1, N, 3, self.num_heads, C // self.num_heads]) # (_, L, N, C, dim * 3) => (dim * 3, N, 3, nH, C // nH)
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4]) # [3, dim*3, nH, N, C // nH]
        q, k, v = tf.unstack(qkv) # each has shape (dim * 3, nH, N, C // nH)
        q = q * self.scale
        attn = tf.einsum('...ij,...kj->...ik', q, k)
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, [-1])) # ((2*W-1)**2 * nH)
        relative_position_bias = tf.reshape(relative_position_bias, [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1]) # (Wh*Ww, Wh*Ww, nH)
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1]) # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = tf.reshape(attn, [-1 // nW, nW, self.num_heads, N, N]) + mask[:, None, :, :]
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)

        x = tf.reshape(tf.transpose(attn @ v, perm=[0, 2, 1, 3]), [-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x