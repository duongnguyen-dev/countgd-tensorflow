import tensorflow as tf

class LinearEmbedding(tf.keras.layers.Layer):
    '''
    This layer is the same as the linear embedding layer as described in this paper: https://arxiv.org/pdf/2010.11929
    It takes the input of size ( B, N, (P*P*C) ) which is the output from the patch partition layer and turn into ( B, N, D ), where D is the embedding dim and N = (H*W)//(P*P) is the number of patches in an image
    '''
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.projection = tf.keras.layers.Dense(embed_dim)
        self.position_embedding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=embed_dim)

    def call(self, patch):
        # patch embeddings
        patches_embed = self.projection(patch) # (B, N, (P*P*C)) => (B, N, D)
        # positional embeddings
        positions = tf.range(start=0, limit=self.num_patches, delta=1) # tensor([0, 1, 2, ... N], shape=(N, ))
        positions_embed = self.position_embedding(positions) # (N, ) => (N, D)

        encoded = patches_embed + positions_embed # (B, N, D)

        return encoded