import tensorflow as tf
import tensorflow_models as tfm 
from countgd.blocks.swint import SwinTransformerBlock
from countgd.blocks.upscale import UpScaleLayer
from countgd.blocks.layers.patch_partition import PatchParition
from countgd.blocks.layers.patch_merging import PatchMerging
from countgd.blocks.layers.linear_embedding import LinearEmbedding
from countgd.config import SwintBConfig

class ImageEncoderBlock(tf.keras.layers.Layer):
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
                drop_path_rate=0.2,
                crop_size=7,
                patch_size=4,
                depths = [2, 2, 18, 2]
        ):
        super().__init__()

        self.patch_size = patch_size
        self.depths = depths

        self.num_patch_x_0 = input_resolution[0] // patch_size
        self.num_patch_y_0 = input_resolution[1] // patch_size
        self.embed_dim_0 = embed_dim

        self.swint_0 = SwinTransformerBlock(
            (self.num_patch_x_0, self.num_patch_y_0), 
            num_heads[0], 
            embed_dim=self.embed_dim_0,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate
        )

        self.num_patch_x_1 = self.num_patch_x_0 // 2
        self.num_patch_y_1 = self.num_patch_y_0 // 2
        self.embed_dim_1 = self.embed_dim_0 * 2

        self.swint_1 = SwinTransformerBlock(
            (self.num_patch_x_1, self.num_patch_y_1), 
            num_heads[1], 
            embed_dim=self.embed_dim_1,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate
        )

        self.num_patch_x_2 = self.num_patch_x_1 // 2
        self.num_patch_y_2 = self.num_patch_y_1 // 2
        self.embed_dim_2 = self.embed_dim_1 * 2

        self.swint_2 = SwinTransformerBlock(
            (self.num_patch_x_2, self.num_patch_y_2), 
            num_heads[2], 
            embed_dim=self.embed_dim_2,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate
        )

        self.num_patch_x_3 = self.num_patch_x_2 // 2
        self.num_patch_y_3 = self.num_patch_y_2 // 2
        self.embed_dim_3 = self.embed_dim_2 * 2

        self.swint_3 = SwinTransformerBlock(
            (self.num_patch_x_3, self.num_patch_y_3), 
            num_heads[3], 
            embed_dim=self.embed_dim_3,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate
        )

        self.upscale = UpScaleLayer()
        self.roi_align = tfm.vision.layers.MultilevelROIAligner(
            crop_size=crop_size, sample_offset=0.5
        )

    def call(self, inputs):
        images = inputs[0]
        examplars = inputs[1]

        # Patch extractor
        patches = PatchParition(self.patch_size)(images)

        # Stage 1
        patches_embed = LinearEmbedding(self.num_patch_x_0 * self.num_patch_y_0, self.embed_dim_0)(patches)

        out_stage_1 = patches_embed
        for _ in range(self.depths[0]):
            out_stage_1 = self.swint_0(out_stage_1)
        
        # Stage 2
        representation = PatchMerging((self.num_patch_x_0, self.num_patch_y_0), channels=self.embed_dim_0)(out_stage_1)
        out_stage_2 = representation

        for _ in range(self.depths[1]):
            out_stage_2 = self.swint_1(out_stage_2)

        # Stage 3
        representation = PatchMerging((self.num_patch_x_1, self.num_patch_y_1), channels=self.embed_dim_1)(out_stage_2)
        out_stage_3 = representation

        for _ in range(self.depths[2]):
            out_stage_3 = self.swint_2(out_stage_3)

        # Stage 4
        representation = PatchMerging((self.num_patch_x_2, self.num_patch_y_2), channels=self.embed_dim_2)(out_stage_3)
        out_stage_4 = representation

        for _ in range(self.depths[3]):
            out_stage_4 = self.swint_3(out_stage_4)

        feature_maps = self.upscale((out_stage_2, out_stage_3, out_stage_4))
        visual_examplars = self.roi_align({"0":feature_maps}, examplars)

        return feature_maps, visual_examplars