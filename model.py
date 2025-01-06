import torch
import torch.nn as nn
from video_mamba import videomamba_tiny
from head import VisualHead

class Model(nn.Module):
    def __init__(
        self,
        pretrained=True,
        width=224, height=224,
        patch_size=[16, 16],
        channels=3,
        drop_rate=0., drop_path_rate=0.,
        norm_epsilon=1e-5,
        ssm_cfg=None,
        initializer_cfg=None,
        fused_add_norm=True, rms_norm=True,
        residual_in_fp32=True,
        bimamba=True,
        kernel_size=1,
        n_frames=256,
        device=None,
        dtype=None,
        use_checkpoint=False,
        checkpoint_num=0,
        head_drop_rate=0.,
        n_classes=1235,
    ):
        super(Model, self).__init__()
        
        # SwinTransformer3D Backbone
        self.backbone = videomamba_tiny(
            pretrained=pretrained,
            width=width, height=height,
            patch_size=patch_size,
            channels=channels,
            drop_rate=drop_rate, drop_path_rate=drop_path_rate,
            norm_epsilon=norm_epsilon,
            ssm_cfg=ssm_cfg, initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm, rms_norm=rms_norm, residual_in_fp32=residual_in_fp32,
            bimamba=bimamba,
            kernel_size=kernel_size,
            n_frames=n_frames,
            device=device,
            dtype=dtype,
            use_checkpoint=use_checkpoint,
            checkpoint_num=checkpoint_num
        )
        
        # VisualHead
        self.head = VisualHead(
            cls_num=n_classes,
            input_size=self.backbone.d_model,
            hidden_size=512,
            ff_size=2048,
            pe=False,
            head_drop_rate=head_drop_rate,
            ff_kernelsize=3,
            pretrained_ckpt=None,
            is_empty=False,
            frozen=False,
            plus_conv_cfg={},
            ssl_projection_cfg={}
        )

    def forward(self, x, sgn_lengths):
        """
        x: Input tensor of shape (B, C, T, H, W)
        sgn_lengths: Signal lengths for each sample in the batch
        """
        # Backbone feature extraction
        backbone_output = self.backbone(x, sgn_lengths)
        
        # Extract features and valid lengths
        # sgn = backbone_output['sgn'].permute(0, 2, 1)
        sgn = backbone_output['sgn']
        sgn_mask = backbone_output['sgn_mask'][-1]
        valid_len_in = backbone_output['valid_len_out'][-1]

        # Head processing
        head_output = self.head(sgn, sgn_mask, valid_len_in)

        return {
            'gloss_logits': head_output['gloss_logits'],
            'gloss_probabilities_log': head_output['gloss_probabilities_log'],
            'valid_len_out': head_output['valid_len_out']
        }