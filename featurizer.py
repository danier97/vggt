import torch
import numpy as np
from submodules.vggt.vggt.models.vggt import VGGT
from utils import center_padding
from einops import rearrange

class VGGTBackbone(VGGT):
    def __init__(self,):

        super().__init__()
        self.load_state_dict(torch.hub.load_state_dict_from_url("https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
        self.requires_grad_(False)
        self.point_head.feature_only = True
        
        self.feat_dims = [2048] * 24 + [128] # 24 transformer layers + DPT layer

    @torch.no_grad()
    def forward(self, batch, layers=None, generator=None, device='cuda:0'):
        '''
        Returns features of all frames.
        '''

        batch_size = batch['pixel_values'].shape[0]
        video_length = batch['pixel_values'].shape[1]
                
        if type(layers) == int:
            layers = [layers]
        assert max(layers) <  25

        # normalize and pad input images to [0, 1]
        frames = batch['pixel_values'].to(device) * 0.5 + 0.5 # (B,F,C,H,W), 0-1
        frames = rearrange(frames, 'b f c h w -> (b f) c h w')  # (B*F,C,H,W), 0-1
        frames_padded, pt, pb, pl, pr = center_padding(frames, 14)  # (B*F,C,H',W'), 0-1
        frames_padded = rearrange(frames_padded, '(b f) c h w -> b f c h w', b=batch_size, f=video_length)  # (B,F,C,H',W'), 0-1
        
        image_h, image_w = frames_padded.shape[3], frames_padded.shape[4]
        patch_h, patch_w = image_h // 14, image_w // 14

        # get features from aa layers
        aggregated_tokens_list, patch_start_idx = self.aggregator(frames_padded)

        embeds = []
        for layer_i, x in enumerate(aggregated_tokens_list):
            if layer_i in layers:
                embeds.append(rearrange(x[:,:,patch_start_idx:], 'b f (h w) c -> b f c h w', h=patch_h, w=patch_w).contiguous())

        if len(embeds) == len(layers):
            return embeds[0] if len(layers) == 1 else embeds
        
        # get features from DPT layer (point head)
        x = self.point_head(
            aggregated_tokens_list, images=frames_padded, patch_start_idx=patch_start_idx, 
        ) # (B,F,C,H',W')
        x = x[:, :, :, pt:x.shape[3]-pb, pl:x.shape[4]-pr] # (B,F,C,H,W)
        embeds.append(x)

        return embeds[0] if len(layers) == 1 else embeds