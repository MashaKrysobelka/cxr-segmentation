"""
Архитектура модели: U-Net декодер + DINO backbone
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from typing import Optional
from config import PATCH_SIZE, DEVICE


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout_p: float = 0.1):
        super().__init__()
        total_in = in_channels + skip_channels

        self.conv1 = nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return x


class DINOSegmentationUNet(nn.Module):
    def __init__(self, backbone_name='microsoft/rad-dino-maira-2', backbone_dim=768, 
                 num_classes=7, decoder_channels=(512, 256, 128, 64), dropout_p=0.1, 
                 device=None, patch_size=PATCH_SIZE, n_decoder_blocks=4):
        super().__init__()
        
        self.device = device if device else DEVICE
        self.patch_size = patch_size
        self.n_decoder_blocks = n_decoder_blocks

        self.backbone = AutoModel.from_pretrained(backbone_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(backbone_name)

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.encoder_proj = nn.Sequential(
            nn.Conv2d(backbone_dim, decoder_channels[0], kernel_size=1),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.skip_projections = nn.ModuleList()
        for i in range(1, n_decoder_blocks):
            target_ch = decoder_channels[i-1]
            self.skip_projections.append(
                nn.Sequential(
                    nn.Conv2d(backbone_dim, target_ch, kernel_size=1),
                    nn.BatchNorm2d(target_ch),
                    nn.ReLU(inplace=True)
                )
            )

        self.decoder_blocks = nn.ModuleList()
        for i in range(n_decoder_blocks):
            if i == 0:
                in_ch = decoder_channels[0]
                skip_ch = 0
            else:
                in_ch = decoder_channels[i-1]
                skip_ch = decoder_channels[i-1]
            
            self.decoder_blocks.append(
                UNetDecoderBlock(in_ch, skip_ch, decoder_channels[i], dropout_p)
            )

        last_ch = decoder_channels[-1]
        self.seg_head = nn.Sequential(
            nn.Conv2d(last_ch, last_ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(last_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(last_ch // 2, num_classes, kernel_size=1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.seg_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def preprocess(self, images):
        inputs = self.processor(images, return_tensors="pt", do_rescale=False)
        return inputs['pixel_values'].to(self.device)

    def _get_features(self, pixel_values):
        self.backbone.eval()
        with torch.no_grad():
            outputs = self.backbone(
                pixel_values=pixel_values.to(self.backbone.device),
                output_hidden_states=True
            )
            features = list(outputs.hidden_states[-self.n_decoder_blocks:])
            return [f[:, 1:, :] for f in features]

    def forward(self, x):
        B, _, H, W = x.shape
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size

        feature_list = self._get_features(x)[::-1]

        def reshape_tokens(tokens):
            n_needed = H_patch * W_patch
            n_current = tokens.shape[1]
            if n_current > n_needed:
                tokens = tokens[:, :n_needed, :]
            elif n_current < n_needed:
                tokens = F.pad(tokens, (0, 0, 0, n_needed - n_current))
            return tokens.reshape(B, H_patch, W_patch, -1).permute(0, 3, 1, 2)

        enc_feat = reshape_tokens(feature_list[0])
        x_dec = self.encoder_proj(enc_feat)

        decoded = self.decoder_blocks[0](x_dec, skip=None)
        
        for i in range(1, len(self.decoder_blocks)):
            decoded = F.interpolate(decoded, scale_factor=2, mode='bilinear', align_corners=False)
            
            skip_feat = reshape_tokens(feature_list[i])
            skip_proj = self.skip_projections[i-1](skip_feat)
            
            decoded = self.decoder_blocks[i](decoded, skip=skip_proj)

        logits = self.seg_head(decoded)
        return F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

    def predict_proba(self, x):
        return torch.sigmoid(self(x))

    def predict(self, x, threshold=0.5):
        return (self.predict_proba(x) > threshold).float()