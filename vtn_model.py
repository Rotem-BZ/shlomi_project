#Created by Adam Goldbraikh - Scalpel Lab Technion
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.vision_transformer import vit_base_patch16_224
import vtn_helper
from transformers import AutoFeatureExtractor, DeiTModel
from torchvision import models


class VTN(nn.Module):
    """
    VTN model builder. It uses ViT-Base as the backbone.
    Daniel Neimark, Omri Bar, Maya Zohar and Dotan Asselmann.
    "Video Transformer Network."
    https://arxiv.org/abs/2102.00719
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(VTN, self).__init__()
        self._construct_network(cfg)

    def _construct_network(self, cfg):
        """
        Builds a VTN model, with a given backbone architecture.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        self.embed_dim = 24
        if cfg.MODEL.ARCH == "VIT":
            # self.backbone = vit_base_patch16_224(pretrained=cfg.VTN.PRETRAINED,
            #                                      num_classes=0,
            #                                      drop_path_rate=cfg.VTN.DROP_PATH_RATE,
            #                                      drop_rate=cfg.VTN.DROP_RATE,
            #                                      img_size=64)

            # self.backbone = timm.create_model('deit_tiny_distilled_patch16_224', img_size=64, pretrained=cfg.VTN.PRETRAINED, in_chans=3)

            # self.backbone = AutoFeatureExtractor.from_pretrained('facebook/deit-tiny-distilled-patch16-224')
            # self.backbone = DeiTModel.from_pretrained("facebook/deit-tiny-distilled-patch16-224", add_pooling_layer=False, image_size=64)

            self.backbone = models.resnet18(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, self.embed_dim)
            self.adjust_embed_dim = torch.nn.Linear(36, self.embed_dim)

        else:
            raise NotImplementedError(f"not supporting {cfg.MODEL.ARCH}")

        # embed_dim = self.backbone.embed_dim
        # embed_dim=1000
        # embed_dim = 768
        # self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.temporal_encoder = vtn_helper.VTNLongformerModel(
            embed_dim=self.embed_dim,
            max_position_embeddings=cfg.VTN.MAX_POSITION_EMBEDDINGS,
            num_attention_heads=cfg.VTN.NUM_ATTENTION_HEADS,
            num_hidden_layers=cfg.VTN.NUM_HIDDEN_LAYERS,
            attention_mode=cfg.VTN.ATTENTION_MODE,
            pad_token_id=cfg.VTN.PAD_TOKEN_ID,
            attention_window=cfg.VTN.ATTENTION_WINDOW,
            intermediate_size=cfg.VTN.INTERMEDIATE_SIZE,
            attention_probs_dropout_prob=cfg.VTN.ATTENTION_PROBS_DROPOUT_PROB,
            hidden_dropout_prob=cfg.VTN.HIDDEN_DROPOUT_PROB)

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(cfg.VTN.HIDDEN_DIM),
        #     nn.Linear(cfg.VTN.HIDDEN_DIM, cfg.VTN.MLP_DIM),
        #     nn.GELU(),
        #     nn.Dropout(cfg.MODEL.DROPOUT_RATE),
        #     nn.Linear(cfg.VTN.MLP_DIM, cfg.MODEL.NUM_CLASSES)
        # )

    def forward(self, x, lengths, bboxes=None):

        device = x.device
        lengths = lengths.to(device)
        # position_ids = torch.arange(x.shape[1])     # x shape: (B, F, C, H, W)

        # spatial backbone
        if back_bone:
            x = x.permute(0, 2, 1, 3, 4)
            B, F, C, H, W = x.shape
            x = x.reshape(B * F, C, H, W)
            x = self.backbone(x.float())
            x = x.reshape(B, F, -1)
        else:
            x = self.adjust_embed_dim(x)

        B, F, E = x.shape
        position_ids = torch.arange(F, device=device).expand(len(lengths), max(lengths))
        # position_ids = torch.arange(F).unsqueeze(0).expand(B, F)
        # x = x.reshape(B * F, C, H, W)
        # x = self.backbone(x.float())
        # x = self.adjust_embed_dim(x)
        # print("x shape:", x.shape)
        # print("PASSED")
        # x = self.backbone(images=x.cpu(), return_tensors="pt")
        x = x.reshape(B, F, -1)

        # temporal encoder (Longformer)
        # B, D, E = x.shape
        # attention_mask = torch.ones((B, D), dtype=torch.long, device=x.device)
        attention_mask = torch.arange(max(lengths), device=x.device).expand(len(lengths), max(lengths)) < lengths.unsqueeze(1)
        # cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_tokens, x), dim=1)
        # cls_atten = torch.ones(1).expand(B, -1).to(x.device)
        # attention_mask = torch.cat((attention_mask, cls_atten), dim=1)
        # attention_mask[:, 0] = 2
        x, attention_mask, position_ids = vtn_helper.pad_to_window_size_local(
            x,
            attention_mask,
            position_ids,
            self.temporal_encoder.config.attention_window[0],
            self.temporal_encoder.config.pad_token_id)
        token_type_ids = torch.zeros(x.size()[:-1], dtype=torch.long, device=device)
        token_type_ids[:, 0] = 1

        # position_ids
        position_ids = position_ids.long()
        mask = attention_mask.ne(0).int()
        max_position_embeddings = self.temporal_encoder.config.max_position_embeddings
        position_ids = position_ids % (max_position_embeddings - 2)
        # position_ids[:, 0] = max_position_embeddings - 2
        position_ids[mask == 0] = max_position_embeddings - 1

        x = self.temporal_encoder(input_ids=None,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  inputs_embeds=x,
                                  output_attentions=None,
                                  output_hidden_states=None,
                                  return_dict=None)
        # MLP head
        x = x["last_hidden_state"]
        # x = self.mlp_head(x[:, 0])
        return x
