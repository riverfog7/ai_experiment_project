from dataclasses import dataclass
from typing import Optional

import timm
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput

from .configs import EfficientMultiTaskClassificationConfig


@dataclass
class MultiTaskOutput(ModelOutput):
    # custom model output dataclass (unconventional classification architecture)
    loss: Optional[torch.FloatTensor] = None
    logits_1: torch.FloatTensor = None
    logits_2: torch.FloatTensor = None


class EfficientMultiTaskClassificationModel(PreTrainedModel):
    config_class = EfficientMultiTaskClassificationConfig

    def __init__(self, config: EfficientMultiTaskClassificationConfig):
        super().__init__(config)

        # Run a dummy input to check output shape
        self.img_size = config.img_size
        if config.in_features is None:
            # Case1: Model Init
            # see https://github.com/huggingface/pytorch-image-models/issues/220
            # no classifier head
            self.backbone = timm.create_model(
                config.backbone_name,
                pretrained=config.pretrained,
                num_classes=0,
                global_pool="",
            )

            with torch.no_grad():
                # If this crashes here, image size is wrong.
                self.backbone.eval()
                dummy_input = torch.randn(1, 3, config.img_size, config.img_size)
                features = self.backbone(dummy_input)
                self.backbone.train()

                # Handle different pooling outputs (some internal timm pooling vs raw features)
                if len(features.shape) == 4:
                    # Shape is (B, C, H, W) -> We need to pool
                    # B,C,H,W -> B,C,1,1 (will be flattened later)
                    self.pool = nn.AdaptiveAvgPool2d(1)
                    in_features = features.shape[1]
                    config.in_features = in_features
                    self.has_internal_pooling = False
                    config.has_internal_pooling = False

                elif len(features.shape) == 2:
                    # Shape is (B, Features) -> Already pooled
                    self.pool = nn.Identity()
                    in_features = features.shape[1]
                    config.in_features = in_features
                    self.has_internal_pooling = True
                    config.has_internal_pooling = True

                else:
                    raise ValueError(
                        f"Unexpected feature shape from backbone: {features.shape}"
                    )
        else:
            # Case2: Loading from pretrained
            self.backbone = timm.create_model(
                config.backbone_name,
                pretrained=False, # loading from pretrained -> do not use imagenet weights
                num_classes=0,
                global_pool="",
            )

            in_features = config.in_features
            if config.has_internal_pooling:
                self.pool = nn.Identity()
            else:
                self.pool = nn.AdaptiveAvgPool2d(1)

        # Assertions for easy debugging
        assert self.pool is not None, "Pooling layer not initialized."
        assert in_features is not None, "in_features not determined."
        self.pool.train()

        # Always unfreeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = True

        # custom classifier head (two head exists because of two classification tasks)
        # activation is handled in loss function (CrossEntropyLoss)
        self.classifier_1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(in_features, config.num_classes_1)
        )
        self.classifier_1.train()

        self.classifier_2 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(in_features, config.num_classes_2)
        )
        self.classifier_2.train()

        # unfreeze classifier parameters explicitly
        for param in self.classifier_1.parameters():
            param.requires_grad = True
        for param in self.classifier_2.parameters():
            param.requires_grad = True

        # use crossentropy loss. Performs better in classification
        self.loss_fn = nn.CrossEntropyLoss() # just use crossentropy for binary classification
        self._init_weights(self.classifier_1)
        self._init_weights(self.classifier_2)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, pixel_values, labels_1=None, labels_2=None, **kwargs) -> MultiTaskOutput:
        features = self.pool(self.backbone(pixel_values))

        logits_1 = self.classifier_1(features)
        logits_2 = self.classifier_2(features)
        loss = None
        if labels_1 is not None and labels_2 is not None:
            # pytorch handles one hot internally
            loss_1 = self.loss_fn(logits_1, labels_1)
            loss_2 = self.loss_fn(logits_2, labels_2)
            loss = loss_1 + loss_2

        # return custom output dataclass
        return MultiTaskOutput(
            loss=loss,
            logits_1=logits_1,
            logits_2=logits_2,
        )
