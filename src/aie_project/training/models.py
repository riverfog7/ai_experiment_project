import timm
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput

from .configs import EfficientClassificationConfig


class EfficientClassificationModel(PreTrainedModel):
    config_class = EfficientClassificationConfig

    def __init__(self, config: EfficientClassificationConfig):
        super().__init__(config)

        self.img_size = config.img_size
        # see https://github.com/huggingface/pytorch-image-models/issues/220
        # no classifier head
        self.backbone = timm.create_model(
            config.backbone_name,
            pretrained=config.pretrained,
            num_classes=0,
            global_pool="",
        )

        # Run a dummy input to check output shape
        self.pool = None
        in_features = None
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
                self.has_internal_pooling = False
            elif len(features.shape) == 2:
                # Shape is (B, Features) -> Already pooled
                self.pool = nn.Identity()
                in_features = features.shape[1]
                self.has_internal_pooling = True
            else:
                raise ValueError(
                    f"Unexpected feature shape from backbone: {features.shape}"
                )
        # Assertions for easy debugging
        assert self.pool is not None, "Pooling layer not initialized."
        assert in_features is not None, "in_features not determined."
        self.pool.train()

        # Always unfreeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = True

        # custom classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(in_features, config.num_classes)
        )
        self.classifier.train()

        # unfreeze classifier parameters explicitly
        for param in self.classifier.parameters():
            param.requires_grad = True

        # use crossentropy loss. Performs better in classification
        self.loss_fn = nn.CrossEntropyLoss()
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, pixel_values, labels=None, **kwargs):
        # Always pool. Will be set to Identity if not needed.
        features = self.pool(self.backbone(pixel_values))

        logits = self.classifier(features)
        loss = None
        if labels is not None:
            # pytorch handles one hot internally
            loss = self.loss_fn(logits, labels)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )
