from transformers import PretrainedConfig
from .constants import IMG_SIZE


class EfficientClassificationConfig(PretrainedConfig):
    model_type = "efficient_classification"

    def __init__(
        self,
        backbone_name: str = "mobilenetv3_large_100",
        num_classes: int = 32,
        pretrained: bool = True,
        img_size: int = IMG_SIZE,
        classifier_dropout: float = 0.2,
        **kwargs,
    ):
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.img_size = img_size
        self.classifier_dropout = classifier_dropout
        super().__init__(**kwargs)
