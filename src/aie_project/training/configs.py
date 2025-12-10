from transformers import PretrainedConfig
from .constants import IMG_SIZE


class EfficientMultiTaskClassificationConfig(PretrainedConfig):
    model_type = "efficient_classification"

    def __init__(
        self,
        backbone_name: str = "mobilenetv3_large_100",
        num_classes_1: int = 14,
        num_classes_2: int = 2,
        id2label_1: dict = None,
        label2id_1: dict = None,
        id2label_2: dict = None,
        label2id_2: dict = None,
        pretrained: bool = True,
        img_size: int = IMG_SIZE,
        classifier_dropout: float = 0.2,
        in_features: int = None,
        has_internal_pooling: bool = None,
        **kwargs,
    ):
        self.backbone_name = backbone_name
        self.num_classes_1 = num_classes_1
        self.num_classes_2 = num_classes_2
        self.id2label_1 = id2label_1
        self.label2id_1 = label2id_1
        self.id2label_2 = id2label_2
        self.label2id_2 = label2id_2
        self.pretrained = pretrained
        self.img_size = img_size
        self.classifier_dropout = classifier_dropout
        self.in_features = in_features
        self.has_internal_pooling = has_internal_pooling

        kwargs.pop("id2label", None)
        kwargs.pop("label2id", None)

        super().__init__(
            label2id=label2id_1,
            id2label=id2label_1,
            **kwargs,
        )
