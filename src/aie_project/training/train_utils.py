from .configs import EfficientMultiTaskClassificationConfig
from .constants import IMG_SIZE
from .models import EfficientMultiTaskClassificationModel


def model_factory(backbone_arch: str = "mobilenetv3_large_100",
                  pretrained: bool = True,
                  img_size: int = IMG_SIZE,
                  num_classes_1: int = 14,
                  num_classes_2: int = 2,
                  label2id_1: dict = None,
                  id2label_1: dict = None,
                  label2id_2: dict = None,
                  id2label_2: dict = None,
                  classifier_dropout: float = 0.2) -> EfficientMultiTaskClassificationModel:
    # utility function to construct model with given parameters
    config = EfficientMultiTaskClassificationConfig(
        backbone_name=backbone_arch,
        pretrained=pretrained,
        img_size=img_size,
        num_classes_1=num_classes_1,
        num_classes_2=num_classes_2,
        label2id_1=label2id_1,
        id2label_1=id2label_1,
        label2id_2=label2id_2,
        id2label_2=id2label_2,
        classifier_dropout=classifier_dropout,
    )
    model = EfficientMultiTaskClassificationModel(config)
    return model
