import torch
from torch import nn

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import  retinanet_resnet50_fpn


def FRCNN_resnet50_fpn(pre_trained=False, pretrained_backbone=True):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pre_trained, pretrained_backbone = pretrained_backbone)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (wheat heads) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def FRCNN_resnet_backbone(backbone_name='resnet_101', pre_trained=True):
    
    # Reference: https://stackoverflow.com/questions/58362892/resnet-18-as-backbone-in-faster-r-cnn
    # https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py
    
    RESNET_SMALL_MODELS = ["resnet18", "resnet34"]

    RESNET_LARGE_MODELS = ["resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d",
                       "wide_resnet50_2", "wide_resnet101_2"]
    

    if backbone_name == 'resnet_18':
        resnet_net = torchvision.models.resnet18(pretrained=pre_trained)    

    elif backbone_name == 'resnet_34':
        resnet_net = torchvision.models.resnet34(pretrained=pre_trained)
    
    elif backbone_name == 'resnet_50':
        resnet_net = torchvision.models.resnet50(pretrained=pre_trained)

    elif backbone_name == 'resnet_101':
        resnet_net = torchvision.models.resnet101(pretrained=pre_trained)

    elif backbone_name == 'resnet_152':
        resnet_net = torchvision.models.resnet152(pretrained=pre_trained)

    elif backbone_name == 'wide_resnet_50_2':
        resnet_net = torchvision.models.wide_resnet_50_2(pretrained=pre_trained)

    elif backbone_name == 'resnext101_32x8d':
        resnet_net = torchvision.models.resnext101_32x8d(pretrained=pre_trained)
    
    elif backbone_name == 'wide_resnet101_2':
        resnet_net = torchvision.models.wide_resnet101_2(pretrained=pre_trained)

    out_channels = 2048
    if(backbone_name in RESNET_SMALL_MODELS):
        out_channels = 512

    modules = list(resnet_net.children())[:-2]
    backbone = nn.Sequential(*modules)
    backbone.out_channels = out_channels

    # sizes=((32, 64, 128, 256, 512),), or in other words defines 1 level/group of 5 anchor-sizes.
    # Why 1 level/group? Because the backbone provides only 1 ouput.
    # The default anchors used in faster-rcnn is ((32,), (64,), (128,), (256,), (512,)) 
    # which means we have 5 levels/groups with 1 anchor size.
    # Why? Because by default FasterRCNN uses a Feature Pyramid as a backbone 
    # which returns 5 outputs (intermediate layers of the original backbone).
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
  

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator)
    return model


def FRCNN_resnetfpn_backbone(backbone_name='resnet101', pre_trained=True):

    # Reference: https://github.com/pytorch/vision/blob/master/torchvision/models/detection/backbone_utils.py

    backbone = resnet_fpn_backbone(backbone_name, pretrained=pre_trained)
    """
    resnet_fpn_bacbone:
    Args:
        backbone_name (string): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
        norm_layer (torchvision.ops): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        trainable_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. default=3
        returned_layers (list of int): The layers of the network to return. Each entry must be in ``[1, 4]``.
            By default all layers are returned.
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names. By
            default a ``LastLevelMaxPool`` is used.
    """
    model = FasterRCNN(backbone,
                       num_classes=2)
    return model

def Retinanet_resnet50_fpn(num_classes=2, pretrained_backbone=True):

    model =  retinanet_resnet50_fpn(pretrained=False, progress=True,
                           num_classes=num_classes, pretrained_backbone=pretrained_backbone)
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    return model


