# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from lib.scene_parser.rcnn.modeling import registry
from torch import nn
import dill


@registry.ROI_RELATION_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        # num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        # self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        # nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        # nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        return cls_logit
        # bbox_pred = self.bbox_pred(x)
        # return cls_logit, bbox_pred


@registry.ROI_RELATION_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

@registry.ROI_RELATION_BOX_PREDICTOR.register("NSMPredictor")
class NSMPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(NSMPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)

        if len(config.MODEL.ROI_HEADS.GROUP_ATTR_RETURN) == 0:
            raise RuntimeError('config.MODEL.ROI_HEADS.GROUP_ATTR_RETURN is None!')
        self.attr_cls = []
        f = open(config.MODEL.ROI_HEADS.GROUP_ATTR_RETURN, 'rb')
        attr_book, attr_to_idx, idx_to_attr, group_size = dill.load(f)
        f.close()
        self.group_size = group_size
        for i, attr_size in enumerate(group_size):
            exec('self.attr_cls_{} = nn.Linear(num_inputs, attr_size+1)'.format(i))
            exec('nn.init.normal_(self.attr_cls_{}.weight, mean=0, std=0.01)'.format(i))
            exec('nn.init.constant_(self.attr_cls_{}.bias, 0)'.format(i))
        # print(self.attr_cls)
        # raise RuntimeError('roi_relation_box_predictors.py line 79')

        # num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        # self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        # nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        # nn.init.constant_(self.bbox_pred.bias, 0)

        # print(self)
        # raise RuntimeError('roi_relation_box_predictors.py line 97')

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        attr_distributions = []
        for i in range(len(self.group_size)):
            exec('attr_distributions.append(self.attr_cls_{}(x))'.format(i))
        return cls_logit, attr_distributions
        # bbox_pred = self.bbox_pred(x)
        # return cls_logit, bbox_pred

def make_roi_relation_box_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)
