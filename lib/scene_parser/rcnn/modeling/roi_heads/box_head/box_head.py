# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

from lib.utils.debug_tools import inspect

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.cfg = cfg
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        # print('len(proposals)', len(proposals))
        # for proposal in proposals:
        #     print(proposal)
        # print('len(features)', len(features))
        # for feature in features:
        #     print(feature.size())
        # print('len(targets)', len(targets))
        # for target in targets:
        #     print(target)
        # raise RuntimeError("box_head.py line 41")
        if self.training: # or not self.cfg.inference:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                # print('box_head.py line 54 proposals before loss_evaluator subsample', proposals)  # [BoxList(num_boxes=1059, image_width=1024, image_height=681, mode=xyxy), BoxList(num_boxes=949, image_width=1024, image_height=768, mode=xyxy)]
                # print(proposals[0].fields())  # ['objectness']
                proposals = self.loss_evaluator.subsample(proposals, targets)
                # print('box_head.py line 56 proposals after loss_evaluator subsample', proposals)  # [BoxList(num_boxes=128, image_width=1024, image_height=681, mode=xyxy), BoxList(num_boxes=128, image_width=1024, image_height=768, mode=xyxy)]
                # print(proposals[0].fields())  # ['objectness', 'labels', 'regression_targets', 'attrs']
                # inspect('objectness', proposals[0].get_field('objectness'))  # type: torch.Tensor, size: torch.Size([128])
                # inspect('labels', proposals[0].get_field('labels'))  # type: torch.Tensor, size: torch.Size([128])
                # inspect('regression_targets', proposals[0].get_field('regression_targets'))  # type: torch.Tensor, size: torch.Size([128, 4])
                # inspect('attrs', proposals[0].get_field('attrs'))

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # inspect('x', x)  # type: torch.Tensor, size: torch.Size([256, 2048, 7, 7])
        # final classifier that converts the features into predictions
        # print('self.predictor', self.predictor)
        # self.predictor FastRCNNPredictor(
        # (avgpool): AdaptiveAvgPool2d(output_size=1)
        # (cls_score): Linear(in_features=2048, out_features=151, bias=True)
        # (bbox_pred): Linear(in_features=2048, out_features=604, bias=True)
        # )
        # raise RuntimeError("STOP!!")
        class_logits, box_regression, attr_distributions = self.predictor(x)
        # inspect('class_logits', class_logits)  # type: torch.Tensor, size: torch.Size([256, 151])
        # inspect('box_regression', box_regression)  # type: torch.Tensor, size: torch.Size([256, 604])
        # inspect('attr_distributions', attr_distributions)  # type: <class 'list'>, len: 77
        # First element of attr_distributions type: torch.Tensor, size: torch.Size([256, 7])

        boxes_per_image = [len(proposal) for proposal in proposals]
        features = x.split(boxes_per_image, dim=0)
        for proposal, feature in zip(proposals, features):
            proposal.add_field("features", self.avgpool(feature))
        if not self.training:
            # if self.cfg.inference:
            result = self.post_processor((class_logits, box_regression), proposals)
            if targets:
                result = self.loss_evaluator.prepare_labels(result, targets)
            return x, result, {}
            # else:
                # return x, proposals, {}

        loss_classifier, loss_box_reg, loss_attr_classifier = self.loss_evaluator(
            [class_logits], [box_regression], attr_distributions
        )
        # inspect('class_logits', class_logits)  # type: torch.Tensor, size: torch.Size([256, 151])
        # inspect('attr_distributions', attr_distributions)  # type: <class 'list'>, len: 77
        # First element of attr_distributions type: torch.Tensor, size: torch.Size([256, 7])
        class_logits = class_logits.split(boxes_per_image, dim=0)  # type: <class 'tuple'>, len: 2
        # inspect('boxes_per_image', boxes_per_image)  # type: <class 'list'>, len: 2
        # First element of boxes_per_image type: <class 'int'>, value: 128
        # First element of class_logits type: torch.Tensor, size: torch.Size([128, 151])
        # inspect('class_logits', class_logits)  # type: torch.Tensor, size: torch.Size([128, 151])
        for proposal, class_logit in zip(proposals, class_logits):
            proposal.add_field("logits", class_logit)
            proposal.add_field("attr_logits", [])

        for attr_distribution in attr_distributions:
            attr_distribution = attr_distribution.split(boxes_per_image, dim=0)
            for proposal, attr_logit in zip(proposals, attr_distribution):
                proposal.get_field("attr_logits").append(attr_logit)
        # inspect('proposals[0].get_field(\'logits\')', proposals[0].get_field('logits'))  # type: torch.Tensor, size: torch.Size([128, 151])
        # inspect('proposals[0].get_field(\'attr_logits\')', proposals[0].get_field('attr_logits'))  # type: <class 'list'>, len: 77
        # # First element of proposals[0].get_field('attr_logits') type: torch.Tensor, size: torch.Size([128, 7])


        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_attr_classifier=loss_attr_classifier),
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
