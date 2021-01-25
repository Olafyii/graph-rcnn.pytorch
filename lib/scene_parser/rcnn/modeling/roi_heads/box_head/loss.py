# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from lib.scene_parser.rcnn.layers import smooth_l1_loss
from lib.scene_parser.rcnn.modeling.box_coder import BoxCoder
from lib.scene_parser.rcnn.modeling.matcher import Matcher
from lib.scene_parser.rcnn.structures.boxlist_ops import boxlist_iou
from lib.scene_parser.rcnn.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from lib.scene_parser.rcnn.modeling.utils import cat

from lib.utils.debug_tools import inspect

class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        # inspect('target loss box_head line 40', target)  # ['labels', 'pred_labels', 'relation_labels', 'attrs']
        # inspect('target.get_field(\'attrs\'', target.get_field('attrs'))  # type: torch.Tensor, size: torch.Size([6, 77])

        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(["labels", 'attrs'])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        attr_dist = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")

            attrs_per_image = matched_targets.get_field("attrs")

            # print('labels_per_image.size()', labels_per_image.size())
            # print('proposal_per_image', proposals_per_image)
            # print('target_per_image', targets_per_image)
            # print('labels_per_image info:', labels_per_image.size(), labels_per_image.min(), labels_per_image.max())
            # print('matched_idxs info:', matched_idxs.size(), matched_idxs.min(), matched_idxs.max())
            # raise RuntimeError("loss.py line 68")

            labels_per_image = labels_per_image.to(dtype=torch.int64)
            attrs_per_image = attrs_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0
            # inspect('labels_per_image in box head loss.py line 83', labels_per_image)  # type: torch.Tensor, size: torch.Size([1404])
            attrs_per_image[bg_inds] = -1  # background, irrelevent to all attributes
            # inspect('attrs_per_image in box head loss.py line 83', attrs_per_image)  # type: torch.Tensor, size: torch.Size([1578, 77])

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler
            attrs_per_image[ignore_inds] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            attr_dist.append(attrs_per_image)

        return labels, regression_targets, attr_dist

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets, attr_dist = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # for label in labels:
        #     print('labels', label.size())
        # for pos_ind in sampled_pos_inds:
        #     print('pos inds', pos_ind.sum(), pos_ind.size())
        # for neg_ind in sampled_neg_inds:
        #     print('neg inds', neg_ind.sum(), neg_ind.size())
        # raise RuntimeError('loss.py line 105')

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, attrs_per_image, proposals_per_image in zip(
            labels, regression_targets, attr_dist, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )
            proposals_per_image.add_field("attrs", attrs_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img.view(-1) | neg_inds_img.view(-1)).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        # print(self._proposals[0], self._proposals[0].get_field('objectness').size(), self._proposals[0].get_field('labels').size(), self._proposals[0].get_field('regression_targets').size())
        # raise RuntimeError('loss.py line 132')
        return proposals

    def prepare_labels(self, proposals, targets):
        """
        This method prepares the ground-truth labels for each bounding box, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets, attr_dist = self.prepare_targets(proposals, targets)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, attrs_per_image, proposals_per_image in zip(
            labels, regression_targets, attr_dist, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )
            proposals_per_image.add_field("attrs", attrs_per_image)

        return proposals

    def __call__(self, class_logits, box_regression, attr_distributions):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        # inspect('attr_distributions in box head loss.py line 189', attr_distributions)  # type: <class 'list'>, len: 77
        # First element of attr_distributions in box head loss.py line 189 type: torch.Tensor, size: torch.Size([256, 7])
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        classification_loss = F.cross_entropy(class_logits, labels)

        attr_loss = 0
        attr_gt = cat([proposal.get_field('attrs') for proposal in proposals], dim=0)
        # inspect('attr_gt in box head loss.py line 207', attr_gt)  # type: torch.Tensor, size: torch.Size([256, 77])
        # inspect('attr_distributions in box head loss.py line 208', attr_distributions)  # type: <class 'list'>, len: 77
        # First element of attr_distributions in box head loss.py line 208 type: torch.Tensor, size: torch.Size([256, 7])
        # inspect('labels in box head loss.py line 209', labels)  # type: torch.Tensor, size: torch.Size([256])
        for i, dist in enumerate(attr_distributions):
            attr_loss += F.cross_entropy(attr_distributions[i], attr_gt[:, i]+1)


        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss, attr_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
