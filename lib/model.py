import os
import datetime
import logging
import time
import numpy as np
import torch
import cv2
from .data.build import build_data_loader
from .scene_parser.parser import build_scene_parser
from .scene_parser.parser import build_scene_parser_optimizer
from .scene_parser.rcnn.utils.metric_logger import MetricLogger
from .scene_parser.rcnn.utils.timer import Timer, get_time_str
from .scene_parser.rcnn.utils.comm import synchronize, all_gather, is_main_process, get_world_size
from .scene_parser.rcnn.utils.visualize import select_top_predictions, overlay_boxes, overlay_class_names
from .data.evaluation import evaluate, evaluate_sg
from .utils.box import bbox_overlaps

from .scene_parser.rcnn.structures.image_list import to_image_list

import dill
from .utils.debug_tools import inspect

class SceneGraphGeneration:
    """
    Scene graph generation
    """
    def __init__(self, cfg, arguments, local_rank, distributed):
        """
        initialize scene graph generation model
        """
        self.cfg = cfg
        self.arguments = arguments.copy()
        self.device = torch.device("cuda")

        # build data loader
        self.data_loader_train = build_data_loader(cfg, split="train", is_distributed=distributed)
        self.data_loader_test = build_data_loader(cfg, split="test", is_distributed=distributed)

        cfg.DATASET.IND_TO_OBJECT = self.data_loader_train.dataset.ind_to_classes
        cfg.DATASET.IND_TO_PREDICATE = self.data_loader_train.dataset.ind_to_predicates

        logger = logging.getLogger("scene_graph_generation.trainer")
        logger.info("Train data size: {}".format(len(self.data_loader_train.dataset)))
        logger.info("Test data size: {}".format(len(self.data_loader_test.dataset)))

        if not os.path.exists("freq_prior.npy"):
            logger.info("Computing frequency prior matrix...")
            fg_matrix, bg_matrix = self._get_freq_prior()
            prob_matrix = fg_matrix.astype(np.float32)
            prob_matrix[:,:,0] = bg_matrix

            prob_matrix[:,:,0] += 1
            prob_matrix /= np.sum(prob_matrix, 2)[:,:,None]
            # prob_matrix /= float(fg_matrix.max())
            np.save("freq_prior.npy", prob_matrix)

        # build scene graph generation model
        self.scene_parser = build_scene_parser(cfg); self.scene_parser.to(self.device)
        # print('self.scene_parser', self.scene_parser)
        """
        self.scene_parser SceneParser(
        (backbone): Sequential(
            (body): ResNet(
            (stem): StemWithFixedBatchNorm(
                (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                (bn1): FrozenBatchNorm2d()
            )
            (layer1): Sequential(
                (0): BottleneckWithFixedBatchNorm(
                (downsample): Sequential(
                    (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (1): FrozenBatchNorm2d()
                )
                (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (1): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (2): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
            )
            (layer2): Sequential(
                (0): BottleneckWithFixedBatchNorm(
                (downsample): Sequential(
                    (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                    (1): FrozenBatchNorm2d()
                )
                (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (1): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (2): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (3): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
            )
            (layer3): Sequential(
                (0): BottleneckWithFixedBatchNorm(
                (downsample): Sequential(
                    (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
                    (1): FrozenBatchNorm2d()
                )
                (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (1): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (2): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (3): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (4): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (5): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (6): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (7): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (8): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (9): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (10): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (11): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (12): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (13): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (14): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (15): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (16): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (17): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (18): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (19): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (20): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (21): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
                (22): BottleneckWithFixedBatchNorm(
                (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn1): FrozenBatchNorm2d()
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): FrozenBatchNorm2d()
                (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (bn3): FrozenBatchNorm2d()
                )
            )
            )
        )
        (rpn): RPNModule(
            (anchor_generator): AnchorGenerator(
            (cell_anchors): BufferList()
            )
            (head): RPNHead(
            (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (cls_logits): Conv2d(1024, 15, kernel_size=(1, 1), stride=(1, 1))
            (bbox_pred): Conv2d(1024, 60, kernel_size=(1, 1), stride=(1, 1))
            )
            (box_selector_train): RPNPostProcessor()
            (box_selector_test): RPNPostProcessor()
        )
        (roi_heads): CombinedROIHeads(
            (box): ROIBoxHead(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (feature_extractor): ResNet50Conv5ROIFeatureExtractor(
                (pooler): Pooler(
                (poolers): ModuleList(
                    (0): ROIAlign(output_size=(14, 14), spatial_scale=0.0625, sampling_ratio=0)
                )
                )
                (head): ResNetHead(
                (layer4): Sequential(
                    (0): BottleneckWithFixedBatchNorm(
                    (downsample): Sequential(
                        (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
                        (1): FrozenBatchNorm2d()
                    )
                    (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                    (bn1): FrozenBatchNorm2d()
                    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (bn2): FrozenBatchNorm2d()
                    (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn3): FrozenBatchNorm2d()
                    )
                    (1): BottleneckWithFixedBatchNorm(
                    (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn1): FrozenBatchNorm2d()
                    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (bn2): FrozenBatchNorm2d()
                    (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn3): FrozenBatchNorm2d()
                    )
                    (2): BottleneckWithFixedBatchNorm(
                    (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn1): FrozenBatchNorm2d()
                    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (bn2): FrozenBatchNorm2d()
                    (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn3): FrozenBatchNorm2d()
                    )
                )
                )
            )
            (predictor): FastRCNNPredictor(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (cls_score): Linear(in_features=2048, out_features=151, bias=True)
                (bbox_pred): Linear(in_features=2048, out_features=604, bias=True)
            )
            (post_processor): PostProcessor()
            )
        )
        (rel_heads): ROIRelationHead(
            (rel_predictor): GRCNN(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (pred_feature_extractor): ResNet50Conv5ROIFeatureExtractor(
                (pooler): Pooler(
                (poolers): ModuleList(
                    (0): ROIAlign(output_size=(14, 14), spatial_scale=0.0625, sampling_ratio=0)
                )
                )
                (head): ResNetHead(
                (layer4): Sequential(
                    (0): BottleneckWithFixedBatchNorm(
                    (downsample): Sequential(
                        (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
                        (1): FrozenBatchNorm2d()
                    )
                    (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                    (bn1): FrozenBatchNorm2d()
                    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (bn2): FrozenBatchNorm2d()
                    (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn3): FrozenBatchNorm2d()
                    )
                    (1): BottleneckWithFixedBatchNorm(
                    (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn1): FrozenBatchNorm2d()
                    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (bn2): FrozenBatchNorm2d()
                    (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn3): FrozenBatchNorm2d()
                    )
                    (2): BottleneckWithFixedBatchNorm(
                    (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn1): FrozenBatchNorm2d()
                    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                    (bn2): FrozenBatchNorm2d()
                    (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (bn3): FrozenBatchNorm2d()
                    )
                )
                )
            )
            (obj_embedding): Sequential(
                (0): Linear(in_features=2048, out_features=1024, bias=True)
                (1): ReLU(inplace=True)
                (2): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (rel_embedding): Sequential(
                (0): Linear(in_features=2048, out_features=1024, bias=True)
                (1): ReLU(inplace=True)
                (2): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (gcn_collect_feat): _GraphConvolutionLayer_Collect(
                (collect_units): ModuleList(
                (0): _Collection_Unit(
                    (fc): Linear(in_features=1024, out_features=1024, bias=True)
                )
                (1): _Collection_Unit(
                    (fc): Linear(in_features=1024, out_features=1024, bias=True)
                )
                (2): _Collection_Unit(
                    (fc): Linear(in_features=1024, out_features=1024, bias=True)
                )
                (3): _Collection_Unit(
                    (fc): Linear(in_features=1024, out_features=1024, bias=True)
                )
                (4): _Collection_Unit(
                    (fc): Linear(in_features=1024, out_features=1024, bias=True)
                )
                )
            )
            (gcn_update_feat): _GraphConvolutionLayer_Update(
                (update_units): ModuleList(
                (0): _Update_Unit()
                (1): _Update_Unit()
                )
            )
            (gcn_collect_score): _GraphConvolutionLayer_Collect(
                (collect_units): ModuleList(
                (0): _Collection_Unit(
                    (fc): Linear(in_features=51, out_features=151, bias=True)
                )
                (1): _Collection_Unit(
                    (fc): Linear(in_features=51, out_features=151, bias=True)
                )
                (2): _Collection_Unit(
                    (fc): Linear(in_features=151, out_features=51, bias=True)
                )
                (3): _Collection_Unit(
                    (fc): Linear(in_features=151, out_features=51, bias=True)
                )
                (4): _Collection_Unit(
                    (fc): Linear(in_features=151, out_features=151, bias=True)
                )
                )
            )
            (gcn_update_score): _GraphConvolutionLayer_Update(
                (update_units): ModuleList(
                (0): _Update_Unit()
                (1): _Update_Unit()
                )
            )
            (obj_predictor): FastRCNNPredictor(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (cls_score): Linear(in_features=1024, out_features=151, bias=True)
            )
            (pred_predictor): FastRCNNPredictor(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (cls_score): Linear(in_features=1024, out_features=51, bias=True)
            )
            )
            (post_processor): PostProcessor()
            (relpn): RelPN(
            (relationshipness): Relationshipness(
                (subj_proj): Sequential(
                (0): Linear(in_features=151, out_features=64, bias=True)
                (1): ReLU(inplace=True)
                (2): Linear(in_features=64, out_features=64, bias=True)
                )
                (obj_prof): Sequential(
                (0): Linear(in_features=151, out_features=64, bias=True)
                (1): ReLU(inplace=True)
                (2): Linear(in_features=64, out_features=64, bias=True)
                )
                (sub_pos_encoder): Sequential(
                (0): Linear(in_features=6, out_features=64, bias=True)
                (1): ReLU(inplace=True)
                (2): Linear(in_features=64, out_features=64, bias=True)
                )
                (obj_pos_encoder): Sequential(
                (0): Linear(in_features=6, out_features=64, bias=True)
                (1): ReLU(inplace=True)
                (2): Linear(in_features=64, out_features=64, bias=True)
                )
            )
            )
        )
        )
        """
        self.sp_optimizer, self.sp_scheduler, self.sp_checkpointer, self.extra_checkpoint_data = \
            build_scene_parser_optimizer(cfg, self.scene_parser, local_rank=local_rank, distributed=distributed)

        self.arguments.update(self.extra_checkpoint_data)

    def _get_freq_prior(self, must_overlap=False):

        fg_matrix = np.zeros((
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
            ), dtype=np.int64)

        bg_matrix = np.zeros((
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
        ), dtype=np.int64)

        for ex_ind in range(len(self.data_loader_train.dataset)):
            gt_classes = self.data_loader_train.dataset.gt_classes[ex_ind].copy()
            gt_relations = self.data_loader_train.dataset.relationships[ex_ind].copy()
            gt_boxes = self.data_loader_train.dataset.gt_boxes[ex_ind].copy()

            # For the foreground, we'll just look at everything
            o1o2 = gt_classes[gt_relations[:, :2]]
            for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
                fg_matrix[o1, o2, gtr] += 1

            # For the background, get all of the things that overlap.
            o1o2_total = gt_classes[np.array(
                self._box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
            for (o1, o2) in o1o2_total:
                bg_matrix[o1, o2] += 1

            if ex_ind % 20 == 0:
                print("processing {}/{}".format(ex_ind, len(self.data_loader_train.dataset)))

        return fg_matrix, bg_matrix

    def _box_filter(self, boxes, must_overlap=False):
        """ Only include boxes that overlap as possible relations.
        If no overlapping boxes, use all of them."""
        n_cands = boxes.shape[0]

        overlaps = bbox_overlaps(torch.from_numpy(boxes.astype(np.float)), torch.from_numpy(boxes.astype(np.float))).numpy() > 0
        np.fill_diagonal(overlaps, 0)

        all_possib = np.ones_like(overlaps, dtype=np.bool)
        np.fill_diagonal(all_possib, 0)

        if must_overlap:
            possible_boxes = np.column_stack(np.where(overlaps))

            if possible_boxes.size == 0:
                possible_boxes = np.column_stack(np.where(all_possib))
        else:
            possible_boxes = np.column_stack(np.where(all_possib))
        return possible_boxes

    def debug(self):
        """
        main body for training scene graph generation model
        """
        start_iter = self.arguments["iteration"]
        logger = logging.getLogger("scene_graph_generation.trainer")
        logger.info("Start training")
        meters = MetricLogger(delimiter="  ")
        max_iter = len(self.data_loader_train)
        self.scene_parser.train()
        start_training_time = time.time()
        end = time.time()
        for i, data in enumerate(self.data_loader_train, start_iter):
            print(i)
            data_time = time.time() - end
            self.arguments["iteration"] = i
            self.sp_scheduler.step()
            imgs, targets, _ = data
            imgs = imgs.to(self.device); targets = [target.to(self.device) for target in targets]

            images = to_image_list(imgs)

            features = self.scene_parser.backbone(images.tensors)
            for feature in features:
                assert not torch.isnan(feature).all(), 'bad feature!'

    def train(self):
        """
        main body for training scene graph generation model
        """
        start_iter = self.arguments["iteration"]
        logger = logging.getLogger("scene_graph_generation.trainer")
        logger.info("Start training")
        meters = MetricLogger(delimiter="  ")
        max_iter = len(self.data_loader_train)
        self.scene_parser.train()
        start_training_time = time.time()
        end = time.time()
        # print('self.scene_parser', self.scene_parser)
        # raise RuntimeError("model.py line 166")
        for i, data in enumerate(self.data_loader_train, start_iter):
            data_time = time.time() - end
            self.arguments["iteration"] = i
            self.sp_scheduler.step()
            imgs, targets, _ = data

            # for target in targets:
            #     print(target.fields())
            #     print(target.get_field('labels'))
            #     print(target.get_field('pred_labels'))
            #     print(target.get_field('relation_labels'))
            # raise RuntimeError('model.py line 163')

            imgs = imgs.to(self.device); targets = [target.to(self.device) for target in targets]

            # images = to_image_list(imgs)
            # features = self.scene_parser.backbone(images.tensors)
            # for feature in features:
            #     assert not torch.isnan(feature).all(), 'bad feature! in model.py line 163'

            loss_dict = self.scene_parser(imgs, targets)
            # print(loss_dict)
            # {'loss_classifier': tensor(5.3355, device='cuda:0', grad_fn=<NllLossBackward>), 'loss_box_reg': tensor(0.3237, device='cuda:0', grad_fn=<DivBackward0>), 'loss_obj_classifier': 0, 'loss_relpn': tensor(1.9932, device='cuda:0', grad_fn=<AddBackward0>), 'loss_pred_classifier': tensor(3.9704, device='cuda:0', grad_fn=<NllLossBackward>), 'loss_objectness': tensor(0.6694, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>), 'loss_rpn_box_reg': tensor(0.2562, device='cuda:0', grad_fn=<DivBackward0>)}
            # raise RuntimeError("model.py line 167")
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = loss_dict
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            self.sp_optimizer.zero_grad()
            losses.backward()
            self.sp_optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - i)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if i % 20 == 0 or i == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "model: {tag}",
                            "eta: {eta}",
                            "iter: {iter}/{max_iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        tag="scene_parser",
                        eta=eta_string,
                        iter=i, max_iter=max_iter,
                        meters=str(meters),
                        lr=self.sp_optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if (i + 1) % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                self.sp_checkpointer.save("checkpoint_{:07d}".format(i), **self.arguments)
            if (i + 1) == max_iter:
                self.sp_checkpointer.save("checkpoint_final", **self.arguments)

    def _accumulate_predictions_from_multiple_gpus(self, predictions_per_gpu):
        all_predictions = all_gather(predictions_per_gpu)
        if not is_main_process():
            return
        # merge the list of dicts
        predictions = {}
        for p in all_predictions:
            predictions.update(p)
        # convert a dict where the key is the index in a list
        image_ids = list(sorted(predictions.keys()))
        if len(image_ids) != image_ids[-1] + 1:
            logger = logging.getLogger("scene_graph_generation.inference")
            logger.warning(
                "Number of images that were gathered from multiple processes is not "
                "a contiguous set. Some images might be missing from the evaluation"
            )

        # convert to a list
        predictions = [predictions[i] for i in image_ids]
        return predictions

    def visualize_detection(self, dataset, img_ids, imgs, predictions):
        visualize_folder = "visualize"
        if not os.path.exists(visualize_folder):
            os.mkdir(visualize_folder)
        for i, prediction in enumerate(predictions):
            top_prediction = select_top_predictions(prediction)
            img = imgs.tensors[i].permute(1, 2, 0).contiguous().cpu().numpy() + np.array(self.cfg.INPUT.PIXEL_MEAN).reshape(1, 1, 3)
            result = img.copy()
            result = overlay_boxes(result, top_prediction)
            result = overlay_class_names(result, top_prediction, dataset.ind_to_classes)
            cv2.imwrite(os.path.join(visualize_folder, "detection_{}.jpg".format(img_ids[i])), result)

    def test(self, timer=None, visualize=False):
        """
        main body for testing scene graph generation model
        """
        logger = logging.getLogger("scene_graph_generation.inference")
        logger.info("Start evaluating")
        self.scene_parser.eval()
        targets_dict = {}
        results_dict = {}
        if self.cfg.MODEL.RELATION_ON:
            results_pred_dict = {}
        cpu_device = torch.device("cpu")
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        reg_recalls = []
        for i, data in enumerate(self.data_loader_test, 0):
            imgs, targets, image_ids = data
            imgs = imgs.to(self.device); targets = [target.to(self.device) for target in targets]
            if i % 10 == 0:
                logger.info("inference on batch {}/{}...".format(i, len(self.data_loader_test)))
            with torch.no_grad():
                if timer:
                    timer.tic()
                output = self.scene_parser(imgs)
                # f = open('C:\\Users\\80409\\Desktop\\temp\\ot.dill', 'wb')
                # dill.dump([output, targets], f)
                # f.close()
                # print(output)
                # print(targets)
                # raise RuntimeError('stop!')
                # print('output', output[0][0].fields(), output)
                # print('output scores', output[0][0].get_field('scores'))
                # print('output labels', output[0][0].get_field('labels'))
                # print('target labels', targets[0].get_field('labels'))
                # print('targets', targets[0].fields())
                # raise RuntimeError('model.py line 282')
                if self.cfg.MODEL.RELATION_ON:
                    output, output_pred = output
                    output_pred = [o.to(cpu_device) for o in output_pred]
                ious = bbox_overlaps(targets[0].bbox, output[0].bbox)
                reg_recall = (ious.max(1)[0] > 0.5).sum().item() / ious.shape[0]
                reg_recalls.append(reg_recall)
                if timer:
                    torch.cuda.synchronize()
                    timer.toc()
                output = [o.to(cpu_device) for o in output]
                # print('output', output)
                # print('output labels', output[0].get_field('labels'))
                if visualize:
                    self.visualize_detection(self.data_loader_test.dataset, image_ids, imgs, output)
                    # print('targets', targets)
                    # print('output', output)
                    # targets[0].add_field('scores', torch.ones((len(targets[0]))))
                    # targets = [t.to(cpu_device) for t in targets]
                    # self.visualize_detection(self.data_loader_test.dataset, image_ids, imgs, targets)
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
            targets_dict.update(
                {img_id: target for img_id, target in zip(image_ids, targets)}
            )
            if self.cfg.MODEL.RELATION_ON:
                results_pred_dict.update(
                    {img_id: result for img_id, result in zip(image_ids, output_pred)}
                )
            if self.cfg.instance > 0 and i > self.cfg.instance:
                break
        synchronize()
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        num_devices = get_world_size()
        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(self.data_loader_test.dataset), num_devices
            )
        )
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(self.data_loader_test.dataset),
                num_devices,
            )
        )
        predictions = self._accumulate_predictions_from_multiple_gpus(results_dict)
        if self.cfg.MODEL.RELATION_ON:
            predictions_pred = self._accumulate_predictions_from_multiple_gpus(results_pred_dict)
        if not is_main_process():
            return

        output_folder = "results"
        if output_folder:
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
            if self.cfg.MODEL.RELATION_ON:
                torch.save(predictions_pred, os.path.join(output_folder, "predictions_pred.pth"))

        extra_args = dict(
            box_only=False if self.cfg.MODEL.RETINANET_ON else self.cfg.MODEL.RPN_ONLY,
            iou_types=("bbox",),
            expected_results=self.cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=self.cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        )
        eval_det_results = evaluate(dataset=self.data_loader_test.dataset,
                        predictions=predictions,
                        output_folder=output_folder,
                        **extra_args)

        if self.cfg.MODEL.RELATION_ON:
            eval_sg_results = evaluate_sg(dataset=self.data_loader_test.dataset,
                            predictions=predictions,
                            predictions_pred=predictions_pred,
                            output_folder=output_folder,
                            **extra_args)

def build_model(cfg, arguments, local_rank, distributed):
    return SceneGraphGeneration(cfg, arguments, local_rank, distributed)
