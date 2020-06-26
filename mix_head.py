import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init

from mmdet.core import force_fp32, multi_apply,multiclass_nms, distance2bbox,images_to_levels
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        force_fp32, images_to_levels, multi_apply,
                        multiclass_nms, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from mmdet.core import bbox_overlaps
from .retina_head import RetinaHead
import numpy as np

INF = 1e8
class anchorBase_retina(RetinaHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                 type='AnchorGenerator',
                 octave_base_scale=4,
                 scales_per_octave=3,
                 ratios=[0.5, 1.0, 2.0],
                 strides=[8, 16, 32, 64, 128]),
                 reg_decoded_bbox=True,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(anchorBase_retina, self).__init__(
            num_classes,
            in_channels,
            stacked_convs=stacked_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            anchor_generator=anchor_generator,
            reg_decoded_bbox=reg_decoded_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

    def anchor_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_weights = bbox_weights[:,0].bool()
        bbox_targets=bbox_targets[bbox_weights]
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_pred = bbox_pred[bbox_weights]
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            anchors = anchors[bbox_weights]
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        # loss_bbox = self.loss_bbox(
        #     bbox_pred,
        #     bbox_targets,
        #     bbox_weights,
        #     avg_factor=num_total_samples)
        return bbox_pred,bbox_targets,loss_cls

    def get_all_anchor(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        bbox_preds,bbox_targets,loss_cls = multi_apply(
            self.anchor_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        bbox_pred=torch.cat((bbox_preds))
        bbox_target=torch.cat((bbox_targets))
        sum_loss=0
        for loss in loss_cls:
            sum_loss+=loss
        loss_cls=sum_loss/len(loss_cls)
        return bbox_pred,bbox_target,loss_cls
class anchorBase(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[8, 16, 32],
                     ratios=[0.5, 1.0, 2.0],
                     strides=[4, 8, 16, 32, 64]),
                 reg_decoded_bbox=False,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(anchorBase, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            reg_decoded_bbox=reg_decoded_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg)

    def anchor_single(self, bbox_pred, anchors,
                    bbox_targets, bbox_weights, ):
        # classification loss
        # labels = labels.reshape(-1)
        # label_weights = label_weights.reshape(-1)
        # cls_score = cls_score.permute(0, 2, 3,
        #                               1).reshape(-1, self.cls_out_channels)
        # loss_cls = self.loss_cls(
        #     cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_weights = bbox_weights[:,0].bool()
        bbox_targets=bbox_targets[bbox_weights]
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_pred = bbox_pred[bbox_weights]
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            anchors = anchors[bbox_weights]
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        # loss_bbox = self.loss_bbox(
        #     bbox_pred,
        #     bbox_targets,
        #     bbox_weights,
        #     avg_factor=num_total_samples)
        return bbox_pred,bbox_targets

    def get_all_anchor(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # num_total_samples = (
        #     num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        bbox_pred,bbox_target = multi_apply(
            self.anchor_single,
            bbox_preds,
            all_anchor_list,
            bbox_targets_list,
            bbox_weights_list,
            )
        bbox_pred=torch.cat((bbox_pred))
        bbox_target=torch.cat((bbox_target))
        return bbox_pred,bbox_target

@HEADS.register_module()
class MixHead(nn.Module):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.

    Example:
        >>> self = MixHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 background_label=None,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 train_cfg=None,
                 test_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[8, 16, 32],
                     ratios=[0.5, 1.0, 2.0],
                     strides=[4, 8, 16, 32, 64]),
    ):
        super(MixHead, self).__init__()
        self.anchor_base=anchorBase_retina(
                 num_classes,
                 in_channels,
                 anchor_generator=anchor_generator,
                 reg_decoded_bbox=True,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.background_label = (
            num_classes if background_label is None else background_label)
        # background_label should be either 0 or num_classes
        assert (self.background_label == 0
                or self.background_label == num_classes)

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        cls_feat_AB=x
        reg_feat_AB=x
        for cls_conv in self.anchor_base.cls_convs:
            cls_feat_AB = cls_conv(cls_feat_AB)
        for reg_conv in self.anchor_base.reg_convs:
            reg_feat_AB = reg_conv(reg_feat_AB)
        cls_score_AB = self.anchor_base.retina_cls(cls_feat_AB)
        bbox_pred_AB = self.anchor_base.retina_reg(reg_feat_AB)
        # cls_score_AB = self.anchor_base.conv_cls(x)
        # bbox_pred_AB = self.anchor_base.conv_reg(x)

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score_AF = self.fcos_cls(cls_feat)
        centerness = self.fcos_centerness(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred_AF = scale(self.fcos_reg(reg_feat)).float().exp()
        cls_score=[cls_score_AB,cls_score_AF]
        bbox_pred=[bbox_pred_AB,bbox_pred_AF]
        return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        cls_scores_AB = [cls_score[0] for cls_score in cls_scores]#anchor base分类得分
        cls_scores_AF = [cls_score[1] for cls_score in cls_scores]#anchor free分类得分
        bbox_preds_AB = [bbox_pred[0] for bbox_pred in bbox_preds]#anchor base回归得分
        bbox_preds_AF = [bbox_pred[1] for bbox_pred in bbox_preds]#anchor free回归得分
        #anchor free部分
        featmap_sizes = [featmap[0].size()[-2:] for featmap in cls_scores]#获得特征图的尺寸
        all_level_points = self.get_points(featmap_sizes, bbox_preds_AF[0].dtype,
                                           bbox_preds_AF[0].device)#获得特征图上的点映射回原图的坐标
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)#获得各级点的标签以及回归目标

        num_imgs = cls_scores_AF[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores_AF#将一个batch中所有图片的分类得分放到一起
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds_AF#将一个batch中所有图片的回归得分放到一起
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses#将一个batch中所有centereness得分放到一起
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)#将各级特征图的分类得分放到同一个张量中
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)#将各级特征图上的回归得分放到同一个张量中
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)#将各个坐标上的分类标签放到同一个张量中
        flatten_bbox_targets = torch.cat(bbox_targets)#将所有回归目标放到同一个张量中
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])#将所有特征图上映射回原图的坐标放到同一个张量中

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes#得到分类类别数
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)#取出flatten_labels中大于 等于0并且小于类别数的
        num_pos = len(pos_inds)#正例子的数量


        pos_bbox_preds = flatten_bbox_preds[pos_inds]#得到正例位置的预测框
        pos_centerness = flatten_centerness[pos_inds]#得到正例对应的centerness值
        bbox_pred,bbox_target,loss_cls_AB=self.anchor_base.get_all_anchor(
            cls_scores_AB,
            bbox_preds_AB,
            gt_bboxes,
            gt_labels,
            img_metas,)#获得anchor base的框,以及回归目标
        loss_cls_AF = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0
        loss_cls = loss_cls_AF + loss_cls_AB

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)#anchor free用于训练的框
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)#回归框的目标
            centerness_target=pos_centerness_targets
            if bbox_pred.shape[0] <= 3000:
                iou = bbox_overlaps(bbox_pred, pos_decoded_bbox_preds)#计算anchor base与anchor free框的Iou
                iou_anchor_max = iou.max(dim=1)
                selected_id = torch.gt(iou_anchor_max.values, 0.2)#选出iou大于0.2的那些anchor base框
                bbox_pred = bbox_pred[selected_id]
                bbox_target = bbox_target[selected_id]#anchor base的目标框
                if bbox_pred.shape[0] > 0:
                    centerness_target = pos_centerness_targets.repeat(bbox_pred.shape[0], 1)
                    centerness_target = centerness_target[0, iou_anchor_max.indices[selected_id]]#得到训练框
                    bbox_pred = torch.cat((bbox_pred,pos_decoded_bbox_preds))
                    bbox_target = torch.cat((bbox_target,pos_decoded_target_preds))
                    centerness_target= torch.cat((centerness_target,pos_centerness_targets))
                else:
                    centerness_target = pos_centerness_targets
                    bbox_pred = pos_decoded_bbox_preds
                    bbox_target = pos_decoded_target_preds
            elif bbox_pred.shape[0] > 3000:
                centerness_target = pos_centerness_targets
                bbox_pred = pos_decoded_bbox_preds
                bbox_target = pos_decoded_target_preds
            # iou = bbox_overlaps(bbox_pred, pos_decoded_bbox_preds)  # 计算anchor base与anchor free框的Iou
            # iou_anchor_max = iou.max(dim=1)
            # selected_id = torch.gt(iou_anchor_max.values, 0.2)#选出iou大于0.2的那些anchor base框
            # bbox_pred = bbox_pred[selected_id]
            # bbox_target = bbox_target[selected_id]#anchor base的目标框
            # if bbox_pred.shape[0] > 0:
            #     centerness_target = pos_centerness_targets.repeat(bbox_pred.shape[0], 1)
            #     centerness_target = centerness_target[0, iou_anchor_max.indices[selected_id]]#得到训练框
            #     bbox_pred = torch.cat((bbox_pred,pos_decoded_bbox_preds))
            #     bbox_target = torch.cat((bbox_target,pos_decoded_target_preds))
            #     centerness_target= torch.cat((centerness_target,pos_centerness_targets))
            # else:
            #     bbox_pred=pos_decoded_bbox_preds
            #     bbox_target=pos_bbox_targets
            loss_bbox = self.loss_bbox(
                bbox_pred,
                bbox_target,
                weight=centerness_target,
                avg_factor=bbox_target.shape[0])
            # loss_bbox = self.loss_bbox(
            #     pos_decoded_bbox_preds,
            #     pos_decoded_target_preds,
            #     weight=pos_centerness_targets,
            #     avg_factor=pos_centerness_targets.sum())
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        # cls_scores_AB = [cls_score[0] for cls_score in cls_scores]
        cls_scores_AF = [cls_score[1] for cls_score in cls_scores]
        # bbox_preds_AB = [bbox_pred[0] for bbox_pred in bbox_preds]
        bbox_preds_AF = [bbox_pred[1] for bbox_pred in bbox_preds]
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores_AF]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds_AF[0].dtype,
                                      bbox_preds_AF[0].device)
        result_list = []
        # featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores_AF]
        # anchors = self.anchor_base.get_anchors(featmap_sizes, img_metas)
        for img_id in range(len(img_metas)):
            cls_score_list_AF = [
                cls_scores_AF[i][img_id].detach() for i in range(num_levels)
            ]
            # cls_score_list_AB = [
            #     cls_scores_AB[i][img_id].detach() for i in range(num_levels)
            # ]
            # cls_score_list=[cls_score_list_AF,cls_score_list_AB]
            bbox_pred_list_AF = [
                bbox_preds_AF[i][img_id].detach() for i in range(num_levels)
            ]
            # bbox_pred_list_AB = [
            #     bbox_preds_AB[i][img_id].detach() for i in range(num_levels)
            # ]
            # bbox_pred_list=[bbox_pred_list_AF,bbox_pred_list_AB]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            # centernesses_AB=[centerness for centerness in bbox_pred_list_AB]
            # for i,centerness in enumerate(centernesses_AB):
            #     centernesses_AB[i]=torch.ones_like(centerness)
            #     centernesses_AB[i]=centernesses_AB[i][0]
            #     centernesses_AB[i]=torch.unsqueeze(centernesses_AB[i],0)
            # centerness_pred_list=[centerness_pred_list,centernesses_AB]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            # anchor=anchors[img_id]
            det_bboxes = self._get_bboxes_single(cls_score_list_AF,
                                                 bbox_pred_list_AF,

                                                 centerness_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        # for cls_score_AB, bbox_pred_AB, points,anchor in zip(
        #         cls_scores[1], bbox_preds[1], mlvl_points,anchors):
        #     assert cls_score_AB.size()[-2:] == bbox_pred_AB.size()[-2:]
        #     scores = cls_score_AB.permute(1, 2, 0).reshape(
        #         -1, self.cls_out_channels).sigmoid()
        #
        #     bbox_pred_AB = bbox_pred_AB.permute(1, 2, 0).reshape(-1, 4)
        #     nms_pre = cfg.get('nms_pre', -1)
        #     if nms_pre > 0 and scores.shape[0] > nms_pre:
        #         max_scores, _ = (scores * 1).max(dim=1)
        #         _, topk_inds = max_scores.topk(nms_pre)
        #         bbox_pred_AB = bbox_pred_AB[topk_inds, :]
        #         scores = scores[topk_inds, :]
        #         anchor = anchor[0][topk_inds, :]
        #     bboxes = self.anchor_base.bbox_coder.decode(anchor, bbox_pred_AB)
        #     mlvl_bboxes.append(bboxes)
        #     mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        # mlvl_centerness_all = torch.ones_like(mlvl_bboxes)[:, 0]
        # mlvl_centerness_all[:len(mlvl_centerness)] = mlvl_centerness
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels
    #计算iou
    def box_iou(self,box1, box2, order='xyxy'):

        # box1=torch.from_numpy(box1).float()
        # box2=torch.from_numpy(box2).float()
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(box1[:, None, :2], box2[:, :2]).int()  # [N,M,2]
        rb = torch.min(box1[:, None, 2:], box2[:, 2:]).int()  # [N,M,2]

        wh = (rb - lt + 1)  # [N,M,2]
        wh=wh.clamp(min=0)
        inter = (wh[:, :, 0] * wh[:, :, 1]).float()  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)  # [N,]
        area2 = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)  # [M,]
        iou = inter / (area1[:, None] + area2 - inter)
        return iou

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._get_points_single(featmap_sizes[i], self.strides[i],
                                        dtype, device))
        return mlvl_points

    def _get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points, ), self.background_label), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.background_label  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
