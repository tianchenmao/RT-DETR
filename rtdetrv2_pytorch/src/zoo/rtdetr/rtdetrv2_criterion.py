"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.distributed
import torch.nn.functional as F 
import torchvision

import copy

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register


@register()
class RTDETRCriterionv2(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher, 
        weight_dict, 
        losses, 
        alpha=0.2, 
        gamma=2.0, 
        num_classes=80, 
        boxes_weight_format=None,
        share_matched_indices=False):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            boxes_weight_format: format for boxes weight (iou, )
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_objectness_trivial(self, outputs, targets, indices, num_boxes):
        assert 'pred_objectness_logits' in outputs

        B, Q = outputs['pred_objectness_logits'].shape[:2]
        obj_pred = outputs['pred_objectness_logits'].squeeze(-1)  # [B, Q]

        # 构造 supervision mask：正样本为 1，其余为 0
        obj_target = torch.zeros((B, Q), dtype=obj_pred.dtype, device=obj_pred.device)
        batch_idx, query_idx = self._get_src_permutation_idx(indices)
        obj_target[batch_idx, query_idx] = 1.0

        # Binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(obj_pred, obj_target, reduction='mean')
        return {'loss_objectness': loss}

    def loss_count_mse(self,outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        pred = outputs['pred_counts'][idx].squeeze(-1)
        target = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)]).to(pred.dtype)
        return {'loss_count_mse': F.mse_loss(pred, target, reduction='mean')}

    def loss_count_l1(self,outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        pred = outputs['pred_counts'][idx].squeeze(-1)
        target = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)]).to(pred.dtype)
        return {'loss_count_l1': F.l1_loss(pred, target, reduction='mean')}

    def loss_count_smooth_l1(self,outputs, targets, indices, num_boxes, beta=1.0):
        idx = self._get_src_permutation_idx(indices)
        pred = outputs['pred_counts'][idx].squeeze(-1)
        target = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)]).to(pred.dtype)
        return {'loss_count_smooth_l1': F.smooth_l1_loss(pred, target, beta=beta, reduction='mean')}

    def loss_count_poisson(self, outputs, targets, indices, num_boxes):
        assert 'pred_counts' in outputs
        idx = self._get_src_permutation_idx(indices)

        # 预测的 count，形状 [N]
        src_counts = outputs['pred_counts'][idx].squeeze(-1)

        # GT：从 label 中读取 count
        target_counts = torch.cat([t['labels'][j] for t, (_, j) in zip(targets, indices)])
        target_counts = target_counts.to(dtype=src_counts.dtype)

        # Poisson NLL Loss
        loss = F.poisson_nll_loss(src_counts, target_counts, log_input=False, full=True, reduction='mean')
        return {'loss_count_poisson': loss}

    def loss_objectness_focal(self, outputs, targets, indices, num_boxes):
        """
        Binary Focal Loss for objectness prediction
        """
        assert 'pred_objectness_logits' in outputs
        B, Q = outputs['pred_objectness_logits'].shape[:2]
        pred_logits = outputs['pred_objectness_logits'].squeeze(-1)  # [B, Q]

        # 构造二值目标：1 为正样本（被 matcher 匹配），其余为 0
        target = torch.zeros((B, Q), dtype=pred_logits.dtype, device=pred_logits.device)
        batch_idx, query_idx = self._get_src_permutation_idx(indices)
        target[batch_idx, query_idx] = 1.0  # positive samples

        # Sigmoid 激活
        prob = torch.sigmoid(pred_logits)
        pt = prob * target + (1 - prob) * (1 - target)  # pt = p if t==1 else 1-p

        # Focal loss 权重项
        focal_weight = (1 - pt) ** self.gamma

        # Alpha 权重
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)

        # Focal Loss
        loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        loss = (focal_weight * alpha_weight * loss).mean()

        return {'loss_objectness': loss}

    def loss_objectness(self, outputs, targets, indices, num_boxes):
        """
        IoU-aware Binary Focal Loss for objectness prediction
        """
        assert 'pred_objectness_logits' in outputs
        B, Q = outputs['pred_objectness_logits'].shape[:2]
        pred_logits = outputs['pred_objectness_logits'].squeeze(-1)  # [B, Q]

        # 构造 soft target（默认值 0）
        target = torch.zeros((B, Q), dtype=pred_logits.dtype, device=pred_logits.device)

        # 获取匹配的 (batch_idx, query_idx) 对应的 IoU，作为 soft objectness label
        batch_idx, query_idx = self._get_src_permutation_idx(indices)

        # 获取预测框和 GT 框
        pred_boxes = outputs['pred_boxes'][batch_idx, query_idx]  # [N, 4]
        tgt_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        # 计算匹配框对之间的 IoU
        ious,_ = box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(tgt_boxes))
        iou_diag = ious.diag().clamp(min=0.0, max=1.0).detach()  # [N]

        # 赋予 soft target
        target[batch_idx, query_idx] = iou_diag  # soft supervision ∈ [0, 1]

        # Sigmoid 激活
        prob = torch.sigmoid(pred_logits)
        pt = prob * target + (1 - prob) * (1 - target)  # pt = p if t==1 else 1-p

        # Focal loss 权重项
        focal_weight = (1 - pt) ** self.gamma

        # Alpha 权重项
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)

        # Binary focal loss
        loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        loss = (focal_weight * alpha_weight * loss).mean()

        return {'loss_objectness': loss}

    def loss_objectness_vfl_count(self, outputs, targets, indices, num_boxes):
        """
        IoU-aware + Count-aware Binary Focal Loss for objectness prediction
        (using predicted group count vs ground truth count similarity as weight)
        """
        assert 'pred_objectness_logits' in outputs
        assert 'pred_counts' in outputs

        B, Q = outputs['pred_objectness_logits'].shape[:2]
        pred_logits = outputs['pred_objectness_logits'].squeeze(-1)  # [B, Q]

        # 构造 soft target（默认值 0）
        target = torch.zeros((B, Q), dtype=pred_logits.dtype, device=pred_logits.device)

        # 获取匹配的 (batch_idx, query_idx)
        batch_idx, query_idx = self._get_src_permutation_idx(indices)

        # 匹配的预测框和 GT 框
        pred_boxes = outputs['pred_boxes'][batch_idx, query_idx]
        tgt_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        # 计算 IoU 作为 soft objectness label
        ious, _ = box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(tgt_boxes))
        iou_diag = ious.diag().clamp(min=0.0, max=1.0).detach()  # [N]

        # =========  Count-aware soft weight =========
        # 获取每个 sample 的预测 count（直接从 pred_counts 平均或 sum）
        pred_count_map = outputs['pred_counts'].squeeze(-1)  # [B, Q]
        pred_counts = pred_count_map.mean(dim=1).clamp(min=1.0)  # [B]

        # GT count: 每个样本中 GT box 数量
        gt_counts = torch.tensor([len(t['labels']) for t in targets],
                                 dtype=pred_counts.dtype,
                                 device=pred_counts.device).clamp(min=1.0)  # [B]

        # count similarity ∈ [0, 1]
        count_sim = torch.minimum(pred_counts, gt_counts) / torch.maximum(pred_counts, gt_counts)  # [B]
        count_sim_per_sample = count_sim[batch_idx]  # [N]

        # 构造 soft supervision label: soft = iou × count_sim
        soft_label = iou_diag * count_sim_per_sample
        target[batch_idx, query_idx] = soft_label

        # ========= Binary Focal Loss =========
        prob = torch.sigmoid(pred_logits)
        pt = prob * target + (1 - prob) * (1 - target)

        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        loss = (focal_weight * alpha_weight * loss).mean()

        return {'loss_objectness': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'objectness': self.loss_objectness,
            'count_poisson': self.loss_count_poisson,
            'count_mse': self.loss_count_mse,
            'count_l1': self.loss_count_l1,
            'count_smooth_l1': self.loss_count_smooth_l1
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Retrieve the matching between the outputs of the last layer and the targets
        matched = self.matcher(outputs_without_aux, targets)
        indices = matched['indices']

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            meta = self.get_loss_meta_info(loss, outputs, targets, indices)            
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **meta)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.share_matched_indices:
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                for loss in self.losses:
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of encoder auxiliary losses. For rtdetr v2
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                matched = self.matcher(aux_outputs, targets)
                indices = matched['indices']
                for loss in self.losses:
                    # Exclude poisson loss for encoder
                    if loss == 'count_poisson':
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
            
            if class_agnostic:
                self.num_classes = orig_num_classes

        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', ):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices
