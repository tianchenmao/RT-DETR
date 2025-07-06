from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
from torchvision.ops import box_iou

class ObjectnessEvaluator:
    def __init__(self, threshold=0.3, iou_threshold=0.5):
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.y_true = []
        self.y_pred = []
        self.count_preds = []
        self.count_gts = []

    def update(self, res, targets):
        for target in targets:
            image_id = target['image_id'].item()
            gt_boxes = target['boxes']
            gt_counts = target['labels']
            gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

            pred = res[image_id]
            boxes = pred['boxes']
            scores = pred['scores']
            counts = pred['labels']

            keep = scores > self.threshold
            boxes = boxes[keep]
            counts = counts[keep]
            scores = scores[keep]

            if len(boxes) == 0:
                continue

            if len(gt_boxes) > 0:
                ious = box_iou(boxes, gt_boxes)
                max_iou, matched_gt_idx = ious.max(dim=1)
                matched_mask = max_iou > self.iou_threshold

                for i in range(len(boxes)):
                    if matched_mask[i]:
                        gt_idx = matched_gt_idx[i].item()
                        if not gt_matched[gt_idx]:
                            self.y_true.append(1)
                            self.y_pred.append(1)
                            self.count_preds.append(counts[i].item())
                            self.count_gts.append(gt_counts[gt_idx].item())
                            gt_matched[gt_idx] = True
                        else:
                            self.y_true.append(0)
                            self.y_pred.append(1)
                    else:
                        self.y_true.append(0)
                        self.y_pred.append(1)
            else:
                self.y_true.extend([0] * len(boxes))
                self.y_pred.extend([1] * len(boxes))

    def compute(self):
        if len(self.y_pred) == 0:
            precision = recall = 0.0
        else:
            precision = np.sum(np.array(self.y_true)) / max(np.sum(np.array(self.y_pred)), 1)
            recall = np.sum(np.array(self.y_true)) / max(len(self.y_true), 1)

        if self.count_preds:
            count_preds = np.array(self.count_preds)
            count_gts = np.array(self.count_gts)
            count_mae = np.mean(np.abs(count_preds - count_gts))
            count_rmse = np.sqrt(np.mean((count_preds - count_gts) ** 2))
        else:
            count_mae = 0.0
            count_rmse = 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "count_mae": round(count_mae, 4),
            "count_rmse": round(count_rmse, 4)
        }

class ObjectnessEvaluatorHungarian:
    def __init__(self, threshold=0.35, iou_threshold=0.5, bin_size=10):
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.bin_size = bin_size

        self.y_true = []         # for precision/recall
        self.y_pred = []
        self.count_preds = []    # for count MAE
        self.count_gts = []
        self.class_preds = []    # for coarse classification accuracy
        self.class_gts = []
        self.fp_counts = []      # for avg_fp_per_image
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, res, targets):
        for target in targets:
            image_id = target['image_id'].item()
            gt_boxes = target['boxes']
            gt_counts = target['labels']  # true count

            pred = res[image_id]
            boxes = pred['boxes']
            scores = pred['scores']
            counts = pred['counts']       # predicted count (float)
            labels = pred['labels']       # predicted coarse bin ID (int)

            keep = scores > self.threshold
            boxes = boxes[keep]
            counts = counts[keep]
            labels = labels[keep]
            scores = scores[keep]

            num_pred = len(boxes)
            num_gt = len(gt_boxes)

            if num_pred == 0:
                self.y_true.extend([0] * num_gt)
                self.y_pred.extend([])
                self.fp_counts.append(0)
                continue
            if num_gt == 0:
                self.y_true.extend([])
                self.y_pred.extend([1] * num_pred)
                self.fp_counts.append(num_pred)
                continue

            ious = box_iou(boxes, gt_boxes)
            cost_matrix = -ious.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_gt = set()
            matched_pred = set()
            fp_count = 0

            for pi, gi in zip(row_ind, col_ind):
                iou = ious[pi, gi].item()
                if iou > self.iou_threshold:
                    self.tp += 1
                    self.y_true.append(1)
                    self.y_pred.append(1)
                    matched_gt.add(gi)
                    matched_pred.add(pi)

                    pred_count = counts[pi].item()
                    gt_count = gt_counts[gi].item()

                    self.count_preds.append(pred_count)
                    self.count_gts.append(gt_count)

                    # ✅ coarse class from label (already bin index)
                    pred_bin = labels[pi].item()
                    gt_bin = int((gt_count - 1) // self.bin_size)
                    self.class_preds.append(pred_bin)
                    self.class_gts.append(gt_bin)
                else:
                    self.fp += 1
                    self.y_true.append(0)
                    self.y_pred.append(1)
                    matched_pred.add(pi)
                    fp_count += 1

            for pi in range(num_pred):
                if pi not in matched_pred:
                    self.y_true.append(0)
                    self.y_pred.append(1)
                    self.fp += 1
                    fp_count += 1

            for gi in range(num_gt):
                if gi not in matched_gt:
                    self.y_true.append(1)
                    self.y_pred.append(0)
                    self.fn += 1

            self.fp_counts.append(fp_count)

    def compute(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        fdr = self.fp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        avg_fp = np.mean(self.fp_counts) if self.fp_counts else 0.0

        if self.count_preds:
            count_preds = np.array(self.count_preds)
            count_gts = np.array(self.count_gts)
            count_mae = np.mean(np.abs(count_preds - count_gts))
            count_rmse = np.sqrt(np.mean((count_preds - count_gts) ** 2))
        else:
            count_mae = -1
            count_rmse = -1

        if self.class_preds:
            class_acc = np.mean(np.array(self.class_preds) == np.array(self.class_gts))
        else:
            class_acc = -1

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "fdr": round(fdr, 4),
            "avg_fp_per_image": round(avg_fp, 4),
            "count_mae": round(count_mae, 4),
            "count_rmse": round(count_rmse, 4),
            "coarse_class_acc": round(class_acc, 4)
        }

def print_eval_metrics(metrics, threshold):
    print("\n" + "="*50)
    print(f"📊 Objectness Evaluation @ threshold = {threshold:.2f}")
    print("-"*50)
    print(f"🎯 Precision      : {metrics['precision']*100:6.2f}%")
    print(f"🎯 Recall         : {metrics['recall']*100:6.2f}%")
    print(f"📏 Count MAE      : {metrics['count_mae']:6.2f}")
    print(f"📏 Count RMSE     : {metrics['count_rmse']:6.2f}")
    print("="*50 + "\n")


def print_eval_metrics_full(metrics, threshold):
    print("\n" + "="*50)
    print(f"📊 Objectness Evaluation @ threshold = {threshold:.2f}")
    print("-"*50)
    print(f"🎯 Precision        : {metrics['precision']*100:6.2f}%")
    print(f"🎯 Recall           : {metrics['recall']*100:6.2f}%")
    print(f"❌ FDR              : {metrics['fdr']*100:6.2f}%")
    print(f"❌ Avg FP / Image   : {metrics['avg_fp_per_image']:6.2f}")
    print(f"📏 Count MAE        : {metrics['count_mae']:6.2f}")
    print(f"📏 Count RMSE       : {metrics['count_rmse']:6.2f}")
    print(f"🧠 Coarse Class Acc : {metrics['coarse_class_acc']*100:6.2f}%")
    print("="*50 + "\n")


