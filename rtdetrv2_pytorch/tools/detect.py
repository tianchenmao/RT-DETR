import os
import re
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sympy.logic.inference import valid
from torchvision.ops import box_convert
import sys
import time
import torchvision.ops as ops
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
from src.zoo.rtdetr import RTDETRPostProcessor, RTDETR
from src.solver import DetSolver
from src.core import YAMLConfig, yaml_utils



def nms(boxes, scores, iou_threshold=0.5, score_threshold=0.0):
    """
    Args:
        boxes (Tensor): shape [N, 4], in xyxy format
        scores (Tensor): shape [N], confidence scores
        iou_threshold (float): IoU threshold for suppression
        score_threshold (float): filter out boxes with score below this
    Returns:
        keep_boxes (Tensor): indices of kept boxes after NMS
    """
    assert boxes.shape[0] == scores.shape[0], "Mismatch in boxes and scores"

    # filter low scores
    keep = scores > score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    # perform NMS
    keep_indices = ops.nms(boxes, scores, iou_threshold)
    return keep.nonzero(as_tuple=False).squeeze(1)[keep_indices]


class ResizeWithPadding:
    def __init__(self, target_size=640, fill=0):
        self.target_size = target_size  # 最终图像为 target_size × target_size
        self.fill = fill                # padding 填充值

    def __call__(self, img):
        """
        Args:
            img: PIL.Image
        Returns:
            padded_img: Tensor, shape [3, target_size, target_size]
            valid_size: Tensor, shape [1, 2], actual resized image size before padding (width, height)
        """
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_img = F.resize(img, [new_h, new_w])  # 等比例缩放
        padded_img = F.pad(resized_img,
                           padding=self._get_padding(new_w, new_h),
                           fill=self.fill,
                           padding_mode='constant')
        img_tensor = F.to_tensor(padded_img)
        valid_size = torch.tensor([[new_w, new_h]], dtype=torch.float32)  # 注意是 [W, H]
        return img_tensor, valid_size

    def _get_padding(self, new_w, new_h):
        """
        Returns padding tuple: (left, top, right, bottom)
        """
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        return (pad_left, pad_top, pad_right, pad_bottom)

# RT-DETR好像只接受方形的输入
transform = T.Compose([
    T.Resize([640,640]),
    T.ToTensor(),
])

def load_images_from_folder(folder_path):
    pattern = re.compile(r'sc(\d+)_det_(\d+)\.jpg')
    images = []
    for fname in os.listdir(folder_path):
        match = pattern.match(fname)
        if match:
            frame = int(match.group(2))
            images.append((frame, os.path.join(folder_path, fname)))
    images.sort()
    return images

def infer_and_plot(model, postprocessor, device, folder, save_dir=None, score_thresh=0.30):
    model.eval()
    images = load_images_from_folder(folder)

    for frame, img_path in images:
        image = Image.open(img_path).convert('RGB')
        # transform_resize_with_padding = ResizeWithPadding(target_size=640)
        # img_tensor, valid_size = transform_resize_with_padding(image)
        img_tensor = transform(image)
        valid_size = torch.tensor([640, 480], dtype=torch.float32).to(device)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            orig_size = valid_size.to(device)  # shape: [1, 2]
            results = postprocessor(outputs, orig_size)

        result = results[0]
        boxes = result['boxes'].cpu()
        scores = result['scores'].cpu()
        labels = result['labels'].cpu()
        counts = result['counts'].cpu()

        keep = nms(boxes, scores, iou_threshold=0.3, score_threshold=score_thresh)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        counts = counts[keep]

        if save_dir is not None:
            image.save(os.path.join(save_dir, f'{frame:06d}.jpg'))

        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for box, score, label, count in zip(boxes, scores, labels, counts):
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f'{count.item()}:{score:.2f}', color='red', fontsize=10)
        ax.axis('off')
        plt.show()
        time.sleep(0.02)


def main(args, ) -> None:
    """main"""
    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
                        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    print('cfg: ', cfg.__dict__)

    solver = DetSolver(cfg)
    solver.eval()

    infer_and_plot(solver.model, solver.postprocessor, solver.device, args.test_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str,
                        default=r'C:\Users\fur\PycharmProjects\RT-DETR\rtdetrv2_pytorch\configs\rtdetrv2\rtdetrv2_hgnetv2_x_6x_coco.yml')
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint',default=r'F:\rtdetrv2_hgnetv2_x_6x_coco_1\checkpoint0140.pth')
    parser.add_argument('-d', '--device', type=str, help='device', )
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-dir', type=str, help='test image directory', default=r'C:\Users\fur\PycharmProjects\UAVGroupTrackingYolo\dataset\test\fullsize\sc1\images')
    parser.add_argument('--test-only', action='store_true', default=False, )

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)

