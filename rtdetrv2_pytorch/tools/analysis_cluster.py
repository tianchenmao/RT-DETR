import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def load_coco_id_to_filename(coco_json_path):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    return id_to_filename

def visualize_batch(json_line, id_to_filename, image_dir, start_epoch=0):
    data = json.loads(json_line)

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()

    for idx, item in enumerate(data):
        image_id = item['image_id']
        bboxes = item['bboxes']
        epoch = item['epoch']
        if epoch > start_epoch:
            break

        file_name = id_to_filename.get(image_id)
        if file_name is None:
            print(f"Image ID {image_id} not found in COCO annotations.")
            axs[idx].axis('off')
            continue

        image_path = os.path.join(image_dir, file_name)
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            axs[idx].axis('off')
            continue

        image = Image.open(image_path).convert("RGB")
        axs[idx].imshow(image)
        axs[idx].set_title(f"ID: {image_id}")
        axs[idx].axis('off')

        w_img, h_img = image.size
        for bbox in bboxes:
            cx, cy, w, h = bbox
            x = (cx - w / 2) * w_img
            y = (cy - h / 2) * h_img
            width = w * w_img
            height = h * h_img

            rect = patches.Rectangle((x, y), width, height, linewidth=2,
                                     edgecolor='red', facecolor='none')
            axs[idx].add_patch(rect)

    for j in range(len(data), 8):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 示例使用
    coco_json_path = r"C:\Users\fur\PycharmProjects\DINO\radarv8_cocostyle\annotations\instances_train2017.json"
    image_dir = r"C:\Users\fur\PycharmProjects\DINO\radarv8_cocostyle\train2017"
    batch_json_path = r"C:\Users\fur\PycharmProjects\RT-DETR\rtdetrv2_pytorch\records\cluster_box_records.json"

    # 创建映射
    id_to_filename = load_coco_id_to_filename(coco_json_path)

    # 逐行处理batch JSON
    with open(batch_json_path, "r") as f:
        for line in f:
            visualize_batch(line, id_to_filename, image_dir,start_epoch=30)
