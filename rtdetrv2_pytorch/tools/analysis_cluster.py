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

def show_batch(group, id_to_filename, image_dir):
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()

    for idx in range(8):
        if idx >= len(group):
            axs[idx].axis('off')
            continue

        item = group[idx]
        image_id = item['image_id']
        bboxes = item['bboxes']

        file_name = id_to_filename.get(image_id)
        if file_name is None:
            axs[idx].set_title(f"ID {image_id} not found")
            axs[idx].axis('off')
            continue

        image_path = os.path.join(image_dir, file_name)
        if not os.path.exists(image_path):
            axs[idx].set_title(f"Missing: {file_name}")
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

    plt.tight_layout()
    plt.show()

def visualize_json_in_chunks(batch_json_path, id_to_filename, image_dir, start_epoch=0):
    with open(batch_json_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data[0]['epoch'] < start_epoch:
                    continue
            except json.JSONDecodeError:
                continue

            # 按8个一组进行显示
            for i in range(0, len(data), 8):
                group = data[i:i+8]
                show_batch(group, id_to_filename, image_dir)
                input("Press Enter to continue...")

if __name__ == "__main__":
    # 示例使用
    coco_json_path = r"C:\Users\fur\PycharmProjects\DINO\radarv8_cocostyle\annotations\instances_train2017.json"
    image_dir = r"C:\Users\fur\PycharmProjects\DINO\radarv8_cocostyle\train2017"
    batch_json_path = r"C:\Users\fur\PycharmProjects\RT-DETR\rtdetrv2_pytorch\records\cluster_box_records.json"

    # 创建映射
    id_to_filename = load_coco_id_to_filename(coco_json_path)
    visualize_json_in_chunks(batch_json_path, id_to_filename, image_dir,start_epoch=50)
