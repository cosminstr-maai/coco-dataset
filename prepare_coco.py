import os
import json
import requests
import random
from tqdm import tqdm
from pathlib import Path

ANNOTATION_URLS = {
    "train": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}
DATA_SPLIT = {"train": 4500, "val": 500}

def download_and_extract_annotations():
    import zipfile
    os.makedirs("annotations", exist_ok=True)
    zip_path = "annotations_trainval2017.zip"
    if not os.path.exists(zip_path):
        print("Downloading annotations...")
        r = requests.get(ANNOTATION_URLS["train"], stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("Extracting annotations...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

def parse_coco_annotations(json_path, image_dir):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Person class only
    person_anns = [ann for ann in data["annotations"] if ann["category_id"] == 1]
    image_id_to_anns = {}
    for ann in person_anns:
        image_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    # Image info
    image_info = {img["id"]: img for img in data["images"] if img["id"] in image_id_to_anns}

    # Return map of image_id -> {info, anns}
    return {
        img_id: {"info": image_info[img_id], "annotations": image_id_to_anns[img_id]}
        for img_id in image_id_to_anns
    }

def download_image(file_name, subset, out_dir):
    url = f"http://images.cocodataset.org/{subset}2017/{file_name}"
    dest_path = out_dir / f"coco_{file_name}"
    if dest_path.exists():
        return
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def coco_to_yolo(ann, img_w, img_h):
    x, y, w, h = ann["bbox"]
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

def prepare_subset(data_map, subset_name, n_images, image_src, out_base):
    out_img_dir = out_base / "images" / subset_name
    out_lbl_dir = out_base / "labels" / subset_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    selected_items = random.sample(list(data_map.items()), n_images)
    for img_id, content in tqdm(selected_items, desc=f"Preparing {subset_name}"):
        file_name = content["info"]["file_name"]
        width, height = content["info"]["width"], content["info"]["height"]
        anns = content["annotations"]

        download_image(file_name, image_src, out_img_dir)

        yolo_labels = [coco_to_yolo(a, width, height) for a in anns]
        label_file = out_lbl_dir / (f"coco_{Path(file_name).stem}.txt")
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_labels))

    return set(k for k, _ in selected_items)

def main():
    random.seed(42)
    download_and_extract_annotations()
    out_base = Path("coco_person")

    print("Parsing annotations...")
    train_data = parse_coco_annotations("annotations/instances_train2017.json", "train2017")
    val_data = parse_coco_annotations("annotations/instances_val2017.json", "val2017")

    # Merge and shuffle
    full_data = {**train_data, **val_data}
    total_ids = list(full_data.keys())
    random.shuffle(total_ids)

    val_ids = total_ids[:DATA_SPLIT["val"]]
    train_ids = total_ids[DATA_SPLIT["val"]:DATA_SPLIT["val"] + DATA_SPLIT["train"]]
    test_ids = total_ids[DATA_SPLIT["val"] + DATA_SPLIT["train"]:]

    val_map = {k: full_data[k] for k in val_ids}
    train_map = {k: full_data[k] for k in train_ids}
    test_map = {k: full_data[k] for k in test_ids}

    prepare_subset(train_map, "train", len(train_map), "train", out_base)
    prepare_subset(val_map, "val", len(val_map), "val", out_base)
    prepare_subset(test_map, "test", len(test_map), "train", out_base)

    print("\n✅ COCO person dataset ready in 'coco_person/'")

if __name__ == "__main__":
    main()
