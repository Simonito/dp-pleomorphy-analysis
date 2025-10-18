import os
import shutil
import yaml
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random


class YOLOFinetuningPreparation:
    """Prepare YOLO detection finetuning dataset from tiled annotations."""

    def __init__(
        self,
        converter_output_dir: str,
        class_mapping_path: str,
        output_dir: str,
        train_val_split: float = 0.8,
        seed: int = 42,
    ):
        """
        Args:
            converter_output_dir: Directory containing tiled images and YOLO txt annotations
            class_mapping_path: Path to class mapping JSON file from converter
            output_dir: Directory to save final YOLO dataset
            train_val_split: Ratio of train/val split
            seed: Random seed
        """
        self.converter_output_dir = converter_output_dir
        self.class_mapping_path = class_mapping_path
        self.output_dir = output_dir
        self.train_val_split = train_val_split
        self.seed = seed

        random.seed(seed)
        os.makedirs(output_dir, exist_ok=True)

    def create_directory_structure(self) -> Dict[str, str]:
        """Create YOLO dataset directory structure."""
        images_train_dir = os.path.join(self.output_dir, "images", "train")
        images_val_dir = os.path.join(self.output_dir, "images", "val")
        labels_train_dir = os.path.join(self.output_dir, "labels", "train")
        labels_val_dir = os.path.join(self.output_dir, "labels", "val")

        os.makedirs(images_train_dir, exist_ok=True)
        os.makedirs(images_val_dir, exist_ok=True)
        os.makedirs(labels_train_dir, exist_ok=True)
        os.makedirs(labels_val_dir, exist_ok=True)

        return {
            "images_train": images_train_dir,
            "images_val": images_val_dir,
            "labels_train": labels_train_dir,
            "labels_val": labels_val_dir,
        }

    def collect_image_label_pairs(self) -> List[Tuple[str, str]]:
        """
        Collect all tiled image-label pairs.
        Supported image extensions: .jpg, .jpeg, .png, .tif, .tiff
        """
        valid_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        valid_double_exts = [".ome.tif", ".ome.tiff"]
        pairs = []

        print(f">>> out_dir = {self.converter_output_dir}")
        for image_file in Path(self.converter_output_dir).rglob("*"):
        #     if image_file.suffix.lower() in valid_exts:
        #         label_file = image_file.with_suffix(".txt")
        #         print('>>< got img: ', image_file, ' looking for label: ', label_file)
        #         if label_file.exists():
        #             pairs.append((str(image_file), str(label_file)))
            matched = False
            for double_ext in valid_double_exts:
                if image_file.name.lower().endswith(double_ext):
                    # remove the double extension
                    fname_without_ext = image_file.name[:-len(double_ext)]
                    label_file = image_file.parent / (fname_without_ext + ".txt")
                    matched = True
                    break

            if not matched and image_file.suffix.lower() in valid_exts:
                label_file = image_file.with_suffix(".txt")
                matched = True
            
            if matched and label_file.exists():
                pairs.append((str(image_file), str(label_file)))

        if not pairs:
            raise RuntimeError(f"No image/label pairs found in {self.converter_output_dir}")

        return pairs

    def split_dataset(self, pairs: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Split dataset into train and val subsets."""
        if len(pairs) == 1:
            return pairs, pairs

        random.shuffle(pairs)
        split_idx = int(len(pairs) * self.train_val_split)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        if not val_pairs:
            val_pairs = [train_pairs[0]]

        return train_pairs, val_pairs

    def copy_pairs(self, pairs: List[Tuple[str, str]], img_dest: str, lbl_dest: str):
        """Copy image-label pairs to destination directories."""
        for img, lbl in pairs:
            shutil.copy(img, os.path.join(img_dest, os.path.basename(img)))
            shutil.copy(lbl, os.path.join(lbl_dest, os.path.basename(lbl)))

    def create_data_yaml(self, class_names: List[str]) -> str:
        """Create data.yaml file for YOLO training."""
        data = {
            "path": os.path.abspath(self.output_dir),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(class_names)},
        }
        yaml_path = os.path.join(self.output_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return yaml_path

    def create_model_yaml(self, num_classes: int) -> str:
        """Create basic YOLOv8 model.yaml."""
        model_config = {
            "nc": num_classes,
            "depth_multiple": 0.33,
            "width_multiple": 0.50,
            "backbone": [
                [-1, 1, "Conv", [64, 6, 2, 2]],
                [-1, 1, "Conv", [128, 3, 2]],
                [-1, 3, "C3", [128]],
                [-1, 1, "Conv", [256, 3, 2]],
                [-1, 6, "C3", [256]],
                [-1, 1, "Conv", [512, 3, 2]],
                [-1, 9, "C3", [512]],
                [-1, 1, "Conv", [1024, 3, 2]],
                [-1, 3, "C3", [1024]],
                [-1, 1, "SPPF", [1024, 5]],
            ],
            "head": [
                [-1, 1, "Conv", [512, 1, 1]],
                [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
                [[-1, 6], 1, "Concat", [1]],
                [-1, 3, "C3", [512, False]],
                [-1, 1, "Conv", [256, 1, 1]],
                [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
                [[-1, 4], 1, "Concat", [1]],
                [-1, 3, "C3", [256, False]],
                [-1, 1, "Conv", [256, 3, 2]],
                [[-1, 14], 1, "Concat", [1]],
                [-1, 3, "C3", [512, False]],
                [-1, 1, "Conv", [512, 3, 2]],
                [[-1, 10], 1, "Concat", [1]],
                [-1, 3, "C3", [1024, False]],
                [[17, 20, 23], 1, "Detect", [num_classes, [8, 16, 32]]],
            ],
        }
        yaml_path = os.path.join(self.output_dir, "model.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(model_config, f, default_flow_style=False, sort_keys=False)
        return yaml_path

    def create_training_script(self) -> str:
        """Generate train.py for YOLOv8 training."""
        script = """#!/usr/bin/env python3
import argparse
from ultralytics import YOLO

def train_yolo(data_yaml, epochs=100, batch_size=16, img_size=640, weights=None):
    if weights:
        model = YOLO(weights)
    else:
        model = YOLO('yolov8n.pt')
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=50,
        save=True,
        device='mps'
    )
    model.val()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model on tiled dataset")
    parser.add_argument("--data", type=str, default="data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()
    train_yolo(args.data, args.epochs, args.batch, args.img_size, args.weights)
"""
        script_path = os.path.join(self.output_dir, "train.py")
        with open(script_path, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o755)
        return script_path

    def process(self) -> Dict[str, str]:
        """Main dataset preparation pipeline."""
        dirs = self.create_directory_structure()
        pairs = self.collect_image_label_pairs()
        train_pairs, val_pairs = self.split_dataset(pairs)

        self.copy_pairs(train_pairs, dirs["images_train"], dirs["labels_train"])
        self.copy_pairs(val_pairs, dirs["images_val"], dirs["labels_val"])

        with open(self.class_mapping_path, "r") as f:
            class_mapping = json.load(f)
        class_names = list(class_mapping.keys())

        data_yaml = self.create_data_yaml(class_names)
        model_yaml = self.create_model_yaml(len(class_names))
        train_script = self.create_training_script()

        return {
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "data_yaml": data_yaml,
            "model_yaml": model_yaml,
            "train_script": train_script,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from tiled converter outputs")
    parser.add_argument("--converter_output_dir", type=str, required=True,
                        help="Path to directory containing tiled images and .txt YOLO annotations")
    parser.add_argument("--class_mapping", type=str, required=True,
                        help="Path to class mapping JSON file from converter")
    parser.add_argument("--output_dir", type=str, default="../../data/processed/yolo_dataset",
                        help="Directory to save YOLO dataset")
    parser.add_argument("--train_val_split", type=float, default=0.8,
                        help="Train/val split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    prep = YOLOFinetuningPreparation(
        converter_output_dir=args.converter_output_dir,
        class_mapping_path=args.class_mapping,
        output_dir=args.output_dir,
        train_val_split=args.train_val_split,
        seed=args.seed,
    )

    result = prep.process()

    print(f"YOLO dataset prepared in: {args.output_dir}")
    print(f"Train images: {result['train_pairs']} | Val images: {result['val_pairs']}")
    print(f"data.yaml: {result['data_yaml']}")
    print(f"model.yaml: {result['model_yaml']}")
    print(f"train.py: {result['train_script']}")
    print("\nTo train the model:")
    print(f"cd {args.output_dir} && python train.py")
