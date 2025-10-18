#!/usr/bin/env python3
"""
Main script for YOLO annotation conversion and finetuning preparation.
This script ties together the annotation conversion and finetuning preparation steps.
"""

import os
import argparse
from pathlib import Path
from convert_annotations import YOLOAnnotationConverter
from prepare_yolo_finetuning import YOLOFinetuningPreparation


def main():
    parser = argparse.ArgumentParser(description="Convert remapped annotations to YOLO format and prepare finetuning")

    parser.add_argument("--remapped_geojson", type=str, 
                        default="outputs/remapped_slide-2024-04-03T07-52-35-R1-S2.geojson",
                        help="Path to the remapped geojson file")
    parser.add_argument("--extracted_region", type=str,
                        default="outputs/remapped_slide-2024-04-03T07-52-35-R1-S2.ome.tif",
                        help="Path to the extracted region image")
    
    # out directories
    parser.add_argument("--annotations_dir", type=str, default="data/processed/yolo2",
                        help="Directory to save the YOLO annotations")
    parser.add_argument("--dataset_dir", type=str, default="data/processed/yolo_dataset2",
                        help="Directory to save the YOLO dataset")
    
    # annotation options
    parser.add_argument("--output_prefix", type=str, default="yolo_annotations",
                        help="Prefix for output files")
    parser.add_argument("--default_margin", type=int, default=20,
                        help="Default margin to add around bounding boxes")
    parser.add_argument("--use_adaptive_margin", action="store_true", default=True,
                        help="Use adaptive margin based on object size")
    
    # dataset options
    parser.add_argument("--train_val_split", type=float, default=0.8,
                        help="Ratio of training data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # skip steps
    parser.add_argument("--skip_conversion", action="store_true",
                        help="Skip annotation conversion step")
    parser.add_argument("--skip_preparation", action="store_true",
                        help="Skip dataset preparation step")
    
    args = parser.parse_args()

    os.makedirs(args.annotations_dir, exist_ok=True)
    os.makedirs(args.dataset_dir, exist_ok=True)

    result_paths = {}

    print("Step 1: Converting annotations to YOLO format...")
    converter = YOLOAnnotationConverter(
        args.remapped_geojson,
        args.extracted_region,
        args.annotations_dir,
        args.default_margin
    )

    result_paths = converter.process(args.output_prefix, args.use_adaptive_margin)

    print(f"YOLO annotations saved to: {result_paths['tiles']}")
    print(f"Class mapping saved to: {result_paths['class_mapping']}")

    print("\nStep 2: Preparing YOLO dataset for finetuning...")
    preparation = YOLOFinetuningPreparation(
        args.annotations_dir,
        result_paths["class_mapping"],
        args.dataset_dir,
        args.train_val_split,
        args.seed
    )
    
    dataset_paths = preparation.process()
    
    print(f"Dataset prepared in: {args.dataset_dir}")
    print(f"Images and annotations placed in {dataset_paths['dataset']} set")
    print(f"Data YAML file created at: {dataset_paths['data_yaml']}")
    print(f"Model YAML file created at: {dataset_paths['model_yaml']}")
    print(f"Training script created at: {dataset_paths['train_script']}")
    print("\nTo train the model, run:")
    print(f"cd {args.dataset_dir} && python train.py")


if __name__ == "__main__":
    main()