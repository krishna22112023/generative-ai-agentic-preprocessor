import glob
import os
from tqdm import tqdm
from PIL import Image
import xml.etree.ElementTree as ET
import json

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

from pathlib import Path
import sys
base_path = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_path)

from src.utils.annotation_converter import convert_voc_to_yolo
from src.utils.metrics import iou

def detect_annotation_format(dataset_dir):
    for root, _, files in os.walk(dataset_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # Detect Pascal VOC (XML)
            if file_name.endswith('.xml'):
                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    if root.tag == 'annotation':
                        return "PascalVOC"
                except:
                    continue

            # Detect YOLO (TXT)
            if file_name.endswith('.txt'):
                try:
                    with open(file_path, 'r') as file:
                        first_line = file.readline().strip().split()
                        if len(first_line) >= 5 and all(part.replace('.', '', 1).isdigit() for part in first_line[1:]):
                            return "YOLO"
                except:
                    continue

            # Detect COCO (JSON)
            if file_name.endswith('.json'):
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                        if all(key in data for key in ["images", "annotations", "categories"]):
                            return "COCO"
                except:
                    continue

    return None


def hf_annotate( input_dir:str, output_dir:str, classes_file:str, model_id:str="IDEA-Research/grounding-dino-base"):
    os.makedirs(output_dir, exist_ok=True)

    # Read class labels
    class_map = {}
    with open(classes_file, 'r') as f:
        classes = f.read().strip().split('\n')
        for idx, class_name in enumerate(classes):
            class_map[class_name] = idx

    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

    annotations = []

    for image_path in tqdm(image_paths, desc=f"Annotating images with {model_id}"):
        image = Image.open(image_path)
        width, height = image.size
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")

        text_labels = [classes]

        inputs = processor(images=image, text=text_labels, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )[0]
        annotations.extend(results)

        # Convert to YOLO format and save
        with open(output_path, 'w') as f:
            for box, label in zip(results['boxes'], results['text_labels']):
                if label in class_map:
                    # YOLO format: class_id center_x center_y width height
                    x1, y1, x2, y2 = box.tolist()
                    
                    # Convert to normalized coordinates
                    center_x = ((x1 + x2) / 2) / width
                    center_y = ((y1 + y2) / 2) / height
                    bbox_width = (x2 - x1) / width
                    bbox_height = (y2 - y1) / height
                    
                    # Write YOLO format line
                    f.write(f"{class_map[label]} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

def yolo_to_absolute(bbox, width=1, height=1):
    """Convert YOLO bbox to absolute coordinates"""
    class_id, x_center, y_center, w, h = bbox
    x1 = (x_center - w/2) * width
    y1 = (y_center - h/2) * height
    x2 = (x_center + w/2) * width
    y2 = (y_center + h/2) * height
    return [class_id, x1, y1, x2, y2]

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics"""
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy

def consolidate_annotations(existing_annotation_dir, pred_annotation_dir, output_dir):
    """
    Consolidate annotations based on IoU scores
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all annotation files
    existing_files = glob.glob(os.path.join(existing_annotation_dir, "*.txt"))
    
    all_true_classes = []
    all_pred_classes = []
    all_ious = []
    
    for existing_file in existing_files:
        base_name = os.path.basename(existing_file)
        pred_file = os.path.join(pred_annotation_dir, base_name)
        
        if not os.path.exists(pred_file):
            continue
            
        # Read annotations
        with open(existing_file, 'r') as f:
            existing_annots = [list(map(float, line.strip().split())) for line in f]
        with open(pred_file, 'r') as f:
            pred_annots = [list(map(float, line.strip().split())) for line in f]
            
        # Convert to absolute coordinates (using normalized coordinates as is)
        existing_boxes = [yolo_to_absolute(bbox) for bbox in existing_annots]
        pred_boxes = [yolo_to_absolute(bbox) for bbox in pred_annots]
        
        # Calculate IoUs for this image
        image_ious = []
        for exist_box in existing_boxes:
            for pred_box in pred_boxes:
                if exist_box[0] == pred_box[0]:  # Same class
                    iou_score = iou(exist_box[1:], pred_box[1:])
                    image_ious.append(iou_score)
                    
                    all_true_classes.append(exist_box[0])
                    all_pred_classes.append(pred_box[0])
        
        # Decide which annotation to keep
        output_file = os.path.join(output_dir, base_name)
        if image_ious and np.mean(image_ious) > 0.5:
            # Keep existing annotation
            with open(existing_file, 'r') as f:
                content = f.read()
        else:
            # Keep predicted annotation
            with open(pred_file, 'r') as f:
                content = f.read()
                
        # Save consolidated annotation
        with open(output_file, 'w') as f:
            f.write(content)
            
        all_ious.extend(image_ious)
    
    # Calculate overall metrics
    if all_true_classes and all_pred_classes:
        precision, recall, f1, accuracy = calculate_metrics(all_true_classes, all_pred_classes)
        mean_iou = np.mean(all_ious) if all_ious else 0.0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'mean_iou': mean_iou
        }
    else:
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'mean_iou': 0.0
        }
    
    return metrics

'''if __name__ == "__main__":

    #detect if there are any existing annotations, if yes, then standardize them to common YOLO format
    annotation_format = detect_annotation_format("/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/datasets/DAWN/")
    print("Annotation files detected!")
    print(f"Detected annotation format: {annotation_format}")

    if annotation_format == "PascalVOC":
        convert_voc_to_yolo(
            voc_dir="/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/datasets/DAWN/Fog/Fog_PASCAL_VOC",
            yolo_dir="/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/outputs/Fog/Fog_YOLO",
            classes_file="/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/datasets/DAWN/classes.txt"
        )
    
    #perform annotation using grounding dino model
    print("performing annotation using grounding dino model")
    hf_annotate(model_id="IDEA-Research/grounding-dino-base", 
                input_dir="/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/outputs/Fog", 
                output_dir="/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/outputs/Fog/Fog_YOLO_Dino",
                classes_file="/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/datasets/DAWN/classes.txt")
    
    #read the existing annotations
    print("consolidating and validating annotations")
    existing_dir = "/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/outputs/Fog/Fog_YOLO"
    pred_dir = "/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/outputs/Fog/Fog_YOLO_Dino"
    output_dir = "/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/outputs/Fog/Fog_YOLO_Dino_consolidated"
    metrics = consolidate_annotations(existing_dir, pred_dir, output_dir)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    
'''