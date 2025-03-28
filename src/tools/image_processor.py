from agents import function_tool
from pathlib import Path
import sys
base_path = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_path)

from src.services.restoration_pipeline import optimize_and_process_folder, preprocess_fog
from src.services.annotation import detect_annotation_format, hf_annotate, consolidate_annotations
from src.utils.annotation_converter import convert_voc_to_yolo

@function_tool
def image_processor(input_directory: str, output_directory: str, model_id: str, metric: str):
    print("The following inputs were provided: \n input_directory: ", input_directory, "\n output_directory: ", output_directory, "\n model_id: ", model_id, "\n metric: ", metric)
    print("Optimizing and processing foggy images...")
    param_grid = {
        'dehaze_filter_size': [5, 7],
        'dehaze_sigma_color': [75, 100],
        'dehaze_sigma_space': [75, 100],
        'contrast_clipLimit': [1.75, 2],
        'contrast_tileGridSize': [(8, 8), (4, 4)]
    }

    optimize_and_process_folder(preprocess_fog, param_grid, input_directory, output_directory, metric)

    #detect if there are any existing annotations, if yes, then standardize them to common YOLO format
    annotation_format = detect_annotation_format(input_directory)
    print("Annotation files detected!")
    print(f"Detected annotation format: {annotation_format}")

    if annotation_format == "PascalVOC":
        convert_voc_to_yolo(
            voc_dir=f"{input_directory}/Fog_PASCAL_VOC",
            yolo_dir=f"{output_directory}/Fog_YOLO",
            classes_file=f"{input_directory}/classes.txt"
        )
    
    #perform annotation using grounding dino model
    print("performing annotation using grounding dino model")
    hf_annotate(model_id=model_id,
                input_dir=output_directory, 
                output_dir=f"{output_directory}/Fog_YOLO_Dino",
                classes_file=f"{input_directory}/classes.txt")
    
    #read the existing annotations
    existing_dir = f"{output_directory}/Fog_YOLO"
    pred_dir = f"{output_directory}/Fog_YOLO_Dino"
    output_dir = f"{output_directory}/Fog_YOLO_Dino_consolidated"
    metrics = consolidate_annotations(existing_dir, pred_dir, output_dir)
    print("Performance of grounding Dino in comparison to existing annotations:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print("Image processing and annotation complete!")