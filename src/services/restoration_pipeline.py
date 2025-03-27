import os
import cv2
import glob
import numpy as np
import torch
import itertools
from tqdm import tqdm

from pathlib import Path
import sys
base_path = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_path)

from src.utils.preprocess import dehaze, denoise, enhance_contrast, sharpen
from src.utils.quality_assess import measure_quality


def convert_cv2_to_tensor(image):
    """
    Convert an OpenCV image (BGR, uint8) to a PyTorch tensor.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor

# Updated restoration functions that accept parameter dictionaries.

def preprocess_fog(image, dehaze_params=None, contrast_params=None):
    """
    Preprocess foggy images.
    Args:
        image (np.ndarray): Input image.
        dehaze_params (dict): Parameters for the dehaze function.
        contrast_params (dict): Parameters for the enhance_contrast function.
    """
    if dehaze_params is None:
        dehaze_params = {'filter_size': 7, 'sigma_color': 75, 'sigma_space': 75}
    if contrast_params is None:
        contrast_params = {'clipLimit': 2.0, 'tileGridSize': (8, 8)}
    
    image = dehaze(image, **dehaze_params)
    image = enhance_contrast(image, **contrast_params)

    params = {'dehaze_filter_size': dehaze_params['filter_size'],
             'dehaze_sigma_color': dehaze_params['sigma_color'],
             'dehaze_sigma_space': dehaze_params['sigma_space'],
             'contrast_clipLimit': contrast_params['clipLimit'],
             'contrast_tileGridSize': contrast_params['tileGridSize']}
    return image, params

def preprocess_snow(image, denoise_params=None, contrast_params=None):
    """
    Preprocess snowy images.
    Args:
        image (np.ndarray): Input image.
        denoise_params (dict): Parameters for the denoise function (expects method "median").
        contrast_params (dict): Parameters for the enhance_contrast function.
    """
    if denoise_params is None:
        denoise_params = {'method': "median", 'kernel_size': 3}
    if contrast_params is None:
        contrast_params = {'clipLimit': 2.0, 'tileGridSize': (8, 8)}
        
    image = denoise(image, **denoise_params)
    image = enhance_contrast(image, **contrast_params)
    params = {'denoise_method': denoise_params['method'],
            'denoise_kernel_size': denoise_params['kernel_size'],
             'contrast_clipLimit': contrast_params['clipLimit'],
             'contrast_tileGridSize': contrast_params['tileGridSize']}
    return image, params

def preprocess_rain(image, denoise_params=None, sharpen_params=None):
    """
    Preprocess rainy images.
    Args:
        image (np.ndarray): Input image.
        denoise_params (dict): Parameters for the denoise function (expects method "gaussian").
        sharpen_params (dict): Parameters for the sharpen function.
    """
    if denoise_params is None:
        denoise_params = {'method': "gaussian", 'kernel_size': 3}
    if sharpen_params is None:
        sharpen_params = {'kernel_size': 3}
    
    image = denoise(image, **denoise_params)
    image = sharpen(image, **sharpen_params)
    params = {'denoise_method': denoise_params['method'],
            'denoise_kernel_size': denoise_params['kernel_size'],
             'sharpen_kernel_size': sharpen_params['kernel_size']}
    return image, params

def preprocess_sandstorm(image, contrast_params=None, denoise_params=None):
    """
    Preprocess images captured in sandstorm conditions.
    Args:
        image (np.ndarray): Input image.
        contrast_params (dict): Parameters for the enhance_contrast function.
        denoise_params (dict): Parameters for the denoise function (expects method "gaussian").
    """
    if contrast_params is None:
        contrast_params = {'clipLimit': 2.0, 'tileGridSize': (8, 8)}
    if denoise_params is None:
        denoise_params = {'method': "gaussian", 'kernel_size': 3}
    
    image = enhance_contrast(image, **contrast_params)
    image = denoise(image, **denoise_params)
    params = {'contrast_clipLimit': contrast_params['clipLimit'],
             'contrast_tileGridSize': contrast_params['tileGridSize'],
             'denoise_method': denoise_params['method'],
             'denoise_kernel_size': denoise_params['kernel_size']}
    return image, params

def optimize_and_process_folder(restore_func, param_grid: dict, input_dir: str, output_dir: str, metric: str = "brisque"):
    """
    For a given restoration function, perform grid search optimization over the candidate parameters
    for each image in the input folder and then save the restored image using the best parameters.
    
    The param_grid should be a flat dictionary where keys are prefixed by the stage, e.g.:
      For preprocess_fog:
         {
           'dehaze_filter_size': [3, 5, 7],
           'dehaze_sigma_color': [50, 75, 100],
           'dehaze_sigma_space': [50, 75, 100],
           'contrast_clipLimit': [1.5, 2.0, 2.5],
           # Optionally, you can also grid over tileGridSize if needed.
         }
    The function splits the parameters based on the prefix (e.g. "dehaze_" or "contrast_") 
    and passes them to the restoration function accordingly.
    
    Args:
        restore_func (function): One of preprocess_fog, preprocess_snow, preprocess_rain, preprocess_sandstorm.
        param_grid (dict): Candidate parameter values.
        input_dir (str): Path to folder with input images.
        output_dir (str): Path to folder where restored images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images (common extensions)
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not load image {img_path}. Skipping.")
            continue
        
        best_score = measure_quality(img_path, metric=metric)
        #skipping preprocessing brisque score is already good enough
        if best_score >= 30:
            best_params = None
            
            # Generate all candidate combinations.
            keys = list(param_grid.keys())
            values = [param_grid[k] for k in keys]
            
            for combo in itertools.product(*values):
                candidate = dict(zip(keys, combo))
                
                # Split candidate into stage-specific parameters.
                stage_params = {}
                # Assume prefixes are separated by "_" e.g. "dehaze_filter_size" or "contrast_clipLimit"
                for key, value in candidate.items():
                    try:
                        prefix, param_name = key.split('_', 1)
                    except ValueError:
                        continue
                    if prefix not in stage_params:
                        stage_params[prefix] = {}
                    stage_params[prefix][param_name] = value
                
                # Depending on the restoration function, the expected parameter keys can be:
                # For preprocess_fog: "dehaze" and "contrast"
                # For preprocess_snow: "denoise" and "contrast"
                # For preprocess_rain: "denoise" and "sharpen"
                # For preprocess_sandstorm: "contrast" and "denoise"
                try:
                    # Call the restoration function with the extracted parameter dictionaries.
                    # Use .get(prefix, {}) to default to an empty dictionary if not provided.
                    if restore_func.__name__ == "preprocess_fog":
                        processed_img, default_params = restore_func(image.copy(),
                                                    dehaze_params=stage_params.get("dehaze", {}),
                                                    contrast_params=stage_params.get("contrast", {}))
                        best_params = default_params
                    elif restore_func.__name__ == "preprocess_snow":
                        processed_img, default_params = restore_func(image.copy(),
                                                    denoise_params=stage_params.get("denoise", {}),
                                                    contrast_params=stage_params.get("contrast", {}))
                        best_params = default_params
                    elif restore_func.__name__ == "preprocess_rain":
                        processed_img, default_params = restore_func(image.copy(),
                                                    denoise_params=stage_params.get("denoise", {}),
                                                    sharpen_params=stage_params.get("sharpen", {}))
                        best_params = default_params
                    elif restore_func.__name__ == "preprocess_sandstorm":
                        processed_img, default_params = restore_func(image.copy(),
                                                    contrast_params=stage_params.get("contrast", {}),
                                                    denoise_params=stage_params.get("denoise", {}))
                        best_params = default_params
                    else:
                        raise ValueError("Unknown restoration function.")
                except Exception as e:
                    print(f"Error with candidate {candidate}: {e}")
                    continue
                
                tensor_img = convert_cv2_to_tensor(processed_img)
                score = measure_quality(tensor_img, metric="brisque")
                # Debug print for candidate parameters.
                print(f"Candidate: {candidate} --> BRISQUE Score: {score}", end='\r')
                
                if score < best_score:
                    best_score = score
                    best_params = candidate.copy()
            
            print(f"Optimized parameters for image {os.path.basename(img_path)}: {best_params} with score {best_score}")
            
            # After obtaining best parameters, split them as before.
            best_stage_params = {}
            for key, value in best_params.items():
                prefix, param_name = key.split('_', 1)
                if prefix not in best_stage_params:
                    best_stage_params[prefix] = {}
                best_stage_params[prefix][param_name] = value
            
            # Apply final transformation with best parameters.
            if restore_func.__name__ == "preprocess_fog":
                final_img,_ = restore_func(image.copy(),
                                        dehaze_params=best_stage_params.get("dehaze", {}),
                                        contrast_params=best_stage_params.get("contrast", {}))
            elif restore_func.__name__ == "preprocess_snow":
                final_img,_ = restore_func(image.copy(),
                                        denoise_params=best_stage_params.get("denoise", {}),
                                        contrast_params=best_stage_params.get("contrast", {}))
            elif restore_func.__name__ == "preprocess_rain":
                final_img,_ = restore_func(image.copy(),
                                        denoise_params=best_stage_params.get("denoise", {}),
                                        sharpen_params=best_stage_params.get("sharpen", {}))
            elif restore_func.__name__ == "preprocess_sandstorm":
                final_img,_ = restore_func(image.copy(),
                                        contrast_params=best_stage_params.get("contrast", {}),
                                        denoise_params=best_stage_params.get("denoise", {}))
            else:
                raise ValueError("Unknown restoration function.")
            
            # Save the final image in the output directory.
            out_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, final_img)
        else:
            print(f"Image {os.path.basename(img_path)} is already clear. Skipping.")
            out_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, image)
# Example usage if run as a script
'''if __name__ == "__main__":
    # Optimize and process folder for foggy images.
    print("Optimizing and processing foggy images...")
    param_grid = {
        'dehaze_filter_size': [5, 7],
        'dehaze_sigma_color': [75, 100],
        'dehaze_sigma_space': [75, 100],
        'contrast_clipLimit': [1.75, 2],
        'contrast_tileGridSize': [(8, 8), (4, 4)]
    }
    
    input_directory = "/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/datasets/DAWN/Fog"
    output_directory = "/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/outputs/Fog"
    
    optimize_and_process_folder(preprocess_fog, param_grid, input_directory, output_directory)


    print("Optimizing and processing rainy images...")
    # Optimize and process folder for Rain images.
    param_grid = {
        'denoise_method': ['gaussian', 'median'],
        'denoise_kernel_size': [3, 5],
        'sharpen_kernel_size': [3, 5]
    }
    
    input_directory = "/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/datasets/DAWN/Rain"
    output_directory = "/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/outputs/Rain"
    
    # For example, optimize the fog restoration pipeline.
    optimize_and_process_folder(preprocess_rain, param_grid, input_directory, output_directory)



    #Optimize and process folder for Snow images.
    print("Optimizing and processing snowy images...")
    param_grid = {
        'denoise_method': ['gaussian', 'median'],
        'denoise_kernel_size': [3, 5],
        'contrast_clipLimit': [1.75,2],
        'contrast_tileGridSize': [(8, 8), (4, 4)]
    }
    
    input_directory = "/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/datasets/DAWN/Snow"
    output_directory = "/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/outputs/Snow"
    
    # For example, optimize the fog restoration pipeline.
    optimize_and_process_folder(preprocess_snow, param_grid, input_directory, output_directory)



    #Optimize and process folder for Sandstorm images.
    print("Optimizing and processing sandstorm images...")
    param_grid = {
        'contrast_clipLimit': [1.5, 1.75],
        'contrast_tileGridSize': [(8, 8), (4, 4)],
        'denoise_method': ['gaussian', 'median'],
        'denoise_kernel_size': [3, 5]
    }
    
    input_directory = "/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/datasets/DAWN/Sand"
    output_directory = "/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/outputs/Sand"
    
    # For example, optimize the fog restoration pipeline.
    optimize_and_process_folder(preprocess_sandstorm, param_grid, input_directory, output_directory)'''

