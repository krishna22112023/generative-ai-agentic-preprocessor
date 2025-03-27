import pyiqa
import torch
from torch import Tensor

def measure_quality(img_tensor_x : Tensor, metric:str):
    """_summary_

    Args:
        metric (str): No reference metric to be used for quality assessment. 
        Refer https://iqa-pytorch.readthedocs.io/en/latest/ModelCard.html for more details.
    """
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

    iqa_metric = pyiqa.create_metric(metric, device=device)

    # Tensor inputs, img_tensor_x: (N, 3, H, W)
    score_nr = iqa_metric(img_tensor_x)

    return score_nr.item()

'''if __name__ == "__main__":
    img = '/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/datasets/DAWN/Fog/foggy-001-processed.jpg'
    print(measure_quality(img, 'brisque'))
    img = '/Users/krishnaiyer/generative-ai-HTX-RCC-preprocess-demo/data/datasets/DAWN/Fog/foggy-001.jpg'
    print(measure_quality(img, 'brisque'))'''