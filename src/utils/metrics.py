def iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    
    Args:
        bbox1 (list or tuple): Bounding box [x1, y1, x2, y2].
        bbox2 (list or tuple): Bounding box [x1, y1, x2, y2].
    
    Returns:
        float: The IoU value.
    """
    # Determine the coordinates of the intersection rectangle.
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Compute the area of intersection rectangle.
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    # Compute the area of both bounding boxes.
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Compute the union area.
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area