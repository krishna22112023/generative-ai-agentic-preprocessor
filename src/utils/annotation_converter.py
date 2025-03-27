import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(voc_dir, yolo_dir, classes_file):

    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    os.makedirs(yolo_dir, exist_ok=True)

    converted_files_count = 0  # Counter for converted files

    for xml_file in os.listdir(voc_dir):
        if not xml_file.endswith('.xml'):
            continue

        # Parse the XML file
        tree = ET.parse(os.path.join(voc_dir, xml_file))
        root = tree.getroot()

        size = root.find('size')
        w_img = int(size.find('width').text)
        h_img = int(size.find('height').text)

        image_filename = root.find('filename').text
        image_name = os.path.splitext(image_filename)[0]

        yolo_file = os.path.join(yolo_dir, f"{image_name}.txt")
        with open(yolo_file, 'w') as f:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in classes:
                    continue

                class_id = classes.index(class_name)
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                x_center = ((xmin + xmax) / 2) / w_img
                y_center = ((ymin + ymax) / 2) / h_img
                box_width = (xmax - xmin) / w_img
                box_height = (ymax - ymin) / h_img


                f.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

        converted_files_count += 1

    # Print summary of the conversion process
    print(f"Conversion completed!")
    print(f"Total files converted: {converted_files_count}")
    print(f"YOLO annotations saved in: {os.path.abspath(yolo_dir)}")

