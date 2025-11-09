import os
import shutil
import json

images_labeled = r"proyecto_final_vision_artificial_202502\proyecto_final\yolo\output_labeled_aug"
label_mapping = {
    "argolla": 0,
    "zeta": 1,
    "tensor": 2,
    "zetaRedonda" : 3
}

for file_name in os.listdir(images_labeled):
    if file_name.endswith(".json"):
        json_path = os.path.join(images_labeled, file_name)
        with open(json_path, "r") as f:
            data = json.load(f)
        
        img_width = data["imageWidth"]
        img_height = data["imageHeight"]

        output_lines = []
        for shape in data["shapes"]:
            label = shape["label"]
            if label not in label_mapping:
                print(f"Etiqueta '{label}' no está en el label_mapping")
                continue
            
            class_id = label_mapping[label]
            points = shape["points"]
            #pasar al formato de YOLO
            x1,y1 = points[0]
            x2,y2 = points[1]

            x_min = min(x1,x2)
            y_min = min(y1,y2)

            x_max = max(x1,x2)
            y_max = max(y1,y2)

            #calcular centroide
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            #Normalizar coordenadas entre 0 y 1
            x_center_norm = x_center / img_width
            y_center_norm   = y_center / img_height
            bbox_width_norm = bbox_width / img_width
            bbox_height_norm = bbox_height / img_height

            line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {bbox_width_norm} {bbox_height_norm}"
            output_lines.append(line)

        txt_file_name = file_name.replace(".json", ".txt")
        txt_path = os.path.join(images_labeled, txt_file_name)
        with open(txt_path, 'w') as f:
            f.write("\n".join(output_lines))