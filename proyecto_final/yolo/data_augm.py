import os
import json
import cv2
import albumentations as A
from datetime import datetime

# Carpeta con las imágenes y .json
input_folder = r"proyecto_final_vision_artificial_202502\proyecto_final\yolo\output_labeled"
output_folder = r"proyecto_final_vision_artificial_202502\proyecto_final\yolo\output_labeled_aug"
os.makedirs(output_folder, exist_ok=True)

# Transformaciones (puedes modificar)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.0), contrast_limit=0.0, p=0.8),
    A.Rotate(limit=15, p=0.5),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

# Función para recortar bbox a límites de imagen (píxeles)
def clip_bbox_to_image(bbox, width, height):
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, min(width - 1, x_min))
    y_min = max(0, min(height - 1, y_min))
    x_max = max(0, min(width - 1, x_max))
    y_max = max(0, min(height - 1, y_max))
    if x_max <= x_min:
        x_max = x_min + 1
    if y_max <= y_min:
        y_max = y_min + 1
    return [x_min, y_min, x_max, y_max]

# Función para convertir puntos (x1,y1,x2,y2) de LabelMe a formato Pascal VOC bbox
def points_to_bbox(points):
    x1, y1 = points[0]
    x2, y2 = points[1]
    return [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]  # [x_min, y_min, x_max, y_max]

# Procesar cada imagen con su .json
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        json_path = os.path.join(input_folder, filename)
        print(f"Procesando: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)

        image_filename = filename.replace(".json", ".png")
        image_path = os.path.join(input_folder, image_filename)
        if not os.path.exists(image_path):
            print(f"Imagen no encontrada: {image_path}")
            continue

        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        shapes = data["shapes"]

        # Extraer bboxes y labels
        bboxes = []
        labels = []
        for shape in shapes:
            if shape["shape_type"] == "rectangle":
                bbox = points_to_bbox(shape["points"])
                bboxes.append(bbox)
                labels.append(shape["label"])

        if not bboxes:
            continue

        # Recortar bboxes para que estén dentro de la imagen
        bboxes_clipped = [clip_bbox_to_image(bbox, width, height) for bbox in bboxes]

        # Aplicar transformación
        augmented = transform(image=image, bboxes=bboxes_clipped, category_ids=labels)
        image_aug = augmented['image']
        bboxes_aug = augmented['bboxes']
        labels_aug = augmented['category_ids']

        # Guardar imagen y json con nuevo nombre
        base_name = os.path.splitext(image_filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_base = f"{base_name}_aug_{timestamp}"
        new_image_path = os.path.join(output_folder, new_base + ".png")
        new_json_path = os.path.join(output_folder, new_base + ".json")

        cv2.imwrite(new_image_path, image_aug)

        # Convertir de nuevo bbox a puntos (2 esquinas) para LabelMe
        new_shapes = []
        for bbox, label in zip(bboxes_aug, labels_aug):
            x_min, y_min, x_max, y_max = bbox
            new_shape = {
                "label": label,
                "text": "",
                "points": [[x_min, y_min], [x_max, y_max]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            new_shapes.append(new_shape)

        new_data = {
            "version": data["version"],
            "flags": {},
            "shapes": new_shapes,
            "imagePath": os.path.basename(new_image_path),
            "imageData": None,
            "imageHeight": image_aug.shape[0],
            "imageWidth": image_aug.shape[1],
            "text": ""
        }

        with open(new_json_path, 'w') as f:
            json.dump(new_data, f, indent=2)

        print(f"✅ Guardado: {new_base}.png y .json")

print("✅ Finalizado el proceso de augmentación.")