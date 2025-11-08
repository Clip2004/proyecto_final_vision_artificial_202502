import os
import shutil

# Ruta de origen con los .jpg y .txt
source_folder = "proyecto_final_vision_artificial_202502\proyecto_final\yolo\output_labeled_aug"

# Rutas destino
dest_images = os.path.join('proyecto_final_vision_artificial_202502','proyecto_final','yolo','test', 'images')
dest_labels = os.path.join('proyecto_final_vision_artificial_202502','proyecto_final','yolo','test', 'labels')

# Crear carpetas destino si no existen
os.makedirs(dest_images, exist_ok=True)
os.makedirs(dest_labels, exist_ok=True)

# Listar y ordenar archivos para mantener orden consistente
files = sorted(os.listdir(source_folder))
for file_name in files:
    source_file = os.path.join(source_folder, file_name)

    if file_name.lower().endswith('.png'):
        # Copiar imagen con nombre original
        shutil.copy2(source_file, os.path.join(dest_images, file_name))
        print(f"Copiado imagen: {file_name} -> {dest_images}")

    elif file_name.lower().endswith('.txt'):
        # Copiar txt con nombre original
        shutil.copy2(source_file, os.path.join(dest_labels, file_name))
        print(f"Copiado label: {file_name} -> {dest_labels}")
