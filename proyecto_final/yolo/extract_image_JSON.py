import os
import shutil

# Ruta a tu carpeta original con imágenes y JSONs
carpeta_origen = "proyecto_final_vision_artificial_202502\proyecto_final\yolo\original_frames"
# Ruta a la carpeta donde quieres copiar los archivos filtrados
carpeta_destino = "proyecto_final_vision_artificial_202502\proyecto_final\yolo\output_labeled"

# Crear carpeta destino si no existe
os.makedirs(carpeta_destino, exist_ok=True)

# Obtener todos los archivos en la carpeta origen
archivos = os.listdir(carpeta_origen)

# Filtrar nombres base de archivos .png que tengan un .json con el mismo nombre
for archivo in archivos:
    if archivo.lower().endswith(".png"):
        nombre_base = os.path.splitext(archivo)[0]
        nombre_json = nombre_base + ".json"

        if nombre_json in archivos:
            # Copiar ambos archivos
            ruta_jpg = os.path.join(carpeta_origen, archivo)
            ruta_json = os.path.join(carpeta_origen, nombre_json)

            shutil.copy2(ruta_jpg, carpeta_destino)
            shutil.copy2(ruta_json, carpeta_destino)

            print(f"Copiado: {archivo} y {nombre_json}")

print("✅ Solo se copiaron pares .jpg + .json con el mismo nombre.")