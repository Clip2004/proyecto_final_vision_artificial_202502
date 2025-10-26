import cv2
import os
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from codigos import *

def main():
    # 1. Leer imagen
    image_path = r"proyecto_final\masking\frames\frame_000293.png"
    image = read_image(image_path)[0]

    # 2. Crear ventana con trackbars
    window_name = "Binary Image"
    create_HSV_trackbarColor(window_name)

    # Establecer posiciones iniciales UNA vez (no en cada iteración)
    cv2.setTrackbarPos("Hmin", window_name, 0)
    cv2.setTrackbarPos("Smin", window_name, 0)
    cv2.setTrackbarPos("Vmin", window_name, 70)
    cv2.setTrackbarPos("Hmax", window_name, 255)
    cv2.setTrackbarPos("Smax", window_name, 255)
    cv2.setTrackbarPos("Vmax", window_name, 255)

    while True:
        # 3. Transformar a HSV
        image_HSV = transform_BGR_HSV(image.copy())

        # 4. Obtener valores de trackbars (ya ajustables por el usuario)
        h_min = cv2.getTrackbarPos("Hmin", window_name)
        s_min = cv2.getTrackbarPos("Smin", window_name)
        v_min = cv2.getTrackbarPos("Vmin", window_name)
        h_max = cv2.getTrackbarPos("Hmax", window_name)
        s_max = cv2.getTrackbarPos("Smax", window_name)
        v_max = cv2.getTrackbarPos("Vmax", window_name)

        # 5. Crear imagen binaria
        image_binary = binaryColor(image_HSV, h_min, s_min, v_min, h_max, s_max, v_max)

        # 6. Mostrar imagen binaria
        cv2.imshow(window_name, image_binary)

        # 7. Esperar interacción del usuario
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Guardar configuración en YAML en la misma carpeta que este archivo
            cfg = {
                "Hmin": int(h_min),
                "Smin": int(s_min),
                "Vmin": int(v_min),
                "Hmax": int(h_max),
                "Smax": int(s_max),
                "Vmax": int(v_max),
            }
            folder = os.path.dirname(__file__)
            yaml_path = os.path.join(folder, "hsv_config.yaml")
            try:
                if _HAS_YAML:
                    with open(yaml_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
                else:
                    # Si no hay PyYAML, usar OpenCV FileStorage (escribir YAML)
                    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_WRITE)
                    for k, v in cfg.items():
                        fs.write(k, v)
                    fs.release()
                print(f"HSV configuration saved to: {yaml_path}")
            except Exception as e:
                print("Error saving YAML:", e)
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()