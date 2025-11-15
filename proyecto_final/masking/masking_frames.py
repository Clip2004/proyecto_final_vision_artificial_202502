import cv2
import os
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from codigos import *

def main():
    # 1. Leer imagen - usar ruta absoluta
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "frames", "frame_final.png")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    image = read_image(image_path)[0]
    if image is None:
        print(f"Error: Failed to read image from {image_path}")
        return
        
    image = cv2.resize(image, (1920//2, 1080//2))
    
    # 2. Crear ventana con trackbars
    window_name = "Adaptive Thresholding"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Method", window_name, 0, 1, lambda x: None)  # 0=Otsu, 1=Adaptive
    cv2.createTrackbar("Block Size", window_name, 11, 99, lambda x: None)  # Para adaptive
    cv2.createTrackbar("C", window_name, 2, 20, lambda x: None)  # Constante para adaptive
    cv2.createTrackbar("Blur", window_name, 1, 50, lambda x: None)
    cv2.createTrackbar("Morph Close", window_name, 5, 50, lambda x: None)
    cv2.createTrackbar("Morph Open", window_name, 2, 50, lambda x: None)
    cv2.createTrackbar("Invert", window_name, 0, 1, lambda x: None)

    while True:
        # 3. Convertir a escala de grises
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        
        # 4. Aplicar desenfoque opcional
        blur_size = cv2.getTrackbarPos("Blur", window_name)
        if blur_size > 0:
            blur_size = blur_size * 2 + 1
            gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        # 5. Aplicar umbralización según método seleccionado
        method = cv2.getTrackbarPos("Method", window_name)
        
        if method == 0:  # Otsu
            _, image_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:  # Adaptive
            block_size = cv2.getTrackbarPos("Block Size", window_name)
            block_size = block_size * 2 + 1  # Asegurar impar
            if block_size < 3:
                block_size = 3
            c_value = cv2.getTrackbarPos("C", window_name)
            
            image_binary = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size, c_value
            )
        
        # Invertir si es necesario
        if cv2.getTrackbarPos("Invert", window_name) == 1:
            image_binary = cv2.bitwise_not(image_binary)
        
        # 6. Operaciones morfológicas
        morph_close = cv2.getTrackbarPos("Morph Close", window_name)
        morph_open = cv2.getTrackbarPos("Morph Open", window_name)
        
        if morph_close > 0:
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
            image_binary = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, kernel_close)
        
        if morph_open > 0:
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
            image_binary = cv2.morphologyEx(image_binary, cv2.MORPH_OPEN, kernel_open)

        # 7. Mostrar resultado
        cv2.imshow(window_name, image_binary)

        # 8. Guardar configuración
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cfg = {
                "method": "adaptive" if method == 1 else "otsu",
                "block_size": int(cv2.getTrackbarPos("Block Size", window_name) * 2 + 1) if method == 1 else 0,
                "c_value": int(cv2.getTrackbarPos("C", window_name)) if method == 1 else 0,
                "blur_size": int(blur_size) if blur_size > 0 else 0,
                "morph_close_size": int(morph_close),
                "morph_open_size": int(morph_open),
                "invert": bool(cv2.getTrackbarPos("Invert", window_name))
            }
            folder = os.path.dirname(__file__)
            yaml_path = os.path.join(folder, "threshold_config.yaml")
            try:
                if _HAS_YAML:
                    with open(yaml_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
                else:
                    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_WRITE)
                    for k, v in cfg.items():
                        if isinstance(v, str):
                            fs.write(k, v)
                        else:
                            fs.write(k, int(v) if isinstance(v, bool) else v)
                    fs.release()
                print(f"Configuration saved to: {yaml_path}")
            except Exception as e:
                print("Error saving YAML:", e)
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()