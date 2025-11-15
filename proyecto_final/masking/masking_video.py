import os
import cv2
import yaml

CONFIG_PATH = r"proyecto_final_vision_artificial_202502\proyecto_final\masking\threshold_config.yaml"

def load_config(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}
    
    # Valores por defecto para Otsu
    defaults = {
        "blur_size": 49,
        "morph_close_size": 0,
        "morph_open_size": 0,
        "invert": False
    }
    
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    
    return cfg

def apply_threshold(gray, config):
    """Aplica umbralización Otsu según configuración"""
    # Aplicar blur si está configurado
    if config["blur_size"] > 0:
        blur_size = config["blur_size"]
        if blur_size % 2 == 0:
            blur_size += 1
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Aplicar umbralización Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invertir si es necesario
    if config["invert"]:
        binary = cv2.bitwise_not(binary)
    
    # Operaciones morfológicas
    if config["morph_close_size"] > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (config["morph_close_size"], config["morph_close_size"]))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    if config["morph_open_size"] > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (config["morph_open_size"], config["morph_open_size"]))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary

def main():
    video_path = r"proyecto_final_vision_artificial_202502\proyecto_final\video_final.mp4"
    
    try:
        config = load_config(CONFIG_PATH)
        print(f"Loaded config: {config}")
    except Exception as e:
        print(f"Error loading config: {e}")
        config = load_config("")  # usar defaults
    
    capture = cv2.VideoCapture(video_path)
    window_name = "Binary Video - Otsu"
    cv2.namedWindow(window_name)

    # Preparar ruta de salida
    abs_video_path = os.path.abspath(video_path)
    dir_path = os.path.dirname(abs_video_path)
    output_name = "video_piezas_binarizado.mp4"
    output_path = os.path.join(dir_path, output_name)

    writer = None
    frame_ready = False

    while capture.isOpened():
        # Solo leer nuevo frame si se presionó 'a' o es el primer frame
        if not frame_ready:
            ret, frame = capture.read()
            if not ret:
                break
            # ESTE RESIZE ES OPCIONAL, DEPENDE DE LA RESOLUCIÓN DEL VIDEO. SI SE HACE, SOLO SE HACE
            # SOBRE UNA COPIA DEL FRAME, NO SOBRE EL FRAME ORIGINAL, SOLO PARA CALCULAR AREAS Y CONTORNOS
            frame_copy = cv2.resize(frame.copy(), (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            frame_ready = True
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        
        # Aplicar umbralización Otsu según configuración
        frame_binary = apply_threshold(gray, config)
        
        cv2.imshow(window_name, frame_binary)
        ## HASTAS AQUÍ USAS EL CODIGO PARA LA CAMERA.PY
    

        # Esperar indefinidamente hasta que se presione una tecla
        key = cv2.waitKey(0) & 0xFF
        if key == ord('a'):
            frame_ready = False
        elif key == ord("q"):
            # Salir del programa
            break
    capture.release()
    cv2.destroyAllWindows()
    
    print(f"Video guardado en: {output_path}")

if __name__ == "__main__":
    main()