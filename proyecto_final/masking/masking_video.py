import os
import cv2
import yaml
from codigos import *

CONFIG_PATH = r"proyecto_final\masking\hsv_config.yaml"  # ajusta ruta si hace falta

def load_config(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}
    hsv_cfg = (cfg.get("hsv") or {})
    defaults = {"Hmin": 0, "Smin": 0, "Vmin": 70, "Hmax": 255, "Smax": 255, "Vmax": 255}
    for k, v in defaults.items():
        hsv_cfg.setdefault(k, v)
    use_trackbars = cfg.get("use_trackbars", True)
    return hsv_cfg, use_trackbars

def main():
    video_path = r"proyecto_final\video_piezas.mp4"
    hsv_cfg, use_trackbars = load_config(CONFIG_PATH)

    capture = cv2.VideoCapture(video_path)
    window_name = "Binary Video"

    # preparar ruta de salida en la misma carpeta del video original
    abs_video_path = os.path.abspath(video_path)
    dir_path = os.path.dirname(abs_video_path)
    output_name = "video_piezas_binarizado.mp4"
    output_path = os.path.join(dir_path, output_name)

    if use_trackbars:
        create_HSV_trackbarColor(window_name)
        # fijar posiciones iniciales según el YAML
        cv2.setTrackbarPos("Hmin", window_name, int(hsv_cfg["Hmin"]))
        cv2.setTrackbarPos("Smin", window_name, int(hsv_cfg["Smin"]))
        cv2.setTrackbarPos("Vmin", window_name, int(hsv_cfg["Vmin"]))
        cv2.setTrackbarPos("Hmax", window_name, int(hsv_cfg["Hmax"]))
        cv2.setTrackbarPos("Smax", window_name, int(hsv_cfg["Smax"]))
        cv2.setTrackbarPos("Vmax", window_name, int(hsv_cfg["Vmax"]))

    writer = None

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        frame_HSV = transform_BGR_HSV(frame.copy())

        if use_trackbars:
            h_min = cv2.getTrackbarPos("Hmin", window_name)
            s_min = cv2.getTrackbarPos("Smin", window_name)
            v_min = cv2.getTrackbarPos("Vmin", window_name)
            h_max = cv2.getTrackbarPos("Hmax", window_name)
            s_max = cv2.getTrackbarPos("Smax", window_name)
            v_max = cv2.getTrackbarPos("Vmax", window_name)
        else:
            # usar valores del YAML sin trackbars
            h_min = int(hsv_cfg["Hmin"])
            s_min = int(hsv_cfg["Smin"])
            v_min = int(hsv_cfg["Vmin"])
            h_max = int(hsv_cfg["Hmax"])
            s_max = int(hsv_cfg["Smax"])
            v_max = int(hsv_cfg["Vmax"])

        frame_binary = binaryColor(frame_HSV, h_min, s_min, v_min, h_max, s_max, v_max)
        cv2.imshow(window_name, frame_binary)

        # preparar y escribir en el VideoWriter (convertir a BGR para compatibilidad)
        if frame_binary is None:
            continue
        if len(frame_binary.shape) == 3:
            gray = cv2.cvtColor(frame_binary, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_binary
        bgr_to_write = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if writer is None:
            height, width = gray.shape
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 20.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height), True)

        writer.write(bgr_to_write)

        if cv2.waitKey(20) == ord("q"):
            break

    if writer is not None:
        writer.release()
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()