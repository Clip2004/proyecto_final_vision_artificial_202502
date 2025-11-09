import os
import cv2
import yaml
import numpy as np
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

    # preparar ruta de salida en la misma carpeta del script (proyecto_final_vision_artificial_202502\proyecto_final\yolo)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # carpeta para guardar frames originales (ahora dentro de ...\proyecto_final\yolo)
    original_dir = os.path.join(script_dir, "original_frames")
    os.makedirs(original_dir, exist_ok=True)

    if use_trackbars:
        create_HSV_trackbarColor(window_name)
        # fijar posiciones iniciales según el YAML
        cv2.setTrackbarPos("Hmin", window_name, int(hsv_cfg["Hmin"]))
        cv2.setTrackbarPos("Smin", window_name, int(hsv_cfg["Smin"]))
        cv2.setTrackbarPos("Vmin", window_name, int(hsv_cfg["Vmin"]))
        cv2.setTrackbarPos("Hmax", window_name, int(hsv_cfg["Hmax"]))
        cv2.setTrackbarPos("Smax", window_name, int(hsv_cfg["Smax"]))
        cv2.setTrackbarPos("Vmax", window_name, int(hsv_cfg["Vmax"]))

    frame_idx = 0
    saved_count = 0  # contador de frames guardados (solo cuando se detecta objeto en el centro)
    object_present = False  # bandera para indicar presencia actual del objeto (cualquier posición)
    saved_in_presence = False  # bandera para guardar solo una vez por aparición

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        frame_idx += 1
        # reducir tamaño para procesamiento
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
        if frame_binary is None:
            continue

        # asegurar máscara en gris
        if len(frame_binary.shape) == 3:
            gray = cv2.cvtColor(frame_binary, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_binary

        # encontrar contornos y contar cada objeto solo una vez por frame
        contours, _ = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 150  # umbral para filtrar ruido; ajustar según necesidad
        dedup_dist = 30  # distancia en píxeles para considerar mismo objeto

        unique_centroids = []
        detected = 0
        annotated = frame.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # comprobar si centro ya pertenece a un objeto detectado
            is_new = True
            for (ux, uy) in unique_centroids:
                if np.hypot(cx - ux, cy - uy) <= dedup_dist:
                    is_new = False
                    break
            if is_new:
                unique_centroids.append((cx, cy))
                detected += 1
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

        # mostrar contador sobre la máscara binaria convertida a color para visibilidad
        bgr_to_write = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.putText(bgr_to_write, f"Detected: {detected}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(window_name, bgr_to_write)

        # actualizar banderas de presencia
        if detected > 0 and not object_present:
            object_present = True
            saved_in_presence = False  # nueva aparición: permitir guardar si centro coincide
        elif detected == 0 and object_present:
            # aparición terminó -> reset
            object_present = False
            saved_in_presence = False

        # calcular región central horizontal (ignorar posición vertical)
        h, w = frame.shape[:2]
        left = int(w * 0.25)
        right = int(w * 0.75)
        # dibujar franja central vertical (ignorar y)
        cv2.rectangle(annotated, (left, 0), (right, h), (255, 0, 0), 1)

        # si hay al menos un centro con coordenada x dentro de la franja central
        center_hit = any(left <= cx <= right for (cx, cy) in unique_centroids)
        if object_present and center_hit and not saved_in_presence:
            saved_count += 1
            saved_path = os.path.join(original_dir, f"frame_saved_{saved_count:06d}.png")
            cv2.imwrite(saved_path, frame)
            cv2.imshow("Saved Original Frame", frame)
            saved_in_presence = True

        # mostrar anotaciones (opcional)
        cv2.imshow("Annotated", annotated)

        if cv2.waitKey(20) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()