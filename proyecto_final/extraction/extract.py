import cv2
import sys
import os
from datetime import datetime

"""
extract.py
Encuentra contornos en un vídeo binarizado, pausa cuando se detecta al menos
un contorno con área >= min_area. En pausa puedes guardar el contorno más grande
con las teclas:
    T / t -> tensores
    Z / z -> zetas
    O / o -> ochos
    A / a -> argollas

Para CONTINUAR (sin guardar) presiona 'n'
Salir con 'q' o ESC.

Uso simple (sin argparse):
    python extract.py [ruta_video] [min_area]
Si no se pasan argumentos usa valores por defecto.
Al iniciar se pedirá la carpeta base para guardar (enter = carpeta por defecto).
"""

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_largest_contour(frame, contour, out_dir, label_letter):
    x, y, w, h = cv2.boundingRect(contour)
    pad = 8
    h_f, w_f = frame.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_f, x + w + pad)
    y2 = min(h_f, y + h + pad)
    roi = frame[y1:y2, x1:x2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{label_letter}_{timestamp}.jpg"
    save_path = os.path.join(out_dir, filename)
    cv2.imwrite(save_path, roi)
    return save_path

def main():
    # Valores por defecto
    default_video = os.path.join("proyecto_final", "video_piezas_binarizado.mp4")
    default_min_area = 15000.0

    # Leer argumentos sencillos
    video_path = sys.argv[1] if len(sys.argv) > 1 else default_video
    try:
        min_area = float(sys.argv[2]) if len(sys.argv) > 2 else default_min_area
    except ValueError:
        print("min_area no es un número válido, usando valor por defecto.")
        min_area = default_min_area

    if not os.path.exists(video_path):
        print(f"Error: el archivo vídeo no existe: {video_path}")
        return

    # Pedir carpeta base para guardar
    default_out_base = os.path.join(os.path.dirname(r"proyecto_final\extraction"), "contornos_database")
    print(f"Ingrese carpeta base para guardar (enter = {default_out_base}): ", end="")
    user_in = input().strip()
    out_base = user_in if user_in else default_out_base
    ensure_dir(out_base)

    # Crear subcarpetas para las clases
    folders = {
        't': os.path.join(out_base, "tensores"),
        'z': os.path.join(out_base, "zetas"),
        'o': os.path.join(out_base, "ochos"),
        'a': os.path.join(out_base, "argollas"),
    }
    for p in folders.values():
        ensure_dir(p)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir el vídeo: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 0 else 30
    win = "Contornos"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("\nControles en pausa:")
    print("  T/t -> guardar en 'tensores'")
    print("  Z/z -> guardar en 'zetas'")
    print("  O/o -> guardar en 'ochos'")
    print("  A/a -> guardar en 'argollas'")
    print("  n   -> continuar sin guardar")
    print("  q / ESC -> salir\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Asegurar imagen en gris/binarizada
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Si no está perfectamente binarizada, aplicar umbral fijo
        _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Encontrar contornos
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) >= min_area]

        # Preparar imagen para mostrar (BGR)
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame.copy()

        # Dibujar contornos válidos y bounding boxes
        for c in valid:
            cv2.drawContours(vis, [c], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 128, 255), 2)

        # Información en pantalla
        text = f"Detectados: {len(valid)} (min_area={min_area}) - Presiona 'q' para salir"
        cv2.putText(vis, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(win, vis)

        # Si se detectó al menos uno, pausar hasta que se presione opción
        if len(valid) > 0:
            # seleccionar el contorno más grande
            largest = max(valid, key=cv2.contourArea)
            area_val = cv2.contourArea(largest)
            hint = "Pausado: T/Z/O/A=guardar, n=continuar, q=salir"
            cv2.putText(vis, hint, (10, vis.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
            cv2.imshow(win, vis)

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q') or key == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Saliendo.")
                    return
                # continuar sin guardar
                if key == ord('n'):
                    break
                # guardar segun tecla (acepta mayus/minus)
                kchar = chr(key).lower() if 0 <= key < 256 else ''
                if kchar in folders:
                    out_dir = folders[kchar]
                    saved = save_largest_contour(frame, largest, out_dir, kchar.upper())
                    print(f"Guardado: {saved}  (area={area_val:.1f})")
                    # opcional: mostrar mini-preview de la ROI por 500ms
                    roi = cv2.imread(saved)
                    if roi is not None:
                        cv2.imshow("Guardado_preview", roi)
                        cv2.waitKey(500)
                        cv2.destroyWindow("Guardado_preview")
                    break
                # si otra tecla, ignorar y seguir esperando
        else:
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q') or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()