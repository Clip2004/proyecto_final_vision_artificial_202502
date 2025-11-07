import cv2
import numpy as np
import joblib
import os
import yaml
from datetime import datetime
from codigos import *
import threading
import time





class Camera1Predictor:
    def __init__(self, model_path, hsv_config_path):
        """
        Inicializa el predictor con el modelo entrenado y la configuración HSV.
        """
        # Cargar modelo con joblib
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        self.model = joblib.load(model_path)
        
        # Cargar configuración HSV
        if not os.path.exists(hsv_config_path):
            raise FileNotFoundError(f"Configuración HSV no encontrada: {hsv_config_path}")
        
        with open(hsv_config_path, 'r') as f:
            self.hsv_config = yaml.safe_load(f)
        
        # Mapeo de clases
        self.class_names = {
            0: 'Argolla',
            1: 'Zeta',
            2: 'Tensor',
            3: 'ZetaR',
        }
        
        # Colores para visualización (BGR)
        self.class_colors = {
            0: (255, 0, 0),    # Azul
            1: (0, 255, 0),    # Verde
            2: (0, 0, 255),    # Rojo
            3: (255, 255, 0),  # Cian
            4: (255, 0, 255)   # Magenta
        }
        
        # Configuración de detección (igual que extract.py)
        self.min_area = 12000.0
        
        # Atributos para la GUI
        self.annotated_frame = None
        self.detected_object_frame = None
        self.last_detection_label = None
        self.is_running = False
        
        self.CONFIG_PATH = r"proyecto_final\masking\hsv_config.yaml"  # ajusta ruta si hace falta

        # Para detección asíncrona / evitar repetición
        self.detection_lock = threading.Lock()
        self._detection_thread = None
        self.last_detection_bbox = None
        self.frames_since_last_detection = 9999
        self.cooldown_frames = 15  # frames entre detecciones para la misma pieza
        self.iou_threshold = 0.3

    def load_config(self, path):
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
    
    def binarize_frame(self, frame):
        """
        Binariza el frame exactamente como extract.py
        """
        # Convertir a escala de grises si es necesario
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Aplicar umbral fijo como en extract.py
        _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        return bw
    
    def count_peaks(self, proj, height=0.0):
        """Cuenta picos en una proyección"""
        arr = np.asarray(proj).flatten()
        if arr.size == 0:
            return 0
        above = arr > height
        peaks = 0
        in_peak = False
        for v in above:
            if v and not in_peak:
                peaks += 1
                in_peak = True
            elif not v:
                in_peak = False
        return peaks
    
    def extract_features_enhanced(self, img_roi_bin, contour):
        """
        Extrae características mejoradas (EXACTAMENTE igual al script de entrenamiento)
        """
        # Características básicas originales
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        h, w = img_roi_bin.shape
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # 1. ANÁLISIS DE ESQUINAS
        corners = cv2.cornerHarris(img_roi_bin, 2, 3, 0.04)
        num_corners = np.sum(corners > 0.01 * corners.max()) if corners.max() > 0 else 0
        corner_density = num_corners / (h * w) if (h * w) > 0 else 0
        
        # 2. ANÁLISIS DE LÍNEAS HORIZONTALES Y VERTICALES
        proj_h = np.sum(img_roi_bin, axis=1)
        proj_v = np.sum(img_roi_bin, axis=0)
        
        h_peaks = self.count_peaks(proj_h, height=np.max(proj_h) * 0.3 if np.max(proj_h) > 0 else 0)
        v_peaks = self.count_peaks(proj_v, height=np.max(proj_v) * 0.3 if np.max(proj_v) > 0 else 0)
        
        # 3. ANÁLISIS DE SIMETRÍA
        left_half = img_roi_bin[:, :w//2]
        right_half = np.fliplr(img_roi_bin[:, w//2:])
        
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_w]
        right_half = right_half[:, :min_w]
        
        horizontal_symmetry = np.sum(left_half == right_half) / (h * min_w) if min_w > 0 else 0
        
        top_half = img_roi_bin[:h//2, :]
        bottom_half = np.flipud(img_roi_bin[h//2:, :])
        
        min_h = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_h, :]
        bottom_half = bottom_half[:min_h, :]
        
        vertical_symmetry = np.sum(top_half == bottom_half) / (min_h * w) if min_h > 0 else 0
        
        # 4. ANÁLISIS DE DENSIDAD POR CUADRANTES
        h_mid, w_mid = h//2, w//2
        
        q1_density = np.sum(img_roi_bin[:h_mid, :w_mid]) / (h_mid * w_mid) if (h_mid * w_mid) > 0 else 0
        q2_density = np.sum(img_roi_bin[:h_mid, w_mid:]) / (h_mid * (w - w_mid)) if (h_mid * (w - w_mid)) > 0 else 0
        q3_density = np.sum(img_roi_bin[h_mid:, :w_mid]) / ((h - h_mid) * w_mid) if ((h - h_mid) * w_mid) > 0 else 0
        q4_density = np.sum(img_roi_bin[h_mid:, w_mid:]) / ((h - h_mid) * (w - w_mid)) if ((h - h_mid) * (w - w_mid)) > 0 else 0
        
        # 5. ANÁLISIS DE CONTORNOS INTERNOS
        contours_all, hierarchy = cv2.findContours(img_roi_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        num_interior = 0
        area_ratio_interior = 0.0
        largest_hole_ratio = 0.0
        
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            hole_areas = []
            
            for i, hinfo in enumerate(hierarchy):
                parent = hinfo[3]
                if parent != -1:
                    num_interior += 1
                    hole_area = cv2.contourArea(contours_all[i])
                    hole_areas.append(hole_area)
                    area_ratio_interior += hole_area / (area + 1e-6)
            
            if num_interior > 0:
                area_ratio_interior /= num_interior
                largest_hole_ratio = max(hole_areas) / (area + 1e-6) if hole_areas else 0
        
        # 6. ANÁLISIS DE BORDES
        top_row_density = np.sum(img_roi_bin[0, :]) / w if w > 0 else 0
        bottom_row_density = np.sum(img_roi_bin[-1, :]) / w if w > 0 else 0
        left_col_density = np.sum(img_roi_bin[:, 0]) / h if h > 0 else 0
        right_col_density = np.sum(img_roi_bin[:, -1]) / h if h > 0 else 0
        
        # 7. CARACTERÍSTICAS DE FORMA ESPECÍFICAS
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
        extent = area / (bbox_w * bbox_h) if (bbox_w * bbox_h) > 0 else 0
        
        # Características de zona
        zone_features = []
        cell_size = 5
        
        for y_cell in range(0, h, cell_size):
            for x_cell in range(0, w, cell_size):
                roi_zone = img_roi_bin[y_cell:y_cell+cell_size, x_cell:x_cell+cell_size]
                if roi_zone.size > 0:
                    density = np.sum(roi_zone) / (roi_zone.size * 255)
                    zone_features.append(density)
        
        # COMBINAR TODAS LAS CARACTERÍSTICAS
        all_features = np.concatenate((
            # Básicas
            [area, perimeter, circularity, aspect_ratio],
            
            # Nuevas características específicas
            [num_corners, corner_density, h_peaks, v_peaks],
            [horizontal_symmetry, vertical_symmetry],
            [q1_density, q2_density, q3_density, q4_density],
            [num_interior, area_ratio_interior, largest_hole_ratio],
            [top_row_density, bottom_row_density, left_col_density, right_col_density],
            [solidity, extent],
            
            # Momentos de Hu
            hu_moments,
            
            # Densidades por zonas
            zone_features
        ))
        
        return all_features.astype(np.float32)
    
    def predict_contour(self, frame, contour):
        """
        Realiza predicción sobre un contorno detectado
        """
        # Extraer ROI con padding (exactamente como extract.py)
        x, y, w, h = cv2.boundingRect(contour)
        pad = 8
        h_f, w_f = frame.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_f, x + w + pad)
        y2 = min(h_f, y + h + pad)
        
        roi = frame[y1:y2, x1:x2]
        
        # Convertir a escala de grises si es necesario
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi.copy()
        
        # Redimensionar a 20x20 (como en el entrenamiento)
        roi_resized = cv2.resize(roi_gray, (20, 20), interpolation=cv2.INTER_NEAREST)
        
        # Extraer características
        features = self.extract_features_enhanced(roi_resized, contour)
        features = features.reshape(1, -1)
        
        # Predecir
        prediction = self.model.predict(features)[0]
        
        # Obtener probabilidades si el modelo lo soporta
        try:
            probabilities = self.model.predict_proba(features)[0]
            confidence = np.max(probabilities) * 100
        except:
            confidence = 0.0
        
        return prediction, confidence, (x, y, w, h), roi
    
    def run_camera(self, video_path):
        """
        Ejecuta la captura y predicción del video (detección en paralelo, sin pausar)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video no encontrado: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Error: No se pudo abrir el video: {video_path}")
        
        self.is_running = True
        
        def bbox_iou(b1, b2):
            if b1 is None or b2 is None:
                return 0.0
            x1,y1,w1,h1 = b1
            x2,y2,w2,h2 = b2
            xa = max(x1, x2)
            ya = max(y1, y2)
            xb = min(x1+w1, x2+w2)
            yb = min(y1+h1, y2+h2)
            inter_w = max(0, xb - xa)
            inter_h = max(0, yb - ya)
            inter = inter_w * inter_h
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - inter
            return inter / union if union > 0 else 0.0

        def async_predict(frame_snapshot, contour_snapshot):
            try:
                pred_class, confidence, bbox, roi = self.predict_contour(frame_snapshot, contour_snapshot)
                class_name = self.class_names.get(pred_class, "Desconocido")
                label = f"{class_name}: {confidence:.1f}%"
                with self.detection_lock:
                    self.last_detection_label = label
                    self.detected_object_frame = roi.copy() if roi is not None else None
                    self.last_detection_bbox = bbox
                    self.frames_since_last_detection = 0
            except Exception:
                # no bloquear loop por errores de predicción
                pass
            finally:
                with self.detection_lock:
                    self._detection_thread = None

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            frame_hsv = transform_BGR_HSV(frame.copy())
            hsv_cfg,_ = self.load_config(self.CONFIG_PATH)
            frame_binary = binaryColor(frame_hsv,
                                       hsv_cfg["Hmin"], hsv_cfg["Smin"], hsv_cfg["Vmin"],
                                       hsv_cfg["Hmax"], hsv_cfg["Smax"], hsv_cfg["Vmax"])

            # Usar la máscara HSV para detección y crear imagen BGR para visualización
            bw = frame_binary.copy()
            contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]

            # Visualización: convertir la máscara binaria a BGR y dibujar sobre ella
            vis = cv2.cvtColor(frame_binary, cv2.COLOR_GRAY2BGR)
            
            # Incrementar contador para cooldown
            self.frames_since_last_detection += 1

            # Procesar cada contorno válido (tomar el más grande)
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Dibujar contorno y bounding box siempre (no pausa)
                cv2.drawContours(vis, [largest_contour], -1, (0, 255, 0), 2)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 128, 255), 2)

                # Decidir si lanzar predicción asíncrona
                with self.detection_lock:
                    should_predict = (self._detection_thread is None and
                                      (self.frames_since_last_detection >= self.cooldown_frames and
                                       bbox_iou(self.last_detection_bbox, (x,y,w,h)) < self.iou_threshold))

                    # si no hay predicción en curso y cumple condiciones, lanzar hilo
                    if should_predict:
                        frame_snap = frame.copy()
                        contour_snap = largest_contour.copy()
                        self._detection_thread = threading.Thread(target=async_predict, args=(frame_snap, contour_snap), daemon=True)
                        self._detection_thread.start()

                # Mostrar estado de clasificación (si ya hay etiqueta o está procesando)
                with self.detection_lock:
                    label = self.last_detection_label or "Clasificando..."
                cv2.putText(vis, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Información general
            info_text = f"Detectados: {len(valid_contours)} (min_area={self.min_area})"
            cv2.putText(vis, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Actualizar frame anotado para la GUI
            self.annotated_frame = vis.copy()
            
            # # Mostrar ventana para debug (opcional)
            # cv2.imshow("Camera1", vis)
            # permitir salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Pequeña pausa para no saturar el CPU (ya hay waitKey)
            # time.sleep(0.005)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def stop(self):
        """
        Detiene la ejecución del video
        """
        self.is_running = False

def main():
    # Obtener directorio actual del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Subir un nivel desde gui/
    
    # Construir rutas relativas
    model_path = os.path.join(base_dir, "train", "models", "family_model.pkl")
    hsv_config_path = os.path.join(base_dir, "masking", "hsv_config.yaml")
    video_path = os.path.join(base_dir, "video_piezas.mp4")
    
    try:
        predictor = Camera1Predictor(model_path, hsv_config_path)
        predictor.run_camera(video_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()