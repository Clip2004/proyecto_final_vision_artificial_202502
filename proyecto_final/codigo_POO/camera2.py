import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import os
import traceback


class RunCamera(threading.Thread):
    """
    Clase para capturar video y realizar detección de objetos en tiempo real
    usando un modelo YOLO.
    """

    def __init__(self, src=0, name="Camera", model_path="proyecto_final/yolo/best.pt"):
        super().__init__()
        self.src = src
        self.name = name
        self.daemon = True
        self.plate_count = 0

        # Estado interno
        self.running = False
        self.cap = None
        self.frame_count = 0
        self.line1_x = 900  # línea izquierda
        self.line2_x = 975  # línea derecha
        self.ebilla_detected = False
        self.last_centroid = None

        # Variables para la GUI
        self.annotated_frame = None
        self.detected_object_frame = None
        self.last_detection_label = None
        self.object_counter = 0
        self.min_area_threshold = 800  # ajustable

       
        # --- Cargar modelo YOLO ---
        print(f"[{self.name}] Cargando modelo YOLO desde: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo YOLO en {model_path}")
        try:
            self.model = YOLO(model_path).to("cuda")
            print(f"[{self.name}] Modelo YOLO cargado correctamente.")
        except Exception as e:
            print(f"[{self.name}] Error al cargar el modelo YOLO: {e}")
            traceback.print_exc()
            self.model = None
            
        self.object_counts = {
            "argolla": 0,
            "tensor": 0,
            "zeta": 0,
            "zetaredonda": 0,
            "Piezas defectuosas": 0
        }

        # (Opcional para futuras ampliaciones)
        self.size_counts = {
            "Argollas": {"S": 0, "M": 0, "L": 0, "XL": 0},
            "Tensores": {"S": 0, "M": 0, "L": 0, "XL": 0},
            "Zetas": {"S": 0, "M": 0, "L": 0, "XL": 0},
            "Zetas redondeadas": {"S": 0, "M": 0, "L": 0, "XL": 0},
            "Piezas defectuosas": {}
        }

        # Control de hilo
        self._stop_event = threading.Event()

        print(f"[{self.name}] RunCamera inicializado (ID: {id(self)})")

    # ================================================================
    # MÉTODO PRINCIPAL DE CAPTURA
    # ================================================================
    def run(self):
        """Hilo principal de captura de video y detección YOLO."""
        self.running = True

        try:
            self.cap = cv2.VideoCapture(self.src)
            if not self.cap.isOpened():
                print(f"[{self.name}] Error: No se pudo abrir la fuente de video {self.src}")
                return

            print(f"[{self.name}] Cámara iniciada correctamente.")

            while self.running and not self._stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print(f"[{self.name}] Fin del video o error en captura.")
                    # Si es un archivo de video, reinicia desde el principio
                    if isinstance(self.src, str):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

                self._process_frame(frame)
                self.frame_count += 1
                time.sleep(0.01)

        except Exception as e:
            print(f"[{self.name}] Error en ejecución: {e}")
            traceback.print_exc()

        finally:
            self._cleanup()

    # ================================================================
    # PROCESAMIENTO DE FRAME
    # ================================================================
    def _process_frame(self, frame):
        """Procesa un frame con YOLO, detecta objetos y aplica lógica de líneas."""

        annotated = frame.copy()
        cv2.line(annotated, (self.line1_x, 0), (self.line1_x, frame.shape[0]), (255, 0, 0), 2)
        cv2.line(annotated, (self.line2_x, 0), (self.line2_x, frame.shape[0]), (255, 0, 0), 2)
        
        if self.model is None:
                print(f"[{self.name}] Modelo YOLO no cargado.")
                return

        if self.modo_familias:
            try:
                results = self.model.predict(frame, conf=0.85, verbose=False)
                result = results[0]
                annotated = result.plot()

                if len(result.boxes) > 0:
                    best_idx = result.boxes.conf.argmax()
                    box = result.boxes[best_idx]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # ROI del objeto detectado
                    roi = frame[y1:y2, x1:x2].copy()
                    

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[class_id]
                    nombre_clase = f"{class_name} ({confidence:.2f})"

                    # Dibuja el rectángulo y el centro
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(annotated, nombre_clase, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # --- Lógica de cruce de líneas ---
                    if cx > self.line2_x and not self.ebilla_detected:
                        self.ebilla_detected = True

                        if roi.size > 0:
                            # Mostrar ROI en la GUI
                            # cv2.imshow("roi", roi)  ############### REVISAR
                            self.detected_object_frame = roi
                            self.last_detection_label = nombre_clase
                            

                            #  Ejecutar análisis de defectos con tu función
                            es_defectuosa = self._analizar_defectos(class_name,frame, roi)

                            # Actualizar etiqueta en GUI
                            # Actualizar etiqueta en GUI
                            estado = "Defectuosa" if es_defectuosa else "OK"
                            self.last_detection_label = f"{class_name} ({estado})"

                            # Actualizar contadores
                            class_map = {
                                "argolla": "Argollas",
                                "tensor": "Tensores",
                                "zeta": "Zetas",
                                "zeta redondeada": "Zetas redondeadas"
                            }

                            # Determinar la clave de familia
                            key = class_map.get(class_name.lower(), class_name.capitalize())

                            # --- Nueva lógica de conteo ---
                            if es_defectuosa:
                                defect_key = "Piezas defectuosas"
                                self.object_counts[defect_key] = self.object_counts.get(defect_key, 0) + 1
                                print(f"[{self.name}] Pieza defectuosa detectada → Total: {self.object_counts[defect_key]}")
                            else:
                                self.object_counts[key] = self.object_counts.get(key, 0) + 1
                                print(f"[{self.name}] {key} detectada → Total: {self.object_counts[key]}")
                            self.object_counter += 1
                            print(f"[{self.name}] Objeto #{self.object_counter} contado")

                    elif cx < self.line1_x and self.ebilla_detected:
                        self.ebilla_detected = False

                # Actualizar frame anotado
                self.annotated_frame = annotated.copy()

            except Exception as e:
                print(f"[{self.name}] Error en modo familias: {e}")
                import traceback
                traceback.print_exc()
                self.annotated_frame = frame.copy()

        elif self.modo_mixto:

            try:
                results = self.model.predict(frame, conf=0.85, verbose=False)
                result = results[0]
                annotated = result.plot()

                if len(result.boxes) > 0:
                    best_idx = result.boxes.conf.argmax()
                    box = result.boxes[best_idx]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # ROI del objeto detectado
                    roi = frame[y1:y2, x1:x2].copy()

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[class_id]
                    nombre_clase = f"{class_name} ({confidence:.2f})"

                    # Dibuja el rectángulo y el centro
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(annotated, nombre_clase, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # --- Lógica de cruce de líneas ---
                    if cx > self.line2_x and not self.ebilla_detected:
                        self.ebilla_detected = True

                        if roi.size > 0:
                            # Mostrar ROI en la GUI
                            self.detected_object_frame = roi
                            self.last_detection_label = nombre_clase
                        

                            #  Ejecutar análisis de defectos con tu función
                            es_defectuosa = self._analizar_defectos(class_name,frame, roi)

                            # Actualizar etiqueta en GUI
                            # Actualizar etiqueta en GUI
                            estado = "Defectuosa" if es_defectuosa else "OK"
                            self.last_detection_label = f"{class_name} ({estado})"

                            # Actualizar contadores
                            class_map = {
                                "argolla": "Argollas",
                                "tensor": "Tensores",
                                "zeta": "Zetas",
                                "zeta redondeada": "Zetas redondeadas"
                            }

                            # Determinar la clave de familia
                            key = class_map.get(class_name.lower(), class_name.capitalize())

                            # --- Nueva lógica de conteo ---
                            if es_defectuosa:
                                defect_key = "Piezas defectuosas"
                                self.object_counts[defect_key] = self.object_counts.get(defect_key, 0) + 1
                                print(f"[{self.name}] Pieza defectuosa detectada → Total: {self.object_counts[defect_key]}")
                            else:
                                    
                                # Clasificar tamaño
                                tamano = self._clasificar_tamano(class_name, frame, roi)

                                # Inicializar estructuras si no existen
                                if not hasattr(self, "size_counts"):
                                    self.size_counts = {}

                                if key not in self.size_counts:
                                    self.size_counts[key] = {"S": 0, "M": 0, "L": 0, "XL": 0}

                                # Incrementar contador de ese tamaño
                                if tamano in self.size_counts[key]:
                                    self.size_counts[key][tamano] += 1
                                    print(f"[{self.name}] {key} tamaño {tamano} → Total: {self.size_counts[key][tamano]}")

                                # También contar pieza total por familia
                                self.object_counts[key] = self.object_counts.get(key, 0) + 1
                            
                            self.object_counter += 1
                            print(f"[{self.name}] Objeto #{self.object_counter} contado")

                    elif cx < self.line1_x and self.ebilla_detected:
                        self.ebilla_detected = False

                # Actualizar frame anotado
                self.annotated_frame = annotated.copy()

            except Exception as e:
                print(f"[{self.name}] Error en modo familias: {e}")
                import traceback
                traceback.print_exc()
                self.annotated_frame = frame.copy()
            

        else:

            self.annotated_frame = annotated.copy()
            self.detected_object_frame = None
            self.last_detection_label = None
    
    # ================================================================
    # Analizar defectos
    # ================================================================

    # esta funcion de agrego para elimianar los contornos que estaban solapados y que causaban
    # un error en la deteccion de defectos en la pieza
    def filtrar_contornos_solapados(self,contours, area_tol=0.1, overlap_thresh=0.8):
        """
        Elimina contornos solapados o duplicados.
        - area_tol: tolerancia relativa de área (0.1 = ±10%)
        - overlap_thresh: porcentaje mínimo de solapamiento (0.8 = 80%)
        """
        filtrados = []
        usados = set()

        for i, c1 in enumerate(contours):
            if i in usados:
                continue

            x1, y1, w1, h1 = cv2.boundingRect(c1)
            area1 = w1 * h1
            rect1 = (x1, y1, x1 + w1, y1 + h1)
            similares = [i]

            for j, c2 in enumerate(contours):
                if j <= i or j in usados:
                    continue

                x2, y2, w2, h2 = cv2.boundingRect(c2)
                area2 = w2 * h2

                # Solapamiento entre rectángulos
                inter_x1 = max(x1, x2)
                inter_y1 = max(y1, y2)
                inter_x2 = min(x1 + w1, x2 + w2)
                inter_y2 = min(y1 + h1, y2 + h2)

                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                overlap = inter_area / float(min(area1, area2))

                # Si tienen área parecida y se solapan bastante, se consideran duplicados
                if abs(area1 - area2) / area1 < area_tol and overlap > overlap_thresh:
                    similares.append(j)
                    usados.add(j)

            # De los contornos similares, conservamos el más grande
            c_final = max([contours[k] for k in similares], key=cv2.contourArea)
            filtrados.append(c_final)
            usados.update(similares)

        return filtrados


    def _analizar_defectos(self, familia, frame, roi):
        """Detecta imperfecciones (poros, roturas) dentro del ROI binarizando la imagen."""
        try:
            # --- Preprocesamiento ---
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2) # revisar esto 

            # --- Encontrar contornos ---
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0 or hierarchy is None:
                return True  # sin contornos → defectuosa

            annotated = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

            # --- Filtro de contornos solapados ---
            contours = self.filtrar_contornos_solapados(contours)

            # --- Recalcular áreas y contorno principal ---
            areas = [cv2.contourArea(c) for c in contours]
            max_idx = int(np.argmax(areas))
            cont_principal = contours[max_idx]

            # --- Contornos hijos y externos ---
            hijos = [i for i, h in enumerate(hierarchy[0]) if h[3] == max_idx]
            contornos_externos = [i for i, h in enumerate(hierarchy[0]) if h[3] == -1]

            # --- Filtro de área mínima ---
            min_area = 20
            hijos_filtrados = [i for i in hijos if cv2.contourArea(contours[i]) > min_area]

            # --- Dibujar contornos visibles ---
            # for idx in contornos_externos:
            #     cv2.drawContours(annotated, [contours[idx]], -1, (255, 0, 0), 2)
            #     M = cv2.moments(contours[idx])
            #     if M["m00"] != 0:
            #         cx = int(M["m10"] / M["m00"])
            #         cy = int(M["m01"] / M["m00"])
            #         cv2.putText(annotated, f"E{idx}", (cx - 10, cy - 10),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # for idx in hijos_filtrados:
            #     cv2.drawContours(annotated, [contours[idx]], -1, (0, 0, 255), 2)

            # --- Diagnóstico ---
            print(f"Externos: {len(contornos_externos)} | Hijos filtrados: {len(hijos_filtrados)}")

            # --- Clasificación por familia ---
            fam = familia.lower()
            defectuosa = False

            if "tensor" in fam:
                defectuosa = not (len(hijos_filtrados) == 2)

            elif "zeta" in fam:
                defectuosa = not (len(hijos_filtrados) == 1)

            elif "argolla" in fam:
                defectuosa = not (len(hijos_filtrados) == 1)

            elif "zetaRedonda" in fam:
                defectuosa = not (len(hijos_filtrados) == 1)

            # cv2.imshow("Análisis de defectos", annotated)
            cv2.waitKey(1)

            return defectuosa

        except Exception:
            return "Desconocido", False
        

    def _clasificar_tamano(self, familia, frame, roi):
        """Clasifica el tamaño de una pieza según su área y proporciones."""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                print(f"[{self.name}] ⚠️ Sin contornos detectados en ROI ({familia})")
                return "Desconocido"

            # Contorno principal
            max_contour = max(contours, key=cv2.contourArea)
            area_contorno = cv2.contourArea(max_contour)
            # Obtener el rectángulo mínimo rotado
            rect = cv2.minAreaRect(max_contour)
            (cx, cy), (w_rot, h_rot), angle = rect
            largo = max(w_rot, h_rot)
            ancho = min(w_rot, h_rot)
            aspect_ratio = largo / ancho if ancho > 0 else 0

            # if w_rot > 1 and h_rot > 1:
            #     box = cv2.boxPoints(rect)
            #     box = box.astype(int)   
            #     debug_img = roi.copy()
            #     cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 2)
            #     cv2.imshow("Rectángulo rotado", debug_img)
            #     cv2.waitKey(1)
            # else:
            #     print(f"[{self.name}] ⚠️ Rectángulo inválido ({familia}): w={w_rot:.1f}, h={h_rot:.1f}")

            # Área total del ROI
            area_roi = roi.shape[0] * roi.shape[1]
            proporcion = area_contorno / float(area_roi)

            # === DEBUG: impresión para calibrar ===
            print("────────────────────────────────────────────")
            # print(f"[{self.name}] Familia: {familia}")
            print(f"• Área del contorno principal: {area_contorno:.1f} píxeles²")
            # print(f"• Área total del ROI: {area_roi} píxeles²")
            # print(f"• Proporción contorno/ROI: {proporcion:.3f}")
            # print(f"• Largo (rotado): {largo:.1f} px | Ancho: {ancho:.1f} px")
            # print(f"• Aspect ratio (ancho/alto): {aspect_ratio:.3f}")
            print("────────────────────────────────────────────")

            # --- Clasificación temporal por área ---
            # Ajusta estos valores después de recolectar datos reales de cada familia


            if "tensor" == familia:

                if area_contorno < 60000:
                   size = "S"
                elif 60000 <= area_contorno < 95000:
                    size = "M"
                elif 95000 <= area_contorno < 150000:
                    size = "L"
                else:
                    size = "XL"
        
            elif "zeta" == familia:

                if area_contorno < 55000:
                   size = "S"
                elif 55000 <= area_contorno < 80000:
                    size = "M"
                elif 80000 <= area_contorno < 120000:
                    size = "L"
                else:
                    size = "XL"

            elif "argolla" == familia:
                if area_contorno < 55000:
                   size = "S"
                elif 55000 <= area_contorno < 80000:
                    size = "M"
                elif 80000 <= area_contorno < 120000:
                    size = "L"
                else:
                    size = "XL"

            elif "zetaRedonda" == familia:

                if area_contorno < 55000:
                   size = "S"
                elif 55000 <= area_contorno < 80000:
                    size = "M"
                elif 80000 <= area_contorno < 120000:
                    size = "L"
                else:
                    size = "XL"
           
            # print(f"[{self.name}] → Tamaño clasificado: {familia} → {size}\n")
            return size

        except Exception as e:
            print(f"[{self.name}] Error en _clasificar_tamano: {e}")
            import traceback
            traceback.print_exc()
            return "Desconocido"


    # ================================================================
    # FINALIZACIÓN Y LIMPIEZA
    # ================================================================
    def stop(self):
        """Detiene la captura de video."""
        self._stop_event.set()
        self.running = False
        print(f"[{self.name}] Captura detenida.")

    def _cleanup(self):
        """Libera recursos de la cámara."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print(f"[{self.name}] Recursos liberados correctamente.")

