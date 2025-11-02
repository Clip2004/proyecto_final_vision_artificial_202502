import threading
import time
import cv2
import numpy as np
import os
import joblib

ruta_ejecutable = os.getcwd()


class RunCamera(threading.Thread):
    def __init__(self, src=0, name="Camera"):
        threading.Thread.__init__(self)
        self.src = src
        self.name = name
        self.daemon = True
        
        
        # Variables para la GUI
        self.annotated_frame = None
        self.detected_object_frame = None
        self.last_detection_label = None
        self.plate_count = 0

        self.detected_plates = set()  # Placas confirmadas (aparecieron +3 veces)
        self.frame_count = 0
        self.process_every_n_frames = 5
        
        self.plate_candidates = {}  # Diccionario: {placa_text: contador}
        self.min_detections = 4     # Mínimo de detecciones para considerar válida
        self.candidate_timeout = 100  # Frames después de los cuales se limpia un candidato
        self.last_seen_candidates = {}  # {placa_text: frame_number}

        self.line1_x = 850  # línea izquierda
        self.line2_x = 900  # línea derecha
        self.ebilla_detected = False
        self.object_counter = 0
        self.last_centroid = None
        self.min_area_threshold = 800  # ajustable

        

        self.modo_familias = False
        self.modo_tamanos = False
        self.modo_mixto = False

        self.argolla_model = None
        self.tensor_model = None
        self.zetas_model = None
        self.ochos_model = None

        self.familias_model = None
        self.number_model = None    
        
        # print(f"✅ Variables de modelos inicializadas")
        
        # Control de thread
        self._stop_event = threading.Event()
        
       
        # Cargar modelos de clasificación
        print(f"Cargando modelos de clasificación...")
        self.load_classification_models()
        
        print(f" RunCamera inicializado completamente - ID: {id(self)}")
    
    def load_classification_models(self):
        """Carga los modelos .pkl"""
        try:
            
            
            # Rutas de los modelos en la carpeta "modelos"
            base_path ='proyecto_final/train_familias'
            
            print(f" Buscando modelos en: {base_path}")
            
            # Rutas de los modelos de TODAS LAS CARACTERÍSTICAS
            familias_model_path = os.path.join(base_path, 'model_familias_100.pkl')
            familias_model_scaler_path = os.path.join(base_path, 'scaler_familias_100.pkl')
            familias_model_mlp_path = os.path.join(base_path, 'best_model_1_fa.pkl')
           
            
            # Verificar que los archivos existen
            required_files = [familias_model_path,familias_model_scaler_path,familias_model_mlp_path]
            
            missing_files = [f for f in required_files if not os.path.exists(f)]
            if missing_files:
                print(f" Archivos faltantes:")
                for f in missing_files:
                    print(f"   - {f}")
                return False
            
            # Cargar modelo de familias
            print(" Cargando modelo de FAMILIAS...")
            self.familias_model = joblib.load(familias_model_path)
            self.familias_model_scaler = joblib.load(familias_model_scaler_path)
            self.familias_model_mlp = joblib.load(familias_model_mlp_path)

            print(f" Familias: Cargado - ID: {id(self)}")
            
     
        
            # VERIFICACIÓN INMEDIATA
            print(f" VERIFICACIÓN POST-CARGA:")
            print(f"   familias_model: {type(self.familias_model) if self.familias_model else 'None'}")
         

            print(" Modelos cargados exitosamente!")
            return True
            
        except Exception as e:
            print(f" Error cargando modelos: {e}")
            import traceback
            traceback.print_exc()
            
            # Inicializar como None para evitar errores
            self.familias_model = None
     
            return False
        

    """
    Extrae los features de clasificación de familias usando
    el mismo set de características que el entrenamiento original.
    
    Parámetros:
        img_roi_bin : ROI ya binarizada y redimensionada (numpy uint8)
        contour     : contorno principal del objeto en el ROI
    
    Retorna:
        vector numpy con 17 features (float32)
    """

    def extract_features_familias(self, img_roi_bin, contour):
     
        try:

            # Contorno principal (ya viene como contour)
            x1_area = cv2.contourArea(contour)
            x2_area = cv2.contourArea(cv2.convexHull(contour))
            x3_perimeter = cv2.arcLength(contour, True)
            x4_circle = x1_area * 4 * np.pi / (x3_perimeter**2 + 1e-6)

            # Hu Moments
            M = cv2.moments(contour)
            Hu = cv2.HuMoments(M)
            x5, x6, x7, x8, x9, x10, x11 = [Hu[i][0] for i in range(7)]

            roi_resized = cv2.resize(img_roi_bin, (128, 128), interpolation=cv2.INTER_NEAREST)
            # Densidad 1/4 altura
            H = img_roi_bin.shape[0]
            x12 = np.sum(img_roi_bin[H//4, :] / 255) / (x1_area + 1e-6)

            # Aspect ratio bounding box del contorno
            x, y, w, h = cv2.boundingRect(contour)
            x13 = w / (h + 1e-6)
            x14 = h / (w + 1e-6)

            # Contornos internos en la ROI
            conts_all, hierarchy2 = cv2.findContours(roi_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Simetría vertical y horizontal
            x15 = np.mean(roi_resized == np.flipud(roi_resized))
            x16 = np.mean(roi_resized == np.fliplr(roi_resized))

            # Área relativa de huecos internos
            num_interior = 0
            area_ratio_interior = 0.0

            if hierarchy2 is not None:
                hierarchy2 = hierarchy2[0]  # aplanar
                for i in range(len(hierarchy2)):
                    if hierarchy2[i][3] != -1:  # tiene padre → contorno interno
                        num_interior += 1
                        area_ratio_interior += cv2.contourArea(conts_all[i]) / (x1_area + 1e-6)

            x17 = area_ratio_interior / num_interior if num_interior > 0 else 0

            # Vector final (MISMO ORDEN QUE EN ENTRENAMIENTO)
            all_features = np.array([
                x1_area, 
                x2_area,
                x3_perimeter, 
                x4_circle,
                x5, 
                x6, 
                x7, 
                x8, 
                x9, 
                x10, 
                x11,
                x12, 
                x13, 
                x14, 
                x15, 
                x16, 
                x17
            ])
            return all_features.astype(np.float32)
            
        except Exception as e:
            print(f"Error extrayendo características mejoradas: {e}")
            return np.zeros(17).astype(np.float32)  # Ajustar según el número real
        
    def extract_features_tamanos(self, familia, img_roi_bin, contour):



        return print( " falta esta parte")


    def predict_character_familias(self, char_image, cnt):
        """Predice un caracter usando el modelo"""
        try:
           
            # Usar las características EXACTAS del entrenamiento
            if self.modo_familias:
                # Para letras: usar características mejoradas
                features = self.extract_features_familias(char_image, cnt)

            if self.modo_mixto:

                features = self.extract_features_familias(char_image, cnt)

            else:

                return None

            # if self.modo_tamanos:

            #     features = self.extract_features_tamanos(familia, char_image, cnt)
      
            # Verificar que las características no estén vacías
            if features is None or len(features) == 0:
                print(" No se pudieron extraer características")
                return None
            
            # Convertir a formato correcto para predicción
            features = features.reshape(1, -1)
            
            # ----- MODO FAMILIAS → modelo familia -----
            if self.modo_familias:

                # features = self.familias_model_scaler.transform(features)
                prediction = self.familias_model_mlp.predict(features)[0]  # modelo familias
                clases = {0:"ARGOLLA", 1:"TENSOR", 2:"ZETA"}

                print("Features:", features[0][:5])  # solo los primeros 5 para ver
                print("Pred:", prediction)
                return clases.get(int(prediction), None)
            
            if self.modo_mixto:

                features = self.familias_model_scaler.transform(features)
                prediction = self.familias_model.predict(features)[0]  # modelo familias
                clases = {0:"ARGOLLA", 1:"TENSOR", 2:"ZETA"}
                return clases.get(int(prediction), None)
                    
                
        except Exception as e:
            print(f"Error en predicción : {e}")
            import traceback
            traceback.print_exc()
            return "?"
    
    


    def process_frame(self, frame):

        print("\n=== PROCESS FRAME START ===")

        if self.familias_model is None:
            print("⚠️ Modelo de familias no cargado")
            self.annotated_frame = frame.copy()
            return
        
        # --- AGREGAR CONTROL DE FRAMES ---
        self.frame_count += 1
        print(f"Frame count: {self.frame_count}")

        if self.frame_count % self.process_every_n_frames != 0:
            print(f"⏭ Saltando frame para optimizar ({self.frame_count})")
            self.annotated_frame = frame.copy()
            return
        
        print("✅ Frame procesado")

        self.cleanup_old_candidates()

        annotated = frame.copy()

        # 1) CONVERSIÓN A GRIS
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("✅ Convertido a gris")

        blur = cv2.GaussianBlur(gray, (5,5), 0)
        # 2) BINARIZACIÓN
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        print("✅ Aplicado Otsu Thresholding")

        cv2.imshow("Mask OTSU", mask)
        cv2.waitKey(1)

        # 3) BUSCAR CONTORNOS
        conts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print(f"🔍 Contornos detectados: {len(conts)}")


        img_bin_or = mask.copy()

        contours_or, _ = cv2.findContours(img_bin_or, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours_or = [c for c in contours_or if cv2.contourArea(c) > self.min_area_threshold]

        if len(filtered_contours_or) > 0:
            maxcont_or = max(filtered_contours_or, key=cv2.contourArea)

            M_or = cv2.moments(maxcont_or)
            if M_or["m00"] != 0: 
                cx = int(M_or["m10"] / M_or["m00"])
                cy = int(M_or["m01"] / M_or["m00"])

                # Dibujar contorno y centro
                cv2.drawContours(annotated, [maxcont_or], -1, (0, 255, 0), 2)
                cv2.circle(annotated, (cx, cy), 5, (0,0,255), -1)

                # Mostrar info
                cv2.putText(annotated, f"Obj {self.object_counter}", (cx-20, cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                # --- LOGICA DE LINEAS ANTIDUPLICADO ---
                if cx > self.line2_x and not self.ebilla_detected:
                    self.ebilla_detected = True

                elif cx < self.line1_x and self.ebilla_detected:
                    self.object_counter += 1
                    self.ebilla_detected = False
                    print(f"✅ Objeto #{self.object_counter} contado")

        # Dibujar líneas de referencia
        cv2.line(annotated, (self.line1_x, 0), (self.line1_x, frame.shape[0]), (255, 0, 0), 2)
        cv2.line(annotated, (self.line2_x, 0), (self.line2_x, frame.shape[0]), (255, 0, 0), 2)

        detected_any = False

        for cnt in conts:
            area = cv2.contourArea(cnt)
            print(f"   ➜ Contorno área: {area}")

            if area < 2000:   
                print(f"   ❌ Contorno ignorado (área pequeña)")
                continue

            print(f"✅ Contorno válido (área={area})")

            # ROI bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            print(f"📦 ROI coords: x={x}, y={y}, w={w}, h={h}")

            roi = mask[y:y+h, x:x+w]

            cv2.imshow("ROI original", roi)

            if roi.size == 0:
                print("❌ ROI vacío, saltando")
                continue

            roi_resized = cv2.resize(roi, (128,128), interpolation=cv2.INTER_NEAREST)

            print("✅ ROI redimensionado")
            print(self.modo_familias, self.modo_tamanos, self.modo_mixto)
            cv2.imshow("ROI" , roi_resized)
            cv2.waitKey(2)

            # --- EXTRAER FEATURES Y CLASIFICAR ---
            conts_roi, _ = cv2.findContours(roi_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(conts_roi)==0:
                print("⚠️ No contorno dentro del ROI")
                continue

            cnt_roi = max(conts_roi, key=cv2.contourArea)

            nombre_clase = self.predict_character_familias(roi_resized, cnt)

            print(f"🔎 Resultado predict_character_familias: {nombre_clase}")

            if nombre_clase is None:
                print("❌ No se obtuvo clase, continuar")
                continue

            detected_any = True

            # DIBUJAR RESULTADOS
            cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(annotated, nombre_clase, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            self.detected_object_frame = frame[y:y+h, x:x+w].copy()

        if not detected_any:
            print("⚠️ Ningún objeto clasificado en este frame")

        self.annotated_frame = annotated
        print("=== PROCESS FRAME END ===")

    def cleanup_old_candidates(self):
        """Limpia candidatos antiguos que no han sido vistos recientemente"""
        current_frame = self.frame_count
        to_remove = []
        
        for plate_text, last_seen in self.last_seen_candidates.items():
            if current_frame - last_seen > self.candidate_timeout:
                to_remove.append(plate_text)
        
        for plate_text in to_remove:
            if plate_text in self.plate_candidates:
                old_count = self.plate_candidates[plate_text]
                #print(f" Limpiando candidato antiguo: {plate_text} (tenía {old_count} detecciones)")
                del self.plate_candidates[plate_text]
            
            if plate_text in self.last_seen_candidates:
                del self.last_seen_candidates[plate_text]

    def run(self):
        """Método principal del thread"""
        cap = cv2.VideoCapture(self.src)
        
        if not cap.isOpened():
            print(f"Error: No se pudo abrir la fuente de video {self.src}")
            return
        
        print(f"Iniciando procesamiento de video: {self.name}")
        
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Fin del video o error leyendo frame")
                break
            
            # Procesar frame
            self.process_frame(frame)
            
            # Pequeña pausa para no saturar el CPU
            time.sleep(0.03)  # ~30 FPS
        
        cap.release()
        print(f"Cámara {self.name} detenida")


    
    def stop(self):
        """Detiene el thread de la cámara"""
        self._stop_event.set()

