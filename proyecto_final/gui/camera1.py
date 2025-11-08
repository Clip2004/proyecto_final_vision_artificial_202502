import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time

class Camera1Predictor:
    """
    Predictor usando YOLO para detección y clasificación de herrajes.
    Compatible con la GUI existente.
    """
    def __init__(self, model_path):
        """
        Args:
            model_path: Ruta al modelo YOLO (.pt)
        """
        try:
            self.model = YOLO(model_path)
            print(f"Modelo YOLO cargado: {model_path}")
        except Exception as e:
            print(f"Error cargando modelo YOLO: {str(e)}")
            raise
        
        # Atributos compartidos con la GUI
        self.annotated_frame = None
        self.detected_object_frame = None
        self.last_detection_label = None
        
        # Control de threads
        self.running = False
        self.cap = None
        
        # Configuración de detección
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

    def run_camera(self, video_source=0):
        """
        Inicia la captura de video y predicción.
        
        Args:
            video_source: 0 para webcam, o ruta a archivo de video
        """
        self.running = True
        
        try:
            self.cap = cv2.VideoCapture(video_source)
            
            if not self.cap.isOpened():
                print(f"Error: No se pudo abrir la fuente de video {video_source}")
                return
            
            print("Cámara iniciada correctamente")
            
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("No se pudo leer el frame")
                    if isinstance(video_source, str):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                
                self._process_frame(frame)
                time.sleep(0.01)
        
        except Exception as e:
            print(f"Error en run_camera: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._cleanup()

    def _process_frame(self, frame):
        """
        Procesa un frame con YOLO y actualiza los atributos para la GUI.
        """
        try:
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            result = results[0]
            annotated = result.plot()
            self.annotated_frame = annotated.copy()
            
            if len(result.boxes) > 0:
                best_idx = result.boxes.conf.argmax()
                box = result.boxes[best_idx]
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2].copy()
                
                if roi.size > 0:
                    self.detected_object_frame = roi
                else:
                    self.detected_object_frame = None
                
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                
                self.last_detection_label = f"{class_name} ({confidence:.2f})"
            else:
                self.detected_object_frame = None
                self.last_detection_label = None
        
        except Exception as e:
            print(f"Error procesando frame: {str(e)}")
            self.annotated_frame = frame.copy()
            self.detected_object_frame = None
            self.last_detection_label = None

    def stop(self):
        """Detiene la captura de video."""
        print("Deteniendo cámara...")
        self.running = False
        time.sleep(0.1)

    def _cleanup(self):
        """Libera recursos."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("Recursos liberados")


if __name__ == "__main__":
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    model_path = os.path.join(base_dir, "yolo", "best.pt")
    video_path = os.path.join(base_dir, "video_piezas.mp4")
    
    print(f"Usando modelo: {model_path}")
    
    predictor = Camera1Predictor(model_path)
    thread = threading.Thread(target=lambda: predictor.run_camera(video_path), daemon=True)
    thread.start()
    
    try:
        while True:
            if predictor.annotated_frame is not None:
                cv2.imshow("Detección YOLO", predictor.annotated_frame)
            
            if predictor.detected_object_frame is not None:
                cv2.imshow("Objeto Detectado", predictor.detected_object_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.03)
    finally:
        predictor.stop()
        cv2.destroyAllWindows()