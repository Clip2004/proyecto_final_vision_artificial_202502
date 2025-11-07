import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import time
import sys

class SimpleGUI:
    """
    GUI minimalista: muestra video original, imagen del contorno detectado y texto con la predicción.
    Controles: Iniciar / Parar.
    """
    def __init__(self, width=1200, height=720):
        self.width = width
        self.height = height

        self.root = tk.Tk()
        self.root.title("Clasificador herrajes - Simple View")
        self.root.geometry(f"{self.width}x{self.height}")

        # Camera thread placeholder (lazy)
        self.camera = None
        self.is_running = False

        # Widgets
        self.create_widgets()

        # Start Tk mainloop
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def create_widgets(self):
        # Video area (original)
        self.video_label = tk.Label(self.root, borderwidth=2, relief="solid")
        self.video_label.place(x=10, y=10, width=800, height=600)

        # Detected contour area (ROI)
        self.detect_label = tk.Label(self.root, borderwidth=2, relief="solid")
        self.detect_label.place(x=830, y=10, width=350, height=350)

        # Prediction text
        self.pred_var = tk.StringVar(value="Predicción: --")
        self.pred_label = tk.Label(self.root, textvariable=self.pred_var, font=("Helvetica", 18))
        self.pred_label.place(x=830, y=380)

        # Start / Stop buttons
        self.btn_start = tk.Button(self.root, text="Iniciar Detección", command=self.start_camera, width=20, bg="#06542a", fg="white")
        self.btn_start.place(x=830, y=430)
        self.btn_stop = tk.Button(self.root, text="Parar Detección", command=self.stop_camera, width=20, bg="#511610", fg="white", state=tk.DISABLED)
        self.btn_stop.place(x=830, y=480)

        # Placeholders
        self.placeholder_video = self._make_placeholder(800, 600)
        self.placeholder_detect = self._make_placeholder(350, 350)
        self.video_label.configure(image=self.placeholder_video)
        self.video_label.image = self.placeholder_video
        self.detect_label.configure(image=self.placeholder_detect)
        self.detect_label.image = self.placeholder_detect

    def _make_placeholder(self, w, h):
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        return ImageTk.PhotoImage(img)

    def start_camera(self):
        # Importar RunCamera dinámicamente
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__)))
            from camera1 import Camera1Predictor
            
            # Construir rutas
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(script_dir)
            
            model_path = os.path.join(base_dir, "train", "models", "family_model2.pkl")
            hsv_config_path = os.path.join(base_dir, "masking", "hsv_config.yaml")
            video_path = os.path.join(base_dir, "video_piezas.mp4")  # <-- CAMBIO: usar video_path
            
            # Crear predictor
            self.camera = Camera1Predictor(model_path, hsv_config_path)
            self.is_running = True
            
            # Ejecutar en un hilo separado para no bloquear la GUI
            def run():
                try:
                    self.camera.run_camera(video_path)  # <-- CAMBIO: pasar video_path
                except Exception as e:
                    print(f"Error en cámara: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            
            # Actualizar botones
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            
            # Iniciar actualización de UI
            self._update_loop()
            
        except ImportError as e:
            messagebox.showerror("Error", f"No se pudo importar Camera1Predictor: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def stop_camera(self):
        if not self.is_running:
            return
        try:
            if self.camera:
                self.camera.stop()
        except Exception:
            pass
        self.is_running = False
        self.camera = None
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

        # reset UI
        self.video_label.configure(image=self.placeholder_video)
        self.video_label.image = self.placeholder_video
        self.detect_label.configure(image=self.placeholder_detect)
        self.detect_label.image = self.placeholder_detect
        self.pred_var.set("Predicción: --")

    def _update_loop(self):
        if not self.is_running or self.camera is None:
            return

        # update main video
        annotated = getattr(self.camera, "annotated_frame", None)
        if annotated is not None:
            try:
                # keep aspect ratio and fit into 800x600 box
                h, w = annotated.shape[:2]
                scale = min(800 / w, 600 / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(annotated, (new_w, new_h))
                # place on black background
                bg = np.zeros((600, 800, 3), dtype=np.uint8)
                sy = (600 - new_h) // 2
                sx = (800 - new_w) // 2
                bg[sy:sy+new_h, sx:sx+new_w] = resized
                img = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                tk_img = ImageTk.PhotoImage(img_pil)
                self.video_label.configure(image=tk_img)
                self.video_label.image = tk_img
            except Exception:
                pass

        # update detected ROI
        obj = getattr(self.camera, "detected_object_frame", None)
        if obj is not None:
            try:
                # obj is expected BGR; fit into 350x350
                h, w = obj.shape[:2]
                scale = min(350 / w, 350 / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(obj, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                bg = np.zeros((350, 350, 3), dtype=np.uint8)
                sy = (350 - new_h) // 2
                sx = (350 - new_w) // 2
                bg[sy:sy+new_h, sx:sx+new_w] = resized
                img = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                tk_img = ImageTk.PhotoImage(img_pil)
                self.detect_label.configure(image=tk_img)
                self.detect_label.image = tk_img
            except Exception:
                pass
        else:
            # keep placeholder if no object
            self.detect_label.configure(image=self.placeholder_detect)
            self.detect_label.image = self.placeholder_detect

        # prediction text
        pred = getattr(self.camera, "last_detection_label", None)
        if pred:
            self.pred_var.set(f"Predicción: {pred}")
        else:
            self.pred_var.set("Predicción: --")

        # schedule next update
        self.root.after(50, self._update_loop)

    def on_close(self):
        # stop camera thread safely
        try:
            if self.camera:
                self.camera.stop()
                # allow thread to finish
                time.sleep(0.1)
        except Exception:
            pass
        self.root.destroy()


def main():
    """Entry point useful when the module is imported (avoids AttributeError)."""
    return SimpleGUI()

if __name__ == "__main__":
    main()