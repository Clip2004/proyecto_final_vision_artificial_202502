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
    """GUI minimalista para clasificación de herrajes con YOLO."""
    
    def __init__(self, width=1200, height=720):
        self.width = width
        self.height = height

        self.root = tk.Tk()
        self.root.title("Clasificador herrajes - Simple View")
        self.root.geometry(f"{self.width}x{self.height}")

        self.camera = None
        self.is_running = False

        self.create_widgets()
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
        self.btn_start = tk.Button(self.root, text="Iniciar Detección", command=self.start_camera, 
                                   width=20, bg="#06542a", fg="white")
        self.btn_start.place(x=830, y=430)
        
        self.btn_stop = tk.Button(self.root, text="Parar Detección", command=self.stop_camera, 
                                  width=20, bg="#511610", fg="white", state=tk.DISABLED)
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
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__)))
            from camera1 import Camera1Predictor
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(script_dir)
            
            model_path = os.path.join(base_dir, "yolo", "best.pt")
            video_path = os.path.join(base_dir, "video_piezas.mp4")
            
            self.camera = Camera1Predictor(model_path)
            self.is_running = True
            
            def run():
                try:
                    self.camera.run_camera(video_path)
                except Exception as e:
                    print(f"Error en cámara: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self._update_loop()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def stop_camera(self):
        if not self.is_running:
            return
        
        if self.camera:
            self.camera.stop()
        
        self.is_running = False
        self.camera = None
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

        self.video_label.configure(image=self.placeholder_video)
        self.video_label.image = self.placeholder_video
        self.detect_label.configure(image=self.placeholder_detect)
        self.detect_label.image = self.placeholder_detect
        self.pred_var.set("Predicción: --")

    def _update_loop(self):
        if not self.is_running or self.camera is None:
            return

        # Update main video
        annotated = getattr(self.camera, "annotated_frame", None)
        if annotated is not None:
            try:
                h, w = annotated.shape[:2]
                scale = min(800 / w, 600 / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(annotated, (new_w, new_h))
                
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

        # Update detected ROI
        obj = getattr(self.camera, "detected_object_frame", None)
        if obj is not None:
            try:
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
            self.detect_label.configure(image=self.placeholder_detect)
            self.detect_label.image = self.placeholder_detect

        # Prediction text
        pred = getattr(self.camera, "last_detection_label", None)
        if pred:
            self.pred_var.set(f"Predicción: {pred}")
        else:
            self.pred_var.set("Predicción: --")

        self.root.after(50, self._update_loop)

    def on_close(self):
        if self.camera:
            self.camera.stop()
            time.sleep(0.1)
        self.root.destroy()


def main():
    return SimpleGUI()

if __name__ == "__main__":
    main()