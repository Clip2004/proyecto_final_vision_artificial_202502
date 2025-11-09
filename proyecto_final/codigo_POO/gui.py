# Guarda este archivo como gui.py
from logger import Logger
import tkinter as tk
from PIL import Image, ImageTk
import tkinter.font as font
from tkinter import ttk
import cv2
#  import camera
import camera2
import numpy as np

class Application(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.logReport = Logger("logGui")
        self.logReport.logger.info("init GUI")
        
        self.master = master
        self.width = 1920
        self.height = 1080
        self.master.geometry("%dx%d" % (self.width, self.height))
        
        self.is_camera_running = False
        
        self.createWidgets()
        self.createFrames()
        self.modo_familias = False
        self.modo_tamanos = False
        self.modo_mixto = False

                
        self.master.mainloop()

    def createFrames(self):
        # Frame para el video principal
        self.labelVideo_1 = tk.Label(self.master, borderwidth=2, relief="solid")
        self.labelVideo_1.place(x=10, y=50)
        
        # Frame para la placa detectada
        self.labelVideo_detection = tk.Label(self.master, borderwidth=2, relief="solid")
        self.labelVideo_detection.place(x=825, y=50)

        # Crear imágenes negras iniciales
        self.imgTk_placeholder = self.createImagePlaceholder(800, 600)
        self.labelVideo_1.configure(image=self.imgTk_placeholder)
        self.labelVideo_1.image = self.imgTk_placeholder

        self.imgTk_detection_placeholder = self.createImagePlaceholder(400, 400)
        self.labelVideo_detection.configure(image=self.imgTk_detection_placeholder)
        self.labelVideo_detection.image = self.imgTk_detection_placeholder
        
    def createImagePlaceholder(self, width, height):
        frame = np.zeros([height, width, 3], dtype=np.uint8)
        img_array = Image.fromarray(frame)
        return ImageTk.PhotoImage(image=img_array)
    def resize_with_aspect_ratio(self, image, max_width, max_height):
        """Redimensiona imagen manteniendo relación de aspecto"""
        h, w = image.shape[:2]
        
        # Calcular escalas
        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h)  # Usar la escala menor
        
        # Nuevas dimensiones
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Redimensionar
        resized = cv2.resize(image, (new_w, new_h))
        
        # Crear imagen de fondo negro del tamaño objetivo
        background = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        
        # Centrar la imagen redimensionada
        start_y = (max_height - new_h) // 2
        start_x = (max_width - new_w) // 2
        background[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return background
    
    def createWidgets(self):

   
        # Fuentes y etiquetas de texto
        self.fontLabelText = font.Font(family='Helvetica', size=12, weight='bold')
        self.fontCounters = font.Font(family='Helvetica', size=16, weight='bold')
        
        tk.Label(self.master, text="Cámara Principal", font=self.fontLabelText).place(x=10, y=20)
        tk.Label(self.master, text="Herraje Detectado", font=self.fontLabelText).place(x=825, y=20)

        # Variables para mostrar información Labels estaticos 
        self.plate_count_var = tk.StringVar(value="Herrajes detectados: 0")
        self.detection_label_var = tk.StringVar(value="Última Detección: --")

        self.defective_var = tk.StringVar(value="Defectuosas detectadas: 0")


        tk.Label(self.master, textvariable=self.plate_count_var, font=self.fontCounters, fg="blue").place(x=825, y=510)
        tk.Label(self.master, textvariable=self.detection_label_var, font=self.fontCounters).place(x=825, y=480)

        # === Inicializar contenedores de labels/botones dinámicos ===
        self.dynamic_labels = []
        self.dynamic_buttons = []
        self.family_labels = {}     # {"Argollas": StringVar, "Tensores": StringVar, ...}
        self.mixed_labels = {}      # {"Argollas": {"S": StringVar, ...}, ...}  



        # === Botones ===
        self.btnInitCamera = tk.Button(self.master, text='Iniciar Detección', font=self.fontLabelText, bg="#06542a", fg='#ffffff', width=22,height=4, command=self.initCamera)
        self.btnInitCamera.place(x=50, y=670)

        self.btnStopCamera = tk.Button(self.master, text='Parar Detección', font=self.fontLabelText, bg="#511610", fg='#ffffff', width=22,height=4, command=self.stopCamera, state=tk.DISABLED)
        self.btnStopCamera.place(x=300, y=670)
        
        self.btnFamilias = tk.Button(self.master, text='Familias', font=self.fontLabelText, bg="#3786e0", fg='#ffffff', width=25, height=3,command=self.showFamilias)
        self.btnFamilias.place(x=1250, y=125)


        self.btnMixto = tk.Button(self.master, text='Mixto', font=self.fontLabelText, bg="#3786e0", fg='#ffffff', width=25, height=3,command=self.showMixto)
        self.btnMixto.place(x=1250, y=200)

    def initCamera(self):
        self.logReport.logger.info("Iniciando detección de placas...")
        
        # Cambiar esta ruta por tu video
        video_path = r'proyecto_final\video_piezas.mp4'
        self.camera_1 = camera2.RunCamera(src=video_path, name="Detector de Placas")
        self.camera_1.start()
        self.is_camera_running = True
        self.camera_1.modo_familias = self.modo_familias  # sincroniza el estado
        self.camera_1.modo_mixto = self.modo_mixto  # sincroniza el estado
        self.btnInitCamera.config(state=tk.DISABLED)
        self.btnStopCamera.config(state=tk.NORMAL)
        
        self.showVideo()

    def showVideo(self):
        if not self.is_camera_running:
            return

        # Actualizar el frame de la cámara principal
        if self.camera_1.annotated_frame is not None:
            frame_resized = cv2.resize(self.camera_1.annotated_frame, (800, 600))
            imgTk = self.convertToFrameTk(frame_resized)
            self.labelVideo_1.configure(image=imgTk)
            self.labelVideo_1.image = imgTk

        # Actualizar el frame de la placa detectada
        if self.camera_1.detected_object_frame is not None:
            obj_frame_resized = cv2.resize(self.camera_1.detected_object_frame, (400, 400))
            imgTk_obj = self.convertToFrameTk(obj_frame_resized)
            self.labelVideo_detection.configure(image=imgTk_obj)
            self.labelVideo_detection.image = imgTk_obj
        else:
            self.labelVideo_detection.configure(image=self.imgTk_detection_placeholder)
            self.labelVideo_detection.image = self.imgTk_detection_placeholder

        # Actualizar contadores y etiquetas
        self.plate_count_var.set(f"Errajes únicos: {self.camera_1.object_counter}")
        if self.camera_1.last_detection_label:
            self.detection_label_var.set(f"Última: {self.camera_1.last_detection_label}")
        else:
            self.detection_label_var.set("Última Detección: --")

        # --- AGREGAR PARA DEBUG (opcional) ---
     # --- Actualizar contadores dinámicos por familia ---
        if hasattr(self.camera_1, 'object_counts') and self.modo_familias:
            for familia, count in self.camera_1.object_counts.items():
                if familia in self.family_labels:
                    self.family_labels[familia].set(f"{familia} detectadas: {count}")

        # --- Actualizar contadores mixtos (familia + tamaño) ---
        if hasattr(self.camera_1, 'size_counts') and self.modo_mixto:
            for fam, sizes in self.camera_1.size_counts.items():
                if fam in self.mixed_labels:
                    for size, count in sizes.items():
                        if size in self.mixed_labels[fam]:
                            self.mixed_labels[fam][size].set(f"{size}: {count}")
            
        if hasattr(self.camera_1, 'object_counts') and "Piezas defectuosas" in self.camera_1.object_counts:
            count_def = self.camera_1.object_counts["Piezas defectuosas"]
            if hasattr(self, 'defective_var'):
                self.defective_var.set(f"Defectuosas detectadas: {count_def}")

        self.labelVideo_1.after(30, self.showVideo)

    def convertToFrameTk(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_array = Image.fromarray(frame_rgb)
        return ImageTk.PhotoImage(image=img_array)

    def stopCamera(self):
        self.logReport.logger.info("Deteniendo detección...")
        if hasattr(self, 'camera_1') and self.camera_1:
            self.camera_1.stop()

        self.is_camera_running = False
        self.btnInitCamera.config(state=tk.NORMAL)
        self.btnStopCamera.config(state=tk.DISABLED)

        self.labelVideo_1.configure(image=self.imgTk_placeholder)
        self.labelVideo_1.image = self.imgTk_placeholder
        self.labelVideo_detection.configure(image=self.imgTk_detection_placeholder)
        self.labelVideo_detection.image = self.imgTk_detection_placeholder


    def clearDynamicElements(self):
        """Elimina los labels o botones dinámicos que estén visibles."""
        for widget in self.dynamic_labels + self.dynamic_buttons:
            widget.destroy()
        self.dynamic_labels.clear()
        self.dynamic_buttons.clear()

    def showFamilias(self):
        """Muestra los contadores por tipo de herraje (familias principales)."""
        self.modo_familias = True
        self.modo_mixto = False
        self.camera_1.modo_familias = True
        self.camera_1.modo_mixto = False
        self.clearDynamicElements()
        self.family_labels.clear()

        familias = ["Argollas", "Tensores", "Zetas", "Zetaredonda", "Piezas defectuosas"]
        start_x, start_y, spacing_y = 825, 550, 35

        for i, fam in enumerate(familias):
            # Crear variable dinámica
            var = tk.StringVar(value=f"{fam} detectadas: 0")
            self.family_labels[fam] = var

            lbl = tk.Label(self.master, textvariable=var, font=self.fontCounters,
                        fg="red" if "defectuosas" in fam.lower() else "black")
            lbl.place(x=start_x, y=start_y + i * spacing_y)
            self.dynamic_labels.append(lbl)
   

    def showMixto(self):
        """Muestra los contadores de familias y tamaños en dos columnas."""
        self.modo_familias = False
        self.modo_mixto = True
        self.camera_1.modo_familias = False
        self.camera_1.modo_mixto = True
        self.clearDynamicElements()
        self.mixed_labels.clear()

        familias_left = ["Argollas", "Tensores"]
        familias_right = ["Zetas", "Zetaredonda"]
        tamanos = ["S", "M", "L", "XL"]

        # --- Posiciones ---
        start_x_left, start_x_right, start_x_middle = 825, 1200, 1000
        start_y, spacing_y, spacing_x, row_gap = 550, 60, 60, 100

        # --- Columna izquierda ---
        for i, fam in enumerate(familias_left):
            y_offset = start_y + i * row_gap
            self.mixed_labels[fam] = {}

            lbl_title = tk.Label(self.master, text=f"{fam} detectadas:", font=self.fontCounters, fg="black")
            lbl_title.place(x=start_x_left, y=y_offset)
            self.dynamic_labels.append(lbl_title)

            for j, t in enumerate(tamanos):
                var = tk.StringVar(value=f"{t}: 0")
                self.mixed_labels[fam][t] = var
                lbl_size = tk.Label(self.master, textvariable=var, font=self.fontLabelText)
                lbl_size.place(x=start_x_left + j * spacing_x + 40, y=y_offset + 30)
                self.dynamic_labels.append(lbl_size)

        # --- Columna derecha ---
        for i, fam in enumerate(familias_right):
            y_offset = start_y + i * row_gap
            self.mixed_labels[fam] = {}

            lbl_title = tk.Label(self.master, text=f"{fam} detectadas:", font=self.fontCounters, fg="black")
            lbl_title.place(x=start_x_right, y=y_offset)
            self.dynamic_labels.append(lbl_title)

            for j, t in enumerate(tamanos):
                var = tk.StringVar(value=f"{t}: 0")
                self.mixed_labels[fam][t] = var
                lbl_size = tk.Label(self.master, textvariable=var, font=self.fontLabelText)
                lbl_size.place(x=start_x_right + j * spacing_x + 40, y=y_offset + 30)
                self.dynamic_labels.append(lbl_size)

        # --- Defectuosas ---

        self.defective_var = tk.StringVar(value="Defectuosas detectadas: 0")
        lbl_def = tk.Label(self.master, textvariable=self.defective_var, font=self.fontCounters, fg="red")
        lbl_def.place(x=start_x_middle, y=start_y + 2 * row_gap)
        self.dynamic_labels.append(lbl_def)


            
def main():
    root = tk.Tk()
    root.title("Detector de Placas - GUI")
    appRunCamera = Application(master=root)

if __name__ == '__main__':
    main()