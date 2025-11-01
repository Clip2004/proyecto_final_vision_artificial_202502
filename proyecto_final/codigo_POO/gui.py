# Guarda este archivo como gui.py
from logger import Logger
import tkinter as tk
from PIL import Image, ImageTk
import tkinter.font as font
from tkinter import ttk
import cv2
#  import camera
import camera1
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
        self.plate_count_var = tk.StringVar(value="Herraje detectadas: 0")
        self.detection_label_var = tk.StringVar(value="Última Detección: --")

        tk.Label(self.master, textvariable=self.plate_count_var, font=self.fontCounters, fg="blue").place(x=825, y=510)
        tk.Label(self.master, textvariable=self.detection_label_var, font=self.fontCounters).place(x=825, y=480)

        # === Inicializar contenedores de labels/botones dinámicos ===
        self.dynamic_labels = []
        self.dynamic_buttons = []

        # === Botones ===
        self.btnInitCamera = tk.Button(self.master, text='Iniciar Detección', font=self.fontLabelText, bg="#06542a", fg='#ffffff', width=22,height=4, command=self.initCamera)
        self.btnInitCamera.place(x=50, y=670)

        self.btnStopCamera = tk.Button(self.master, text='Parar Detección', font=self.fontLabelText, bg="#511610", fg='#ffffff', width=22,height=4, command=self.stopCamera, state=tk.DISABLED)
        self.btnStopCamera.place(x=300, y=670)
        
        self.btnFamilias = tk.Button(self.master, text='Familias', font=self.fontLabelText, bg="#3786e0", fg='#ffffff', width=25, height=3,command=self.showFamilias)
        self.btnFamilias.place(x=1250, y=50)

        self.btnTamanos = tk.Button(self.master, text='Tamaños', font=self.fontLabelText, bg="#3786e0", fg='#ffffff', width=25,height=3, command=self.showTamanos)
        self.btnTamanos.place(x=1250, y=125)

        self.btnMixto = tk.Button(self.master, text='Mixto', font=self.fontLabelText, bg="#3786e0", fg='#ffffff', width=25, height=3,command=self.showMixto)
        self.btnMixto.place(x=1250, y=200)

    def initCamera(self):
        self.logReport.logger.info("Iniciando detección de placas...")
        
        # Cambiar esta ruta por tu video
        video_path = r'proyecto_final\video_piezas.mp4'
        self.camera_1 = camera1.RunCamera(src=video_path, name="Detector de Placas")
        self.camera_1.start()
        self.is_camera_running = True
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
        self.plate_count_var.set(f"Placas únicas: {self.camera_1.plate_count}")
        if self.camera_1.last_detection_label:
            self.detection_label_var.set(f"Última: {self.camera_1.last_detection_label}")
        else:
            self.detection_label_var.set("Última Detección: --")

        # --- AGREGAR PARA DEBUG (opcional) ---
        if hasattr(self.camera_1, 'get_detection_stats'):
            stats = self.camera_1.get_detection_stats()
            #if len(stats['detected_plates']) > 1:
                # print(f" Placas detectadas: {stats['detected_plates']}")

        self.labelVideo_1.after(10, self.showVideo)

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
        """Muestra los contadores por tipo de herraje."""
        self.clearDynamicElements()
        labels_texts = [
            "Argollas detectadas: 0",
            "Tensores detectados: 0",
            "Zetas detectadas: 0",
            "Ochos detectados: 0",
            "Piezas defectuosas: 0"
        ]

        start_x = 825
        start_y = 550
        spacing_y = 35

        for i, text in enumerate(labels_texts):
            lbl = tk.Label(self.master, text=text, font=self.fontCounters, fg="black")
            if text == "Piezas defectuosas: 0":
                lbl.config(fg="red")
            lbl.place(x=start_x, y=start_y + i * spacing_y)
            self.dynamic_labels.append(lbl)

    def showTamanos(self):
        """Muestra botones para las categorías de tamaño."""
        self.clearDynamicElements()
        sizes = ["Tamaño S", "Tamaño M", "Tamaño L", "Tamaño XL"]
        start_x = 825
        start_y = 550
        spacing_y = 50

        for i, size in enumerate(sizes):
            btn = tk.Button(self.master, text=size, width=20, bg="#1E70AE", fg="white")
            btn.place(x=start_x, y=start_y + i * spacing_y)
            self.dynamic_buttons.append(btn)

    def showMixto(self):
        """Muestra los contadores de familias y sus tamaños en dos columnas."""
        self.clearDynamicElements()

        familias_left = ["Argollas", "Tensores"]
        familias_right = ["Zetas", "Ochos"]
        tamanos = ["S", "M", "L", "XL"]

        # Posiciones base
        start_x_left = 825
        start_x_right = 1200
        start_x_middle = 1000
        start_y = 550
        spacing_y = 60      # distancia vertical entre bloques
        spacing_x = 60      # separación horizontal entre tamaños
        row_gap = 100       # separación entre grupos de familias

        # --- Columna izquierda ---
        for i, fam in enumerate(familias_left):
            y_offset = start_y + i * row_gap

            # Label principal
            lbl_title = tk.Label(self.master, text=f"{fam} detectadas:", font=self.fontCounters, fg="black")
            lbl_title.place(x=start_x_left, y=y_offset)
            self.dynamic_labels.append(lbl_title)

            # Labels de tamaños (debajo del título)
            for j, t in enumerate(tamanos):
                lbl_size = tk.Label(self.master, text=f"{t}: 0", font=self.fontLabelText)
                lbl_size.place(x=start_x_left + j * spacing_x + 40, y=y_offset + 30)
                self.dynamic_labels.append(lbl_size)

        # --- Columna derecha ---
        for i, fam in enumerate(familias_right):
            y_offset = start_y + i * row_gap

            lbl_title = tk.Label(self.master, text=f"{fam} detectadas:", font=self.fontCounters, fg="black")
            lbl_title.place(x=start_x_right, y=y_offset)
            self.dynamic_labels.append(lbl_title)

            for j, t in enumerate(tamanos):
                lbl_size = tk.Label(self.master, text=f"{t}: 0", font=self.fontLabelText)
                lbl_size.place(x=start_x_right + j * spacing_x + 40, y=y_offset + 30)
                self.dynamic_labels.append(lbl_size)

        # --- Solo "Defectuosas detectadas" sin tamaños ---
        lbl_def = tk.Label(self.master, text="Defectuosas detectadas: 0", font=self.fontCounters, fg="red")
        lbl_def.place(x=start_x_middle, y=start_y + 2 * row_gap)
        self.dynamic_labels.append(lbl_def)


            
def main():
    root = tk.Tk()
    root.title("Detector de Placas - GUI")
    appRunCamera = Application(master=root)

if __name__ == '__main__':
    main()