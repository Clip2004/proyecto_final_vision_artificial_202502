import os
import cv2
import sys

# Ajusta si necesitas instalar dependencias:
# pip install torch torchvision opencv-python
# opcional: pip install ultralytics
# o usar yolov5 via torch.hub (necesita internet la primera vez)

MODEL_REL = "best.pt"
VIDEO_REL = r"proyecto_final_vision_artificial_202502\proyecto_final\video_piezas.mp4"

def load_model(model_path):
    try:
        # intento ultralytics (si está instalado)
        from ultralytics import YOLO
        return ("ultralytics", YOLO(model_path))
    except Exception:
        try:
            import torch
            model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=False)
            return ("yolov5", model)
        except Exception as e:
            print("No se pudo cargar modelo (instala 'ultralytics' o habilita acceso a internet para torch.hub).", file=sys.stderr)
            raise e

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_REL)
    video_path = os.path.join(VIDEO_REL)

    if not os.path.exists(model_path):
        print("Modelo no encontrado en:", model_path)
        return
    if not os.path.exists(video_path):
        print("Video no encontrado en:", video_path)
        return

    backend, model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(script_dir, "yolo_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    window = "YOLO Inference"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if backend == "ultralytics":
                # ultralytics YOLO: model(frame) -> Results, results[0].plot() returns annotated image
                res = model(frame)  # uses numpy BGR frames directly
                # obtener imagen anotada (ploteada)
                ann = res[0].plot()  # RGB
                ann_bgr = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)
                detected = len(res[0].boxes) if hasattr(res[0], "boxes") else 0
            else:
                # yolov5 from torch.hub: model(frame) -> results; .render() draws on imgs
                results = model(frame[..., ::-1])  # convert BGR->RGB for consistency
                # results.render() updates results.imgs (RGB)
                results.render()
                ann = results.imgs[0]
                ann_bgr = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)
                # detecciones en results.xyxy[0]
                detected = 0
                if hasattr(results, "xyxy") and len(results.xyxy) > 0:
                    detected = len(results.xyxy[0])

            # mostrar y guardar
            cv2.putText(ann_bgr, f"Detections: {detected}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow(window, ann_bgr)
            writer.write(ann_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print("Salida guardada en:", out_path)

if __name__ == "__main__":
    main()