import os
import sys
from glob import glob
import cv2
import numpy as np
import xlsxwriter
import random

# Asegurar que podamos importar las funciones del módulo existente
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# --- Funciones de aumentación y extracción (originales) ---
def count_peaks(proj, height=0.0):
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

def augment_image(image):
    """
    Toma una imagen y devuelve una lista de imágenes aumentadas (incluida la original).
    MEJORADO: Ahora genera muchos más patrones y variaciones.
    """
    augmented_images = []
    h, w = image.shape
    center = (w // 2, h // 2)
    
    # 1. Añadir la imagen original
    augmented_images.append(image.copy())

    # 2. ROTACIONES (más ángulos)
    rotation_angles = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, -7, 7, -6, 6]
    for angle in rotation_angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=0)
        augmented_images.append(rotated)

    # 3. ESCALADO (diferentes tamaños)
    scale_factors = [0.85, 0.9, 0.95, 1.05, 1.1, 1.15]
    for scale in scale_factors:
        M = cv2.getRotationMatrix2D(center, 0, scale)
        scaled = cv2.warpAffine(image, M, (w, h), borderValue=0)
        augmented_images.append(scaled)

    # 4. TRASLACIONES (movimientos pequeños)
    translations = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1), 
                   (-2, -1), (-2, 0), (-2, 1), (2, -1), (2, 0), (2, 1)]
    for tx, ty in translations:
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(image, M, (w, h), borderValue=0)
        augmented_images.append(translated)

    # 5. RUIDO GAUSSIANO (múltiples niveles)
    noise_levels = [1, 2, 3, 5]
    for noise_level in noise_levels:
        noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
        noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        augmented_images.append(noisy_image)

    # 6. AJUSTES DE BRILLO (más variaciones)
    brightness_adjustments = [-30, -20, -10, 10, 20, 30, -15, 15, -25, 25]
    for brightness in brightness_adjustments:
        if brightness > 0:
            bright_img = cv2.add(image, np.array([brightness], dtype=np.uint8))
        else:
            bright_img = cv2.subtract(image, np.array([abs(brightness)], dtype=np.uint8))
        augmented_images.append(bright_img)

    # 7. EROSIÓN Y DILATACIÓN (simular grosor de línea)
    kernels = [
        np.ones((2,2), np.uint8),
        np.ones((3,3), np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    ]
    
    for kernel in kernels:
        # Erosión (líneas más delgadas)
        eroded = cv2.erode(image, kernel, iterations=1)
        augmented_images.append(eroded)
        
        # Dilatación (líneas más gruesas)
        dilated = cv2.dilate(image, kernel, iterations=1)
        augmented_images.append(dilated)

    # 8. DEFORMACIONES ELÁSTICAS SIMPLES
    # Simulan pequeñas deformaciones de la placa
    shear_values = [-0.1, 0.1, -0.05, 0.05]
    for shear in shear_values:
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[shear*h, 0], [w+shear*h, 0], [0, h], [w, h]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        sheared = cv2.warpPerspective(image, M, (w, h), borderValue=0)
        augmented_images.append(sheared)

    # 9. COMBINACIONES (rotación + ruido, escala + brillo, etc.)
    # Crear algunas combinaciones aleatorias
    for _ in range(10):  # 10 combinaciones aleatorias
        temp_img = image.copy()
        
        # Rotación aleatoria pequeña
        angle = random.uniform(-3, 3)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        temp_img = cv2.warpAffine(temp_img, M, (w, h), borderValue=0)
        
        # Ruido aleatorio
        noise = np.random.normal(0, random.randint(1, 3), temp_img.shape).astype(np.int16)
        temp_img = np.clip(temp_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Brillo aleatorio
        brightness = random.randint(-15, 15)
        if brightness > 0:
            temp_img = cv2.add(temp_img, np.array([brightness], dtype=np.uint8))
        else:
            temp_img = cv2.subtract(temp_img, np.array([abs(brightness)], dtype=np.uint8))
        
        augmented_images.append(temp_img)

    # 10. BLUR Y SHARPENING
    # Simulan diferentes calidades de cámara
    blur_kernels = [(3,3), (5,5)]
    for kernel_size in blur_kernels:
        blurred = cv2.GaussianBlur(image, kernel_size, 0)
        augmented_images.append(blurred)
    
    # Sharpening kernel
    kernel_sharpen = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel_sharpen)
    augmented_images.append(sharpened)

    print(f"    🔄 Generadas {len(augmented_images)} variaciones de la imagen original")
    return augmented_images

def extract_features(img_roi_bin, contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    h, w = img_roi_bin.shape
    aspect_ratio = w / float(h)
    zone_features = []
    cell_size = 5

    contours_all, hierarchy = cv2.findContours(img_roi_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num_interior = 0
    area_ratio_interior = 0.0

    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, hinfo in enumerate(hierarchy):
            parent = hinfo[3]
            if parent != -1:
                num_interior += 1
                area_ratio_interior += cv2.contourArea(contours_all[i]) / (area + 1e-6)
    if num_interior > 0:
        area_ratio_interior /= num_interior

    ys, xs = np.where(img_roi_bin[:, :max(1, w//4)] > 0)
    r2_left = 0.0
    if len(xs) > 10:
        A = np.vstack([ys, np.ones_like(ys)]).T
        m, b = np.linalg.lstsq(A, xs, rcond=None)[0]
        x_pred = m*ys + b
        ss_res = np.sum((xs - x_pred)**2)
        ss_tot = np.sum((xs - xs.mean())**2) + 1e-6
        r2_left = 1.0 - ss_res/ss_tot

    for y in range(0, img_roi_bin.shape[0], cell_size):
        for x in range(0, img_roi_bin.shape[1], cell_size):
            roi_zone = img_roi_bin[y:y+cell_size, x:x+cell_size]
            density = cv2.countNonZero(roi_zone) / (cell_size * cell_size)
            zone_features.append(density)
    
    all_features = np.concatenate((
        [area, perimeter, circularity, aspect_ratio, r2_left, num_interior, area_ratio_interior],
        hu_moments,
        zone_features
    ))
    return all_features.astype(np.float32)

def extract_features_enhanced(img_roi_bin, contour):
    """Versión mejorada con características específicas para letras similares"""
    
    # Características básicas originales
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    h, w = img_roi_bin.shape
    aspect_ratio = w / float(h)
    
    # ==========================================
    # NUEVAS CARACTERÍSTICAS PARA DISTINGUIR LETRAS SIMILARES
    # ==========================================
    
    # 1. ANÁLISIS DE ESQUINAS (para distinguir O de Q, P, etc.)
    # Detectar esquinas usando Harris
    corners = cv2.cornerHarris(img_roi_bin, 2, 3, 0.04)
    num_corners = np.sum(corners > 0.01 * corners.max())
    corner_density = num_corners / (h * w)
    
    # 2. ANÁLISIS DE LÍNEAS HORIZONTALES Y VERTICALES
    # Para distinguir E, F, H, etc.
    
    # Proyecciones
    proj_h = np.sum(img_roi_bin, axis=1)  # Horizontal
    proj_v = np.sum(img_roi_bin, axis=0)  # Vertical
    
    # Contar picos en proyecciones (líneas horizontales/verticales)
    h_peaks = count_peaks(proj_h, height=np.max(proj_h) * 0.3)
    v_peaks = count_peaks(proj_v, height=np.max(proj_v) * 0.3)
    
    # 3. ANÁLISIS DE SIMETRÍA
    # Para distinguir letras simétricas vs asimétricas
    
    # Simetría horizontal (izq vs der)
    left_half = img_roi_bin[:, :w//2]
    right_half = np.fliplr(img_roi_bin[:, w//2:])
    
    # Ajustar tamaños si son diferentes
    min_w = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_w]
    right_half = right_half[:, :min_w]
    
    horizontal_symmetry = np.sum(left_half == right_half) / (h * min_w) if min_w > 0 else 0
    
    # Simetría vertical (arriba vs abajo)
    top_half = img_roi_bin[:h//2, :]
    bottom_half = np.flipud(img_roi_bin[h//2:, :])
    
    min_h = min(top_half.shape[0], bottom_half.shape[0])
    top_half = top_half[:min_h, :]
    bottom_half = bottom_half[:min_h, :]
    
    vertical_symmetry = np.sum(top_half == bottom_half) / (min_h * w) if min_h > 0 else 0
    
    # 4. ANÁLISIS DE DENSIDAD POR CUADRANTES
    # Para detectar patrones específicos de cada letra
    
    h_mid, w_mid = h//2, w//2
    
    q1_density = np.sum(img_roi_bin[:h_mid, :w_mid]) / (h_mid * w_mid)  # Superior izq
    q2_density = np.sum(img_roi_bin[:h_mid, w_mid:]) / (h_mid * (w - w_mid))  # Superior der
    q3_density = np.sum(img_roi_bin[h_mid:, :w_mid]) / ((h - h_mid) * w_mid)  # Inferior izq
    q4_density = np.sum(img_roi_bin[h_mid:, w_mid:]) / ((h - h_mid) * (w - w_mid))  # Inferior der
    
    # 5. ANÁLISIS DE CONTORNOS INTERNOS (para O vs Q vs P)
    contours_all, hierarchy = cv2.findContours(img_roi_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    num_interior = 0
    area_ratio_interior = 0.0
    largest_hole_ratio = 0.0
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        hole_areas = []
        
        for i, hinfo in enumerate(hierarchy):
            parent = hinfo[3]
            if parent != -1:  # Es un contorno interno
                num_interior += 1
                hole_area = cv2.contourArea(contours_all[i])
                hole_areas.append(hole_area)
                area_ratio_interior += hole_area / (area + 1e-6)
        
        if num_interior > 0:
            area_ratio_interior /= num_interior
            largest_hole_ratio = max(hole_areas) / (area + 1e-6) if hole_areas else 0
    
    # 6. ANÁLISIS DE BORDE SUPERIOR E INFERIOR
    # Para distinguir C, G, O, Q, etc.
    
    top_row_density = np.sum(img_roi_bin[0, :]) / w
    bottom_row_density = np.sum(img_roi_bin[-1, :]) / w
    left_col_density = np.sum(img_roi_bin[:, 0]) / h
    right_col_density = np.sum(img_roi_bin[:, -1]) / h
    
    # 7. CARACTERÍSTICAS DE FORMA ESPECÍFICAS
    
    # Compactness mejorada
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Extent (relación área/bounding box)
    x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
    extent = area / (bbox_w * bbox_h)
    
    # Características originales (simplificadas)
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
def extract_main_contour(img_binary):
    if img_binary is None or img_binary.size == 0:
        return None
    # Binarizar si es necesario
    if img_binary.dtype != np.uint8 or img_binary.max() > 1:
        _, th = cv2.threshold(img_binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        th = (img_binary * 255).astype(np.uint8)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) > 50:
        return main_contour
    return None

# Configuración: carpeta con los contornos originales
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
SOURCE_DB = os.path.join(PROJECT_ROOT, "contornos_database")
DEST_DB = os.path.join(PROJECT_ROOT,"data_augmentation", "contornos_database_aug")
OUTPUT_XLSX = os.path.join(HERE, "contornos_features_aug.xlsx")

CLASSES = ["argollas", "zetas", "tensores","zetasr"]

# Número máximo de aumentos por pasada por imagen (puede repetirse en pasadas posteriores)
MAX_AUG_PER_IMAGE = 20

# Número objetivo de muestras por clase en la base aumentada
TARGET_PER_CLASS = 1000

# Evita loops infinitos: máximo de pasadas sobre las originales para generar augmentaciones
MAX_PASSES = 50

os.makedirs(DEST_DB, exist_ok=True)

def process():
    all_rows = []
    total_original = 0

    for label_idx, cls in enumerate(CLASSES):
        src_folder = os.path.join(SOURCE_DB, cls)
        dst_folder = os.path.join(DEST_DB, cls)
        os.makedirs(dst_folder, exist_ok=True)

        # Limpiar destino para asegurar control exacto del número final
        for f in glob(os.path.join(dst_folder, "*")):
            try:
                os.remove(f)
            except Exception:
                pass

        img_paths = sorted(glob(os.path.join(src_folder, "*.*")))
        print(f"Procesando clase '{cls}' ({len(img_paths)} archivos). Objetivo: {TARGET_PER_CLASS} muestras")

        if not img_paths:
            print(f"  ⚠️ No hay imágenes en {src_folder}, se salta clase.")
            continue

        saved = 0
        pass_idx = 0
        total_original += len(img_paths)

        # Generar aumentos pasando repetidamente por las originales hasta alcanzar TARGET_PER_CLASS
        while saved < TARGET_PER_CLASS and pass_idx < MAX_PASSES:
            random.shuffle(img_paths)
            for img_path in img_paths:
                if saved >= TARGET_PER_CLASS:
                    break

                img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_gray is None:
                    continue

                base = os.path.splitext(os.path.basename(img_path))[0]

                # Generar augmentaciones (la función ya incluye la original como primer elemento)
                augmented_images = augment_image(img_gray)

                # Limitar por imagen para no salvar demasiadas en una sola pasada
                if len(augmented_images) > MAX_AUG_PER_IMAGE:
                    # conservar original + una muestra aleatoria del resto
                    original = augmented_images[0]
                    rest = augmented_images[1:]
                    k_rest = MAX_AUG_PER_IMAGE - 1
                    sampled = random.sample(rest, k_rest) if k_rest > 0 else []
                    augmented_images = [original] + sampled
                else:
                    # mezclar para evitar patrones repetidos
                    random.shuffle(augmented_images)

                for aug in augmented_images:
                    if saved >= TARGET_PERClass if False else saved >= TARGET_PER_CLASS:  # guard de seguridad
                        break

                    save_name = f"{base}_aug{saved:04d}.png"
                    save_path = os.path.join(dst_folder, save_name)
                    cv2.imwrite(save_path, aug)
                    saved += 1

                    # extraer contorno y características
                    contour = extract_main_contour(aug)
                    if contour is None:
                        continue

                    x, y, w, h = cv2.boundingRect(contour)
                    roi = aug[y:y+h, x:x+w]
                    if roi.size == 0:
                        continue

                    roi_resized = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_NEAREST)
                    try:
                        features = extract_features_enhanced(roi_resized, contour)
                    except Exception:
                        features = None
                    if features is None:
                        continue

                    row = np.concatenate(([label_idx], features)).tolist()
                    all_rows.append(row)

                    if saved >= TARGET_PER_CLASS:
                        break

                if saved >= TARGET_PER_CLASS:
                    break

            pass_idx += 1

            # si en una pasada no se generó ninguna nueva muestra, salir
            if pass_idx > 1 and saved == 0:
                print("  ⚠️ No se pudieron generar nuevas augmentaciones, saliendo.")
                break

        print(f"  -> guardadas {saved} muestras en {dst_folder} (pasadas realizadas: {pass_idx})")

    # Escribir Excel con características
    if all_rows:
        wb = xlsxwriter.Workbook(OUTPUT_XLSX)
        ws = wb.add_worksheet("features")
        for r, row in enumerate(all_rows):
            for c, val in enumerate(row):
                ws.write(r, c, float(val))
        wb.close()
        print(f"\nCaracterísticas guardadas en: {OUTPUT_XLSX}")
    else:
        print("\nNo se extrajeron características.")

    print(f"\nResumen: {total_original} originales procesadas. Salida aumentada en: {DEST_DB}")

if __name__ == "__main__":
    process()