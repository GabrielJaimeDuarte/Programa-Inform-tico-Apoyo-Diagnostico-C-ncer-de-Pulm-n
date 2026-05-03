import os
import json
import numpy as np
import cv2
import random
import shutil

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
DIR_DATASET_NPY = r"D:\IMAGENES PROYECTO DE GRADO\DATASET_NPY"
DIR_YOLO_DATA   = r"D:\Modelo_Prueba_2\data"

DIRS_TO_CREATE = [
    os.path.join(DIR_YOLO_DATA, "images", "train"),
    os.path.join(DIR_YOLO_DATA, "images", "val"),
    os.path.join(DIR_YOLO_DATA, "labels", "train"),
    os.path.join(DIR_YOLO_DATA, "labels", "val"),
]

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def preparar_carpetas():
    if os.path.exists(DIR_YOLO_DATA):
        shutil.rmtree(DIR_YOLO_DATA)
    for d in DIRS_TO_CREATE:
        os.makedirs(d, exist_ok=True)
    print("Carpetas de YOLO creadas (limpias).")


def aplicar_ventana_pulmonar(imagen_hu):
    vmin, vmax = -1350, 150
    imagen_recortada   = np.clip(imagen_hu, vmin, vmax)
    imagen_normalizada = (imagen_recortada - vmin) / (vmax - vmin)
    return (imagen_normalizada * 255).astype(np.uint8)


def guardar_yolo(split, nombre_base, slice_2d, cajas):
    """Guarda imagen PNG y archivo de etiquetas YOLO."""
    ruta_img = os.path.join(DIR_YOLO_DATA, "images", split, f"{nombre_base}.png")
    ruta_txt = os.path.join(DIR_YOLO_DATA, "labels", split, f"{nombre_base}.txt")
    cv2.imwrite(ruta_img, slice_2d)
    with open(ruta_txt, "w") as f:
        for caja in cajas:
            f.write(f"0 {caja['x']:.6f} {caja['y']:.6f} {caja['w']:.6f} {caja['h']:.6f}\n")


def rotar_imagen_y_cajas(slice_2d, cajas, angulo):
    """
    Rota la imagen y ajusta las coordenadas YOLO.
    Soporta ángulos: 90, 180, 270.
    """
    h, w = slice_2d.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1.0)
    rotada = cv2.warpAffine(slice_2d, M, (w, h))

    cajas_rot = []
    for c in cajas:
        cx, cy = c["x"], c["y"]
        if angulo == 90:
            cx_new, cy_new = 1.0 - cy, cx
        elif angulo == 180:
            cx_new, cy_new = 1.0 - cx, 1.0 - cy
        else:  # 270
            cx_new, cy_new = cy, 1.0 - cx
        cajas_rot.append({"x": cx_new, "y": cy_new, "w": c["w"], "h": c["h"]})

    return rotada, cajas_rot


def guardar_nodulo_con_augmentation(split, nombre_base, slice_2d, cajas):
    """
    Guarda un slice positivo (con nódulo) en 4 versiones:
    original + rotaciones 90°, 180°, 270°.
    Esto cuadruplica los ejemplos positivos y reduce los FN.
    """
    # Original
    guardar_yolo(split, nombre_base, slice_2d, cajas)

    # Rotaciones
    for angulo in [90, 180, 270]:
        img_rot, cajas_rot = rotar_imagen_y_cajas(slice_2d, cajas, angulo)
        guardar_yolo(split, f"{nombre_base}_rot{angulo}", img_rot, cajas_rot)


# ==========================================
# BUCLE PRINCIPAL
# ==========================================
def procesar_dataset():
    preparar_carpetas()

    carpetas_origen   = ["con_nodulos", "sin_nodulos"]
    todos_los_pacientes = []

    for subcarpeta in carpetas_origen:
        ruta_sub = os.path.join(DIR_DATASET_NPY, subcarpeta)
        if not os.path.exists(ruta_sub):
            continue
        for archivo in os.listdir(ruta_sub):
            if archivo.endswith("_meta.json"):
                todos_los_pacientes.append((ruta_sub, archivo))

    random.seed(42)
    random.shuffle(todos_los_pacientes)
    split_idx = int(len(todos_los_pacientes) * 0.8)

    splits = [
        ("train", todos_los_pacientes[:split_idx]),
        ("val",   todos_los_pacientes[split_idx:]),
    ]

    for split_nombre, lista_pacientes in splits:
        print(f"\nProcesando '{split_nombre}' — {len(lista_pacientes)} pacientes...")

        conteo_pos = 0
        conteo_neg = 0

        for ruta_sub, archivo_json in lista_pacientes:
            ruta_json = os.path.join(ruta_sub, archivo_json)
            ruta_npy  = os.path.join(ruta_sub, archivo_json.replace("_meta.json", ".npy"))
            if not os.path.exists(ruta_npy):
                continue

            with open(ruta_json, "r") as f:
                meta = json.load(f)

            volumen  = None
            shape_y  = meta["shape_ZYX"][1]
            shape_x  = meta["shape_ZYX"][2]
            spacing_x = meta["spacing_XYZ_mm"][0]

            # --------------------------------------------------
            # 1. POSITIVOS: Multi-slice (Z-1, Z, Z+1) + rotaciones
            #    Cada nódulo genera hasta 3 slices × 4 rotaciones = 12 imágenes
            # --------------------------------------------------
            if "annotations" in meta and "nodulos" in meta["annotations"]:
                volumen = np.load(ruta_npy)

                for idx, nodulo in enumerate(meta["annotations"]["nodulos"]):
                    z = int(nodulo["voxel_Z"])
                    x = int(nodulo["voxel_X"])
                    y = int(nodulo["voxel_Y"])
                    diametro_px = nodulo["diameter_mm"] / spacing_x

                    caja_yolo = {
                        "x": x / shape_x,
                        "y": y / shape_y,
                        "w": diametro_px / shape_x,
                        "h": diametro_px / shape_y,
                    }

                    # Guardar slice central + vecinos inmediatos
                    for offset in [-1, 0, 1]:
                        z_off = z + offset
                        if not (0 <= z_off < volumen.shape[0]):
                            continue
                        slice_2d  = aplicar_ventana_pulmonar(volumen[z_off, :, :])
                        nombre    = f"{meta['nombre_paciente']}_nodulo_{idx}_Z{z_off}"

                        # En validación NO aumentamos para medir métricas reales
                        if split_nombre == "train":
                            guardar_nodulo_con_augmentation("train", nombre, slice_2d, [caja_yolo])
                            conteo_pos += 4  # original + 3 rotaciones
                        else:
                            guardar_yolo("val", nombre, slice_2d, [caja_yolo])
                            conteo_pos += 1

            # --------------------------------------------------
            # 2. NEGATIVOS: Hard Negative Mining
            #    4 candidatos negativos por paciente (tejido sano difícil)
            # --------------------------------------------------
            if "candidates_v2" in meta and "candidatos" in meta["candidates_v2"]:
                cands_neg = [c for c in meta["candidates_v2"]["candidatos"] if c["class"] == 0]
                if cands_neg:
                    muestreo = random.sample(cands_neg, min(len(cands_neg), 4))
                    if volumen is None:
                        volumen = np.load(ruta_npy)

                    for i, cand in enumerate(muestreo):
                        z_neg    = int(cand["voxel_Z"])
                        slice_neg = aplicar_ventana_pulmonar(volumen[z_neg, :, :])
                        nombre   = f"{meta['nombre_paciente']}_neg_{i}_Z{z_neg}"
                        guardar_yolo(split_nombre, nombre, slice_neg, [])
                        conteo_neg += 1

        print(f"  Positivos guardados : {conteo_pos}")
        print(f"  Negativos guardados : {conteo_neg}")
        ratio = conteo_neg / conteo_pos if conteo_pos > 0 else 0
        print(f"  Ratio neg/pos       : {ratio:.2f}  (ideal: 1-3x)")

    print("\n¡Preprocesamiento finalizado con multi-slice + augmentation!")


if __name__ == "__main__":
    procesar_dataset()