import os
import shutil
import cv2
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# ==========================================
# MODELO C — YOLOv8 SMALL v22
# Rol en el ensemble: filtro de falsos positivos
# Parte de: yolov8s.pt (desde cero)
# Resultado: yolo_pulmon_small_v22
# ==========================================

RUTA_DATASET     = r"D:/Modelo_Prueba_2/config_dataset.yaml"

DIR_TRAIN_IMGS   = r"D:\Modelo_Prueba_2\data\images\train"
DIR_TRAIN_LABELS = r"D:\Modelo_Prueba_2\data\labels\train"
DIR_VAL_IMGS     = r"D:\Modelo_Prueba_2\data\images\val"
DIR_VAL_LABELS   = r"D:\Modelo_Prueba_2\data\labels\val"

# Nódulos difíciles — para mantener sensibilidad
PACIENTES_DIFICILES = {
    "paciente_040", "paciente_045", "paciente_079",
    "paciente_115", "paciente_129", "paciente_150",
    "paciente_163", "paciente_166", "paciente_208",
    "paciente_232", "paciente_236", "paciente_323",
    "paciente_337", "paciente_402", "paciente_537",
    "paciente_562", "paciente_594", "paciente_672",
    "paciente_691", "paciente_692", "paciente_709",
    "paciente_757", "paciente_765", "paciente_800",
}

# Negativos persistentes — para subir precisión
NEGATIVOS_PERSISTENTES = {
    "paciente_015", "paciente_034", "paciente_045",
    "paciente_047", "paciente_063", "paciente_066",
    "paciente_087", "paciente_129", "paciente_134",
    "paciente_136", "paciente_137", "paciente_143",
    "paciente_161", "paciente_163", "paciente_203",
    "paciente_208", "paciente_223", "paciente_293",
    "paciente_300", "paciente_311", "paciente_323",
    "paciente_324", "paciente_342", "paciente_346",
    "paciente_388", "paciente_389", "paciente_390",
    "paciente_393", "paciente_395", "paciente_427",
    "paciente_469", "paciente_485", "paciente_537",
    "paciente_567", "paciente_594", "paciente_625",
    "paciente_628", "paciente_672", "paciente_692",
    "paciente_693", "paciente_704", "paciente_709",
    "paciente_751", "paciente_802", "paciente_818",
    "paciente_839", "paciente_854", "paciente_867",
}


def get_paciente(nombre):
    partes = nombre.split("_")
    return f"{partes[0]}_{partes[1]}" if len(partes) >= 2 else ""


def inyectar_casos():
    """
    Inyecta en train con variantes para que el small aprenda
    tanto los nódulos difíciles como el tejido que NO es nódulo.
    """
    pos = 0
    neg = 0
    archivos_val = [f for f in os.listdir(DIR_VAL_IMGS) if f.endswith(".png")]

    for nombre in archivos_val:
        pac  = get_paciente(nombre)
        base = nombre.replace(".png", "")
        ruta_img = os.path.join(DIR_VAL_IMGS, nombre)
        ruta_lbl = os.path.join(DIR_VAL_LABELS, nombre.replace(".png", ".txt"))

        if not os.path.exists(ruta_lbl):
            continue
        img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Nódulos difíciles — 4 variantes
        if "_nodulo_" in nombre and pac in PACIENTES_DIFICILES:
            for factor, ruido, sufijo in [
                (1.00, False, "_sm0"),
                (1.15, False, "_sm1"),
                (0.85, False, "_sm2"),
                (1.08, True,  "_sm3"),
            ]:
                img_v = np.clip(img.astype(np.float32) * factor, 0, 255)
                if ruido:
                    img_v = np.clip(img_v + np.random.normal(0, 4, img_v.shape), 0, 255)
                cv2.imwrite(os.path.join(DIR_TRAIN_IMGS, f"{base}{sufijo}.png"),
                            img_v.astype(np.uint8))
                shutil.copy(ruta_lbl, os.path.join(DIR_TRAIN_LABELS, f"{base}{sufijo}.txt"))
                pos += 1

        # Negativos persistentes — 4 variantes
        elif "_neg_" in nombre and pac in NEGATIVOS_PERSISTENTES:
            for factor, ruido, sufijo in [
                (1.00, False, "_sn0"),
                (1.12, False, "_sn1"),
                (0.88, False, "_sn2"),
                (1.06, True,  "_sn3"),
            ]:
                img_v = np.clip(img.astype(np.float32) * factor, 0, 255)
                if ruido:
                    img_v = np.clip(img_v + np.random.normal(0, 3, img_v.shape), 0, 255)
                cv2.imwrite(os.path.join(DIR_TRAIN_IMGS, f"{base}{sufijo}.png"),
                            img_v.astype(np.uint8))
                shutil.copy(ruta_lbl, os.path.join(DIR_TRAIN_LABELS, f"{base}{sufijo}.txt"))
                neg += 1

    print(f"  Positivos inyectados : {pos}")
    print(f"  Negativos inyectados : {neg}")
    print(f"  Ratio neg/pos        : {neg/pos:.2f}x" if pos > 0 else "")
    return pos + neg


def limpiar():
    eliminados = 0
    for carpeta in [DIR_TRAIN_IMGS, DIR_TRAIN_LABELS]:
        for f in os.listdir(carpeta):
            if any(s in f for s in ["_sm", "_sn"]):
                os.remove(os.path.join(carpeta, f))
                eliminados += 1
    print(f"  Archivos eliminados: {eliminados}")


def entrenar():
    print("=" * 60)
    print("  Entrenamiento Small v22 — Modelo C del ensemble")
    print("=" * 60)

    print("\nPaso 1: inyectando positivos y negativos difíciles...")
    n = inyectar_casos()
    if n == 0:
        print("  Sin casos encontrados.")
        return

    print("\nPaso 2: entrenando YOLOv8s desde cero...")
    # Desde cero con yolov8s — más capacidad que nano para
    # aprender la frontera difícil entre nódulos y tejido confuso
    modelo = YOLO("yolov8s.pt")

    resultados = modelo.train(
        data=RUTA_DATASET,

        # Imagen más pequeña — más rápido en CPU
        imgsz=320,

        # Duración con parada temprana
        epochs=30,
        patience=8,

        # CPU / 8 GB
        batch=4,
        workers=0,
        cache=False,
        device="cpu",
        half=False,

        # Congelar primeras 10 capas — small es más pesado que nano
        freeze=10,

        # Optimizador
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=0.0005,

        # Pérdida con más peso en clasificación
        # cls=2.0 para aprender bien la distinción nódulo/no-nódulo
        cls=2.0,
        box=7.5,
        dfl=1.5,

        # Augmentation completa
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.05,
        degrees=25.0,
        translate=0.1,
        scale=0.3,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.0,

        save_period=5,
        resume=False,
        project="D:/Modelo_Prueba_2/models",
        name="yolo_pulmon_small_v22",
    )

    print("\nPaso 3: limpiando inyecciones...")
    limpiar()

    metricas = resultados.results_dict
    print("\n" + "=" * 60)
    print("  Small v22 finalizado")
    print("=" * 60)
    print(f"  mAP50     : {metricas.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  Precisión : {metricas.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall    : {metricas.get('metrics/recall(B)', 0):.4f}")
    print(f"\n  Modelo C: D:/Modelo_Prueba_2/models/yolo_pulmon_small_v22/weights/best.pt")
    print("\n  En 03_inferencia_yolo.py este modelo va como RUTA_MODELO_C")
    print("  Su rol es de FILTRO — veta detecciones de baja confianza de A y B")


if __name__ == "__main__":
    entrenar()
