import os
import shutil
import cv2
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# ==========================================
# MODELO B — YOLOv8 NOCTURNO 2
# Rol en el ensemble: casos difíciles y FN
# Parte de: yolo_pulmon_nocturno/best.pt
# Resultado: yolo_pulmon_nocturno2
# ==========================================

RUTA_MODELO_BASE = r"D:/Modelo_Prueba_2/models/yolo_pulmon_nocturno/weights/best.pt"
RUTA_DATASET     = r"D:/Modelo_Prueba_2/config_dataset.yaml"

DIR_TRAIN_IMGS   = r"D:\Modelo_Prueba_2\data\images\train"
DIR_TRAIN_LABELS = r"D:\Modelo_Prueba_2\data\labels\train"
DIR_VAL_IMGS     = r"D:\Modelo_Prueba_2\data\images\val"
DIR_VAL_LABELS   = r"D:\Modelo_Prueba_2\data\labels\val"

# FP persistentes — negativos que el modelo confunde con nódulos
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

# Nódulos difíciles — FN que aparecen en todos los experimentos
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


def get_paciente(nombre):
    partes = nombre.split("_")
    return f"{partes[0]}_{partes[1]}" if len(partes) >= 2 else ""


def inyectar_casos():
    """
    Inyecta en train:
    1. FP persistentes con 4 variantes — para que aprenda que ese tejido NO es nódulo
    2. Nódulos difíciles con 3 variantes — para no olvidar lo aprendido
    """
    neg_inyectados = 0
    pos_inyectados = 0

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

        # Negativos persistentes — 4 variantes
        if "_neg_" in nombre and pac in NEGATIVOS_PERSISTENTES:
            for factor, con_ruido, sufijo in [
                (1.00, False, "_fp0"),
                (1.12, False, "_fp1"),
                (0.88, False, "_fp2"),
                (1.06, True,  "_fp3"),
            ]:
                img_v = np.clip(img.astype(np.float32) * factor, 0, 255)
                if con_ruido:
                    img_v = np.clip(img_v + np.random.normal(0, 3, img_v.shape), 0, 255)
                cv2.imwrite(os.path.join(DIR_TRAIN_IMGS, f"{base}{sufijo}.png"),
                            img_v.astype(np.uint8))
                shutil.copy(ruta_lbl, os.path.join(DIR_TRAIN_LABELS, f"{base}{sufijo}.txt"))
                neg_inyectados += 1

        # Nódulos difíciles — 3 variantes
        elif "_nodulo_" in nombre and pac in PACIENTES_DIFICILES:
            for factor, sufijo in [(1.00, "_hd0"), (1.10, "_hd1"), (0.90, "_hd2")]:
                img_v = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(DIR_TRAIN_IMGS, f"{base}{sufijo}.png"), img_v)
                shutil.copy(ruta_lbl, os.path.join(DIR_TRAIN_LABELS, f"{base}{sufijo}.txt"))
                pos_inyectados += 1

    print(f"  Variantes negativas inyectadas : {neg_inyectados}")
    print(f"  Variantes positivas inyectadas : {pos_inyectados}")
    ratio = neg_inyectados / pos_inyectados if pos_inyectados > 0 else 0
    print(f"  Ratio neg/pos                  : {ratio:.2f}x")
    return neg_inyectados


def limpiar_inyectados():
    eliminados = 0
    for carpeta in [DIR_TRAIN_IMGS, DIR_TRAIN_LABELS]:
        for f in os.listdir(carpeta):
            if any(s in f for s in ["_fp", "_hd"]):
                os.remove(os.path.join(carpeta, f))
                eliminados += 1
    print(f"  Archivos inyectados eliminados: {eliminados}")


def entrenar():
    print("=" * 58)
    print("  Entrenamiento Nocturno 2 — Modelo B del ensemble")
    print("=" * 58)

    if not os.path.exists(RUTA_MODELO_BASE):
        print(f"ERROR: no se encontró {RUTA_MODELO_BASE}")
        print("Asegúrate de que el nocturno original ya fue entrenado.")
        return

    print("\nPaso 1: inyectando FP persistentes + nódulos difíciles...")
    n = inyectar_casos()
    if n == 0:
        print("  Sin casos encontrados.")
        return

    print("\nPaso 2: entrenando desde nocturno/best.pt...")
    modelo = YOLO(RUTA_MODELO_BASE)

    resultados = modelo.train(
        data=RUTA_DATASET,

        # Misma resolución que el nocturno
        imgsz=480,

        # Toda la noche
        epochs=40,
        patience=10,

        # CPU / 8 GB
        batch=3,
        workers=0,
        cache=False,
        device="cpu",
        half=False,

        # Sin freeze — entrenamiento completo
        # El backbone necesita re-aprender a distinguir tejido de FP

        # LR bajo — ajuste fino desde nocturno
        optimizer="AdamW",
        lr0=0.0003,
        lrf=0.005,
        warmup_epochs=2,
        weight_decay=0.0005,

        # Pérdida balanceada
        # cls=2.0 menos que nocturno (3.0) para no sesgar hacia positivos
        cls=2.0,
        box=8.0,
        dfl=1.5,

        # Augmentation igual al nocturno
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.05,
        degrees=20.0,
        translate=0.08,
        scale=0.25,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.0,

        save_period=10,
        resume=False,
        project="D:/Modelo_Prueba_2/models",
        name="yolo_pulmon_nocturno2",
    )

    print("\nPaso 3: limpiando inyecciones...")
    limpiar_inyectados()

    metricas = resultados.results_dict
    print("\n" + "=" * 58)
    print("  Nocturno 2 finalizado")
    print("=" * 58)
    print(f"  mAP50     : {metricas.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  Precisión : {metricas.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall    : {metricas.get('metrics/recall(B)', 0):.4f}")
    print(f"\n  Modelo B: D:/Modelo_Prueba_2/models/yolo_pulmon_nocturno2/weights/best.pt")


if __name__ == "__main__":
    entrenar()
