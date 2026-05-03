import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# ==========================================
# MODELO A — YOLOv8 NANO
# Rol en el ensemble: sensibilidad base
# Resultado: mejor.pt en yolo_pulmon_nano
# ==========================================

RUTA_MODELO_PREVIO = r"D:/Modelo_Prueba_2/models/yolo_pulmon_nano/weights/last.pt"
RUTA_DATASET       = r"D:/Modelo_Prueba_2/config_dataset.yaml"


def entrenar():
    print("=" * 55)
    print("  Entrenamiento Nano — Modelo A del ensemble")
    print("=" * 55)

    if os.path.exists(RUTA_MODELO_PREVIO):
        print("Entrenamiento previo encontrado. Reanudando...")
        modelo = YOLO(RUTA_MODELO_PREVIO)
        resume_training = True
    else:
        print("Iniciando desde yolov8n.pt (nano)...")
        modelo = YOLO("yolov8n.pt")
        resume_training = False

    resultados = modelo.train(
        data=RUTA_DATASET,

        # Imagen pequeña — cabe bien en 8 GB
        imgsz=320,

        # Duración con parada temprana
        epochs=30,
        patience=7,

        # CPU / 8 GB
        batch=4,
        workers=0,
        cache=False,
        device="cpu",
        half=False,

        # Congelar solo las 3 primeras capas
        freeze=3,

        # Optimizador
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,

        # Pérdida
        cls=1.5,
        box=7.5,
        dfl=1.5,

        # Augmentation médica
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.05,
        degrees=30.0,
        translate=0.1,
        scale=0.3,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.0,

        save_period=5,
        resume=resume_training,
        project="D:/Modelo_Prueba_2/models",
        name="yolo_pulmon_nano",
    )

    metricas = resultados.results_dict
    print("\n" + "=" * 55)
    print("  Nano finalizado")
    print("=" * 55)
    print(f"  mAP50     : {metricas.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  Precisión : {metricas.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall    : {metricas.get('metrics/recall(B)', 0):.4f}")
    print(f"\n  Modelo A: D:/Modelo_Prueba_2/models/yolo_pulmon_nano/weights/best.pt")


if __name__ == "__main__":
    entrenar()
