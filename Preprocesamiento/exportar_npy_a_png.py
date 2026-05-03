import os
import json
import numpy as np
import cv2

# ==========================================
# CONFIGURACIÓN
# ==========================================
DIR_DATASET_NPY = r"D:\IMAGENES PROYECTO DE GRADO\DATASET_NPY"
DIR_SALIDA      = r"D:\IMAGENES PROYECTO DE GRADO\NPY_A_PNG"

# Subcarpetas origen
SUBCARPETAS = ["con_nodulos", "sin_nodulos"]


def aplicar_ventana_pulmonar(imagen_hu):
    """Misma ventana que se usó para entrenar el modelo."""
    vmin, vmax = -1350, 150
    imagen_recortada   = np.clip(imagen_hu, vmin, vmax)
    imagen_normalizada = (imagen_recortada - vmin) / (vmax - vmin)
    return (imagen_normalizada * 255).astype(np.uint8)


def exportar_paciente(ruta_npy, ruta_json, carpeta_salida):
    """
    Exporta TODOS los slices de un paciente como PNG.
    Estructura: carpeta_salida/paciente_XXX/slice_000.png, slice_001.png, ...
    """
    # Cargar volumen
    volumen = np.load(ruta_npy)  # shape: (Z, Y, X)
    n_slices = volumen.shape[0]

    # Crear carpeta del paciente
    os.makedirs(carpeta_salida, exist_ok=True)

    # Cargar meta para marcar slices con nódulo
    nodulos_z = set()
    if os.path.exists(ruta_json):
        with open(ruta_json, "r") as f:
            meta = json.load(f)
        if "annotations" in meta and "nodulos" in meta["annotations"]:
            for nod in meta["annotations"]["nodulos"]:
                z = int(nod["voxel_Z"])
                # Marcar Z-1, Z, Z+1
                for offset in [-1, 0, 1]:
                    if 0 <= z + offset < n_slices:
                        nodulos_z.add(z + offset)

    # Exportar cada slice
    for z in range(n_slices):
        slice_2d  = aplicar_ventana_pulmonar(volumen[z, :, :])
        nombre    = f"slice_{z:03d}.png"
        ruta_png  = os.path.join(carpeta_salida, nombre)
        cv2.imwrite(ruta_png, slice_2d)

    return n_slices, len(nodulos_z)


def exportar_todo():
    print("=" * 60)
    print("  Exportador NPY → PNG completo por paciente")
    print("=" * 60)
    print(f"\nOrigen : {DIR_DATASET_NPY}")
    print(f"Destino: {DIR_SALIDA}\n")

    # Crear carpetas de destino
    for sub in SUBCARPETAS:
        os.makedirs(os.path.join(DIR_SALIDA, sub), exist_ok=True)

    total_pacientes  = 0
    total_slices     = 0
    errores          = []

    for subcarpeta in SUBCARPETAS:
        ruta_sub = os.path.join(DIR_DATASET_NPY, subcarpeta)
        if not os.path.exists(ruta_sub):
            print(f"  [AVISO] No existe: {ruta_sub}")
            continue

        # Listar todos los .npy
        archivos_npy = sorted([f for f in os.listdir(ruta_sub) if f.endswith(".npy")])
        print(f"Procesando '{subcarpeta}' — {len(archivos_npy)} pacientes...")

        for idx, archivo_npy in enumerate(archivos_npy):
            nombre_paciente = archivo_npy.replace(".npy", "")
            ruta_npy  = os.path.join(ruta_sub, archivo_npy)
            ruta_json = os.path.join(ruta_sub, archivo_npy.replace(".npy", "_meta.json"))

            # Carpeta destino: NPY_A_PNG/con_nodulos/paciente_002/
            carpeta_pac = os.path.join(DIR_SALIDA, subcarpeta, nombre_paciente)

            try:
                n_slices, n_nod_slices = exportar_paciente(ruta_npy, ruta_json, carpeta_pac)
                total_pacientes += 1
                total_slices    += n_slices

                # Progreso cada 50 pacientes
                if (idx + 1) % 50 == 0 or (idx + 1) == len(archivos_npy):
                    print(f"  [{idx+1:3d}/{len(archivos_npy)}] {nombre_paciente} "
                          f"— {n_slices} slices exportados")

            except Exception as e:
                errores.append(f"{nombre_paciente}: {e}")
                print(f"  [ERROR] {nombre_paciente}: {e}")

    print("\n" + "=" * 60)
    print(f"  EXPORTACIÓN COMPLETADA")
    print("=" * 60)
    print(f"  Pacientes exportados : {total_pacientes}")
    print(f"  Slices totales       : {total_slices:,}")
    print(f"  Errores              : {len(errores)}")
    print(f"\n  Estructura de salida:")
    print(f"  {DIR_SALIDA}/")
    print(f"    con_nodulos/")
    print(f"      paciente_002/")
    print(f"        slice_000.png")
    print(f"        slice_001.png")
    print(f"        ...")
    print(f"    sin_nodulos/")
    print(f"      paciente_001/")
    print(f"        slice_000.png")
    print(f"        ...")

    if errores:
        print(f"\n  Errores encontrados:")
        for e in errores:
            print(f"    - {e}")

    print(f"\n  Para usar en la app, sube los PNG de cualquier")
    print(f"  carpeta de paciente directamente desde:")
    print(f"  {DIR_SALIDA}\\con_nodulos\\paciente_XXX\\")


if __name__ == "__main__":
    exportar_todo()
