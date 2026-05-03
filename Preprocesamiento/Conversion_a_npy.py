"""
============================================================
  LUNA16 → NPY CONVERTER  (version completa con 3 CSVs)
  Proyecto de Grado - Deteccion de Cancer de Pulmon
============================================================

  Utiliza los 3 archivos de anotacion del dataset LUNA16:
    - annotations.csv    -> nodulos CONFIRMADOS (ground truth con diametro)
    - candidates.csv     -> candidatos v1 (class 0=FP, class 1=nodulo real)
    - candidates_V2.csv  -> candidatos v2 (class 0=FP, class 1=nodulo real)

  Por cada paciente guarda:
    - paciente_XXX.npy        -> volumen 3D completo en HU float32 (sin perdida)
    - paciente_XXX_meta.json  -> todos los metadatos + nodulos + candidatos

  Carpetas de salida:
    DATASET_NPY/
    ├── con_nodulos/
    ├── sin_nodulos/
    ├── nodulos_por_paciente.csv   <- tabla maestra de referencia
    └── resumen.txt

============================================================
"""

import os
import json
import time
import SimpleITK as sitk
import numpy as np
import pandas as pd

# ============================================================
#  CONFIGURACION — ajusta estas rutas a tu maquina
# ============================================================

DATASET_PATH      = r"D:\IMAGENES PROYECTO DE GRADO\DATASET"
OUTPUT_PATH       = r"D:\IMAGENES PROYECTO DE GRADO\DATASET_NPY"

ANNOTATIONS_CSV   = os.path.join(DATASET_PATH, "annotations.csv")
CANDIDATES_CSV    = os.path.join(DATASET_PATH, "candidates.csv")
CANDIDATES_V2_CSV = os.path.join(DATASET_PATH, "candidates_V2.csv")

SUBSET_FOLDERS = [f"subset{i}" for i in range(10)]  # subset0 ... subset9

# ============================================================
#  CARGA DE ANOTACIONES
# ============================================================

def cargar_annotations(csv_path):
    """
    annotations.csv: seriesuid, coordX, coordY, coordZ, diameter_mm
    Nodulos confirmados por radiólogos. Retorna dict {uid: [lista nodulos]}
    """
    if not os.path.exists(csv_path):
        print(f"  [AVISO] No se encontro: {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        uid = row['seriesuid']
        entry = {
            "coordX_mm":   float(row['coordX']),
            "coordY_mm":   float(row['coordY']),
            "coordZ_mm":   float(row['coordZ']),
            "diameter_mm": float(row['diameter_mm']),
        }
        result.setdefault(uid, []).append(entry)
    print(f"  -> annotations.csv:    {len(df)} registros | {len(result)} pacientes con nodulos")
    return result


def cargar_candidates(csv_path, nombre):
    """
    candidates.csv / candidates_V2.csv:
      seriesuid, coordX, coordY, coordZ, class  (1=nodulo, 0=falso positivo)
    Retorna dict {uid: [lista candidatos]}
    """
    if not os.path.exists(csv_path):
        print(f"  [AVISO] No se encontro: {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        uid = row['seriesuid']
        entry = {
            "coordX_mm": float(row['coordX']),
            "coordY_mm": float(row['coordY']),
            "coordZ_mm": float(row['coordZ']),
            "class":     int(row['class']),  # 1=nodulo real, 0=falso positivo
        }
        result.setdefault(uid, []).append(entry)
    reales = df[df['class'] == 1]
    print(f"  -> {nombre}: {len(df)} registros | "
          f"{len(reales)} nodulos reales (class=1) | "
          f"{len(df) - len(reales)} falsos positivos (class=0)")
    return result


# ============================================================
#  CONVERSION DE COORDENADAS MM -> VOXEL
# ============================================================

def mm_a_voxel(lista_puntos, itk_image):
    """
    Convierte coordenadas en mm al espacio de voxel del TAC.
    Critico para localizar nodulos dentro del array numpy despues.
    """
    resultado = []
    for punto in lista_puntos:
        xyz_mm = (punto["coordX_mm"], punto["coordY_mm"], punto["coordZ_mm"])
        try:
            voxel = itk_image.TransformPhysicalPointToIndex(xyz_mm)
            extra = {
                "voxel_X": int(voxel[0]),
                "voxel_Y": int(voxel[1]),
                "voxel_Z": int(voxel[2]),
            }
        except Exception:
            extra = {"voxel_X": -1, "voxel_Y": -1, "voxel_Z": -1}
        resultado.append({**punto, **extra})
    return resultado


# ============================================================
#  PROCESAMIENTO DE UN PACIENTE
# ============================================================

def procesar_paciente(mhd_path, uid, nombre_paciente, out_dir,
                      anotaciones, candidates, candidates_v2):
    """
    Lee el TAC, lo guarda como float32 .npy (valores HU exactos)
    y genera su .json de metadatos con toda la informacion.
    """
    # Leer imagen con SimpleITK
    itk_image = sitk.ReadImage(mhd_path)
    array = sitk.GetArrayFromImage(itk_image).astype(np.float32)
    # Shape resultante: (Z, Y, X)

    # Metadatos espaciales — CRITICOS para reconstruccion y localizacion
    spacing   = list(itk_image.GetSpacing())    # mm por voxel en (X, Y, Z)
    origin    = list(itk_image.GetOrigin())     # origen en mm
    direction = list(itk_image.GetDirection())  # matriz de direccion (9 valores)

    # Convertir todas las coordenadas de mm a voxel
    nod_ann  = mm_a_voxel(anotaciones.get(uid, []),   itk_image)
    nod_cv1  = mm_a_voxel(candidates.get(uid, []),    itk_image)
    nod_cv2  = mm_a_voxel(candidates_v2.get(uid, []), itk_image)

    # Estadisticas HU del volumen completo
    stats = {
        "hu_min":  float(array.min()),
        "hu_max":  float(array.max()),
        "hu_mean": float(round(float(array.mean()), 4)),
        "hu_std":  float(round(float(array.std()),  4)),
    }

    # Guardar volumen 3D como .npy
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{nombre_paciente}.npy"), array)

    # Guardar metadatos como .json
    meta = {
        "nombre_paciente":        nombre_paciente,
        "seriesuid_original":     uid,
        "tiene_nodulos":          len(nod_ann) > 0,
        "archivo_fuente":         mhd_path,
        "dtype":                  "float32",
        "unidad_valores":         "Hounsfield Units (HU)",
        "shape_ZYX":              list(array.shape),
        "spacing_XYZ_mm":         spacing,
        "origin_XYZ_mm":          origin,
        "direction_matrix_9vals": direction,
        "estadisticas_HU":        stats,

        # Nodulos confirmados por radiólogos (ground truth)
        "annotations": {
            "descripcion": "Nodulos confirmados. Coordenadas en mm y voxel + diametro.",
            "cantidad":    len(nod_ann),
            "nodulos":     nod_ann,
        },

        # Candidatos version 1
        "candidates_v1": {
            "descripcion":      "Candidatos v1. class=1 nodulo real, class=0 falso positivo.",
            "cantidad_total":   len(nod_cv1),
            "cantidad_nodulos": sum(1 for c in nod_cv1 if c["class"] == 1),
            "cantidad_fp":      sum(1 for c in nod_cv1 if c["class"] == 0),
            "candidatos":       nod_cv1,
        },

        # Candidatos version 2 (refinados)
        "candidates_v2": {
            "descripcion":      "Candidatos v2 refinados. class=1 nodulo real, class=0 falso positivo.",
            "cantidad_total":   len(nod_cv2),
            "cantidad_nodulos": sum(1 for c in nod_cv2 if c["class"] == 1),
            "cantidad_fp":      sum(1 for c in nod_cv2 if c["class"] == 0),
            "candidatos":       nod_cv2,
        },
    }

    with open(os.path.join(out_dir, f"{nombre_paciente}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    return array.shape


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=" * 62)
    print("   LUNA16 -> NPY  (annotations + candidates v1 y v2)")
    print("=" * 62)

    # 1. Cargar los 3 CSVs
    print("\n[1/4] Cargando archivos de anotaciones...")
    anotaciones   = cargar_annotations(ANNOTATIONS_CSV)
    candidates    = cargar_candidates(CANDIDATES_CSV,    "candidates.csv  ")
    candidates_v2 = cargar_candidates(CANDIDATES_V2_CSV, "candidates_V2.csv")
    uids_con_nodulos = set(anotaciones.keys())

    # 2. Crear carpetas
    print("\n[2/4] Preparando carpetas de salida...")
    dir_con = os.path.join(OUTPUT_PATH, "con_nodulos")
    dir_sin = os.path.join(OUTPUT_PATH, "sin_nodulos")
    os.makedirs(dir_con, exist_ok=True)
    os.makedirs(dir_sin, exist_ok=True)
    print(f"  -> {OUTPUT_PATH}")

    # 3. Buscar todos los .mhd
    print("\n[3/4] Buscando archivos .mhd en los subsets...")
    archivos_mhd = []
    for subset in SUBSET_FOLDERS:
        subset_path = os.path.join(DATASET_PATH, subset)
        if not os.path.isdir(subset_path):
            print(f"  [AVISO] Carpeta no encontrada: {subset_path}")
            continue
        # Busca .mhd directo en la carpeta O dentro de subcarpeta con el mismo nombre
        posibles_rutas = [
            subset_path,
            os.path.join(subset_path, subset)  # ej: subset3/subset3/
        ]
        for ruta in posibles_rutas:
            if not os.path.isdir(ruta):
                continue
            for archivo in sorted(os.listdir(ruta)):
                if archivo.endswith(".mhd"):
                    archivos_mhd.append({
                        "uid":    archivo.replace(".mhd", ""),
                        "path":   os.path.join(ruta, archivo),
                        "subset": subset,
                    })

    total = len(archivos_mhd)
    print(f"  -> Total de TACs encontrados: {total}")

    # 4. Procesar
    print(f"\n[4/4] Convirtiendo {total} TACs a .npy ...\n")
    resumen_filas = []
    errores = []
    cont_con = cont_sin = 0
    t0 = time.time()

    for idx, info in enumerate(archivos_mhd, start=1):
        uid    = info["uid"]
        path   = info["path"]
        subset = info["subset"]
        tiene  = uid in uids_con_nodulos
        nombre = f"paciente_{idx:03d}"
        out_dir = dir_con if tiene else dir_sin

        n_ann = len(anotaciones.get(uid, []))
        n_cv1 = len(candidates.get(uid, []))
        n_cv2 = len(candidates_v2.get(uid, []))

        print(f"  [{idx:03d}/{total}] {nombre} | {subset} | "
              f"{'CON' if tiene else 'SIN'} nod | "
              f"ann={n_ann} cand_v1={n_cv1} cand_v2={n_cv2}")

        try:
            shape  = procesar_paciente(path, uid, nombre, out_dir,
                                       anotaciones, candidates, candidates_v2)
            estado = "OK"
            if tiene:
                cont_con += 1
            else:
                cont_sin += 1
        except Exception as e:
            estado = f"ERROR: {e}"
            errores.append({"paciente": nombre, "uid": uid, "error": str(e)})
            shape  = (0, 0, 0)
            print(f"    !! ERROR: {e}")

        resumen_filas.append({
            "nombre_paciente":    nombre,
            "seriesuid_original": uid,
            "subset":             subset,
            "tiene_nodulos":      tiene,
            "n_nodulos_ann":      n_ann,
            "n_candidates_v1":    n_cv1,
            "n_candidates_v2":    n_cv2,
            "shape_Z":            shape[0] if shape else 0,
            "shape_Y":            shape[1] if len(shape) > 1 else 0,
            "shape_X":            shape[2] if len(shape) > 2 else 0,
            "estado":             estado,
        })

    # Guardar CSV maestro
    df_res = pd.DataFrame(resumen_filas)
    df_res.to_csv(os.path.join(OUTPUT_PATH, "nodulos_por_paciente.csv"),
                  index=False, encoding="utf-8")

    # Guardar log
    t_total = time.time() - t0
    with open(os.path.join(OUTPUT_PATH, "resumen.txt"), "w", encoding="utf-8") as f:
        f.write("RESUMEN CONVERSION LUNA16 -> NPY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total TACs procesados : {total}\n")
        f.write(f"  CON nodulos         : {cont_con}\n")
        f.write(f"  SIN nodulos         : {cont_sin}\n")
        f.write(f"  Errores             : {len(errores)}\n")
        f.write(f"Tiempo total          : {t_total/60:.1f} minutos\n\n")
        if errores:
            f.write("ERRORES:\n")
            for e in errores:
                f.write(f"  {e['paciente']} | {e['uid']}\n    -> {e['error']}\n")

    # Resumen consola
    print("\n" + "=" * 62)
    print("   CONVERSION COMPLETADA")
    print("=" * 62)
    print(f"  Total procesados  : {total}")
    print(f"  CON nodulos       : {cont_con}")
    print(f"  SIN nodulos       : {cont_sin}")
    print(f"  Errores           : {len(errores)}")
    print(f"  Tiempo            : {t_total/60:.1f} minutos")
    print(f"\n  Salida en         : {OUTPUT_PATH}")
    print(f"  Tabla maestra     : nodulos_por_paciente.csv")
    print(f"  Log               : resumen.txt")
    print("=" * 62)


if __name__ == "__main__":
    main()