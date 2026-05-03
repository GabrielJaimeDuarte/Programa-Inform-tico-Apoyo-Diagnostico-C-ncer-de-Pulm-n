from ultralytics import YOLO
import os
import cv2
import numpy as np
from collections import defaultdict

# ==========================================
# CONFIGURACIÓN — TRIPLE ENSEMBLE V31
# ==========================================
RUTA_IMAGENES  = r"D:\Modelo_Prueba_2\data\images\val"
RUTA_MODELO_A  = r"D:/Modelo_Prueba_2/models/yolo_pulmon_nano/weights/best.pt"
RUTA_MODELO_B  = r"D:/Modelo_Prueba_2/models/yolo_pulmon_nocturno2/weights/best.pt"
RUTA_MODELO_C  = r"D:/Modelo_Prueba_2/models/yolo_pulmon_small_v22/weights/best.pt"
RUTA_REPORTE   = r"D:\Modelo_Prueba_2\reporte_tesis_v31_final.txt"
DIR_BASE       = r"D:\Modelo_Prueba_2\RESULTADOS_VISUALES_V31"

RUTAS_VISUALES = {
    "ACIERTOS":           os.path.join(DIR_BASE, "ACIERTOS_NODULOS"),
    "ACIERTOS_NEGATIVOS": os.path.join(DIR_BASE, "ACIERTOS_NEGATIVOS"),
    "FALSAS_ALARMAS":     os.path.join(DIR_BASE, "FALSAS_ALARMAS"),
    "ESCAPES":            os.path.join(DIR_BASE, "ESCAPES"),
}

CONF_NANO_BASE     = 0.26
CONF_NANO_RUIDOSO  = 0.40
CONF_FINE_BASE     = 0.38
CONF_FINE_RUIDOSO  = 0.52
CONF_SMALL_BASE    = 0.35
CONF_SMALL_RUIDOSO = 0.50
UMBRAL_RUIDO       = 1


def parsear_nombre(nombre):
    base   = nombre.replace(".png", "")
    partes = base.split("_")
    try:
        paciente = f"{partes[0]}_{partes[1]}"
        tipo     = partes[2]
        id_obj   = partes[3]
        z        = int(partes[-1].replace("Z", ""))
        return paciente, tipo, id_obj, z
    except (IndexError, ValueError):
        return None, None, None, None


def id_nodulo_unico(paciente, id_obj):
    return f"{paciente}_nodulo_{id_obj}"


def id_negativo_unico(paciente, id_obj):
    return f"{paciente}_neg_{id_obj}"


def inferencia_tta(modelo, ruta_img, conf_umbral):
    img_orig = cv2.imread(ruta_img)
    if img_orig is None:
        return False, None, 0.0
    rutas_tmp     = []
    confianza_max = 0.0
    detecto       = False
    r_orig = modelo(ruta_img, conf=conf_umbral, verbose=False)[0]
    if len(r_orig.boxes) > 0:
        detecto       = True
        confianza_max = float(r_orig.boxes.conf.max())
    for flip_code, sufijo in [(1, "_tta_h"), (0, "_tta_v")]:
        img_flip = cv2.flip(img_orig, flip_code)
        ruta_tmp = ruta_img.replace(".png", f"{sufijo}.png")
        cv2.imwrite(ruta_tmp, img_flip)
        rutas_tmp.append(ruta_tmp)
        r = modelo(ruta_tmp, conf=conf_umbral, verbose=False)[0]
        if len(r.boxes) > 0:
            detecto       = True
            confianza_max = max(confianza_max, float(r.boxes.conf.max()))
    for t in rutas_tmp:
        if os.path.exists(t):
            os.remove(t)
    return detecto, r_orig, confianza_max


def calibrar_umbrales(modelo, nombres, conf_scan=0.12):
    ruido = defaultdict(int)
    for nombre in nombres:
        if "_neg_" not in nombre:
            continue
        paciente, _, _, _ = parsear_nombre(nombre)
        if paciente is None:
            continue
        ruta = os.path.join(RUTA_IMAGENES, nombre)
        r    = modelo(ruta, conf=conf_scan, verbose=False)[0]
        if len(r.boxes) > 0:
            ruido[paciente] += 1
    return {p for p, n in ruido.items() if n >= UMBRAL_RUIDO}


def decidir_deteccion(det_a, det_b, det_c,
                      conf_max_a, conf_max_b, conf_max_c,
                      es_ruidoso):
    if conf_max_a >= 0.55 or conf_max_b >= 0.60:
        decision_base = det_a or det_b
    elif det_a and det_b:
        decision_base = True
    elif det_a and conf_max_a >= 0.38:
        decision_base = True
    elif det_b and conf_max_b >= 0.52:
        decision_base = True
    else:
        decision_base = False

    if not decision_base:
        return False

    confianza_maxima_ab = max(conf_max_a, conf_max_b)
    if es_ruidoso:
        if not det_c and conf_max_c < 0.01 and confianza_maxima_ab < 0.45:
            return False
    else:
        if not det_c and conf_max_c < 0.03 and confianza_maxima_ab < 0.52:
            return False
    return True


def calcular_metricas():
    for ruta in RUTAS_VISUALES.values():
        os.makedirs(ruta, exist_ok=True)

    for ruta in [RUTA_MODELO_A, RUTA_MODELO_B, RUTA_MODELO_C]:
        if not os.path.exists(ruta):
            print(f"ERROR: no se encontró {ruta}")
            return

    print(f"Modelo A : {RUTA_MODELO_A}")
    print(f"Modelo B : {RUTA_MODELO_B}")
    print(f"Modelo C : {RUTA_MODELO_C}")
    print(f"Modo: TRIPLE ENSEMBLE V31 + métricas por paciente\n")

    modelo_a = YOLO(RUTA_MODELO_A)
    modelo_b = YOLO(RUTA_MODELO_B)
    modelo_c = YOLO(RUTA_MODELO_C)
    nombres  = sorted([f for f in os.listdir(RUTA_IMAGENES) if f.endswith(".png")])

    print("Calibrando umbrales...")
    ruidosos_a = calibrar_umbrales(modelo_a, nombres)
    ruidosos_b = calibrar_umbrales(modelo_b, nombres)
    ruidosos   = ruidosos_a & ruidosos_b
    print(f"  Pacientes ruidosos: {len(ruidosos)}\n")
    print(f"Evaluando {len(nombres)} slices...\n")

    # ---- estructuras por nódulo ----
    nodulos_detectados  = defaultdict(float)
    nodulos_total       = set()
    negativos_fp        = defaultdict(float)
    negativos_total     = set()
    mejor_slice_nodulo  = {}
    mejor_slice_negfp   = {}
    escape_slice        = {}
    acierto_neg_slice   = {}

    # ---- estructuras por paciente ----
    # pacientes_con_nodulos: pacientes que TIENEN nódulos reales
    # pacientes_sin_nodulos: pacientes que NO tienen nódulos
    # nodulos_detectados_por_paciente: nódulos que el modelo encontró por paciente
    # fp_por_paciente: falsas alarmas por paciente
    pacientes_con_nodulos         = set()
    pacientes_sin_nodulos         = set()
    nodulos_detectados_x_paciente = defaultdict(set)  # paciente -> {nid detectados}
    nodulos_totales_x_paciente    = defaultdict(set)  # paciente -> {nid totales}
    fp_x_paciente                 = defaultdict(set)  # paciente -> {neg_id con FP}

    for i, nombre in enumerate(nombres, 1):
        if i % 300 == 0:
            print(f"  {i}/{len(nombres)}...")

        paciente, tipo, id_obj, z = parsear_nombre(nombre)
        if paciente is None:
            continue

        ruta_img   = os.path.join(RUTA_IMAGENES, nombre)
        es_ruidoso = paciente in ruidosos

        conf_a = CONF_NANO_RUIDOSO  if es_ruidoso else CONF_NANO_BASE
        conf_b = CONF_FINE_RUIDOSO  if es_ruidoso else CONF_FINE_BASE
        conf_c = CONF_SMALL_RUIDOSO if es_ruidoso else CONF_SMALL_BASE

        det_a, res_a, conf_max_a = inferencia_tta(modelo_a, ruta_img, conf_a)
        det_b, res_b, conf_max_b = inferencia_tta(modelo_b, ruta_img, conf_b)
        det_c, res_c, conf_max_c = inferencia_tta(modelo_c, ruta_img, conf_c)

        detecto       = decidir_deteccion(det_a, det_b, det_c,
                                          conf_max_a, conf_max_b, conf_max_c,
                                          es_ruidoso)
        confianza_max = max(conf_max_a, conf_max_b, conf_max_c)
        confs         = [(conf_max_a, res_a), (conf_max_b, res_b), (conf_max_c, res_c)]
        resultado     = max(confs, key=lambda x: x[0])[1]
        img_cajas     = resultado.plot() if resultado is not None else cv2.imread(ruta_img)

        if tipo == "nodulo":
            nid = id_nodulo_unico(paciente, id_obj)
            nodulos_total.add(nid)
            nodulos_totales_x_paciente[paciente].add(nid)
            pacientes_con_nodulos.add(paciente)

            if detecto:
                if confianza_max > nodulos_detectados[nid]:
                    nodulos_detectados[nid]        = confianza_max
                    mejor_slice_nodulo[nid]        = (nombre, img_cajas)
                nodulos_detectados_x_paciente[paciente].add(nid)
            else:
                if nid not in escape_slice:
                    escape_slice[nid] = (nombre, cv2.imread(ruta_img))

        elif tipo == "neg":
            nid = id_negativo_unico(paciente, id_obj)
            negativos_total.add(nid)
            pacientes_sin_nodulos.add(paciente)

            if detecto:
                if confianza_max > negativos_fp.get(nid, 0):
                    negativos_fp[nid]       = confianza_max
                    mejor_slice_negfp[nid]  = (nombre, img_cajas)
                fp_x_paciente[paciente].add(nid)
            else:
                if nid not in acierto_neg_slice:
                    acierto_neg_slice[nid] = (nombre, cv2.imread(ruta_img))

    # Pacientes sin nódulos = solo aparecen en negativos, nunca en positivos
    pacientes_sin_nodulos = pacientes_sin_nodulos - pacientes_con_nodulos

    # ---- conteo por nódulo ----
    nodulos_detectados_set = set(nodulos_detectados.keys())
    nodulos_perdidos_set   = nodulos_total - nodulos_detectados_set
    negativos_fp_set       = set(negativos_fp.keys())
    negativos_tn_set       = negativos_total - negativos_fp_set

    TP_nod = len(nodulos_detectados_set)
    FN_nod = len(nodulos_perdidos_set)
    FP_nod = len(negativos_fp_set)
    TN_nod = len(negativos_tn_set)

    # ---- conteo por PACIENTE ----
    # Paciente detectado = al menos 1 nódulo detectado
    # Paciente escapado  = tenía nódulos pero ninguno detectado
    # Paciente FP        = no tiene nódulos pero el modelo disparó
    # Paciente TN        = no tiene nódulos y el modelo no disparó nada

    pac_tp = set()   # tenía nódulos Y detectó al menos uno
    pac_fn = set()   # tenía nódulos Y no detectó ninguno
    pac_fp = set()   # no tenía nódulos Y el modelo disparó
    pac_tn = set()   # no tenía nódulos Y el modelo no disparó

    for pac in pacientes_con_nodulos:
        if nodulos_detectados_x_paciente[pac]:
            pac_tp.add(pac)
        else:
            pac_fn.add(pac)

    for pac in pacientes_sin_nodulos:
        if fp_x_paciente[pac]:
            pac_fp.add(pac)
        else:
            pac_tn.add(pac)

    # ---- pacientes con detección PARCIAL ----
    # Detectó algunos nódulos pero no todos
    pac_parcial = {
        pac for pac in pac_tp
        if len(nodulos_detectados_x_paciente[pac]) < len(nodulos_totales_x_paciente[pac])
    }
    pac_completo = pac_tp - pac_parcial

    # ---- guardar imágenes ----
    for nid, (nombre, img) in mejor_slice_nodulo.items():
        cv2.imwrite(os.path.join(RUTAS_VISUALES["ACIERTOS"], nombre), img)
    for nid, (nombre, img) in escape_slice.items():
        if nid in nodulos_perdidos_set:
            cv2.imwrite(os.path.join(RUTAS_VISUALES["ESCAPES"], nombre), img)
    for nid, (nombre, img) in mejor_slice_negfp.items():
        cv2.imwrite(os.path.join(RUTAS_VISUALES["FALSAS_ALARMAS"], nombre), img)
    for nid, (nombre, img) in acierto_neg_slice.items():
        if nid in negativos_tn_set:
            cv2.imwrite(os.path.join(RUTAS_VISUALES["ACIERTOS_NEGATIVOS"], nombre), img)

    # ---- métricas por nódulo ----
    total_nod    = TP_nod + TN_nod + FP_nod + FN_nod
    exactitud    = (TP_nod + TN_nod) / total_nod * 100 if total_nod > 0 else 0
    sensibilidad = TP_nod / (TP_nod + FN_nod) * 100 if (TP_nod + FN_nod) > 0 else 0
    precision    = TP_nod / (TP_nod + FP_nod) * 100 if (TP_nod + FP_nod) > 0 else 0
    especif      = TN_nod / (TN_nod + FP_nod) * 100 if (TN_nod + FP_nod) > 0 else 0
    f1           = (2 * precision * sensibilidad / (precision + sensibilidad)
                    if (precision + sensibilidad) > 0 else 0)

    # ---- métricas por paciente ----
    total_pac       = len(pac_tp) + len(pac_tn) + len(pac_fp) + len(pac_fn)
    exact_pac       = (len(pac_tp) + len(pac_tn)) / total_pac * 100 if total_pac > 0 else 0
    sens_pac        = len(pac_tp) / (len(pac_tp) + len(pac_fn)) * 100 if (len(pac_tp) + len(pac_fn)) > 0 else 0
    prec_pac        = len(pac_tp) / (len(pac_tp) + len(pac_fp)) * 100 if (len(pac_tp) + len(pac_fp)) > 0 else 0
    especif_pac     = len(pac_tn) / (len(pac_tn) + len(pac_fp)) * 100 if (len(pac_tn) + len(pac_fp)) > 0 else 0
    f1_pac          = (2 * prec_pac * sens_pac / (prec_pac + sens_pac)
                       if (prec_pac + sens_pac) > 0 else 0)

    # ---- reporte ----
    reporte = (
        "==============================================\n"
        "  REPORTE FINAL V31 — TRIPLE ENSEMBLE\n"
        "  Métricas por nódulo Y por paciente\n"
        "==============================================\n"
        f"Modelo A : {RUTA_MODELO_A}\n"
        f"Modelo B : {RUTA_MODELO_B}\n"
        f"Modelo C : {RUTA_MODELO_C}\n\n"

        "----------------------------------------------\n"
        "  MÉTRICAS POR NÓDULO ÚNICO\n"
        "----------------------------------------------\n"
        f"Nódulos únicos en validación : {len(nodulos_total)}\n"
        f"Negativos únicos             : {len(negativos_total)}\n\n"
        "Matriz de confusión:\n"
        f"  TP (nódulos detectados)    : {TP_nod}\n"
        f"  TN (sanos correctos)       : {TN_nod}\n"
        f"  FP (falsas alarmas)        : {FP_nod}\n"
        f"  FN (nódulos perdidos)      : {FN_nod}\n\n"
        f"  Exactitud      : {exactitud:.2f}%\n"
        f"  Sensibilidad   : {sensibilidad:.2f}%\n"
        f"  Precisión      : {precision:.2f}%\n"
        f"  Especificidad  : {especif:.2f}%\n"
        f"  F1-Score       : {f1:.2f}%\n\n"

        "----------------------------------------------\n"
        "  MÉTRICAS POR PACIENTE COMPLETO\n"
        "----------------------------------------------\n"
        f"Pacientes con nódulos (reales)    : {len(pacientes_con_nodulos)}\n"
        f"Pacientes sin nódulos (reales)    : {len(pacientes_sin_nodulos)}\n"
        f"Total pacientes evaluados         : {total_pac}\n\n"
        "Matriz de confusión por paciente:\n"
        f"  TP pacientes detectados          : {len(pac_tp)}\n"
        f"     - Detectó TODOS sus nódulos   : {len(pac_completo)}\n"
        f"     - Detectó ALGUNOS (parcial)   : {len(pac_parcial)}\n"
        f"  TN pacientes sanos correctos     : {len(pac_tn)}\n"
        f"  FP pacientes con falsa alarma    : {len(pac_fp)}\n"
        f"  FN pacientes sin detección       : {len(pac_fn)}\n\n"
        f"  Exactitud por paciente    : {exact_pac:.2f}%\n"
        f"  Sensibilidad por paciente : {sens_pac:.2f}%\n"
        f"  Precisión por paciente    : {prec_pac:.2f}%\n"
        f"  Especificidad por paciente: {especif_pac:.2f}%\n"
        f"  F1 por paciente           : {f1_pac:.2f}%\n\n"

        "----------------------------------------------\n"
        "  PACIENTES CON TODOS LOS NÓDULOS DETECTADOS\n"
        "----------------------------------------------\n"
    )
    for pac in sorted(pac_completo):
        n_total = len(nodulos_totales_x_paciente[pac])
        reporte += f"  {pac}  ({n_total}/{n_total} nódulos)\n"

    reporte += (
        "\n----------------------------------------------\n"
        "  PACIENTES CON DETECCIÓN PARCIAL\n"
        "----------------------------------------------\n"
    )
    for pac in sorted(pac_parcial):
        n_det   = len(nodulos_detectados_x_paciente[pac])
        n_total = len(nodulos_totales_x_paciente[pac])
        reporte += f"  {pac}  ({n_det}/{n_total} nódulos detectados)\n"

    reporte += (
        "\n----------------------------------------------\n"
        "  PACIENTES CON NÓDULOS QUE NO SE DETECTARON\n"
        "----------------------------------------------\n"
    )
    for pac in sorted(pac_fn):
        n_total = len(nodulos_totales_x_paciente[pac])
        reporte += f"  {pac}  (0/{n_total} nódulos detectados)\n"

    reporte += (
        "\n----------------------------------------------\n"
        "  PACIENTES SANOS CON FALSA ALARMA\n"
        "----------------------------------------------\n"
    )
    for pac in sorted(pac_fp):
        n_fp = len(fp_x_paciente[pac])
        reporte += f"  {pac}  ({n_fp} falsa(s) alarma(s))\n"

    reporte += (
        "\n----------------------------------------------\n"
        "  PACIENTES SANOS CORRECTAMENTE IDENTIFICADOS\n"
        "----------------------------------------------\n"
        f"  Total: {len(pac_tn)} pacientes sanos sin ninguna alarma\n"
    )

    with open(RUTA_REPORTE, "w", encoding="utf-8") as f:
        f.write(reporte)

    print("\n" + reporte)
    print(f"Reporte  : {RUTA_REPORTE}")
    print(f"Visuales : {DIR_BASE}")


if __name__ == "__main__":
    calcular_metricas()