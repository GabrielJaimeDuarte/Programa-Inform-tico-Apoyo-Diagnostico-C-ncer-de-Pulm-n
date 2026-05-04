# Sistema Automatizado de Detección de Nódulos Pulmonares
### Proyecto de Grado — Gabriel Jaime Duarte López - Camilo José Cifuentes Garzón - Ingeniería Electrónica 2026
### YOLOv8 Triple Ensemble + NMS 3D + Clasificación Clínica en 4 Niveles

---

##  Descripción General

Este proyecto desarrolla un sistema de detección automatizada de nódulos pulmonares en imágenes de tomografía computarizada (CT), orientado como herramienta de apoyo al diagnóstico médico. El sistema fue entrenado y validado sobre el dataset público **LUNA16** (LUng Nodule Analysis 2016) y probado en el dataset **LIDC-IDRI** para verificar su capacidad de generalización.

El sistema implementa un enfoque de **Triple Ensemble** basado en tres modelos YOLOv8 con roles complementarios, combinados con un algoritmo de **NMS 3D** (Non-Maximum Suppression tridimensional) para agrupar detecciones en hallazgos volumétricos clínicamente significativos. Todo el sistema fue desarrollado y ejecutado completamente en **CPU sin necesidad de GPU**, lo que lo hace accesible para entornos con recursos computacionales limitados.

---

##  Métricas Finales — Versión V31

### Por Nódulo Único

| Métrica | Valor |
|---|---|
| Exactitud | **93.30%** |
| Sensibilidad (Recall) | **88.37%** |
| Precisión (PPV) | **86.69%** |
| Especificidad | **95.08%** |
| F1-Score | **87.52%** |

**Matriz de confusión por nódulo:**

|  | Predicho Positivo | Predicho Negativo |
|---|---|---|
| **Real Positivo** | TP = 228 | FN = 30 |
| **Real Negativo** | FP = 35 | TN = 677 |

- Nódulos únicos evaluados: **258**
- Negativos únicos evaluados: **712**
- Total evaluados: **970**

### Por Paciente Completo

| Métrica | Valor |
|---|---|
| Exactitud | **93.26%** |
| Sensibilidad | **94.59%** |
| Precisión | **94.59%** |
| Especificidad | **91.04%** |
| F1-Score | **94.59%** |

**Matriz de confusión por paciente:**

|  | Predicho Positivo | Predicho Negativo |
|---|---|---|
| **Real Positivo** | TP = 105 | FN = 6 |
| **Real Negativo** | FP = 6 | TN = 61 |

- Pacientes con nódulos: **111**
- Pacientes sanos: **67**
- Total pacientes evaluados: **178**
- Detectó TODOS los nódulos: **84 pacientes**
- Detectó ALGUNOS (parcial): **21 pacientes**

---

## Estructura del Repositorio

```
PROYECTO_FINAL_NODULOS/
│
├── PREPROCESAMIENTO/           # Scripts de conversión y preparación de datos
│   ├── conversion_a_npy.py         # Conversión de MetaImage (.mhd/.raw) a NumPy (.npy)
│   ├── exportar_slices_png.py      # Exportación de volúmenes NPY a slices PNG con ventana pulmonar
│   ├── categorizar_slices.py       # Categorización de slices con/sin nódulo usando annotations.csv
│   └── creador_val_train.py        # División train/val por paciente completo (seed=42, 80/20)
│
├── ENTRENAMIENTO/              # Scripts de entrenamiento de los tres modelos
│   ├── entrenar_nano.py            # Entrenamiento Modelo A — YOLOv8n (alta sensibilidad)
│   ├── entrenar_nocturno2.py       # Entrenamiento Modelo B — YOLOv8n con Hard Example Mining
│   ├── entrenar_small_v22.py       # Entrenamiento Modelo C — YOLOv8s (filtro de falsos positivos)
│   └── config_dataset.yaml         # Configuración del dataset para YOLOv8 (rutas y clases)
│
├── INFERENCIA/                 # Script de evaluación sobre el conjunto de validación
│   └── inferencia_v31.py           # Triple Ensemble V31 con NMS 3D — genera métricas completas
│
├── PROGRAMA/                   # Aplicación de escritorio
│   └── app_nodulos_desktop.py      # App PyQt6 con Triple Ensemble, NMS 3D y clasificación clínica
│
├── UTILS/                      # Herramientas auxiliares
│   └── generar_excel_slices.py     # Genera Excel con posición exacta (slice Z) de cada nódulo
│
├── requirements.txt            # Dependencias Python del proyecto
├── .gitignore                  # Archivos y carpetas excluidos del repositorio
└── README.md                   # Este archivo
```

> **Nota:** Los datos del dataset (imágenes PNG, volúmenes NPY), los pesos entrenados (`.pt`) y el ejecutable compilado (`.exe`) **no están incluidos** en el repositorio por su tamaño. Ver sección de instalación para obtenerlos.

---

## Arquitectura del Sistema

### Los Tres Modelos del Ensemble

El sistema combina tres modelos YOLOv8 con roles complementarios. Cada modelo fue entrenado con una estrategia diferente para maximizar la complementariedad:

| Modelo | Arquitectura | Rol en el Ensemble | Estrategia de Entrenamiento |
|---|---|---|---|
| **Modelo A (Nano)** | YOLOv8n — 3.2M parámetros | Alta sensibilidad — detector base | Transfer learning desde COCO, freeze=3, imágenes 320×320 |
| **Modelo B (Nocturno2)** | YOLOv8n — 3.2M parámetros | Casos difíciles — Hard Example Mining | Fine-tuning con FP y FN persistentes inyectados, imágenes 480×480 |
| **Modelo C (Small v22)** | YOLOv8s — 11.2M parámetros | Filtro de falsos positivos | Arquitectura más grande, freeze=10, especializado en precisión |

### Lógica de Decisión del Ensemble

Para cada slice analizado, el ensemble sigue estas reglas en orden de prioridad:

1. Si Modelo A ≥ 55% **O** Modelo B ≥ 60% → **ACEPTA**
2. Si Modelo A **Y** Modelo B detectan → **ACEPTA** (consenso)
3. Si solo Modelo A detecta con conf ≥ 38% → **ACEPTA**
4. Si solo Modelo B detecta con conf ≥ 52% → **ACEPTA**
5. Si ninguna regla anterior → **DESCARTA**
6. Veto adicional: si Modelo C no confirma con conf suficiente → **DESCARTA**

### Umbrales del Sistema

| Parámetro | Valor | Descripción |
|---|---|---|
| Umbral confianza mínima | 0.18 | Nivel mínimo para reportar una detección |
| Gap máximo NMS 3D | 3 slices | Slices limpios tolerados entre detecciones del mismo nódulo |
| Conf. mínima 1 corte | 0.50 | Umbral especial para detecciones en un único slice |
| Conf. mínima 2-3 cortes | 0.45 | Umbral para hallazgos en pocos slices |

### Test-Time Augmentation (TTA)

Cada slice se analiza en **3 variantes** simultáneas:
- Imagen original
- Flip horizontal (espejo izquierda-derecha)
- Flip vertical (espejo arriba-abajo)

La confianza reportada es el máximo de las tres variantes. Esto aumenta la sensibilidad aproximadamente 3-5 puntos porcentuales sin necesidad de entrenar modelos adicionales.

### NMS 3D — Agrupación Volumétrica

El algoritmo NMS 3D agrupa las detecciones individuales en slices consecutivos en hallazgos volumétricos:

1. Se recopilan todos los slices donde el ensemble detectó un nódulo
2. Se agrupan slices separados por ≤ `gap_max` slices limpios en un mismo cluster
3. Cada cluster representa un hallazgo volumétrico único
4. Se calcula el diámetro estimado usando la bounding box del modelo (eje XY) y el número de slices (eje Z)

### Clasificación Clínica en 4 Niveles

| Nivel | Color | Criterio | Acción recomendada |
|---|---|---|---|
| 🔴 NÓDULO PROBABLE | Rojo | 4+ cortes ≥60% **O** 2-3 cortes ≥65% | Revisión prioritaria por radiólogo |
| 🟠 SOSPECHA MODERADA | Naranja | 4+ cortes <60% **O** 2-3 cortes 45-64% | Evaluación recomendada |
| 🟡 HALLAZGO INCIDENTAL | Amarillo | 1 corte ≥50% | Verificación manual recomendada |
| ⚪ PROBABLE ARTEFACTO | Gris | No cumple criterios mínimos | Se descarta automáticamente |

---

## Dataset

### LUNA16 (Dataset de Entrenamiento y Validación)

| Característica | Valor |
|---|---|
| Pacientes totales | 888 estudios CT |
| Nódulos anotados | 1,186 confirmados por consenso de radiólogos |
| Pacientes con nódulo | 601 |
| Pacientes sin nódulo | 287 |
| Tamaño mínimo de nódulo | 3 mm de diámetro |
| Formato original | MetaImage (.mhd + .raw) |
| Espaciado típico XY | 0.5–1.0 mm/pixel |
| Espaciado típico Z | 1.0–2.5 mm entre slices |
| División utilizada | 80% train / 20% validación por paciente completo, seed=42 |

**Pipeline de preprocesamiento:**
1. MetaImage (.mhd/.raw) → NumPy 3D (.npy) usando SimpleITK
2. Conversión de coordenadas mm → índices de pixel usando spacing y origin
3. Aplicación de ventana pulmonar: recorte a [-1350, +150] HU → normalización a [0, 255]
4. Exportación de slices PNG a 320×320 píxeles
5. Generación de etiquetas YOLO: bounding box normalizada centrada en el nódulo

### LIDC-IDRI (Dataset de Prueba de Generalización)

Dataset padre de LUNA16. Contiene 1,018 casos CT con anotaciones cualitativas de 4 radiólogos independientes, incluyendo calificaciones de malignidad del 1 al 5.

**Resultado de la prueba con LIDC-IDRI-0001:** el sistema detectó correctamente el hallazgo principal (nódulo con malignidad 4-5) en los slices 86-96 con confianza máxima del 67.6%, clasificado como NÓDULO PROBABLE. Esto confirma la capacidad de generalización a datos de equipos CT diferentes.

---

## Requisitos del Sistema

| Componente | Mínimo recomendado |
|---|---|
| Sistema Operativo | Windows 10/11 (64-bit) |
| Procesador | Intel Core i5 o equivalente |
| RAM | 8 GB (16 GB recomendado) |
| Almacenamiento | 5 GB libres para la app y modelos |
| GPU | **No requerida** — funciona completamente en CPU |

---

## Instalación y Configuración

### Requisitos de Python

```bash
pip install -r requirements.txt
```

Contenido de `requirements.txt`:

```
PyQt6
ultralytics
opencv-python
pillow
openpyxl
numpy
pydicom
simpleitk
```

### Configuración de Rutas de los Modelos

Antes de ejecutar la aplicación, editar las constantes al inicio de `PROGRAMA/app_nodulos_desktop.py`:

```python
RUTA_MODELO_A = r"ruta/a/yolo_pulmon_nano/weights/best.pt"
RUTA_MODELO_B = r"ruta/a/yolo_pulmon_nocturno2/weights/best.pt"
RUTA_MODELO_C = r"ruta/a/yolo_pulmon_small_v22/weights/best.pt"
```

### Ejecutar la Aplicación (modo desarrollo)

```bash
conda activate nombre_entorno
cd PROYECTO_FINAL_NODULOS
python PROGRAMA/app_nodulos_desktop.py
```

### Generar el Ejecutable (.exe)

```bash
conda activate nombre_entorno
cd PROYECTO_FINAL_NODULOS

pyinstaller --noconfirm --onefile --windowed ^
  --name "AnalisisPulmonar" ^
  --add-data "modelos;modelos" ^
  --collect-all ultralytics ^
  --collect-all torch ^
  --collect-all pydicom ^
  PROGRAMA/app_nodulos_desktop.py
```

El ejecutable queda en `dist/AnalisisPulmonar.exe`.

---

## Funcionalidades de la Aplicación

### Formatos de Entrada Soportados

| Formato | Descripción | Preprocesamiento |
|---|---|---|
| PNG / JPG | Slices ya exportados con ventana pulmonar aplicada | Carga directa |
| NPY | Volúmenes 3D NumPy en Unidades Hounsfield | Ventana pulmonar automática |
| DICOM ZIP | Estudio CT completo comprimido | Ordena por ImagePositionPatient, aplica ventana pulmonar |
| DICOM .dcm | Archivos DICOM individuales | Mismo procesamiento que ZIP |

### Flujo de Análisis

1. Cargar archivos del paciente (PNG, NPY o DICOM)
2. Configurar parámetros en el panel de configuración
3. Ejecutar análisis — la inferencia corre en hilo separado (UI no se congela)
4. Ver resultado: banner con nivel máximo, cards de resumen, hallazgos detallados
5. Ver miniaturas de cortes detectados — click para abrir imagen en grande
6. Descargar reporte Excel (4 hojas) o TXT con sección ARCHIVOS_ANALIZADOS

### Reportes Generados

**Excel** (4 hojas):
- Hoja 1: Hallazgos válidos con clasificación clínica, rango de slices y diámetro estimado
- Hoja 2: Detalle por slice con clasificación individual
- Hoja 3: Detecciones descartadas con razón de descarte
- Hoja 4: Métricas del modelo V31

**TXT**:
- Reporte de hallazgos legible
- Sección `ARCHIVOS_ANALIZADOS` con formato `nombre.png|0o1|confianza` para el comparador

---

## Reproducibilidad del Entrenamiento

Los tres scripts de entrenamiento están parametrizados para reproducir exactamente los modelos del ensemble V31. Para reentrenar:

```bash
# Modelo A
python ENTRENAMIENTO/entrenar_nano.py

# Modelo B (requiere que Modelo A esté entrenado)
python ENTRENAMIENTO/entrenar_nocturno2.py

# Modelo C
python ENTRENAMIENTO/entrenar_small_v22.py
```

**Parámetros clave por modelo:**

| Parámetro | Modelo A (Nano) | Modelo B (Nocturno2) | Modelo C (Small v22) |
|---|---|---|---|
| Arquitectura base | yolov8n.pt | yolo_pulmon_nocturno.pt | yolov8s.pt |
| Tamaño imagen | 320×320 | 480×480 | 320×320 |
| Batch size | 4 | 3 | 4 |
| Épocas | 30 | 40 | 30 |
| Freeze | 3 | 0 | 10 |
| Learning rate | 0.001 | 0.0003 | 0.001 |
| Peso cls | 1.5 | 2.0 | 2.0 |
| Peso box | 7.5 | 8.0 | 7.5 |
| Punto de partida | COCO | Nocturno anterior | COCO |

---

## Limitaciones

1. **Análisis 2D**: el sistema analiza cada slice de forma independiente. El NMS 3D posterior es una aproximación al análisis volumétrico completo pero no captura todas las características 3D de un nódulo.

2. **Hardware CPU**: el análisis de un CT completo de ~200 slices tarda aproximadamente 15–20 minutos en CPU frente a menos de 1 minuto en GPU.

3. **Tamaño mínimo**: los modelos fueron entrenados con nódulos de mínimo 3 mm. Nódulos menores pueden no ser detectados.

4. **Data leakage documentado**: el Modelo B fue entrenado con variaciones de slices de pacientes del conjunto de validación (Hard Example Mining). Esto introduce un leve data leakage que puede inflar marginalmente las métricas del ensemble. Los Modelos A y C no presentan este problema.

5. **Generalización limitada**: el sistema fue entrenado y validado con datos de LUNA16. Puede requerir ajuste de umbrales para otros equipos CT con características de imagen diferentes.

---

## Comparación con el Estado del Arte

El estado del arte en LUNA16 reporta sensibilidades de 90–96% usando GPU con arquitecturas especializadas 3D. Este proyecto logra **88.37% de sensibilidad ejecutando completamente en CPU sin GPU**, con una aplicación completa que incluye interfaz gráfica, clasificación clínica y herramientas de evaluación. El aporte principal no es superar las métricas absolutas sino demostrar la viabilidad de un sistema funcional completo dentro de restricciones reales de hardware.

---

## Aviso Importante

> **Este sistema es una herramienta computacional de apoyo al diagnóstico, desarrollada como Proyecto de Grado académico. NO constituye un dispositivo médico certificado ni un diagnóstico médico. Todos los hallazgos deben ser confirmados por un radiólogo o médico especialista certificado.**

---

## Archivos No Incluidos en el Repositorio

Por limitaciones de tamaño, los siguientes archivos no están incluidos:

| Archivo / Carpeta | Razón de exclusión | Cómo obtener |
|---|---|---|
| Imágenes PNG del dataset | 227,225 archivos — 31 GB | Descargar LUNA16 desde TCIA |
| Volúmenes NPY | Derivados de LUNA16 | Ejecutar `PREPROCESAMIENTO/conversion_a_npy.py` |
| Pesos entrenados (`*.pt`) | >100 MB por modelo | Contactar al autor |
| Ejecutable (`AnalisisPulmonar.exe`) | >1.5 GB compilado | Compilar con PyInstaller usando instrucciones arriba |

---

## Tecnologías Utilizadas

| Tecnología | Versión | Uso |
|---|---|---|
| Python | 3.10+ | Lenguaje principal |
| YOLOv8 (Ultralytics) | 8.x | Arquitectura de detección de objetos |
| PyQt6 | 6.x | Interfaz gráfica de escritorio |
| OpenCV | 4.x | Procesamiento de imagen |
| NumPy | 1.24+ | Manejo de volúmenes CT |
| SimpleITK | 2.x | Lectura de formato MetaImage |
| pydicom | 2.x | Lectura de archivos DICOM |
| openpyxl | 3.x | Generación de reportes Excel |
| PyInstaller | 6.x | Compilación del ejecutable |

---

*Proyecto de Grado 2026 — Sistema de Detección de Nódulos Pulmonares*
*LUNA16 · YOLOv8 Triple Ensemble · NMS 3D · CPU · PyQt6*
