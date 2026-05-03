PROYECTO DE GRADO - DETECCION DE NODULOS PULMONARES
====================================================

Hardware : CPU Intel, 8 GB RAM (sin GPU)
Dataset  : LUNA16 - 888 pacientes, 1186 nodulos
Split    : 80% train / 20% val (seed=42)

ESTRUCTURA:
  01_DATASET\          Referencia a los datos pesados y CSVs
  02_PREPROCESAMIENTO\ Scripts para preparar el dataset YOLO
  03_ENTRENAMIENTO\    Scripts de los 3 modelos del ensemble
  04_MODELOS\          Rutas a los pesos entrenados (.pt)
  05_INFERENCIA\       Script de evaluacion y reporte final
  06_APP\              Aplicacion Streamlit de demostracion
  07_RESULTADOS\       Excel de analisis y referencia a visuales
  08_UTILS\            Herramientas auxiliares

FLUJO:
  NPY -> preprocesar -> entrenar x3 -> inferencia -> app

METRICAS FINALES V31:
  Exactitud    93.20% / 93.26%  |  Sensibilidad 88.76% / 94.59%
  Precision    86.09% / 94.59%  |  F1-Score     87.40% / 94.59%
