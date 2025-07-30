import cv2
import numpy as np
import time
import math
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO
from norfair import Detection, Tracker

# ========================
# Configuración general
# ========================
modelo = YOLO(r"G:\TRANSPORTE\runs\detect\train3\weights\best.pt")
tracker = Tracker(distance_function="euclidean", distance_threshold=30)

clases = modelo.names
umbrales_confianza = {
    1: 0.5,  # Auto
    2: 0.6,  # Bus
    3: 0.8,  # Camión
    4: 0.5,  # Camioneta
    5: 0.5,  # Coaster
    6: 0.3,  # Combi
    7: 0.6,  # Moto
    8: 0.8,  # MotoTaxi
    9: 0.8,  # TrailerM
    10: 0.8  # TrailerX
}

colores_clases = {
    1: (0, 255, 255), 2: (0, 0, 255), 3: (255, 0, 0),
    4: (0, 255, 0),   5: (255, 255, 0), 6: (255, 0, 255),
    7: (0, 165, 255), 8: (128, 0, 128), 9: (0, 100, 255),
    10: (75, 0, 130)
}

# ========================
# Definir segmentos personalizados con puntos
# ========================
SEGMENTOS = {
    "SUR": [(255, 110), (450, 110)],
    "NORTE": [(200, 380), (400, 380)],
    "OESTE": [(200, 150), (200, 350)],
    "ESTE": [(500, 150), (500, 300)]
}

# ========================
# Trackeo y conteo
# ========================
track_id_to_real_id = {}
real_id_to_last_pos = {}
clase_fija_por_id = {}
estado_vehiculo = {}
conteo_direcciones = {
    "SUR→NORTE": defaultdict(int),
    "NORTE→SUR": defaultdict(int),
    "ESTE→OESTE": defaultdict(int),
    "OESTE→ESTE": defaultdict(int),
}

next_real_id = 1
conteo_intervalos = []
intervalo_actual = 1
lapso = 15
start_time = time.time()
duracion_total = 180

cap = cv2.VideoCapture(4)

def distancia(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def punto_en_segmento(punto, segmento):
    x0, y0 = punto
    (x1, y1), (x2, y2) = segmento
    d1 = math.hypot(x0 - x1, y0 - y1)
    d2 = math.hypot(x0 - x2, y0 - y2)
    linea = math.hypot(x2 - x1, y2 - y1)
    return abs((d1 + d2) - linea) <= 10

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or (time.time() - start_time > duracion_total):
        break

    tiempo_transcurrido = int(time.time() - start_time)
    resultados = modelo(frame)
    detecciones = []

    # Dibujar segmentos con texto
    for nombre, (p1, p2) in SEGMENTOS.items():
        color = (255, 255, 0) if nombre == "SUR" else (0, 255, 255) if nombre == "NORTE" else (255, 0, 255) if nombre == "OESTE" else (0, 255, 0)
        cv2.line(frame, p1, p2, color, 2)
        cv2.putText(frame, nombre, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for result in resultados:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            clase_id = int(box.cls[0])
            conf = float(box.conf[0])

            if clase_id not in umbrales_confianza or conf < umbrales_confianza[clase_id]:
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            nombre_clase = clases[clase_id]

            detecciones.append(Detection(
                points=np.array([[cx, cy]]),
                data=(nombre_clase, clase_id, conf, x1, y1, x2, y2, cx, cy)
            ))

    objetos = tracker.update(detecciones)

    for obj in objetos:
        norfair_id = obj.id
        nombre_clase, clase_id, conf, x1, y1, x2, y2, cx, cy = obj.last_detection.data

        if norfair_id not in track_id_to_real_id:
            assigned = False
            for real_id, (last_cx, last_cy) in real_id_to_last_pos.items():
                if distancia((cx, cy), (last_cx, last_cy)) < 50:
                    track_id_to_real_id[norfair_id] = real_id
                    assigned = True
                    break
            if not assigned:
                track_id_to_real_id[norfair_id] = next_real_id
                next_real_id += 1

        real_id = track_id_to_real_id[norfair_id]
        real_id_to_last_pos[real_id] = (cx, cy)

        if real_id not in clase_fija_por_id:
            clase_fija_por_id[real_id] = (nombre_clase, clase_id)
        clase_fija, clase_id_fijo = clase_fija_por_id[real_id]
        color = colores_clases.get(clase_id_fijo, (255, 255, 255))

        if real_id not in estado_vehiculo:
            estado_vehiculo[real_id] = {
                "sur": False, "norte": False, "este": False, "oeste": False,
                "contado_sn": False, "contado_ns": False,
                "contado_oe": False, "contado_eo": False
            }

        if punto_en_segmento((cx, cy), SEGMENTOS["SUR"]):
            estado_vehiculo[real_id]["sur"] = True
        if punto_en_segmento((cx, cy), SEGMENTOS["NORTE"]):
            estado_vehiculo[real_id]["norte"] = True
        if punto_en_segmento((cx, cy), SEGMENTOS["ESTE"]):
            estado_vehiculo[real_id]["este"] = True
        if punto_en_segmento((cx, cy), SEGMENTOS["OESTE"]):
            estado_vehiculo[real_id]["oeste"] = True

        if estado_vehiculo[real_id]["sur"] and estado_vehiculo[real_id]["norte"] and not estado_vehiculo[real_id]["contado_sn"]:
            conteo_direcciones["SUR→NORTE"][clase_fija] += 1
            estado_vehiculo[real_id]["contado_sn"] = True

        if estado_vehiculo[real_id]["norte"] and estado_vehiculo[real_id]["sur"] and not estado_vehiculo[real_id]["contado_ns"]:
            conteo_direcciones["NORTE→SUR"][clase_fija] += 1
            estado_vehiculo[real_id]["contado_ns"] = True

        if estado_vehiculo[real_id]["este"] and estado_vehiculo[real_id]["oeste"] and not estado_vehiculo[real_id]["contado_eo"]:
            conteo_direcciones["ESTE→OESTE"][clase_fija] += 1
            estado_vehiculo[real_id]["contado_eo"] = True

        if estado_vehiculo[real_id]["oeste"] and estado_vehiculo[real_id]["este"] and not estado_vehiculo[real_id]["contado_oe"]:
            conteo_direcciones["OESTE→ESTE"][clase_fija] += 1
            estado_vehiculo[real_id]["contado_oe"] = True

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        texto = f"{clase_fija} ID:{real_id} ({conf:.2%})"
        cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    y_offset = 220
    for direccion, conteo in conteo_direcciones.items():
        cv2.putText(frame, direccion, (60, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 25
        for clase, cantidad in conteo.items():
            color = colores_clases[list(clases.keys())[list(clases.values()).index(clase)]]
            cv2.putText(frame, f"{clase}: {cantidad}", (60, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 20
        y_offset += 10

    if tiempo_transcurrido // lapso + 1 > intervalo_actual:
        fila = {"Tiempo (s)": tiempo_transcurrido}
        for direccion, conteo in conteo_direcciones.items():
            for clase in clases.values():
                key = f"{direccion} - {clase}"
                fila[key] = conteo.get(clase, 0)
        conteo_intervalos.append(fila)
        intervalo_actual += 1

    cv2.imshow("Conteo por Dirección", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Guardar CSV
df = pd.DataFrame(conteo_intervalos)
nombre_csv = f"conteo_direccion_{time.strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(nombre_csv, index=False)
print(f"\n✅ CSV guardado como: {nombre_csv}")
