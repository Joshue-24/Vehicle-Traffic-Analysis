import cv2
import numpy as np
import time
import math
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO

# ========================
# Configuración general
# ========================
modelo = YOLO("best.pt")
modelo.tracker = "bytetrack.yaml"
modelo.conf = 0.3
modelo.iou = 0.5
modelo.track=True

# Sistema de trails optimizado
trail_points = {}
max_trail_length = 10
frame_skip = 2
frame_count = 0

# Configuración de resolución
display_width = 640
display_height = 480
process_width = 384
process_height = 288

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

# Optimización de segmentos usando numpy
scale_x = display_width / process_width
scale_y = display_height / process_height

SEGMENTOS = {
    "SUR": np.array([(int(255/scale_x), int(110/scale_y)), (int(450/scale_x), int(110/scale_y))]),
    "NORTE": np.array([(int(200/scale_x), int(380/scale_y)), (int(400/scale_x), int(380/scale_y))]),
    "OESTE": np.array([(int(200/scale_x), int(150/scale_y)), (int(200/scale_x), int(350/scale_y))]),
    "ESTE": np.array([(int(500/scale_x), int(150/scale_y)), (int(500/scale_x), int(300/scale_y))])
}

# ========================
# Configuración de conteo por intervalos
# ========================
duracion_total = 180  # 3 minutos
lapso = 30  # Intervalos de 30 segundos
start_time = time.time()
conteo_nseo = defaultdict(lambda: defaultdict(int))

# ========================
# Funciones auxiliares
# ========================
def punto_en_segmento(punto, segmento):
    x0, y0 = punto
    (x1, y1), (x2, y2) = segmento
    d1 = math.hypot(x0 - x1, y0 - y1)
    d2 = math.hypot(x0 - x2, y0 - y2)
    linea = math.hypot(x2 - x1, y2 - y1)
    return abs((d1 + d2) - linea) <= 10

# Captura de video
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Cache para colores de trail
trail_colors_cache = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or (time.time() - start_time > duracion_total):
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Frames para visualización y procesamiento
    frame_display = cv2.resize(frame, (display_width, display_height))
    frame_process = cv2.resize(frame, (process_width, process_height))
    
    tiempo_transcurrido = int(time.time() - start_time)
    intervalo_actual = tiempo_transcurrido // lapso
    
    resultados = modelo.track(frame_process, 
                            persist=True,
                            verbose=False,
                            augment=False,
                            half=True,
                            imgsz=(process_width, process_height))

    # Dibujar segmentos
    for nombre, puntos in SEGMENTOS.items():
        color = {
            "NORTE": (0, 255, 255),
            "SUR": (255, 255, 0),
            "ESTE": (0, 255, 0),
            "OESTE": (255, 0, 255)
        }.get(nombre, (255, 255, 255))
        p1 = (int(puntos[0][0] * scale_x), int(puntos[0][1] * scale_y))
        p2 = (int(puntos[1][0] * scale_x), int(puntos[1][1] * scale_y))
        cv2.line(frame_display, p1, p2, color, 2)
        cv2.putText(frame_display, nombre, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if resultados[0].boxes.id is not None:
        boxes = resultados[0].boxes
        for idx, (box, track_id) in enumerate(zip(boxes.xyxy, boxes.id)):
            x1 = int(box[0].item() * scale_x)
            y1 = int(box[1].item() * scale_y)
            x2 = int(box[2].item() * scale_x)
            y2 = int(box[3].item() * scale_y)
            
            clase_id = int(boxes.cls[idx])
            conf = float(boxes.conf[idx])
            track_id = int(track_id)

            if clase_id not in umbrales_confianza or conf < umbrales_confianza[clase_id]:
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            clase_nombre = clases[clase_id]
            
            # Verificar posición y contar
            pos = (cx/scale_x, cy/scale_y)
            for segmento in ["NORTE", "SUR", "ESTE", "OESTE"]:
                if punto_en_segmento(pos, SEGMENTOS[segmento]):
                    conteo_nseo[intervalo_actual][f"{segmento}-{clase_nombre}"] += 1
                    break

            # Dibujar bounding box y etiqueta
            color = colores_clases.get(clase_id, (255, 255, 255))
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 1)
            texto = f"{clase_nombre[:3]}{track_id}"
            cv2.putText(frame_display, texto, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Mostrar conteo actual
    y_offset = 30
    cv2.putText(frame_display, f"Intervalo: {intervalo_actual*30}-{(intervalo_actual+1)*30}s", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    y_offset += 25
    for segmento in ["NORTE", "SUR", "ESTE", "OESTE"]:
        for clase in clases.values():
            key = f"{segmento}-{clase}"
            cantidad = conteo_nseo[intervalo_actual][key]
            if cantidad > 0:
                cv2.putText(frame_display, f"{key}: {cantidad}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                y_offset += 20

    cv2.imshow("Conteo N-S-E-O", frame_display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Preparar y guardar resultados
intervalos_data = []
for i in range(6):  # 6 intervalos de 30 segundos
    fila = {"Intervalo": f"{i*30}-{(i+1)*30}s"}
    for segmento in ["NORTE", "SUR", "ESTE", "OESTE"]:
        for clase in clases.values():
            key = f"{segmento}-{clase}"
            fila[key] = conteo_nseo[i].get(key, 0)
    intervalos_data.append(fila)

df = pd.DataFrame(intervalos_data)
nombre_csv = f"conteo_NSEO_{time.strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(nombre_csv, index=False)
print(f"\n✅ CSV guardado como: {nombre_csv}")