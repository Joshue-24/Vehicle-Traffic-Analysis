# Sistema de Análisis de Tráfico Vehicular Inteligente

![Portada del Proyecto](Trafic_1.png)

Sistema avanzado de monitoreo y análisis de tráfico vehicular en tiempo real que utiliza visión por computadora e inteligencia artificial para el conteo y seguimiento de vehículos en diferentes segmentos viales.

## 🚀 Características Principales

- 🚗 **Detección en Tiempo Real**: Utiliza YOLOv8 para la detección precisa de vehículos.
- 📍 **Seguimiento de Objetos**: Implementa ByteTrack para el seguimiento consistente de vehículos entre fotogramas.
- 🛣️ **Análisis por Segmentos**: Permite definir múltiples segmentos de carretera para análisis independiente.
- 📊 **Estadísticas en Tiempo Real**: Muestra conteos y métricas de tráfico actualizadas.
- 💾 **Exportación de Datos**: Guarda los resultados en formato CSV para análisis posteriores.

## 📸 Demostración Visual

### Primera Versión: Detección Básica
![Detección Inicial](trafic_3.jpeg)

*Primera versión mostrando la detección básica de vehículos.*

### Versión Final: Sistema Completo
![Sistema Completo](trafic_2.jpeg)

*Versión final con seguimiento, conteo y análisis de direcciones.*

### Análisis de Semaforización
![Análisis de Semaforización](trafic_4.png)

*Sistema de semaforización inteligente basado en el análisis de flujo vehicular.*

## 🛠️ Requisitos Técnicos

- Python 3.8 o superior
- OpenCV
- Ultralytics YOLOv8
- PyTorch
- Numpy
- Pandas

## 🚀 Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/Joshue-24/Vehicle-Traffic-Analysis.git
cd Vehicle-Traffic-Analysis
```

2. Instala las dependencias:
```bash
pip install ultralytics opencv-python numpy pandas
```

## 🏃‍♂️ Uso

1. Ejecuta el script principal:
```bash
python TRAKING32.py
```

2. Configura los parámetros en el código según sea necesario:
   - Ajusta `modelo.conf` para cambiar el umbral de confianza (0-1)
   - Modifica `modelo.iou` para ajustar la intersección sobre unión
   - Personaliza los segmentos de análisis en el código

## 📁 Estructura del Proyecto

```
Vehicle-Traffic-Analysis/
├── TRAKING32.py       # Script principal de análisis de tráfico
├── 4 segmentos.py     # Versión con 4 puntos de segmentación
├── best.pt           # Modelo YOLOv8 pre-entrenado
├── bytetrack.yaml     # Configuración del tracker
├── detec/            # Módulos de detección
├── Trafic_1.png      # Imagen de portada
├── trafic_2.jpeg     # Captura del sistema completo
└── trafic_3.jpeg     # Captura de la primera versión
```

## 🚦 Sistema de Semaforización Inteligente

### Análisis de Flujo Vehicular
El sistema implementa un esquema de semaforización adaptativo que analiza en tiempo real:
- **Volumen de tráfico** por carril y dirección
- **Tiempos de espera** de los vehículos
- **Patrones de giro** (izquierda, derecha, recto)
- **Densidad vehicular** en cada segmento de la intersección

### Índices de Desempeño
El sistema calcula y optimiza los siguientes índices:
1. **Nivel de Servicio (LOS)**: Evalúa la calidad del flujo vehicular (de A a F)
2. **Tiempo de Espera Promedio**: Por carril y dirección
3. **Tasa de Flujo**: Vehículos por hora por carril
4. **Proporción de Giro**: Porcentaje de vehículos que giran a izquierda/derecha

### Esquema de Giros
El sistema detecta y clasifica los giros en:
- **Giro a la Izquierda**: Vehículos que se desplazan hacia el cuadrante superior izquierdo
- **Giro a la Derecha**: Vehículos que se desplazan hacia el cuadrante inferior derecho
- **Flujo Recto**: Vehículos que mantienen su trayectoria horizontal

### Algoritmo de Control
1. **Detección en Tiempo Real**: Usando YOLOv8 + ByteTrack
2. **Clasificación de Trayectorias**: Análisis de vectores de movimiento
3. **Cálculo de Métricas**: Por carril y dirección
4. **Optimización de Tiempos**: Ajuste dinámico de los semáforos

## 📊 Salida de Datos

El sistema genera un archivo CSV con las siguientes métricas por vehículo detectado:
- ID del vehículo
- Coordenadas (x, y)
- Velocidad estimada (km/h)
- Dirección de movimiento (grados)
- Tipo de giro (izquierda/derecha/recto)
- Tiempo de espera (segundos)
- Carril de origen y destino
- Timestamp (formato ISO 8601)

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Por favor, abre un issue para discutir los cambios que te gustaría realizar.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## ✉️ Contacto

Para consultas o soporte, por favor abre un issue en el repositorio.
