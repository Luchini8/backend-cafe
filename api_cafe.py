from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2
from typing import List, Dict

# Inicializar FastAPI
app = FastAPI(title="Coffee Cherry Detection API")

# Configurar CORS para permitir peticiones desde Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica el dominio de tu app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo YOLOv8n-seg al iniciar
print("Cargando modelo YOLOv8n-seg...")
model = YOLO("best.pt")
print("✅ Modelo cargado exitosamente")

# Mapeo de clases
CLASS_NAMES = {
    0: "maduro",
    1: "pinton",
    2: "seco",
    3: "sobremaduro",
    4: "verde"
}

# Configuración de umbrales (replicando el código de Colab)
CONF_THRESHOLD = 0.10  # Umbral de confianza mínimo para MOSTRAR una predicción
IOU_BBOX_THRESHOLD = 0.5  # Umbral IoU para NMS interno basado en BBox
MASK_IOU_THRESHOLD = 0.1  # Umbral IoU para NMS basado en máscaras


# --- Funciones auxiliares para NMS basado en máscaras ---
def calculate_mask_iou(mask1_coords, mask2_coords, img_shape):
    """Calcula IoU entre dos máscaras basándose en sus coordenadas de polígono"""
    mask1_img = np.zeros(img_shape[:2], dtype=np.uint8)
    mask2_img = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask1_img, [mask1_coords.astype(np.int32)], 1)
    cv2.fillPoly(mask2_img, [mask2_coords.astype(np.int32)], 1)
    intersection = np.logical_and(mask1_img, mask2_img).sum()
    union = np.logical_or(mask1_img, mask2_img).sum()
    if union == 0:
        return 0.0
    return intersection / union


def apply_mask_nms(detections, img_shape, iou_threshold):
    """Aplica Non-Maximum Suppression basado en IoU de máscaras"""
    if not detections:
        return []
    
    # Ordenar por confianza descendente
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep_indices = []
    suppressed = np.zeros(len(detections), dtype=bool)
    
    for i in range(len(detections)):
        if suppressed[i]:
            continue
        keep_indices.append(i)
        
        # Suprimir detecciones con alto IoU con esta detección
        for j in range(i + 1, len(detections)):
            if suppressed[j]:
                continue
            iou = calculate_mask_iou(
                detections[i]['mask_coords'],
                detections[j]['mask_coords'],
                img_shape
            )
            if iou >= iou_threshold:
                suppressed[j] = True
    
    final_detections = [detections[idx] for idx in keep_indices]
    return final_detections


@app.get("/")
async def root():
    """Endpoint de prueba"""
    return {
        "message": "Coffee Cherry Detection API",
        "status": "online",
        "model": "YOLOv8n-seg",
        "classes": CLASS_NAMES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recibe imagen comprimida y devuelve JSON optimizado con:
    - Coordenadas de polígonos de segmentación
    - Clases y confianzas
    - Conteo por clase
    
    Replica la lógica exacta del código de Colab con NMS basado en máscaras
    """
    try:
        # Leer imagen desde el archivo subido
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convertir a numpy array (OpenCV format)
        image_np = np.array(image)
        if len(image_np.shape) == 2:  # Grayscale
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        img_shape = image_np.shape
        
        # --- PASO 1: Predicción inicial con conf muy bajo (como en Colab) ---
        # conf=0.01 para capturar candidatos, iou=0.5 para NMS de BBox
        results = model.predict(image_np, conf=0.01, iou=IOU_BBOX_THRESHOLD, verbose=False)
        result = results[0]
        
        # --- PASO 2: Recopilar detecciones y filtrar por confianza >= 0.10 ---
        raw_detections = []
        
        if result.masks is not None and result.boxes is not None:
            for i in range(len(result.boxes)):
                confidence = float(result.boxes.conf[i])
                
                # Filtrado inicial por confianza (como en Colab)
                if confidence >= CONF_THRESHOLD:
                    cls_id = int(result.boxes.cls[i])
                    
                    # Asegurar que hay máscara para esta detección
                    if i < len(result.masks.xy):
                        bbox = result.boxes.xyxy[i].cpu().numpy().astype(int)
                        mask_coords = result.masks.xy[i].astype(int)
                        
                        raw_detections.append({
                            'confidence': confidence,
                            'cls_id': cls_id,
                            'class_name': CLASS_NAMES[cls_id],
                            'bbox': bbox,
                            'mask_coords': mask_coords
                        })
        
        # --- PASO 3: Aplicar NMS basado en IoU de máscaras (como en Colab) ---
        final_detections = apply_mask_nms(raw_detections, img_shape, MASK_IOU_THRESHOLD)
        
        # --- PASO 4: Preparar respuesta optimizada ---
        detections = []
        class_counts = {name: 0 for name in CLASS_NAMES.values()}
        
        for det in final_detections:
            class_id = det['cls_id']
            class_name = det['class_name']
            confidence = det['confidence']
            mask_coords = det['mask_coords']
            bbox = det['bbox']
            
            # Incrementar contador de clase
            class_counts[class_name] += 1
            
            # Convertir coordenadas de máscara a lista para JSON
            polygon = [[float(x), float(y)] for x, y in mask_coords]
            
            # Añadir detección (JSON compacto)
            detections.append({
                "class_id": class_id,
                "class": class_name,
                "confidence": round(confidence, 3),
                "polygon": [[round(x, 1), round(y, 1)] for x, y in polygon],
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            })
        
        # Calcular estadísticas
        total_detections = len(detections)
        avg_confidence = round(
            sum(d["confidence"] for d in detections) / total_detections, 3
        ) if total_detections > 0 else 0.0
        
        # Respuesta final optimizada
        response = {
            "success": True,
            "image_size": {
                "width": image.width,
                "height": image.height
            },
            "total_detections": total_detections,
            "avg_confidence": avg_confidence,
            "class_counts": class_counts,
            "detections": detections
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.get("/health")
async def health_check():
    """Verificar que el servidor y modelo están funcionando"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)