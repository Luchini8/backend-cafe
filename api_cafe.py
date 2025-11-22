import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from collections import Counter
from typing import List, Dict, Any

app = FastAPI(title="Coffee Bean Detection API (OpenCV Optimized)")

# ---------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# ---------------------------------------------------------
MODEL_PATH = "best.pt"  # Asegúrate de que este archivo exista en el directorio
try:
    # Cargamos el modelo al iniciar
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"ERROR CRÍTICO: No se pudo cargar el modelo en {MODEL_PATH}. {e}")
    model = None

# Umbrales definidos en los requerimientos
CONF_THRESHOLD = 0.10
IOU_BOX_THRESHOLD = 0.5
IOU_MASK_THRESHOLD = 0.1

# ---------------------------------------------------------
# UTILIDADES DE FILTRADO (NMS & IOU)
# ---------------------------------------------------------

def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calcula la Intersección sobre Unión (IoU) de dos máscaras binarias.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def polygon_to_mask(polygon, img_shape):
    """
    Convierte coordenadas de polígono a una máscara binaria (bitmap).
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    # Convertir polígono a formato entero para cv2.fillPoly
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)

def apply_mask_nms(detections: List[Dict], iou_thresh: float, img_shape: tuple) -> List[Dict]:
    """
    Aplica Non-Maximum Suppression basado en la superposición de Máscaras.
    Prioriza detecciones con mayor confianza.
    """
    if not detections:
        return []

    # Ordenar por confianza descendente
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep = []
    
    # Cache de máscaras binarias para evitar regenerarlas
    mask_cache = {}

    for i, current_det in enumerate(detections):
        current_poly = current_det['polygon']
        
        # Generar o recuperar máscara
        if i not in mask_cache:
            mask_cache[i] = polygon_to_mask(current_poly, img_shape)
        
        curr_mask = mask_cache[i]
        should_keep = True

        for j in keep:
            # Recuperar máscara de la detección ya aceptada
            prev_mask = mask_cache[j['original_index']]
            
            # Calcular IoU
            iou = calculate_mask_iou(curr_mask, prev_mask)
            
            if iou > iou_thresh:
                should_keep = False
                break
        
        if should_keep:
            # Guardamos el índice original para referencia en caché
            current_det['original_index'] = i
            keep.append(current_det)

    # Limpiar campo temporal
    for item in keep:
        del item['original_index']
        
    return keep

# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")

    try:
        # 1. CAMBIO A OPENCV PURO (Lectura de bytes directa)
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        
        # Decodificar directamente a BGR (OpenCV default)
        # Esto es CRÍTICO para coincidir con el entrenamiento si se usó cv2
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="El archivo no es una imagen válida.")

        height, width = img.shape[:2]

        # 2. INFERENCIA
        # Ultralytics maneja la conversión interna, pero al pasar numpy BGR,
        # respetará los canales si el modelo fue entrenado así o estandarizado.
        results = model.predict(
            source=img, 
            conf=CONF_THRESHOLD, 
            iou=IOU_BOX_THRESHOLD,
            retina_masks=True # Mejora la calidad de los polígonos
        )[0]

        raw_detections = []

        # Verificar si hay máscaras detectadas
        if results.masks is not None:
            # Extraer datos
            # results.masks.xy devuelve una lista de arrays con coordenadas de polígonos
            polygons = results.masks.xy 
            boxes = results.boxes
            names = results.names

            for i, poly in enumerate(polygons):
                # Si el polígono está vacío o muy pequeño, saltar
                if len(poly) < 3: 
                    continue

                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                label = names[cls_id]

                raw_detections.append({
                    "class": label,
                    "confidence": round(conf, 4),
                    "polygon": poly.tolist() # Convertir numpy array a lista para JSON
                })

        # 3. FILTRADO (Mask NMS)
        # Mantenemos la lógica de filtrar duplicados que se solapan físicamente
        final_detections = apply_mask_nms(raw_detections, IOU_MASK_THRESHOLD, (height, width))

        # 4. CÁLCULO DE ESTADÍSTICAS MEJORADAS
        total_count = len(final_detections)
        avg_conf = 0.0
        
        # Conteo de clases
        class_counts_raw = Counter(d['class'] for d in final_detections)
        
        # Ordenar conteos de MAYOR a MENOR
        sorted_class_counts = dict(sorted(class_counts_raw.items(), key=lambda item: item[1], reverse=True))

        # Calcular porcentajes ordenados
        class_percentages = {}
        if total_count > 0:
            avg_conf = sum(d['confidence'] for d in final_detections) / total_count
            for cls_name, count in sorted_class_counts.items():
                percent = (count / total_count) * 100
                class_percentages[cls_name] = f"{percent:.1f}%"

        # 5. RESPUESTA JSON
        return JSONResponse(content={
            "success": True,
            "total_detections": total_count,
            "avg_confidence": round(avg_conf, 4),
            "class_counts": sorted_class_counts,     # Requerimiento: Ordenado descendente
            "class_percentages": class_percentages,  # Requerimiento: Nuevo campo
            "detections": final_detections
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500, 
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    # Ejecución directa para pruebas
    uvicorn.run(app, host="0.0.0.0", port=8000)