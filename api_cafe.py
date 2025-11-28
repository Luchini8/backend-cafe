import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse # Importamos esto para errores personalizados
from ultralytics import YOLO
import uvicorn
from typing import List, Dict, Any

app = FastAPI(title="API Detección Café - OpenCV Optimized & Retina")

# --- 0. CONFIGURACIÓN ---
CONF_THRESHOLD = 0.30       # Umbral mínimo de confianza
IOU_BBOX_THRESHOLD = 0.5    # NMS interno de YOLO (Cajas)
MASK_IOU_THRESHOLD = 0.1    # NMS personalizado (Máscaras)

# Cargar modelo
try:
    model = YOLO('best.pt')
    print("✅ Modelo YOLOv8 cargado exitosamente (Modo OpenCV + Retina Masks).")
except Exception as e:
    print(f"❌ Error crítico cargando el modelo: {e}")
    model = None

# --- 1. FUNCIONES DE FILTRADO OPTIMIZADAS ---

def polygon_to_mask(polygon, img_shape):
    """Convierte coordenadas de polígono a una máscara binaria (bitmap)."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)

def calculate_mask_iou_optimized(mask1_bool, mask2_bool):
    """Calcula IoU usando operaciones booleanas (mucho más rápido)."""
    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    if union == 0: return 0.0
    return intersection / union

def apply_mask_nms(detections: List[Dict], iou_thresh: float, img_shape: tuple) -> List[Dict]:
    """
    NMS basado en máscaras con caché para alto rendimiento.
    """
    if not detections: return []

    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep = []
    
    mask_cache = {}

    for i, current_det in enumerate(detections):
        if i not in mask_cache:
            mask_cache[i] = polygon_to_mask(current_det['polygon'], img_shape)
        
        curr_mask = mask_cache[i]
        should_keep = True

        for kept_item in keep:
            kept_idx = kept_item['original_index']
            prev_mask = mask_cache[kept_idx]
            
            iou = calculate_mask_iou_optimized(curr_mask, prev_mask)
            
            if iou > iou_thresh:
                should_keep = False
                break
        
        if should_keep:
            current_det['original_index'] = i
            keep.append(current_det)

    for item in keep:
        if 'original_index' in item:
            del item['original_index']
            
    return keep

# --- 2. ENDPOINT ---

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "El modelo no está cargado en el servidor."}
        )

    try:
        # A. Leer imagen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Imagen corrupta o formato no soportado."}
            )

        height, width = img.shape[:2]

        # B. Inferencia
        results = model.predict(
            source=img, 
            conf=CONF_THRESHOLD, 
            iou=IOU_BBOX_THRESHOLD,
            retina_masks=True,
            verbose=False
        )[0]

        raw_detections = []

        # C. Extracción
        if results.masks is not None:
            polygons = results.masks.xy 
            boxes = results.boxes
            names = results.names

            for i, poly in enumerate(polygons):
                if len(poly) < 3: continue

                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                label = names[cls_id]

                raw_detections.append({
                    "class": label,
                    "confidence": round(conf, 4),
                    "polygon": poly.tolist()
                })

        # D. NMS Máscaras
        final_detections = apply_mask_nms(raw_detections, MASK_IOU_THRESHOLD, (height, width))

        # E. Estadísticas
        class_counts = {}
        total_detections = len(final_detections)
        
        for det in final_detections:
            c_name = det['class']
            class_counts[c_name] = class_counts.get(c_name, 0) + 1

        summary_list = []
        sorted_counts = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)

        for name, count in sorted_counts:
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            summary_list.append({
                "class_name": name,
                "count": count,
                "percentage": round(percentage, 1)
            })

        # F. Respuesta JSON (CORREGIDA)
        return {
            "success": True,              # <--- ¡IMPORTANTE! Esto faltaba
            "status": "success",          # Mantenemos ambos por compatibilidad
            "total_detections": total_detections,
            "summary": summary_list,      
            "detections": final_detections
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Devolvemos un JSON con success: False en caso de error
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)