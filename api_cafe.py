import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn
from typing import List, Dict, Any
import os

app = FastAPI(title="API Café Pro - Multi Modelo")

# --- 0. CONFIGURACIÓN ---
CONF_THRESHOLD = 0.15       # Un poco más alto para evitar falsos positivos en modelos rápidos
IOU_BBOX_THRESHOLD = 0.5
MASK_IOU_THRESHOLD = 0.1

# --- CARGA DE MÚLTIPLES MODELOS ---
models = {}

# Función para cargar modelos de forma segura
def load_models():
    print("⏳ Cargando modelos...")
    
    # 1. Modelo Preciso (best.pt)
    if os.path.exists('best.pt'):
        try:
            models['best'] = YOLO('best.pt')
            print("✅ Modelo 'best' (Precisión) cargado.")
        except Exception as e:
            print(f"❌ Error cargando 'best.pt': {e}")
    else:
        print("⚠️ Advertencia: 'best.pt' no encontrado.")

    # 2. Modelo Rápido (fast.pt)
    # IMPORTANTE: Debes subir tu modelo ligero y llamarlo 'fast.pt'
    if os.path.exists('fast.pt'):
        try:
            models['fast'] = YOLO('fast.pt')
            print("✅ Modelo 'fast' (Velocidad) cargado.")
        except Exception as e:
            print(f"❌ Error cargando 'fast.pt': {e}")
    else:
        print("⚠️ Advertencia: 'fast.pt' no encontrado. Se usará 'best.pt' como respaldo.")
        if 'best' in models:
            models['fast'] = models['best'] # Fallback para que no falle la app

load_models()

# --- 1. FUNCIONES DE FILTRADO ---

def polygon_to_mask(polygon, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)

def calculate_mask_iou_optimized(mask1_bool, mask2_bool):
    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    if union == 0: return 0.0
    return intersection / union

def apply_mask_nms(detections: List[Dict], iou_thresh: float, img_shape: tuple) -> List[Dict]:
    if not detections: return []
    # Ordenar por confianza
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
        if 'original_index' in item: del item['original_index']
    return keep

# --- 2. ENDPOINT ---

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: str = Form("best") 
):
    # Selección segura del modelo
    selected_model = models.get(model_id)
    if selected_model is None:
        # Si falla, intentamos el otro
        selected_model = models.get('best') or models.get('fast')
        if selected_model is None:
             return JSONResponse(status_code=500, content={"success": False, "error": "No hay modelos cargados en el servidor."})

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"success": False, "error": "Imagen corrupta."})

        height, width = img.shape[:2]

        # Inferencia
        results = selected_model.predict(
            source=img, 
            conf=CONF_THRESHOLD, 
            iou=IOU_BBOX_THRESHOLD,
            retina_masks=True, 
            verbose=False
        )[0]

        raw_detections = []

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

        final_detections = apply_mask_nms(raw_detections, MASK_IOU_THRESHOLD, (height, width))

        # Estadísticas
        class_counts = {}
        total_detections = len(final_detections)
        
        for det in final_detections:
            c_name = det['class']
            class_counts[c_name] = class_counts.get(c_name, 0) + 1
            det['confidence_percentage'] = round(det['confidence'] * 100, 1)

        summary_list = []
        # Mapeo de colores para el orden (Opcional, ayuda a mantener consistencia)
        priority_order = {'maduro': 1, 'pinton': 2, 'verde': 3, 'sobremaduro': 4, 'seco': 5}
        
        # Ordenar por prioridad de negocio (Maduro primero) o cantidad
        sorted_counts = sorted(class_counts.items(), key=lambda item: priority_order.get(item[0].lower(), 99))

        for name, count in sorted_counts:
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            summary_list.append({
                "class_name": name,
                "count": count,
                "percentage": round(percentage, 1)
            })

        return {
            "success": True,
            "model_used": model_id,
            "total_detections": total_detections,
            "summary": summary_list,
            "detections": final_detections
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)