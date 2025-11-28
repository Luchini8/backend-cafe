import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn
from typing import List, Dict, Any

app = FastAPI(title="API Café Pro - Multi Modelo")

# --- 0. CONFIGURACIÓN ---
CONF_THRESHOLD = 0.10
IOU_BBOX_THRESHOLD = 0.5
MASK_IOU_THRESHOLD = 0.1

# --- CARGA DE MÚLTIPLES MODELOS ---
# Cargamos un diccionario de modelos para permitir al usuario elegir.
# 'best': Tu modelo entrenado de alta precisión (88MB).
# 'fast': Un modelo nano estándar para pruebas rápidas (o una versión cuantizada).
models = {}

try:
    print("⏳ Cargando modelos...")
    models['best'] = YOLO('best.pt')  # Tu modelo principal
    # Si no tienes un segundo modelo entrenado, usa 'yolov8n-seg.pt' como placeholder rápido
    # O simplemente carga el mismo 'best.pt' con otro nombre si solo tienes uno por ahora.
    models['fast'] = YOLO('yolov8n-seg.pt') 
    print("✅ Modelos 'best' y 'fast' cargados exitosamente.")
except Exception as e:
    print(f"⚠️ Advertencia: Hubo un problema cargando los modelos: {e}")
    # Fallback: intentar cargar al menos uno si el otro falla
    if 'best' not in models and 'fast' not in models:
        print("❌ Error crítico: No hay modelos disponibles.")

# --- 1. FUNCIONES DE FILTRADO OPTIMIZADAS ---

def polygon_to_mask(polygon, img_shape):
    """Convierte coordenadas de polígono a una máscara binaria."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)

def calculate_mask_iou_optimized(mask1_bool, mask2_bool):
    """Calcula IoU usando operaciones booleanas."""
    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    if union == 0: return 0.0
    return intersection / union

def apply_mask_nms(detections: List[Dict], iou_thresh: float, img_shape: tuple) -> List[Dict]:
    """NMS basado en máscaras con caché."""
    if not detections: return []

    # Ordenar por confianza descendente
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

    # Limpieza final
    for item in keep:
        if 'original_index' in item: del item['original_index']
            
    return keep

# --- 2. ENDPOINT PRINCIPAL ---

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: str = Form("best") # Recibe el parámetro del usuario (default: best)
):
    # 1. Selección del Modelo
    selected_model = models.get(model_id)
    if selected_model is None:
        # Si piden un modelo que no existe, usamos el 'best' por defecto
        selected_model = models.get('best')
        if selected_model is None:
             return JSONResponse(status_code=500, content={"success": False, "error": "Modelos no disponibles."})

    try:
        # A. Leer imagen con OpenCV
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"success": False, "error": "Imagen inválida."})

        height, width = img.shape[:2]

        # B. Inferencia
        # Usamos retina_masks=True para bordes suaves (calidad visual "profesional")
        results = selected_model.predict(
            source=img, 
            conf=CONF_THRESHOLD, 
            iou=IOU_BBOX_THRESHOLD,
            retina_masks=True, 
            verbose=False
        )[0]

        raw_detections = []

        # C. Extracción de datos
        if results.masks is not None:
            polygons = results.masks.xy 
            boxes = results.boxes
            names = results.names

            for i, poly in enumerate(polygons):
                if len(poly) < 3: continue

                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                label = names[cls_id]

                # Guardamos datos crudos
                raw_detections.append({
                    "class": label,
                    "confidence": round(conf, 4), # Decimal 0.9521
                    "polygon": poly.tolist()
                })

        # D. NMS de Máscaras (Filtrado avanzado)
        final_detections = apply_mask_nms(raw_detections, MASK_IOU_THRESHOLD, (height, width))

        # E. Cálculo de Estadísticas para la Interfaz
        class_counts = {}
        total_detections = len(final_detections)
        
        # Agregar campo de porcentaje legible para la App
        for det in final_detections:
            c_name = det['class']
            class_counts[c_name] = class_counts.get(c_name, 0) + 1
            
            # FIX: Agregamos el porcentaje directo para que la App no tenga que calcularlo
            # Multiplicamos por 100 y redondeamos a 1 decimal (ej: 95.2)
            det['confidence_percentage'] = round(det['confidence'] * 100, 1)

        # Resumen ordenado para el "Dashboard" de la App
        summary_list = []
        sorted_counts = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)

        for name, count in sorted_counts:
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            summary_list.append({
                "class_name": name,
                "count": count,
                "percentage": round(percentage, 1) # Porcentaje del total de granos
            })

        # F. Respuesta JSON Final
        return {
            "success": True,
            "model_used": model_id,
            "total_detections": total_detections,
            "summary": summary_list,       # Para la tabla de estadísticas
            "detections": final_detections # Para dibujar los polígonos
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)