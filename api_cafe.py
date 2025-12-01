import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn
from typing import List, Dict
import os

app = FastAPI(title="API Café Pro - Solo Fast")

# --- 0. CONFIGURACIÓN ---
# Umbral de confianza (ajustable si el modelo fast es muy "tímido" o muy "lanzado")
CONF_THRESHOLD = 0.20 
MASK_IOU_THRESHOLD = 0.1

# --- CARGA DEL MODELO ---
model = None

def load_model():
    global model
    print("⏳ Cargando modelo fast.pt...")
    
    # Buscamos solo el modelo 'fast.pt'
    if os.path.exists('fast.pt'):
        try:
            model = YOLO('fast.pt')
            print("✅ Modelo 'fast.pt' cargado correctamente.")
        except Exception as e:
            print(f"❌ Error crítico cargando 'fast.pt': {e}")
    else:
        print("⚠️ ERROR: No se encuentra el archivo 'fast.pt' en la raíz.")

# Cargar al iniciar
load_model()

# --- 1. FUNCIONES DE FILTRADO (NMS) ---

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
async def predict(file: UploadFile = File(...)):
    # Verificación de seguridad del modelo
    if model is None:
        return JSONResponse(
            status_code=500, 
            content={"success": False, "error": "El modelo no está cargado en el servidor."}
        )

    try:
        # Leer imagen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                status_code=400, 
                content={"success": False, "error": "El archivo no es una imagen válida."}
            )

        height, width = img.shape[:2]

        # Inferencia (Solo con modelo fast)
        results = model.predict(
            source=img, 
            conf=CONF_THRESHOLD, 
            retina_masks=True, 
            verbose=False
        )[0]

        raw_detections = []

        # Procesar resultados si hay detecciones
        if results.masks is not None:
            polygons = results.masks.xy 
            boxes = results.boxes
            names = results.names

            for i, poly in enumerate(polygons):
                if len(poly) < 3: continue # Ignorar polígonos rotos

                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                label = names[cls_id]

                raw_detections.append({
                    "class": label,
                    "confidence": round(conf, 4),
                    "polygon": poly.tolist()
                })

        # Aplicar filtro de superposición
        final_detections = apply_mask_nms(raw_detections, MASK_IOU_THRESHOLD, (height, width))
        total_detections = len(final_detections)

        # --- LÓGICA DE VALIDACIÓN DE "IMAGEN SIN SENTIDO" ---
        # Si después del filtro no queda NINGÚN grano, asumimos que la imagen no es de café
        # o está muy borrosa/lejana.
        if total_detections == 0:
            return {
                "success": False,
                "is_coffee": False, # Bandera para que la App sepa qué mensaje mostrar
                "error": "No se detectaron granos. Por favor, asegúrate de subir una foto clara de frutos de café.",
                "total_detections": 0,
                "summary": [],
                "detections": []
            }

        # Estadísticas
        class_counts = {}
        for det in final_detections:
            c_name = det['class']
            class_counts[c_name] = class_counts.get(c_name, 0) + 1

        summary_list = []
        # Orden personalizado para mostrar (según tesis: Inmaduro, Pintón, Maduro, Sobremaduro)
        # Ajusta estos nombres EXACTAMENTE a como salen de tu modelo (ej: 'verde', 'maduro')
        priority_order = {'maduro': 1, 'pinton': 2, 'verde': 3, 'sobremaduro': 4, 'seco': 5}
        
        sorted_counts = sorted(class_counts.items(), key=lambda item: priority_order.get(item[0].lower(), 99))

        for name, count in sorted_counts:
            percentage = (count / total_detections * 100)
            summary_list.append({
                "class_name": name,
                "count": count,
                "percentage": round(percentage, 1)
            })

        return {
            "success": True,
            "is_coffee": True,
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