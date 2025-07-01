import os
import cv2

# --- Configuración inicial ---
personas = ['personaA', 'personaB']  # Reemplaza con tus nombres reales
emociones = ['triste', 'pensativo', 'con_ira', 'cansado', 'sorprendido', 'riendo', 'alegre']
conjuntos = ['train', 'val']  # Ahora procesa ambos conjuntos

# --- Funciones de apoyo ---
def detectar_y_recortar_cara(imagen_path):
    """Detecta y recorta caras usando Haar Cascade."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    imagen = cv2.imread(imagen_path)
    
    if imagen is None:
        print(f"Error leyendo: {imagen_path}")
        return None
    
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    caras = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(caras) == 0:
        print(f"No se detectaron caras en: {imagen_path}")
        return None
    
    x, y, w, h = caras[0]  # Tomamos la primera cara detectada
    cara = imagen[y:y+h, x:x+w]
    return cv2.resize(cara, (224, 224))  # Normalizamos tamaño

def aumentar_imagen(imagen):
    """Aplica aumentaciones aleatorias."""
    import random
    # Flip horizontal (50% de probabilidad)
    if random.random() > 0.5:
        imagen = cv2.flip(imagen, 1)
    # Puedes añadir más aumentaciones aquí (rotación, brillo, etc.)
    return imagen

# --- Procesamiento principal ---
def procesar_todos_conjuntos():
    for conjunto in conjuntos:  # Procesa train y val
        for persona in personas:
            for emocion in emociones:
                ruta_carpeta = os.path.join("data", conjunto, persona, emocion)
                
                if not os.path.exists(ruta_carpeta):
                    print(f"⚠️ Carpeta no encontrada: {ruta_carpeta}")
                    continue
                
                print(f"\nProcesando: {ruta_carpeta}...")
                for img_name in os.listdir(ruta_carpeta):
                    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    
                    full_path = os.path.join(ruta_carpeta, img_name)
                    
                    # 1. Detección y recorte de cara
                    cara_recortada = detectar_y_recortar_cara(full_path)
                    if cara_recortada is not None:
                        cv2.imwrite(full_path, cara_recortada)  # Sobreescribe original
                    
                    # 2. Aumentación (opcional)
                    try:
                        imagen = cv2.imread(full_path)
                        if imagen is not None:
                            imagen_aug = aumentar_imagen(imagen)
                            nombre_aug = os.path.splitext(img_name)[0] + "_proc.jpg"
                            cv2.imwrite(os.path.join(ruta_carpeta, nombre_aug), imagen_aug)
                    except Exception as e:
                        print(f"❌ Error en {img_name}: {str(e)}")

# --- Ejecución ---
if __name__ == "__main__":
    print("🚀 Iniciando procesamiento de TODAS las carpetas (train + val)...")
    procesar_todos_conjuntos()
    print("✅ ¡Procesamiento completado!")
