# 1) Importaciones y configuración
import os
import cv2
import logging
import time
import numpy as np
import pandas as pd
from mtcnn import MTCNN
from tqdm import tqdm
from tensorflow.keras import Sequential, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import load_model # Importar para cargar el modelo entrenado

logging.basicConfig(level=logging.INFO)

personas = ['personaA', 'personaB']
emociones = [
    'alegre','triste','pensativo',
    'con_ira','cansado','sorprendido','riendo'
]

# IMPORTANTE: Ajusta estas rutas a la ubicación de tus carpetas 'data/train' y 'data/test'
# Si tu script está en el mismo directorio que la carpeta 'data', usa 'data/train' y 'data/test'.
# Si 'data' está en un directorio padre, podrías necesitar '../data/train', o una ruta absoluta.
base_train = 'data/train' # Asumiendo que la carpeta 'data' está en el mismo directorio que tu script Python
base_test = 'data/test'   # Asumiendo la carpeta 'data/test' para inferencia

# 2) Detector MTCNN y función de recorte
detector = MTCNN()

def detectar_y_recortar_cara(ruta_img,
                             max_size=2000,
                             max_time=3.0,
                             target_size=(224,224)):
    img = cv2.imread(ruta_img)
    if img is None:
        logging.warning(f"No se pudo cargar la imagen: {ruta_img}")
        return None
    # Si la imagen ya tiene el tamaño deseado, la devuelve directamente
    if img.shape[0:2] == target_size: return img
    # Si la imagen es demasiado grande, la ignora
    if img.shape[0] > max_size or img.shape[1] > max_size:
        logging.warning(f"Imagen demasiado grande para procesar: {ruta_img}")
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    start = time.time()
    resultados = detector.detect_faces(rgb)
    # Si no se detectan caras o la detección excede el tiempo máximo, ignora la imagen
    if not resultados or time.time()-start > max_time:
        logging.warning(f"No se detectaron caras o el tiempo de detección excedió el límite para: {ruta_img}")
        return None
    # Extrae las coordenadas de la primera cara detectada
    x,y,w,h = resultados[0]['box']
    # Asegura que las coordenadas no sean negativas
    x, y = max(0,x), max(0,y)
    # Recorta la región de interés (ROI)
    roi = img[y:y+h, x:x+w]
    # Redimensiona la ROI al tamaño objetivo
    return cv2.resize(roi, target_size)

# 3) Preprocesa in-place todas las imágenes con MTCNN
# Esta función modificará tus imágenes originales. Asegúrate de tener copias de seguridad si es necesario.
def procesar_train():
    logging.info("Iniciando preprocesamiento de imágenes de entrenamiento...")
    for p in personas:
        for e in emociones:
            carpeta = os.path.join(base_train, p, e)
            if not os.path.isdir(carpeta):
                logging.warning(f"Directorio no encontrado: {carpeta}. Saltando.")
                continue
            # Itera sobre los archivos en la carpeta con una barra de progreso
            for f in tqdm(os.listdir(carpeta), desc=f"Procesando {p}/{e}"):
                # Verifica si el archivo es una imagen
                if not f.lower().endswith(('.jpg','jpeg','png')):
                    continue
                ruta = os.path.join(carpeta, f)
                rec = detectar_y_recortar_cara(ruta)
                if rec is not None:
                    try:
                        # Guarda la imagen recortada y redimensionada en la misma ruta, sobrescribiendo la original
                        cv2.imwrite(ruta, rec)
                    except Exception as err:
                        logging.error(f"Error al guardar la imagen procesada {ruta}: {err}")
                else:
                    # Opcionalmente, puedes mover o registrar imágenes que no pudieron ser procesadas
                    logging.info(f"No se pudo procesar la imagen (no se detectó cara o error): {ruta}")
    logging.info("Preprocesamiento de entrenamiento completado.")

# Ejecuta el preprocesamiento de las imágenes de entrenamiento
procesar_train()

# 4) Construye DataFrame con rutas y etiquetas
logging.info("Construyendo DataFrame con rutas y etiquetas...")
filepaths, labels = [], []
for p in personas:
    for e in emociones:
        carpeta = os.path.join(base_train, p, e)
        if not os.path.isdir(carpeta): continue
        for f in os.listdir(carpeta):
            if not f.lower().endswith(('.jpg','jpeg','png')):
                continue
            filepaths.append(os.path.join(carpeta, f))
            labels.append(f"{p}_{e}")

df = pd.DataFrame({'filename': filepaths, 'class': labels})
logging.info(f"DataFrame creado con {len(df)} entradas.")

# 5) Split estratificado en train/val
logging.info("Realizando split estratificado de datos...")
df_train, df_val = train_test_split(
    df, test_size=0.2,
    stratify=df['class'],
    random_state=42
)
logging.info(f"Muestras de entrenamiento: {len(df_train)}, Muestras de validación: {len(df_val)}")

# 6) Generadores con augmentations moderadas
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normaliza los píxeles a un rango de 0 a 1
    rotation_range=30,        # Rota imágenes aleatoriamente hasta 30 grados
    width_shift_range=0.2,    # Desplaza imágenes horizontalmente
    height_shift_range=0.2,   # Desplaza imágenes verticalmente
    zoom_range=0.2,           # Aplica zoom aleatorio
    horizontal_flip=True,     # Voltea imágenes horizontalmente
    brightness_range=(0.8,1.2), # Ajusta el brillo aleatoriamente
    fill_mode='nearest'       # Rellena los puntos nuevos creados por las transformaciones
)
val_datagen = ImageDataGenerator(rescale=1./255) # Solo normaliza para validación

logging.info("Configurando generadores de imágenes...")
train_gen = train_datagen.flow_from_dataframe(
    df_train, x_col='filename', y_col='class',
    target_size=(224,224), batch_size=batch_size,
    class_mode='categorical', shuffle=True
)
val_gen = val_datagen.flow_from_dataframe(
    df_val, x_col='filename', y_col='class',
    target_size=(224,224), batch_size=batch_size,
    class_mode='categorical', shuffle=False
)

# 7) Cálculo de class_weight
logging.info("Calculando pesos de clase...")
weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weight_dict = dict(enumerate(weights))
logging.info(f"Pesos de clase calculados: {class_weight_dict}")

# 8) Definición de la CNN (ligeramente mayor capacidad)
num_classes = df['class'].nunique() # Número único de clases (ej: personaA_alegre, personaB_triste, etc.)

model = Sequential([
    layers.Conv2D(32, 3, activation='relu',
                  input_shape=(224,224,3)), # Primera capa convolucional
    layers.MaxPooling2D(), layers.Dropout(0.3), # Capa de MaxPooling y Dropout para regularización

    layers.Conv2D(64, 3, activation='relu'),    # Segunda capa convolucional
    layers.MaxPooling2D(), layers.Dropout(0.3),

    layers.Conv2D(128, 3, activation='relu'),   # Tercera capa convolucional
    layers.GlobalAveragePooling2D(),            # Reduce las dimensiones espaciales

    layers.Dense(128, activation='relu'),      # Capa densa con activación ReLU
    layers.Dropout(0.4),

    layers.Dense(num_classes, activation='softmax') # Capa de salida con activación Softmax para clasificación multiclase
])
logging.info("Modelo CNN definido.")
model.summary() # Muestra un resumen de la arquitectura del modelo

# 9) Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
early_stop = EarlyStopping(
    monitor='val_loss',         # Monitorea la pérdida de validación
    patience=8,                 # Espera 8 épocas sin mejora antes de detener
    restore_best_weights=True   # Restaura los pesos del modelo con la mejor pérdida de validación
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',         # Monitorea la pérdida de validación
    factor=0.5,                 # Reduce la tasa de aprendizaje a la mitad
    patience=4,                 # Espera 4 épocas sin mejora antes de reducir
    min_lr=1e-6                 # Tasa de aprendizaje mínima
)
checkpoint = ModelCheckpoint(
    'mejor_modelo.h5',          # El modelo se guardará en el mismo directorio que tu script
    monitor='val_loss',         # Monitorea la pérdida de validación
    save_best_only=True         # Guarda solo el mejor modelo
)
logging.info("Callbacks configurados.")

# 10) Compilación con learning rate fijo inicial
from tensorflow.keras.optimizers import Adam
opt = Adam(learning_rate=1e-3) # Optimizador Adam con una tasa de aprendizaje inicial de 0.001

model.compile(
    optimizer=opt,
    loss='categorical_crossentropy', # Función de pérdida para clasificación multiclase
    metrics=['accuracy']             # Métrica a monitorear durante el entrenamiento
)
logging.info("Modelo compilado.")

# 11) Entrenamiento (20-30 épocas)
logging.info("Iniciando entrenamiento del modelo...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30, # Número máximo de épocas de entrenamiento
    callbacks=[early_stop, checkpoint, reduce_lr], # Callbacks a utilizar
    class_weight=class_weight_dict, # Pesos de clase para manejar desequilibrio
    verbose=2 # 0 = silencioso, 1 = barra de progreso, 2 = una línea por época
)
logging.info("Entrenamiento completado.")

# 12) Mapeo índice → etiqueta
idx_to_label = {v: k for k, v in train_gen.class_indices.items()}
logging.info("Mapeo de índice a etiqueta creado.")

# Cargar el mejor modelo después del entrenamiento
try:
    model = load_model('mejor_modelo.h5')
    logging.info("Mejor modelo cargado exitosamente.")
except Exception as e:
    logging.error(f"Error al cargar el mejor modelo: {e}. Asegúrate de que 'mejor_modelo.h5' exista.")

# 13) Inferencia sobre nuevas imágenes
def predecir_emocion_y_persona(ruta_img):
    face = detectar_y_recortar_cara(ruta_img)
    if face is None:
        print(f"No se detectó cara o hubo un error de procesamiento en {ruta_img}")
        return None, None # Retorna None para ambos si no se detecta cara
    x = face.astype('float32') / 255.0 # Normaliza los píxeles
    x = np.expand_dims(x, axis=0)       # Agrega una dimensión de lote (batch)
    probs = model.predict(x, verbose=0)[0] # Realiza la predicción, verbose=0 para una salida limpia
    idx = np.argmax(probs)              # Obtiene el índice de la clase con mayor probabilidad
    label = idx_to_label[idx]           # Mapea el índice a la etiqueta de la clase
    person = label.split('_')[0]
    emotion = label.split('_')[1]
    return person, emotion

print("\n--- Inferencia en 'data/test' ---")
# Verifica si el directorio de prueba existe
if os.path.isdir(base_test):
    # Filtra solo los archivos de imagen
    test_images = [f for f in os.listdir(base_test) if f.lower().endswith(('.jpg','jpeg','png'))]
    if test_images:
        # Itera sobre las imágenes de prueba con una barra de progreso
        for f in tqdm(test_images, desc="Realizando inferencia"):
            person_detected, emotion_detected = predecir_emocion_y_persona(os.path.join(base_test, f))
            if person_detected and emotion_detected:
                print(f"{os.path.basename(f)} -> Persona: {person_detected}, Emoción: {emotion_detected}")
    else:
        print(f"No se encontraron imágenes en '{base_test}'.")
else:
    print(f"Directorio de prueba '{base_test}' no encontrado. No se realizará inferencia.")