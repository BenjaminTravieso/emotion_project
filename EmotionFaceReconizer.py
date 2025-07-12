# 1) Clone el repo y prepare el entorno
#!git clone https://github.com/BenjaminTravieso/emotion_project
#%cd emotion_project
#!pip install mtcnn opencv-python-headless tensorflow pandas

#aqui esta el codigo del proyecto pasado, a lo mejor tengas que cambiar la direccion de la carpeta aqui, para que funcione, porque yo lo probe solo en el colab
#posdata trabajalo a parte porque puedes alterar las imagenes en el repo que son las que usa en el colab este proyecto
#poposdata los comentarios de arriba no lo debes de ejecutar esos eran para descargar el repo y las dependencias en el colab

# 2) Imports y configuración
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

logging.basicConfig(level=logging.INFO)

personas  = ['personaA', 'personaB']
emociones = [
    'alegre','triste','pensativo',
    'con_ira','cansado','sorprendido','riendo'
]
base_train = 'data/train'

# 3) Detector MTCNN y función de recorte
detector = MTCNN()

def detectar_y_recortar_cara(ruta_img,
                             max_size=2000,
                             max_time=3.0,
                             target_size=(224,224)):
    img = cv2.imread(ruta_img)
    if img is None: return None
    if img.shape[0:2] == target_size: return img
    if img.shape[0] > max_size or img.shape[1] > max_size:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    start = time.time()
    resultados = detector.detect_faces(rgb)
    if not resultados or time.time()-start > max_time:
        return None
    x,y,w,h = resultados[0]['box']
    x, y = max(0,x), max(0,y)
    roi = img[y:y+h, x:x+w]
    return cv2.resize(roi, target_size)

# 4) Preprocesa in-place todas las imágenes con MTCNN
def procesar_train():
    for p in personas:
        for e in emociones:
            carpeta = os.path.join(base_train, p, e)
            if not os.path.isdir(carpeta): continue
            for f in os.listdir(carpeta):
                if not f.lower().endswith(('.jpg','jpeg','png')):
                    continue
                ruta = os.path.join(carpeta, f)
                rec = detectar_y_recortar_cara(ruta)
                if rec is not None:
                    cv2.imwrite(ruta, rec)

procesar_train()

# 5) Construye DataFrame con rutas y etiquetas
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

# 6) Split estratificado en train/val
df_train, df_val = train_test_split(
    df, test_size=0.2,
    stratify=df['class'],
    random_state=42
)

# 7) Generadores con augmentations moderadas
batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.8,1.2),
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

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

# 8) Cálculo de class_weight
weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weight_dict = dict(enumerate(weights))

# 9) Definición de la CNN (ligeramente mayor capacidad)
num_classes = df['class'].nunique()

model = Sequential([
    layers.Conv2D(32, 3, activation='relu',
                  input_shape=(224,224,3)),
    layers.MaxPooling2D(), layers.Dropout(0.3),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(), layers.Dropout(0.3),

    layers.Conv2D(128, 3, activation='relu'),
    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),

    layers.Dense(num_classes, activation='softmax')
])

# 10) Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-6
)
checkpoint = ModelCheckpoint(
    'mejor_modelo.h5',
    monitor='val_loss',
    save_best_only=True
)

# 11) Compilación con learning rate fijo inicial
from tensorflow.keras.optimizers import Adam
opt = Adam(learning_rate=1e-3)

model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 12) Entrenamiento (20-30 épocas)
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weight_dict,
    verbose=2
)

# 13) Mapeo índice → etiqueta
idx_to_label = {v: k for k, v in train_gen.class_indices.items()}

# 14) Inferencia sobre nuevas imágenes
def predecir_emocion_y_persona(ruta_img):
    face = detectar_y_recortar_cara(ruta_img)
    if face is None:
        print(f"No se detectó cara en {ruta_img}")
        return
    x = face.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    probs = model.predict(x)[0]
    idx = np.argmax(probs)
    label = idx_to_label[idx]
    print(f"{os.path.basename(ruta_img)} → {label} ({probs[idx]:.2f})")

print("\n--- Inferencia en 'data/test' ---")
for f in os.listdir('data/test'):
    if f.lower().endswith(('.jpg','jpeg','png')):
        predecir_emocion_y_persona(os.path.join('data/test', f))