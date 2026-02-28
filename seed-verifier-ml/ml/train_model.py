import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

def f05_score(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)
    return (1.25 * p * r) / (0.25 * p + r + 1e-7)

def build_v15():
    # Load PRETRAINED weights from ImageNet
    base = MobileNetV3Small(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base.trainable = False 
    
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Lambda(preprocess_input),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=[f05_score])
    
    # Phase 2: Unfreeze for Fine-tuning
    base.trainable = True
    for layer in base.layers[:-30]: layer.trainable = False
    
    model.save("app/seed_model_v15.h5")
    print("Pretrained base adapted and saved.")

if __name__ == "__main__":
    build_v15()