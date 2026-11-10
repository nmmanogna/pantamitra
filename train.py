"""Train a crop disease classifier (MobileNetV2 transfer learning).
Usage: python train.py <path-to-PlantVillage-folder>
Folder layout: one sub-folder per class, e.g. Tomato___Early_blight/
"""
import json
import sys
import tensorflow as tf
from tensorflow.keras import layers

data_dir = sys.argv[1] if len(sys.argv) > 1 else "PlantVillage"
IMG = (224, 224)

train = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=42,
    image_size=IMG, batch_size=32)
val = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=42,
    image_size=IMG, batch_size=32)

classes = train.class_names
json.dump(classes, open("classes.json", "w"))

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG + (3,), include_top=False, weights="imagenet")
base.trainable = False

inp = tf.keras.Input(IMG + (3,))
x = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])(inp)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(len(classes), activation="softmax")(x)
model = tf.keras.Model(inp, out)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train.prefetch(tf.data.AUTOTUNE),
          validation_data=val.prefetch(tf.data.AUTOTUNE), epochs=5)
loss, acc = model.evaluate(val)
print(f"Validation accuracy: {acc:.3f}  (quote this in your PPT)")
model.save("crop_model.keras")
