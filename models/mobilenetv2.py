# ============================================
# 🍌 Banana Ripeness Detection using MobileNetV2
# ============================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

print("✅ TensorFlow version:", tf.__version__)

# ------------------------------------------------------
# Step 1: Dataset paths (UPDATE if needed)
# ------------------------------------------------------
train_dir = r"C:\Users\Venkatesh P\Downloads\archive (1)\Banana Ripeness Classification Dataset\train"
val_dir   = r"C:\Users\Venkatesh P\Downloads\archive (1)\Banana Ripeness Classification Dataset\valid"
test_dir  = r"C:\Users\Venkatesh P\Downloads\archive (1)\Banana Ripeness Classification Dataset\test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ------------------------------------------------------
# Step 2: Data preprocessing & augmentation (FIXED)
# ------------------------------------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_gen = val_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Save class indices for inference
with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

print("📌 Class indices:", train_gen.class_indices)

NUM_CLASSES = train_gen.num_classes

# ------------------------------------------------------
# Step 3: Compute class weights (CRITICAL FIX)
# ------------------------------------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)

class_weight_dict = dict(enumerate(class_weights))
print("⚖️ Class weights:", class_weight_dict)

# ------------------------------------------------------
# Step 4: Build MobileNetV2 model
# ------------------------------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # Freeze base initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ------------------------------------------------------
# Step 5: Compile model
# ------------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------------------------------------
# Step 6: Callbacks
# ------------------------------------------------------
checkpoint = ModelCheckpoint(
    "best_mobilenet_banana.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

# ------------------------------------------------------
# Step 7: Initial training
# ------------------------------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=[checkpoint, early_stop],
    class_weight=class_weight_dict
)

# ------------------------------------------------------
# Step 8: Fine-tuning (IMPORTANT)
# ------------------------------------------------------
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[checkpoint, early_stop],
    class_weight=class_weight_dict
)



# ------------------------------------------------------
# Step 9: Evaluate on test set
# ------------------------------------------------------
test_loss, test_acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy: {test_acc:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")


# ------------------------------------------------------
# Step 10: Save final model
# ------------------------------------------------------
model.save("mobilenet_banana_final.keras")
print("📁 Final model saved as mobilenet_banana_final.keras")

# ------------------------------------------------------
# Step 11: Plot accuracy & loss
# ------------------------------------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# ✅ SAVE plot instead of only showing
plt.savefig("../results/mobilenetv2_training_curves.png", dpi=300)
plt.close()

print("📊 Training curves saved to results/mobilenetv2_training_curves.png")