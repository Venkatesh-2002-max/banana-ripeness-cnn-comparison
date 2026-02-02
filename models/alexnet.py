# ============================================
# AlexNet Model for Banana Ripeness Detection
# ============================================

import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam

print("✅ TensorFlow version:", tf.__version__)

# ------------------------------------------------------
# Step 1: Dataset paths
# ------------------------------------------------------
train_dir = r"C:\Users\Venkatesh P\Downloads\archive (1)\Banana Ripeness Classification Dataset\train"
val_dir   = r"C:\Users\Venkatesh P\Downloads\archive (1)\Banana Ripeness Classification Dataset\valid"
test_dir  = r"C:\Users\Venkatesh P\Downloads\archive (1)\Banana Ripeness Classification Dataset\test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4

# ------------------------------------------------------
# Step 2: Data preprocessing
# ------------------------------------------------------
datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Save class indices (overwrite is fine)
with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

print("📌 Class indices:", train_gen.class_indices)

# ------------------------------------------------------
# Step 3: Build AlexNet model
# ------------------------------------------------------
model = Sequential([
    Conv2D(96, (11, 11), strides=4, activation="relu",
           input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(3, 3), strides=2),

    Conv2D(256, (5, 5), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(3, 3), strides=2),

    Conv2D(384, (3, 3), padding="same", activation="relu"),
    Conv2D(384, (3, 3), padding="same", activation="relu"),
    Conv2D(256, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(3, 3), strides=2),

    Flatten(),
    Dense(4096, activation="relu"),
    Dropout(0.5),
    Dense(4096, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

# ------------------------------------------------------
# Step 4: Compile model
# ------------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------------------------------------
# Step 5: Train model
# ------------------------------------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15
)

# ------------------------------------------------------
# Step 6: Evaluate model
# ------------------------------------------------------
test_loss, test_acc = model.evaluate(test_gen)
print(f"✅ AlexNet Test Accuracy: {test_acc:.4f}")
print(f"✅ AlexNet Test Loss: {test_loss:.4f}")
# ------------------------------------------------------
# Step 8: Save model
# ------------------------------------------------------
# ------------------------------------------------------
model.save("alexnet_banana.keras")
print("📁 AlexNet model saved as alexnet_banana.keras")



# ------------------------------------------------------
# Step 7: Plot accuracy & loss (for GitHub)
# ------------------------------------------------------

plt.figure(figsize=(10, 4))


# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("AlexNet Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("AlexNet Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

# Save plot
os.makedirs("../results", exist_ok=True)
plt.savefig("../results/alexnet_training_curves.png")
plt.show()

