"""Pneumonia Xray detection - 0.0.2"""

from datetime import datetime
from dataclasses import dataclass
from os import path
from glob import glob
from sys import argv

import numpy as np
import pandas as pd
from seaborn import heatmap
from keras.callbacks import TensorBoard
from keras import Sequential, models
from keras.src.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample


MODEL_VERSION: str = "0_0_2"
MODEL_PATH: str = f"trained_model_{MODEL_VERSION}.keras"
FORCE_RETRAINING: bool = '--train' in argv

df: pd.DataFrame = pd.DataFrame(glob("data/chest_xray" + "/*/*/*.jpeg"), columns=["PATH"])
df["LABEL"] = df["PATH"].apply(lambda path_: path_.split("\\")[2].strip())


@dataclass(frozen=True)
class Labels:
    NORMAL: str = "NORMAL"
    PNEUMONIA: str = "PNEUMONIA"


df_normal: pd.DataFrame = df[df["LABEL"] == Labels.NORMAL]
df_pneumonia: pd.DataFrame = df[df["LABEL"] == Labels.PNEUMONIA]

df_pneumonia_balanced: pd.DataFrame = resample(df_pneumonia, replace=False, n_samples=len(df_normal), random_state=42)

df_balanced: pd.DataFrame = pd.concat([df_normal, df_pneumonia_balanced])
df_balanced["LABEL"] = np.where(df_balanced["LABEL"] == Labels.PNEUMONIA, "1", "0")

df_shuffled: pd.DataFrame = df_balanced.sample(frac=1.0, random_state=42)

df_train, df_temp = train_test_split(df_shuffled, test_size=0.2, random_state=42)
df_test, df_valid = train_test_split(df_temp, test_size=0.5, random_state=42)
print(f"Training set shapes: {df_train.shape}")
print(f"Testing set shapes: {df_test.shape}")
print(f"Validation set shapes: {df_valid.shape}")

# Transform pixels values from 0-255 to 0-1.
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

target_size = (150, 150)

# Create generators which will load train, valid and test xray images.
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col="PATH",
    y_col="LABEL",
    target_size=target_size,
    batch_size=32,
    class_mode="binary",
    color_mode="rgb",
    # rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # fill_mode='nearest',
    shuffle=True,
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    x_col="PATH",
    y_col="LABEL",
    target_size=target_size,
    color_mode="rgb",
    batch_size=64,
    class_mode="binary",
    shuffle=False,
)

valid_generator = test_datagen.flow_from_dataframe(
    dataframe=df_valid,
    x_col="PATH",
    y_col="LABEL",
    target_size=target_size,
    batch_size=32,
    color_mode="rgb",
    class_mode="binary",
    shuffle=False,
)

if not path.exists(MODEL_PATH) or FORCE_RETRAINING:
    # Build the model.
    model: Sequential = Sequential(
        [
            # First convolutional layer with reducing layer.
            Conv2D(32, (3, 3), activation="relu", input_shape=(target_size[0], target_size[1], 3)),
            MaxPooling2D((2, 2)),
            # Second convolutional layer with reducing layer.
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            # Third convolutional layer with reducing layer.
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            # First hidden layer.
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            # Output hidden layer.
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # Train the model.
    tensorboard = TensorBoard(log_dir=f"logs/{datetime.now().strftime('%Y-%m-%d %H-%M')}", histogram_freq=1)
    history = model.fit(train_generator, epochs=10, validation_data=valid_generator, verbose=1, callbacks=[tensorboard])

    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("training_assets/training_val_accuracy.png")
    plt.show()

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_assets/training_val_loss.png")
    plt.show()

    model.save(MODEL_PATH)
else:
    model = models.load_model(MODEL_PATH)
    model.summary()

# Validate model accuracy.
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Execute final test.
test_generator.reset()
Y_pred = model.predict(test_generator)
y_pred = np.round(Y_pred).astype(int)

y_true = test_generator.classes

print(
    f"Classification report:\n{classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys())}"
)
cm = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix: \n{cm}")

plt.figure(figsize=(10, 7))
heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=test_generator.class_indices.keys(),
    yticklabels=test_generator.class_indices.keys(),
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("training_assets/confusion_matrix.png")
plt.show()


def plot_final_test_results(test_generator, model, num_images=10) -> None:
    test_generator.reset()
    X_test, y_test = next(test_generator)
    y_pred = model.predict(X_test)
    y_pred_labels = np.round(y_pred).astype(int)

    plt.figure(figsize=(num_images * 2, num_images))
    for i in range(num_images):
        ax = plt.subplot(2, num_images // 2, i + 1)
        plt.imshow(X_test[i])
        plt.title(f"Labeled: {y_test[i]}, Predicted: {y_pred_labels[i][0]}")
        plt.axis("off")
    plt.savefig("training_assets/final_test_results.png")
    plt.show()


plot_final_test_results(test_generator, model, num_images=10)
