"""Pneumonia Xray detection - 0.0.1"""

from dataclasses import dataclass
from os import path
from glob import glob

import numpy as np
import pandas as pd
from seaborn import heatmap
from keras import Sequential, models
from keras.src.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.utils import resample

MODEL_VERSION: str = "0_0_1"
MODEL_PATH: str = f"trained_model_{MODEL_VERSION}.h5"
FORCE_RETRAINING: bool = False

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
print(f"Validation set shapes: {df_valid.shape}")
print(f"Testing set shapes: {df_test.shape}")

# Transform pixels values from 0-255 to 0-1.
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Create generators which will load train, valid and test xray images.
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col="PATH",
    y_col="LABEL",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    color_mode="rgb",
    shuffle=True,
)

valid_generator = test_datagen.flow_from_dataframe(
    dataframe=df_valid,
    x_col="PATH",
    y_col="LABEL",
    target_size=(150, 150),
    batch_size=32,
    color_mode="rgb",
    class_mode="binary",
    shuffle=False,
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    x_col="PATH",
    y_col="LABEL",
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=64,
    class_mode="binary",
    shuffle=False,
)

if not path.exists(MODEL_PATH) or FORCE_RETRAINING:
    # Build model.
    model: Sequential = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # Train model.
    history = model.fit(train_generator, epochs=10, validation_data=valid_generator, verbose=1)

    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    model.save(MODEL_PATH)
else:
    model = models.load_model(MODEL_PATH)

# Validate model accuracy.
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Execute final test.
test_generator.reset()
Y_pred = model.predict(test_generator)
y_pred = np.round(Y_pred).astype(int)

y_true = test_generator.classes

# Classification Report
print(f"Classification report:{classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys())}")
cm = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix: {cm}")

plt.figure(figsize=(10, 7))
heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
plt.savefig("training_assets/confusion_matrix.png")


def plot_final_test_results(test_generator, model, num_images=10) -> None:
    test_generator.reset()
    X_test, y_test = next(test_generator)
    y_pred = model.predict(X_test)
    y_pred_labels = np.round(y_pred).astype(int)

    plt.figure(figsize=(20, 10))
    for i in range(num_images):
        ax = plt.subplot(2, num_images // 2, i + 1)
        plt.imshow(X_test[i])
        plt.title(f"Labeled: {y_test[i]}, Predicted: {y_pred_labels[i][0]}")
        plt.axis("off")
    plt.show()
    plt.savefig("training_assets/final_test_results.png")


plot_final_test_results(test_generator, model, num_images=10)
