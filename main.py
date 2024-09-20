"""Pneumonia Xray detection - 1.0.0"""

from datetime import datetime
from dataclasses import dataclass
from os import path
from glob import glob
from sys import argv

import numpy as np
import pandas as pd
import tensorflow as tf
from seaborn import heatmap
from keras.callbacks import EarlyStopping, TensorBoard
from keras import Sequential, models
from keras.src.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

tf.random.set_seed(42)

# Constants.
MODEL_VERSION: str = "1_0_0"
MODEL_PATH: str = f"models/pneumonia_xray_detector_{MODEL_VERSION}.keras"
FORCE_RETRAINING: bool = "--train" in argv
EPOCHS: int = 20
TARGET_SIZE: tuple[int, int] = (150, 150)  # Will resize image to 150x150 pixels.


@dataclass(frozen=True)
class Labels:
    NORMAL: str = "NORMAL"
    PNEUMONIA: str = "PNEUMONIA"


def load_dataset() -> pd.DataFrame:
    df: pd.DataFrame = pd.DataFrame(glob("data/chest_xray" + "/*/*/*.jpeg"), columns=["PATH"])
    return df.assign(LABEL=df["PATH"].apply(lambda path_: path_.split("\\")[2].strip()))


def balance_dataset(df_loaded: pd.DataFrame) -> pd.DataFrame:
    df_normal: pd.DataFrame = df_loaded[df_loaded["LABEL"] == Labels.NORMAL]
    df_pneumonia: pd.DataFrame = df_loaded[df_loaded["LABEL"] == Labels.PNEUMONIA]
    df_pneumonia_balanced: pd.DataFrame = resample(
        df_pneumonia, replace=False, n_samples=len(df_normal), random_state=42
    )
    df_balanced: pd.DataFrame = pd.concat([df_normal, df_pneumonia_balanced])
    df_balanced["LABEL"] = np.where(df_balanced["LABEL"] == Labels.PNEUMONIA, "1", "0")
    return df_balanced


def split_balanced(df_balanced: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the data into train (80%), test (10%), and validation (10%) sets."""
    df_shuffled: pd.DataFrame = df_balanced.sample(frac=1.0, random_state=42)
    df_train, df_temp = train_test_split(df_shuffled, test_size=0.2, random_state=42)
    df_test, df_valid = train_test_split(df_temp, test_size=0.5, random_state=42)
    print(f"TRAINING shape: {df_train.shape}\nVALIDATION shape: {df_valid.shape}\nTESTING shape: {df_test.shape}")
    return {"TRAIN": df_train, "VALID": df_valid, "TEST": df_test}


def create_img_generators(training_sets: dict[str, pd.DataFrame]) -> dict[str, ImageDataGenerator]:
    """Create generators which will be used to load and process xray images in real-time during training."""
    # Normalize pixel values from 0-255 to 0-1.
    img_data_gen: ImageDataGenerator = ImageDataGenerator(rescale=1.0 / 255)

    img_gen_kw = {
        "x_col": "PATH",
        "y_col": "LABEL",
        "target_size": TARGET_SIZE,
        "class_mode": "binary",
        "color_mode": "rgb",
    }
    return {
        "TRAIN_GENERATOR": img_data_gen.flow_from_dataframe(
            dataframe=training_sets["TRAIN"],
            batch_size=32,
            shuffle=True,
            horizontal_flip=True,
            **img_gen_kw,
        ),
        "VALID_GENERATOR": img_data_gen.flow_from_dataframe(
            dataframe=training_sets["VALID"],
            batch_size=32,
            shuffle=False,
            **img_gen_kw,
        ),
        "TEST_GENERATOR": img_data_gen.flow_from_dataframe(
            dataframe=training_sets["TEST"],
            batch_size=64,
            shuffle=False,
            **img_gen_kw,
        ),
    }


def create_model() -> Sequential:
    model: Sequential = Sequential(
        [
            # Convolutional layers extract features from simple to complex (edges, textures to more complex patterns).
            Conv2D(32, (3, 3), activation="relu", input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)),
            MaxPooling2D((2, 2)),  # Reduce spatial dimensions.
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),

            # Transform 3D feature maps into 1D vector.
            Flatten(),
            # Fully connected deep layers process extracted features to detect abstract patterns.
            Dense(64, activation="relu"),
            Dropout(0.5),  # Drop randomly 50% of neurons to prevent overfitting.
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(256, activation="relu"),
            Dropout(0.5),

            # Output layer in form of binary classification (0 or 1).
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def train_model(generators: dict[str, ImageDataGenerator], model: Sequential):
    return model.fit(
        generators["TRAIN_GENERATOR"],
        epochs=EPOCHS,
        validation_data=generators["VALID_GENERATOR"],
        verbose=1,
        callbacks=[
            TensorBoard(log_dir=f"logs/{datetime.now().strftime('%Y-%m-%d %H-%M')}", histogram_freq=1),
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
        ],
    )


def plot_accuracy(training_history) -> None:
    plt.plot(training_history.history["accuracy"], label="Training Accuracy")
    plt.plot(training_history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("training_assets/training_val_accuracy.png")
    plt.show()

    # Plot training and validation loss.
    plt.plot(training_history.history["loss"], label="Training Loss")
    plt.plot(training_history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_assets/training_val_loss.png")
    plt.show()


def evaluate_model(model: Sequential, test_generator: ImageDataGenerator) -> str:
    """Validate model accuracy on test set."""
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy:.4f}\nTest Loss: {test_loss:.4f}")

    # Make predictions on the test set.
    test_generator.reset()
    Y_pred = model.predict(test_generator)
    y_pred = np.round(Y_pred).astype(int)
    y_true = test_generator.classes

    cm = confusion_matrix(y_true, y_pred)
    classification_report_ = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys())
    print(f"Confusion Matrix: \n{cm}\nClassification report:\n{classification_report_}")
    return cm


def plot_confusion_matrix(cm, test_generator: ImageDataGenerator) -> None:
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


def plot_final_test_results(model: Sequential, test_generator: ImageDataGenerator, num_images: int = 10) -> None:
    test_generator.reset()
    X_test, y_test = next(test_generator)
    y_pred = model.predict(X_test)
    y_pred_labels = np.round(y_pred).astype(int)

    plt.figure(figsize=(num_images * 2, num_images))
    for i in range(num_images):
        plt.subplot(2, num_images // 2, i + 1)
        plt.imshow(X_test[i])
        plt.title(f"Labeled: {y_test[i]}, Predicted: {y_pred_labels[i][0]}")
        plt.axis("off")
    plt.savefig("training_assets/final_test_results.png")
    plt.show()


if __name__ == "__main__":
    df_loaded: pd.DataFrame = load_dataset()
    df_balanced: pd.DataFrame = balance_dataset(df_loaded)
    training_sets: dict[str, pd.DataFrame] = split_balanced(df_balanced)
    img_generators: dict[str, ImageDataGenerator] = create_img_generators(training_sets)

    if not path.exists(MODEL_PATH) or FORCE_RETRAINING:
        model: Sequential = create_model()
        training_history = train_model(img_generators, model)
        plot_accuracy(training_history)
        model.save(MODEL_PATH)
    else:
        model = models.load_model(MODEL_PATH)
        model.summary()

    cm = evaluate_model(model, img_generators["TEST_GENERATOR"])
    plot_confusion_matrix(cm, img_generators["TEST_GENERATOR"])
    plot_final_test_results(model=model, test_generator=img_generators["TEST_GENERATOR"], num_images=20)
