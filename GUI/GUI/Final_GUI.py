from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import os
import cv2
import tensorflow as tf
import pandas as pd
import keras
from keras import layers, regularizers
import numpy as np
import shutil
from keras.applications import efficientnet_v2
from pathlib import Path
import threading
from PIL import Image, ImageTk
import tkinter as tk
# import funct
from tkinter import filedialog, messagebox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Text, Button, PhotoImage

from keras.applications import efficientnet_v2


@keras.saving.register_keras_serializable()
class CLAHE(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, active=True):
        # THIS WRAPPER IS MANDATORY TO SET THE SHAPE OF THE OUTPUT TENSOR 
        # https://stackoverflow.com/questions/42590431/output-from-tensorflow-py-func-has-unknown-rank-shape
        def tf_wrapper(img):
            @tf.py_function(Tout=tf.float32)
            def apply_CLAHE(img):
                t_img = img.numpy().astype('uint8')
                gray = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
                clahe_img = clahe.apply(gray)
                clahe_img_colored = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
                return tf.convert_to_tensor(clahe_img_colored, dtype=tf.float32)

            new_img = apply_CLAHE(img)
            new_img.set_shape(img.shape)
            return new_img

        _ndims = inputs.get_shape().ndims
        if active == True:
            if _ndims == 3:
                return tf_wrapper(inputs)
            elif _ndims == 4:
                return tf.map_fn(tf_wrapper, inputs)
        else:
            return inputs

@keras.saving.register_keras_serializable()
class WristMURA(keras.Model):
    def __init__(self,
                 preprocess_fn,
                 base_model,
                 apply_CLAHE,
                 apply_aug,
                 top_layers_trainable,
                 top_layers_trainable_num,
                 trainable=True,
                 name=None,
                 dtype=None
                 ):
        super().__init__(trainable=trainable, name=name, dtype=dtype)

        self.preprocess_fn = preprocess_fn

        # Model layers
        self.clahe_layer = CLAHE()
        self.data_augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.2),
        ])

        self.base_model = base_model
        self.dense1_layer = keras.layers.Dense(512, kernel_regularizer=keras.regularizers.L2(1e-4), activation='relu',
                                               name="Dense_layer_1")
        self.dropout1_layer = keras.layers.Dropout(.4, name="Dropout_layer_1")
        self.dense2_layer = keras.layers.Dense(256, kernel_regularizer=keras.regularizers.L2(1e-4), activation='relu',
                                               name="Dense_layer_2")
        self.dropout2_layer = keras.layers.Dropout(.4, name="Dropout_layer_2")
        self.output_layer = keras.layers.Dense(1, activation='sigmoid', name="Output_layer")

        # Model configs
        self.base_model.trainable = False

        self.apply_CLAHE = apply_CLAHE
        self.apply_aug = apply_aug

        self.top_layers_trainable_num = top_layers_trainable_num
        self.top_layers_trainable = top_layers_trainable

    def call(self, inputs, training=None):

        inputs = self.clahe_layer(inputs, active=self.apply_CLAHE)
        inputs = self.data_augmentation(inputs, training=(self.apply_aug and training))

        inputs = self.preprocess_fn(inputs)

        x = self.base_model(inputs, training=training)
        x = self.dense1_layer(x)
        x = self.dropout1_layer(x, training=training)
        x = self.dense2_layer(x)
        x = self.dropout2_layer(x, training=training)

        output = self.output_layer(x)

        return output

    @property
    def top_layers_trainable(self):
        return self._top_layers_trainable

    @top_layers_trainable.setter
    def top_layers_trainable(self, value):
        self._top_layers_trainable = value
        if value == True:
            self.base_model.trainable = True
            for layer in self.base_model.layers[:-self.top_layers_trainable_num]:
                layer.trainable = False
        else:
            self.base_model.trainable = False

    def get_config(self):
        base_config = super().get_config()
        config = {
            "preprocess_fn": keras.saving.serialize_keras_object(self.preprocess_fn),
            "base_model": keras.saving.serialize_keras_object(self.base_model),
            "apply_CLAHE": keras.saving.serialize_keras_object(self.apply_CLAHE),
            "apply_aug": keras.saving.serialize_keras_object(self.apply_aug),
            "top_layers_trainable": keras.saving.serialize_keras_object(self.top_layers_trainable),
            "top_layers_trainable_num": keras.saving.serialize_keras_object(self.top_layers_trainable_num),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config['preprocess_fn'] = keras.saving.deserialize_keras_object(config['preprocess_fn'])
        config['base_model'] = keras.saving.deserialize_keras_object(config['base_model'])
        config['apply_CLAHE'] = keras.saving.deserialize_keras_object(config['apply_CLAHE'])
        config['apply_aug'] = keras.saving.deserialize_keras_object(config['apply_aug'])
        config['top_layers_trainable'] = keras.saving.deserialize_keras_object(config['top_layers_trainable'])
        config['top_layers_trainable_num'] = keras.saving.deserialize_keras_object(config['top_layers_trainable_num'])
        return cls(**config)


@keras.saving.register_keras_serializable()
def preprocess_fn_wrapper(x):
    return efficientnet_v2.preprocess_input(x)


# Define Paths and Constants 
BoneType_Path = "D:\GP\Seminar 2\Models\BoneType_ResNet50.keras"  # 97.62% Accuracy
Shoulder_Path = "D:\GP\Seminar 2\Models\DenseNetShoulder.keras"
Finger_Path = "D:\GP\Seminar 2\Models\ResNet50Finger80.keras"  # 80% Accuracy
Elbow_Path = "D:\GP\Seminar 2\Models\ElbowResNet50.keras"
Wrist_Path = "D:\GP\Seminar 2\Models\EfficientNetWrist.keras"
Hand_Path = "D:\GP\Seminar 2\Models\HandResNet50NewTest.keras"


# Functions Declarations
# Define required Preprocessing Funcs
def Clahe_Enhance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    x, y, w, h = cv2.boundingRect(thresh)
    x, y, w, h = x, y, w + 20, h + 20
    img = img[y:y + h, x:x + w]
    gray_crop = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_crop)
    return clahe_img


def Normalize_IMG(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = img.astype('float32') / 255.0
    return img


# For Kaggle
def Copy_Model(source_path, name):
    destination_path = f"/kaggle/working/{name}.keras"
    shutil.copyfile(source_path, destination_path)
    return destination_path


def load_image_wrist(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    return img


# Models Functions

def BoneType_Predict(img):
    # categories list
    categories_parts = ["Elbow", "Finger", "Hand", "Shoulder", "Wrist"]
    img_resized = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img_array = preprocess_input(img_array)  # Preprocess the image using ResNet50 preprocessing function
    prediction = np.argmax(BoneType_Model.predict(preprocessed_img_array), axis=1)
    prediction_str = categories_parts[prediction.item()]
    return prediction_str


def Shoulder_Predict(img):
    img = Clahe_Enhance(img)
    img = Normalize_IMG(img)
    img = np.expand_dims(img, axis=0)
    prediction = Shoulder_Model.predict(img)
    binary_prediction = "Fractured" if prediction > 0.5 else "Non-Fractured"
    return binary_prediction


def Finger_Predict(img):
    img_resized = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img_array = keras.applications.resnet50.preprocess_input(img_array)
    prediction = Finger_Model.predict(preprocessed_img_array)
    binary_prediction = "Fractured" if prediction > 0.5 else "Non-Fractured"
    return binary_prediction


# def Hand_Predict(img):
#     img_resized = cv2.resize(img, (224, 224))
#     img_array = image.img_to_array(img_resized)
#     img_array = np.expand_dims(img_array, axis=0)
#     prediction = Hand_Model.predict(img_array)
#     binary_prediction = "Fractured" if prediction > 0.72 else "Non-Fractured"
#     return binary_prediction

def Hand_Predict(img):
    img_resized = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img_array = keras.applications.resnet50.preprocess_input(img_array)
    prediction = Finger_Model.predict(preprocessed_img_array)
    binary_prediction = "Fractured" if prediction > 0.5 else "Non-Fractured"
    return binary_prediction


def Elbow_Predict(img):
    img = Clahe_Enhance(img)
    img_resized = cv2.resize(img, (224, 224))
    # Convert image to RGB if it's grayscale
    if len(img_resized.shape) == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img_array = keras.applications.resnet50.preprocess_input(img_array)
    prediction = Elbow_Model.predict(preprocessed_img_array)
    binary_prediction = "Fractured" if prediction > 0.5 else "Non-Fractured"

    return binary_prediction


def Wrist_Predict(img):
    img = np.expand_dims(img, 0)
    prediction = Wrist_Model.predict(img, verbose=0)
    print(prediction)
    binary_prediction = "Fractured" if prediction > 0.5 else "Non-Fractured"
    return binary_prediction


def predict_fracture(img):
    Predicted_bonetype = BoneType_Predict(img)
    print(f"Predicted bone-type is {Predicted_bonetype}")
    if Predicted_bonetype == "Shoulder":
        final_p = Shoulder_Predict(img)
    elif Predicted_bonetype == "Finger":
        final_p = Finger_Predict(img)
    elif Predicted_bonetype == "Wrist":
        wrist_img = load_image_wrist(file_path)
        final_p = Wrist_Predict(wrist_img)
    elif Predicted_bonetype == "Hand":
        final_p = Hand_Predict(img)
    elif Predicted_bonetype == "Elbow":
        final_p = Elbow_Predict(img)
    print("Predicted Status: ", final_p)
    return Predicted_bonetype, final_p


def load_models():
    global BoneType_Model, Shoulder_Model, Finger_Model, Hand_Model, Elbow_Model, Wrist_Model

    # BoneType_Model_Path = Copy_Model(BoneType_Path,"BoneType")
    # Shoulder_Model_Path = Copy_Model(Shoulder_Path,"Shoulder")
    # Finger_Model_Path = Copy_Model(Finger_Path,"Finger")
    # Hand_Model_Path = Copy_Model(Hand_Path,"Hand")
    # Elbow_Model_Path = Copy_Model(Elbow_Path,"Elbow")
    # Wrist_Model_Path = Copy_Model(Wrist_Path,"Wrist")
    # Models Loading
    BoneType_Model = load_model(BoneType_Path)
    Shoulder_Model = load_model(Shoulder_Path)
    Finger_Model = load_model(Finger_Path)
    Hand_Model = load_model(Hand_Path)
    # shutil.copyfile('/kaggle/input/alexnettrial/tensorflow2/alex2/1/alexNet.h5', '/kaggle/working/alexNet.h5')
    # Hand_Model = model = tf.keras.models.load_model('/kaggle/working/alexNet.h5')
    Elbow_Model = load_model(Elbow_Path)
    Wrist_Model = load_model(Wrist_Path)
    # return BoneType_Model,Shoulder_Model,Finger_Model, Hand_Model,Elbow_Model,Wrist_Model


# Data Loading
######################################### GUI ###########################################################
load_models()
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"D:\GP\Seminar 2\GUI\GUI\assets")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


def load_xray():
    global file_path
    file_path = filedialog.askopenfilename()
    print(file_path)
    # Clear text_1 and text_2
    text_1.configure(state="normal")
    text_1.delete(1.0, tk.END)
    text_1.configure(state="disabled")
    text_2.configure(state="normal")
    text_2.delete(1.0, tk.END)
    text_2.configure(state="disabled")


def predict():
    global bone_type, fracture
    print(file_path)
    img = cv2.imread(file_path)
    bone_type, fracture = predict_fracture(img)
    update_text_widgets()


def update_text_widgets():
    text_1.configure(state="normal")
    text_1.delete(1.0, tk.END)  # Clear previous content
    text_1.insert(tk.END, bone_type)
    text_1.tag_configure("white", foreground="white",
                         font=("Arial", 30, "bold"))  # Set the font size to 20 and color to white
    text_1.delete(1.0, tk.END)
    text_1.insert(tk.END, bone_type, "white")
    text_1.configure(state="disabled")
    text_2.configure(state="normal")
    text_2.delete(1.0, tk.END)
    text_2.insert(tk.END, fracture)

    # Set font color based on the value of fracture
    if fracture == "Non-Fractured":
        text_2.tag_configure("green", foreground="green", font=("Arial", 30, "bold"))
        text_2.delete(1.0, tk.END)
        text_2.insert(tk.END, fracture, "green")
    elif fracture == "Fractured":
        text_2.tag_configure("red", foreground="red", font=("Arial", 30, "bold"))
        text_2.delete(1.0, tk.END)
        text_2.insert(tk.END, fracture, "red")

    text_2.configure(state="disabled")


def second_page(master):
    global text_1, text_2
    frame = tk.Frame(master, bg="#000000")
    frame.place(x=0, y=0, relwidth=1, relheight=1)

    canvas = Canvas(
        frame,
        bg="#000000",
        height=600,
        width=800,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )

    canvas.place(x=0, y=0)

    canvas.create_text(
        45.0,
        300.0,
        anchor="nw",
        text="Bone Type:",
        fill="#FFFFFF",
        font=("Inter Bold", 32)
    )

    text_1 = tk.Text(
        frame,
        bg="black",  # Set background color to black
        fg="#FFFFFF",
        wrap=tk.WORD,
        state="disabled",
        borderwidth=0,  # Set border width to 0
        highlightthickness=0
    )
    text_1.place(
        x=47.0,
        y=348.0,
        width=350.0,
        height=78.0,

    )

    canvas.create_text(
        45.0,
        441.0,
        anchor="nw",
        text="Status:",
        fill="#FFFFFF",
        font=("Inter Bold", 32)
    )

    text_2 = tk.Text(
        frame,
        bg="black",  # Set background color to black
        fg="#FFFFFF",
        wrap=tk.WORD,
        state="disabled",
        borderwidth=0,  # Set border width to 0
        highlightthickness=0
    )
    text_2.place(
        x=43.0,
        y=489.0,
        width=350.0,
        height=78.0,

    )

    button_3 = Button(
        frame,
        text="Load X-ray",
        command=lambda: load_xray(),
        relief="flat",
        bg="#A3A3A3",
        font=("Inter Bold", 32),
        fg="#FFFFFF"
    )
    button_3.place(
        x=47.0,
        y=54.0,
        width=350.0,
        height=80.0
    )

    button_4 = Button(
        frame,
        text="Predict",
        command=lambda: predict(),
        relief="flat",
        bg="#A3A3A3",
        font=("Inter Bold", 32),
        fg="#FFFFFF"
    )
    button_4.place(
        x=47.0,
        y=177.0,
        width=350.0,
        height=80.0
    )

    file = relative_to_assets("sec.gif")
    gif_list = []
    delay = 0
    current = None

    def ready_gif():
        nonlocal delay, current
        gif = Image.open(file)

        for r in range(0, gif.n_frames):
            gif.seek(r)
            gif_list.append(ImageTk.PhotoImage(gif.copy()))
            delay = gif.info['duration']
        play()

    count = -1

    def play():
        nonlocal count, current

        if count >= len(gif_list) - 1:
            count = -1
            play()
        else:
            count += 1
            current = gif_list[count]
            gflbl.config(image=current, borderwidth=0, highlightthickness=0)
            frame.after(delay, play)

    gflbl = tk.Label(frame, borderwidth=0, highlightthickness=0)
    gflbl.pack(fill=tk.BOTH)
    gflbl.place(x=463, y=0)

    threading.Thread(target=ready_gif).start()


window = tk.Tk()

window.geometry("800x600")
window.configure(bg="#000000")

canvas = Canvas(
    window,
    bg="#FFFFFF",
    height=600,
    width=800,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

canvas.place(x=0, y=0)

file = relative_to_assets("first.gif")  # Adjusted file path
gif_list = []
delay = 0
current = None


def ready_gif():
    global delay, current
    gif = Image.open(file)

    for r in range(0, gif.n_frames):
        gif.seek(r)
        gif_list.append(ImageTk.PhotoImage(gif.copy()))
        delay = gif.info['duration']
    play()


count = -1


def play():
    global count, current

    if count >= len(gif_list) - 1:
        count = -1
        play()
    else:
        count += 1
        current = gif_list[count]
        gflbl.config(image=current, borderwidth=0, highlightthickness=0)
        window.after(delay, play)


gflbl = tk.Label(window)
gflbl.pack(fill=tk.BOTH)
gflbl.place(x=0, y=0)

threading.Thread(target=ready_gif).start()

header_canvas = Canvas(
    window,
    bg="#000000",
    height=100,
    width=800,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
header_canvas.place(x=0, y=0)

header_canvas.create_text(
    400,  # Center X coordinate
    50,  # Y coordinate
    anchor="center",  # Anchor point at center
    text="Bone Fracture Detection",  # Text content
    fill="white",  # Font color
    font=("Arial", 32, "bold")  # Font settings
)

button_image_1 = PhotoImage(
    file=relative_to_assets("start.png"))  # Adjusted file path
button_1 = Button(
    window,
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: second_page(window),
    relief="flat"
)
button_1.place(
    x=226.0,
    y=467.0,
    width=350.0,
    height=80.0
)

window.resizable(False, False)
window.mainloop()
