import os
import uuid
import flask
import urllib

import tensorflow as tf
import skimage
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, send_file
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np
app = Flask(__name__)
model = load_model('D:\\FYP\\py\\gan_graphics_real\\MCcolor\\mceffnet_model.h5')
print("Model summary:")
model.summary()

# 检查模型权重
print("\nChecking model weights...")
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        print(f"\nLayer: {layer.name}")
        for i, w in enumerate(weights):
            print(f"Weight {i} shape: {w.shape}")
            print(f"Weight {i} min: {w.min()}, max: {w.max()}")
            print(f"Weight {i} mean: {w.mean()}")
ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXT


classes = ['REAL', 'COMPUTER GRAPHICS', 'GAN']


def scale0to255(image):
    converted_image = image.copy()
    min_1 = np.min(converted_image[:, :, 0])
    max_1 = np.max(converted_image[:, :, 0])
    converted_image[:, :, 0] = np.round(((converted_image[:, :, 0] - min_1) / (max_1 - min_1)) * 255)

    min_2 = np.min(converted_image[:, :, 1])
    max_2 = np.max(converted_image[:, :, 1])
    converted_image[:, :, 1] = np.round(((converted_image[:, :, 1] - min_2) / (max_2 - min_2)) * 255)

    min_3 = np.min(converted_image[:, :, 2])
    max_3 = np.max(converted_image[:, :, 2])
    converted_image[:, :, 2] = np.round(((converted_image[:, :, 2] - min_3) / (max_3 - min_3)) * 255)
    return converted_image


def colorFunction2(image):
    color_transf_image = skimage.color.rgb2hsv(image)
    scaled_image = scale0to255(color_transf_image)
    return scaled_image


def colorFunction1(image):
    color_transf_image = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
    scaled_image = scale0to255(color_transf_image)
    return scaled_image


def prepare_inputs(img_path):
    # 加载图像
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array.astype('float32')

    # RGB输入
    rgb_input = img_array.copy()
    rgb_input = rgb_input / 255.0
    rgb_input = np.expand_dims(rgb_input, axis=0)  # 添加batch维度

    # XYZ输入 (对应训练时的colorFunction1)
    xyz_input = colorFunction1(img_array.astype(np.uint8))
    xyz_input = xyz_input.astype('float32')
    xyz_input = xyz_input / 255.0
    xyz_input = np.expand_dims(xyz_input, axis=0)  # 添加batch维度

    # HSV输入 (对应训练时的colorFunction2)
    hsv_input = colorFunction2(img_array.astype(np.uint8))
    hsv_input = hsv_input.astype('float32')
    hsv_input = hsv_input / 255.0
    hsv_input = np.expand_dims(hsv_input, axis=0)  # 添加batch维度

    # 打印输入形状用于调试
    print("Input shapes:")
    print(f"RGB shape: {rgb_input.shape}")
    print(f"XYZ shape: {xyz_input.shape}")
    print(f"HSV shape: {hsv_input.shape}")

    return [rgb_input, xyz_input, hsv_input]



def predict(filename, model):
    # 准备输入
    inputs = prepare_inputs(filename)

    # 打印输入形状和值范围用于调试
    print("\nInput shapes and ranges:")
    for i, inp in enumerate(inputs):
        print(f"Input {i} shape: {inp.shape}")
        print(f"Input {i} min: {inp.min()}, max: {inp.max()}")
        print(f"Input {i} mean: {inp.mean()}\n")

    # 进行预测
    result = model.predict(inputs)
    print("\nRaw prediction result:", result)

    # 获取预测结果
    predictions = result[0]

    # 创建类别和概率的对应关系
    class_prob = list(zip(classes, predictions))

    # 按概率排序
    class_prob.sort(key=lambda x: x[1], reverse=True)

    # 获取前三个结果
    top3 = class_prob[:3]

    # 分离类别和概率
    class_result = [item[0] for item in top3]
    prob_result = [(item[1] * 100).round(2) for item in top3]

    return class_result, prob_result


def predict(filename, model):
    inputs = prepare_inputs(filename)
    print("Input shapes:", [i.shape for i in inputs])

    result = model.predict(inputs)
    print("Raw predictions:", result)

    # 获取预测结果
    predictions = result[0]
    print("Predictions:", predictions)

    # 创建类别和概率的对应关系
    class_prob = list(zip(classes, predictions))
    print("Class probabilities:", class_prob)

    # 按概率排序
    class_prob.sort(key=lambda x: x[1], reverse=True)

    # 获取前三个结果
    top3 = class_prob[:3]
    print("Top 3 results:", top3)

    # 分离类别和概率
    class_result = [item[0] for item in top3]
    prob_result = [(item[1] * 100).round(2) for item in top3]

    return class_result, prob_result


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/images')
    if request.method == 'POST':
        if (request.form):
            link = request.form.get('link')
            try:
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                output = open(img_path, "wb")
                output.write(resource.read())
                output.close()
                img = filename
                class_result, prob_result = predict(img_path, model)
                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                }
            except Exception as e:
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'
            if (len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)

        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                img = file.filename
                class_result, prob_result = predict(img_path, model)
                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                }
            else:
                error = "Please upload images of jpg , jpeg and png extension only"
            if (len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)