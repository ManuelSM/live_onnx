import numpy as np
import onnxruntime as rt
from PIL import Image
import os

def preprocessTF(image, resize_size:tuple=(224,224), float_mod:bool=False):

    image = image.resize(resize_size)


    if not float_mod:
        np_image = np.array(image, dtype=np.uint8)
        return np.expand_dims(np_image, axis=0)

    np_image = np.array(image)

    if float_mod:
        norm_img_data = (np_image - np.min(np_image))/(np.max(np_image)-np.min(np_image))
        norm_img_data = np.array(norm_img_data, dtype=np.float32)
    else:
        norm_img_data = np.zeros(np_image.shape).astype('uint8')

    np_image = np.expand_dims(norm_img_data, axis=0)
    return np_image


def livenessRS(img):
    sessionRS = rt.InferenceSession("./models/farRS.onnx")
    input_name = sessionRS.get_inputs()[0].name
    output_name = sessionRS.get_outputs()[0].name
    img_np = preprocessTF(img)
    out = sessionRS.run([output_name], {input_name: img_np})

    return out[0].flatten()[0]/255


def livenessRI(img):

    sessionRI = rt.InferenceSession("./models/farRI_v2.onnx")
    input_name = sessionRI.get_inputs()[0].name
    output_name = sessionRI.get_outputs()[0].name
    img_np = preprocessTF(img)
    out = sessionRI.run([output_name], {input_name: img_np})

    return out[0].flatten()[0]/255


if __name__ == "__main__":
    images_path = './images'
    list_images = os.listdir(images_path)

    for img in list_images:
        load_img = Image.open(os.path.join(images_path, img))
        
        RS_res = livenessRS(load_img)
        RI_res = livenessRI(load_img)

        print(f"RS: {RS_res}, RI: {RI_res}")
