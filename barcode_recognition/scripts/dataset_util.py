import random

import numpy as np
from keras.preprocessing import image


def load_img_array(image_name, grayscale=False, target_size=None):
    img = image.load_img(image_name, grayscale, target_size)
    img_array = image.img_to_array(img) / 255.0
    return img_array


def train_data_generator(data_dir, image_names, batch_size, target_size=(224, 224), augment=False, angle_pos=3):
    img_cnt = 0
    while True:
        random.shuffle(image_names)

        for start in range(0, len(image_names), batch_size):
            end = min(start + batch_size, len(image_names))
            image_names_batch = image_names[start: end]

            image_batch = []
            y_batch = []
            for image_name in image_names_batch:
                # image
                image_pathname = data_dir + "/" + image_name
                image_array = load_img_array(image_pathname, grayscale=False, target_size=target_size)

                angle = image_name.split("_")[angle_pos]
                y = float(angle)

                if augment:
                    pass

                image_batch.append(image_array)
                y_batch.append(y)

                img_cnt += 1
            image_batch = np.array(image_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            yield image_batch, y_batch
