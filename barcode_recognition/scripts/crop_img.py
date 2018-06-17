import os

from PIL import Image
from parse_coordinate import parse_coordinate
from calculate_angle import calculate_angle


def crop_border_img(image_name, num_barcode, coordinates, input_dir, output_dir):
    img = Image.open(input_dir + os.sep + image_name)
    assert num_barcode == len(coordinates), \
        "Number of num_barcode does not match the number of coordinates."

    for i in range(num_barcode):
        barcode_coordinate = coordinates[i]
        assert 8 == len(barcode_coordinate), \
            "Number of barcode_coordinate is not correct."

        x1 = min(barcode_coordinate[0::2])
        y1 = min(barcode_coordinate[1::2])
        x2 = max(barcode_coordinate[0::2])
        y2 = max(barcode_coordinate[1::2])

        cropped_img = img.crop((x1, y1, x2, y2))

        angle = calculate_angle(barcode_coordinate)

        cropped_image_name = image_name.split(".")[0] + "_" + str(i) + "_" + str(angle) + "." + image_name.split(".")[1]
        cropped_img.save(output_dir + os.sep + cropped_image_name)


def crop_square_img(image_name, image_dir, cropped_image_dir):
    img = Image.open(image_dir + os.sep + image_name)
    w, h = img.size

    cropped_size = min(img.size)

    if w == cropped_size:
        x1 = 0
        x2 = w
        y1 = (h - w) // 2
        y2 = y1 + cropped_size
    else:
        y1 = 0
        y2 = h
        x1 = (w - h) // 2
        x2 = x1 + cropped_size

    cropped_img = img.crop((x1, y1, x2, y2))

    cropped_image_name = '.'.join(image_name.split(".")[:-1]) + "_" + "square" + "." + image_name.split(".")[-1]
    cropped_img.save(cropped_image_dir + os.sep + cropped_image_name)


def crop_all_img():
    pass


if __name__ == '__main__':
    metadata_dir = "/home/xrq/prog/DeepLearning/barcode_recognition/data/metadata"
    input_imgs_dir = "/home/xrq/prog/DeepLearning/barcode_recognition/data/input_imgs"
    output_border_imgs_dir = "/home/xrq/prog/DeepLearning/barcode_recognition/data/output_border_imgs"
    output_square_imgs_dir = "/home/xrq/prog/DeepLearning/barcode_recognition/data/output_square_imgs"

    for image_name_, num_barcode_, coordinates_ in parse_coordinate(metadata_dir + os.sep + "merged_metadata.csv"):
        crop_border_img(image_name_, num_barcode_, coordinates_, input_imgs_dir, output_border_imgs_dir)

    for image_name_ in os.listdir(output_border_imgs_dir):
        crop_square_img(image_name_, output_border_imgs_dir, output_square_imgs_dir)
