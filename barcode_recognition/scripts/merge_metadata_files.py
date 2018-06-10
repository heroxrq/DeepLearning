import os


def merge_metadata_files(image_name_fname, coordinate_fname, output_fname, sep1=" ", sep2=" "):
    image_names = []
    num_barcodes = []
    coordinates = []

    with open(image_name_fname) as f:
        for line in f:
            image_name = line.strip('\n') + ".jpg"
            image_names.append(image_name)

    with open(coordinate_fname) as f:
        for line in f:
            coordinate = []
            fields = line.strip('\n').split(sep1)
            for index, value in enumerate(fields):
                if index == 0:
                    num_barcodes.append(value)
                elif index % 10 == 0 or index % 10 == 9:
                    pass
                else:
                    coordinate.append(value)
            coordinates.append(coordinate)

    assert len(image_names) == len(num_barcodes) and len(image_names) == len(coordinates), \
        "The number of lines in the two files is not equal!"

    with open(output_fname, 'w') as f:
        header = "image_name" + "," + "num_barcode" + "," + "coordinate" + "\n"
        f.write(header)
        for i in range(len(image_names)):
            out_line = image_names[i] + "," + num_barcodes[i] + "," + sep2.join(coordinates[i]) + "\n"
            f.write(out_line)


if __name__ == "__main__":
    metadata_dir = "/home/xrq/prog/DeepLearning/barcode_recognition/data/metadata"
    merge_metadata_files(metadata_dir + os.sep + "image_name_metadata.txt",
                         metadata_dir + os.sep + "coordinate_metadata.txt",
                         metadata_dir + os.sep + "merged_metadata.csv")
