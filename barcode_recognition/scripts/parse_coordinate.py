import os


def parse_coordinate(coordinate_fname, sep=" ", skip_header=True):

    with open(coordinate_fname) as f:
        for line in f:
            if skip_header:
                skip_header = False
                continue

            fields = line.strip("\n").split(",")

            image_name = fields[0]
            num_barcode = int(fields[1])
            coordinate = fields[2]

            coordinates = []
            barcode_coordinate = []
            for index, value in enumerate(coordinate.split(sep)):
                barcode_coordinate.append(int(value))

                if index % 8 == 7:
                    coordinates.append(barcode_coordinate)
                    barcode_coordinate = []

            yield (image_name, num_barcode, coordinates)


if __name__ == "__main__":
    metadata_dir = "/home/xrq/prog/DeepLearning/barcode_recognition/data/metadata"
    for image_name_, num_barcode_, coordinates_ in parse_coordinate(metadata_dir + os.sep + "merged_metadata.csv"):
        print(image_name_, num_barcode_, coordinates_)
