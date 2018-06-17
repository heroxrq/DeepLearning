import os

# --------------------------------------------------
# path config
# --------------------------------------------------
BASE_DIR = os.path.abspath("..")

DATA_DIR = BASE_DIR + "/data"
METADATA_DIR = DATA_DIR + "/metadata"
INPUT_IMGS_DIR = DATA_DIR + "/input_imgs"
OUTPUT_BORDER_IMGS_DIR = DATA_DIR + "/output_border_imgs"
OUTPUT_SQUARE_IMGS_DIR = DATA_DIR + "/output_square_imgs"

MODEL_DIR = BASE_DIR + "/model"
BEST_MODEL_FILE = MODEL_DIR + '/best_model.hdf5'
BEST_WEIGHTS_FILE = MODEL_DIR + '/best_weights.hdf5'
MODEL_FILE = MODEL_DIR + "/model.json"

# --------------------------------------------------
# model config
# --------------------------------------------------
TRAIN_BATCH_SIZE = 8
EPOCHS = 200
