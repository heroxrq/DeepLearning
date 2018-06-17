import datetime
import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from dataset_util import train_data_generator
from config import *
import numpy as np

# callbacks = [EarlyStopping(monitor='mae',
#                            patience=10,
#                            verbose=1,
#                            min_delta=0.0001,
#                            mode='min'),
#              ReduceLROnPlateau(monitor='mae',
#                                factor=0.5,
#                                patience=3,
#                                verbose=1,
#                                epsilon=0.0001,
#                                cooldown=0,
#                                mode='min'),
#              ModelCheckpoint(monitor='val_dice_coef',
#                              filepath=BEST_WEIGHTS_FILE,
#                              save_best_only=True,
#                              save_weights_only=True,
#                              verbose=1,
#                              mode='max'),
#              TensorBoard(log_dir=TF_LOG_DIR)]


def train():
    start_time = datetime.datetime.now()

    all_train_images = os.listdir(OUTPUT_SQUARE_IMGS_DIR)
    train_images, validation_images = train_test_split(all_train_images, train_size=0.8, test_size=0.2, random_state=42)

    print("Number of train_images: {}".format(len(train_images)))
    print("Number of validation_images: {}".format(len(validation_images)))

    train_gen = train_data_generator(OUTPUT_SQUARE_IMGS_DIR, train_images, TRAIN_BATCH_SIZE, augment=False)
    validation_gen = train_data_generator(OUTPUT_SQUARE_IMGS_DIR, validation_images, TRAIN_BATCH_SIZE, augment=False)


    from keras.applications.inception_v3 import InceptionV3
    from keras.preprocessing import image
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D
    from keras import backend as K

    # create the base pre-trained model
    base_model = InceptionV3(weights=None, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(256, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(1, activation='relu')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)


    model.compile(optimizer=RMSprop(lr=0.0005), loss='mse', metrics=['mae'])
    save_model(model, MODEL_FILE)




    steps_per_epoch = np.ceil(len(train_images) / TRAIN_BATCH_SIZE)
    validation_steps = np.ceil(len(validation_images) / TRAIN_BATCH_SIZE)
    print("steps_per_epoch:", steps_per_epoch)
    print("validation_steps:", validation_steps)

    model.fit_generator(generator=train_gen, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                        # callbacks=callbacks,
                        validation_data=validation_gen, validation_steps=validation_steps)

    end_time = datetime.datetime.now()
    cost_time = end_time - start_time
    print("train cost time:", cost_time)

    return model


def save_model(model, model_file):
    model_json_string = model.to_json()
    with open(model_file, 'w') as mf:
        mf.write(model_json_string)


if __name__ == '__main__':
    train()
