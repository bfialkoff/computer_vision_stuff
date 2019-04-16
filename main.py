from pathlib import Path

from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy
from keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16

from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt


def imshow(img):
    plt.imshow(img)
    plt.show()


def get_vgg():
    vgg = VGG16()
    model = Sequential()
    for layer in vgg.layers:
        layer.trainable = False
        model.add(layer)
    model.layers.pop()
    model.add(Dense(1, activation='sigmoid'))
    return model


def get_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(64, 64, 3)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    return model


def get_generators():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    training_set_gen = train_datagen.flow_from_directory(directory=training_path,
                                                         target_size=img_shape, class_mode='binary')
    test_set_gen = test_datagen.flow_from_directory(test_path, target_size=img_shape, shuffle=False,
                                                    class_mode='binary', batch_size=batch_size)

    return test_set_gen, training_set_gen


def train_model(model, should_load=True):
    test_set_gen, training_set_gen = get_generators()
    if weights_path.exists() and should_load:
        print('loading weights')
        model.load_weights(str(weights_path))
    else:
        check_point = ModelCheckpoint(str(weights_path), monitor='val_loss', verbose=0, save_best_only=False,
                                      save_weights_only=False, mode='auto', period=1)

        history = model.fit_generator(training_set_gen, steps_per_epoch=20, epochs=10,
                                      callbacks=[check_point],
                                      validation_data=test_set_gen,
                                      validation_steps=step_size // batch_size)
    return model


def confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = _confusion_matrix(y_true, y_pred).ravel()
    print(tp, fp)
    print(fn, tn)


if __name__ == '__main__':
    data_path = Path(__file__).joinpath('..', 'data').resolve()
    training_path = Path(str(data_path)).joinpath('training_set').resolve()
    test_path = Path(str(data_path)).joinpath('test_set').resolve()
    weights_path = Path(str(data_path)).joinpath('weights.hdf5').resolve()

    img_shape = (64, 64)
    batch_size = 4
    step_size = 12
    model = get_model()
    model = train_model(model)
    vgg = get_vgg()
    vgg = train_model(vgg, should_load=False)
    print(vgg.summary())
    test_set_gen, training_set_gen = get_generators()
    evaluation = model.evaluate_generator(generator=test_set_gen, steps=1)

    test_set_gen.reset()
    num_of_test_samples = 23

    num_steps = 2 * (num_of_test_samples // batch_size) + 1  # num_step * batch_size == num_val_imgs
    y_pred = model.predict_generator(generator=test_set_gen, steps=num_steps)
    mask = y_pred > 0.5
    y_pred[mask] = 1
    y_pred[~mask] = 0

    test_set_gen.reset()

    confusion_matrix(test_set_gen.classes, y_pred)
    print(classification_report(test_set_gen.classes, y_pred, target_names=['cat', 'dog']))
    print(evaluation)
