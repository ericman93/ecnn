import os
import pickle
import numpy as np
from PIL import Image
from cnn.network import CnnNetwork
from cnn.steps import ConvolutionalStep
from cnn.steps import ReluActivation
from cnn.steps import MaxPoolingStep
from cnn.steps import OutputStep
from cnn.steps import Flatten
from cnn.network import CnnNetwork
from cnn.steps.activation import Sigmoid
from cnn.steps.activation import Relu
from cnn.steps.activation import Tanh
from cnn.steps.activation import Softmax
from cnn.cost_functions import MeanSquared
from cnn.cost_functions import CrossEntropyLogisticRegression
from cnn.cost_functions import CrossEntropy

from tensorflow.python import debug as tf_debug

cards_dir = '/home/eric/dev/pokemon/cards/'
categories = {'trainer': 0, 'pokemon': 1, 'energy': 2}
image_size = (150, 110)
# image_size = (150, 150)


def save_steps(inputs, network):
    data = network.to_3d_shape(np.array(inputs))
    for i, step in enumerate(network.steps):
        data = step.forward_propagation(data)

        if len(data.shape) == 3:
            for j in range(data.shape[0]):
                filter = data[j, :, :].dot(200)
                shape = filter.shape
                filter = np.array([np.uint8(px) for px in filter.reshape(filter.size)]).reshape(shape)

                a = 2

                file_name = f"step#{i}@{step.__class__.__name__}_filter#{j}"
                Image.fromarray(filter).save(f'output/{file_name}.bmp')

    return data


def get_image_matrix(image_path, size, graysacle=False):
    image = Image.open(image_path)

    revese_size = (size[1], size[0])
    image.thumbnail(revese_size, Image.ANTIALIAS)

    if graysacle:
        return np.asarray(image.convert("L"))
    else:
        result = np.zeros((3,) + size)
        matrix = np.asarray(image)
        red = matrix[:, :, 0]
        green = matrix[:, :, 1]
        blue = matrix[:, :, 2]

        # Image.fromarray(red).save('red.bmp')
        # Image.fromarray(green).save('green.bmp')
        Image.fromarray(blue).save('blue.bmp')

        merged = np.stack([red, green, blue], axis=0)

        result[:merged.shape[0], :merged.shape[1], :merged.shape[2]] = merged
        return result.dot(1 / 256)
        # return result

        # return merged.dot(1 / 256)


def get_cards(X, y):
    print("Getting cards")

    for category, tag in categories.items():
        print(category)
        file_names = os.listdir(os.path.join(cards_dir, category))
        category_x = []
        category_y = []

        # change to 900
        for index, image_path in enumerate(file_names[0:900]):
            # for index, image_path in enumerate(file_names):
            image_path = os.path.join(cards_dir, category, image_path)

            image_matrix = get_image_matrix(image_path, image_size)

            category_x.append(image_matrix)
            category_y.append(tag)

        # When taking 900 pictures
        if category == 'energy':
            category_x = category_x * 7 # change to 7 if 900
            category_y = category_y * 7

        # if tag == 2:
        X += category_x
        y += category_y


def validate(save=None):
    network = CnnNetwork([])
    network.load('pokemon-cards.b')
    # network.load('network.b')

    # dir = os.path.join(cards_dir, 'test')
    dir = 'validation/v2'

    file_names = os.listdir(dir)
    for file_name in file_names:
        if '.' not in file_name:
            continue

        image_path = os.path.join(dir, file_name)

        image_matrix = get_image_matrix(image_path, image_size)

        if save and file_name == save:
            save_steps(image_matrix, network)

         # print('--')
        predictions = network.forward_propagation(image_matrix)
        # print(predictions)
        predicted_class, score = network.predict(image_matrix)
        class_name = [name for name, value in categories.items() if value == predicted_class][0]

        print(f"{file_name} - {class_name} ({predictions})")
        # print('--')
        # card_type = [k for k,v in categories.items() if v == prediction[0]][0]
        # print(f"{file_name} is a {card_type} ({prediction[1]})")


def keras_fit():
    import keras

    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    X = []
    y = []
    get_cards(X, y)

    sample = 5
    y = np.atleast_2d(keras.utils.to_categorical(y)[sample])
    X = np.atleast_2d(X[sample].reshape(X[sample].size))

    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(3, input_dim=X.size, activation='softmax', use_bias=True,
                           kernel_initializer=keras.initializers.Zeros())
    )
    sgd = keras.optimizers.SGD(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    model.fit(X, y, epochs=10, shuffle=True, verbose=2)


def tensorflow_fit():
    import tensorflow as tf

    X = []
    y = []
    get_cards(X, y)

    sample = 5
    weights = np.ones((3, X[sample].size))
    X = X[sample].reshape(X[sample].size)
    Y = y[sample]

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    train_op = optimizer.minimize(loss_op)


def fit():
    X = []
    y = []
    get_cards(X, y)

    filter_size = (6, 6)
    padding = 0
    stride = 1
    x0 = 'random'

    steps = [
        ConvolutionalStep(filter_size=filter_size, padding=padding, stride=stride, num_of_kernels=4, x0=x0,
                          activation=Relu),
        MaxPoolingStep(3),
        ConvolutionalStep(filter_size=filter_size, padding=padding, stride=stride, num_of_kernels=7, x0=x0,
                          activation=Relu),
        MaxPoolingStep(3),
        # ConvolutionalStep(filter_size=(3, 3), stride=3, num_of_kernels=3, x0='random', activation=Relu),
        # MaxPoolingStep(2),
        # ConvolutionalStep(filter_size=(3, 3), num_of_kernels=6, x0='random', activation=Relu),
        # MaxPoolingStep(3),
        # ConvolutionalStep(filter_size=filter_size, padding=padding, stride=stride, num_of_kernels=4, x0='random', activation=Relu),
        # MaxPoolingStep(3),
        Flatten(),
        OutputStep(x0=x0, activation=Softmax)
    ]

    # network = CnnNetwork(steps)
    network = CnnNetwork([])
    network.load('pokemon-cards.b')

    # network.fit(X, y, MeanSquared, iterations=1000, batch_size=32)

    # network.load('network.b')
    sample = 4
    network.fit(
        # [X[sample]], [y[sample]],
        X, y,
        CrossEntropy,
        iterations=100,
        batch_size=len(X),
        learning_rate=1e-5,
        verbose=1,

    )
    # print("start fit")
    # network.fit(X, y, MeanSquared, iterations=15, batch_size=int(len(X)), learning_rate=0.00001)
    # network.save('network.b')
    # network.save('network.b')


def get_biggets_sizes():
    height = width = 0
    for category, tag in categories.items():
        file_names = os.listdir(os.path.join(cards_dir, category))

        for index, image_path in enumerate(file_names[0:900]):
            image_path = os.path.join(cards_dir, category, image_path)
            image = Image.open(image_path)

            if image.size[1] > height:
                height = image.size[1]

            if image.size[0] > width:
                width = image.size[0]

    print(f"{height}, {width}")
    print(f"Ratio {width/height}")
    size = 150
    print(f"if height: {size} then width is {size*(width/height)}")


if __name__ == '__main__':
    # get_biggets_sizes()
    fit()
    # tensorflow_fit()
    # keras_fit()
    # validate(save='p2.jpg')
