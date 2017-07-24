from steps.convolutional_step import ConvolutionalStep
import numpy as np

if __name__ == '__main__':
    # s = ConvolutionalStep((5, 5), 4)
    images = np.array(
        [
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]
            ],
            [
                [13, 14, 15, 16],
                [17, 18, 19, 20],
                [20, 21, 22, 23]
            ],
        ]
    )

    print(images.shape)

    # s.forward_propagation()
