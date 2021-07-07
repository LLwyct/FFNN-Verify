from keras.models import load_model
import numpy as np
import os


input_mean_values = np.array([1.9791091e+04, 0.0, 0.0, 650.0, 600.0])
input_ranges = np.array([60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0])
output_mean = 7.5188840201005975
output_range = 373.94992


def acas_denormalise_output(value):
    return value * output_range + output_mean

if __name__ == "__main__":
    input = [
        [
            -0.30353115613746867,
            0.009139156821477343,
            0.4933803235848431,
            0.3,
            0.34368909377181106
        ]
    ]
    input1 = [
        [
            -0.29855281193475053,
            -0.004758424582022636,
            0.49999989597792077,
            0.3,
            0.39474407455536803
        ]
    ]

    networkFileName = "acas_1_1.h5"
    networkFilePath = os.path.abspath(os.path.join("../../resources/Acas", networkFileName))
    print(networkFilePath)
    model = load_model(networkFilePath)
    Y = model.predict(np.array(input1))
    print(Y)
    denor_Y = acas_denormalise_output(Y)
    print(denor_Y)
    '''
    y_0 -0.00829189424779031
    y_1 -0.008291894247792938
    y_2 -0.008291894247792935
    y_3  0.0006895530961594492
    y_4 -0.006488597509305857
    X_0 -0.30353115613746867
    X_1 0.009139156821477343
    X_2 0.4933803235848431
    X_3 0.3
    X_4 0.34368909377181106
    '''