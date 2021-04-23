from keras.models import load_model
import numpy as np
import os
from keras import Model

if __name__ == "__main__":
    input = [
        -0.30353115613746867,
        0.009139156821477343,
        0.4933803235848431,
        0.3,
        0.34368909377181106
    ]

    networkFileName = "acas_1_6.h5"
    networkFilePath = os.path.abspath(os.path.join("../../resources/Acas", networkFileName))
    print(networkFilePath)
    model = load_model(networkFilePath)
    assert isinstance(model, Model)
    model.predict(np.array(input))

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