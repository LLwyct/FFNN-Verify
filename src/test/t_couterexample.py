from keras.models import load_model
import numpy as np
import os
from keras import Model

if __name__ == "__main__":
    input = [
        -0.29855281193475053,
        -0.009549296585513092,
        0.4981472954290255,
        0.5,
        0.3,
    ]

    networkFileName = "acas_1_7.h5"
    networkFilePath = os.path.abspath(os.path.join("../../resources/Acas", networkFileName))
    print(networkFilePath)
    model = load_model(networkFilePath)
    assert isinstance(model, Model)
    model.predict(np.array(input))