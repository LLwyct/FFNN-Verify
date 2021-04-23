import numpy as np
from numpy import ndarray
# from Layer import InputLayer

def compute_lower(weights_plus, weights_neg, input_lb, input_ub):
    # l_hat = w_+ * l_{i-1} + w_- * u_{i-1}
    return weights_plus.dot(input_lb) + weights_neg.dot(input_ub)

def compute_upper(weights_plus, weights_neg, input_lb, input_ub):
    # u_hat = w_+ * u_{i-1} + w_- * l_{i-1}
    return weights_plus.dot(input_ub) + weights_neg.dot(input_lb)

class LinearFunctions:
    def __init__(self, matrix: ndarray, offset: ndarray):
        self.size: int = matrix[0].size
        self.matrix: ndarray = matrix
        self.offset: ndarray = offset

    def computeMaxBoundsValue(self, inputLayer):
        input_ub = inputLayer.var_bounds_out["ub"]
        input_lb = inputLayer.var_bounds_out["lb"]

        weights_plus = np.maximum(self.matrix, np.zeros(self.matrix.shape))
        weights_neg = np.minimum(self.matrix, np.zeros(self.matrix.shape))

        # u_hat = w_+ * u_{i-1} + w_- * l_{i-1}
        return compute_upper(weights_plus, weights_neg, input_lb, input_ub) + self.offset

    def computeMinBoundsValue(self, inputLayer):
        input_ub = inputLayer.var_bounds_out["ub"]
        input_lb = inputLayer.var_bounds_out["lb"]

        weights_plus = np.maximum(self.matrix, np.zeros(self.matrix.shape))
        weights_neg = np.minimum(self.matrix, np.zeros(self.matrix.shape))

        # l_hat = w_+ * l_{i-1} + w_- * u_{i-1}
        return compute_lower(weights_plus, weights_neg, input_lb, input_ub) + self.offset

    '''def computeMinValues(self, inputLayer):
        input_lower = inputLayer.var_bounds_out["lb"]
        input_upper = inputLayer.var_bounds_out["ub"]
        weights_plus = np.maximum(self.matrix, np.zeros(self.matrix.shape))
        weights_neg = np.minimum(self.matrix, np.zeros(self.matrix.shape))
        return compute_lower(weights_plus, weights_neg, input_lower, input_upper) + self.offset

    def computeMaxValues(self, inputLayer):
        input_lower = inputLayer.var_bounds_out["lb"]
        input_upper = inputLayer.var_bounds_out["ub"]
        weights_plus = np.maximum(self.matrix, np.zeros(self.matrix.shape))
        weights_neg = np.minimum(self.matrix, np.zeros(self.matrix.shape))
        return compute_upper(weights_plus, weights_neg, input_lower, input_upper) + self.offset'''

    def getUpperOutEqThroughRelu(self, inputLayer):
        lower = self.computeMinBoundsValue(inputLayer)
        upper = self.computeMaxBoundsValue(inputLayer)
        matrix = self.matrix
        offset = self.offset

        for i in range(self.size):
            if upper[i] <= 0:
                matrix[i, :] = 0
                offset[i] = 0
            elif lower >= 0:
                continue
            else:
