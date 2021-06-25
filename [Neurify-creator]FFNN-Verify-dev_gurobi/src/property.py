import numpy as np
from typing import List, Dict
from ConstraintFormula import *
'''
全局约束，这是这五个input首先都不能违反的外界条件下的约束，其次才是要验证安全属性的约束
0<= input[0] <= 62000
-3.141592<= input[1] <= +3.141692
-3.141592<= input[2] <= +3.141692
0<= input[3] <= 1200
0<= input[4] <= 1200
'''

acas_properties: List[Dict] = [
    {},
    # 1: If the intruder is distant and is significantly slower than the ownship,
    # the score of a COC advisory will always be below a certain fixed threshold
    {
        "name": "Property 1",
        "input": "",
        "output": "",
        "input_bounds": {
            "lower": [55947.691, -3.141592, -3.141592, 1145, 0],
            "upper": [62000, 3.141592, 3.141592, 1200, 60]
        }
    },

    # 2: If the intruder is distant and significantly slower than the ownship,
    # the score of a COC advisory will never be maximal
    #
    # Output constrains: the score for COC is not the maximal score
    {
        "name": "Property 2",
        "input": "",
        "output": "",
        "input_bounds": {
            "lower": [55947.691, -3.141592, -3.141592, 1145, 0],
            "upper": [62000, 3.141592, 3.141592, 1200, 60]
        }
    },

    # 3: If the intruder is directly ahead and is moving towards the ownship,
    # the score for COC will not be minimal
    #
    # Output constraints: the score for COC is not the minimal score
    {
        "name": "Property 3",
        "input": "",
        "output": "",
        "input_bounds": {
            "lower": [1500, -0.06, 3.10, 980, 960],
            "upper": [1800, 0.06, 3.141592, 1200, 1200]
        },
        "outputConstraints": [
            Disjunctive (
                [
                    ("VarVar", "Y0", "GT", "Y1"),
                    ("VarVar", "Y0", "GT", "Y2"),
                    ("VarVar", "Y0", "GT", "Y3"),
                    ("VarVar", "Y0", "GT", "Y4"),
                ]
            )
        ]
    },
    {
        "name": "Property 4",
        "input": "",
        "output": "",
        "input_bounds": {
            "lower": [1500, -0.06, 0, 1000, 700],
            "upper": [1800, 0.06, 3.141592, 1200, 800]
        }
    },

    # 5: If the intruder is near and approaching from the left,
    # the network advises "strong right"
    #
    # Output constraints: the score for "strong right" is the minimal score
    {
        "name": "Property 5",
        "input": "",
        "output": "",
        "input_bounds": {
            "lower": [250, 0.2, -3.141592, 100, 0],
            "upper": [400, 0.4, -3.141592+0.005, 400, 400]
        },
        "outputConstraints": [
            Conjunctive (
                [
                    ("VarVar", "Y4", "LT", "Y0"),
                    ("VarVar", "Y4", "LT", "Y1"),
                    ("VarVar", "Y4", "LT", "Y2"),
                    ("VarVar", "Y4", "LT", "Y3"),
                ]
            )
        ]
    },
]

input_mean_values = np.array([1.9791091e+04, 0.0, 0.0, 650.0, 600.0])
input_ranges = np.array([60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0])
output_mean = 7.5188840201005975
output_range = 373.94992

def acas_normalise_input(values):
    return (values - input_mean_values)/input_ranges


def acas_denormalise_input(values):
    return values * input_ranges + input_mean_values


def acas_normalise_output(value):
    return (value - output_mean) / output_range


def acas_denormalise_output(value):
    return value * output_range + output_mean


def getNormaliseInput(property = 0) -> list:
    return [
        acas_normalise_input(np.array(acas_properties[property]["input_bounds"]["lower"])),
        acas_normalise_input(np.array(acas_properties[property]["input_bounds"]["upper"]))
    ]
