import numpy as np

acas_properties = [
    {},
    # 1: If the intruder is distant and is significantly slower than the ownship,
    # the score of a COC advisory will always be below a certain fixed threshold
    {
        "name": "Property 1",
        "input": "",
        "output": "",
        "raw_bounds": {
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
        "raw_bounds": {
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
        "raw_bounds": {
            "lower": [1500, -0.06, 3.10, 980, 960],
            "upper": [1800, 0.06, 3.141592, 1200, 1200]
        },
    }
]

input_mean_values = np.array([1.9791091e+04, 0.0, 0.0, 650.0, 600.0])
input_ranges = np.array([60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0])
output_mean = 7.5188840201005975
output_range = 373.94992

def acas_normalise_input(values):
    return (values -input_mean_values)/input_ranges


def acas_denormalise_input(values):
    return values * input_ranges + input_mean_values


def acas_normalise_output(value):
    return (value - output_mean) / output_range


def acas_denormalise_output(value):
    return value * output_range + output_mean


def getNormaliseInput(property = 0) :
    return [
        acas_normalise_input(acas_properties[property]["raw_bounds"]["lower"]),
        acas_normalise_input(acas_properties[property]["raw_bounds"]["upper"])
        ]
