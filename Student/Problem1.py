import numpy as np
import matplotlib.pyplot as plt


def norm_histogram(histogram):
    """
    takes a list of counts and converts to a list of probabilities, outputs the probability list.
    :param histogram: a numpy ndarray object
    :return: list
    """

    numSamples = 0
    normListofProbabilities = []

    for i in histogram:
        numSamples += i

    for n in histogram:
        normListofProbabilities.append(float(n) / numSamples)

    return normListofProbabilities

    


def compute_j(histogram, bin_width, num_samples):
    """
    takes list of counts, uses norm_histogram function to output the histogram of probabilities,
    then calculates compute_j for one specific bin width (reference: histogram.pdf page19)
    :param histogram: list
    :param bin_width: float
    :param num_samples: int
    :return: float
    """

    probabilities = norm_histogram(histogram)

    sumofProbabilitiesSquared = 0
    for p in probabilities:
        sumofProbabilitiesSquared = sumofProbabilitiesSquared + p**2

    j = ( 2 / ( (num_samples - 1) * bin_width ) ) - ( (num_samples + 1) / ( (num_samples - 1) * bin_width ) ) * sumofProbabilitiesSquared

    return j

def sweep_n(data, min_val, max_val, min_bins, max_bins):
    """
    find the optimal bin
    calculate compute_j for a full sweep [min_bins to max_bins]
    please make sure max_bins is included in your sweep

    The variable "data" is the raw data that still needs to be "processed"
    with matplotlib.pyplot.hist to output the histogram

    You must utilize the variables (data, min_val, max_val, min_bins, max_bins)
    in your code for 'sweep_n' to determine the correct input to the function 'matplotlib.pyplot.hist',
    specifically the values to (x, bins, range).
    Other input variables of 'matplotlib.pyplot.hist' can be set as default value.

    :param data: list
    :param min_val: int
    :param max_val: int
    :param min_bins: int
    :param max_bins: int
    :return: list
    """

    jList = []

    dataRange = max_val - min_val

    for numBins in range(min_bins, max_bins + 1):
        # Compute the bin width
        bin_width = dataRange / numBins

        hist, bin_edges, patches = plt.hist(data, bins=numBins, range=(min_val, max_val))

        numSamples = sum(hist)

        j = compute_j(hist, bin_width, numSamples)

        # Append the result to the list
        jList.append(j)

    return jList




def find_min(l):
    """
    takes a list of numbers and returns the three smallest number in that list and their index.
    return a dict i.e.
    {index_of_the_smallest_value: the_smallest_value, index_of_the_second_smallest_value: the_second_smallest_value, ...}

    For example:
        A list(l) is [14,27,15,49,23,41,147]
        Then you should return {0: 14, 2: 15, 4: 23}

    :param l: list
    :return: dict: {int: float}
    """

    threeSmallest = {}

    indexed = list(enumerate(l))
    for _ in range(3):
        smallI, smallV = min(indexed, key=lambda x: x[1])
        threeSmallest[smallI] = smallV
        indexed.remove((smallI, smallV))

    return threeSmallest