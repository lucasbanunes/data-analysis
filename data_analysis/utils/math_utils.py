"""This model contains mathemtical utilities for the package"""

import numpy as np

def euclidean_algorithm(a, b):
    """Computes the GCD between two integers using the euclidean algorithm"""
    a = int(a)
    b = int(b)
    divided = max(a, b)
    divisor = min(a, b)
    remainder = divided%divisor
    if remainder == 0:
        return divisor
    else:
        return euclidean_algorithm(divisor, remainder)

def trapezoid_integration(x, y):
    """Integrates the function given by the discrete sequences of x and y using the trapezoid rule"""
    area=0
    for k in range(1,len(x)):
        dx = x[k] - x[k-1]
        bases = y[k] + y[k-1]
        area += bases*dx/2

    return area