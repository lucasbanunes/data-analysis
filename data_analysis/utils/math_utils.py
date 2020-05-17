"""This model contains mathemtical utilities for the package"""

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