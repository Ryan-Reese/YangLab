import numpy, scipy
from numpy import random, linalg

def zero_counter(matrix, matrix_size, tolerance):
    boolean = numpy.isclose(matrix, numpy.zeros(matrix_size), atol=tolerance)
    print(boolean)
    return numpy.count_nonzero(boolean)

def huckel_constructor(matrix_size, alpha, beta):
    middle = numpy.diag(numpy.full(matrix_size, alpha))
    upper_lower = numpy.diag(numpy.full(matrix_size-1, beta), k=1) + numpy.diag(numpy.full(matrix_size-1, beta), k=-1)
    return middle + upper_lower

def main():
    huckel = huckel_constructor(5, 1, 2)
    print(huckel)
    invhuckel = linalg.inv(huckel)
    print(invhuckel)
    product = numpy.matmul(huckel, invhuckel)
    print(product)
    print(zero_counter(product, product.shape[0], 1e-10))

main()


