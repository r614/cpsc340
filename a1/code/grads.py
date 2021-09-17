import numpy as np


def example(x):
    return np.sum(x ** 2)


def example_grad(x):
    return 2 * x


def foo(x):
    result = 1
    λ = 4  # this is here to make sure you're using Python 3
    # ...but in general, it's probably better practice to stick to plaintext
    # names. (Can you distinguish each of λ𝛌𝜆𝝀𝝺𝞴 at a glance?)
    for x_i in x:
        result += x_i ** λ
    return result


def foo_grad(x):
    result = [] 
    for x_i in x: 
        result.append(float("{:.8f}".format(4*(x_i ** 3))))
    return result


def bar(x):
    return np.prod(x)


def bar_grad(x):
    p = np.prod(x)
    res = []
    for num in x: 
        res.append(float("{:.8f}".format(p/num))) if num != 0 else 1
    return res
# Hint: This is a bit tricky - what if one of the x[i] is zero?
