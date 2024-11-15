from simplegrad import Space
import math

EPSILON = 1e-6


## testleri sil teker teker yaz make test
def test_space_creation():
    x = Space(5.0)
    assert abs(x.data() - 5.0) < EPSILON
    assert abs(x.grad() - 0.0) < EPSILON


def test_space_addition():
    x = Space(5.0)
    y = Space(3.0)
    z = x + y
    assert abs(z.data() - 8.0) < EPSILON
    assert abs(z.grad() - 0.0) < EPSILON


def test_space_subtraction():
    x = Space(5.0)
    y = Space(3.0)
    z = x - y
    assert abs(z.data() - 2.0) < EPSILON
    assert abs(z.grad() - 0.0) < EPSILON


def test_space_multiplication():
    x = Space(5.1156)
    y = Space(3.5225)
    z = x * y
    assert abs(z.data() - 5.1156 * 3.5225) < EPSILON
    assert abs(z.grad() - 0.0) < EPSILON


def test_space_division():
    x = Space(5.0)
    y = Space(0.0)
    z = x / y
    assert abs(z.data() - 5.0 / 1e-9) < EPSILON
    assert abs(z.grad() - 0.0) < EPSILON

def test_compound_ops():
    a = Space(5.0)
    b = Space(3.0)
    
    d = a * b
    d = d + 1.0
    assert abs(d.data() - 16.0) < EPSILON


def test_relu_ops():
    a = Space(-5.0)
    b = Space(3.0)
    sum_ab = b + a
    relu_result = sum_ab.relu()
    assert abs(relu_result.data() - 0.0) < EPSILON



