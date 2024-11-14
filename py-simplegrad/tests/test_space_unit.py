from simplegrad import Space
import pytest

EPSILON = 1e-6



def test_space_creation():
    x = Space(5.0)
    assert x.data() == 5.0
    assert x.grad() == 0.0


def test_addition():
    x = simplegrad.Space(2.0)
    y = simplegrad.Space(3.0)
    z = x + y
    assert z.data == 5.0
    z.backward()
    assert x.grad == 1.0
    assert y.grad == 1.0


def test_multiplication():
    x = simplegrad.Space(3.0)
    y = simplegrad.Space(4.0)
    z = x * y
    assert z.data == 12.0
    z.backward()
    assert x.grad == 4.0
    assert y.grad == 3.0


def test_power():
    x = simplegrad.Space(2.0)
    z = x.pow(3.0)
    assert z.data == 8.0
    z.backward()
    assert x.grad == 12.0  # derivative of x^3 is 3x^2


def test_negation():
    x = simplegrad.Space(2.0)
    z = -x
    assert z.data == -2.0
    z.backward()
    assert x.grad == -1.0


def test_chained_operations():
    x = simplegrad.Space(2.0)
    y = simplegrad.Space(3.0)
    z = x * y + x
    assert z.data == 8.0
    z.backward()
    assert x.grad == 4.0  # derivative with respect to x
    assert y.grad == 2.0  # derivative with respect to y


def test_scalar_operations():
    x = simplegrad.Space(2.0)
    z = x + 3.0
    assert z.data == 5.0
    z.backward()
    assert x.grad == 1.0


def test_more_complex_expression():
    x = simplegrad.Space(2.0)
    y = simplegrad.Space(3.0)
    z = (x * y).pow(2.0) + y
    assert z.data == 39.0  # (2*3)^2 + 3 = 36 + 3 = 39
    z.backward()
    assert abs(x.grad - 24.0) < 1e-6  # partial derivative with respect to x
    assert abs(y.grad - 13.0) < 1e-6  # partial derivative with respect to y


def test_multiple_uses():
    x = simplegrad.Space(2.0)
    y = x * x
    z = y + x
    assert z.data == 6.0  # 2^2 + 2 = 6
    z.backward()
    assert x.grad == 5.0  # derivative = 2x + 1 = 5


def test_zero_gradient():
    x = simplegrad.Space(2.0)
    y = simplegrad.Space(3.0)
    z = x + y
    assert x.grad == 0.0
    assert y.grad == 0.0
