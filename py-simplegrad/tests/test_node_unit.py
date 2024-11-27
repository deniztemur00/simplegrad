from simplegrad import Node
import math
import pytest

EPSILON = 1e-6


## testleri sil teker teker yaz make test
def test_space_creation():
    x = Node(5.0)
    assert abs(x.data() - 5.0) < EPSILON
    assert abs(x.grad() - 0.0) < EPSILON


def test_space_addition():
    x = Node(5.0)
    y = Node(3.0)
    z = x + y
    assert abs(z.data() - 8.0) < EPSILON
    assert abs(z.grad() - 0.0) < EPSILON


def test_space_subtraction():
    x = Node(5.0)
    y = Node(3.0)
    z = x - y
    assert abs(z.data() - 2.0) < EPSILON
    assert abs(z.grad() - 0.0) < EPSILON


def test_space_multiplication():
    x = Node(5.1156)
    y = Node(3.5225)
    z = x * y
    assert abs(z.data() - 5.1156 * 3.5225) < EPSILON
    assert abs(z.grad() - 0.0) < EPSILON


def test_space_division():
    x = Node(5.0)
    y = Node(0.0)
    z = x / y
    assert abs(z.data() - 5.0 / 1e-9) < EPSILON
    assert abs(z.grad() - 0.0) < EPSILON


def test_compound_ops():
    a = Node(5.0)
    b = Node(3.0)

    d = a * b
    d = d + 1.0
    assert abs(d.data() - 16.0) < EPSILON


def test_compound_mul():
    a = Node(5.0)
    b = Node(3.0)

    d = a * b
    d = d * 3.0
    assert abs(d.data() - 45.0) < EPSILON


@pytest.mark.xfail(reason="To be implemented")
def test_compound_div():
    a = Node(5.0)
    b = Node(3.0)

    d = a / b
    d = d / 3.0
    assert abs(d.data() - 5.0 / 3.0 / 3.0) < EPSILON


def test_relu_ops():
    a = Node(-5.0)
    b = Node(3.0)
    sum_ab = b + a
    relu_result = sum_ab.relu()
    assert abs(relu_result.data() - 0.0) < EPSILON


def test_backward():
    a = Node(5.0)
    b = Node(3.0)
    c = a * b
    c.backward()
    assert abs(a.grad() - 3.0) < EPSILON
    assert abs(b.grad() - 5.0) < EPSILON


def test_addition_backward():
    a = Node(5.0)
    b = Node(3.0)
    c = a + b
    c.backward()
    assert abs(a.grad() - 1.0) < EPSILON
    assert abs(b.grad() - 1.0) < EPSILON

    # Chained addition


def test_chained_add():
    a = Node(2.0)
    b = Node(3.0)
    c = Node(4.0)
    d = a + b + c
    assert abs(d.data() - 9.0) < EPSILON
    d.backward()
    assert abs(a.grad() - 1.0) < EPSILON
    assert abs(b.grad() - 1.0) < EPSILON
    assert abs(c.grad() - 1.0) < EPSILON


# Multiple use
def test_multiple_use():
    a = Node(2.0)
    b = Node(3.0)
    c = a + b
    d = c + a
    assert abs(d.data() - 7.0) < EPSILON
    d.backward()
    assert abs(a.grad() - 2.0) < EPSILON
    assert abs(b.grad() - 1.0) < EPSILON


# Zero addition
def test_zero_add():
    a = Node(2.0)
    b = Node(0.0)
    c = a + b
    assert abs(c.data() - 2.0) < EPSILON
    c.backward()
    assert abs(a.grad() - 1.0) < EPSILON
    assert abs(b.grad() - 1.0) < EPSILON


def test_multiplication_backward():
    a = Node(5.0)
    b = Node(3.0)
    c = a * b
    c.backward()
    print(a.grad())
    print(b.grad())
    assert abs(a.grad() - 3.0) < 1e-6
    assert abs(b.grad() - 5.0) < 1e-6


def test_addition_backward():
    a = Node(5.0)
    b = Node(3.0)
    c = a + b
    c.backward()
    print(a, b, c)
    print(a.op(), b.op(), c.op())
    print(c.prev())
    assert abs(a.grad() - 1.0) < 1e-6
    assert abs(b.grad() - 1.0) < 1e-6


def test_chained_add():
    a = Node(2.0)
    b = Node(3.0)
    c = Node(4.0)
    d = a + b + c


def test_subtraction_backward():
    a = Node(5.24)
    b = -Node(3.14)
    c = a - b
    c.backward()
    print(a, b, c)

    assert abs(a.grad() - 1.0) < 1e-6
    assert abs(b.grad() + 1.0) < 1e-6
    assert abs(c.grad() - 1.0) < 1e-6


@pytest.mark.xfail(reason="Low precision of exponentiation")
def test_power_backward():
    a = Node(5.0)
    b = Node(3.0)
    c = a**b
    c.backward()
    print(a.grad())
    print(b.grad())
    assert abs(a.grad() - (3.0 * 5.0**2)) < 1e-6
    res = (5.0**3) * math.log(5.0)
    assert (
        abs(b.grad() - ((5.0**3) * math.log(5.0))) < 1e-6
    )  # Difference between b grad vs actual result:  9.480893709223892e-06


def test_division_backward():
    a = Node(5.0)
    b = Node(3.0)
    c = a / b
    c.backward()
    print(a.grad())
    print(b.grad())
    assert abs(a.grad() - 1.0 / 3.0) < 1e-6
    assert abs(b.grad() + 5.0 / 9.0) < 1e-6


def test_power_with_float_exponent():
    a = Node(2.0)
    c = a**3.0
    c.backward()
    assert abs(a.grad() - (3.0 * 2.0**2)) < EPSILON

@pytest.mark.xfail(reason="Gradients exceed the epsilon at infinity")
def test_division_by_zero():
    a = Node(5.0)
    b = Node(0.0)
    c = a / b
    c.backward()
    assert abs(c.data() - (5.0 / 1e-9)) < EPSILON
    assert abs(a.grad() - (1.0 / 1e-9)) < EPSILON
    assert abs(b.grad() - (-5.0 / (1e-9) ** 2)) < EPSILON


def test_backward_complex_expression():
    a = Node(2.0)
    b = Node(3.0)
    c = Node(4.0)
    d = a * b + c
    d.backward()
    assert abs(a.grad() - 3.0) < EPSILON
    assert abs(b.grad() - 2.0) < EPSILON
    assert abs(c.grad() - 1.0) < EPSILON


def test_negation():
    a = Node(5.0)
    b = -a
    assert abs(b.data() + 5.0) < EPSILON
    b.backward()
    assert abs(a.grad() + 1.0) < EPSILON


def test_relu_negative_input():
    a = Node(-5.0)
    b = a.relu()
    assert abs(b.data() - 0.0) < EPSILON
    b.backward()
    assert abs(a.grad() - 0.0) < EPSILON


def test_relu_positive_input():
    a = Node(5.0)
    b = a.relu()
    assert abs(b.data() - 5.0) < EPSILON
    b.backward()
    assert abs(a.grad() - 1.0) < EPSILON


def test_gradient_accumulation():
    a = Node(1.0)
    b = a + a
    b.backward()
    assert abs(a.grad() - 2.0) < EPSILON
