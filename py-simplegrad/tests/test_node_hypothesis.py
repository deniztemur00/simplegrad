from hypothesis import given, strategies as st
from simplegrad import Node
import pytest
EPSILON = 1e-5


floats = st.floats(
    min_value=-255, max_value=255, allow_infinity=False, allow_nan=False, width=16
)

integers = st.integers(
    min_value=-10000, max_value=10000
)


@given(x=[floats], y=floats)
def test_addition_random(x, y):
    a = Node(x)
    b = Node(y)
    c = a + b
    assert abs(c.data() - (x + y)) < EPSILON
    c.backward()
    assert abs(a.grad() - 1.0) < EPSILON
    assert abs(b.grad() - 1.0) < EPSILON


@given(x=floats, y=floats)
def test_multiplication_random(x, y):
    a = Node(x)
    b = Node(y)
    c = a * b
    assert abs(c.data() - (x * y)) < EPSILON
    c.backward()
    assert abs(a.grad() - y) < EPSILON
    assert abs(b.grad() - x) < EPSILON

@given(x=integers, y=integers)
def test_addition_random(x, y):
    a = Node(x)
    b = Node(y)
    c = a + b
    assert abs(c.data() - (x + y)) < EPSILON
    c.backward()
    assert abs(a.grad() - 1.0) < EPSILON
    assert abs(b.grad() - 1.0) < EPSILON


@given(x=integers, y=integers)
@pytest.mark.xfail(reason="Scientific notation exceeds tolerance value")
def test_multiplication_random(x, y):
    a = Node(x)
    b = Node(y)
    c = a * b
    assert abs(c.data() - (x * y)) < EPSILON
    c.backward()
    assert abs(a.grad() - y) < EPSILON
    assert abs(b.grad() - x) < EPSILON
