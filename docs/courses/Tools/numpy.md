# Tools - NumPy

!!! Info ""
    NumPy is the fundamental library for scientific computing with Python. NumPy is centered around a powerful N-dimensional array object, and it also contains useful linear algebra, Fourier transform, and random number functions.

## **Creating Arrays**

!!! Info ""
    Now let's import `numpy`. Most people import it as `np`:

```py
import numpy as np
```

### **np.zeros**

!!! Info ""
    The `zeros` function creates an array containing any number of zeros:

```py
np.zeros(5)
```

??? Output "Output"
    array([0., 0., 0., 0., 0.])

!!! Info ""
    It's just as easy to create a 2D array (ie. a matrix) by providing a tuple with the desired number of rows and columns. For example, here's a 3x4 matrix:

```py
np.zeros((3,4))
```
??? Output "Output"
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])

### **Some vocabulary**

!!! Info ""

    - In NumPy, each dimension is called an ***axis***.

    - The number of axes is called the ***rank***.

        - For example, the above 3x4 matrix is an array of rank 2 (it is 2-dimensional).

        - The first axis has length 3, the second has length 4.

    - An array's list of axis lengths is called the ***shape*** of the array.

        - For example, the above matrix's shape is `(3, 4)`.

        - The rank is equal to the shape's length.

    - The ***size*** of an array is the total number of elements, which is the product of all axis lengths (eg. 3*4=12)

```py
a = np.zeros((3,4))
a
```
??? Output "Output"

    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])

```py
a.shape
```
??? Output "Output"
    (3, 4)

```py
a.ndim  # equal to len(a.shape)
```
??? Output "Output"
    2

```py
a.size
```
??? Output "Output"
    12

### **N-dimensional arrays**

!!! Info ""
    You can also create an N-dimensional array of arbitrary rank. For example, here's a 3D array (rank=3), with shape `(2,3,4)`:

```py
np.zeros((2,3,4))
```

??? Output "Output"
    array([[[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]],

           [[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]]])

### **Array type**

!!! Info ""
    NumPy arrays have the type `ndarrays`:

```py
type(np.zeros((3,4)))
```

??? Output "Output"
    numpy.ndarray

### **np.ones**

!!! Info ""
    Many other NumPy functions create ndarrays.

    Here's a 3x4 matrix full of ones:

```py
np.ones((3,4))
```

??? Output "Output"
    array([[1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.]])
        
### **np.full**

!!! Info ""
    Creates an array of the given shape initialized with the given value. Here's a 3x4 matrix full of `π`.

```py
np.full((3,4), np.pi)
```

??? Output "Output"
    array([[3.14159265, 3.14159265, 3.14159265, 3.14159265],
           [3.14159265, 3.14159265, 3.14159265, 3.14159265],
           [3.14159265, 3.14159265, 3.14159265, 3.14159265]])

### **np.empty**

!!! Info ""
    An uninitialized 2x3 array (its content is not predictable, as it is whatever is in memory at that point):

```py
np.empty((2,3))
```

??? Output "Output"
    array([[0., 0., 0.],
           [0., 0., 0.]])

### **np.array**

!!! Info ""
    Of course you can initialize an `ndarray` using a regular python array. Just call the `array` function:

```py
np.array([[1,2,3,4], [10, 20, 30, 40]])
```

??? Output "Output"
    array([[ 1,  2,  3,  4],
           [10, 20, 30, 40]])

### **np.arange**

!!! Info ""
    You can create an `ndarray` using NumPy's `arange` function, which is similar to python's built-in `range` function:

```py
np.arange(1, 5)
```

??? Output "Output"
    array([1, 2, 3, 4])

!!! Info ""
    It also works with floats:

```py
np.arange(1.0, 5.0)
```

??? Output "Output"
    array([1., 2., 3., 4.])

!!! Info ""
    Of course you can provide a step parameter:

```py
np.arange(1, 5, 0.5)
```

??? Output "Output"
    array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5])

!!! Info ""
    However, when dealing with floats, the exact number of elements in the array is not always predictible. For example, consider this:

```py
print(np.arange(0, 5/3, 1/3)) # depending on floating point errors, the max value is 4/3 or 5/3.
print(np.arange(0, 5/3, 0.333333333))
print(np.arange(0, 5/3, 0.333333334))
```

??? Output "Output"
    [0.         0.33333333 0.66666667 1.         1.33333333 1.66666667]
    [0.         0.33333333 0.66666667 1.         1.33333333 1.66666667]
    [0.         0.33333333 0.66666667 1.         1.33333334]

### **np.linspace**

!!! Info ""
    For this reason, it is generally preferable to use the `linspace` function instead of `arange` when working with floats. The `linspace` function returns an array containing a specific number of points evenly distributed between two values (note that the maximum value is included, contrary to `arange`):


```py
print(np.linspace(0, 5/3, 6))
```
??? Output "Output"
    [0.         0.33333333 0.66666667 1.         1.33333333 1.66666667]

### **np.rand and np.randn**

!!! Info ""
    A number of functions are available in NumPy's `random` module to create `ndarray`s initialized with random values. For example, here is a 3x4 matrix initialized with random floats between 0 and 1 (uniform distribution):

```py
np.random.rand(3,4)
```

??? Output "Output"
    array([[0.07951522, 0.82516403, 0.54524215, 0.46662691],
           [0.12016334, 0.74912183, 0.183234  , 0.105027  ],
           [0.22051959, 0.26931151, 0.02739192, 0.4721405 ]])

!!! Info ""
    Here's a 3x4 matrix containing random floats sampled from a univariate [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) (Gaussian distribution) of mean 0 and variance 1:


```py
np.random.randn(3,4)
```

??? Output "Output"
    array([[ 0.09545957,  0.14828368, -0.91504156, -0.36224068],
           [ 0.55434999,  0.41143633,  0.84385243, -0.3652369 ],
           [ 1.48071803, -1.45297797,  1.24551713,  0.4508626 ]])

!!! Info ""  
    To give you a feel of what these distributions look like, let's use matplotlib (see the [matplotlib tutorial](https://colab.research.google.com/drive/tools_matplotlib.ipynb) for more details):

```py
%matplotlib inline
import matplotlib.pyplot as plt
```

```py
plt.hist(np.random.rand(100000), density=True, bins=100, histtype="step", color="blue", label="rand")
plt.hist(np.random.randn(100000), density=True, bins=100, histtype="step", color="red", label="randn")
plt.axis([-2.5, 2.5, 0, 1.1])
plt.legend(loc = "upper left")
plt.title("Random distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()
```

??? Output "Output"
    ![Image](../../assets/images/numpy1.png)


### **np.fromfunction**

!!! Info "" 
    You can also initialize an `ndarray` using a function:

```py
def my_function(z, y, x):
    return x + 10 * y + 100 * z

np.fromfunction(my_function, (3, 2, 10))
```

??? Output "Output"
    array([[[  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.],
            [ 10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.]],

           [[100., 101., 102., 103., 104., 105., 106., 107., 108., 109.],
            [110., 111., 112., 113., 114., 115., 116., 117., 118., 119.]],

           [[200., 201., 202., 203., 204., 205., 206., 207., 208., 209.],
            [210., 211., 212., 213., 214., 215., 216., 217., 218., 219.]]])

!!! Info "" 
    NumPy first creates three `ndarrays` (one per dimension), each of shape `(3, 2, 10)`. Each array has values equal to the coordinate along a specific axis. For example, all elements in the `z` array are equal to their z-coordinate:

```md
[[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
  [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]

 [[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]

 [[ 2.  2.  2.  2.  2.  2.  2.  2.  2.  2.]
  [ 2.  2.  2.  2.  2.  2.  2.  2.  2.  2.]]]
```

!!! Info "" 
    So the terms `x`, `y` and `z` in the expression `x + 10 * y + 100 * z` above are in fact `ndarrays` (we will discuss arithmetic operations on arrays below). The point is that the function `my_function` is only called once, instead of once per element. This makes initialization very efficient.

## **Array data**

### **dtype**

!!! Info "" 
    NumPy's `ndarrays` are also efficient in part because all their elements must have the same type (usually numbers). You can check what the data type is by looking at the `dtype` attribute:

```py
c = np.arange(1, 5)
print(c.dtype, c)
```

??? Output "Output"
    int64 [1 2 3 4]

```py
c = np.arange(1.0, 5.0)
print(c.dtype, c)
```
??? Output "Output"
    float64 [ 1.  2.  3.  4.]

!!! Info "" 
    Instead of letting NumPy guess what data type to use, you can set it explicitly when creating an array by setting the `dtype` parameter:

??? Output "Output"
    complex64 [ 1.+0.j  2.+0.j  3.+0.j  4.+0.j]

!!! Info "" 
    Available data types include `int8`, `int16`, `int32`, `int64`, `uint8`|`16`|`32`|`64`, `float16`|`32`|`64` and `complex64`|`128`. Check out [the documentation](http://docs.scipy.org/doc/numpy-1.10.1/user/basics.types.html) for the full list.

### **itemsize**

!!! Info ""
    The `itemsize` attribute returns the size (in bytes) of each item:

```py
e = np.arange(1, 5, dtype=np.complex64)
e.itemsize
```

??? Output "Output"
    8

### **data buffer**

!!! Info ""
    An array's data is actually stored in memory as a flat (one dimensional) byte buffer. It is available via the `data` attribute (you will rarely need it, though).

```py
f = np.array([[1,2],[1000, 2000]], dtype=np.int32)
f.data
```

??? Output "Output"
    <read-write buffer for 0x10f8a18a0, size 16, offset 0 at 0x10f9dbbb0>

!!! Info ""
    In python 2, `f.data` is a buffer. In python 3, it is a memoryview.

```py
if (hasattr(f.data, "tobytes")):
    data_bytes = f.data.tobytes() # python 3
else:
    data_bytes = memoryview(f.data).tobytes() # python 2

data_bytes
```

??? Output "Output"
    '\x01\x00\x00\x00\x02\x00\x00\x00\xe8\x03\x00\x00\xd0\x07\x00\x00'

!!! Info ""
    Several `ndarrays` can share the same data buffer, meaning that modifying one will also modify the others. We will see an example in a minute.



## **Reshaping an array**

### **In place**

!!! Info ""
    Changing the shape of an `ndarray` is as simple as setting its `shape` attribute. However, the array's size must remain the same.

```py
g = np.arange(24)
print(g)
print("Rank:", g.ndim)
```

??? Output "Output"
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
    Rank: 1

```py
g.shape = (6, 4)
print(g)
print("Rank:", g.ndim)
```

??? Output "Output"
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]
     [16 17 18 19]
     [20 21 22 23]]
    Rank: 2

```py
g.shape = (2, 3, 4)
print(g)
print("Rank:", g.ndim)
```

??? Output "Output"
    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]

     [[12 13 14 15]
      [16 17 18 19]
      [20 21 22 23]]]
    Rank: 3

### **reshape**

!!! Info ""
    The `reshape` function returns a new `ndarray` object pointing at the same data. This means that modifying one array will also modify the other.

```py
g2 = g.reshape(4,6)
print(g2)
print("Rank:", g2.ndim)
```

??? Output "Output"
    [[ 0  1  2  3  4  5]
     [ 6  7  8  9 10 11]
     [12 13 14 15 16 17]
     [18 19 20 21 22 23]]
    Rank: 2

!!! Info ""
    Set item at row 1, col 2 to 999 (more about indexing below).

```py
g2[1, 2] = 999
g2
```

??? Output "Output"
    array([[  0,   1,   2,   3,   4,   5],
           [  6,   7, 999,   9,  10,  11],
           [ 12,  13,  14,  15,  16,  17],
           [ 18,  19,  20,  21,  22,  23]])

!!! Info ""
    The corresponding element in `g` has been modified.

??? Output "Output"
    array([[[  0,   1,   2,   3],
            [  4,   5,   6,   7],
            [999,   9,  10,  11]],

           [[ 12,  13,  14,  15],
            [ 16,  17,  18,  19],
            [ 20,  21,  22,  23]]])
        
### **ravel**

!!! Info ""
    Finally, the `ravel` function returns a new one-dimensional `ndarray` that also points to the same data:

```py
g.ravel()
```

??? Output "Output"
    array([  0,   1,   2,   3,   4,   5,   6,   7, 999,   9,  10,  11,  12,
            13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23])


## **Arithmetic operations**

!!! Info ""
    All the usual arithmetic operators (`+`, `-`, `*`, `/`, `//`, `**`, etc.) can be used with `ndarray`s. They apply elementwise:

```py
a = np.array([14, 23, 32, 41])
b = np.array([5,  4,  3,  2])
print("a + b  =", a + b)
print("a - b  =", a - b)
print("a * b  =", a * b)
print("a / b  =", a / b)
print("a // b  =", a // b)
print("a % b  =", a % b)
print("a ** b =", a ** b)
```

??? Output "Output"
    a + b  = [19 27 35 43]
    a - b  = [ 9 19 29 39]
    a * b  = [70 92 96 82]
    a / b  = [  2.8          5.75        10.66666667  20.5       ]
    a // b  = [ 2  5 10 20]
    a % b  = [4 3 2 1]
    a ** b = [537824 279841  32768   1681]

!!! Info ""
    Note that the multiplication is not a matrix multiplication. We will discuss matrix operations below.

    The arrays must have the same shape. If they do not, NumPy will apply the broadcasting rules.


## **Broadcasting**

!!! Info ""
    In general, when NumPy expects arrays of the same shape but finds that this is not the case, it applies the so-called broadcasting rules:

### **First rule**

!!! Info ""
    If the arrays do not have the same rank, then a 1 will be prepended to the smaller ranking arrays until their ranks match.

```py
h = np.arange(5).reshape(1, 1, 5)
h
```

??? Output "Output"
    array([[[0, 1, 2, 3, 4]]])

!!! Info ""
    Now let's try to add a 1D array of shape `(5,)` to this 3D array of shape `(1,1,5)`. Applying the first rule of broadcasting!

```py
h + [10, 20, 30, 40, 50]  # same as: h + [[[10, 20, 30, 40, 50]]]
```

??? Output "Output"
    array([[[10, 21, 32, 43, 54]]])

### **Second rule**

!!! Info ""
    Arrays with a 1 along a particular dimension act as if they had the size of the array with the largest shape along that dimension. The value of the array element is repeated along that dimension.

```py
k = np.arange(6).reshape(2, 3)
k
```

??? Output "Output"
    array([[0, 1, 2],
           [3, 4, 5]])

!!! Info ""
    Let's try to add a 2D array of shape `(2,1)` to this 2D `ndarray` of shape `(2, 3)`. NumPy will apply the second rule of broadcasting:

```py
k + [[100], [200]]  # same as: k + [[100, 100, 100], [200, 200, 200]]
```

??? Output "Output"
    array([[100, 101, 102],
           [203, 204, 205]])

!!! Info ""
    Combining rules 1 & 2, we can do this:

```py
k + [100, 200, 300]  # after rule 1: [[100, 200, 300]], and after rule 2: [[100, 200, 300], [100, 200, 300]]
```

??? Output "Output"
    array([[100, 201, 302],
           [103, 204, 305]])

!!! Info ""
    And also, very simply:

```py
k + 1000  # same as: k + [[1000, 1000, 1000], [1000, 1000, 1000]]
```

??? Output "Output"
    array([[1000, 1001, 1002],
           [1003, 1004, 1005]])
    
### **Third rule**

!!! Info ""
    After rules 1 & 2, the sizes of all arrays must match.

```py
try:
    k + [33, 44]
except ValueError as e:
    print(e)
```

??? Output "Output"
    operands could not be broadcast together with shapes (2,3) (2,) 

!!! Info ""
    Broadcasting rules are used in many NumPy operations, not just arithmetic operations, as we will see below. For more details about broadcasting, check out [the documentation](https://docs.scipy.org/doc/numpy-dev/user/basics.broadcasting.html).

### **Upcasting**

!!! Info ""
    When trying to combine arrays with different `dtype`s, NumPy will upcast to a type capable of handling all possible values (regardless of what the actual values are).

```py
k1 = np.arange(0, 5, dtype=np.uint8)
print(k1.dtype, k1)
```
??? Output "Output"
    uint8 [0 1 2 3 4]

```py
k2 = k1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)
print(k2.dtype, k2)
```

??? Output "Output"
    int16 [ 5  7  9 11 13]

!!! Info ""
    Note that `int16` is required to represent all possible `int8` and `uint8` values (from -128 to 255), even though in this case a uint8 would have sufficed.

```py
k3 = k1 + 1.5
print(k3.dtype, k3)
```

??? Output "Output"
    float64 [ 1.5  2.5  3.5  4.5  5.5]


## **Conditional operators**

!!! Info ""
    The conditional operators also apply elementwise:

```py
m = np.array([20, -5, 30, 40])
m < [15, 16, 35, 36]
```

??? Output "Output"
    array([False,  True,  True, False], dtype=bool)

!!! Info ""
    And using broadcasting:

```py
m < 25  # equivalent to m < [25, 25, 25, 25]
```

??? Output "Output"
    array([ True,  True, False, False], dtype=bool)

!!! Info ""
    This is most useful in conjunction with boolean indexing (discussed below).

```py
m[m < 25]
```

??? Output "Output"
    array([20, -5])


## **Mathematical and statistical functions**

!!! Info ""
    Many mathematical and statistical functions are available for `ndarray`s.

### **ndarray methods**

!!! Info ""
    Some functions are simply `ndarray` methods, for example:

```py
a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
print(a)
print("mean =", a.mean())
```

??? Output "Output"
    [[ -2.5   3.1   7. ]
     [ 10.   11.   12. ]]
    mean = 6.76666666667

!!! Info ""
    Note that this computes the mean of all elements in the `ndarray`, regardless of its shape.

    Here are a few more useful `ndarray` methods:

```py
for func in (a.min, a.max, a.sum, a.prod, a.std, a.var):
    print(func.__name__, "=", func())
```

??? Output "Output"
    min = -2.5
    max = 12.0
    sum = 40.6
    prod = -71610.0
    std = 5.08483584352
    var = 25.8555555556

!!! Info ""
    These functions accept an optional argument `axis` which lets you ask for the operation to be performed on elements along the given axis. For example:

```py
c=np.arange(24).reshape(2,3,4)
c
```

??? Output "Output"
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],

           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
        
```py
c.sum(axis=0)  # sum across matrices
```

??? Output "Output"
    array([[12, 14, 16, 18],
           [20, 22, 24, 26],
           [28, 30, 32, 34]])

```py
c.sum(axis=1)  # sum across rows
```

??? Output "Output"
    array([[12, 15, 18, 21],
           [48, 51, 54, 57]])

!!! Info ""
    You can also sum over multiple axes:

```py
c.sum(axis=(0,2))  # sum across matrices and columns
```

??? Output "Output"
    array([ 60,  92, 124])

```py
0+1+2+3 + 12+13+14+15, 4+5+6+7 + 16+17+18+19, 8+9+10+11 + 20+21+22+23
```

??? Output "Output"
    (60, 92, 124)


### **Universal functions**

!!! Info ""
    NumPy also provides fast elementwise functions called universal functions, or **ufunc**. They are vectorized wrappers of simple functions. For example `square` returns a new `ndarray` which is a copy of the original `ndarray` except that each element is squared:

```py
a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
np.square(a)
```

??? Output "Output"
    array([[   6.25,    9.61,   49.  ],
           [ 100.  ,  121.  ,  144.  ]])
        
!!! Info ""
    Here are a few more useful unary ufuncs:

```py
print("Original ndarray")
print(a)
for func in (np.abs, np.sqrt, np.exp, np.log, np.sign, np.ceil, np.modf, np.isnan, np.cos):
    print("\n", func.__name__)
    print(func(a))
```

??? Output "Output"
    Original ndarray
    [[ -2.5   3.1   7. ]
     [ 10.   11.   12. ]]

    absolute
    [[  2.5   3.1   7. ]
     [ 10.   11.   12. ]]

    sqrt
    [[        nan  1.76068169  2.64575131]
     [ 3.16227766  3.31662479  3.46410162]]

    exp
    [[  8.20849986e-02   2.21979513e+01   1.09663316e+03]
     [  2.20264658e+04   5.98741417e+04   1.62754791e+05]]

    log
    [[        nan  1.13140211  1.94591015]
     [ 2.30258509  2.39789527  2.48490665]]

    sign
    [[-1.  1.  1.]
     [ 1.  1.  1.]]

    ceil
    [[ -2.   4.   7.]
     [ 10.  11.  12.]]

    modf
    (array([[-0.5,  0.1,  0. ],
           [ 0. ,  0. ,  0. ]]), array([[ -2.,   3.,   7.],
           [ 10.,  11.,  12.]]))

    isnan
    [[False False False]
     [False False False]]

    cos
    [[-0.80114362 -0.99913515  0.75390225]
     [-0.83907153  0.0044257   0.84385396]]
    -c:5: RuntimeWarning: invalid value encountered in sqrt
    -c:5: RuntimeWarning: invalid value encountered in log

### **Binary ufuncs**

!!! Info ""
    There are also many binary ufuncs, that apply elementwise on two `ndarray`s. Broadcasting rules are applied if the arrays do not have the same shape:

```py
a = np.array([1, -2, 3, 4])
b = np.array([2, 8, -1, 7])
np.add(a, b)  # equivalent to a + b
```

??? Output "Output"
    array([ 3,  6,  2, 11])

```py
np.greater(a, b)  # equivalent to a > b
```

??? Output "Output"
    array([False, False,  True, False], dtype=bool)

```py
np.maximum(a, b)
```

??? Output "Output"
    array([2, 8, 3, 7])

```py
np.copysign(a, b)
```

??? Output "Output"
    array([ 1.,  2., -3.,  4.])


## **Array indexing**

## **Iterating**

## **Stacking arrays**

## **Splitting arrays**

## **Transposing arrays**

## **Linear algebra**

## **Vectorization**

## **Saving and loading**

## **What next?**