---
comments: true
---

# NumPy


NumPy is the fundamental library for scientific computing with Python. NumPy is centered around a powerful N-dimensional array object, and it also contains useful linear algebra, Fourier transform, and random number functions.

## **Creating Arrays**

Now let's import `numpy`. Most people import it as `np`:

```py
import numpy as np
```

### **np.zeros**

The `zeros` function creates an array containing any number of zeros:

```py
np.zeros(5)
```

??? Output "Output"
    array([0., 0., 0., 0., 0.])

It's just as easy to create a 2D array (ie. a matrix) by providing a tuple with the desired number of rows and columns. For example, here's a 3x4 matrix:

```py
np.zeros((3,4))
```
??? Output "Output"
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])

### **Some vocabulary**

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

NumPy arrays have the type `ndarrays`:

```py
type(np.zeros((3,4)))
```

??? Output "Output"
    numpy.ndarray

### **np.ones**

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

Creates an array of the given shape initialized with the given value. Here's a 3x4 matrix full of `π`.

```py
np.full((3,4), np.pi)
```

??? Output "Output"
    array([[3.14159265, 3.14159265, 3.14159265, 3.14159265],
           [3.14159265, 3.14159265, 3.14159265, 3.14159265],
           [3.14159265, 3.14159265, 3.14159265, 3.14159265]])

### **np.empty**

An uninitialized 2x3 array (its content is not predictable, as it is whatever is in memory at that point):

```py
np.empty((2,3))
```

??? Output "Output"
    array([[0., 0., 0.],
           [0., 0., 0.]])

### **np.array**

Of course you can initialize an `ndarray` using a regular python array. Just call the `array` function:

```py
np.array([[1,2,3,4], [10, 20, 30, 40]])
```

??? Output "Output"
    array([[ 1,  2,  3,  4],
           [10, 20, 30, 40]])

### **np.arange**

You can create an `ndarray` using NumPy's `arange` function, which is similar to python's built-in `range` function:

```py
np.arange(1, 5)
```

??? Output "Output"
    array([1, 2, 3, 4])

It also works with floats:

```py
np.arange(1.0, 5.0)
```

??? Output "Output"
    array([1., 2., 3., 4.])

Of course you can provide a step parameter:

```py
np.arange(1, 5, 0.5)
```

??? Output "Output"
    array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5])

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

For this reason, it is generally preferable to use the `linspace` function instead of `arange` when working with floats. The `linspace` function returns an array containing a specific number of points evenly distributed between two values (note that the maximum value is included, contrary to `arange`):


```py
print(np.linspace(0, 5/3, 6))
```
??? Output "Output"
    [0.         0.33333333 0.66666667 1.         1.33333333 1.66666667]

### **np.rand and np.randn**

A number of functions are available in NumPy's `random` module to create `ndarray`s initialized with random values. For example, here is a 3x4 matrix initialized with random floats between 0 and 1 (uniform distribution):

```py
np.random.rand(3,4)
```

??? Output "Output"
    array([[0.07951522, 0.82516403, 0.54524215, 0.46662691],
           [0.12016334, 0.74912183, 0.183234  , 0.105027  ],
           [0.22051959, 0.26931151, 0.02739192, 0.4721405 ]])

Here's a 3x4 matrix containing random floats sampled from a univariate [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) (Gaussian distribution) of mean 0 and variance 1:


```py
np.random.randn(3,4)
```

??? Output "Output"
    array([[ 0.09545957,  0.14828368, -0.91504156, -0.36224068],
           [ 0.55434999,  0.41143633,  0.84385243, -0.3652369 ],
           [ 1.48071803, -1.45297797,  1.24551713,  0.4508626 ]])
 
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

NumPy first creates three `ndarrays` (one per dimension), each of shape `(3, 2, 10)`. Each array has values equal to the coordinate along a specific axis. For example, all elements in the `z` array are equal to their z-coordinate:

```md
[[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
  [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]

 [[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]

 [[ 2.  2.  2.  2.  2.  2.  2.  2.  2.  2.]
  [ 2.  2.  2.  2.  2.  2.  2.  2.  2.  2.]]]
```

So the terms `x`, `y` and `z` in the expression `x + 10 * y + 100 * z` above are in fact `ndarrays` (we will discuss arithmetic operations on arrays below). The point is that the function `my_function` is only called once, instead of once per element. This makes initialization very efficient.

## **Array data**

### **dtype**

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

Instead of letting NumPy guess what data type to use, you can set it explicitly when creating an array by setting the `dtype` parameter:

??? Output "Output"
    complex64 [ 1.+0.j  2.+0.j  3.+0.j  4.+0.j]

Available data types include `int8`, `int16`, `int32`, `int64`, `uint8`|`16`|`32`|`64`, `float16`|`32`|`64` and `complex64`|`128`. Check out [the documentation](http://docs.scipy.org/doc/numpy-1.10.1/user/basics.types.html) for the full list.

### **itemsize**

The `itemsize` attribute returns the size (in bytes) of each item:

```py
e = np.arange(1, 5, dtype=np.complex64)
e.itemsize
```

??? Output "Output"
    8

### **data buffer**

An array's data is actually stored in memory as a flat (one dimensional) byte buffer. It is available via the `data` attribute (you will rarely need it, though).

```py
f = np.array([[1,2],[1000, 2000]], dtype=np.int32)
f.data
```

??? Output "Output"
    <read-write buffer for 0x10f8a18a0, size 16, offset 0 at 0x10f9dbbb0>

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

Several `ndarrays` can share the same data buffer, meaning that modifying one will also modify the others. We will see an example in a minute.



## **Reshaping an array**

### **In place**

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

The corresponding element in `g` has been modified.

??? Output "Output"
    array([[[  0,   1,   2,   3],
            [  4,   5,   6,   7],
            [999,   9,  10,  11]],

           [[ 12,  13,  14,  15],
            [ 16,  17,  18,  19],
            [ 20,  21,  22,  23]]])
        
### **ravel**

Finally, the `ravel` function returns a new one-dimensional `ndarray` that also points to the same data:

```py
g.ravel()
```

??? Output "Output"
    array([  0,   1,   2,   3,   4,   5,   6,   7, 999,   9,  10,  11,  12,
            13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23])


## **Arithmetic operations**

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

Note that the multiplication is not a matrix multiplication. We will discuss matrix operations below.

The arrays must have the same shape. If they do not, NumPy will apply the broadcasting rules.


## **Broadcasting**

In general, when NumPy expects arrays of the same shape but finds that this is not the case, it applies the so-called broadcasting rules:

### **First rule**

If the arrays do not have the same rank, then a 1 will be prepended to the smaller ranking arrays until their ranks match.

```py
h = np.arange(5).reshape(1, 1, 5)
h
```

??? Output "Output"
    array([[[0, 1, 2, 3, 4]]])

Now let's try to add a 1D array of shape `(5,)` to this 3D array of shape `(1,1,5)`. Applying the first rule of broadcasting!

```py
h + [10, 20, 30, 40, 50]  # same as: h + [[[10, 20, 30, 40, 50]]]
```

??? Output "Output"
    array([[[10, 21, 32, 43, 54]]])

### **Second rule**

Arrays with a 1 along a particular dimension act as if they had the size of the array with the largest shape along that dimension. The value of the array element is repeated along that dimension.

```py
k = np.arange(6).reshape(2, 3)
k
```

??? Output "Output"
    array([[0, 1, 2],
           [3, 4, 5]])

Let's try to add a 2D array of shape `(2,1)` to this 2D `ndarray` of shape `(2, 3)`. NumPy will apply the second rule of broadcasting:

```py
k + [[100], [200]]  # same as: k + [[100, 100, 100], [200, 200, 200]]
```

??? Output "Output"
    array([[100, 101, 102],
           [203, 204, 205]])

Combining rules 1 & 2, we can do this:

```py
k + [100, 200, 300]  # after rule 1: [[100, 200, 300]], and after rule 2: [[100, 200, 300], [100, 200, 300]]
```

??? Output "Output"
    array([[100, 201, 302],
           [103, 204, 305]])

And also, very simply:

```py
k + 1000  # same as: k + [[1000, 1000, 1000], [1000, 1000, 1000]]
```

??? Output "Output"
    array([[1000, 1001, 1002],
           [1003, 1004, 1005]])
    
### **Third rule**

After rules 1 & 2, the sizes of all arrays must match.

```py
try:
    k + [33, 44]
except ValueError as e:
    print(e)
```

??? Output "Output"
    operands could not be broadcast together with shapes (2,3) (2,) 

Broadcasting rules are used in many NumPy operations, not just arithmetic operations, as we will see below. For more details about broadcasting, check out [the documentation](https://docs.scipy.org/doc/numpy-dev/user/basics.broadcasting.html).

### **Upcasting**

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

Note that `int16` is required to represent all possible `int8` and `uint8` values (from -128 to 255), even though in this case a uint8 would have sufficed.

```py
k3 = k1 + 1.5
print(k3.dtype, k3)
```

??? Output "Output"
    float64 [ 1.5  2.5  3.5  4.5  5.5]


## **Conditional operators**

The conditional operators also apply elementwise:

```py
m = np.array([20, -5, 30, 40])
m < [15, 16, 35, 36]
```

??? Output "Output"
    array([False,  True,  True, False], dtype=bool)

And using broadcasting:

```py
m < 25  # equivalent to m < [25, 25, 25, 25]
```

??? Output "Output"
    array([ True,  True, False, False], dtype=bool)

This is most useful in conjunction with boolean indexing (discussed below).

```py
m[m < 25]
```

??? Output "Output"
    array([20, -5])


## **Mathematical and statistical functions**

Many mathematical and statistical functions are available for `ndarray`s.

### **ndarray methods**

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

NumPy also provides fast elementwise functions called universal functions, or **ufunc**. They are vectorized wrappers of simple functions. For example `square` returns a new `ndarray` which is a copy of the original `ndarray` except that each element is squared:

```py
a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
np.square(a)
```

??? Output "Output"
    array([[   6.25,    9.61,   49.  ],
           [ 100.  ,  121.  ,  144.  ]])
        
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

### **One-dimensional arrays**

One-dimensional NumPy arrays can be accessed more or less like regular python arrays:

```py
a = np.array([1, 5, 3, 19, 13, 7, 3])
a[3]
```

??? Output "Output"
    19

```py
a[2:5]
```

??? Output "Output"
    array([ 3, 19, 13])

```py
a[2:-1]
```

??? Output "Output"
    array([ 3, 19, 13,  7])

```py
a[:2]
```

??? Output "Output"
    array([1, 5])

```py
a[2::2]
```

??? Output "Output"
    array([ 3, 13,  3])

```py
a[::-1]
```

??? Output "Output"
    array([ 3,  7, 13, 19,  3,  5,  1])


!!! Info ""
    Of course, you can modify elements:

```py
a[3]=999
a
```

??? Output "Output"
    array([  1,   5,   3, 999,  13,   7,   3])

!!! Info ""
    You can also modify an `ndarray` slice:

```py
a[2:5] = [997, 998, 999]
a
```

??? Output "Output"
    array([  1,   5, 997, 998, 999,   7,   3])

### **Differences with regular python arrays**

Contrary to regular python arrays, if you assign a single value to an `ndarray` slice, it is copied across the whole slice, thanks to broadcasting rules discussed above.

```py
a[2:5] = -1
a
```

??? Output "Output"
    array([ 1,  5, -1, -1, -1,  7,  3])

Also, you cannot grow or shrink `ndarrays` this way:

```py
try:
    a[2:5] = [1,2,3,4,5,6]  # too long
except ValueError as e:
    print(e)
```

??? Output "Output"
    cannot copy sequence with size 6 to array axis with dimension 3

You cannot delete elements either:

```py
try:
    del a[2:5]
except ValueError as e:
    print(e)
```

??? Output "Output"
    cannot delete array elements

Last but not least, `ndarray` **slices are actually views** on the same data buffer. This means that if you create a slice and modify it, you are actually going to modify the original `ndarray` as well!


```py
a_slice = a[2:6]
a_slice[1] = 1000
a  # the original array was modified!
```

??? Output "Output"
    array([   1,    5,   -1, 1000,   -1,    7,    3])

```py
a[3] = 2000
a_slice  # similarly, modifying the original array modifies the slice!
```

??? Output "Output"
    array([  -1, 2000,   -1,    7])

If you want a copy of the data, you need to use the `copy` method:

```py
another_slice = a[2:6].copy()
another_slice[1] = 3000
a  # the original array is untouched
```

??? Output "Output"
    array([   1,    5,   -1, 2000,   -1,    7,    3])

```py
a[3] = 4000
another_slice  # similary, modifying the original array does not affect the slice copy
```

??? Output "Output"
    array([  -1, 3000,   -1,    7])

### **Multi-dimensional arrays**

Multi-dimensional arrays can be accessed in a similar way by providing an index or slice for each axis, separated by commas:

```py
b = np.arange(48).reshape(4, 12)
b
```

??? Output "Output"
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
           [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]])
        
```py
b[1, 2]  # row 1, col 2
```

??? Output "Output"
    14

```py
b[1, :]  # row 1, all columns
```

??? Output "Output"
    array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

```py
b[:, 1]  # all rows, column 1
```

??? Output "Output"
    array([ 1, 13, 25, 37])

!!! Warning "Caution"
    note the subtle difference between these two expressions:

```py
b[1, :]
```

??? Output "Output"
    array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

```py
b[1:2, :]
```

??? Output "Output"
    array([[12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]])

The first expression returns row 1 as a 1D array of shape `(12,)`, while the second returns that same row as a 2D array of shape `(1, 12)`.

### **Fancy indexing**

You may also specify a list of indices that you are interested in. This is referred to as fancy indexing.

```py
b[(0,2), 2:5]  # rows 0 and 2, columns 2 to 4 (5-1)
```

??? Output "Output"
    array([[ 2,  3,  4],
           [26, 27, 28]])

```py
b[:, (-1, 2, -1)]  # all rows, columns -1 (last), 2 and -1 (again, and in this order)
```

??? Output "Output"
    array([[11,  2, 11],
           [23, 14, 23],
           [35, 26, 35],
           [47, 38, 47]])

If you provide multiple index arrays, you get a 1D `ndarray` containing the values of the elements at the specified coordinates.

```py
b[(-1, 2, -1, 2), (5, 9, 1, 9)]  # returns a 1D array with b[-1, 5], b[2, 9], b[-1, 1] and b[2, 9] (again)
```

??? Output "Output"
    array([41, 33, 37, 33])

### **Higher dimensions**

Everything works just as well with higher dimensional arrays, but it's useful to look at a few examples:

```py
c = b.reshape(4,2,6)
c
```

??? Output "Output"
    array([[[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11]],

           [[12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23]],

           [[24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35]],

           [[36, 37, 38, 39, 40, 41],
            [42, 43, 44, 45, 46, 47]]])

```py
c[2, 1, 4]  # matrix 2, row 1, col 4
```

??? Output "Output"
    34

```py
c[2, :, 3]  # matrix 2, all rows, col 3
```

??? Output "Output"
    array([27, 33])

If you omit coordinates for some axes, then all elements in these axes are returned:

```py
c[2, 1]  # Return matrix 2, row 1, all columns.  This is equivalent to c[2, 1, :]
```

??? Output "Output"
    array([30, 31, 32, 33, 34, 35])

### **Ellipsis (...)**

You may also write an ellipsis (`...`) to ask that all non-specified axes be entirely included.

```py
c[2, ...]  #  matrix 2, all rows, all columns.  This is equivalent to c[2, :, :]
```

??? Output "Output"
    array([[24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35]])

```py
c[2, 1, ...]  # matrix 2, row 1, all columns.  This is equivalent to c[2, 1, :]
```

??? Output "Output"
    array([30, 31, 32, 33, 34, 35])

```py
c[2, ..., 3]  # matrix 2, all rows, column 3.  This is equivalent to c[2, :, 3]
```

??? Output "Output"
    array([27, 33])

```py
c[..., 3]  # all matrices, all rows, column 3.  This is equivalent to c[:, :, 3]
```

??? Output "Output"
    array([[ 3,  9],
           [15, 21],
           [27, 33],
           [39, 45]])
        
### **Boolean indexing**

You can also provide an `ndarray` of boolean values on one axis to specify the indices that you want to access.

```py
b = np.arange(48).reshape(4, 12)
b
```

??? Output "Output"
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
           [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]])
        
```py
rows_on = np.array([True, False, True, False])
b[rows_on, :]  # Rows 0 and 2, all columns. Equivalent to b[(0, 2), :]
```

??? Output "Output"
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
           [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]])

```py
cols_on = np.array([False, True, False] * 4)
b[:, cols_on]  # All rows, columns 1, 4, 7 and 10
```

??? Output "Output"
    array([[ 1,  4,  7, 10],
           [13, 16, 19, 22],
           [25, 28, 31, 34],
           [37, 40, 43, 46]])

### **np.ix_**

You cannot use boolean indexing this way on multiple axes, but you can work around this by using the `ix_` function:

```py
b[np.ix_(rows_on, cols_on)]
```

??? Output "Output"
    array([[ 1,  4,  7, 10],
           [25, 28, 31, 34]])

```py
np.ix_(rows_on, cols_on)
```

??? Output "Output"
    (array([[0],
            [2]]), array([[ 1,  4,  7, 10]]))

If you use a boolean array that has the same shape as the `ndarray`, then you get in return a 1D array containing all the values that have `True` at their coordinate. This is generally used along with conditional operators:

```py
b[b % 3 == 1]
```

??? Output "Output"
    array([ 1,  4,  7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46])


## **Iterating**

Iterating over `ndarrays` is very similar to iterating over regular python arrays. Note that iterating over multidimensional arrays is done with respect to the first axis.

```py
c = np.arange(24).reshape(2, 3, 4)  # A 3D array (composed of two 3x4 matrices)
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
for m in c:
    print("Item:")
    print(m)
```

??? Output "Output"
    Item:
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    Item:
    [[12 13 14 15]
     [16 17 18 19]
     [20 21 22 23]]

```py
for i in range(len(c)):  # Note that len(c) == c.shape[0]
    print("Item:")
    print(c[i])
```

??? Output "Output"
    Item:
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    Item:
    [[12 13 14 15]
     [16 17 18 19]
     [20 21 22 23]]

If you want to iterate on all elements in the `ndarray`, simply iterate over the `flat` attribute:

```py
for i in c.flat:
    print("Item:", i)
```

??? Output "Output"
    Item: 0
    Item: 1
    Item: 2
    Item: 3
    Item: 4
    Item: 5
    Item: 6
    Item: 7
    Item: 8
    Item: 9
    Item: 10
    Item: 11
    Item: 12
    Item: 13
    Item: 14
    Item: 15
    Item: 16
    Item: 17
    Item: 18
    Item: 19
    Item: 20
    Item: 21
    Item: 22
    Item: 23



## **Stacking arrays**

It is often useful to stack together different arrays. NumPy offers several functions to do just that. Let's start by creating a few arrays.

```py
q1 = np.full((3,4), 1.0)
q1
```

??? Output "Output"
    array([[ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.]])

```py
q2 = np.full((4,4), 2.0)
q2
```

??? Output "Output"
    array([[ 2.,  2.,  2.,  2.],
           [ 2.,  2.,  2.,  2.],
           [ 2.,  2.,  2.,  2.],
           [ 2.,  2.,  2.,  2.]])

```py
q3 = np.full((3,4), 3.0)
q3
```

??? Output "Output"
    array([[ 3.,  3.,  3.,  3.],
           [ 3.,  3.,  3.,  3.],
           [ 3.,  3.,  3.,  3.]])


### **vstack**

Now let's stack them vertically using `vstack`:

```py
q4 = np.vstack((q1, q2, q3))
q4
```

??? Output "Output"
    array([[ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 2.,  2.,  2.,  2.],
           [ 2.,  2.,  2.,  2.],
           [ 2.,  2.,  2.,  2.],
           [ 2.,  2.,  2.,  2.],
           [ 3.,  3.,  3.,  3.],
           [ 3.,  3.,  3.,  3.],
           [ 3.,  3.,  3.,  3.]])
        
```py
q4.shape
```

??? Output "Output"
    (10, 4)

This was possible because q1, q2 and q3 all have the same shape (except for the vertical axis, but that's ok since we are stacking on that axis).

### **hstack**

We can also stack arrays horizontally using `hstack`:

```py
q5 = np.hstack((q1, q3))
q5
```

??? Output "Output"
    array([[ 1.,  1.,  1.,  1.,  3.,  3.,  3.,  3.],
           [ 1.,  1.,  1.,  1.,  3.,  3.,  3.,  3.],
           [ 1.,  1.,  1.,  1.,  3.,  3.,  3.,  3.]])

```py
q5.shape
```

??? Output "Output"
    (3, 8)

This is possible because q1 and q3 both have 3 rows. But since q2 has 4 rows, it cannot be stacked horizontally with q1 and q3:

```py
try:
    q5 = np.hstack((q1, q2, q3))
except ValueError as e:
    print(e)
```

??? Output "Output"
    all the input array dimensions except for the concatenation axis must match exactly

### **concatenate**

The `concatenate` function stacks arrays along any given existing axis.

```py
q7 = np.concatenate((q1, q2, q3), axis=0)  # Equivalent to vstack
q7
```

??? Output "Output"
    array([[ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 2.,  2.,  2.,  2.],
           [ 2.,  2.,  2.,  2.],
           [ 2.,  2.,  2.,  2.],
           [ 2.,  2.,  2.,  2.],
           [ 3.,  3.,  3.,  3.],
           [ 3.,  3.,  3.,  3.],
           [ 3.,  3.,  3.,  3.]])

```py
q7.shape
```

??? Output "Output"
    (10, 4)

As you might guess, `hstack` is equivalent to calling `concatenate` with `axis=1`.


### **stack**

The `stack` function stacks arrays along a new axis. All arrays have to have the same shape.

```py
q8 = np.stack((q1, q3))
q8
```

??? Output "Output"
    array([[[ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.],
            [ 1.,  1.,  1.,  1.]],

           [[ 3.,  3.,  3.,  3.],
            [ 3.,  3.,  3.,  3.],
            [ 3.,  3.,  3.,  3.]]])

```py
q8.shape
```

??? Output "Output"
    (2, 3, 4)



## **Splitting arrays**

Splitting is the opposite of stacking. For example, let's use the `vsplit` function to split a matrix vertically.

First let's create a 6x4 matrix:

```py
r = np.arange(24).reshape(6,4)
r
```

??? Output "Output"
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23]])

Now let's split it in three equal parts, vertically:

```py
r1, r2, r3 = np.vsplit(r, 3)
r1
```

??? Output "Output"
    array([[0, 1, 2, 3],
           [4, 5, 6, 7]])

```py
r2
```

??? Output "Output"
    array([[ 8,  9, 10, 11],
           [12, 13, 14, 15]])

```py
r3
```

??? Output "Output"
    array([[16, 17, 18, 19],
           [20, 21, 22, 23]])

There is also a `split` function which splits an array along any given axis. Calling `vsplit` is equivalent to calling `split` with `axis=0`. There is also an `hsplit` function, equivalent to calling `split` with `axis=1`:

```py
r4, r5 = np.hsplit(r, 2)
r4
```

??? Output "Output"
    array([[ 0,  1],
           [ 4,  5],
           [ 8,  9],
           [12, 13],
           [16, 17],
           [20, 21]])

```py
r5
```

??? Output "Output"
    array([[ 2,  3],
           [ 6,  7],
           [10, 11],
           [14, 15],
           [18, 19],
           [22, 23]])



## **Transposing arrays**

The `transpose` method creates a new view on an `ndarray`'s data, with axes permuted in the given order.

For example, let's create a 3D array:

```py
t = np.arange(24).reshape(4,2,3)
t
```

??? Output "Output"
    array([[[ 0,  1,  2],
            [ 3,  4,  5]],

           [[ 6,  7,  8],
            [ 9, 10, 11]],

           [[12, 13, 14],
            [15, 16, 17]],

           [[18, 19, 20],
            [21, 22, 23]]])

Now let's create an `ndarray` such that the axes `0, 1, 2` (depth, height, width) are re-ordered to `1, 2, 0` (depth→width, height→depth, width→height):

```py
t1 = t.transpose((1,2,0))
t1
```

??? Output "Output"
    array([[[ 0,  6, 12, 18],
            [ 1,  7, 13, 19],
            [ 2,  8, 14, 20]],

           [[ 3,  9, 15, 21],
            [ 4, 10, 16, 22],
            [ 5, 11, 17, 23]]])

```py
t1.shape
```

??? Output "Output"
    (2, 3, 4)

By default, `transpose` reverses the order of the dimensions:

```py
t2 = t.transpose()  # equivalent to t.transpose((2, 1, 0))
t2
```

??? Output "Output"
    array([[[ 0,  6, 12, 18],
            [ 3,  9, 15, 21]],

           [[ 1,  7, 13, 19],
            [ 4, 10, 16, 22]],

           [[ 2,  8, 14, 20],
            [ 5, 11, 17, 23]]])

```py
t2.shape
```

??? Output "Output"
    (3, 2, 4)

NumPy provides a convenience function `swapaxes` to swap two axes. For example, let's create a new view of `t` with depth and height swapped:

```py
t3 = t.swapaxes(0,1)  # equivalent to t.transpose((1, 0, 2))
t3
```

??? Output "Output"
    array([[[ 0,  1,  2],
            [ 6,  7,  8],
            [12, 13, 14],
            [18, 19, 20]],

           [[ 3,  4,  5],
            [ 9, 10, 11],
            [15, 16, 17],
            [21, 22, 23]]])

```py
t3.shape
```

??? Output "Output"
    (2, 4, 3)



## **Linear algebra**

NumPy 2D arrays can be used to represent matrices efficiently in python. We will just quickly go through some of the main matrix operations available. For more details about Linear Algebra, vectors and matrics, go through the [Linear Algebra tutorial](https://colab.research.google.com/drive/math_linear_algebra.ipynb).

### **Matrix transpose**

The `T` attribute is equivalent to calling `transpose()` when the rank is ≥2:

```py
m1 = np.arange(10).reshape(2,5)
m1
```

??? Output "Output"
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])

```py
m1.T
```

??? Output "Output"
    array([[0, 5],
           [1, 6],
           [2, 7],
           [3, 8],
           [4, 9]])

The `T` attribute has no effect on rank 0 (empty) or rank 1 arrays:

```py
m2 = np.arange(5)
m2
```

??? Output "Output"
    array([0, 1, 2, 3, 4])

```py
m2.T
```

??? Output "Output"
    array([0, 1, 2, 3, 4])

We can get the desired transposition by first reshaping the 1D array to a single-row matrix (2D):

```py
m2r = m2.reshape(1,5)
m2r
```

??? Output "Output"
    array([[0, 1, 2, 3, 4]])

```py
m2r.T
```

??? Output "Output"
    array([[0],
          [1],
          [2],
          [3],
          [4]])
    
### **Matrix multiplication**

Let's create two matrices and execute a [matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication) using the `dot()` method.

```py
n1 = np.arange(10).reshape(2, 5)
n1
```

??? Output "Output"
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])

```py
n2 = np.arange(15).reshape(5,3)
n2
```

??? Output "Output"
    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11],
           [12, 13, 14]])
        
```py
n1.dot(n2)
```

??? Output "Output"
    array([[ 90, 100, 110],
           [240, 275, 310]])

!!! Warning "Caution"
    As mentionned previously, `n1*n2` is not a matric multiplication, it is an elementwise product (also called a [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))).

### **Matrix inverse and pseudo-inverse**

Many of the linear algebra functions are available in the `numpy.linalg` module, in particular the `inv` function to compute a square matrix's inverse:

```py
import numpy.linalg as linalg

m3 = np.array([[1,2,3],[5,7,11],[21,29,31]])
m3
```

??? Output "Output"
    array([[ 1,  2,  3],
           [ 5,  7, 11],
           [21, 29, 31]])

```py
linalg.inv(m3)
```

??? Output "Output"
    array([[-2.31818182,  0.56818182,  0.02272727],
           [ 1.72727273, -0.72727273,  0.09090909],
           [-0.04545455,  0.29545455, -0.06818182]])

You can also compute the [pseudoinverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse) using `pinv`:

```py
linalg.pinv(m3)
```

??? Output "Output"
    array([[-2.31818182,  0.56818182,  0.02272727],
           [ 1.72727273, -0.72727273,  0.09090909],
           [-0.04545455,  0.29545455, -0.06818182]])
        
### **Identity matrix**

The product of a matrix by its inverse returns the identiy matrix (with small floating point errors):

```py
m3.dot(linalg.inv(m3))
```

??? Output "Output"
    array([[  1.00000000e+00,  -1.11022302e-16,  -6.93889390e-18],
           [ -1.33226763e-15,   1.00000000e+00,  -5.55111512e-17],
           [  2.88657986e-15,   0.00000000e+00,   1.00000000e+00]])
     
You can create an identity matrix of size NxN by calling `eye`:

```py
np.eye(3)
```

??? Output "Output"
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

### **QR decomposition**
  
The `qr` function computes the [QR decomposition](https://en.wikipedia.org/wiki/QR_decomposition) of a matrix:

```py
q, r = linalg.qr(m3)
q
```

??? Output "Output"
    array([[-0.04627448,  0.98786672,  0.14824986],
           [-0.23137241,  0.13377362, -0.96362411],
           [-0.97176411, -0.07889213,  0.22237479]])

```py
r
```

??? Output "Output"
    array([[-21.61018278, -29.89331494, -32.80860727],
           [  0.        ,   0.62427688,   1.9894538 ],
           [  0.        ,   0.        ,  -3.26149699]])

```py
q.dot(r)  # q.r equals m3
```

??? Output "Output"
    array([[  1.,   2.,   3.],
           [  5.,   7.,  11.],
           [ 21.,  29.,  31.]])

### **Determinant**
 
The `det` function computes the [matrix determinant](https://en.wikipedia.org/wiki/Determinant):

```py
linalg.det(m3)  # Computes the matrix determinant
```

??? Output "Output"
    43.999999999999972

### **Eigenvalues and eigenvectors**
 
The `eig` function computes the [eigenvalues and eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) of a square matrix:

```py
eigenvalues, eigenvectors = linalg.eig(m3)
eigenvalues # λ
```

??? Output "Output"
    array([ 42.26600592,  -0.35798416,  -2.90802176])

```py
eigenvectors # v
```

??? Output "Output"
    array([[-0.08381182, -0.76283526, -0.18913107],
           [-0.3075286 ,  0.64133975, -0.6853186 ],
           [-0.94784057, -0.08225377,  0.70325518]])

```py
m3.dot(eigenvectors) - eigenvalues * eigenvectors  # m3.v - λ*v = 0
```

??? Output "Output"
    array([[  8.88178420e-15,   2.49800181e-15,  -3.33066907e-16],
           [  1.77635684e-14,  -1.66533454e-16,  -3.55271368e-15],
           [  3.55271368e-14,   3.61516372e-15,  -4.44089210e-16]])

### **Singular Value Decomposition**

The `svd` function takes a matrix and returns its [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition):

```py
m4 = np.array([[1,0,0,0,2], [0,0,3,0,0], [0,0,0,0,0], [0,2,0,0,0]])
m4
```

??? Output "Output"
    array([[1, 0, 0, 0, 2],
           [0, 0, 3, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 2, 0, 0, 0]])

```py
U, S_diag, V = linalg.svd(m4)
U
```

??? Output "Output"
    array([[ 0.,  1.,  0.,  0.],
           [ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  0., -1.],
           [ 0.,  0.,  1.,  0.]])

```py
S_diag
```

??? Output "Output"
    array([ 3.        ,  2.23606798,  2.        ,  0.        ])

The `svd` function just returns the values in the diagonal of Σ, but we want the full Σ matrix, so let's create it:

```py
S = np.zeros((4, 5))
S[np.diag_indices(4)] = S_diag
S  # Σ
```

??? Output "Output"
    array([[ 3.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  2.23606798,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  2.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])

```py
V
```

??? Output "Output"
    array([[-0.        ,  0.        ,  1.        , -0.        ,  0.        ],
           [ 0.4472136 ,  0.        ,  0.        ,  0.        ,  0.89442719],
           [-0.        ,  1.        ,  0.        , -0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ,  0.        ],
           [-0.89442719,  0.        ,  0.        ,  0.        ,  0.4472136 ]])
    
```py
U.dot(S).dot(V) # U.Σ.V == m4
```

??? Output "Output"
    array([[ 1.,  0.,  0.,  0.,  2.],
           [ 0.,  0.,  3.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  2.,  0.,  0.,  0.]])

### **Diagonal and trace**

```py
np.diag(m3)  # the values in the diagonal of m3 (top left to bottom right)
```

??? Output "Output"
    array([ 1,  7, 31])

```py
np.trace(m3)  # equivalent to np.diag(m3).sum()
```

??? Output "Output"
    39

### **Solving a system of linear scalar equations**

The `solve` function solves a system of linear scalar equations, such as:

* $2x + 6y = 6$
* $5x + 3y = -9$

```py
coeffs  = np.array([[2, 6], [5, 3]])
depvars = np.array([6, -9])
solution = linalg.solve(coeffs, depvars)
solution
```

??? Output "Output"
    array([-3.,  2.])

!!! Info ""
    Let's check the solution:

```py
coeffs.dot(solution), depvars  # yep, it's the same
```

??? Output "Output"
    (array([ 6., -9.]), array([ 6, -9]))

!!! Info ""
    Looks good! Another way to check the solution:

```py
np.allclose(coeffs.dot(solution), depvars)
```

??? Output "Output"
    True


## **Vectorization**

Instead of executing operations on individual array items, one at a time, your code is much more efficient if you try to stick to array operations. This is called *vectorization*. This way, you can benefit from NumPy's many optimizations.

For example, let's say we want to generate a 768x1024 array based on the formula $sin(xy/40.5)$. A **bad** option would be to do the math in python using nested loops:

```py
import math
data = np.empty((768, 1024))
for y in range(768):
    for x in range(1024):
        data[y, x] = math.sin(x*y/40.5)  # BAD! Very inefficient.
```

Sure, this works, but it's terribly inefficient since the loops are taking place in pure python. Let's vectorize this algorithm. First, we will use NumPy's `meshgrid` function which generates coordinate matrices from coordinate vectors.

```py
x_coords = np.arange(0, 1024)  # [0, 1, 2, ..., 1023]
y_coords = np.arange(0, 768)   # [0, 1, 2, ..., 767]
X, Y = np.meshgrid(x_coords, y_coords)
X
```

??? Output "Output"
    array([[   0,    1,    2, ..., 1021, 1022, 1023],
           [   0,    1,    2, ..., 1021, 1022, 1023],
           [   0,    1,    2, ..., 1021, 1022, 1023],
           ..., 
           [   0,    1,    2, ..., 1021, 1022, 1023],
           [   0,    1,    2, ..., 1021, 1022, 1023],
           [   0,    1,    2, ..., 1021, 1022, 1023]])
    
```py
Y
```

??? Output "Output"
    array([[  0,   0,   0, ...,   0,   0,   0],
           [  1,   1,   1, ...,   1,   1,   1],
           [  2,   2,   2, ...,   2,   2,   2],
           ..., 
           [765, 765, 765, ..., 765, 765, 765],
           [766, 766, 766, ..., 766, 766, 766],
           [767, 767, 767, ..., 767, 767, 767]])

As you can see, both `X` and `Y` are 768x1024 arrays, and all values in `X` correspond to the horizontal coordinate, while all values in `Y` correspond to the the vertical coordinate.

Now we can simply compute the result using array operations:

```py
data = np.sin(X*Y/40.5)
```

Now we can plot this data using matplotlib's `imshow` function

```py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
fig = plt.figure(1, figsize=(7, 6))
plt.imshow(data, cmap=cm.hot, interpolation="bicubic")
plt.show()
```

## **Saving and loading**

NumPy makes it easy to save and load `ndarray`s in binary or text format.

### **Binary `.npy` format**

Let's create a random array and save it.

```py
a = np.random.rand(2,3)
a
```

??? Output "Output"
    array([[ 0.41307972,  0.20933385,  0.32025581],
           [ 0.19853514,  0.408001  ,  0.6038287 ]])

```py
np.save("my_array", a)
```

Done! Since the file name contains no file extension was provided, NumPy automatically added `.npy`. Let's take a peek at the file content:

```py
with open("my_array.npy", "rb") as f:
    content = f.read()

content
```

??? Output "Output"
    "\x93NUMPY\x01\x00F\x00{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }          \n\xa8\x96\x1d\xeb\xe5o\xda? \x06W\xa1s\xcb\xca?*\xdeB>\x12\x7f\xd4?x<h\x81\x99i\xc9?@\xa4\x027\xb0\x1c\xda?<P\x05\x8f\x90R\xe3?"

To load this file into a NumPy array, simply call `load`:


```py
a_loaded = np.load("my_array.npy")
a_loaded
```

??? Output "Output"
    array([[ 0.41307972,  0.20933385,  0.32025581],
           [ 0.19853514,  0.408001  ,  0.6038287 ]])

### **Text format**

Let's try saving the array in text format:

```py
np.savetxt("my_array.csv", a)
```

Now let's look at the file content:

```py
with open("my_array.csv", "rt") as f:
    print(f.read())
```

??? Output "Output"
    4.130797191668116319e-01 2.093338525574361952e-01 3.202558143634371968e-01
    1.985351449843368865e-01 4.080009972772735694e-01 6.038286965726977762e-01

This is a CSV file with tabs as delimiters. You can set a different delimiter:

```py
np.savetxt("my_array.csv", a, delimiter=",")
```

To load this file, just use `loadtxt`:

```py
a_loaded = np.loadtxt("my_array.csv", delimiter=",")
a_loaded
```

??? Output "Output"
    array([[ 0.41307972,  0.20933385,  0.32025581],
           [ 0.19853514,  0.408001  ,  0.6038287 ]])
    
### **Zipped `.npz` format**

It is also possible to save multiple arrays in one zipped file:

```py
b = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
b
```

??? Output "Output"
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],

           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]], dtype=uint8)

```py
np.savez("my_arrays", my_a=a, my_b=b)
```

Again, let's take a peek at the file content. Note that the `.npz` file extension was automatically added.

```py
with open("my_arrays.npz", "rb") as f:
    content = f.read()

repr(content)[:180] + "[...]"
```

??? Output "Output"
    u'"PK\\x03\\x04\\x14\\x00\\x00\\x00\\x00\\x00x\\x94cH\\xb6\\x96\\xe4{h\\x00\\x00\\x00h\\x00\\x00\\x00\\x08\\x00\\x00\\x00my_b.npy\\x93NUMPY\\x01\\x00F\\x00{\'descr\': \'|u1\', \'fortran_order\': False, \'shape\': (2,[...]'

You then load this file like so:

```py
my_arrays = np.load("my_arrays.npz")
my_arrays
```
??? Output "Output"
    <numpy.lib.npyio.NpzFile at 0x10fa4d4d0>

This is a dict-like object which loads the arrays lazily:

```py
my_arrays.keys()
```

??? Output "Output"
    ['my_b', 'my_a']

```py
my_arrays["my_a"]
```

??? Output "Output"
    array([[ 0.41307972,  0.20933385,  0.32025581],
           [ 0.19853514,  0.408001  ,  0.6038287 ]])

## **What next?**

Now you know all the fundamentals of NumPy, but there are many more options available. The best way to learn more is to experiment with NumPy, and go through the excellent [reference documentation](http://docs.scipy.org/doc/numpy/reference/index.html) to find more functions and features you may be interested in.