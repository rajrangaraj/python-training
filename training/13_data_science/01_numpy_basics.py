"""
Demonstration of NumPy fundamentals and array operations.
"""

import numpy as np
from typing import List, Tuple, Any
import time
import sys

# Basic array creation
def demonstrate_array_creation():
    """Examples of different ways to create NumPy arrays."""
    
    # From Python lists
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    
    # Using NumPy functions
    zeros = np.zeros((3, 4))  # 3x4 array of zeros
    ones = np.ones((2, 3))    # 2x3 array of ones
    empty = np.empty((2, 2))  # 2x2 uninitialized array
    
    # Range-based arrays
    range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
    linspace = np.linspace(0, 1, 5)  # 5 evenly spaced points between 0 and 1
    
    # Special arrays
    identity = np.eye(3)        # 3x3 identity matrix
    random = np.random.rand(3, 3)  # 3x3 random array
    
    return {
        'basic_1d': arr1,
        'basic_2d': arr2,
        'zeros': zeros,
        'ones': ones,
        'empty': empty,
        'range': range_arr,
        'linspace': linspace,
        'identity': identity,
        'random': random
    }

# Array operations and mathematics
def demonstrate_array_operations():
    """Examples of array operations and mathematical functions."""
    
    # Create sample arrays
    arr1 = np.array([1, 2, 3, 4])
    arr2 = np.array([5, 6, 7, 8])
    matrix1 = np.array([[1, 2], [3, 4]])
    matrix2 = np.array([[5, 6], [7, 8]])
    
    # Basic operations
    addition = arr1 + arr2
    subtraction = arr2 - arr1
    multiplication = arr1 * arr2
    division = arr2 / arr1
    
    # Matrix operations
    dot_product = np.dot(matrix1, matrix2)
    matrix_multiply = matrix1 @ matrix2  # Same as dot product
    
    # Mathematical functions
    sqrt = np.sqrt(arr1)
    exp = np.exp(arr1)
    log = np.log(arr1)
    sin = np.sin(arr1)
    
    # Statistical operations
    mean = np.mean(arr1)
    std = np.std(arr1)
    var = np.var(arr1)
    min_val = np.min(arr1)
    max_val = np.max(arr1)
    
    return {
        'addition': addition,
        'subtraction': subtraction,
        'multiplication': multiplication,
        'division': division,
        'dot_product': dot_product,
        'matrix_multiply': matrix_multiply,
        'sqrt': sqrt,
        'exp': exp,
        'log': log,
        'sin': sin,
        'stats': {
            'mean': mean,
            'std': std,
            'var': var,
            'min': min_val,
            'max': max_val
        }
    }

# Array manipulation
def demonstrate_array_manipulation():
    """Examples of array reshaping, indexing, and slicing."""
    
    # Create a sample array
    arr = np.arange(12)
    
    # Reshaping
    reshaped_2d = arr.reshape(3, 4)
    reshaped_3d = arr.reshape(2, 2, 3)
    
    # Indexing and slicing
    single_element = arr[5]
    slice_elements = arr[2:7]
    stride_elements = arr[::2]
    
    # 2D array operations
    row = reshaped_2d[1]      # Second row
    column = reshaped_2d[:, 1]  # Second column
    block = reshaped_2d[1:3, 1:3]  # 2x2 block
    
    # Boolean indexing
    bool_mask = arr > 5
    filtered = arr[bool_mask]
    
    # Fancy indexing
    indices = [1, 3, 5]
    selected = arr[indices]
    
    return {
        'original': arr,
        'reshaped_2d': reshaped_2d,
        'reshaped_3d': reshaped_3d,
        'single_element': single_element,
        'slice': slice_elements,
        'stride': stride_elements,
        'row': row,
        'column': column,
        'block': block,
        'filtered': filtered,
        'selected': selected
    }

# Performance comparison
def compare_performance():
    """Compare performance between Python lists and NumPy arrays."""
    
    size = 1000000
    
    # Python list operations
    start_time = time.time()
    python_list = list(range(size))
    python_result = [x * 2 for x in python_list]
    python_time = time.time() - start_time
    
    # NumPy array operations
    start_time = time.time()
    numpy_array = np.arange(size)
    numpy_result = numpy_array * 2
    numpy_time = time.time() - start_time
    
    # Memory usage
    python_memory = sys.getsizeof(python_list) + sum(sys.getsizeof(x) for x in python_list)
    numpy_memory = numpy_array.nbytes
    
    return {
        'python_time': python_time,
        'numpy_time': numpy_time,
        'python_memory': python_memory,
        'numpy_memory': numpy_memory,
        'speedup': python_time / numpy_time,
        'memory_ratio': python_memory / numpy_memory
    }

if __name__ == '__main__':
    # Array creation examples
    arrays = demonstrate_array_creation()
    print("\nArray Creation Examples:")
    for name, arr in arrays.items():
        print(f"\n{name}:\n{arr}")
    
    # Array operations examples
    operations = demonstrate_array_operations()
    print("\nArray Operations Examples:")
    for name, result in operations.items():
        print(f"\n{name}:\n{result}")
    
    # Array manipulation examples
    manipulations = demonstrate_array_manipulation()
    print("\nArray Manipulation Examples:")
    for name, result in manipulations.items():
        print(f"\n{name}:\n{result}")
    
    # Performance comparison
    performance = compare_performance()
    print("\nPerformance Comparison:")
    print(f"Python List Time: {performance['python_time']:.4f} seconds")
    print(f"NumPy Array Time: {performance['numpy_time']:.4f} seconds")
    print(f"Speedup: {performance['speedup']:.2f}x")
    print(f"Memory Usage Ratio: {performance['memory_ratio']:.2f}x") 