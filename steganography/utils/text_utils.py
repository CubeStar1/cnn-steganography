import numpy as np

def text_to_bits(text, data_depth):
    """Convert text to binary representation."""
    binary = ''.join(format(ord(char), '08b') for char in text)
    if len(binary) > data_depth:
        raise ValueError(f"Text too long for data_depth of {data_depth}")
    binary = binary.ljust(data_depth, '0')
    return np.array([float(bit) * 2 - 1 for bit in binary]) 