# File: matrixbuffer/matrixbuffer/MatrixBuffer.py

import numpy as np
import multiprocessing
import ctypes
import torch
import time


class MultiprocessSafeTensorBuffer:
    def __init__(self, n=None, m=None, initial_data=None, mode="numerical", dtype=torch.float32):
        self._lock = multiprocessing.Lock()
        self._update_event = multiprocessing.Event()
        self._mode = mode.lower()
        if self._mode not in ["numerical", "rgb"]:
            raise ValueError("Invalid mode. Must be 'numerical' or 'rgb'.")
        self._n = None
        self._m = None
        self._dtype = None
        self._numpy_dtype = None
        self._ctype = None
        self._element_size_bytes = None
        self._bytes_per_pixel = None
        self._buffer_size_bytes = None
        self._shared_array = None

        if initial_data is not None:
            if isinstance(initial_data, np.ndarray):
                initial_tensor = torch.from_numpy(initial_data.copy())
            elif isinstance(initial_data, torch.Tensor):
                initial_tensor = initial_data
            else:
                raise TypeError("initial_data must be a NumPy array or a PyTorch tensor.")
            self._initialize_from_tensor(initial_tensor, mode)
        else:
            if not isinstance(n, int) or n <= 0:
                raise ValueError("N (rows) must be a positive integer when initial_data is not provided.")
            if not isinstance(m, int) or m <= 0:
                raise ValueError("M (columns) must be a positive integer when initial_data is not provided.")
            self._n = n
            self._m = m
            if self._mode == "numerical":
                self._dtype = dtype
                self._numpy_dtype = self._get_numpy_dtype(self._dtype)
                self._ctype = self._get_ctype(self._dtype)
                if self._ctype is None:
                    raise ValueError(f"Unsupported torch.dtype for numerical mode: {self._dtype}")
                self._element_size_bytes = ctypes.sizeof(self._ctype)
                initial_tensor = torch.zeros((n, m), dtype=self._dtype)
            elif self._mode == "rgb":
                self._dtype = torch.uint8
                self._numpy_dtype = np.uint8
                self._ctype = ctypes.c_uint8
                self._bytes_per_pixel = 3
                self._element_size_bytes = ctypes.sizeof(self._ctype)
                initial_tensor = torch.zeros((n, m, self._bytes_per_pixel), dtype=self._dtype)

            self._buffer_size_bytes = initial_tensor.numel() * self._element_size_bytes
            self._shared_array = multiprocessing.Array(self._ctype, self._buffer_size_bytes)
            self._write_to_shared_array(initial_tensor)

    def _initialize_from_tensor(self, tensor, mode):
        if mode == "numerical":
            if tensor.ndim != 2:
                raise ValueError(f"Numerical mode expects a 2D tensor, but got {tensor.ndim}D.")
            self._n, self._m = tensor.shape
            self._dtype = tensor.dtype
            self._numpy_dtype = self._get_numpy_dtype(self._dtype)
            self._ctype = self._get_ctype(self._dtype)
            if self._ctype is None:
                raise ValueError(f"Unsupported torch.dtype for numerical mode: {self._dtype}")
            self._element_size_bytes = ctypes.sizeof(self._ctype)
        elif mode == "rgb":
            if tensor.ndim != 3 or tensor.shape[2] != 3:
                raise ValueError(f"RGB mode expects a 3D tensor with 3 channels (N, M, 3), but got shape {tensor.shape}.")
            if tensor.dtype != torch.uint8:
                raise ValueError("RGB components must be unsigned 8-bit integers (torch.uint8).")
            self._n, self._m, self._bytes_per_pixel = tensor.shape
            self._dtype = torch.uint8
            self._numpy_dtype = np.uint8
            self._ctype = ctypes.c_uint8
            self._element_size_bytes = ctypes.sizeof(self._ctype)

        self._buffer_size_bytes = tensor.numel() * self._element_size_bytes
        self._shared_array = multiprocessing.Array(self._ctype, self._buffer_size_bytes)
        self._write_to_shared_array(tensor)

    def get_update_event(self):
        return self._update_event

    def _get_ctype(self, torch_dtype):
        if torch_dtype == torch.float32:
            return ctypes.c_float
        elif torch_dtype == torch.float64:
            return ctypes.c_double
        elif torch_dtype == torch.int32:
            return ctypes.c_int32
        elif torch_dtype == torch.int64:
            return ctypes.c_int64
        elif torch_dtype == torch.uint8:
            return ctypes.c_uint8
        elif torch_dtype == torch.bool:
            return ctypes.c_bool
        else:
            return None

    def _get_numpy_dtype(self, torch_dtype):
        if torch_dtype == torch.float32:
            return np.float32
        elif torch_dtype == torch.float64:
            return np.float64
        elif torch_dtype == torch.int32:
            return np.int32
        elif torch_dtype == torch.int64:
            return np.int64
        elif torch_dtype == torch.uint8:
            return np.uint8
        elif torch_dtype == torch.bool:
            return np.bool_
        else:
            return None

    def _write_to_shared_array(self, tensor):
        if self._mode == "rgb":
            expected_shape = (self._n, self._m, self._bytes_per_pixel)
        else:
            expected_shape = (self._n, self._m)

        if tensor.shape != expected_shape:
            raise ValueError(f"Tensor shape must be {expected_shape}, but got {tensor.shape}.")
        if tensor.dtype != self._dtype:
            raise ValueError(f"Tensor dtype must be {self._dtype}, but got {tensor.dtype}.")

        np_array = tensor.cpu().numpy()
        np_shared = np.frombuffer(self._shared_array.get_obj(), dtype=self._numpy_dtype)
        np_shared[:] = np_array.flatten()
        self._update_event.set()

    def read_matrix(self):
        with self._lock:
            np_shared = np.frombuffer(self._shared_array.get_obj(), dtype=self._numpy_dtype)
            tensor_shape = (self._n, self._m) if self._mode == "numerical" else (self._n, self._m, self._bytes_per_pixel)
            return torch.from_numpy(np_shared.copy().reshape(tensor_shape)).to(self._dtype)

    def write_matrix(self, new_tensor):
        with self._lock:
            self._write_to_shared_array(new_tensor)

    def add_matrix(self, other_tensor):
        with self._lock:
            current_tensor = self.read_matrix()
            if current_tensor.shape != other_tensor.shape or current_tensor.dtype != other_tensor.dtype:
                raise ValueError("Shape and dtype must match for addition.")
            result_tensor = current_tensor + other_tensor
            self._write_to_shared_array(result_tensor)

    def get_dimensions(self):
        return (self._n, self._m)

    def get_mode(self):
        return self._mode

    def get_dtype(self):
        return self._dtype

    def get_buffer_size_bytes(self):
        return self._buffer_size_bytes

    def empty_like(self, bg_color=None):
        return LocalTensorBuffer(self._n, self._m, mode=self._mode, dtype=self._dtype, bg_color=bg_color)

    def inplace_matrix(self):
        return self._tensor  # Returns reference to underlying tensor for in-place modifications


class LocalTensorBuffer:
    """
    Local tensor buffer for off-screen rendering.
    """
    def __init__(self, n, m, mode="rgb", dtype=torch.uint8, bg_color=None):
        self._n = n
        self._m = m
        self._mode = mode
        self._dtype = dtype
        if mode == "rgb":
            self._tensor = torch.zeros((n, m, 3), dtype=dtype)
            if bg_color is not None:
                self._tensor[:] = torch.tensor(bg_color, dtype=dtype)
        else:
            self._tensor = torch.zeros((n, m), dtype=dtype)

    def read_matrix(self):
        return self._tensor.clone()

    def write_matrix(self, new_tensor):
        self._tensor = new_tensor.clone()

    def get_dimensions(self):
        return (self._n, self._m)

    def get_mode(self):
        return self._mode

    def get_dtype(self):
        return self._dtype

    def inplace_matrix(self):
        return self._tensor
