# File: matrixbuffer/matrixbuffer/MatrixBuffer.py

import pygame
import numpy as np
import threading
import time
import torch
import multiprocessing
import ctypes

BUFFER_UPDATED_EVENT = pygame.USEREVENT + 1

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
        expected_shape = (self._n, self._m)
        if self._mode == "rgb":
            expected_shape = (self._n, self._m, self._bytes_per_pixel)

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
            if self._mode == "numerical":
                tensor_shape = (self._n, self._m)
            elif self._mode == "rgb":
                tensor_shape = (self._n, self._m, self._bytes_per_pixel)
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

class Render:
    def __init__(self, buffer: MultiprocessSafeTensorBuffer, display_surface: pygame.Surface):
        if buffer.get_mode() != "rgb":
            raise ValueError("Render class is currently only supported for 'rgb' mode buffers.")

        self._buffer = buffer
        self._display = display_surface
        self._n, self._m = self._buffer.get_dimensions()
        self._tensor_surface = pygame.Surface((self._m, self._n))
        self._blit_buffer_to_display()

    def _blit_buffer_to_display(self):
        try:
            tensor_data = self._buffer.read_matrix()
            np_data = tensor_data.cpu().numpy()
            np_data_transposed = np.transpose(np_data, (1, 0, 2))
            temp_surface = pygame.surfarray.make_surface(np_data_transposed)
            scaled_surface = pygame.transform.scale(temp_surface, self._display.get_size())
            self._display.blit(scaled_surface, (0, 0))
            pygame.display.flip()
        except ValueError as e:
            print(f"Rendering error: {e}")

    def render(self):
        self._blit_buffer_to_display()

def update_buffer_process(shared_buffer: MultiprocessSafeTensorBuffer, update_event: multiprocessing.Event, stop_event: multiprocessing.Event):
    height, width = shared_buffer.get_dimensions()
    if shared_buffer.get_mode() != "rgb":
        print("Error: The buffer is not in 'rgb' mode.")
        return

    while not stop_event.is_set():
        new_data = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8)
        shared_buffer.write_matrix(new_data)
        time.sleep(0.01)