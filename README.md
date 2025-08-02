# MatrixBuffer

MatrixBuffer is a Python package that provides a multiprocess-safe buffer for PyTorch tensors, specifically designed for rendering RGB matrices using Pygame. This package allows for efficient sharing of tensor data between processes, making it suitable for applications that require real-time rendering and updates.

## Features

- **Multiprocess Safe**: Utilizes shared memory and locks to ensure safe access to tensor data across multiple processes.
- **Flexible Modes**: Supports both numerical and RGB modes for tensor data.
- **Easy Integration**: Designed to work seamlessly with Pygame for rendering visual data.

## Installation

You can install the MatrixBuffer package using pip. Run the following command:

```
pip install matrixbuffer
```

## Usage

Here is a simple example of how to use the MatrixBuffer package:

```python
import pygame
from matrixbuffer.MatrixBuffer import MultiprocessSafeTensorBuffer, Render, update_buffer_process
import multiprocessing

# Initialize Pygame and create a window
pygame.init()
screen = pygame.display.set_mode((800, 600))

# Create a multiprocess-safe tensor buffer
rgb_buffer = MultiprocessSafeTensorBuffer(n=240, m=320, mode="rgb")

# Create a renderer
renderer = Render(rgb_buffer, screen)

# Start the worker process to update the buffer
stop_event = multiprocessing.Event()
worker_process = multiprocessing.Process(target=update_buffer_process, args=(rgb_buffer, stop_event))
worker_process.start()

# Main loop for rendering
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    renderer.render()

# Clean up
stop_event.set()
worker_process.join()
pygame.quit()
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.