# pymatgraph

pymatgraph is a Python package that provides a multiprocess-safe buffer for PyTorch tensors, specifically designed for rendering RGB matrices and tables using Pygame. This package allows for efficient sharing of tensor data between processes, making it suitable for applications that require real-time rendering and updates.

## Features

- **Multiprocess Safe**: Utilizes shared memory and locks to ensure safe access to tensor data across multiple processes.
- **Flexible Modes**: Supports both numerical and RGB modes for tensor data.
- **Table Rendering**: Built-in utilities to render structured tabular data directly on the screen.
- **Easy Integration**: Designed to work seamlessly with Pygame for rendering visual data.

## Installation

You can install the pymatgraph package using pip:

```bash
pip install pymatgraph
```

## Usage
Here is a simple example of how to use the pymatgraph package:
```python
width, height = 640, 480
buffer = MultiprocessSafeTensorBuffer(n=height, m=width, mode="rgb")
buffer.write_matrix(torch.zeros((height,width,3), dtype=torch.uint8))

g = Graphics(width=width, height=height, bg_color=(30,30,30))

text1 = Text("Custom Rendering Engine!", x=50, y=50, font_size=32, color=(255,255,0))
table1 = Table(
    data=[["Name","Age"], ["Alice","24"], ["Bob","30"]],
    x=50, y=120, cell_width=120, cell_height=40,
    bg_color=(50,50,100), grid_color=(255,255,255)
)

text1.render_to_tensor(buffer)
table1.render_to_tensor(buffer)

g.run(buffer)
```
