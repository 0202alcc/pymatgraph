from pymatgraph.MatrixBuffer import MultiprocessSafeTensorBuffer
from pymatgraph.Graphics import Graphics, Text, Table
import torch

width, height = 640, 480
buffer = MultiprocessSafeTensorBuffer(n=height, m=width, mode="rgb")
buffer.write_matrix(torch.zeros((height, width, 3), dtype=torch.uint8))

# Choose one backend: "pygame" or "kivy"
g = Graphics(width=width, height=height, bg_color=(30,30,30), backend="pygame")

text1 = Text("Hello World!", x=50, y=50, font_size=32, color=(255,255,0))
table1 = Table([["Name","Age"], ["Alice","24"], ["Bob","30"]], x=50, y=120)

# ⚠️ Use render_to_buffer, not render_to_tensor
text1.render_to_buffer(buffer)
table1.render_to_buffer(buffer)

g.run(buffer)
