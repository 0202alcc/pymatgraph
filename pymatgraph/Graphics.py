# File: matrixbuffer/Graphics.py

import torch
from PIL import Image, ImageDraw, ImageFont
from pymatgraph.MatrixBuffer import MultiprocessSafeTensorBuffer

# For font path
import importlib.resources as pkg_resources
from pathlib import Path
import pymatgraph

class Text:
    """
    Rasterizes text into a MultiprocessSafeTensorBuffer (RGB).
    """
    def __init__(self, text, x, y, font_path=None, font_size=16, color=(255,255,255)):
        self.text = text
        self.x = x
        self.y = y
        self.font_size = font_size
        self.color = color

        # Default font if none is provided
        if font_path is None:
            with pkg_resources.path(pymatgraph, "fonts/ComicMono.ttf") as p:
                font_path = str(p)

        try:
            self.font = ImageFont.truetype(font_path, font_size)
        except Exception:
            self.font = ImageFont.load_default()

    def render_to_buffer(self, buffer: MultiprocessSafeTensorBuffer):
        """Draws the text onto the given tensor buffer."""
        if not self.text:
            return

        # Measure text size
        bbox = self.font.getbbox(self.text)
        text_w, text_h = max(1, bbox[2]-bbox[0]), max(1, bbox[3]-bbox[1])

        # Create RGBA image for text
        img = Image.new("RGBA", (text_w, text_h), (0,0,0,0))
        draw = ImageDraw.Draw(img)
        draw.text((-bbox[0], -bbox[1]), self.text, font=self.font, fill=(*self.color, 255))

        # Convert to tensor
        arr = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).reshape(text_h, text_w, 4)
        text_rgb = arr[...,:3].to(torch.float32)
        alpha = arr[...,3:4].to(torch.float32) / 255.0

        buf = buffer.read_matrix().to(torch.float32)
        H, W = buffer.get_dimensions()

        # Clamp text region into buffer
        x0, y0 = max(0,self.x), max(0,self.y)
        x1, y1 = min(W, self.x+text_w), min(H, self.y+text_h)
        if x0>=x1 or y0>=y1:
            return

        tx0, ty0 = 0, 0
        tx1, ty1 = x1-x0, y1-y0

        # Alpha blend text into buffer
        region = buf[y0:y1, x0:x1]
        buf[y0:y1, x0:x1] = text_rgb[ty0:ty1, tx0:tx1]*alpha[ty0:ty1, tx0:tx1] + region*(1-alpha[ty0:ty1, tx0:tx1])
        buffer.write_matrix(buf.to(torch.uint8))


class Table:
    """
    Rasterizes tabular data into a MultiprocessSafeTensorBuffer (RGB).
    """
    def __init__(self, data, x, y, font_path=None, font_size=16,
                 cell_width=100, cell_height=40, grid_color=(200,200,200),
                 bg_color=None, text_color=(255,255,255), expand_cells=False):
        self.data = data
        self.x = x
        self.y = y
        self.font_size = font_size
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.grid_color = grid_color
        self.bg_color = bg_color
        self.text_color = text_color
        self.font_path = font_path
        if font_path is None:
            # Load the font from package data
            with pkg_resources.path(pymatgraph, "fonts/ComicMono.ttf") as p:
                self.font_path = str(p)
        self.expand_cells = expand_cells
        self.column_widths = [cell_width] * max(len(r) for r in self.data)

    def render_to_buffer(self, buffer: MultiprocessSafeTensorBuffer):
        """Draws the table into the given tensor buffer."""
        H, W = buffer.get_dimensions()
        buf = buffer.read_matrix().to(torch.float32)
        rows, cols = len(self.data), max(len(r) for r in self.data)
        ch = self.cell_height

        # Expand cells dynamically if requested
        if self.expand_cells:
            for c in range(cols):
                max_text_width = 0
                for r in range(rows):
                    if c < len(self.data[r]):
                        text = str(self.data[r][c])
                        font = ImageFont.truetype(self.font_path, self.font_size)
                        bbox = font.getbbox(text)
                        w = bbox[2] - bbox[0]
                        max_text_width = max(max_text_width, w)
                self.column_widths[c] = max(self.column_widths[c], max_text_width + 8)

        # Table bounds
        total_width = sum(self.column_widths)
        x_start, x_end = self.x, min(W, self.x + total_width)
        y_start, y_end = self.y, min(H, self.y + rows*ch)

        # Fill background
        if self.bg_color:
            buf[y_start:y_end, x_start:x_end] = torch.tensor(self.bg_color, dtype=torch.float32)

        # Draw grid lines
        row_mask = torch.zeros(y_end - y_start, dtype=torch.bool)
        for r in range(rows+1):
            y_pos = r * ch
            if y_pos < y_end-y_start:
                row_mask[y_pos] = True

        col_mask = torch.zeros(x_end - x_start, dtype=torch.bool)
        current_x = 0
        for c_width in self.column_widths:
            if current_x < x_end-x_start:
                col_mask[current_x] = True
            current_x += c_width
        if current_x < x_end-x_start:
            col_mask[current_x] = True

        grid_mask = row_mask[:, None] | col_mask[None, :]
        buf[y_start:y_end, x_start:x_end][grid_mask] = torch.tensor(self.grid_color, dtype=torch.float32)

        # Render all text into a single image
        table_width = x_end - x_start
        table_height = y_end - y_start
        text_layer = Image.new("RGBA", (table_width, table_height), (0,0,0,0))
        draw = ImageDraw.Draw(text_layer)

        for r, row in enumerate(self.data):
            current_x_offset = 0
            for c, cell_text in enumerate(row):
                if not cell_text:
                    continue

                font = ImageFont.truetype(self.font_path, self.font_size)
                bbox = font.getbbox(str(cell_text))
                text_width = bbox[2] - bbox[0]

                cell_x = current_x_offset
                if r > 0 and c > 0:  # right-align numeric cells
                    cell_x += self.column_widths[c] - text_width - 4

                cell_y = r * ch
                draw.text((cell_x, cell_y), str(cell_text), font=font, fill=(*self.text_color,255))
                current_x_offset += self.column_widths[c]

        # Convert text layer to tensor
        arr = torch.ByteTensor(torch.ByteStorage.from_buffer(text_layer.tobytes())).reshape(table_height, table_width, 4)
        text_rgb = arr[...,:3].to(torch.float32)
        alpha = arr[...,3:4].to(torch.float32) / 255.0

        # Blend onto buffer
        buf[y_start:y_end, x_start:x_end] = text_rgb*alpha + buf[y_start:y_end, x_start:x_end]*(1-alpha)
        buffer.write_matrix(buf.to(torch.uint8))

class Graphics:
    def __init__(self, width=800, height=600, bg_color=(0,0,0), backend="pygame"):
        self.width = width
        self.height = height
        self.bg_color = bg_color

        if backend == "pygame":
            from pymatgraph.backends.pygame_backend import PygameRenderer
            self.renderer = PygameRenderer(width, height, bg_color)

        elif backend == "kivy":
            from pymatgraph.backends.kivy_backend import KivyRenderer
            self.renderer = KivyRenderer(width, height, bg_color)

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def add(self, obj):
        self.renderer.add(obj)

    def run(self, buffer: MultiprocessSafeTensorBuffer):
        self.renderer.run(buffer)