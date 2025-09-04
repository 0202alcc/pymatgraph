# Graphics.py
# File: pymatgraph/Graphics.py

import torch
from PIL import Image, ImageDraw, ImageFont
from pymatgraph.MatrixBuffer import MultiprocessSafeTensorBuffer, LocalTensorBuffer
from importlib.resources import files
import pymatgraph


class GraphicObject:
    def __init__(self, x=0, y=0, visible=True, opacity=1.0, z_index=0):
        self.x = x
        self.y = y
        self.visible = visible
        self.opacity = float(opacity)
        self.z_index = int(z_index)

    def render_to_buffer(self, buffer):
        raise NotImplementedError


class Text(GraphicObject):
    def __init__(self, text, x, y, font_path=None, font_size=16, color=(255, 255, 255),
                 visible=True, opacity=1.0, z_index=0):
        super().__init__(x, y, visible, opacity, z_index)
        self.text = str(text)
        self.font_size = int(font_size)
        self.color = tuple(map(int, color))

        if font_path is None:
            font_path = files(pymatgraph) / "fonts" / "ComicMono.ttf"
        try:
            self.font = ImageFont.truetype(str(font_path), self.font_size)
        except Exception:
            self.font = ImageFont.load_default()

    def render_to_buffer(self, buffer):
        if not self.visible or not self.text:
            return

        H, W = buffer.shape[:2] if isinstance(buffer, torch.Tensor) else buffer.get_dimensions()
        buf = buffer if isinstance(buffer, torch.Tensor) else buffer.inplace_matrix()

        bbox = self.font.getbbox(self.text)
        text_w, text_h = max(1, bbox[2]-bbox[0]), max(1, bbox[3]-bbox[1])
        img = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((-bbox[0], -bbox[1]), self.text, font=self.font, fill=(*self.color, 255))

        arr = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).reshape(text_h, text_w, 4)
        text_rgb = arr[..., :3].to(torch.float32)
        alpha = (arr[..., 3:4].to(torch.float32)/255.0) * self.opacity

        x0, y0 = int(self.x), int(self.y)
        x1, y1 = x0 + text_w, y0 + text_h
        vis_x0, vis_y0 = max(0, -x0), max(0, -y0)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x0 >= x1 or y0 >= y1:
            return

        text_slice = text_rgb[vis_y0:vis_y0+(y1-y0), vis_x0:vis_x0+(x1-x0)]
        alpha_slice = alpha[vis_y0:vis_y0+(y1-y0), vis_x0:vis_x0+(x1-x0)]
        region = buf[y0:y1, x0:x1].to(torch.float32)
        buf[y0:y1, x0:x1] = (text_slice*alpha_slice + region*(1-alpha_slice)).to(torch.uint8)


class Table(GraphicObject):
    def __init__(self, data, x, y, font_path=None, font_size=16,
                 cell_width=100, cell_height=40, grid_color=(200,200,200),
                 bg_color=None, text_color=(255,255,255), expand_cells=False,
                 visible=True, opacity=1.0, z_index=0):
        super().__init__(x, y, visible, opacity, z_index)
        self.data = data
        self.font_size = int(font_size)
        self.cell_width = int(cell_width)
        self.cell_height = int(cell_height)
        self.grid_color = tuple(map(int, grid_color))
        self.bg_color = tuple(map(int, bg_color)) if bg_color is not None else None
        self.text_color = tuple(map(int, text_color))
        self.expand_cells = bool(expand_cells)

        if font_path is None:
            self.font_path = files(pymatgraph) / "fonts" / "ComicMono.ttf"
        else:
            self.font_path = font_path
        try:
            self._font = ImageFont.truetype(str(self.font_path), self.font_size)
        except Exception:
            self._font = ImageFont.load_default()

        self.column_widths = [self.cell_width]*max(len(r) for r in self.data) if self.data else [self.cell_width]

    def render_to_buffer(self, buffer):
        if not self.visible or not self.data:
            return

        H, W = buffer.shape[:2] if isinstance(buffer, torch.Tensor) else buffer.get_dimensions()
        buf = buffer if isinstance(buffer, torch.Tensor) else buffer.inplace_matrix()

        rows = len(self.data)
        cols = max(len(r) for r in self.data)
        ch = self.cell_height

        if self.expand_cells:
            for c in range(cols):
                max_text_w = 0
                for r in range(rows):
                    if c < len(self.data[r]):
                        bbox = self._font.getbbox(str(self.data[r][c]))
                        max_text_w = max(max_text_w, bbox[2]-bbox[0])
                self.column_widths[c] = max(self.column_widths[c], max_text_w + 8)

        total_width = sum(self.column_widths)
        x_start, y_start = int(self.x), int(self.y)
        x_end, y_end = x_start + total_width, y_start + rows*ch

        vis_x0, vis_y0 = max(0, -x_start), max(0, -y_start)
        x0, y0 = max(0, x_start), max(0, y_start)
        x1, y1 = min(W, x_end), min(H, y_end)
        if x0 >= x1 or y0 >= y1:
            return

        # Background
        if self.bg_color is not None:
            buf[y0:y1, x0:x1] = torch.tensor(self.bg_color, dtype=torch.uint8)

        # Grid lines
        for r in range(rows+1):
            y = y_start + r*ch
            if y0 <= y < y1:
                buf[y:y+1, x0:x1] = torch.tensor(self.grid_color, dtype=torch.uint8)
        cur_x = x_start
        for w in self.column_widths:
            if x0 <= cur_x < x1:
                buf[y0:y1, cur_x:cur_x+1] = torch.tensor(self.grid_color, dtype=torch.uint8)
            cur_x += w

        table_w, table_h = x_end - x_start, y_end - y_start
        text_layer = Image.new("RGBA", (table_w, table_h), (0,0,0,0))
        draw = ImageDraw.Draw(text_layer)
        for r, row in enumerate(self.data):
            cur_x_off = 0
            for c, cell in enumerate(row):
                if cell is None:
                    cur_x_off += self.column_widths[c]
                    continue
                s = str(cell)
                bbox = self._font.getbbox(s)
                text_w = max(1, bbox[2]-bbox[0])
                cell_x = cur_x_off
                if r > 0 and c > 0:
                    cell_x += max(0, self.column_widths[c]-text_w-4)
                draw.text((cell_x, r*ch), s, font=self._font, fill=(*self.text_color, 255))
                cur_x_off += self.column_widths[c]

        arr = torch.ByteTensor(torch.ByteStorage.from_buffer(text_layer.tobytes())).reshape(table_h, table_w, 4)
        text_rgb = arr[..., :3].to(torch.float32)
        alpha = (arr[..., 3:4].to(torch.float32)/255.0)*self.opacity

        text_slice = text_rgb[vis_y0:vis_y0+(y1-y0), vis_x0:vis_x0+(x1-x0)]
        alpha_slice = alpha[vis_y0:vis_y0+(y1-y0), vis_x0:vis_x0+(x1-x0)]
        region = buf[y0:y1, x0:x1].to(torch.float32)
        buf[y0:y1, x0:x1] = (text_slice*alpha_slice + region*(1-alpha_slice)).to(torch.uint8)


class ImageObject(GraphicObject):
    def __init__(self, image, x=0, y=0, width=None, height=None,
                 visible=True, opacity=1.0, z_index=0):
        super().__init__(x, y, visible, opacity, z_index)

        # Accept either PIL.Image.Image or a file path
        if isinstance(image, Image.Image):
            img = image.convert("RGBA")
        else:
            img = Image.open(image).convert("RGBA")

        if width and height:
            img = img.resize((int(width), int(height)), Image.LANCZOS)
        self.img = img


    def render_to_buffer(self, buffer):
        if not self.visible:
            return

        H, W = buffer.shape[:2] if isinstance(buffer, torch.Tensor) else buffer.get_dimensions()
        buf = buffer if isinstance(buffer, torch.Tensor) else buffer.inplace_matrix()

        arr = torch.ByteTensor(torch.ByteStorage.from_buffer(self.img.tobytes())).reshape(
            self.img.height, self.img.width, 4
        )
        img_rgb = arr[..., :3].to(torch.float32)
        alpha = (arr[..., 3:4].to(torch.float32)/255.0)*self.opacity

        x0, y0 = int(self.x), int(self.y)
        x1, y1 = x0 + self.img.width, y0 + self.img.height
        vis_x0, vis_y0 = max(0, -x0), max(0, -y0)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x0 >= x1 or y0 >= y1:
            return

        img_slice = img_rgb[vis_y0:vis_y0+(y1-y0), vis_x0:vis_x0+(x1-x0)]
        alpha_slice = alpha[vis_y0:vis_y0+(y1-y0), vis_x0:vis_x0+(x1-x0)]
        region = buf[y0:y1, x0:x1].to(torch.float32)
        buf[y0:y1, x0:x1] = (img_slice*alpha_slice + region*(1-alpha_slice)).to(torch.uint8)


class Graphics:
    def __init__(self, width=800, height=600, bg_color=(0,0,0), backend="pygame"):
        self.width = int(width)
        self.height = int(height)
        self.bg_color = tuple(map(int, bg_color))
        self.objects = []

        if backend=="pygame":
            from pymatgraph.backends.pygame_backend import PygameRenderer
            self.renderer = PygameRenderer(self.width, self.height, self.bg_color)
        elif backend=="kivy":
            from pymatgraph.backends.kivy_backend import KivyRenderer
            self.renderer = KivyRenderer(self.width, self.height, self.bg_color)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def add(self, obj: GraphicObject):
        self.objects.append(obj)
        self.objects.sort(key=lambda o: o.z_index)

    def _make_staging(self):
        staging = LocalTensorBuffer(self.height, self.width, mode="rgb")
        staging.inplace_matrix()[:, :] = torch.tensor(self.bg_color, dtype=torch.uint8)
        return staging

    def run(self, shared_buffer: MultiprocessSafeTensorBuffer):
        staging = self._make_staging()
        for obj in self.objects:
            obj.render_to_buffer(staging)
        shared_buffer.write_matrix(staging.inplace_matrix())
        self.renderer.run(shared_buffer)
