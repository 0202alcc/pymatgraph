# Graphics.py

import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


class Graphics:
    def __init__(self, matrix_buffer, font_path=None, font_size=16):
        self.buffer = matrix_buffer

        # Resolve default font path: <package_root>/fonts/ComicMono.ttf
        if font_path is None:
            package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            font_path = os.path.join(package_root, "fonts", "ComicMono.ttf")

        # Load font
        try:
            self.font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"Failed to load font at {font_path}: {e}. Falling back to default font.")
            self.font = ImageFont.load_default()

        # Cache buffer dimensions (n rows = height, m cols = width)
        self.buf_h, self.buf_w = self.buffer.get_dimensions()

        # Ensure we are using an RGB buffer
        if getattr(self.buffer, "get_mode", lambda: None)() != "rgb":
            raise ValueError("Graphics requires an 'rgb' mode MultiprocessSafeTensorBuffer.")

    def draw_text(self, text, start_x, start_y, color=(255, 255, 255)):
        """Render a single text string at (start_x, start_y) onto buffer."""
        if not text:
            return

        # Measure & rasterize text with PIL
        bbox = self.font.getbbox(text)
        text_w = max(1, bbox[2] - bbox[0])
        text_h = max(1, bbox[3] - bbox[1])

        img = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((-bbox[0], -bbox[1]), text, font=self.font,
                  fill=(color[0], color[1], color[2], 255))

        arr = np.array(img, dtype=np.uint8)  # (H, W, 4)
        text_rgb = torch.from_numpy(arr[..., :3].copy()).to(torch.float32)  # (H, W, 3)
        alpha = torch.from_numpy(arr[..., 3:4].copy()).to(torch.float32) / 255.0  # (H, W, 1)

        # Safe clipping
        x0, y0 = max(0, start_x), max(0, start_y)
        x1, y1 = min(self.buf_w, start_x + text_w), min(self.buf_h, start_y + text_h)
        if x0 >= x1 or y0 >= y1:
            return

        tx0, ty0 = x0 - start_x, y0 - start_y
        tx1, ty1 = tx0 + (x1 - x0), ty0 + (y1 - y0)

        text_rgb = text_rgb[ty0:ty1, tx0:tx1]
        alpha = alpha[ty0:ty1, tx0:tx1]

        # Blend
        buf = self.buffer.read_matrix()  # (H, W, 3), uint8
        region = buf[y0:y1, x0:x1].to(torch.float32)
        blended = text_rgb * alpha + region * (1.0 - alpha)
        buf[y0:y1, x0:x1] = blended.to(torch.uint8)
        self.buffer.write_matrix(buf)

    def draw_table_batched_full(
        self, data, start_x=0, start_y=0,
        grid_color=(200, 200, 200), bg_color=None
    ):
        """
        Fully vectorized table renderer: grid, cell background, and text.
        Uses PyTorch for blending. PIL/NumPy only for text rasterization.
        """
        buf = self.buffer.read_matrix()
        H, W, _ = buf.shape
        rows = len(data)
        cols = max(len(r) for r in data)

        cell_h = max(H // rows, 1)
        cell_w = max(W // cols, 1)

        # --- 1. Fill table background ---
        if bg_color is not None:
            y1 = min(start_y + rows * cell_h, H)
            x1 = min(start_x + cols * cell_w, W)
            buf[start_y:y1, start_x:x1] = torch.tensor(bg_color, dtype=torch.uint8)

        # --- 2. Draw grid lines ---
        row_pattern = torch.zeros(H, dtype=torch.bool)
        row_pattern[start_y : start_y + rows * cell_h + 1 : cell_h] = True
        col_pattern = torch.zeros(W, dtype=torch.bool)
        col_pattern[start_x : start_x + cols * cell_w + 1 : cell_w] = True
        grid_mask = row_pattern[:, None] | col_pattern[None, :]
        buf[grid_mask] = torch.tensor(grid_color, dtype=torch.uint8)

        # --- 3. Text overlay ---
        overlay = torch.zeros_like(buf, dtype=torch.float32)
        alpha_overlay = torch.zeros(H, W, 1, dtype=torch.float32)

        text_imgs, positions = [], []
        for r, row in enumerate(data):
            for c, text in enumerate(row):
                if not text:
                    continue
                bbox = self.font.getbbox(text)
                text_w, text_h = max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])

                img = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.text((-bbox[0], -bbox[1]), text, font=self.font, fill=(255, 255, 255, 255))
                arr = np.array(img, dtype=np.uint8)

                arr_t = torch.from_numpy(arr.copy())
                text_imgs.append(arr_t)

                cell_top, cell_left = r * cell_h + start_y, c * cell_w + start_x
                text_x = cell_left + (cell_w - text_w) // 2
                text_y = cell_top + (cell_h - text_h) // 2 - bbox[1]
                positions.append((text_y, text_x))

        for arr, (y0, x0) in zip(text_imgs, positions):
            h_clip, w_clip = min(arr.shape[0], H - y0), min(arr.shape[1], W - x0)
            if h_clip <= 0 or w_clip <= 0:
                continue
            arr_rgb = arr[:h_clip, :w_clip, :3].to(torch.float32)
            arr_alpha = arr[:h_clip, :w_clip, 3:4].to(torch.float32) / 255.0
            overlay[y0:y0+h_clip, x0:x0+w_clip] = arr_rgb
            alpha_overlay[y0:y0+h_clip, x0:x0+w_clip] = arr_alpha

        buf_float = buf.to(torch.float32)
        buf_float = overlay * alpha_overlay + buf_float * (1.0 - alpha_overlay)
        self.buffer.write_matrix(buf_float.to(torch.uint8))


if __name__ == "__main__":
    import pygame
    from matrixbuffer.MatrixBuffer import MultiprocessSafeTensorBuffer
    import torch

    pygame.init()
    window_width, window_height = 800, 600
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Fully Vectorized Table Render")

    buffer_width, buffer_height = 400, 300
    buffer = MultiprocessSafeTensorBuffer(n=buffer_height, m=buffer_width, mode="rgb")

    background = torch.zeros((buffer_height, buffer_width, 3), dtype=torch.uint8)
    background[:, :] = torch.tensor([20, 30, 60], dtype=torch.uint8)
    buffer.write_matrix(background)

    graphics = Graphics(buffer, font_size=16)
    data = [
        ["Name", "Age", "Score", "Country"],
        ["Alice", "23", "95", "USA"],
        ["Bob", "30", "88", "UK"],
        ["Carol", "27", "92", "Canada"],
        ["Dave", "35", "85", "Australia"]
    ]

    graphics.draw_table_batched_full(
        data, start_x=0, start_y=0,
        grid_color=(200, 200, 200),
        bg_color=(50, 50, 100)
    )

    running, clock = True, pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        tensor_data = buffer.read_matrix()
        np_data = tensor_data.cpu().numpy()
        np_data_transposed = np.transpose(np_data, (1, 0, 2))  # (W,H,3)
        surface = pygame.surfarray.make_surface(np_data_transposed)
        surface_scaled = pygame.transform.scale(surface, (window_width, window_height))
        screen.blit(surface_scaled, (0, 0))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
