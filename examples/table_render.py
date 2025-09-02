import pygame
from pymatgraph.MatrixBuffer import MultiprocessSafeTensorBuffer
from pymatgraph.Graphics import Graphics
import torch
import numpy as np

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