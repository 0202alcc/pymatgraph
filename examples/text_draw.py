import pygame
import numpy as np
import torch
from matrixbuffer.MatrixBuffer import MultiprocessSafeTensorBuffer, Render
from matrixbuffer.Graphics import Graphics

# Initialize Pygame
pygame.init()

# Create a window
window_width, window_height = 800, 600
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Graphics Text Test")

# Create an RGB buffer (smaller than window for performance)
buffer_width, buffer_height = 400, 300
buffer = MultiprocessSafeTensorBuffer(n=buffer_height, m=buffer_width, mode="rgb")

# Create renderer
renderer = Render(buffer, screen)

# Create Graphics instance
graphics = Graphics(buffer, font_size=24)

# Fill background with a dark blue color
background = torch.zeros((buffer_height, buffer_width, 3), dtype=torch.uint8)
background[:, :] = torch.tensor([20, 30, 60], dtype=torch.uint8)
buffer.write_matrix(background)

# Draw some test text
graphics.draw_text("Hello, MatrixBuffer!", 50, 50, color=(255, 255, 255))
graphics.draw_text("Graphics Test", 100, 100, color=(255, 200, 100))
graphics.draw_text("Alpha Blending Works!", 80, 150, color=(100, 255, 100))

# Test text clipping at edges
graphics.draw_text("This text goes off the right edge...", 250, 200, color=(255, 100, 255))
graphics.draw_text("Bottom edge text", 50, 280, color=(100, 200, 255))

# Main loop
clock = pygame.time.Clock()
running = True
frame_count = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Animate some text
    animated_x = 150 + int(50 * np.sin(frame_count * 0.05))
    animated_y = 220
    
    # Clear the animated text area
    buffer_data = buffer.read_matrix()
    buffer_data[animated_y-20:animated_y+40, max(0, animated_x-100):min(buffer_width, animated_x+200)] = torch.tensor([20, 30, 60], dtype=torch.uint8)
    buffer.write_matrix(buffer_data)
    
    # Draw animated text
    graphics.draw_text("Moving Text!", animated_x, animated_y, color=(255, 255, 0))
    
    # Render to screen
    renderer.render()
    
    clock.tick(60)  # 60 FPS
    frame_count += 1

pygame.quit()