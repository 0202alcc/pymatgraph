# This script demonstrates how to create a tilted projection from a larger RGB matrix
# using the matrixbuffer package for multiprocess-safe rendering.
#
# Before running this script, ensure you have Pygame, PyTorch, and the matrixbuffer
# package installed:
# pip install pygame torch matrixbuffer

import pygame
import torch
import torch.nn.functional as F
import numpy as np
import math
import multiprocessing
import time

try:
    # Attempt to import the user's matrixbuffer package
    from pymatgraph.MatrixBuffer import MultiprocessSafeTensorBuffer, Render
except ImportError:
    print("Error: The 'matrixbuffer' package could not be found.")
    print("Please make sure you have installed it by running 'pip install .' in your project's root directory.")
    exit()

def tilted_projection(source_tensor, theta_degrees, target_width, target_height, center_offset):
    """
    Performs a tilted projection (rotation) of a portion of a source tensor.

    This function uses PyTorch's affine grid and grid_sample to perform the
    transformation efficiently on the GPU if available.

    Args:
        source_tensor (torch.Tensor): The original RGB matrix. Must be a 3D tensor
                                      of shape (height, width, channels).
        theta_degrees (float): The angle of rotation in degrees.
        target_width (int): The width of the projected view.
        target_height (int): The height of the projected view.
        center_offset (tuple): A tuple (x_offset, y_offset) to control the
                               center of the view within the source matrix.

    Returns:
        torch.Tensor: The new, rotated tensor of shape (target_height, target_width, 3).
    """
    # Convert degrees to radians for torch functions
    theta_rad = math.radians(theta_degrees)

    # Get the source tensor's height and width
    source_height, source_width, _ = source_tensor.shape
    
    # Calculate the rotation matrix. We also include a translation to pan the view.
    # The matrix is defined for a normalized grid (-1 to 1).
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)
    
    # Create the 2x3 affine transformation matrix for rotation and translation.
    # The first two columns handle rotation and scaling. The last column is for translation.
    # Translation values are normalized to [-1, 1].
    x_offset_norm = center_offset[0] / source_width
    y_offset_norm = center_offset[1] / source_height
    
    # The matrix is constructed to map the target view coordinates to source coordinates.
    # This involves a reverse rotation.
    rot_matrix = torch.tensor([
        [cos_theta, sin_theta, -x_offset_norm],
        [-sin_theta, cos_theta, -y_offset_norm]
    ], dtype=torch.float32)

    # Add a batch dimension to the rotation matrix
    rot_matrix = rot_matrix.unsqueeze(0)

    # Add a batch and channel dimension to the source tensor for grid_sample
    source_tensor = source_tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Create a grid of coordinates to sample from the source tensor
    grid = F.affine_grid(rot_matrix, size=(1, 3, target_height, target_width), align_corners=False)

    # Use grid_sample to perform the actual projection and resampling.
    # 'bilinear' mode uses bilinear interpolation for smooth results.
    output_tensor = F.grid_sample(source_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Remove the extra batch and channel dimensions and permute back
    # Also, convert back to uint8 (0-255)
    output_tensor = (output_tensor.squeeze(0).permute(1, 2, 0) * 255.0).to(torch.uint8)

    return output_tensor

def update_buffer_process(buffer, stop_event, source_matrix):
    """
    Worker process to continuously update the buffer with a tilted projection.
    
    This function simulates a separate process generating data.
    """
    angle = 0
    while not stop_event.is_set():
        # Define the view parameters
        view_width, view_height = 400, 300
        angle = (angle + 1) % 360  # Animate the angle
        # Animate the center of the view to create a panning effect
        x_offset = int(100 * math.sin(math.radians(angle)))
        y_offset = int(100 * math.cos(math.radians(angle * 1.5)))
        center_offset = (x_offset, y_offset)

        # Generate the tilted projection
        tilted_view = tilted_projection(source_matrix, float(angle), view_width, view_height, center_offset)

        # Update the multiprocess-safe buffer
        # FIX: The method name is write_matrix, not update
        buffer.write_matrix(tilted_view)
        
        # Add a small delay to control the update rate
        time.sleep(0.01)

if __name__ == '__main__':
    # Initialize Pygame and set up the display
    pygame.init()
    VIEW_WIDTH, VIEW_HEIGHT = 400, 300
    screen = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))
    pygame.display.set_caption("MatrixBuffer Tilted Projection Example")

    # Define the dimensions of the original source matrix
    SOURCE_WIDTH, SOURCE_HEIGHT = 800, 600

    # Create the initial source RGB matrix with a gradient pattern
    # This matrix will be what we project from
    source_x_vals = torch.arange(SOURCE_WIDTH).view(1, SOURCE_WIDTH, 1)
    source_y_vals = torch.arange(SOURCE_HEIGHT).view(SOURCE_HEIGHT, 1, 1)
    
    # Expand r_channel to match the height of the source matrix
    r_channel = (source_x_vals.expand(SOURCE_HEIGHT, -1, -1) / SOURCE_WIDTH) * 255
    
    # Expand g_channel to match the width of the source matrix
    g_channel = (source_y_vals.expand(-1, SOURCE_WIDTH, -1) / SOURCE_HEIGHT) * 255
    
    # b_channel is already the correct shape
    b_channel = torch.zeros(SOURCE_HEIGHT, SOURCE_WIDTH, 1) + 128
    
    source_matrix = torch.cat((r_channel, g_channel, b_channel), dim=2).to(torch.uint8)

    # Create a multiprocess-safe tensor buffer
    projected_buffer = MultiprocessSafeTensorBuffer(VIEW_HEIGHT, VIEW_WIDTH, mode="rgb")

    # Create a renderer for the projected buffer
    renderer = Render(projected_buffer, screen)

    # Start the worker process to update the buffer
    stop_event = multiprocessing.Event()
    worker_process = multiprocessing.Process(target=update_buffer_process, args=(projected_buffer, stop_event, source_matrix))
    worker_process.start()

    # Main loop for rendering
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # The renderer reads from the buffer, which is updated by the worker process
        renderer.render()

    # Clean up and terminate the worker process
    stop_event.set()
    worker_process.join()
    pygame.quit()
