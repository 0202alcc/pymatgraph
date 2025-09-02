# matrixbuffer/backends/pygame_backend.py
import pygame
import torch

class PygameRenderer:
    def __init__(self, width, height, bg_color=(0,0,0)):
        pygame.init()
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MatrixBuffer - Pygame")
        self.clock = pygame.time.Clock()
        self.objects = []

    def add(self, obj):
        self.objects.append(obj)

    def run(self, buffer):
        running = True
        while running:
            self.screen.fill(self.bg_color)
            tensor_data = buffer.read_matrix()
            surf = pygame.surfarray.make_surface(tensor_data.cpu().numpy().swapaxes(0,1))
            self.screen.blit(surf, (0,0))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()
