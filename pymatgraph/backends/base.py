# matrixbuffer/backends/base.py

from abc import ABC, abstractmethod

class Renderer(ABC):
    """Abstract rendering backend interface."""

    def __init__(self, width, height, bg_color=(0, 0, 0)):
        self.width = width
        self.height = height
        self.bg_color = bg_color

    @abstractmethod
    def draw_buffer(self, tensor_data):
        """Draw the buffer (RGB tensor) to the screen."""
        pass

    @abstractmethod
    def handle_events(self):
        """Handle window events, return False if the app should exit."""
        pass

    @abstractmethod
    def flip(self):
        """Refresh the display."""
        pass

    @abstractmethod
    def quit(self):
        """Clean up backend resources."""
        pass
