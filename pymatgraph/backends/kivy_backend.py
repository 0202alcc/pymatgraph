# matrixbuffer/backends/kivy_backend.py
from kivy.app import App
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import torch

class KivyRenderer(App):
    def __init__(self, width, height, bg_color=(0,0,0), **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.objects = []
        self.buffer = None

    def add(self, obj):
        self.objects.append(obj)

    def run(self, buffer):
        self.buffer = buffer
        super().run()

    def build(self):
        self.img = Image(size=(self.width, self.height))
        Clock.schedule_interval(self.update, 1/30)
        return self.img

    def update(self, dt):
        if self.buffer is None: return
        tensor_data = self.buffer.read_matrix()
        H, W, _ = tensor_data.shape
        arr = tensor_data.cpu().numpy().astype('uint8')
        texture = Texture.create(size=(W,H))
        texture.blit_buffer(arr.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()  # flips the image to match Kivy coordinates

        self.img.texture = texture
