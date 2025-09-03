# main.py
import torch
import time
import multiprocessing as mp
from pymatgraph.Graphics import Graphics, Text, Table, ImageObject
from pymatgraph.MatrixBuffer import MultiprocessSafeTensorBuffer


def animation_process(buffer, width, height):
    txt = Text("Hello, PyMatGraph!", x=50, y=50, font_size=24, color=(255, 255, 0))

    data = [
        ["Name", "Score", "Level"],
        ["Alice", 95, 5],
        ["Bob", 87, 4],
        ["Carol", 92, 5]
    ]
    tbl = Table(
        data, x=50, y=100, font_size=18, cell_width=120, cell_height=40,
        grid_color=(200, 200, 200), bg_color=(50, 50, 50),
        text_color=(255, 255, 255)
    )

    img = ImageObject("photo.png", x=400, y=50, width=200, height=150)

    vx = 3
    x_pos = 50

    while True:
        # --- Build next frame completely ---
        bg_color = torch.tensor([30, 30, 30], dtype=torch.uint8)
        next_frame = bg_color.repeat(height, width, 1)

        # Move objects
        x_pos += vx
        if x_pos + 200 > width or x_pos < 0:
            vx = -vx

        txt.x = x_pos
        tbl.x = x_pos

        # Render into frame
        txt.render_to_buffer(next_frame)
        tbl.render_to_buffer(next_frame)
        img.render_to_buffer(next_frame)

        # Swap buffer
        buffer.write_matrix(next_frame)

        time.sleep(1/30)




if __name__ == "__main__":
    mp.set_start_method("spawn")  # safer for macOS

    width, height = 800, 600
    buffer = MultiprocessSafeTensorBuffer(n=height, m=width, mode="rgb")


    # Fill background initially
    buf = buffer.read_matrix()
    buf[:, :] = torch.tensor([30, 30, 30], dtype=torch.uint8)
    buffer.write_matrix(buf)

    # Start animation process
    anim_proc = mp.Process(target=animation_process, args=(buffer, width, height))
    anim_proc.start()

    # Pygame renderer runs in the main process
    gfx = Graphics(width, height, bg_color=(30, 30, 30), backend="pygame")
    gfx.run(buffer)

    anim_proc.terminate()
    anim_proc.join()
