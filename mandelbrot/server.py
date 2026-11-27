# ************************************************************
# Server in python to run the mandelbrot computation lib and
# support the web interface for PC display.
#
# Date : 2026-11-26
#
# ************************************************************


import ctypes
import io
import time
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn

# Initialize the FastAPI web server
app = FastAPI()

# Load the compiled CUDA shared library
lib = ctypes.CDLL("./libmandelbrot.so")

# Specify the exact C argument types expected by render_mandelbrot
lib.render_mandelbrot.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3, flags="C_CONTIGUOUS"),  # Output RGB pixel array
    ctypes.c_int, ctypes.c_int,                                             # Image width and height
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,    # Complex plane boundaries
    ctypes.c_int                                                            # Maximum iterations
]

# Serve the frontend HTML page on the root URL
@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())

# Real-time WebSocket endpoint handling viewport updates and binary image streaming
@app.websocket("/ws")

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_res = (0, 0)
    img_array = None

    while True:
        # Receive viewport state and desired resolution from the browser client
        data = await websocket.receive_json()
        cx, cy, zoom, max_iter = data["cx"], data["cy"], data["zoom"], data["iter"]
        width, height = int(data["width"]), int(data["height"])

        # Reallocate the NumPy host buffer only if the resolution changed
        if (width, height) != current_res:
            current_res = (width, height)
            img_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Compute complex plane bounds preserving the selected aspect ratio
        aspect = width / height
        span_y = 2.5 / zoom
        span_x = span_y * aspect

        # Measure GPU execution + Device-to-Host memory copy latency
        t0 = time.perf_counter()
        lib.render_mandelbrot(
            img_array, width, height,
            cx - span_x / 2, cx + span_x / 2,
            cy - span_y / 2, cy + span_y / 2,
            max_iter
        )
        gpu_ms = (time.perf_counter() - t0) * 1000

        # Send timing telemetry as JSON, followed by the compressed JPEG binary frame
        await websocket.send_json({"gpu_ms": round(gpu_ms, 2)})
        img = Image.fromarray(img_array)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        await websocket.send_bytes(buf.getvalue())

# Run the ASGI server listening on all network interfaces
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
