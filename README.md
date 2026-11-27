# Mandelbrot-RTX

A Mandelbrot fractal explorer capable of zooming up to \(10^{-14}\), remotely displayed in a web browser.

This project was developed as a proof of concept to execute complex mathematical equations (such as the Mandelbrot set) on an **NVIDIA RTX 5000 series GPU** using Docker. It has been tested on a **QNAP NAS** equipped with an **RTX 5060 Ti 16GB** GPU.

<img width="917" height="650" alt="screenshot" src="https://github.com/user-attachments/assets/13142cf3-08ec-4e79-bce6-858875729903" />


## Architecture

The project uses minimal code divided into three main components, orchestrated via `docker-compose.yml`:
* **`mandelbrot.cu`**: CUDA mathematical engine (compiled at container startup into `libmandelbrot.so`).
* **`server.py`**: Backend server interfacing with the shared library to generate the fractal.
* **`index.html`**: Web interface for interactive graphical display.

## Installation

1. Create a `rtx_gpu` folder in your Container directory:
   ```text
   /share/Container/container-station-data/application/rtx_gpu
   ```
   *(Path example for QNAP Container Station)*

2. Copy the `docker-compose.yml` file into this folder and create a `mandelbrot` subfolder.

3. Place the following three files inside the `mandelbrot` subfolder:
   * `server.py`
   * `mandelbrot.cu`
   * `index.html`

4. Build and launch the container using the following command:
   ```bash
   docker compose up -d
   ```
   > **Note:** The `-d` flag runs the container in detached mode (in the background). Omit it if you want to view the console output directly for debugging.
