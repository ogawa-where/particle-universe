#!/usr/bin/env python3
"""
Particle Universe - GPU-Accelerated Artificial Life Simulator
Web server with WebSocket frame streaming
"""

import ctypes
import io
import json
import os
import random
import struct
import threading
import time

from flask import Flask, send_from_directory
from flask_sock import Sock
from PIL import Image

# ============================================================
# Load CUDA shared library
# ============================================================

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libparticle_sim.so")
lib = ctypes.CDLL(LIB_PATH)

# Define function signatures
lib.sim_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.sim_init.restype = ctypes.c_int

lib.sim_set_attraction.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
lib.sim_set_attraction.restype = None

lib.sim_get_attraction.argtypes = [ctypes.c_int, ctypes.c_int]
lib.sim_get_attraction.restype = ctypes.c_float

lib.sim_set_friction.argtypes = [ctypes.c_float]
lib.sim_set_friction.restype = None

lib.sim_set_force_scale.argtypes = [ctypes.c_float]
lib.sim_set_force_scale.restype = None

lib.sim_set_cutoff_radius.argtypes = [ctypes.c_float]
lib.sim_set_cutoff_radius.restype = None

lib.sim_set_repulsion_radius.argtypes = [ctypes.c_float]
lib.sim_set_repulsion_radius.restype = None

lib.sim_set_dt.argtypes = [ctypes.c_float]
lib.sim_set_dt.restype = None

lib.sim_set_num_particles.argtypes = [ctypes.c_int]
lib.sim_set_num_particles.restype = None

lib.sim_step.argtypes = [ctypes.c_int]
lib.sim_step.restype = None

lib.sim_get_framebuffer.argtypes = []
lib.sim_get_framebuffer.restype = ctypes.POINTER(ctypes.c_uint8)

lib.sim_get_width.argtypes = []
lib.sim_get_width.restype = ctypes.c_int

lib.sim_get_height.argtypes = []
lib.sim_get_height.restype = ctypes.c_int

lib.sim_reinit_particles.argtypes = []
lib.sim_reinit_particles.restype = None

lib.sim_cleanup.argtypes = []
lib.sim_cleanup.restype = None

# ============================================================
# Simulation Configuration
# ============================================================

WIDTH = 1280
HEIGHT = 720
NUM_PARTICLES = 30000
NUM_TYPES = 6
TARGET_FPS = 30

# ============================================================
# Flask App
# ============================================================

app = Flask(__name__, static_folder="static")
sock = Sock(app)

# Global state
sim_lock = threading.Lock()
use_trails = True
running = True


def randomize_rules():
    """Generate random attraction/repulsion rules."""
    for i in range(NUM_TYPES):
        for j in range(NUM_TYPES):
            val = random.uniform(-1.0, 1.0)
            lib.sim_set_attraction(i, j, val)


def get_rules():
    """Get current attraction matrix as list of lists."""
    rules = []
    for i in range(NUM_TYPES):
        row = []
        for j in range(NUM_TYPES):
            row.append(round(float(lib.sim_get_attraction(i, j)), 3))
        rules.append(row)
    return rules


def preset_symmetric():
    """Generate symmetric rules (more stable patterns)."""
    for i in range(NUM_TYPES):
        for j in range(i, NUM_TYPES):
            val = random.uniform(-1.0, 1.0)
            lib.sim_set_attraction(i, j, val)
            lib.sim_set_attraction(j, i, val)


def preset_chains():
    """Each type is attracted to the next, creating chains."""
    for i in range(NUM_TYPES):
        for j in range(NUM_TYPES):
            lib.sim_set_attraction(i, j, -0.5)
    for i in range(NUM_TYPES):
        lib.sim_set_attraction(i, (i + 1) % NUM_TYPES, 1.0)
        lib.sim_set_attraction(i, i, 0.2)


def preset_predator_prey():
    """Rock-paper-scissors style dynamics."""
    for i in range(NUM_TYPES):
        for j in range(NUM_TYPES):
            lib.sim_set_attraction(i, j, -0.3)
    for i in range(NUM_TYPES):
        lib.sim_set_attraction(i, (i + 1) % NUM_TYPES, 0.8)
        lib.sim_set_attraction(i, (i - 1) % NUM_TYPES, -0.8)
        lib.sim_set_attraction(i, i, 0.1)


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


@sock.route("/ws")
def websocket(ws):
    global use_trails

    print("[WS] Client connected")

    # Send initial state
    ws.send(json.dumps({
        "type": "init",
        "width": WIDTH,
        "height": HEIGHT,
        "num_types": NUM_TYPES,
        "rules": get_rules(),
    }))

    frame_time = 1.0 / TARGET_FPS
    last_frame = 0

    try:
        while running:
            # Check for incoming messages (non-blocking)
            try:
                msg = ws.receive(timeout=0.001)
                if msg:
                    data = json.loads(msg)
                    handle_message(data)
            except TimeoutError:
                pass
            except Exception:
                pass

            # Rate limit
            now = time.time()
            if now - last_frame < frame_time:
                time.sleep(0.001)
                continue
            last_frame = now

            # Simulate and render
            with sim_lock:
                lib.sim_step(1 if use_trails else 0)

                w = lib.sim_get_width()
                h = lib.sim_get_height()
                fb_ptr = lib.sim_get_framebuffer()
                fb_size = w * h * 3
                fb_data = ctypes.string_at(fb_ptr, fb_size)

            # Encode as JPEG
            img = Image.frombuffer("RGB", (w, h), fb_data, "raw", "RGB", 0, 1)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            jpeg_data = buf.getvalue()

            # Send binary frame
            ws.send(jpeg_data)

    except Exception as e:
        print(f"[WS] Disconnected: {e}")

    print("[WS] Client disconnected")


def handle_message(data):
    global use_trails

    msg_type = data.get("type", "")

    if msg_type == "set_param":
        name = data.get("name")
        value = data.get("value")
        with sim_lock:
            if name == "friction":
                lib.sim_set_friction(float(value))
            elif name == "force_scale":
                lib.sim_set_force_scale(float(value))
            elif name == "cutoff_radius":
                lib.sim_set_cutoff_radius(float(value))
            elif name == "repulsion_radius":
                lib.sim_set_repulsion_radius(float(value))
            elif name == "dt":
                lib.sim_set_dt(float(value))
            elif name == "num_particles":
                lib.sim_set_num_particles(int(value))
            elif name == "trails":
                use_trails = bool(value)
        print(f"[Param] {name} = {value}")

    elif msg_type == "randomize":
        with sim_lock:
            randomize_rules()
        print("[Rules] Randomized")

    elif msg_type == "preset":
        preset_name = data.get("name", "")
        with sim_lock:
            if preset_name == "symmetric":
                preset_symmetric()
            elif preset_name == "chains":
                preset_chains()
            elif preset_name == "predator_prey":
                preset_predator_prey()
            else:
                randomize_rules()
        print(f"[Rules] Preset: {preset_name}")

    elif msg_type == "set_rule":
        i = data.get("i", 0)
        j = data.get("j", 0)
        v = data.get("value", 0.0)
        with sim_lock:
            lib.sim_set_attraction(int(i), int(j), float(v))

    elif msg_type == "reset":
        with sim_lock:
            lib.sim_reinit_particles()
        print("[Sim] Particles reset")

    elif msg_type == "get_rules":
        # Rules will be sent with next frame
        pass


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Particle Universe - GPU-Accelerated Life Simulator")
    print("=" * 60)

    # Initialize simulation
    print(f"Initializing: {NUM_PARTICLES} particles, {NUM_TYPES} types, {WIDTH}x{HEIGHT}")
    lib.sim_init(NUM_PARTICLES, NUM_TYPES, WIDTH, HEIGHT)

    # Set initial random rules
    randomize_rules()

    # Set initial parameters
    lib.sim_set_friction(0.5)
    lib.sim_set_force_scale(1.0)
    lib.sim_set_cutoff_radius(80.0)
    lib.sim_set_repulsion_radius(20.0)
    lib.sim_set_dt(1.0)

    print(f"Starting server on http://localhost:5000")
    print("Open your browser to see the simulation!")

    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        lib.sim_cleanup()
        print("Shutdown complete.")
