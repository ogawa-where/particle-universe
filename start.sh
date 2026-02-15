#!/bin/bash
# Particle Universe - GPU-Accelerated Artificial Life Simulator
# Start script

cd "$(dirname "$0")"

# Compile if needed
if [ ! -f libparticle_sim.so ] || [ particle_sim.cu -nt libparticle_sim.so ]; then
    echo "Compiling CUDA kernels..."
    make
fi

echo ""
echo "============================================"
echo "  Particle Universe"
echo "  GPU-Accelerated Artificial Life Simulator"
echo "============================================"
echo ""
echo "  Open http://localhost:5000 in your browser"
echo ""

python3 server.py
