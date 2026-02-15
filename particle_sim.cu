#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

// ============================================================
// Particle Universe - GPU-Accelerated Artificial Life Simulator
// ============================================================

#define MAX_TYPES 8
#define GRID_SIZE 128

struct SimParams {
    int num_particles;
    int num_types;
    int width;
    int height;
    float dt;
    float friction;
    float force_scale;
    float cutoff_radius;
    float repulsion_radius;
    float repulsion_strength;
    // interaction matrix: attraction[i * MAX_TYPES + j]
    float attraction[MAX_TYPES * MAX_TYPES];
    // type colors (RGB)
    float colors[MAX_TYPES * 3];
};

// Device-side particles
static float *d_px = nullptr, *d_py = nullptr;
static float *d_vx = nullptr, *d_vy = nullptr;
static int   *d_type = nullptr;

// Grid for spatial partitioning
static int *d_grid_counts = nullptr;    // GRID_SIZE * GRID_SIZE
static int *d_grid_offsets = nullptr;   // GRID_SIZE * GRID_SIZE
static int *d_grid_particles = nullptr; // num_particles
static int *d_sorted_indices = nullptr; // num_particles

// Render buffer
static uint8_t *d_framebuffer = nullptr;
static uint8_t *h_framebuffer = nullptr;

// Params on device
static SimParams *d_params = nullptr;
static SimParams h_params;

// RNG states
static curandState *d_rng = nullptr;

static bool initialized = false;
static int max_particles = 0;

// ============================================================
// CUDA Kernels
// ============================================================

__global__ void init_rng_kernel(curandState *states, int n, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        curand_init(seed, i, 0, &states[i]);
    }
}

__global__ void init_particles_kernel(float *px, float *py, float *vx, float *vy,
                                       int *type, int n, int num_types,
                                       int width, int height, curandState *rng) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        curandState local_rng = rng[i];
        px[i] = curand_uniform(&local_rng) * width;
        py[i] = curand_uniform(&local_rng) * height;
        vx[i] = 0.0f;
        vy[i] = 0.0f;
        type[i] = (int)(curand_uniform(&local_rng) * num_types) % num_types;
        rng[i] = local_rng;
    }
}

__global__ void clear_grid_kernel(int *grid_counts, int total_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_cells) {
        grid_counts[i] = 0;
    }
}

__global__ void count_grid_kernel(float *px, float *py, int *grid_counts,
                                   int n, float cell_w, float cell_h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int gx = min(max((int)(px[i] / cell_w), 0), GRID_SIZE - 1);
        int gy = min(max((int)(py[i] / cell_h), 0), GRID_SIZE - 1);
        atomicAdd(&grid_counts[gy * GRID_SIZE + gx], 1);
    }
}

// Simple prefix sum for grid offsets (run on single thread - grid is small)
__global__ void prefix_sum_kernel(int *counts, int *offsets, int total_cells) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < total_cells; i++) {
            offsets[i] = sum;
            sum += counts[i];
        }
    }
}

__global__ void assign_grid_kernel(float *px, float *py, int *grid_counts,
                                    int *grid_offsets, int *grid_particles,
                                    int n, float cell_w, float cell_h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int gx = min(max((int)(px[i] / cell_w), 0), GRID_SIZE - 1);
        int gy = min(max((int)(py[i] / cell_h), 0), GRID_SIZE - 1);
        int cell = gy * GRID_SIZE + gx;
        int idx = atomicAdd(&grid_counts[cell], 1);
        grid_particles[grid_offsets[cell] + idx] = i;
    }
}

__global__ void compute_forces_kernel(float *px, float *py, float *vx, float *vy,
                                       int *type, int *grid_counts, int *grid_offsets,
                                       int *grid_particles, SimParams *params,
                                       float cell_w, float cell_h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = px[i], yi = py[i];
    int ti = type[i];
    float fx = 0.0f, fy = 0.0f;
    float cutoff = params->cutoff_radius;
    float cutoff2 = cutoff * cutoff;
    float rep_r = params->repulsion_radius;
    float rep_r2 = rep_r * rep_r;
    float rep_s = params->repulsion_strength;
    float force_scale = params->force_scale;
    int w = params->width, h = params->height;

    int gx = min(max((int)(xi / cell_w), 0), GRID_SIZE - 1);
    int gy = min(max((int)(yi / cell_h), 0), GRID_SIZE - 1);

    // Search radius in grid cells
    int search_r = (int)ceilf(cutoff / fminf(cell_w, cell_h)) + 1;

    for (int dy = -search_r; dy <= search_r; dy++) {
        for (int dx = -search_r; dx <= search_r; dx++) {
            int nx = gx + dx;
            int ny = gy + dy;

            // Wrap around (toroidal)
            if (nx < 0) nx += GRID_SIZE;
            if (nx >= GRID_SIZE) nx -= GRID_SIZE;
            if (ny < 0) ny += GRID_SIZE;
            if (ny >= GRID_SIZE) ny -= GRID_SIZE;

            int cell = ny * GRID_SIZE + nx;
            int start = grid_offsets[cell];
            int count = grid_counts[cell];

            for (int k = 0; k < count; k++) {
                int j = grid_particles[start + k];
                if (j == i) continue;

                // Compute distance with wrapping
                float djx = px[j] - xi;
                float djy = py[j] - yi;

                // Toroidal wrapping
                if (djx > w * 0.5f) djx -= w;
                if (djx < -w * 0.5f) djx += w;
                if (djy > h * 0.5f) djy -= h;
                if (djy < -h * 0.5f) djy += h;

                float d2 = djx * djx + djy * djy;
                if (d2 > cutoff2 || d2 < 1e-6f) continue;

                float d = sqrtf(d2);
                float inv_d = 1.0f / d;
                float nx_dir = djx * inv_d;
                float ny_dir = djy * inv_d;

                int tj = type[j];
                float attr = params->attraction[ti * MAX_TYPES + tj];

                float force;
                if (d < rep_r) {
                    // Repulsion zone: linear ramp from repulsion to 0
                    force = rep_s * (d / rep_r - 1.0f);
                } else {
                    // Attraction/repulsion zone: smooth curve
                    float t = (d - rep_r) / (cutoff - rep_r);
                    // Bell curve peaking at midpoint
                    force = attr * (1.0f - fabsf(2.0f * t - 1.0f));
                }

                force *= force_scale;
                fx += nx_dir * force;
                fy += ny_dir * force;
            }
        }
    }

    // Update velocity
    vx[i] += fx * params->dt;
    vy[i] += fy * params->dt;

    // Apply friction
    vx[i] *= params->friction;
    vy[i] *= params->friction;
}

__global__ void update_positions_kernel(float *px, float *py, float *vx, float *vy,
                                         int n, int width, int height, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;

    // Toroidal wrapping
    if (px[i] < 0) px[i] += width;
    if (px[i] >= width) px[i] -= width;
    if (py[i] < 0) py[i] += height;
    if (py[i] >= height) py[i] -= height;
}

__global__ void clear_framebuffer_kernel(uint8_t *fb, int total_pixels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_pixels) {
        // Dark background with slight blue tint
        fb[i * 3 + 0] = 8;   // R
        fb[i * 3 + 1] = 8;   // G
        fb[i * 3 + 2] = 15;  // B
    }
}

__global__ void render_particles_kernel(uint8_t *fb, float *px, float *py, int *type,
                                         SimParams *params, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int x = (int)px[i];
    int y = (int)py[i];
    int w = params->width;
    int h = params->height;

    if (x < 0 || x >= w || y < 0 || y >= h) return;

    int ti = type[i];
    uint8_t r = (uint8_t)(params->colors[ti * 3 + 0] * 255.0f);
    uint8_t g = (uint8_t)(params->colors[ti * 3 + 1] * 255.0f);
    uint8_t b = (uint8_t)(params->colors[ti * 3 + 2] * 255.0f);

    // Draw a 2x2 pixel dot for visibility
    for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
            int px2 = x + dx;
            int py2 = y + dy;
            if (px2 >= 0 && px2 < w && py2 >= 0 && py2 < h) {
                int idx = (py2 * w + px2) * 3;
                fb[idx + 0] = r;
                fb[idx + 1] = g;
                fb[idx + 2] = b;
            }
        }
    }
}

// Render with glow/trail effect - fade existing buffer instead of clearing
__global__ void fade_framebuffer_kernel(uint8_t *fb, int total_pixels, int fade_amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_pixels) {
        int r = fb[i * 3 + 0];
        int g = fb[i * 3 + 1];
        int b = fb[i * 3 + 2];
        fb[i * 3 + 0] = (uint8_t)max(r - fade_amount, 5);
        fb[i * 3 + 1] = (uint8_t)max(g - fade_amount, 5);
        fb[i * 3 + 2] = (uint8_t)max(b - fade_amount, 8);
    }
}

// ============================================================
// Host-side API (exported as C functions for ctypes)
// ============================================================

extern "C" {

int sim_init(int num_particles, int num_types, int width, int height) {
    if (initialized) return 0;

    max_particles = num_particles;
    h_params.num_particles = num_particles;
    h_params.num_types = min(num_types, MAX_TYPES);
    h_params.width = width;
    h_params.height = height;
    h_params.dt = 1.0f;
    h_params.friction = 0.5f;
    h_params.force_scale = 1.0f;
    h_params.cutoff_radius = 80.0f;
    h_params.repulsion_radius = 20.0f;
    h_params.repulsion_strength = 5.0f;

    // Default colors (vibrant palette)
    float default_colors[] = {
        1.0f, 0.2f, 0.2f,   // Red
        0.2f, 0.8f, 0.2f,   // Green
        0.3f, 0.5f, 1.0f,   // Blue
        1.0f, 0.8f, 0.1f,   // Yellow
        0.9f, 0.3f, 0.9f,   // Magenta
        0.1f, 0.9f, 0.9f,   // Cyan
        1.0f, 0.5f, 0.1f,   // Orange
        0.7f, 0.7f, 0.7f,   // Gray
    };
    memcpy(h_params.colors, default_colors, sizeof(default_colors));

    // Zero out attraction matrix
    memset(h_params.attraction, 0, sizeof(h_params.attraction));

    // Allocate device memory
    cudaMalloc(&d_px, num_particles * sizeof(float));
    cudaMalloc(&d_py, num_particles * sizeof(float));
    cudaMalloc(&d_vx, num_particles * sizeof(float));
    cudaMalloc(&d_vy, num_particles * sizeof(float));
    cudaMalloc(&d_type, num_particles * sizeof(int));
    cudaMalloc(&d_rng, num_particles * sizeof(curandState));

    int total_cells = GRID_SIZE * GRID_SIZE;
    cudaMalloc(&d_grid_counts, total_cells * sizeof(int));
    cudaMalloc(&d_grid_offsets, total_cells * sizeof(int));
    cudaMalloc(&d_grid_particles, num_particles * sizeof(int));

    int total_pixels = width * height;
    cudaMalloc(&d_framebuffer, total_pixels * 3);
    h_framebuffer = (uint8_t *)malloc(total_pixels * 3);

    cudaMalloc(&d_params, sizeof(SimParams));

    // Init RNG
    int threads = 256;
    int blocks = (num_particles + threads - 1) / threads;
    init_rng_kernel<<<blocks, threads>>>(d_rng, num_particles, 42);
    cudaDeviceSynchronize();

    // Init particles
    init_particles_kernel<<<blocks, threads>>>(d_px, d_py, d_vx, d_vy, d_type,
                                                num_particles, num_types, width, height, d_rng);
    cudaDeviceSynchronize();

    // Clear framebuffer
    blocks = (total_pixels + threads - 1) / threads;
    clear_framebuffer_kernel<<<blocks, threads>>>(d_framebuffer, total_pixels);
    cudaDeviceSynchronize();

    initialized = true;
    printf("[CUDA] Initialized: %d particles, %d types, %dx%d\n",
           num_particles, num_types, width, height);
    return 0;
}

void sim_set_attraction(int type_a, int type_b, float value) {
    if (type_a < MAX_TYPES && type_b < MAX_TYPES) {
        h_params.attraction[type_a * MAX_TYPES + type_b] = value;
    }
}

float sim_get_attraction(int type_a, int type_b) {
    if (type_a < MAX_TYPES && type_b < MAX_TYPES) {
        return h_params.attraction[type_a * MAX_TYPES + type_b];
    }
    return 0.0f;
}

void sim_set_friction(float f) { h_params.friction = f; }
void sim_set_force_scale(float s) { h_params.force_scale = s; }
void sim_set_cutoff_radius(float r) { h_params.cutoff_radius = r; }
void sim_set_repulsion_radius(float r) { h_params.repulsion_radius = r; }
void sim_set_dt(float dt) { h_params.dt = dt; }
void sim_set_num_particles(int n) { h_params.num_particles = min(n, max_particles); }

void sim_step(int use_trails) {
    if (!initialized) return;

    int n = h_params.num_particles;
    int w = h_params.width;
    int h = h_params.height;
    int threads = 256;

    // Upload params
    cudaMemcpy(d_params, &h_params, sizeof(SimParams), cudaMemcpyHostToDevice);

    // Build spatial grid
    float cell_w = (float)w / GRID_SIZE;
    float cell_h = (float)h / GRID_SIZE;
    int total_cells = GRID_SIZE * GRID_SIZE;

    // Clear grid counts
    int grid_blocks = (total_cells + threads - 1) / threads;
    clear_grid_kernel<<<grid_blocks, threads>>>(d_grid_counts, total_cells);
    cudaDeviceSynchronize();

    // Count particles per cell
    int p_blocks = (n + threads - 1) / threads;
    count_grid_kernel<<<p_blocks, threads>>>(d_px, d_py, d_grid_counts, n, cell_w, cell_h);
    cudaDeviceSynchronize();

    // Prefix sum for offsets
    prefix_sum_kernel<<<1, 1>>>(d_grid_counts, d_grid_offsets, total_cells);
    cudaDeviceSynchronize();

    // Reset counts and assign particles to grid
    clear_grid_kernel<<<grid_blocks, threads>>>(d_grid_counts, total_cells);
    cudaDeviceSynchronize();
    assign_grid_kernel<<<p_blocks, threads>>>(d_px, d_py, d_grid_counts, d_grid_offsets,
                                               d_grid_particles, n, cell_w, cell_h);
    cudaDeviceSynchronize();

    // Compute forces and update velocities
    compute_forces_kernel<<<p_blocks, threads>>>(d_px, d_py, d_vx, d_vy, d_type,
                                                  d_grid_counts, d_grid_offsets,
                                                  d_grid_particles, d_params,
                                                  cell_w, cell_h, n);
    cudaDeviceSynchronize();

    // Update positions
    update_positions_kernel<<<p_blocks, threads>>>(d_px, d_py, d_vx, d_vy,
                                                    n, w, h, h_params.dt);
    cudaDeviceSynchronize();

    // Render
    int total_pixels = w * h;
    int fb_blocks = (total_pixels + threads - 1) / threads;

    if (use_trails) {
        fade_framebuffer_kernel<<<fb_blocks, threads>>>(d_framebuffer, total_pixels, 15);
    } else {
        clear_framebuffer_kernel<<<fb_blocks, threads>>>(d_framebuffer, total_pixels);
    }
    cudaDeviceSynchronize();

    render_particles_kernel<<<p_blocks, threads>>>(d_framebuffer, d_px, d_py, d_type,
                                                    d_params, n);
    cudaDeviceSynchronize();

    // Copy framebuffer to host
    cudaMemcpy(h_framebuffer, d_framebuffer, total_pixels * 3, cudaMemcpyDeviceToHost);
}

uint8_t* sim_get_framebuffer() {
    return h_framebuffer;
}

int sim_get_width() { return h_params.width; }
int sim_get_height() { return h_params.height; }

void sim_add_force_at(float x, float y, float fx_add, float fy_add, float radius) {
    // This would need a kernel - for simplicity we'll handle mouse interaction differently
    // by adding particles at mouse position
}

void sim_reinit_particles() {
    if (!initialized) return;
    int n = h_params.num_particles;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    init_particles_kernel<<<blocks, threads>>>(d_px, d_py, d_vx, d_vy, d_type,
                                                n, h_params.num_types,
                                                h_params.width, h_params.height, d_rng);
    cudaDeviceSynchronize();
}

void sim_cleanup() {
    if (!initialized) return;
    cudaFree(d_px); cudaFree(d_py);
    cudaFree(d_vx); cudaFree(d_vy);
    cudaFree(d_type); cudaFree(d_rng);
    cudaFree(d_grid_counts); cudaFree(d_grid_offsets); cudaFree(d_grid_particles);
    cudaFree(d_framebuffer); cudaFree(d_params);
    free(h_framebuffer);
    initialized = false;
    printf("[CUDA] Cleaned up.\n");
}

} // extern "C"
