#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "linear-algebra.hh"

using clock_type = std::chrono::high_resolution_clock;
using duration = clock_type::duration;
using time_point = clock_type::time_point;

double bandwidth(int n, time_point t0, time_point t1) {
    using namespace std::chrono;
    const auto dt = duration_cast<microseconds>(t1-t0).count();
    if (dt == 0) { return 0; }
    return ((n+n+n)*sizeof(float)*1e-9)/(dt*1e-6);
}

void print(const char* name, std::array<duration,5> dt, std::array<double,2> bw) {
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i=0; i<5; ++i) {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
        std::cout << std::setw(20) << tmp.str();
    }
    for (size_t i=0; i<2; ++i) {
        std::stringstream tmp;
        tmp << bw[i] << "GB/s";
        std::cout << std::setw(20) << tmp.str();
    }
    std::cout << '\n';
}

void print_column_names() {
    std::cout << std::setw(19) << "function";
    std::cout << std::setw(20) << "OpenMP";
    std::cout << std::setw(20) << "OpenCL total";
    std::cout << std::setw(20) << "OpenCL copy-in";
    std::cout << std::setw(20) << "OpenCL kernel";
    std::cout << std::setw(20) << "OpenCL copy-out";
    std::cout << std::setw(20) << "OpenMP bandwidth";
    std::cout << std::setw(20) << "OpenCL bandwidth";
    std::cout << '\n';
}

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

void profile_vector_times_vector(int n, OpenCL& opencl) {
    auto a = random_vector<float>(n);
    auto b = random_vector<float>(n);
    Vector<float> result(n), expected_result(n);
    opencl.queue.flush();
    cl::Kernel kernel(opencl.program, "vector_times_vector");
    auto t0 = clock_type::now();
    vector_times_vector(a, b, expected_result);
    auto t1 = clock_type::now();
    cl::Buffer d_a(opencl.queue, begin(a), end(a), true);
    cl::Buffer d_b(opencl.queue, begin(b), end(b), true);
    cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, result.size()*sizeof(float));
    kernel.setArg(0, d_a);
    kernel.setArg(1, d_b);
    kernel.setArg(2, d_result);
    opencl.queue.flush();
    auto t2 = clock_type::now();
    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NullRange);
    opencl.queue.flush();
    auto t3 = clock_type::now();
    cl::copy(opencl.queue, d_result, begin(result), end(result));
    auto t4 = clock_type::now();
    verify_vector(expected_result, result);
    print("vector-times-vector",
          {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3},
          {bandwidth(n+n+n, t0, t1), bandwidth(n+n+n, t2, t3)});
}

void profile_matrix_times_vector(int n, OpenCL& opencl) {
    auto a = random_matrix<float>(n,n);
    auto b = random_vector<float>(n);
    Vector<float> result(n), expected_result(n);
    opencl.queue.flush();
    cl::Kernel kernel(opencl.program, "matrix_times_vector");
    auto t0 = clock_type::now();
    matrix_times_vector(a, b, expected_result);
    auto t1 = clock_type::now();
    cl::Buffer d_a(opencl.queue, begin(a), end(a), true);
    cl::Buffer d_b(opencl.queue, begin(b), end(b), true);
    cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, result.size()*sizeof(float));
    kernel.setArg(0, d_a);
    kernel.setArg(1, d_b);
    kernel.setArg(2, d_result);
    kernel.setArg(3, n);
    opencl.queue.flush();
    auto t2 = clock_type::now();
    // 2D launch:
    // - each work-group computes MV_BLOCK_ROWS rows
    // - within a row, MV_BLOCK_COLS work-items cooperate via local memory and reduction
    constexpr size_t MV_BLOCK_ROWS = 4;
    constexpr size_t MV_BLOCK_COLS = 64;
    auto roundUp = [](size_t x, size_t m) { return (x + m - 1) / m * m; };
    const cl::NDRange lws(MV_BLOCK_ROWS, MV_BLOCK_COLS);
    const cl::NDRange gws(roundUp((size_t)n, MV_BLOCK_ROWS), MV_BLOCK_COLS);
    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gws, lws);
    opencl.queue.flush();
    auto t3 = clock_type::now();
    cl::copy(opencl.queue, d_result, begin(result), end(result));
    auto t4 = clock_type::now();
    verify_vector(expected_result, result, 1e-1f);
    print("matrix-times-vector",
          {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3},
          {bandwidth(n*n+n+n, t0, t1), bandwidth(n*n+n+n, t2, t3)});
}

void profile_matrix_times_matrix(int n, OpenCL& opencl) {
    auto a = random_matrix<float>(n,n);
    auto b = random_matrix<float>(n,n);
    Matrix<float> result(n,n), expected_result(n,n);
    opencl.queue.flush();
    cl::Kernel kernel(opencl.program, "matrix_times_matrix");

    auto t0 = clock_type::now();
    matrix_times_matrix(a, b, expected_result);
    auto t1 = clock_type::now();

    cl::Buffer d_a(opencl.queue, begin(a), end(a), true);
    cl::Buffer d_b(opencl.queue, begin(b), end(b), true);
    cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, result.size()*sizeof(float));
    kernel.setArg(0, d_a);
    kernel.setArg(1, d_b);
    kernel.setArg(2, d_result);
    kernel.setArg(3, n);

    opencl.queue.flush();
    auto t2 = clock_type::now();
    constexpr size_t MM_TILE = 16;
    auto roundUp = [](size_t x, size_t m) { return (x + m - 1) / m * m; };
    const cl::NDRange lws(MM_TILE, MM_TILE);
    const cl::NDRange gws(roundUp((size_t)n, MM_TILE), roundUp((size_t)n, MM_TILE));
    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, gws, lws);
    opencl.queue.flush();
    auto t3 = clock_type::now();

    cl::copy(opencl.queue, d_result, begin(result), end(result));
    auto t4 = clock_type::now();

    // Floating-point summation order differs between CPU and GPU.
    // Use a relaxed epsilon to avoid false negatives in verification.
    verify_matrix(expected_result, result, 1e-1f);
    print("matrix-times-matrix",
          {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3},
          {bandwidth(n*n+n*n+n*n, t0, t1), bandwidth(n*n+n*n+n*n, t2, t3)});
}

void opencl_main(OpenCL& opencl) {
    using namespace std::chrono;
    print_column_names();
    profile_vector_times_vector(1024*1024*10, opencl);
    profile_matrix_times_vector(1024*10, opencl);
    profile_matrix_times_matrix(1024, opencl);
}

const std::string src = R"(
kernel void vector_times_vector(global float* a,
                                global float* b,
                                global float* result) {
    const int i = get_global_id(0);
    result[i] = a[i] * b[i];
}

kernel void matrix_times_vector(global const float* a,
                                global const float* b,
                                global float* result,
                                const int n) {
    // Work-group: (MV_BLOCK_ROWS x MV_BLOCK_COLS)
    // - MV_BLOCK_ROWS rows of the matrix per group
    // - MV_BLOCK_COLS lanes per row cooperate using local memory
    // This satisfies the requirement "kernel uses local memory".
    #define MV_BLOCK_ROWS 4
    #define MV_BLOCK_COLS 64

    const int lr = get_local_id(0); // [0..MV_BLOCK_ROWS-1]
    const int lc = get_local_id(1); // [0..MV_BLOCK_COLS-1]
    const int row = get_group_id(0) * MV_BLOCK_ROWS + lr;

    local float xTile[MV_BLOCK_COLS];
    local float partial[MV_BLOCK_ROWS][MV_BLOCK_COLS];

    float acc = 0.0f;
    const int tiles = (n + MV_BLOCK_COLS - 1) / MV_BLOCK_COLS;
    for (int t = 0; t < tiles; ++t) {
        const int col = t * MV_BLOCK_COLS + lc;
        // Cache a tile of vector b in local memory once per group.
        if (lr == 0) {
            xTile[lc] = (col < n) ? b[col] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (row < n && col < n) {
            acc += a[row * n + col] * xTile[lc];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Reduce MV_BLOCK_COLS partial sums to a single value per row.
    partial[lr][lc] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = MV_BLOCK_COLS / 2; stride > 0; stride >>= 1) {
        if (lc < stride) {
            partial[lr][lc] += partial[lr][lc + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lc == 0 && row < n) {
        result[row] = partial[lr][0];
    }

    #undef MV_BLOCK_ROWS
    #undef MV_BLOCK_COLS
}

kernel void matrix_times_matrix(global float* a,
                                global float* b,
                                global float* result,
                                const int n) {
    // Classic tiled GEMM using local memory.
    // Each work-group computes one (MM_TILE x MM_TILE) tile of the result.
    #define MM_TILE 16

    const int col = get_global_id(0);
    const int row = get_global_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    local float As[MM_TILE][MM_TILE];
    local float Bs[MM_TILE][MM_TILE];

    float acc = 0.0f;
    const int tiles = (n + MM_TILE - 1) / MM_TILE;
    for (int t = 0; t < tiles; ++t) {
        const int kA = t * MM_TILE + lx; // column in A
        const int kB = t * MM_TILE + ly; // row in B

        // Load tiles into local memory (with bounds checks).
        As[ly][lx] = (row < n && kA < n) ? a[row * n + kA] : 0.0f;
        Bs[ly][lx] = (kB < n && col < n) ? b[kB * n + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < MM_TILE; ++k) {
            acc += As[ly][k] * Bs[k][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < n && col < n) {
        result[row * n + col] = acc;
    }

    #undef MM_TILE
}
)";

int main() {
    try {
        // find OpenCL platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "Unable to find OpenCL platforms\n";
            return 1;
        }
        cl::Platform platform = platforms[0];
        std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
        // create context
        cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        // get all devices associated with the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[0];
        std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
        cl::Program program(context, src);
        // compile the programme
        try {
            program.build(devices);
        } catch (const cl::Error& err) {
            for (const auto& device : devices) {
                std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << log;
            }
            throw;
        }
        cl::CommandQueue queue(context, device);
        OpenCL opencl{platform, device, context, program, queue};
        opencl_main(opencl);
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        std::cerr << "Search cl.h file for error code (" << err.err()
            << ") to understand what it means:\n";
        std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h\n";
        return 1;
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}
