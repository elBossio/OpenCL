#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>   // if missing, try <CL/opencl.hpp>
#endif

#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "filter.hh"
#include "linear-algebra.hh"
#include "reduce-scan.hh"

using clock_type = std::chrono::high_resolution_clock;
using duration   = clock_type::duration;
using time_point = clock_type::time_point;

static size_t round_up(size_t x, size_t m) { return (x + m - 1) / m * m; }

void print(const char* name, std::array<duration,5> dt) {
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i=0; i<5; ++i) {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
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
    std::cout << '\n';
}

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

// Windows-friendly copy: pinned host staging buffer
template <class T>
cl::Buffer copy_to_device_pinned(OpenCL& opencl, const T* src, size_t count, cl_mem_flags device_flags) {
    const size_t bytes = count * sizeof(T);

    cl::Buffer staging(opencl.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes);

    void* mapped = opencl.queue.enqueueMapBuffer(staging, CL_TRUE, CL_MAP_WRITE, 0, bytes);
    std::memcpy(mapped, src, bytes);
    opencl.queue.enqueueUnmapMemObject(staging, mapped);

    cl::Buffer device(opencl.context, device_flags, bytes);
    opencl.queue.enqueueCopyBuffer(staging, device, 0, 0, bytes);

    opencl.queue.finish();
    return device;
}

// ===================== GPU inclusive scan for int =====================
static void gpu_scan_inclusive_int(OpenCL& opencl, const cl::Buffer& d_in, cl::Buffer& d_out, int n) {
    constexpr int BLOCK = 256;

    const int num_blocks = (n + BLOCK - 1) / BLOCK;
    const size_t global  = (size_t)num_blocks * (size_t)BLOCK;
    const size_t local   = (size_t)BLOCK;

    cl::Buffer d_block_sums(opencl.context, CL_MEM_READ_WRITE, sizeof(int) * (size_t)num_blocks);

    // 1) scan inside blocks + block sums
    cl::Kernel k_scan(opencl.program, "scan_inclusive_int");
    k_scan.setArg(0, d_in);
    k_scan.setArg(1, d_out);
    k_scan.setArg(2, d_block_sums);
    k_scan.setArg(3, n);

    opencl.queue.enqueueNDRangeKernel(k_scan, cl::NullRange, cl::NDRange(global), cl::NDRange(local));

    // 2) scan block sums recursively + add offsets
    if (num_blocks > 1) {
        cl::Buffer d_block_prefix(opencl.context, CL_MEM_READ_WRITE, sizeof(int) * (size_t)num_blocks);

        gpu_scan_inclusive_int(opencl, d_block_sums, d_block_prefix, num_blocks);

        cl::Kernel k_add(opencl.program, "add_offsets_int");
        k_add.setArg(0, d_out);
        k_add.setArg(1, d_block_prefix);
        k_add.setArg(2, n);

        opencl.queue.enqueueNDRangeKernel(k_add, cl::NullRange, cl::NDRange(global), cl::NDRange(local));
    }
}

// ===================== FILTER (stream compaction) =====================
void profile_filter(int n, OpenCL& opencl) {
    auto input = random_std_vector<float>(n);

    // CPU reference
    std::vector<float> expected;
    expected.reserve(n);

    auto t0 = clock_type::now();
    filter(input, expected, [] (float x) { return x > 0.0f; });
    auto t1 = clock_type::now();

    // OpenCL copy-in
    auto t1_start = t1;

    cl::Buffer d_in = copy_to_device_pinned(opencl, input.data(), input.size(), CL_MEM_READ_ONLY);

    cl::Buffer d_flags (opencl.context, CL_MEM_READ_WRITE, sizeof(int)   * (size_t)n);
    cl::Buffer d_prefix(opencl.context, CL_MEM_READ_WRITE, sizeof(int)   * (size_t)n);
    cl::Buffer d_out   (opencl.context, CL_MEM_READ_WRITE, sizeof(float) * (size_t)n); // max n

    auto t2 = clock_type::now();

    constexpr int BLOCK = 256;
    const size_t global = round_up((size_t)n, (size_t)BLOCK);
    const size_t local  = (size_t)BLOCK;

    // 1) flags[i] = 1 if input[i] > 0 else 0
    {
        cl::Kernel k_flags(opencl.program, "make_flags_positive");
        k_flags.setArg(0, d_in);
        k_flags.setArg(1, d_flags);
        k_flags.setArg(2, n);
        opencl.queue.enqueueNDRangeKernel(k_flags, cl::NullRange, cl::NDRange(global), cl::NDRange(local));
    }

    // 2) prefix = inclusive scan(flags)
    gpu_scan_inclusive_int(opencl, d_flags, d_prefix, n);

    // 3) scatter to compact array
    {
        cl::Kernel k_scatter(opencl.program, "scatter_positive");
        k_scatter.setArg(0, d_in);
        k_scatter.setArg(1, d_flags);
        k_scatter.setArg(2, d_prefix);
        k_scatter.setArg(3, d_out);
        k_scatter.setArg(4, n);
        opencl.queue.enqueueNDRangeKernel(k_scatter, cl::NullRange, cl::NDRange(global), cl::NDRange(local));
    }

    opencl.queue.finish();
    auto t3 = clock_type::now();

    // Copy-out: result size is prefix[n-1]
    int count = 0;
    opencl.queue.enqueueReadBuffer(d_prefix, CL_TRUE, sizeof(int) * (size_t)(n - 1), sizeof(int), &count);

    std::vector<float> result((size_t)count);
    if (count > 0) {
        opencl.queue.enqueueReadBuffer(d_out, CL_TRUE, 0, sizeof(float) * (size_t)count, result.data());
    }

    auto t4 = clock_type::now();

    verify_vector(expected, result);

    print("filter", {t1-t0, t4-t1_start, t2-t1_start, t3-t2, t4-t3});
}

void opencl_main(OpenCL& opencl) {
    print_column_names();
    profile_filter(1024 * 1024, opencl);
}

const std::string src = R"(
#define BLOCK 256

kernel void make_flags_positive(global const float* input,
                                global int* flags,
                                int n) {
    const int i = (int)get_global_id(0);
    if (i < n) {
        flags[i] = (input[i] > 0.0f) ? 1 : 0;
    }
}

// inclusive scan inside each block + block sums
kernel void scan_inclusive_int(global const int* in,
                               global int* out,
                               global int* block_sums,
                               int n) {
    local int tmp[BLOCK];

    const int lid = (int)get_local_id(0);
    const int gid = (int)get_group_id(0);
    const int idx = gid * BLOCK + lid;

    tmp[lid] = (idx < n) ? in[idx] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Hillis–Steele inclusive scan (simple and readable)
    for (int offset = 1; offset < BLOCK; offset <<= 1) {
        int add = (lid >= offset) ? tmp[lid - offset] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        tmp[lid] += add;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (idx < n) {
        out[idx] = tmp[lid];
    }

    if (lid == BLOCK - 1) {
        block_sums[gid] = tmp[lid];
    }
}

// add scanned block sums to each element (offsets)
kernel void add_offsets_int(global int* data,
                            global const int* block_prefix,
                            int n) {
    const int lid = (int)get_local_id(0);
    const int gid = (int)get_group_id(0);
    const int idx = gid * BLOCK + lid;

    if (gid == 0) return;
    if (idx >= n) return;

    data[idx] += block_prefix[gid - 1];
}

kernel void scatter_positive(global const float* input,
                             global const int* flags,
                             global const int* prefix,
                             global float* out,
                             int n) {
    const int i = (int)get_global_id(0);
    if (i < n && flags[i]) {
        const int pos = prefix[i] - 1; // inclusive -> 0-based
        out[pos] = input[i];
    }
}
)";

int main() {
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "Unable to find OpenCL platforms\n";
            return 1;
        }

        cl::Platform platform = platforms[0];
        std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';

        cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0 };
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);

        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[0];
        std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';

        cl::Program program(context, src);

        try {
            program.build(devices);
        } catch (const cl::Error&) {
            for (const auto& d : devices) {
                std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(d);
            }
            throw;
        }

        cl::CommandQueue queue(context, device);
        OpenCL opencl{platform, device, context, program, queue};
        opencl_main(opencl);

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        return 1;
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}
