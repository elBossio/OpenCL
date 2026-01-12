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
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "linear-algebra.hh"
#include "reduce-scan.hh"

using clock_type = std::chrono::high_resolution_clock;
using duration   = clock_type::duration;
using time_point = clock_type::time_point;

double bandwidth(int n, time_point t0, time_point t1) {
    using namespace std::chrono;
    const auto dt = duration_cast<microseconds>(t1 - t0).count();
    if (dt == 0) return 0;
    return ((n + n + n) * sizeof(float) * 1e-9) / (dt * 1e-6);
}

void print(const char* name, std::array<duration, 5> dt, std::array<double, 2> bw) {
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i = 0; i < 5; ++i) {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
        std::cout << std::setw(20) << tmp.str();
    }
    for (size_t i = 0; i < 2; ++i) {
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

// -------- Windows-friendly copy: pinned host staging --------
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

// ===================== REDUCE (fast, local mem, GPU finish) =====================
static void gpu_reduce(OpenCL& opencl, cl::Buffer& d_in, cl::Buffer& d_tmp, cl::Buffer& d_result, int n) {
    constexpr int WG = 256;

    cl::Kernel k(opencl.program, "reduce_pass");

    cl::Buffer* in  = &d_in;
    cl::Buffer* out = &d_tmp;
    int cur_n = n;

    while (true) {
        const int num_groups = (cur_n + (WG * 2 - 1)) / (WG * 2);
        const size_t global = (size_t)num_groups * (size_t)WG;
        const size_t local  = (size_t)WG;

        k.setArg(0, *in);
        k.setArg(1, *out);
        k.setArg(2, d_result);
        k.setArg(3, cur_n);

        opencl.queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(global), cl::NDRange(local));

        if (num_groups == 1) break;
        cur_n = num_groups;
        std::swap(in, out);
    }
}

static void gpu_scan_inclusive_serial(OpenCL& opencl, const cl::Buffer& d_in, cl::Buffer& d_out, int n) {
    constexpr int BLOCK = 256;

    cl::Kernel k(opencl.program, "scan_inclusive_serial");
    k.setArg(0, d_in);
    k.setArg(1, d_out);
    k.setArg(2, n);

    // Один work-group обрабатывает весь массив чанками.
    opencl.queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(BLOCK), cl::NDRange(BLOCK));
}

void profile_reduce(int n, OpenCL& opencl) {
    auto a = random_vector<float>(n);

    float expected = 0.0f;
    float actual   = 0.0f;

    // CPU (OpenMP)
    auto t0 = clock_type::now();
    expected = reduce(a);
    auto t1 = clock_type::now();

    // OpenCL copy-in
    auto t1_start = t1;
    cl::Buffer d_a = copy_to_device_pinned(opencl, std::begin(a), (size_t)a.size(), CL_MEM_READ_ONLY);

    cl::Buffer d_tmp(opencl.context, CL_MEM_READ_WRITE, sizeof(float) * (size_t)a.size());
    cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, sizeof(float));

    auto t2 = clock_type::now();

    // OpenCL kernel(s)
    gpu_reduce(opencl, d_a, d_tmp, d_result, n);
    opencl.queue.finish();
    auto t3 = clock_type::now();

    // OpenCL copy-out
    opencl.queue.enqueueReadBuffer(d_result, CL_TRUE, 0, sizeof(float), &actual);
    auto t4 = clock_type::now();

    // Verification for reduce: allow small relative tolerance (OpenMP SIMD sums differently)
    const float diff = std::abs(expected - actual);
    const float tol  = 1e-4f * (std::abs(expected) + 1.0f);
    if (diff > tol) {
        std::stringstream msg;
        msg << "Bad reduce value: expected=" << expected << ", actual=" << actual
            << " (diff=" << diff << ", tol=" << tol << ")";
        throw std::runtime_error(msg.str());
    }

    print("reduce",
          {t1 - t0, t4 - t1_start, t2 - t1_start, t3 - t2, t4 - t3},
          {bandwidth(n, t0, t1), bandwidth(n, t2, t3)});
}

void profile_scan_inclusive(int n, OpenCL& opencl) {
    auto a = random_vector<float>(n);
    Vector<float> expected(a), actual(a);

    // CPU
    auto t0 = clock_type::now();
    scan_inclusive(expected);
    auto t1 = clock_type::now();

    // OpenCL copy-in
    auto t1_start = t1;
    cl::Buffer d_in  = copy_to_device_pinned(opencl, std::begin(actual), (size_t)actual.size(), CL_MEM_READ_ONLY);
    cl::Buffer d_out(opencl.context, CL_MEM_READ_WRITE, sizeof(float) * (size_t)actual.size());
    auto t2 = clock_type::now();

    // OpenCL kernel
    gpu_scan_inclusive_serial(opencl, d_in, d_out, n);
    opencl.queue.finish();
    auto t3 = clock_type::now();

    // OpenCL copy-out
    opencl.queue.enqueueReadBuffer(d_out, CL_TRUE, 0,
                                   sizeof(float) * (size_t)actual.size(),
                                   std::begin(actual));
    auto t4 = clock_type::now();

    // Verification is strict (eps=1e-6)
    verify_vector(expected, actual);

    print("scan-inclusive",
          {t1 - t0, t4 - t1_start, t2 - t1_start, t3 - t2, t4 - t3},
          {bandwidth(n, t0, t1), bandwidth(n, t2, t3)});
}

void opencl_main(OpenCL& opencl) {
    print_column_names();
    profile_reduce(1024 * 1024 * 10, opencl);
    profile_scan_inclusive(1024 * 1024 * 10, opencl);
}

const std::string src = R"(
#define WG 256
#define BLOCK 256

kernel void reduce_pass(global const float* a,
                        global float* b,
                        global float* result,
                        int n) {
    // Tree reduction inside one work-group using local memory.
    local float sdata[WG];

    const int lid = (int)get_local_id(0);
    const int gid = (int)get_group_id(0);

    const int base = gid * (WG * 2) + lid;

    float sum = 0.0f;
    if (base < n)      sum += a[base];
    if (base + WG < n) sum += a[base + WG];

    sdata[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = WG / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            sdata[lid] += sdata[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        b[gid] = sdata[0];
        // GPU-finish: last pass writes final result
        if (get_num_groups(0) == 1) {
            result[0] = sdata[0];
        }
    }
}

kernel void scan_inclusive_serial(global const float* a,
                                  global float* out,
                                  int n) {
    // Exact inclusive scan matching CPU order.
    // One work-group scans the whole array in chunks of BLOCK, using local memory.
    local float tmp[BLOCK];

    const int lid = (int)get_local_id(0);

    float carry = 0.0f;

    for (int base = 0; base < n; base += BLOCK) {
        const int idx = base + lid;

        tmp[lid] = (idx < n) ? a[idx] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0) {
            float sum = carry;
            for (int i = 0; i < BLOCK; ++i) {
                sum += tmp[i];
                tmp[i] = sum;
            }
            carry = tmp[BLOCK - 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (idx < n) {
            out[idx] = tmp[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
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

        cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0
        };
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
