#include "profiler_lifecycle.h"
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cstdio>

// Function pointer to the real cudaLaunchKernel
typedef cudaError_t (*cudaLaunchKernel_t)(const void*, dim3, dim3, void**, size_t, cudaStream_t);

static cudaLaunchKernel_t GetRealCudaLaunchKernel() {
    static cudaLaunchKernel_t real_func = nullptr;
    if (!real_func) {
        real_func = (cudaLaunchKernel_t)dlsym(RTLD_NEXT, "cudaLaunchKernel");
        if (!real_func) {
            fprintf(stderr, "ERROR: Could not find real cudaLaunchKernel\n");
        }
    }
    return real_func;
}

// Interposed cudaLaunchKernel (passthrough)
extern "C" cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                                        void** args, size_t sharedMem, cudaStream_t stream) {
    auto real_func = GetRealCudaLaunchKernel();
    if (real_func) {
        return real_func(func, gridDim, blockDim, args, sharedMem, stream);
    }
    return cudaErrorInvalidValue;
}

// Library constructor - called when LD_PRELOAD loads the library
static void __attribute__((constructor)) InitProfiler() {
    ProfileWrapper::Initialize();
}

// Library destructor - called when library is unloaded
static void __attribute__((destructor)) CleanupProfiler() {
    ProfileWrapper::Finalize();
}
