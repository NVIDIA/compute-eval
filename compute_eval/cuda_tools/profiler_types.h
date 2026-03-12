#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <cstddef>

namespace ProfileWrapper {

// Individual kernel invocation record
struct KernelInvocation {
    uint64_t duration_ns;
    uint64_t start_ns;
    std::string nvtx_range;
};

// Aggregated statistics for a kernel
struct KernelStats {
    std::string demangled_name;
    std::vector<KernelInvocation> invocations;
    uint64_t total_duration_ns;
    size_t count;
};

// Individual memory copy record
struct MemcpyRecord {
    std::string kind;          // "HtoD", "DtoH", "DtoD", "Other"
    uint64_t duration_ns;
    size_t bytes;
    std::string nvtx_range;
};

// Aggregated statistics for an NVTX range
struct NvtxRangeStats {
    std::string parent_range;          // immediately enclosing named range, "" if top-level
    uint64_t total_wall_clock_ns;
    uint64_t total_kernel_time_ns;     // includes time from all descendant ranges
    uint64_t total_memcpy_time_ns;     // includes time from all descendant ranges
    size_t kernel_invocations;         // includes invocations from all descendant ranges
    size_t memcpy_count;               // includes transfers from all descendant ranges
    uint64_t total_memcpy_bytes;       // includes bytes from all descendant ranges
};

// Active NVTX range on a thread
struct ActiveNvtxRange {
    std::string name;
    uint64_t start_timestamp;
};

} // namespace ProfileWrapper
