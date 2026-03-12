#include "profiler_state.h"
#include <cxxabi.h>
#include <chrono>
#include <vector>

namespace ProfileWrapper {

// Static storage
static std::map<std::string, KernelStats> s_kernel_stats;
static std::vector<MemcpyRecord> s_memcpy_records;
static std::map<std::string, NvtxRangeStats> s_nvtx_range_stats;
static std::unordered_map<uint64_t, std::vector<ActiveNvtxRange>> s_thread_active_ranges;

static uint32_t s_next_external_id = 1;
static std::unordered_map<uint32_t, std::string> s_external_id_to_nvtx_name;
static std::unordered_map<uint32_t, uint32_t> s_external_id_to_parent_id;
static std::unordered_map<uint32_t, uint32_t> s_cuda_corr_to_external_id;

static CUpti_SubscriberHandle s_subscriber;

// Application lifetime tracking
static uint64_t s_app_start_ns = 0;
static uint64_t s_app_end_ns = 0;

// Helper: Get nanosecond timestamp
static uint64_t GetTimestampNs() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
}

// Helper: Demangle C++ kernel names
static std::string Demangle(const char* mangled) {
    int status;
    char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);

    if (status == 0 && demangled) {
        std::string result(demangled);
        free(demangled);

        // Simplify - extract just the function name
        size_t paren = result.find('(');
        if (paren != std::string::npos) {
            result = result.substr(0, paren);
        }

        return result;
    }

    return std::string(mangled);
}

// Lifecycle timing
void ProfilerState::RecordStartTime() {
    s_app_start_ns = GetTimestampNs();
}

void ProfilerState::RecordEndTime() {
    s_app_end_ns = GetTimestampNs();
}

uint64_t ProfilerState::GetApplicationDurationNs() {
    return s_app_end_ns - s_app_start_ns;
}

// Kernel statistics
std::map<std::string, KernelStats>& ProfilerState::GetKernelStats() {
    return s_kernel_stats;
}

void ProfilerState::RecordKernelInvocation(const std::string& mangled_name,
                                           const KernelInvocation& invocation) {
    auto& stats = s_kernel_stats[mangled_name];

    // Initialize on first use
    if (stats.demangled_name.empty()) {
        stats.demangled_name = Demangle(mangled_name.c_str());
        stats.total_duration_ns = 0;
        stats.count = 0;
    }

    stats.invocations.push_back(invocation);
    stats.total_duration_ns += invocation.duration_ns;
    stats.count++;

    // Update NVTX range stats for the innermost range
    if (!invocation.nvtx_range.empty()) {
        UpdateNvtxRangeStats(invocation.nvtx_range, invocation.duration_ns, 0, 0);
    }
}

// Memory transfers
std::vector<MemcpyRecord>& ProfilerState::GetMemcpyRecords() {
    return s_memcpy_records;
}

// Memory transfers - skip NVTX tracking if range is empty
void ProfilerState::RecordMemcpy(const MemcpyRecord& record) {
    s_memcpy_records.push_back(record);

    // Only track NVTX breakdown if there's an active range
    if (!record.nvtx_range.empty()) {
        UpdateNvtxRangeStats(record.nvtx_range, 0, record.duration_ns, record.bytes);
    }
}

// NVTX range statistics
std::map<std::string, NvtxRangeStats>& ProfilerState::GetNvtxRangeStats() {
    return s_nvtx_range_stats;
}

void ProfilerState::UpdateNvtxRangeStats(const std::string& range_name,
                                         uint64_t kernel_time_ns,
                                         uint64_t memcpy_time_ns,
                                         size_t memcpy_bytes) {
    auto& stats = s_nvtx_range_stats[range_name];
    stats.total_kernel_time_ns += kernel_time_ns;
    stats.total_memcpy_time_ns += memcpy_time_ns;
    stats.total_memcpy_bytes += memcpy_bytes;

    if (kernel_time_ns > 0) {
        stats.kernel_invocations++;
    }
    if (memcpy_time_ns > 0) {
        stats.memcpy_count++;
    }
}

void ProfilerState::RecordNvtxWallClockTime(const std::string& range_name,
                                            uint64_t duration_ns) {
    s_nvtx_range_stats[range_name].total_wall_clock_ns += duration_ns;
}

// NVTX thread tracking
std::unordered_map<uint64_t, std::vector<ActiveNvtxRange>>&
ProfilerState::GetThreadActiveRanges() {
    return s_thread_active_ranges;
}

// External correlation
uint32_t ProfilerState::GetNextExternalId() {
    return s_next_external_id++;
}

void ProfilerState::RegisterExternalId(uint32_t external_id, const std::string& nvtx_name,
                                       uint32_t parent_id) {
    s_external_id_to_nvtx_name[external_id] = nvtx_name;
    if (parent_id != 0) {
        s_external_id_to_parent_id[external_id] = parent_id;
    }
}

std::string ProfilerState::GetNvtxNameForExternalId(uint32_t external_id) {
    auto it = s_external_id_to_nvtx_name.find(external_id);
    return (it != s_external_id_to_nvtx_name.end()) ? it->second : "";
}

uint32_t ProfilerState::GetParentExternalId(uint32_t external_id) {
    auto it = s_external_id_to_parent_id.find(external_id);
    return (it != s_external_id_to_parent_id.end()) ? it->second : 0;
}

std::vector<std::string> ProfilerState::GetNvtxChainForCudaCorrelation(uint32_t cuda_corr_id) {
    std::vector<std::string> chain;
    auto it = s_cuda_corr_to_external_id.find(cuda_corr_id);
    if (it == s_cuda_corr_to_external_id.end()) return chain;

    uint32_t ext_id = it->second;
    while (ext_id != 0) {
        std::string name = GetNvtxNameForExternalId(ext_id);
        if (!name.empty()) chain.push_back(name);
        ext_id = GetParentExternalId(ext_id);
    }
    return chain;
}

void ProfilerState::SetNvtxRangeParent(const std::string& range_name,
                                       const std::string& parent_name) {
    auto& stats = s_nvtx_range_stats[range_name];
    if (stats.parent_range.empty()) {
        stats.parent_range = parent_name;
    }
}

void ProfilerState::MapCudaCorrelationToExternal(uint32_t cuda_corr_id, uint32_t external_id) {
    s_cuda_corr_to_external_id[cuda_corr_id] = external_id;
}

std::string ProfilerState::GetNvtxNameForCudaCorrelation(uint32_t cuda_corr_id) {
    auto it = s_cuda_corr_to_external_id.find(cuda_corr_id);
    if (it != s_cuda_corr_to_external_id.end()) {
        return GetNvtxNameForExternalId(it->second);
    }
    return "";
}

// CUPTI subscriber
void ProfilerState::SetSubscriber(CUpti_SubscriberHandle handle) {
    s_subscriber = handle;
}

CUpti_SubscriberHandle ProfilerState::GetSubscriber() {
    return s_subscriber;
}

} // namespace ProfileWrapper
