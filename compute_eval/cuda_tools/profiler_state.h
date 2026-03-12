#pragma once

#include "profiler_types.h"
#include <cupti.h>
#include <unordered_map>
#include <vector>

namespace ProfileWrapper {

// Centralized state management for the profiler
class ProfilerState {
public:
    // Lifecycle timing
    static void RecordStartTime();
    static void RecordEndTime();
    static uint64_t GetApplicationDurationNs();

    // Kernel statistics
    static std::map<std::string, KernelStats>& GetKernelStats();
    static void RecordKernelInvocation(const std::string& mangled_name,
                                       const KernelInvocation& invocation);

    // Memory transfer records
    static std::vector<MemcpyRecord>& GetMemcpyRecords();
    static void RecordMemcpy(const MemcpyRecord& record);

    // NVTX range statistics
    static std::map<std::string, NvtxRangeStats>& GetNvtxRangeStats();
    static void UpdateNvtxRangeStats(const std::string& range_name,
                                     uint64_t kernel_time_ns,
                                     uint64_t memcpy_time_ns,
                                     size_t memcpy_bytes);
    static void RecordNvtxWallClockTime(const std::string& range_name,
                                        uint64_t duration_ns);

    // NVTX correlation tracking
    static std::unordered_map<uint64_t, std::vector<ActiveNvtxRange>>& GetThreadActiveRanges();

    // External correlation (NVTX push/pop to CUDA operations)
    static uint32_t GetNextExternalId();
    static void RegisterExternalId(uint32_t external_id, const std::string& nvtx_name,
                                   uint32_t parent_id = 0);
    static std::string GetNvtxNameForExternalId(uint32_t external_id);
    static uint32_t GetParentExternalId(uint32_t external_id);

    static void MapCudaCorrelationToExternal(uint32_t cuda_corr_id, uint32_t external_id);
    static std::string GetNvtxNameForCudaCorrelation(uint32_t cuda_corr_id);

    // Returns the full ancestry chain for a CUDA correlation ID, innermost range first.
    static std::vector<std::string> GetNvtxChainForCudaCorrelation(uint32_t cuda_corr_id);

    // Records the immediately enclosing named range for a given range (first write wins).
    static void SetNvtxRangeParent(const std::string& range_name, const std::string& parent_name);

    // CUPTI subscriber handle
    static void SetSubscriber(CUpti_SubscriberHandle handle);
    static CUpti_SubscriberHandle GetSubscriber();

private:
    ProfilerState() = delete;  // Static only
};

} // namespace ProfileWrapper
