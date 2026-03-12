#include "activity_processing.h"
#include "profiler_state.h"
#include <cstdlib>
#include <cstdio>
#include <map>

namespace ProfileWrapper {

void CUPTIAPI BufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
    *size = 8 * 1024 * 1024;  // 8MB buffer
    *buffer = (uint8_t*)malloc(*size);
    *maxNumRecords = 0;
}

void CUPTIAPI BufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer,
                              size_t size, size_t validSize) {
    CUpti_Activity* record = nullptr;

    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {

        // Process external correlation records (NVTX -> CUDA mapping)
        if (record->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
            CUpti_ActivityExternalCorrelation* extCorr =
                (CUpti_ActivityExternalCorrelation*)record;
            ProfilerState::MapCudaCorrelationToExternal(
                extCorr->correlationId, extCorr->externalId);
        }

        // Process NVTX marker records (wall-clock timing)
        else if (record->kind == CUPTI_ACTIVITY_KIND_MARKER) {
            CUpti_ActivityMarker2* marker = (CUpti_ActivityMarker2*)record;

            if (marker->flags & CUPTI_ACTIVITY_FLAG_MARKER_START) {
                // START markers: check name is present and it's a thread marker
                if (marker->name != nullptr && marker->objectKind == CUPTI_ACTIVITY_OBJECT_THREAD) {
                    uint64_t thread_key = ((uint64_t)marker->objectId.pt.processId << 32) |
                                          marker->objectId.pt.threadId;

                    ActiveNvtxRange active;
                    active.name = std::string(marker->name);
                    active.start_timestamp = marker->timestamp;

                    auto& thread_ranges = ProfilerState::GetThreadActiveRanges()[thread_key];
                    thread_ranges.push_back(active);
                }
            }
            else if (marker->flags & CUPTI_ACTIVITY_FLAG_MARKER_END) {
                // END markers: only check object kind (name might not be populated)
                if (marker->objectKind == CUPTI_ACTIVITY_OBJECT_THREAD) {
                    uint64_t thread_key = ((uint64_t)marker->objectId.pt.processId << 32) |
                                          marker->objectId.pt.threadId;

                    auto& thread_ranges = ProfilerState::GetThreadActiveRanges()[thread_key];

                    if (!thread_ranges.empty()) {
                        ActiveNvtxRange& active = thread_ranges.back();
                        uint64_t duration_ns = marker->timestamp - active.start_timestamp;
                        ProfilerState::RecordNvtxWallClockTime(active.name, duration_ns);
                        thread_ranges.pop_back();
                    }
                }
            }
        }

        // Process kernel records
        else if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
                 record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            CUpti_ActivityKernel4* kernel = (CUpti_ActivityKernel4*)record;

            std::string mangled_name(kernel->name);
            uint64_t duration_ns = kernel->end - kernel->start;

            // Build the full ancestry chain: chain[0] = innermost range, chain[n] = outermost
            std::vector<std::string> nvtx_chain =
                ProfilerState::GetNvtxChainForCudaCorrelation(kernel->correlationId);

            KernelInvocation inv;
            inv.duration_ns = duration_ns;
            inv.start_ns = kernel->start;
            inv.nvtx_range = nvtx_chain.empty() ? "" : nvtx_chain[0];

            // RecordKernelInvocation handles attribution to the innermost range
            ProfilerState::RecordKernelInvocation(mangled_name, inv);

            // Roll up kernel time to all ancestor ranges so parent totals are inclusive
            for (size_t i = 1; i < nvtx_chain.size(); i++) {
                ProfilerState::UpdateNvtxRangeStats(nvtx_chain[i], duration_ns, 0, 0);
            }
        }

        // Process memory copy records
        else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            CUpti_ActivityMemcpy* memcpy = (CUpti_ActivityMemcpy*)record;

            // Build the full ancestry chain for this transfer
            std::vector<std::string> nvtx_chain =
                ProfilerState::GetNvtxChainForCudaCorrelation(memcpy->correlationId);

            MemcpyRecord rec;
            rec.duration_ns = memcpy->end - memcpy->start;
            rec.bytes = memcpy->bytes;
            rec.nvtx_range = nvtx_chain.empty() ? "" : nvtx_chain[0];

            switch (memcpy->copyKind) {
                case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
                    rec.kind = "HtoD";
                    break;
                case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
                    rec.kind = "DtoH";
                    break;
                case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
                    rec.kind = "DtoD";
                    break;
                default:
                    rec.kind = "Other";
            }

            // RecordMemcpy handles attribution to the innermost range
            ProfilerState::RecordMemcpy(rec);

            // Roll up memcpy time/bytes to all ancestor ranges
            for (size_t i = 1; i < nvtx_chain.size(); i++) {
                ProfilerState::UpdateNvtxRangeStats(nvtx_chain[i], 0, rec.duration_ns, rec.bytes);
            }
        }
    }

    free(buffer);
}

} // namespace ProfileWrapper
