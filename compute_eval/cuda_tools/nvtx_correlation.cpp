#include "nvtx_correlation.h"
#include "profiler_state.h"
#include <cstdio>
#include <vector>

namespace ProfileWrapper {

// Per-thread push stack. Stores the external_id that was pushed for each NVTX push
// callback, or 0 if the push had no extractable message and therefore did not call
// cuptiActivityPushExternalCorrelationId. The pop callback uses this to decide
// whether to call the corresponding CUPTI pop, keeping the two stacks in sync.
static thread_local std::vector<uint32_t> s_push_stack;

static void push_correlation(const char* message) {
    if (message) {
        uint32_t parent_id = 0;
        for (auto it = s_push_stack.rbegin(); it != s_push_stack.rend(); ++it) {
            if (*it != 0) { parent_id = *it; break; }
        }

        uint32_t external_id = ProfilerState::GetNextExternalId();
        ProfilerState::RegisterExternalId(external_id, std::string(message), parent_id);
        cuptiActivityPushExternalCorrelationId(
            CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, external_id);
        s_push_stack.push_back(external_id);

        if (parent_id != 0) {
            std::string parent_name = ProfilerState::GetNvtxNameForExternalId(parent_id);
            if (!parent_name.empty()) {
                ProfilerState::SetNvtxRangeParent(std::string(message), parent_name);
            }
        }
    } else {
        s_push_stack.push_back(0);
    }
}

static void pop_correlation() {
    if (!s_push_stack.empty()) {
        uint32_t pushed_id = s_push_stack.back();
        s_push_stack.pop_back();
        if (pushed_id != 0) {
            uint64_t popped_id;
            cuptiActivityPopExternalCorrelationId(
                CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &popped_id);
        }
    }
}

// Called from the nvtxDomainRangePushEx LD_PRELOAD interposition for the
// null-domain case. CUPTI does not fire a push callback when domain == nullptr
// so we handle it here instead.
void HandleNullDomainPush(const char* message) {
    push_correlation(message);
}

void CUPTIAPI NvtxCallbackHandler(void* userdata,
                                  CUpti_CallbackDomain domain,
                                  CUpti_CallbackId cbid,
                                  const void* cbdata) {
    // Driver API: manually record correlationId -> active NVTX external_id for
    // kernel launch calls. cuLaunchKernelEx and similar newer driver APIs (used by
    // JIT-compiled kernels via nvrtc/PTX) may not generate
    // CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION records automatically, so we
    // supplement the cuptiActivityPushExternalCorrelationId mechanism here.
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        const CUpti_CallbackData* data = (const CUpti_CallbackData*)cbdata;
        if (data->callbackSite != CUPTI_API_ENTER) return;

        bool is_launch = false;
        switch (cbid) {
            case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
            case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
            case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
            case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz:
            case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
            case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz:
                is_launch = true;
                break;
            default:
                break;
        }

        if (is_launch) {
            // Scan the thread-local stack for the innermost active external_id
            uint32_t active_id = 0;
            for (auto it = s_push_stack.rbegin(); it != s_push_stack.rend(); ++it) {
                if (*it != 0) { active_id = *it; break; }
            }
            if (active_id != 0) {
                ProfilerState::MapCudaCorrelationToExternal(
                    data->correlationId, active_id);
            }
        }
        return;
    }

    if (domain != CUPTI_CB_DOMAIN_NVTX) {
        return;
    }

    const CUpti_NvtxData* nvtxData = (const CUpti_NvtxData*)cbdata;

    // Helper to extract message from event attributes
    auto extract_message = [](const nvtxEventAttributes_t* attr) -> const char* {
        if (attr && attr->messageType == NVTX_MESSAGE_TYPE_ASCII) {
            return attr->message.ascii;
        }
        return nullptr;
    };

    // Handle different NVTX callback types
    switch (cbid) {
        // C API: nvtxRangePushA
        case CUPTI_CBID_NVTX_nvtxRangePushA: {
            const nvtxRangePushA_params_st* params =
                (const nvtxRangePushA_params_st*)nvtxData->functionParams;
            push_correlation(params ? params->message : nullptr);
            break;
        }

        // C API: nvtxRangePushEx
        case CUPTI_CBID_NVTX_nvtxRangePushEx: {
            const nvtxRangePushEx_params_st* params =
                (const nvtxRangePushEx_params_st*)nvtxData->functionParams;
            push_correlation(extract_message(params ? params->eventAttrib : nullptr));
            break;
        }

        // C API: nvtxRangePop
        case CUPTI_CBID_NVTX_nvtxRangePop: {
            pop_correlation();
            break;
        }

        // C++ API: nvtxDomainRangePushEx
        case CUPTI_CBID_NVTX_nvtxDomainRangePushEx: {
            const nvtxDomainRangePushEx_params_st* params =
                (const nvtxDomainRangePushEx_params_st*)nvtxData->functionParams;
            push_correlation(extract_message(params ? params->core.eventAttrib : nullptr));
            break;
        }

        // C++ API: nvtxDomainRangePop
        case CUPTI_CBID_NVTX_nvtxDomainRangePop: {
            pop_correlation();
            break;
        }

        default:
            break;
    }
}

} // namespace ProfileWrapper
