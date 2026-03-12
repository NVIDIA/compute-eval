#include "profiler_lifecycle.h"
#include "profiler_state.h"
#include "nvtx_correlation.h"
#include "activity_processing.h"
#include "json_output.h"
#include <cupti.h>
#include <cstdio>

namespace ProfileWrapper {

void Initialize() {
    // Record application start time
    ProfilerState::RecordStartTime();

    // Subscribe to CUPTI callbacks
    CUpti_SubscriberHandle subscriber;
    CUptiResult res = cuptiSubscribe(&subscriber,
                                     (CUpti_CallbackFunc)NvtxCallbackHandler,
                                     nullptr);
    if (res != CUPTI_SUCCESS) {
        fprintf(stderr, "Failed to subscribe to CUPTI: %d\n", res);
        return;
    }
    ProfilerState::SetSubscriber(subscriber);

    // Enable callback domains
    res = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_NVTX);
    if (res == CUPTI_SUCCESS) {
        fprintf(stderr, "Enabled NVTX domain: %d\n", res);
    }

    res = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    if (res == CUPTI_SUCCESS) {
        fprintf(stderr, "[DEBUG] cuptiEnableDomain(RUNTIME_API) returned: %d\n", res);
    }

    res = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
    if (res == CUPTI_SUCCESS) {
        fprintf(stderr, "[DEBUG] cuptiEnableDomain(DRIVER_API) returned: %d\n", res);
    }

    // Register activity API callbacks
    cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted);

    // Enable activity kinds
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER_DATA);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);

    res = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
    if (res == CUPTI_SUCCESS) {
        fprintf(stdout, "External correlation enabled ✓\n");
    }

    fprintf(stdout, "CUDA Profiling initialized\n");

    // Debug: Verify external correlation is working
    CUpti_ActivityKind kind = CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION;
    size_t size = sizeof(kind);
    CUptiResult attrRes = cuptiActivityGetAttribute(
        CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &size, &kind);
    fprintf(stderr, "[DEBUG] cuptiActivityGetAttribute check: %d\n", attrRes);
}

void Finalize() {
    // Flush all pending activities
    cuptiActivityFlushAll(0);

    // Record application end time
    ProfilerState::RecordEndTime();

    // Write results
    PrintAndSaveJson();

    // Disable activities
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MARKER);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MARKER_DATA);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);

    // Unsubscribe from callbacks
    cuptiUnsubscribe(ProfilerState::GetSubscriber());
}

} // namespace ProfileWrapper
