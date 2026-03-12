#pragma once

#include <cupti.h>

// NVTX v3 headers — nvtx3/ path available since CUDA 12.x;
// CUDA 12.9 removed the v2 top-level headers entirely.
// This library is compiled with g++ (not nvcc) so CUDA_VERSION
// is not defined; always use the nvtx3/ path.
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtSync.h>

#include <generated_nvtx_meta.h>

namespace ProfileWrapper {

// Handles NVTX push/pop callbacks and external correlation
void CUPTIAPI NvtxCallbackHandler(void* userdata,
                                  CUpti_CallbackDomain domain,
                                  CUpti_CallbackId cbid,
                                  const void* cbdata);

// Called from the nvtxDomainRangePushEx LD_PRELOAD interposition for the
// null-domain case, where CUPTI does not fire a push callback.
void HandleNullDomainPush(const char* message);

} // namespace ProfileWrapper
