// nvtx_interpose.cpp — NVTX injection entry point for libcuda_profile.so.
//
// nvtx3::scoped_range (C++ API, default/global domain) calls
// nvtxDomainRangePushEx(nullptr, ...) for push. CUPTI fires a pop callback
// (cbid=34) for the corresponding nvtxDomainRangePop(nullptr), but silently
// drops the push callback when domain == nullptr.
//
// NVTX_NO_IMPL must be defined before any NVTX headers to suppress inline
// function definitions that would conflict with our extern "C" symbols.
#define NVTX_NO_IMPL

#include "nvtx_correlation.h"
#include <dlfcn.h>
#include <cstdio>

typedef const void* (*nvtxGetExportTable_fn)(uint32_t exportTableId);
typedef int (*InitializeInjectionNvtx2_fn)(nvtxGetExportTable_fn);

static const unsigned int kPushExIdx = NVTX_CBID_CORE2_DomainRangePushEx;

typedef int (*nvtxDomainRangePushEx_fn)(nvtxDomainHandle_t,
                                        const nvtxEventAttributes_t*);

static nvtxDomainRangePushEx_fn g_next_nvtxDomainRangePushEx = nullptr;

static int OurNvtxDomainRangePushEx(nvtxDomainHandle_t domain,
                                    const nvtxEventAttributes_t* eventAttrib) {
    if (domain == nullptr) {
        // CUPTI does not fire a push callback for null-domain calls.
        // Handle it here so the external correlation ID is pushed before the
        // application launches any GPU work inside this range.
        const char* message =
            (eventAttrib && eventAttrib->messageType == NVTX_MESSAGE_TYPE_ASCII)
            ? eventAttrib->message.ascii : nullptr;
        ProfileWrapper::HandleNullDomainPush(message);
    }
    // For non-null domain, do nothing — CUPTI fires cbid=33 normally.
    // Chain to the next hook (CUPTI's) in all cases.
    return g_next_nvtxDomainRangePushEx
           ? g_next_nvtxDomainRangePushEx(domain, eventAttrib)
           : 0;
}

extern "C" int InitializeInjectionNvtx2(nvtxGetExportTable_fn getExportTable) {
    // --- Step 1: chain to CUPTI's InitializeInjectionNvtx2 ---
    // Use dladdr to locate the shared library that owns cuptiSubscribe (which
    // is already loaded by our constructor), then dlsym for CUPTI's init.
    {
        Dl_info info = {};
        if (dladdr(reinterpret_cast<void*>(cuptiSubscribe), &info) && info.dli_fname) {
            void* h = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
            if (h) {
                auto cupti_init = reinterpret_cast<InitializeInjectionNvtx2_fn>(
                    dlsym(h, "InitializeInjectionNvtx2"));
                if (cupti_init) {
                    cupti_init(getExportTable);
                }
                dlclose(h);
            }
        }
    }

    // --- Step 2: get the NVTX callbacks export table ---
    const NvtxExportTableCallbacks* etbl =
        static_cast<const NvtxExportTableCallbacks*>(
            getExportTable(NVTX_ETID_CALLBACKS));
    if (!etbl || !etbl->GetModuleFunctionTable) {
        return 1; // CUPTI hooks still work; our null-domain hook is absent
    }

    // --- Step 3: hook nvtxDomainRangePushEx in the CORE2 function table ---
    NvtxFunctionTable tbl = nullptr;
    unsigned int count = 0;
    if (etbl->GetModuleFunctionTable(NVTX_CB_MODULE_CORE2, &tbl, &count) == 1
        && tbl && count > kPushExIdx) {
        NvtxFunctionPointer* slot = tbl[kPushExIdx];
        if (slot) {
            // Save whatever is in slot 4 now (CUPTI's hook after Step 1).
            g_next_nvtxDomainRangePushEx =
                reinterpret_cast<nvtxDomainRangePushEx_fn>(*slot);
            // Install our wrapper.
            *slot = reinterpret_cast<NvtxFunctionPointer>(
                OurNvtxDomainRangePushEx);
        }
    }

    return 1;
}
