#pragma once

#include <cupti.h>
#include <cstdint>
#include <cstddef>

namespace ProfileWrapper {

// CUPTI activity buffer lifecycle callbacks
void CUPTIAPI BufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords);
void CUPTIAPI BufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t* buffer,
                              size_t size, size_t validSize);

} // namespace ProfileWrapper
