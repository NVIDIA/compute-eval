#pragma once

namespace ProfileWrapper {

// Initialize profiler (called at library load)
void Initialize();

// Finalize profiler and output results (called at library unload)
void Finalize();

} // namespace ProfileWrapper
