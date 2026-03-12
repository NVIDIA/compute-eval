#pragma once

#include <cstdio>

namespace ProfileWrapper {

// Write profiling results as JSON to file handle
void WriteJson(FILE* out);

// Write to both stdout and file
void PrintAndSaveJson();

} // namespace ProfileWrapper
