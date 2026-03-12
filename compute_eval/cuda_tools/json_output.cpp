#include "json_output.h"
#include "profiler_state.h"
#include <cstdio>
#include <map>

namespace ProfileWrapper {

void WriteJson(FILE* out) {
    const auto& kernel_stats = ProfilerState::GetKernelStats();
    const auto& memcpy_records = ProfilerState::GetMemcpyRecords();
    const auto& nvtx_range_stats = ProfilerState::GetNvtxRangeStats();

    fprintf(out, "{\n");

    // Kernel statistics
    fprintf(out, "  \"kernels\": [\n");
    bool first_kernel = true;
    uint64_t total_kernel_time_ns = 0;

    for (const auto& entry : kernel_stats) {
        if (!first_kernel) fprintf(out, ",\n");
        first_kernel = false;

        const auto& stats = entry.second;
        total_kernel_time_ns += stats.total_duration_ns;

        double total_ms = stats.total_duration_ns / 1e6;
        double avg_ms = total_ms / stats.count;

        fprintf(out, "    {\n");
        fprintf(out, "      \"name\": \"%s\",\n", stats.demangled_name.c_str());
        fprintf(out, "      \"invocations\": %zu,\n", stats.count);
        fprintf(out, "      \"total_duration_ms\": %.6f,\n", total_ms);
        fprintf(out, "      \"avg_duration_ms\": %.6f\n", avg_ms);
        fprintf(out, "    }");
    }
    fprintf(out, "\n  ],\n");

    // Memory transfers
    fprintf(out, "  \"memory_transfers\": {\n");

    std::map<std::string, std::vector<MemcpyRecord>> transfers_by_kind;
    for (const auto& rec : memcpy_records) {
        transfers_by_kind[rec.kind].push_back(rec);
    }

    bool first_kind = true;
    for (const auto& entry : transfers_by_kind) {
        if (!first_kind) fprintf(out, ",\n");
        first_kind = false;

        const std::string& kind = entry.first;
        const auto& transfers = entry.second;

        uint64_t total_bytes = 0;
        uint64_t total_time_ns = 0;
        std::map<std::string, uint64_t> nvtx_bytes;
        std::map<std::string, uint64_t> nvtx_time_ns;

        for (const auto& rec : transfers) {
            total_bytes += rec.bytes;
            total_time_ns += rec.duration_ns;
            nvtx_bytes[rec.nvtx_range] += rec.bytes;
            nvtx_time_ns[rec.nvtx_range] += rec.duration_ns;
        }

        double bandwidth_gbps = 0.0;
        if (total_time_ns > 0) {
            double time_s = total_time_ns / 1e9;
            bandwidth_gbps = (total_bytes / time_s) / 1e9;
        }

        fprintf(out, "    \"%s\": {\n", kind.c_str());
        fprintf(out, "      \"count\": %zu,\n", transfers.size());
        fprintf(out, "      \"total_bytes\": %lu,\n", total_bytes);
        fprintf(out, "      \"total_time_ms\": %.6f,\n", total_time_ns / 1e6);
        fprintf(out, "      \"bandwidth_gbps\": %.3f,\n", bandwidth_gbps);
        fprintf(out, "      \"nvtx\": {\n");

        bool first_nvtx = true;
        for (const auto& nvtx_entry : nvtx_bytes) {
            if (!first_nvtx) fprintf(out, ",\n");
            first_nvtx = false;

            const std::string& range_name = nvtx_entry.first;
            uint64_t range_bytes = nvtx_entry.second;
            uint64_t range_time_ns = nvtx_time_ns[range_name];

            double range_bw_gbps = 0.0;
            if (range_time_ns > 0) {
                double time_s = range_time_ns / 1e9;
                range_bw_gbps = (range_bytes / time_s) / 1e9;
            }

            fprintf(out, "        \"%s\": {\n", range_name.c_str());
            fprintf(out, "          \"bytes\": %lu,\n", range_bytes);
            fprintf(out, "          \"time_ms\": %.6f,\n", range_time_ns / 1e6);
            fprintf(out, "          \"bandwidth_gbps\": %.3f\n", range_bw_gbps);
            fprintf(out, "        }");
        }
        fprintf(out, "\n      }\n");
        fprintf(out, "    }");
    }
    fprintf(out, "\n  },\n");

    // NVTX ranges summary
    fprintf(out, "  \"nvtx_ranges\": {\n");
    bool first_nvtx_summary = true;
    for (const auto& entry : nvtx_range_stats) {
        if (!first_nvtx_summary) fprintf(out, ",\n");
        first_nvtx_summary = false;

        const std::string& range_name = entry.first;
        const NvtxRangeStats& stats = entry.second;

        fprintf(out, "    \"%s\": {\n", range_name.c_str());

        if (!stats.parent_range.empty()) {
            fprintf(out, "      \"parent\": \"%s\",\n", stats.parent_range.c_str());
        }

        if (stats.total_wall_clock_ns > 0) {
            fprintf(out, "      \"wall_clock_time_ms\": %.6f,\n",
                    stats.total_wall_clock_ns / 1e6);
        }

        fprintf(out, "      \"kernel_time_ms\": %.6f,\n", stats.total_kernel_time_ns / 1e6);
        fprintf(out, "      \"kernel_invocations\": %zu,\n", stats.kernel_invocations);
        fprintf(out, "      \"memcpy_time_ms\": %.6f,\n", stats.total_memcpy_time_ns / 1e6);
        fprintf(out, "      \"memcpy_count\": %zu,\n", stats.memcpy_count);
        fprintf(out, "      \"memcpy_bytes\": %lu", stats.total_memcpy_bytes);
        fprintf(out, "\n    }");
    }
    fprintf(out, "\n  },\n");

    // Summary
    fprintf(out, "  \"summary\": {\n");
    fprintf(out, "    \"total_kernel_time_ms\": %.6f,\n", total_kernel_time_ns / 1e6);
    fprintf(out, "    \"total_kernels\": %zu,\n", kernel_stats.size());

    size_t total_invocations = 0;
    for (const auto& entry : kernel_stats) {
        total_invocations += entry.second.count;
    }
    fprintf(out, "    \"total_kernel_invocations\": %zu,\n", total_invocations);
    fprintf(out, "    \"total_memory_transfers\": %zu,\n", memcpy_records.size());

    // Application lifetime
    uint64_t app_duration_ns = ProfilerState::GetApplicationDurationNs();
    fprintf(out, "    \"application_duration_ms\": %.6f\n", app_duration_ns / 1e6);

    fprintf(out, "  }\n");
    fprintf(out, "}\n");
}

void PrintAndSaveJson() {
    // Write to stdout
    WriteJson(stdout);

    // Also write to file
    const char* filename = "profile.json";
    FILE* f = fopen(filename, "w");
    if (f) {
        WriteJson(f);
        fclose(f);
        fprintf(stderr, "Profile saved to: %s\n", filename);
    } else {
        fprintf(stderr, "Warning: Could not write to %s\n", filename);
    }
}

} // namespace ProfileWrapper
