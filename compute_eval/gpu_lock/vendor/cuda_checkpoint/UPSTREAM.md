# cuda-checkpoint

- Repository: https://github.com/NVIDIA/cuda-checkpoint
- Commit: `00d5cce84c628088d6caa203fc4af40c1538b6f7`
- Minimum NVIDIA display driver: 550

## Bundled binaries

| Platform      | Path                            | SHA-256                                                            |
| ------------- | ------------------------------- | ----------------------------------------------------------------- |
| Linux x86-64  | `x86_64_linux/cuda-checkpoint`  | `707fa7f54136824d6c1d6dd724b9b1717610f831033c00d06da474de363a06db` |
| Linux aarch64 | `aarch64_linux/cuda-checkpoint` | `4636b2615135b87e5ff524d2e80ecc0d4a4e599f1e61fab07374747e3b3c8414` |

The x86-64 binary is byte-identical to the one previously vendored from commit
`1c71b048c779d30921f01172dda96e04033bb8ac`; the pin was advanced to
`00d5cce84c628088d6caa203fc4af40c1538b6f7` only to pick up the aarch64 binary,
which upstream added in that commit.

The binaries and license are copied without modification from the upstream
repository. `scheduler.py` selects the binary matching `platform.machine()`.
