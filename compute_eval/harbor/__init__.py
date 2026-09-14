import os

# Harbor 0.20 enables anonymous usage reporting by default. ComputeEval runs can
# contain internal model and benchmark metadata, so keep reporting off unless an
# operator explicitly opts in before starting the harness.
os.environ.setdefault("HARBOR_TELEMETRY", "off")

__all__: list[str] = []
