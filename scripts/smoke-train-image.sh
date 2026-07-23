#!/usr/bin/env bash
# CPU-only smoke gate for the train image (Dockerfile --target train), run by
# the release workflow before the -train tag is pushed — and runnable locally:
#
#   docker build --target train -t argus-forge:train-smoke .
#   scripts/smoke-train-image.sh argus-forge:train-smoke
#
# CI has no GPU, so this is deliberately short of a real training step (see
# README "CI / Release"). What it does gate:
#
#   1. the image serves: /health answers, and still reports training disabled —
#      the train variant must keep the demo-safe default;
#   2. the trainer stack is present and importable: torch, accelerate, and the
#      sd-scripts library (via its editable install), plus the launch script
#      train.sh will exec;
#   3. the validation path of a live run: forge a kohya config from a minimal
#      export, then `run --dry-run` it — prepare_run() resolving the forged
#      train.sh is exactly what POST /run does before launching.
set -euo pipefail

IMAGE="${1:?usage: smoke-train-image.sh <image-ref>}"

# tmpfs at the export root: /health only advertises a root whose directory
# exists at startup, so this also gates that the baked ARGUS_FORGE_EXPORT_ROOT
# survived the stage split (and gives the forged smoke config somewhere to land).
# No --rm: a container that crashes at startup must survive long enough for the
# trap to dump its logs — otherwise a boot failure surfaces in CI as a bare "No
# such container" and diagnosing it means rebuilding the multi-GB image locally.
cid="$(docker run -d --tmpfs /data/out "$IMAGE")"
trap 'rc=$?;
  if [ "$rc" -ne 0 ]; then
    echo "==> smoke failed (exit $rc); container logs:" >&2
    docker logs "$cid" >&2 || true
  fi
  docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT

echo "==> /health answers and keeps the demo-safe default"
# Polled from inside the container: the image has no curl, and publishing a
# host port would only add a collision to fail on.
docker exec -i "$cid" python - <<'PY'
import json, time, urllib.request

for _ in range(30):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8103/health", timeout=2) as r:
            health = json.load(r)
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("no /health answer after 30 attempts (~30-90s)")

print(health)
assert health["training"] == "disabled", "train image must default to demo-safe (ARGUS_FORGE_READONLY=1)"
assert health["export_root"] == "/data/out", "baked ARGUS_FORGE_EXPORT_ROOT must be advertised"
PY

echo "==> trainer stack imports (CPU-only: no CUDA device expected here)"
docker exec "$cid" bash -ec '
  test -f "$SD_SCRIPTS_DIR/sdxl_train_network.py"
  # On a GPU host triton (via bitsandbytes, kohya default AdamW8bit) JIT-
  # compiles a driver stub at import time and dies without a working C
  # toolchain. This CPU-only run cannot take that path, so gate on a test
  # compile — presence of cc alone misses missing libc headers.
  echo "int main(void){return 0;}" | cc -x c - -o /tmp/cc-check && rm /tmp/cc-check
  python -c "
import accelerate, bitsandbytes, torch, torchvision
import library.train_util  # sd-scripts editable install
# The requirements.txt install resolves against default PyPI: a future pin
# conflict could silently swap the cu124 wheels for PyPI ones and imports would
# still pass. Keep these in sync with the Dockerfile torch layer.
assert torch.__version__ == \"2.6.0+cu124\", f\"torch is {torch.__version__}, not the cu124 pin\"
assert torchvision.__version__ == \"0.21.0+cu124\", f\"torchvision is {torchvision.__version__}, not the cu124 pin\"
print(f\"torch {torch.__version__}, accelerate {accelerate.__version__}, cuda available: {torch.cuda.is_available()}\")
"
  # Import the module train.sh actually launches, plus the network_module the
  # forged config loads dynamically — library.train_util alone stops one layer
  # short of the real entrypoint (a bump-added import there would otherwise
  # surface as ModuleNotFoundError on the first real GPU run). The cd is
  # mandatory: the editable install exposes only the `library` package, and
  # sdxl_train_network.py resolves from the checkout root, exactly as train.sh
  # runs it. Import-time execution is guarded under __main__ at the pinned SHA.
  (cd "$SD_SCRIPTS_DIR" && python -c "import sdxl_train_network, networks.lora")
'

echo "==> forge a kohya config and validate the run path (prepare_run via --dry-run)"
docker exec "$cid" bash -ec '
  mkdir -p /data/out/smoke
  # A real (1x1) PNG — same bytes as tests/conftest.py PNG_1PX: manifest-less
  # fallback ingests the bare folder.
  echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" \
    | base64 -d > /data/out/smoke/img.png
  argus-forge config /data/out/smoke --trainer kohya
  argus-forge run /data/out/smoke --trainer kohya --dry-run
'

echo "OK: train image smoke passed"
