FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN rm -rf dist build || true

RUN pip install --no-cache-dir build scikit-build-core pybind11 numpy

RUN python3 -m build --wheel

RUN python3 - <<'PY'
import glob, sys, os, subprocess
wh = glob.glob('dist/*.whl')
if not wh:
    print("ERROR: no wheel found in dist/; build failed. Dist contents:", os.listdir('dist') if os.path.isdir('dist') else 'dist missing')
    sys.exit(2)
wheel = wh[0]
print("Installing wheel:", wheel)
res = subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", wheel])
if res.returncode != 0:
    print("pip install failed")
    sys.exit(res.returncode)
print("Installed", wheel)
PY

RUN rm -rf /app/tensor_ops

CMD ["python3", "tests/test_mac.py"]