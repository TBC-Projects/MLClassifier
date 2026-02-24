# Deploying an ONNX Model to NVIDIA Jetson (ONNX Runtime + TensorRT)

This guide covers the full deployment flow with extra detail on **model transfer**. Steps are written so you can copy/paste commands.

## Prerequisites
- Jetson is flashed with JetPack (CUDA, cuDNN, TensorRT installed).
- Jetson and your dev machine are on the same network.
- You can SSH into the Jetson (user + IP).
- You have an exported `.onnx` model on your dev machine.

## Step-by-step deployment process

### 1) Confirm JetPack and CUDA are installed
On the Jetson:
```
nvcc --version
dpkg -l | grep -E "nvidia-jetpack|tensorrt|cudnn"
```

### 2) Create the ONNX file (export or convert)
On your dev machine, export from your training framework and validate the ONNX.
Make sure required packages are installed (examples): `pip install onnx onnxruntime tf2onnx`.

**Option A — PyTorch export**
```
python3 - <<'PY'
import torch

model = ...  # load your model
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
print("Saved model.onnx")
PY
```

**Option B — TensorFlow / Keras export**
```
python3 - <<'PY'
import tensorflow as tf
import tf2onnx

model = ...  # load your model
spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input"),)

tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=17,
    output_path="model.onnx",
)
print("Saved model.onnx")
PY
```

**Validate ONNX locally before transfer**
```
python3 - <<'PY'
import onnx
import onnxruntime as ort

onnx.checker.check_model("model.onnx")
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
print("ONNX load OK:", [i.name for i in sess.get_inputs()])
PY
```

Adjust input shapes, opset version, and names to match your model.

### 3) Transfer the ONNX model to the Jetson
Follow the transfer steps below (this is the most detailed section).

#### 3.1) Identify the Jetson on your network
On the Jetson:
```
hostname -I
```
Note the IP address (e.g., `192.168.1.50`).

#### 3.2) Create a target directory on the Jetson
On the Jetson:
```
mkdir -p ~/models/my_model
```

#### 3.3) (Recommended) Record the model file size on your dev machine
On your dev machine:
```
ls -lh /path/to/model.onnx
```
Windows (PowerShell):
```
Get-Item "C:\path\to\model.onnx" | Select-Object Name, Length
```

#### 3.4) Transfer the model
Pick one method.

**Option A — SCP (simple and common):**
```
scp "/path/to/model.onnx" username@192.168.1.50:~/models/my_model/
```

**Option B — rsync (faster for repeat transfers):**
```
rsync -avh --progress "/path/to/model.onnx" username@192.168.1.50:~/models/my_model/
```

**Option C — SFTP (interactive):**
```
sftp username@192.168.1.50
cd models/my_model
put /path/to/model.onnx
bye
```

#### 3.5) Verify the file on the Jetson
On the Jetson:
```
ls -lh ~/models/my_model/model.onnx
```
Confirm the size matches your dev machine.

#### 3.6) (Recommended) Verify integrity with checksums
On your dev machine:
```
sha256sum "/path/to/model.onnx"
```
Windows (PowerShell):
```
certutil -hashfile "C:\path\to\model.onnx" SHA256
```
On the Jetson:
```
sha256sum ~/models/my_model/model.onnx
```
The hashes should match exactly.

#### 3.7) (Optional) Transfer labels/config files
If your model uses labels or a config file:
```
scp "/path/to/labels.txt" username@192.168.1.50:~/models/my_model/
scp "/path/to/config.json" username@192.168.1.50:~/models/my_model/
```

#### 3.8) (Optional) Package multiple assets into a folder
On your dev machine:
```
mkdir -p model_bundle
cp /path/to/model.onnx model_bundle/
cp /path/to/labels.txt model_bundle/
cp /path/to/config.json model_bundle/
tar -czf model_bundle.tgz model_bundle
```
Transfer and extract on Jetson:
```
scp model_bundle.tgz username@192.168.1.50:~/
ssh username@192.168.1.50 "tar -xzf ~/model_bundle.tgz -C ~/"
```

### 4) Install ONNX Runtime with TensorRT support
On the Jetson:
- Install the ONNX Runtime wheel that matches your JetPack version.
- Verify providers:
```
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
You should see `TensorrtExecutionProvider` and `CUDAExecutionProvider`.

### 5) Run a smoke test inference (Python)
On the Jetson:
```
python3 - <<'PY'
import onnxruntime as ort
import numpy as np

providers = [
    ("TensorrtExecutionProvider", {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "./trt_cache",
    }),
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

sess = ort.InferenceSession("models/my_model/model.onnx", providers=providers)
input_name = sess.get_inputs()[0].name
x = np.random.rand(1, 3, 224, 224).astype(np.float32)
_ = sess.run(None, {input_name: x})
print("Inference OK")
PY
```
Replace the input shape and preprocessing to match your model.

### 6) Measure performance
- Warm up with 5–10 runs, then time inference.
- Separate preprocessing time from model runtime.

### 7) Package for deployment
- Bundle model, labels, config, and inference script.
- Optionally run as a systemd service or inside a Jetson Docker image.

## Common issues
- **Permission denied**: ensure SSH works and you have write access to `~/models/my_model`.
- **Slow transfer**: use `rsync` or a wired network.
- **Mismatched checksum**: re-transfer and avoid unstable Wi‑Fi.
- **Missing TensorRT provider**: install the correct ONNX Runtime wheel for your JetPack.
