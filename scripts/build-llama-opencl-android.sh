#!/bin/bash
# Build llama.cpp for Android with the Adreno OpenCL backend (+ optimized CPU).
# Follows docs/backend/OPENCL.md, adapted to NDK r27 on macOS host.
set -uo pipefail
NDK=~/Library/Android/sdk/ndk/27.1.12297006
SYSROOT=$NDK/toolchains/llvm/prebuilt/darwin-x86_64/sysroot
WORK=~/np2-aibench2
SRC=$WORK/llama.cpp
LOG=$WORK/build-gpu.log
: > "$LOG"
log(){ echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

command -v cmake >/dev/null || { log "FATAL: no cmake"; exit 1; }
NINJA=""
command -v ninja >/dev/null && NINJA="-G Ninja"

# 1) OpenCL headers into NDK sysroot
if [ ! -d "$SYSROOT/usr/include/CL" ]; then
  log "=== OpenCL-Headers ==="
  git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers "$WORK/OpenCL-Headers" >>"$LOG" 2>&1
  cp -r "$WORK/OpenCL-Headers/CL" "$SYSROOT/usr/include/" || { log "FATAL: header copy"; exit 1; }
else
  log "OpenCL headers already in sysroot"
fi

# 2) OpenCL ICD loader (libOpenCL.so to link against; at runtime the vendor lib is used)
ICD_OUT=$SYSROOT/usr/lib/aarch64-linux-android/libOpenCL.so
if [ ! -f "$ICD_OUT" ]; then
  log "=== OpenCL-ICD-Loader ==="
  git clone --depth 1 https://github.com/KhronosGroup/OpenCL-ICD-Loader "$WORK/OpenCL-ICD-Loader" >>"$LOG" 2>&1
  cd "$WORK/OpenCL-ICD-Loader"
  cmake -B b $NINJA -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DOPENCL_ICD_LOADER_HEADERS_DIR=$SYSROOT/usr/include \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=24 -DANDROID_STL=c++_shared >>"$LOG" 2>&1 \
    || { log "FATAL: ICD cmake"; exit 1; }
  cmake --build b >>"$LOG" 2>&1 || { log "FATAL: ICD build"; exit 1; }
  cp b/libOpenCL.so "$ICD_OUT" || { log "FATAL: ICD install"; exit 1; }
else
  log "ICD loader already installed"
fi
log "ICD: $(ls -la "$ICD_OUT" | awk '{print $5}') bytes"

# 3) llama.cpp with OpenCL + optimized CPU
cd "$SRC"
log "=== llama.cpp cmake (OpenCL + armv8.2 dotprod+i8mm+fp16 CPU) ==="
cmake -B bgpu $NINJA \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF \
  -DGGML_OPENCL=ON \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv8.2-a+dotprod+i8mm+fp16" >>"$LOG" 2>&1 \
  || { log "FATAL: llama cmake"; tail -20 "$LOG"; exit 1; }
cmake --build bgpu --target llama-bench llama-cli llama-mtmd-cli -j8 >>"$LOG" 2>&1 \
  || { log "FATAL: llama build"; tail -20 "$LOG"; exit 1; }
ls -la bgpu/bin/llama-bench bgpu/bin/llama-cli bgpu/bin/llama-mtmd-cli 2>/dev/null | tee -a "$LOG"
log "### GPU BUILD DONE ###"
