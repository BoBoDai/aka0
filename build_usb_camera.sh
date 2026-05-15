#!/bin/bash

# 设置交叉编译工具链
export CC=riscv64-unknown-linux-musl-gcc
export CXX=riscv64-unknown-linux-musl-g++

# 设置SDK路径，请根据实际情况修改
TPU_SDK_PATH=${TPU_SDK_PATH:-"/home/junbo_dai/cvitek_tpu_sdk"}
OPENCV_PATH=${TPU_SDK_PATH}/opencv

echo "Using TPU_SDK_PATH: $TPU_SDK_PATH"
echo "Using OPENCV_PATH: $OPENCV_PATH"

# 检查 OpenCV videoio 模块
if [ ! -d "$OPENCV_PATH/lib" ]; then
    echo "Error: OPENCV_PATH/lib not found at $OPENCV_PATH"
    exit 1
fi

echo "Available OpenCV libs:"
ls $OPENCV_PATH/lib/ | grep opencv

# 清理之前的构建
rm -rf build_usb_camera
mkdir -p build_usb_camera
cd build_usb_camera

# 配置cmake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_C_FLAGS="-O2 -mcpu=c906fdv -mabi=lp64d" \
    -DCMAKE_CXX_FLAGS="-O2 -mcpu=c906fdv -mabi=lp64d -DUSE_USB_CAMERA=1" \
    -DCMAKE_CROSSCOMPILING=ON \
    -DTPU_SDK_PATH=${TPU_SDK_PATH} \
    -DOPENCV_PATH=${OPENCV_PATH} \
    -DUSE_USB_CAMERA=1 \
    -DUSE_ESP32_UART=1

# 编译
make -j$(nproc)

echo "USB camera test compiled successfully!"
echo "Executable: $(pwd)/usb_camera_test"
