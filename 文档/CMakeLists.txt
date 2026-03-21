cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(inference)

# ================== C++ 标准 ==================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ================== ONNX Runtime GPU 配置 ==================
# 确保这是 onnxruntime-win-x64-gpu-1.23.2
set(ONNXRUNTIME_DIR "D:/inference_cpp/onnxruntime-win-x64-gpu-1.23.2")
set(ONNXRUNTIME_INCLUDE "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

message(STATUS "ONNX Runtime Include: ${ONNXRUNTIME_INCLUDE}")
message(STATUS "ONNX Runtime Lib Dir: ${ONNXRUNTIME_LIB_DIR}")

# ================== cuDNN 9.x 配置 ==================
# 指向 cuDNN bin/12.9 目录
set(CUDNN_BIN_DIR
    "C:/Program Files/NVIDIA/CUDNN/v9.17/bin/12.9"
)

message(STATUS "cuDNN Bin Dir: ${CUDNN_BIN_DIR}")

# ================== 头文件 ==================
include_directories(${ONNXRUNTIME_INCLUDE})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ================== 库目录 ==================
link_directories(${ONNXRUNTIME_LIB_DIR})

# ================== 可执行文件 ==================
add_executable(inference inference.cpp)

# GPU / CPU 统一链接 onnxruntime.lib
target_link_libraries(inference onnxruntime)

# ================== 编码（解决 C4819） ==================
add_compile_options(/utf-8)

# ================== 输出目录 ==================
set_target_properties(inference PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ================== Windows：复制运行时 DLL ==================
if (WIN32)

    # ---- 1. 复制 ONNX Runtime GPU DLL ----
    add_custom_command(TARGET inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll"
            $<TARGET_FILE_DIR:inference>

        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_LIB_DIR}/onnxruntime_providers_shared.dll"
            $<TARGET_FILE_DIR:inference>

        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_LIB_DIR}/onnxruntime_providers_cuda.dll"
            $<TARGET_FILE_DIR:inference>
    )

    # ---- 2. 复制 cuDNN 9.x 全部 DLL（关键） ----
    file(GLOB CUDNN_DLLS
        "${CUDNN_BIN_DIR}/cudnn*.dll"
    )

    add_custom_command(TARGET inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CUDNN_DLLS}
            $<TARGET_FILE_DIR:inference>
    )

endif()

# ================== VS 调试时 CUDA PATH（可选） ==================
# cuDNN DLL 已在 exe 同目录，这里只需要 CUDA Runtime
if (WIN32)
    set_target_properties(inference PROPERTIES
        VS_DEBUGGER_ENVIRONMENT
        "PATH=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin;%PATH%"
    )
endif()
