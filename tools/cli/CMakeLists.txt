# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

project(
    ntc-cli
    DESCRIPTION "Neural Texture Compression Command Line Interface"
    LANGUAGES CXX
)

set (OPTIX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/optix/include")
set (LIBNTC_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libraries/RTXNTC-Library/include")

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 

# Compile a CUDA file to PTX using NVCC
function ( NVCC_COMPILE cuda_src extra_headers ptx_generated extra_nvcc_args )
    get_filename_component ( cuda_src_we ${cuda_src} NAME_WE )
    get_filename_component ( cuda_src_dir ${cuda_src} DIRECTORY )
    set (cuda_ptx "${CMAKE_CURRENT_BINARY_DIR}/${cuda_src_we}.ptx" )
    set (cuda_ptx_h "${CMAKE_CURRENT_BINARY_DIR}/${cuda_src_we}Ptx.h")
    set (${ptxlist} ${${ptxlist}} ${cuda_ptx} )
    set (${ptx_generated} ${cuda_ptx} PARENT_SCOPE)
    set(PTX_GENERATED_PATH "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "")
    file (GLOB cuda_headers "${cuda_src_dir}/*.h")
    list (APPEND cuda_headers ${extra_headers})
    find_program(BIN_TO_C bin2c)

    message (STATUS "CUDAToolkit_NVCC_EXECUTABLE: ${CUDAToolkit_NVCC_EXECUTABLE}")

    add_custom_command (OUTPUT ${cuda_ptx}
        COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
            "-I${LIBNTC_INCLUDE_DIR}"
            "-I${OPTIX_INCLUDE_DIR}"
            -m64 
            -ptx
            --std=c++17
            -dc --use_fast_math
            --expt-relaxed-constexpr
            ${cuda_src} -o ${cuda_ptx}
        COMMAND ${BIN_TO_C} --type int --name CUDA_PTX --const ${cuda_ptx} > ${cuda_ptx_h}
        MAIN_DEPENDENCY ${cuda_src}
        DEPENDS ${cuda_src} ${cuda_headers}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endfunction ()

add_executable(ntc-cli)

target_compile_definitions(ntc-cli PRIVATE
    NTC_WITH_OPTIX=$<BOOL:${NTC_WITH_OPTIX}>)

target_include_directories(ntc-cli PRIVATE
    ${PTX_GENERATED_PATH} ${CMAKE_CURRENT_BINARY_DIR} ${OPTIX_INCLUDE_DIR})

set_target_properties(ntc-cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY                "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${NTC_BINARY_DIR}"
)

target_link_libraries(ntc-cli PRIVATE CUDA::cudart_static CUDA::cuda_driver libntc stb tinyexr argparse donut_app donut_engine lodepng)
target_link_libraries(ntc-cli PRIVATE ntc-utils)

target_sources(ntc-cli PRIVATE 
    NtcCommandLine.cpp
    GraphicsPasses.cpp
    GraphicsPasses.h
    Utils.cpp
    Utils.h
)

if (NTC_WITH_OPTIX)
    # OptixDecompressor.cpp includes the generated PTX header and OptiX headers
    # unconditionally, so only compile it when OptiX is enabled.
    target_sources(ntc-cli PRIVATE
        OptixDecompressionKernel.h
        OptixDecompressor.cpp
        OptixDecompressor.h)

    set (cuda_src
        OptixDecompressionKernel.cu
        )

    set (ntc_cuda_headers
        OptixDecompressionKernel.h
        "${LIBNTC_INCLUDE_DIR}/libntc/shaders/InferenceDataOptix.h"
        "${LIBNTC_INCLUDE_DIR}/libntc/shaders/InferenceOptix.h"
        )

    NVCC_COMPILE ( "${cuda_src}" "${ntc_cuda_headers}" ptx_generated "" )
    message (STATUS "ptx_generated: ${ptx_generated}")
    add_definitions( -DCUDA_PTX_FILE="${ptx_generated}" )
    list (APPEND ptx_list ${ptx_generated})

    add_custom_target (ntc_ptx ALL
        DEPENDS ${ptx_list} ${ntc_cuda_headers} ${cuda_ptx} ${cuda_ptx_h}
        SOURCES ${cuda_src}
        )

    install (FILES ${ptx_list}
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

    add_dependencies(ntc-cli ntc_ptx)
endif()

if (DONUT_WITH_DX12)
    add_dependencies(ntc-cli dx12-agility-sdk)
endif()
