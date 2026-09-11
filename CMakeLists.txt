cmake_minimum_required(VERSION 3.16.0)
project(nll_loss_op_prj)
find_package(ASC REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ARCH32_COMPUTE_UNITS ascend910b ascend910_93)
set(ARCH35_COMPUTE_UNITS ascend950)

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ${ARCH32_COMPUTE_UNITS} ${ARCH35_COMPUTE_UNITS})
endif()
set(package_name nll_loss_custom)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ASCEND_ARCH_DIR x86_64-linux)
else()
    set(ASCEND_ARCH_DIR aarch64-linux)
endif()

 npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_host")
    add_subdirectory(op_host)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel")
    add_subdirectory(op_kernel)
endif()

message(WARNING "cmake 'make' does NOT build kernel binary by default. Use: bash build.sh --soc=<soc>")
