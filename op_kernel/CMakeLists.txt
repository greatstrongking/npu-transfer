file(GLOB_RECURSE ALL_KERNEL_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE OP
    KERNEL_DIR .
    KERNEL_FILE ${ALL_KERNEL_FILES}
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

# 本机 conda 无 /usr/include 时，bisheng 预编译找不到 libc/libstdc++ 头。只加到 kernel，避免污染 host g++ include_next。
set(_CONDA_SYSROOT_INC "")
if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/x86_64-conda-linux-gnu/sysroot/usr/include/stdio.h")
    set(_CONDA_SYSROOT_INC "$ENV{CONDA_PREFIX}/x86_64-conda-linux-gnu/sysroot/usr/include")
    set(_CONDA_ROOT "$ENV{CONDA_PREFIX}")
elseif(EXISTS "$ENV{HOME}/miniconda3/x86_64-conda-linux-gnu/sysroot/usr/include/stdio.h")
    set(_CONDA_SYSROOT_INC "$ENV{HOME}/miniconda3/x86_64-conda-linux-gnu/sysroot/usr/include")
    set(_CONDA_ROOT "$ENV{HOME}/miniconda3")
endif()
file(GLOB _CONDA_CXX_INC "${_CONDA_ROOT}/lib/gcc/x86_64-conda-linux-gnu/*/include/c++")
list(SORT _CONDA_CXX_INC)
list(REVERSE _CONDA_CXX_INC)
if(_CONDA_SYSROOT_INC)
    set(_KERNEL_ISYSTEM -isystem${_CONDA_SYSROOT_INC})
    if(_CONDA_CXX_INC)
        list(GET _CONDA_CXX_INC 0 _CXX_INC)
        list(APPEND _KERNEL_ISYSTEM -isystem${_CXX_INC} -isystem${_CXX_INC}/x86_64-conda-linux-gnu)
    endif()
    npu_op_kernel_options(ascendc_kernels NllLoss OPTIONS ${_KERNEL_ISYSTEM})
endif()

npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)
