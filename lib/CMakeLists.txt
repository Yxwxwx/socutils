# Top-level CMake for the socutils C/C++ libraries, modelled on
# pyscf/lib/CMakeLists.txt.
#
# It centralises the common build configuration -- build type, warning and
# optimisation flags, BLAS/LAPACK and OpenMP discovery, and an $ORIGIN RPATH --
# then delegates each shared library to its own subdirectory through
# add_subdirectory().  Following the PySCF convention, every library is emitted
# flat into this directory (socutils/lib/) and loaded from there by
# ``socutils.lib.load_library``:
#
#   x2camf  ->  lib/libx2camf_c.so    (import x2camf, the SOC integral backend)
#   ccsdt   ->  lib/libccsdt_clib.so  (socutils.cc CCSDT kernels)
#   zquatev ->  lib/libzquatev.so     (Kramers-restricted spinor SCF)
#   ao2mo   ->  lib/libnrr_opt.so     (socutils.lib.ao2mo.nrr_fast spinor AO2MO)
#
# The subdirectory CMakeLists.txt files are thin (just the target plus its
# link/output), exactly as in pyscf/lib, so they slot into pyscf/lib unchanged
# if contributed upstream.
cmake_minimum_required(VERSION 3.5)
project(socutils C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()
set(CMAKE_VERBOSE_MAKEFILE OFF)

if(CMAKE_COMPILER_IS_GNUCC)
  if(UNIX AND NOT APPLE AND NOT DEFINED ENV{LDFLAGS})
    # Emit DT_RPATH instead of DT_RUNPATH so the PySCF wheel directory is
    # also searched for indirect OpenBLAS dependencies (notably its hashed
    # libgfortran).  RUNPATH is not transitive on glibc.
    set(CMAKE_SHARED_LINKER_FLAGS
        "-Wl,--no-as-needed -Wl,--disable-new-dtags ${CMAKE_SHARED_LINKER_FLAGS}")
  endif()
endif()
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "-O3 -funroll-loops")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -funroll-loops -g")
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Architecture-specific overrides (optional), as in pyscf.
if(EXISTS "${PROJECT_SOURCE_DIR}/cmake.arch.inc")
  include("${PROJECT_SOURCE_DIR}/cmake.arch.inc")
endif()

# --- BLAS / LAPACK -----------------------------------------------------------
# Honour a user-supplied BLAS_LIBRARIES (e.g. -DBLAS_LIBRARIES=mkl_rt or
# OpenBLAS); otherwise discover one.  -DBLA_VENDOR=OpenBLAS|Intel10_64lp|...
# selects a threaded vendor library.
set(_blas_user_provided TRUE)
if(NOT BLAS_LIBRARIES)
  set(_blas_user_provided FALSE)
  # Prefer the BLAS bundled with the selected PySCF wheel.  Loading a host MKL
  # here can otherwise conflict at runtime with the MKL version linked into an
  # official Block2 wheel.  An explicit BLAS_LIBRARIES still takes precedence.
  if(NOT PYTHON_EXECUTABLE)
    find_program(PYTHON_EXECUTABLE NAMES python3 python)
  endif()
  if(PYTHON_EXECUTABLE)
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c
              "from pyscf import lib; import os; print(os.path.dirname(os.path.abspath(lib.__file__)))"
      OUTPUT_VARIABLE PYSCF_LIB_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _pyscf_lib_rc)
  endif()
  if(_pyscf_lib_rc EQUAL 0 AND PYSCF_LIB_DIR)
    file(GLOB _pyscf_blas "${PYSCF_LIB_DIR}/libopenblas*.so")
  endif()
  if(_pyscf_blas)
    list(GET _pyscf_blas 0 BLAS_LIBRARIES)
    set(_blas_user_provided TRUE)
    message(STATUS "BLAS/LAPACK from selected PySCF: ${BLAS_LIBRARIES}")
  else()
    find_package(BLAS REQUIRED)
    message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
  endif()
endif()
if(NOT LAPACK_LIBRARIES)
  if(_blas_user_provided)
    # An explicitly supplied BLAS (mkl_rt, OpenBLAS) already provides LAPACK;
    # do not drag in a separate reference LAPACK.
    set(LAPACK_LIBRARIES "")
    message(STATUS "Using user BLAS_LIBRARIES for LAPACK too: ${BLAS_LIBRARIES}")
  else()
    find_package(LAPACK REQUIRED)
    message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
  endif()
endif()

# --- OpenMP ------------------------------------------------------------------
option(ENABLE_OPENMP "Compile with OpenMP" ON)
set(OPENMP_C_PROPERTIES "")
if(ENABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(HAVE_OPENMP 1)
    set(OPENMP_C_PROPERTIES OpenMP::OpenMP_C)
  endif()
endif()

# --- RPATH: each .so locates its (possibly bundled) deps next to itself ------
if(WIN32)
  # nothing
elseif(APPLE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/deps/lib")
else()
  set(CMAKE_SKIP_BUILD_RPATH FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/deps/lib:${PYSCF_LIB_DIR}")
endif()

add_subdirectory(x2camf)
add_subdirectory(ccsdt)
add_subdirectory(zquatev)
add_subdirectory(ao2mo)
