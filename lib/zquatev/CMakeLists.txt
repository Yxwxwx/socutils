# zquatev: Toru Shiozaki's quaternion eigensolver, for Kramers-restricted
# spinor SCF (spinor_hf.KRHF).  Thin subdir CMake (pyscf/lib style); common
# configuration (build type, flags, BLAS/LAPACK, C++ standard, RPATH) comes
# from the parent lib/CMakeLists.txt.
#
# The csrc/ sources are a verbatim copy of zquatev and remain under
# BSD-2-Clause (see LICENSE and the per-file headers); the capi/ shim is part
# of socutils (Apache-2.0).
set(ZQUATEV_SRC
  csrc/zquatev.cc
  csrc/blocked.cc
  csrc/unblocked.cc
  csrc/transpose.cc
  capi/zquatev_capi.cc
)

add_library(zquatev SHARED ${ZQUATEV_SRC})
target_include_directories(zquatev PRIVATE csrc capi)
set_target_properties(zquatev PROPERTIES
  PREFIX "lib" OUTPUT_NAME "zquatev"
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND UNIX AND NOT APPLE)
  # The site Python may already have loaded an older libstdc++ (for example
  # from Anaconda).  Keep this small bundled C++ library independent of that
  # process-global runtime to avoid GLIBCXX symbol-version conflicts.
  set_property(TARGET zquatev APPEND_STRING PROPERTY LINK_FLAGS
               " -static-libstdc++ -static-libgcc")
endif()
target_link_libraries(zquatev PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
