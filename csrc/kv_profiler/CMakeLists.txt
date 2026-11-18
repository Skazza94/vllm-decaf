# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
cmake_minimum_required(VERSION 3.26)
set(KV_PROFILER_STANDALONE FALSE)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
  set(KV_PROFILER_STANDALONE TRUE)
  project(kv_profiler_core LANGUAGES CXX)
  find_package(Python REQUIRED COMPONENTS Interpreter)
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "KV profiler requires Linux tmpfs")
endif()
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import sysconfig; print(1 if sysconfig.get_config_var('Py_GIL_DISABLED') else 0)"
  OUTPUT_VARIABLE KV_PROFILER_GIL_DISABLED
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE KV_PROFILER_PROBE_RESULT)
if(NOT KV_PROFILER_PROBE_RESULT EQUAL 0)
  message(FATAL_ERROR "Cannot determine KV profiler interpreter GIL mode")
endif()
if(KV_PROFILER_GIL_DISABLED)
  message(STATUS "Skipping KV profiler: free-threaded Python is not supported")
  return()
endif()
if(KV_PROFILER_STANDALONE)
  find_package(Python REQUIRED COMPONENTS Development.Module Development.SABIModule)
endif()
Python_add_library(_kv_profiler MODULE USE_SABI 3.11 WITH_SOABI
  "${CMAKE_CURRENT_LIST_DIR}/recorder.cpp")
target_compile_features(_kv_profiler PRIVATE cxx_std_20)
install(TARGETS _kv_profiler LIBRARY DESTINATION vllm COMPONENT _kv_profiler)
