// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cstdint>
#include <dlfcn.h>

struct Wrapped {
  PyMemAllocatorEx original;
  uint64_t calls;
};
static Wrapped allocators[3];
static thread_local bool active = false;
static void* counted_malloc(void* ctx, size_t n) {
  auto* w = (Wrapped*)ctx;
  if (active) ++w->calls;
  return w->original.malloc(w->original.ctx, n);
}
static void* counted_calloc(void* ctx, size_t n, size_t size) {
  auto* w = (Wrapped*)ctx;
  if (active) ++w->calls;
  return w->original.calloc(w->original.ctx, n, size);
}
static void* counted_realloc(void* ctx, void* ptr, size_t size) {
  auto* w = (Wrapped*)ctx;
  if (active) ++w->calls;
  return w->original.realloc(w->original.ctx, ptr, size);
}
static void counted_free(void* ctx, void* ptr) {
  auto* w = (Wrapped*)ctx;
  w->original.free(w->original.ctx, ptr);
}
static PyObject* measure(PyObject*, PyObject* args) {
  PyObject *callable, *batches;
  if (!PyArg_ParseTuple(args, "OO", &callable, &batches)) return nullptr;
  if (!PyList_Check(batches)) {
    PyErr_SetString(PyExc_TypeError, "argument batches must be a list");
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(batches); ++i) {
    if (!PyTuple_Check(PyList_GET_ITEM(batches, i))) {
      PyErr_SetString(PyExc_TypeError, "argument batch must be a tuple");
      return nullptr;
    }
  }
  auto begin = (void (*)())dlsym(RTLD_DEFAULT, "kvprof_alloc_begin");
  auto end = (uint64_t (*)())dlsym(RTLD_DEFAULT, "kvprof_alloc_end");
  if (!begin || !end) {
    PyErr_SetString(PyExc_RuntimeError, "allocation interposer not loaded");
    return nullptr;
  }
  for (int i = 0; i < 3; ++i) {
    PyMem_GetAllocator((PyMemAllocatorDomain)i, &allocators[i].original);
    allocators[i].calls = 0;
    PyMemAllocatorEx hook{&allocators[i], counted_malloc, counted_calloc,
                          counted_realloc, counted_free};
    PyMem_SetAllocator((PyMemAllocatorDomain)i, &hook);
  }
  bool failed = false;
  active = false;  // Initialize measurement TLS before enabling libc counters.
  begin();
  active = true;
  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(batches); ++i) {
    PyObject* tuple = PyList_GET_ITEM(batches, i);
    PyObject* value =
        PyObject_Vectorcall(callable, ((PyTupleObject*)tuple)->ob_item,
                            PyTuple_GET_SIZE(tuple), nullptr);
    if (!value) {
      failed = true;
      break;
    }
    Py_DECREF(value);
  }
  active = false;
  uint64_t libc_calls = end();
  for (int i = 0; i < 3; ++i)
    PyMem_SetAllocator((PyMemAllocatorDomain)i, &allocators[i].original);
  if (failed) return nullptr;
  return Py_BuildValue("(KKKK)", (unsigned long long)allocators[0].calls,
                       (unsigned long long)allocators[1].calls,
                       (unsigned long long)allocators[2].calls,
                       (unsigned long long)libc_calls);
}
static PyMethodDef methods[] = {{"measure", measure, METH_VARARGS, nullptr},
                                {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_profiler_allocations",
                             nullptr, -1, methods};
PyMODINIT_FUNC PyInit__profiler_allocations() {
  return PyModule_Create(&module);
}
