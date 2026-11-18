// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "recorder.h"
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <new>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <time.h>
#include <unistd.h>

namespace {
constexpr uint64_t MAGIC = 0x3152464f5250564bULL;
constexpr uint64_t MAX_MEMORY = 256ULL << 20;
constexpr uint32_t MAX_ID = 512;
constexpr uint32_t CHUNK = 72;
constexpr uint32_t PROBES = 32;
enum Counter {
  ATTEMPTS,
  EVENTS,
  FULL,
  CONTENTION,
  IDS,
  HANDLES,
  BAD_ID,
  CONFLICT,
  CLOSED,
  UNKNOWN_HANDLE,
  COUNTERS
};
struct alignas(64) Header {
  uint64_t magic, memory, owner;
  uint32_t version, capacity, id_limit, id_bytes, handle_limit;
  uint64_t ids_offset, arena_offset, handles_offset;
  std::atomic<uint64_t> head{0}, tail{0};
  std::atomic_flag producer = ATOMIC_FLAG_INIT;
  std::atomic<uint64_t> counters[COUNTERS]{};
  std::atomic<uint32_t> closing{0}, writer_state{0}, writer_errno{0};
  std::atomic<uint64_t> id_count{0}, arena_used{0}, handle_count{0};
  uint64_t next_transfer = 1, active_handle_epoch = 0;
  std::atomic<uint64_t> handle_epoch{0};
  std::atomic<uint32_t> handles_disabled{0};
};
struct Record {
  uint64_t v[16];
};
struct Identity {
  uint64_t hash = 0, id = 0;
  uint32_t offset = 0, length = 0, encoding = 0, unused = 0;
};
struct Handle {
  uint64_t key = 0, id = 0, state = 0, epoch = 0;
};
static_assert(sizeof(Record) == 128);
static_assert(sizeof(Header) <= 4096);
constexpr size_t START = 4096;
uint64_t now_ns() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ULL + t.tv_nsec;
}
uint64_t mix(uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
uint64_t hash_id(const KvprofIdentity& v) {
  uint64_t h = 1469598103934665603ULL ^ v.encoding;
  for (uint32_t i = 0; i < v.length; ++i)
    h = (h ^ uint8_t(v.data[i])) * 1099511628211ULL;
  return mix(h);
}
Record* records(Header* h) { return (Record*)((char*)h + START); }
Identity* ids(Header* h) { return (Identity*)((char*)h + h->ids_offset); }
char* arena(Header* h) { return (char*)h + h->arena_offset; }
Handle* handles(Header* h) { return (Handle*)((char*)h + h->handles_offset); }
int drop(Header* h, Counter c) {
  h->counters[c].fetch_add(1, std::memory_order_relaxed);
  return int(c);
}
struct Guard {
  Header* h;
  ~Guard() { h->producer.clear(std::memory_order_release); }
};
bool same(const KvprofIdentity& a, const KvprofIdentity& b) {
  return a.length == b.length && a.encoding == b.encoding &&
         !memcmp(a.data, b.data, a.length);
}
}  // namespace

extern "C" int kvprof_record(void* mapping, const KvprofEvent* e) {
  auto* h = (Header*)mapping;
  if (h->owner != (uint64_t)getpid()) return -1;
  h->counters[ATTEMPTS].fetch_add(1, std::memory_order_relaxed);
  if (e->kind < 2 || e->kind > 21 || e->kind == 20 || e->transport > 3 ||
      e->status > 255 || (e->valid & ~127u) ||
      ((e->kind == 6 || e->kind == 21) && !e->has_handle))
    return drop(h, BAD_ID);
  if (h->closing.load(std::memory_order_acquire)) return drop(h, CLOSED);
  uint64_t entered_epoch = h->handle_epoch.load(std::memory_order_acquire);
  if (h->producer.test_and_set(std::memory_order_acquire)) {
    if (e->has_handle && (e->kind == 6 || e->kind == 21)) {
      // A lost lifecycle boundary invalidates every earlier association.
      if (h->handle_epoch.fetch_add(1, std::memory_order_acq_rel) == UINT64_MAX)
        h->handles_disabled.store(1, std::memory_order_release);
    }
    return drop(h, CONTENTION);
  }
  Guard guard{h};
  if (h->closing.load(std::memory_order_acquire)) return drop(h, CLOSED);
  uint64_t epoch = h->handle_epoch.load(std::memory_order_acquire);
  bool uncertain = entered_epoch != epoch ||
                   h->handles_disabled.load(std::memory_order_acquire);
  Handle* hs = nullptr;
  uint64_t transfer = 0;
  bool bind = e->kind == 6, retire = e->kind == 21;
  if (h->active_handle_epoch != epoch) {
    h->active_handle_epoch = epoch;
    h->handle_count = 0;
  }
  if (e->has_handle && !uncertain) {
    Handle* vacant = nullptr;
    uint64_t hash = mix(e->handle);
    for (uint32_t i = 0; i < PROBES; ++i) {
      auto* c = &handles(h)[(hash + i) & (2 * h->handle_limit - 1)];
      uint64_t state = c->epoch == epoch ? c->state : 0;
      if (state == 1 && c->key == e->handle) {
        hs = c;
        break;
      }
      if (state != 1 && !vacant) vacant = c;
      if (!state) break;
    }
    if (bind) {
      if (hs) {
        hs->state = 2;
        --h->handle_count;
        return drop(h, CONFLICT);
      }
      if (!vacant || h->handle_count == h->handle_limit ||
          h->next_transfer == UINT64_MAX)
        return drop(h, HANDLES);
      hs = vacant;
      transfer = h->next_transfer;
    } else if (hs)
      transfer = hs->id;
    else
      h->counters[UNKNOWN_HANDLE].fetch_add(1, std::memory_order_relaxed);
  }
  if (e->has_handle && uncertain)
    h->counters[UNKNOWN_HANDLE].fetch_add(1, std::memory_order_relaxed);
  struct Retirement {
    Header* h;
    Handle* slot;
    bool retire;
    ~Retirement() {
      if (retire && slot) {
        slot->state = 2;
        --h->handle_count;
      }
    }
  } retirement{h, hs, retire};
  for (auto& s : e->identities)
    if (s.data && (s.length > MAX_ID || s.encoding > 1)) return drop(h, BAD_ID);

  Identity* slots[3]{};
  KvprofIdentity pending[3]{};
  uint64_t resolved[3]{}, hashes[3]{};
  uint32_t count = 0, bytes = 0, needed = 1;
  for (int i = 0; i < 3; ++i) {
    auto& s = e->identities[i];
    if (!s.data) continue;
    bool duplicate = false;
    for (uint32_t j = 0; j < count; ++j) {
      if (same(s, pending[j])) {
        resolved[i] = h->id_count + j + 1;
        duplicate = true;
        break;
      }
    }
    if (duplicate) continue;
    uint64_t hash = hash_id(s);
    Identity* slot = nullptr;
    for (uint32_t n = 0; n < PROBES; ++n) {
      auto* candidate = &ids(h)[(hash + n) & (2 * h->id_limit - 1)];
      bool reserved = false;
      for (uint32_t j = 0; j < count; ++j) reserved |= slots[j] == candidate;
      if (reserved) continue;
      if (!candidate->id) {
        slot = candidate;
        break;
      }
      if (candidate->hash == hash && candidate->length == s.length &&
          candidate->encoding == s.encoding &&
          !memcmp(arena(h) + candidate->offset, s.data, s.length)) {
        resolved[i] = candidate->id;
        break;
      }
    }
    if (resolved[i]) continue;
    if (!slot || h->id_count + count == h->id_limit) return drop(h, IDS);
    slots[count] = slot;
    hashes[count] = hash;
    pending[count] = s;
    resolved[i] = h->id_count + count + 1;
    ++count;
    bytes += s.length;
    needed += (s.length + CHUNK - 1) / CHUNK + (s.length == 0);
  }
  if (h->arena_used + bytes > h->id_bytes) return drop(h, IDS);

  uint64_t head = h->head.load(std::memory_order_relaxed);
  uint64_t tail = h->tail.load(std::memory_order_acquire);
  if (head > UINT64_MAX - needed || head - tail + needed > h->capacity) {
    return drop(h, FULL);
  }
  uint64_t stamp = now_ns();
  auto next = [&]() -> Record& {
    Record& r = records(h)[head & (h->capacity - 1)];
    memset(&r, 0, sizeof(r));
    r.v[0] = stamp;
    r.v[1] = ++head;
    return r;
  };
  // Definitions and the event commit with one release-store of head.
  for (uint32_t j = 0; j < count; ++j) {
    auto& s = pending[j];
    uint64_t id = ++h->id_count;
    memcpy(arena(h) + h->arena_used, s.data, s.length);
    *slots[j] = {hashes[j], id,         uint32_t(h->arena_used),
                 s.length,  s.encoding, 0};
    h->arena_used += s.length;
    uint32_t offset = 0;
    do {
      auto& r = next();
      uint32_t n = s.length - offset < CHUNK ? s.length - offset : CHUNK;
      r.v[2] = (1ULL << 48) | (1ULL << 32);
      r.v[3] = id;
      r.v[4] = s.length;
      r.v[5] = offset;
      r.v[6] = (uint64_t(s.encoding) << 32) | n;
      memcpy((char*)&r + 56, s.data + offset, n);
      offset += n;
    } while (offset < s.length);
  }
  if (bind && hs) {
    *hs = {e->handle, h->next_transfer++, 1, epoch};
    ++h->handle_count;
  }
  auto& r = next();
  r.v[2] = (1ULL << 48) | (e->kind << 32) | (e->transport << 24) |
           (e->status << 16) | e->valid |
           ((e->has_handle && !transfer) ? (1ULL << 15) : 0);
  r.v[3] = e->context;
  r.v[4] = resolved[0];
  r.v[5] = resolved[1];
  r.v[6] = transfer;
  r.v[7] = resolved[2];
  r.v[8] = e->subject;
  memcpy(r.v + 9, e->values, sizeof(e->values));
  h->counters[EVENTS].fetch_add(1, std::memory_order_relaxed);
  h->head.store(head, std::memory_order_release);
  return 0;
}

namespace {
struct Ring {
  PyObject_HEAD Header* h;
  size_t size;
  int fd;
  bool producer;
  bool reader_lease;
};
PyTypeObject* ring_type = nullptr;
bool power(uint32_t n) { return n >= 2 && !(n & (n - 1)); }
PyObject* error() { return PyErr_SetFromErrno(PyExc_OSError); }
PyObject* create(PyObject*, PyObject* args) {
  const char* path;
  unsigned cap, limit, bytes, handle_limit;
  int require_tmpfs = 1;
  if (!PyArg_ParseTuple(args, "sIIII|p", &path, &cap, &limit, &bytes,
                        &handle_limit, &require_tmpfs))
    return nullptr;
  if (!power(cap) || !power(limit) || !power(handle_limit) || !bytes ||
      cap > (1u << 20) || limit > 65536 || handle_limit > 65536) {
    PyErr_SetString(PyExc_ValueError, "invalid profiler capacity");
    return nullptr;
  }
  size_t io = START + size_t(cap) * 128;
  size_t ao = io + size_t(2 * limit) * sizeof(Identity);
  size_t ho = (ao + bytes + 63) & ~size_t(63);
  size_t total = ho + size_t(2 * handle_limit) * sizeof(Handle);
  if (total > MAX_MEMORY) {
    PyErr_SetString(PyExc_ValueError, "profiler memory exceeds 256 MiB");
    return nullptr;
  }
#ifdef Py_GIL_DISABLED
  PyErr_SetString(PyExc_RuntimeError, "profiler requires GIL-enabled CPython");
  return nullptr;
#endif
  int fd = open(path, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd < 0) return error();
  struct statfs fs;
  if (fstatfs(fd, &fs) || (require_tmpfs && fs.f_type != 0x01021994)) {
    close(fd);
    unlink(path);
    PyErr_SetString(PyExc_ValueError, "profiler ring requires tmpfs");
    return nullptr;
  }
  if (ftruncate(fd, total)) {
    close(fd);
    unlink(path);
    return error();
  }
  int reserved = posix_fallocate(fd, 0, total);
  if (reserved) {
    close(fd);
    unlink(path);
    errno = reserved;
    return error();
  }
  void* ptr = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    close(fd);
    unlink(path);
    return error();
  }
  memset(ptr, 0, total);  // Prefault before recording starts.
  auto* h = new (ptr) Header{};
  h->magic = MAGIC;
  h->version = 2;
  h->memory = total;
  h->owner = getpid();
  h->capacity = cap;
  h->id_limit = limit;
  h->id_bytes = bytes;
  h->handle_limit = handle_limit;
  h->ids_offset = io;
  h->arena_offset = ao;
  h->handles_offset = ho;
  for (uint32_t i = 0; i < cap; ++i) new (&records(h)[i]) Record{};
  for (uint32_t i = 0; i < 2 * limit; ++i) new (&ids(h)[i]) Identity{};
  for (uint32_t i = 0; i < 2 * handle_limit; ++i) new (&handles(h)[i]) Handle{};
  if (!h->head.is_lock_free() || !h->counters[0].is_lock_free()) {
    munmap(ptr, total);
    close(fd);
    unlink(path);
    PyErr_SetString(PyExc_RuntimeError, "profiler requires lock-free atomics");
    return nullptr;
  }
  now_ns();
  auto* out = PyObject_New(Ring, ring_type);
  if (!out) {
    munmap(ptr, total);
    close(fd);
    unlink(path);
    return nullptr;
  }
  out->h = h;
  out->size = total;
  out->fd = fd;
  out->producer = true;
  out->reader_lease = false;
  return (PyObject*)out;
}
PyObject* attach(PyObject*, PyObject* arg) {
  const char* path = PyUnicode_AsUTF8AndSize(arg, nullptr);
  if (!path) return nullptr;
  int fd = open(path, O_RDWR | O_CLOEXEC);
  if (fd < 0) return error();
  struct stat st;
  if (fstat(fd, &st)) {
    close(fd);
    return error();
  }
  if (st.st_size < (off_t)START || st.st_size > (off_t)MAX_MEMORY) {
    close(fd);
    PyErr_SetString(PyExc_ValueError, "invalid ring size");
    return nullptr;
  }
  void* ptr =
      mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    close(fd);
    return error();
  }
  auto* h = (Header*)ptr;
  if (h->magic != MAGIC || h->version != 2 ||
      h->memory != (uint64_t)st.st_size || !power(h->capacity) ||
      START + uint64_t(h->capacity) * 128 > h->memory) {
    munmap(ptr, st.st_size);
    close(fd);
    PyErr_SetString(PyExc_ValueError, "invalid profiler ring");
    return nullptr;
  }
  auto* out = PyObject_New(Ring, ring_type);
  if (!out) {
    munmap(ptr, st.st_size);
    close(fd);
    return nullptr;
  }
  out->h = h;
  out->fd = fd;
  out->size = st.st_size;
  out->producer = false;
  out->reader_lease = false;
  return (PyObject*)out;
}
bool number(PyObject* p, uint64_t& out) {
  if (!PyLong_Check(p) || PyBool_Check(p)) {
    PyErr_SetString(PyExc_TypeError, "profiler fields must be integers");
    return false;
  }
  out = PyLong_AsUnsignedLongLong(p);
  return !PyErr_Occurred();
}
bool encode(PyObject* p, char* buffer, KvprofIdentity& out) {
  if (p == Py_None) {
    out = {nullptr, 0, 0};
    return true;
  }
  if (PyBytes_Check(p)) {
    out = {PyBytes_AsString(p), uint32_t(PyBytes_Size(p)), 0};
    if (PyBytes_Size(p) > MAX_ID) out.length = MAX_ID + 1;
    return true;
  }
  if (!PyUnicode_Check(p)) {
    PyErr_SetString(PyExc_TypeError, "identity must be str, bytes or None");
    return false;
  }
  uint32_t n = 0;
  Py_ssize_t length = PyUnicode_GetLength(p);
  for (Py_ssize_t i = 0; i < length; ++i) {
    Py_UCS4 ch = PyUnicode_ReadChar(p, i);
    uint32_t width = ch < 0x80 ? 1 : ch < 0x800 ? 2 : ch < 0x10000 ? 3 : 4;
    if (n + width > MAX_ID || (ch >= 0xd800 && ch <= 0xdfff)) {
      out = {buffer, MAX_ID + 1, 1};
      return true;
    }
    if (ch < 0x80)
      buffer[n++] = char(ch);
    else if (ch < 0x800) {
      buffer[n++] = char(0xc0 | (ch >> 6));
      buffer[n++] = char(0x80 | (ch & 63));
    } else if (ch < 0x10000) {
      buffer[n++] = char(0xe0 | (ch >> 12));
      buffer[n++] = char(0x80 | ((ch >> 6) & 63));
      buffer[n++] = char(0x80 | (ch & 63));
    } else {
      buffer[n++] = char(0xf0 | (ch >> 18));
      buffer[n++] = char(0x80 | ((ch >> 12) & 63));
      buffer[n++] = char(0x80 | ((ch >> 6) & 63));
      buffer[n++] = char(0x80 | (ch & 63));
    }
  }
  out = {buffer, n, 1};
  return true;
}
PyObject* record(Ring* self, PyObject* const* args, Py_ssize_t n) {
  if (!self->h) Py_RETURN_NONE;
  if (!self->producer) {
    PyErr_SetString(PyExc_RuntimeError, "only the creating process may record");
    return nullptr;
  }
  if (n != 16) {
    PyErr_SetString(PyExc_TypeError, "record requires 16 positional fields");
    return nullptr;
  }
  KvprofEvent e{};
  char strings[3][MAX_ID];
  if (!number(args[0], e.kind) || !number(args[1], e.context)) return nullptr;
  if (e.kind < 2 || e.kind > 21 || e.kind == 20) {
    PyErr_SetString(PyExc_ValueError, "invalid event kind");
    return nullptr;
  }
  for (int i = 0; i < 3; ++i)
    if (!encode(args[i + 2], strings[i], e.identities[i])) return nullptr;
  e.has_handle = args[5] != Py_None;
  if (e.has_handle && !number(args[5], e.handle)) return nullptr;
  if ((e.kind == 6 || e.kind == 21) && !e.has_handle) {
    PyErr_SetString(PyExc_ValueError, "bind/retire requires a handle");
    return nullptr;
  }
  for (int i = 0; i < 7; ++i) {
    if (args[6 + i] == Py_None) continue;
    if (!number(args[6 + i], e.values[i])) return nullptr;
    e.valid |= 1u << i;
  }
  if (!number(args[13], e.transport) || !number(args[14], e.status) ||
      !number(args[15], e.subject))
    return nullptr;
  if (e.transport > 3 || e.status > 255) {
    PyErr_SetString(PyExc_ValueError, "invalid transport or status");
    return nullptr;
  }
  if (kvprof_record(self->h, &e) == -1) {
    PyErr_SetString(PyExc_RuntimeError, "ring belongs to another process");
    return nullptr;
  }
  Py_RETURN_NONE;
}
const char* counter_names[] = {
    "attempts",    "events",         "drop_full",     "drop_contention",
    "drop_ids",    "drop_handles",   "drop_identity", "drop_conflict",
    "drop_closed", "unknown_handles"};
PyObject* stats(Ring* self, PyObject*) {
  if (!self->h) {
    PyErr_SetString(PyExc_ValueError, "closed ring");
    return nullptr;
  }
  auto* d = PyDict_New();
  if (!d) return nullptr;
  auto add = [&](const char* key, uint64_t v) {
    auto* p = PyLong_FromUnsignedLongLong(v);
    if (!p) return false;
    int rc = PyDict_SetItemString(d, key, p);
    Py_DECREF(p);
    return rc == 0;
  };
  auto* h = self->h;
  for (int i = 0; i < COUNTERS; ++i)
    if (!add(counter_names[i], h->counters[i].load())) {
      Py_DECREF(d);
      return nullptr;
    }
  if (!add("head", h->head.load()) || !add("tail", h->tail.load()) ||
      !add("ids_used", h->id_count.load()) ||
      !add("id_bytes_used", h->arena_used.load()) ||
      !add("live_handles", h->handle_count.load()) ||
      !add("handle_epoch", h->handle_epoch.load()) ||
      !add("handles_disabled", h->handles_disabled.load()) ||
      !add("capacity", h->capacity) || !add("memory_bytes", h->memory) ||
      !add("closing", h->closing.load()) ||
      !add("writer_state", h->writer_state.load()) ||
      !add("writer_errno", h->writer_errno.load())) {
    Py_DECREF(d);
    return nullptr;
  }
  return d;
}
PyObject* peek(Ring* self, PyObject* arg) {
  if (!self->h) {
    PyErr_SetString(PyExc_ValueError, "closed ring");
    return nullptr;
  }
  if (!self->reader_lease) {
    PyErr_SetString(PyExc_RuntimeError, "reader lease required");
    return nullptr;
  }
  unsigned long limit = PyLong_AsUnsignedLong(arg);
  if (PyErr_Occurred()) return nullptr;
  if (!limit || limit > 4096) {
    PyErr_SetString(PyExc_ValueError, "peek limit must be 1..4096");
    return nullptr;
  }
  auto* h = self->h;
  uint64_t tail = h->tail.load(std::memory_order_acquire);
  uint64_t head = h->head.load(std::memory_order_acquire);
  if (head < tail || head - tail > h->capacity) {
    PyErr_SetString(PyExc_ValueError, "invalid ring cursors");
    return nullptr;
  }
  uint64_t n = head - tail < limit ? head - tail : limit;
  auto* data = PyBytes_FromStringAndSize(nullptr, n * 128);
  if (!data) return nullptr;
  for (uint64_t i = 0; i < n; ++i)
    memcpy(PyBytes_AsString(data) + i * 128,
           &records(h)[(tail + i) & (h->capacity - 1)], 128);
  return Py_BuildValue("(KKN)", (unsigned long long)tail,
                       (unsigned long long)(tail + n), data);
}
PyObject* ack(Ring* self, PyObject* args) {
  unsigned long long begin, end;
  if (!PyArg_ParseTuple(args, "KK", &begin, &end)) return nullptr;
  if (!self->h) {
    PyErr_SetString(PyExc_ValueError, "closed ring");
    return nullptr;
  }
  if (!self->reader_lease) {
    PyErr_SetString(PyExc_RuntimeError, "reader lease required");
    return nullptr;
  }
  if (begin != self->h->tail.load() || end < begin ||
      end > self->h->head.load(std::memory_order_acquire)) {
    PyErr_SetString(PyExc_ValueError, "stale or invalid acknowledgement");
    return nullptr;
  }
  uint64_t expected = begin;
  if (!self->h->tail.compare_exchange_strong(expected, end)) {
    PyErr_SetString(PyExc_ValueError, "stale acknowledgement");
    return nullptr;
  }
  Py_RETURN_NONE;
}
PyObject* acquire_reader(Ring* self, PyObject*) {
  if (!self->h) {
    PyErr_SetString(PyExc_ValueError, "closed ring");
    return nullptr;
  }
  if (!self->reader_lease) {
    if (flock(self->fd, LOCK_EX | LOCK_NB)) {
      if (errno == EWOULDBLOCK) Py_RETURN_FALSE;
      return error();
    }
    self->reader_lease = true;
  }
  Py_RETURN_TRUE;
}
PyObject* finish(Ring* self, PyObject*) {
  if (self->h) self->h->closing.store(1, std::memory_order_release);
  Py_RETURN_NONE;
}
PyObject* writer_state(Ring* self, PyObject* args) {
  unsigned state, err = 0;
  if (!PyArg_ParseTuple(args, "I|I", &state, &err)) return nullptr;
  if (!self->h) {
    PyErr_SetString(PyExc_ValueError, "closed ring");
    return nullptr;
  }
  self->h->writer_errno.store(err);
  self->h->writer_state.store(state, std::memory_order_release);
  Py_RETURN_NONE;
}
PyObject* quiescent(Ring* self, PyObject*) {
  if (!self->h) Py_RETURN_TRUE;
  return PyBool_FromLong(!self->h->producer.test(std::memory_order_acquire));
}
PyObject* address(Ring* self, PyObject*) { return PyLong_FromVoidPtr(self->h); }
void unmap(Ring* self) {
  if (self->h) {
    munmap(self->h, self->size);
    close(self->fd);
    self->h = nullptr;
  }
}
PyObject* close_ring(Ring* self, PyObject*) {
  unmap(self);
  Py_RETURN_NONE;
}
void destroy(Ring* self) {
  unmap(self);
  auto* type = Py_TYPE((PyObject*)self);
  PyObject_Del(self);
  Py_DECREF(type);
}
PyMethodDef methods[] = {
    {"record", (PyCFunction)record, METH_FASTCALL, nullptr},
    {"stats", (PyCFunction)stats, METH_NOARGS, nullptr},
    {"acquire_reader", (PyCFunction)acquire_reader, METH_NOARGS, nullptr},
    {"peek", (PyCFunction)peek, METH_O, nullptr},
    {"ack", (PyCFunction)ack, METH_VARARGS, nullptr},
    {"finish", (PyCFunction)finish, METH_NOARGS, nullptr},
    {"set_writer_state", (PyCFunction)writer_state, METH_VARARGS, nullptr},
    {"quiescent", (PyCFunction)quiescent, METH_NOARGS, nullptr},
    {"address", (PyCFunction)address, METH_NOARGS, nullptr},
    {"close", (PyCFunction)close_ring, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};
PyMethodDef functions[] = {{"create", create, METH_VARARGS, nullptr},
                           {"attach", attach, METH_O, nullptr},
                           {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_kv_profiler", nullptr, -1,
                      functions};
}  // namespace
PyMODINIT_FUNC PyInit__kv_profiler() {
  PyType_Slot slots[] = {
      {Py_tp_methods, methods}, {Py_tp_dealloc, (void*)destroy}, {0, nullptr}};
  PyType_Spec spec = {"vllm._kv_profiler.Ring", sizeof(Ring), 0,
                      Py_TPFLAGS_DEFAULT, slots};
  auto* m = PyModule_Create(&module);
  if (!m) return nullptr;
  ring_type = (PyTypeObject*)PyType_FromSpec(&spec);
  if (!ring_type) {
    Py_DECREF(m);
    return nullptr;
  }
  if (PyModule_AddObject(m, "Ring", (PyObject*)ring_type) < 0) {
    Py_DECREF(ring_type);
    Py_DECREF(m);
    return nullptr;
  }
  PyModule_AddIntConstant(m, "EVENT_SIZE", sizeof(KvprofEvent));
  return m;
}
