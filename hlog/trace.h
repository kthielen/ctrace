/*
 * trace : serialize/deserialize type-directed information in a forward-only trace file, by representing storage as a sequence of variant values
 *         (from the variant of all possible types), and allowing the variant type to be extended with new constructors
 *
 * The main purpose of this lib is to minimize NFS traffic while writing structured logs.  These files will be less convenient to read
 * and query than fregion files, but should typically be smaller and result in fewer writes to NFS (because fregion writes are indirect
 * through mmap, which performs poorly with some NFS configurations).
 */

#ifndef HOBBES_TRACE_H_INCLUDED
#define HOBBES_TRACE_H_INCLUDED

#include "reflect.h"
#include <chrono>
#include <string>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include <functional>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <signal.h>

BEGIN_HLOG_NAMESPACE
namespace trace {

#define HTRACE_MAGIC_PREFIX "HTRACEV2"
#define HTRACE_OLD_MAGIC_PREFIX "HTRACEV1" /* this is just like the v2 format except symbol table resets aren't allowed */

// convenience for raising errors out of errno
inline void raiseSysError(const std::string& msg, const std::string& fname, int e = errno) {
  std::ostringstream ss;
  ss << fname << ": " << msg << " (" << strerror(e) << ")" << std::flush;
  if (e == ENOMEM) {
    // when we're out of memory, die in a way that lets any controlling process know we hit OOM
    std::cerr << ss.str() << " -- ABORT ON OOM" << std::endl;
    kill(getpid(), SIGKILL);
  }
  throw std::runtime_error(ss.str());
}

// to save some space, we cut some sizes down where we assume we can get away with it
// but just to be sure, let's at least fail loudly prior to doing the wrong thing
template <typename U, typename T>
inline U truncate(T t, const char* explain = "truncate failure") {
  if (t >= std::numeric_limits<U>::max()) {
    throw std::runtime_error(explain);
  }
  return static_cast<U>(t);
}

// the writer/reader classes mediate access to trace files
// (forward-declare so that we can declare types for per-constructor input/output)
class writer;
class reader;

// a single constructor out of the (informally defined) variant for all trace values
// (variant values have a unique constructor ID in the range 1..2^32-1)
typedef uint32_t ctorid;

// type-annotated writers for individual constructors
// writers will either be statically typed, where input types are known C++ types
// or they can be "dynamically typed", where types are known at allocation time
// ("dynamically typed" values must be boxed in some arbitrary universal type)
class wseriesi {
public:
  wseriesi(ctorid cid) : cid(cid) { }
  virtual ~wseriesi() { }
  ctorid id() const { return this->cid; }
private:
  ctorid cid;
};
template <typename T>
class wseries : public wseriesi {
public:
  wseries(writer*, ctorid);
  static ty::desc type();
  void operator()(datetime_t, const T&);
private:
  writer* w;
};
template <typename U>
class dwseries : public wseriesi {
public:
  typedef std::function<void(writer*, const U&)> WriteF;
  dwseries(writer*, ctorid, const WriteF&);
  void operator()(datetime_t, const U&);
private:
  writer* w;
  WriteF  writeF;
};

// describe the type T (to persist or check a stored type description)
template <typename T>
ty::desc descSeriesType();

// abstract over file writes (so trace data can be buffered if needed)
struct WriteTraceData {
  virtual ~WriteTraceData() { }
  virtual const std::string& pathDesc() = 0;
  virtual void write(const uint8_t* data, size_t dataSize) = 0;
  virtual void writeAtOffset(off_t offset, const uint8_t* data, size_t dataSize) = 0;
};
typedef std::shared_ptr<WriteTraceData> WriteTraceDataPtr;

class WriteFileData : public WriteTraceData {
public:
  WriteFileData(const std::string& path, bool writeHeader = true) : path(path) {
    // open a file for writing, but refuse to write into an existing trace file
    // except if it already exists and it's only 0 bytes
    // (this is how we atomically allocate a unique file name)
    this->fd = open(path.c_str(), O_WRONLY|O_CREAT, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
    if (this->fd < 0) {
      raiseSysError("Failed to open file for writing", path);
    }
    struct stat sb;
    if (::fstat(this->fd, &sb) < 0) {
      raiseSysError("Failed to stat opened trace file", path);
    }
    if (sb.st_size != 0) {
      throw std::runtime_error("Can't write to trace file that already exists: " + path);
    }

    if (writeHeader) {
      // write the magic prefix so that readers can identify this file correctly
      static const char   prefix[]  = HTRACE_MAGIC_PREFIX;
      static const size_t prefixLen = sizeof(prefix) - 1; // ignore the '\0' char

      if (::write(this->fd, prefix, prefixLen) != ssize_t(prefixLen)) {
        raiseSysError("Failed to write prefix to trace file", path);
      }
    }
  }
  ~WriteFileData() {
    ::fsync(this->fd);
    ::close(this->fd);
  }

  const std::string& pathDesc() override {
    return this->path;
  }

  void write(const uint8_t* data, size_t dataSize) override {
    size_t k = 0;
    while (k < dataSize) {
      auto n = ::write(this->fd, data + k, dataSize - k);
      if (n < 0 || (n == 0 && errno != 0)) {
        raiseSysError("Failed to write to file", this->path);
      }
      k += n;
    }
  }

  void writeAtOffset(off_t offset, const uint8_t* data, size_t dataSize) override {
    if (::lseek(this->fd, offset, SEEK_SET) == static_cast<off_t>(-1)) {
      raiseSysError("Failed to seek in trace to update", this->path);
    }
    this->write(data, dataSize);
    if (::lseek(this->fd, 0, SEEK_END) == static_cast<off_t>(-1)) {
      raiseSysError("Failed to reset write position in trace", this->path);
    }
  }
private:
  int         fd;
  std::string path;
};

struct BufferTraceData : public trace::WriteTraceData {
  std::vector<uint8_t> buffer;

  BufferTraceData(bool writeHeader = true) {
    if (writeHeader) {
      static const char prefix[] = HTRACE_MAGIC_PREFIX;
      this->write(reinterpret_cast<const uint8_t*>(prefix), sizeof(prefix) - 1);
      uint32_t c = 0;
      this->write(reinterpret_cast<const uint8_t*>(&c), sizeof(c));
    }
  }
  const std::string& pathDesc() override {
    static const thread_local std::string desc = "buffer";
    return desc;
  }
  void write(const uint8_t* data, size_t dataSize) override {
    this->buffer.insert(this->buffer.end(), data, data + dataSize);
  }
  void writeAtOffset(off_t offset, const uint8_t* data, size_t dataSize) override {
    if (offset + dataSize > this->buffer.size()) {
      std::ostringstream ss;
      ss << reinterpret_cast<uint64_t>(this) << ": Trace buffer access out of bounds (with offset=" << offset << ", dataSize=" << dataSize << ", totalBufferSize=" << this->buffer.size() << ")";
      throw std::runtime_error(ss.str());
    }
    memcpy(this->buffer.data() + offset, data, dataSize);
  }
};

// write a trace file
//
//  to write a trace file into 'filename.log', construct a variable like:
//    trace::writer w("filename.log");
//
//  to make a constructor to write into the trace with type T, open a "series" from the file:
//    auto& s = w.series<T>();
//
//  then with the series in hand, write into it as much as necessary via function calls, like:
//    T t;
//    ...
//    /* fill out 't' */
//    ...
//    s(t);
//
//  valid types T are primitives, char*, std::string, pairs, fixed-length arrays, vectors,
//  tuples, variants, reflective structs/variants, opaque type aliases, and any type that has a
//  specialization of 'trace::store<T>' defined (usually users will not need to specialize
//  this to get serialization, but it's an option)
//
//  output is buffered, and you can change the buffer limit to X in bytes in the constructor:
//    trace::writer w("filename.log", X);
//  or later with:
//    w.bufferLimit(X);
//
//  the buffer is flushed at the end of a whole-type write if the buffer size is beyond the limit
//  (this means that a large value could push the buffer beyond its limit briefly)
//
//  to force a buffer flush, just call:
//    w.flush();
//
class writer {
public:
  writer(const WriteTraceDataPtr& out, size_t buffer_lim = 32*4096)
  : out(out)
  , buffer_lim(buffer_lim)
  {
  }
  writer(const std::string& path, size_t buffer_lim = 32*4096) : writer(std::make_shared<WriteFileData>(path, true), buffer_lim) {
    // initially start with 0 constructors defined
    // (this is a special area in the file header that will be updated with each new constructor)
    updateConstructorCount(0);
  }
  ~writer() {
    for (auto* p : this->series_refs) {
      delete p;
    }
    this->series_refs.clear();

    flush();
  }

  const std::string& filePath() const {
    return this->out->pathDesc();
  }

  void flush() {
    this->out->write(this->buffer.data(), this->buffer.size());
    this->buffer.clear();
  }
  void flushAtLimit() {
    if (this->buffer.size() >= this->buffer_lim) {
      flush();
    }
  }
  size_t bufferLimit() const { return this->buffer_lim; }
  void bufferLimit(size_t x) { this->buffer_lim = x; flushAtLimit(); }

  // the next available constructor ID
  // it gets ID of size+1 because 0 is reserved for the "new series" constructor
  // (typically not a user API)
  ctorid nextConstructorID() const {
    return ctorid(this->series_refs.size() + 1);
  }

  // allocate an object to write just one constructor value
  // (optional size argument just for compatibility with fregion::writer interface)
  template <typename T>
    wseries<T>& series(const std::string& name, size_t unusedSegSize = 0) {
      // allocate this writer and register it with encoders
      // it gets ID of size+1 because 0 is reserved for the "new series" constructor
      auto* p = new wseries<T>(this, nextConstructorID());
      registerSeries(name, descSeriesType<T>(), p);
      return *p;
    }

  // allocate an object to write a "dynamically typed" value
  // (the type description must be provided at the point of allocation here)
  // it will work similarly to a statically typed series,
  // but its input can be represented by a homogenous universal type
  template <typename U, typename F>
    dwseries<U>& dynamicSeries(const std::string& name, const ty::desc& ty, const F& writeF) {
      auto* p = new dwseries<U>(this, nextConstructorID(), typename function_traits<F>::fn(writeF));
      registerSeries(name, ty, p);
      return *p;
    }

  // register a generic series writer (not a user API, use series<T>(...) instead)
  void registerSeries(const std::string& name, const ty::desc& ty, wseriesi* p) {
    if (p->id() != nextConstructorID()) {
      throw std::runtime_error("Internal error, write series must be registered in order");
    }

    // serialize the type description so it can be persisted
    ty::bytes tdesc;
    ty::encode(ty, &tdesc);

    // write out this type-registration message
    writeCtor(0); writeByte(0);
    auto nsz = truncate<uint16_t>(name.size(), "Constructor name too large (greater than 2^16 characters)");
    writeBytes(reinterpret_cast<const uint8_t*>(&nsz), sizeof(nsz));
    writeBytes(reinterpret_cast<const uint8_t*>(name.data()), nsz);

    auto tsz = truncate<uint32_t>(tdesc.size(), "Type description size too large (greater than 2^32 bytes)");
    writeBytes(reinterpret_cast<const uint8_t*>(&tsz), sizeof(tsz));
    writeBytes(tdesc.data(), tdesc.size());

    // add the writer for this constructor to the total sequence
    this->series_refs.push_back(p);
    this->series_names.push_back(name);

    // this might affect our recorded constructor ID sizes going forward
    if (this->series_refs.size() == (1UL << 8UL*this->ctorsz)) {
      ++this->ctorsz;
    }

    // and leave a hint to readers that need to know how many constructors are defined in total
    updateConstructorCount(ctorid(this->series_refs.size()));
  }

  // mark the start of a new payload by writing as many bits as are necessary now for a constructor
  void writeCtor(ctorid id) {
    writeBytes(reinterpret_cast<const uint8_t*>(&id), this->ctorsz);
  }

  // write some bytes to the trace file
  // (this call should only be made in the context of a meaningful 'writeCtor' preceding it)
  void writeByte(uint8_t b) {
    this->buffer.push_back(b);
  }
  void writeBytes(const uint8_t* bs, size_t n) {
    this->buffer.insert(this->buffer.end(), bs, bs + n);
  }

  // write a symbol (avoid repeating expensive variable-length strings)
  void writeSymbol(const std::string& x) {
    auto s = this->symbol_ids.find(x);
    if (s != this->symbol_ids.end()) {
      // already interned, just output the index
      writeBytes(reinterpret_cast<const uint8_t*>(&s->second), this->symbolsz);
    } else {
      // new symbol, emit the definition
      size_t newID = 0;
      writeBytes(reinterpret_cast<const uint8_t*>(&newID), this->symbolsz);

      size_t len = x.size();
      writeBytes(reinterpret_cast<const uint8_t*>(&len), sizeof(size_t));
      writeBytes(reinterpret_cast<const uint8_t*>(x.data()), len);

      // symbol IDs will be monotonically increasing from 1
      // (id=0 is reserved for the "new symbol" code)
      newID = 1 + this->symbol_ids.size();
      this->symbol_ids[x] = newID;

      // symbol ID size will grow as we cross new byte boundaries
      if (newID >= (1UL << size_t(this->symbolsz*8UL))) {
        ++this->symbolsz;
      }
    }
  }

  // write the current timestamp
  // it will be stored as a variable-length diff from the previously recorded timestamp
  // if first two bits are 00, then 8 bytes of absolute time in microseconds
  // if first two bits are 01, then 6 remaining bits + 8 more bits as diff *2 from previous timestamp
  // if first two bits are 10, then 6 remaining bits + 16 more bits as diff *2 from previous timestamp
  // if first two bits are 11, then 6 remaining bits + 24 more bits as diff *2 from previous timestamp
  void writeTimestamp(uint64_t ts) {
    uint64_t dt = (ts - this->lastts) >> 1;

    if (ts < this->lastts || dt >= (1UL << 30UL)) {
      uint8_t reset = 0;
      writeBytes(&reset, 1);
      writeBytes(reinterpret_cast<const uint8_t*>(&ts), 8);
      this->lastts = ts;
    } else {
      this->lastts += dt << 1;

      if (dt < (1UL << 14UL)) {
        dt = 1 | (dt << 2);
        writeBytes(reinterpret_cast<const uint8_t*>(&dt), 2);
      } else if (dt < (1UL << 22UL)) {
        dt = 2 | (dt << 2);
        writeBytes(reinterpret_cast<const uint8_t*>(&dt), 3);
      } else {
        dt = 3 | (dt << 2);
        writeBytes(reinterpret_cast<const uint8_t*>(&dt), 4);
      }
    }
  }

  // not intended for general use
  void unsafeSetWriter(const std::string& name, wseriesi* p) {
    auto id = p->id();
    if (this->series_refs.size() <= id) {
      this->series_refs.resize(id + 1);
      this->series_names.resize(id + 1);
    }
    this->series_refs[id] = p;
    this->series_names[id] = name;

    // this might affect our recorded constructor ID sizes going forward
    while (this->series_refs.size() >= (1UL << 8UL*this->ctorsz)) {
      ++this->ctorsz;
    }
  }
  wseriesi* unsafeGetWriter(const std::string& name) {
    for (size_t i = 0; i < this->series_refs.size(); ++i) {
      if (this->series_names[i] == name) {
        return this->series_refs[i];
      }
    }
    throw std::runtime_error("The series '" + name + "' is not defined in this trace file");
  }
  void resetSymbols() {
    this->symbol_ids.clear();
    this->symbolsz = 1;

    // write a message to let the reader know that we reset the symbol table
    writeCtor(0); writeByte(1);
  }
private:
  std::string          path;       // the output path for our file
  WriteTraceDataPtr    out;        // the handler for output bytes
  uint8_t              ctorsz = 1; // how many bytes of ctor IDs should we write?
  std::vector<uint8_t> buffer;     // buffered output bytes (to reduce write() call spam)
  size_t               buffer_lim; // how far we should write before flushing

  // update the count of constructors defined in the file
  // (this should be rare enough in long runs that the expensive seek/reset on the file is amortized out)
  void updateConstructorCount(uint32_t c) {
    this->out->writeAtOffset(sizeof(HTRACE_MAGIC_PREFIX) - 1, reinterpret_cast<const uint8_t*>(&c), sizeof(c));
  }

  // optionally intern strings
  typedef std::unordered_map<std::string, size_t> SymbolIDs;
  SymbolIDs symbol_ids;
  uint8_t   symbolsz = 1;

  // encode timestamps as relative diffs between encodings
  uint64_t lastts = 0;

  // objects allocated to write event values by constructor
  typedef std::vector<wseriesi*> SeriesRefs;
  SeriesRefs series_refs;

  typedef std::vector<std::string> SeriesNames;
  SeriesNames series_names;

  writer() { }
};

// type-annotated readers for individual constructors
// all mapping to a base where type information is unknown
class rseriesi {
public:
  rseriesi(const std::string& name) : sname(name) { }
  virtual ~rseriesi() { }
  const std::string& name() const { return this->sname; }
  virtual ty::desc type() const = 0;
  virtual void read(reader*) = 0;
private:
  std::string sname;
};
template <typename T>
class rseries : public rseriesi {
public:
  rseries(const std::string&, const std::function<void(datetime_t, const T&)>&);
  ty::desc type() const override;
  void read(reader*) override;
private:
  std::function<void(datetime_t, const T&)> handler;
};

// read a trace file
//
//  to read a trace file from 'filename.log', just construct a variable like:
//    trace::reader r("filename.log");
//
//  to make a deconstructor to read out of the trace with type T, attach a "series" to the file:
//    r.series<T>([](const T& t) { ... });
//
//  the order of these series constructions must match the order when the file was originally written
//
//  with all series attached, scan through the file and call each series handler with:
//    r.run();
//
//  valid types T are primitives, char*, std::string, pairs, fixed-length arrays, vectors,
//  tuples, variants, reflective structs/variants, opaque type aliases, and any type that has a
//  specialization of 'trace::store<T>' defined (usually users will not need to specialize
//  this to get serialization, but it's an option)
//
class reader {
public:
  reader(const std::string& path, size_t buffer_sz = 32*4096)
  : path(path),
    buffer_sz(std::max<size_t>(4096, buffer_sz))
  {
    // open the file for reading
    this->fd = open(path.c_str(), O_RDONLY, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
    if (this->fd < 0) {
      raiseSysError("Failed to open file for reading", path);
    }

    // read the magic prefix (else it's not a valid trace file)
    constexpr size_t prefix_len = sizeof(HTRACE_MAGIC_PREFIX) - 1;
    char prefix_buf[prefix_len + 1];

    if (::read(this->fd, prefix_buf, prefix_len) != ssize_t(prefix_len)) {
      ::close(this->fd);
      raiseSysError("Failed to read prefix from trace file", path);
    }
    if (strncmp(prefix_buf, HTRACE_OLD_MAGIC_PREFIX, prefix_len) == 0) {
      // this is the old V1 trace format, where symbol table resets aren't allowed
      // we can read it like a normal V2 trace file but we can't look for symbol table resets
      this->allowSymbolTableResets = false;
    } else if (strncmp(prefix_buf, HTRACE_MAGIC_PREFIX, prefix_len) != 0) {
      ::close(this->fd);
      throw std::runtime_error("Not a valid trace file: " + path);
    }

    // read the expected constructor count (in case we need to know how far to read for all valid definitions)
    if (::read(this->fd, reinterpret_cast<uint8_t*>(&this->expected_ctorsz), sizeof(this->expected_ctorsz)) != sizeof(this->expected_ctorsz)) {
      ::close(this->fd);
      raiseSysError("Failed to read prefix from trace file", path);
    }
  }
  ~reader() {
    for (auto* r : this->series_refs) {
      delete r;
    }
    this->series_refs.clear();

    ::close(this->fd);
  }

  // how many constructor definitions are predicted by this file's header?
  uint32_t expectedConstructorCount() const { return this->expected_ctorsz; }

  // how many constructor definitions have we counted so far?
  uint32_t constructorsDefined() const { return uint32_t(this->ctorc); }

  // add a handler for another constructor (matched in order with the write side)
  template <typename F>
  void series(const std::string& n, const F& f) {
    typedef typename function_traits<F>::argl Argl;
    typedef typename tupType<1, Argl>::type CArg0R;
    typedef typename std::remove_reference<CArg0R>::type CArg0;
    typedef typename std::remove_const<CArg0>::type T;

    this->series_refs.push_back(new rseries<T>(n, f));
  }

  void seekToInitialSegment() {
    if (::lseek(this->fd, sizeof(HTRACE_MAGIC_PREFIX) - 1 + sizeof(this->expected_ctorsz), SEEK_SET) == static_cast<off_t>(-1)) {
      raiseSysError("Failed to seek in trace file", this->path);
    }
  }

  // try to find the series with a given name
  // (raises an error if the name isn't given to any series)
  rseriesi* seriesWithName(const std::string& n) const {
    for (size_t i = 0; i < this->series_refs.size(); ++i) {
      if (this->series_refs[i]->name() == n) {
        return this->series_refs[i];
      }
    }
    std::ostringstream err;
    err << "No such series ";
    if (this->series_refs.size() < this->expectedConstructorCount()) {
      err << "yet ";
    }
    err << "defined: '" << n << "'";
    throw std::runtime_error(err.str());
  }

  // read and process a single constructor
  bool step() {
    ctorid c = 0;
    if (!maybeReadCtor(&c)) return false;
    if (this->series_refs.size() <= c) {
      throw std::runtime_error("Invalid constructor for trace file (corruption?): " + string::from(c));
    }
    this->series_refs[c]->read(this);
    return true;
  }

  // read and process the file data until EOF
  void run() {
    while (step());
  }

  // read some bytes from the trace file
  // (this call should only be made in the context of a meaningful 'readCtor' preceding it)
  uint8_t readByte() {
    uint8_t r = 0;
    readBytes(&r, 1);
    return r;
  }
  void readBytes(uint8_t* bs, size_t n) {
    this->bytes_read += n;

    while (n > 0) {
      size_t k = std::min<size_t>(this->buffer.size(), this->buffer_pos + n);
      size_t m = k - this->buffer_pos;
      memcpy(bs, this->buffer.data() + this->buffer_pos, m);
      bs += m;
      n  -= m;

      this->buffer_pos += m;
      if (this->buffer_pos == this->buffer.size()) {
        readNextBlock();

        if (n > 0 && this->buffer.size() == 0) {
          throw std::runtime_error("Attempt to read past EOF"); // TODO: optionally block and wait for more data?
        }
      }
    }
  }

  // how many bytes have we read out of this file so far?
  size_t bytesRead() const {
    return this->bytes_read;
  }

  // read a symbol (avoids repeating expensive variable-length strings)
  const std::string& readSymbol() {
    size_t sid = 0;
    readBytes(reinterpret_cast<uint8_t*>(&sid), this->symbolsz);

    if (sid != 0) {
      sid -= 1; // normalize to remove new-symbol command

      if (sid < this->symbol_by_id.size()) {
        return this->symbol_by_id[sid];
      } else {
        throw std::runtime_error("Error in file, reference to invalid symbol #" + string::from(sid));
      }
    }

    // this is a new symbol
    this->symbol_by_id.push_back(std::string());
    std::string& sym = this->symbol_by_id.back();

    // resize the size of symbol IDs if necessary
    if (this->symbolsz < sizeof(size_t) && this->symbol_by_id.size() >= (1UL << size_t(this->symbolsz*8UL))) {
      ++this->symbolsz;
    }

    // read the new symbol
    size_t n = 0;
    readBytes(reinterpret_cast<uint8_t*>(&n), sizeof(size_t));
    sym.resize(n);
    if (n > 0) {
      readBytes(reinterpret_cast<uint8_t*>(&sym.front()), n);
    }
    return sym;
  }

  // read a timestamp encoded as a relative diff (see writer::writeTimestamp)
  uint64_t readTimestamp() {
    uint8_t c = 0;
    readBytes(&c, 1);

    uint64_t diff = 0;
    if ((c & 3) == 1) {
      readBytes(reinterpret_cast<uint8_t*>(&diff), 1);
      diff = (diff << 6) | (c >> 2);
      this->lastts += diff << 1;
    } else if ((c & 3) == 2) {
      readBytes(reinterpret_cast<uint8_t*>(&diff), 2);
      diff = (diff << 6) | (c >> 2);
      this->lastts += diff << 1;
    } else if ((c & 3) == 3) {
      readBytes(reinterpret_cast<uint8_t*>(&diff), 3);
      diff = (diff << 6) | (c >> 2);
      this->lastts += diff << 1;
    } else {
      readBytes(reinterpret_cast<uint8_t*>(&this->lastts), sizeof(this->lastts));
    }
    return this->lastts;
  }
protected:
  std::string          path;
  int                  fd = -1;
  uint32_t             expected_ctorsz = 0;
  std::vector<uint8_t> buffer;
  size_t               buffer_pos = 0;
  size_t               buffer_sz = 32*4096;
  size_t               bytes_read = 0;
  bool                 allowSymbolTableResets = true;

  // keep track of symbol definitions and ID size (widened as we cross byte thresholds)
  typedef std::vector<std::string> Symbols;
  Symbols symbol_by_id;
  uint8_t symbolsz = 1;

  // keep track of timestamp differences to read minimal diffs as much as possible
  uint64_t lastts = 0;

  // count constructors and size IDs just as we cross byte thresholds
  size_t  ctorc  = 0;
  uint8_t ctorsz = 1;

  // keep track of all registered series readers (assumed closed)
  typedef std::vector<rseriesi*> SeriesRefs;
  SeriesRefs series_refs;

  // by default, we assume all readers are installed with correct types up front
  // so this step only performs validation (to ensure that installed readers have the right name and type)
  //
  // but a reasonable implementation could be given to _derive_ readers from stored name and type
  virtual rseriesi* makeReaderAt(size_t i, const std::string& name, const ty::desc& ty) {
    // we're at a new constructor definition
    // do we actually have one registered here?
    if (this->series_refs.size() <= i) {
      throw std::runtime_error("Trace file has registered series beyond the expected set");
    }
    rseriesi* s = this->series_refs[i];

    // if this constructor name doesn't match what we expect, we have to reject the file for this reader
    if (name != s->name()) {
      throw std::runtime_error("Trace file defines series #" + string::from(i) + " with name '" + name + "' but reader expected '" + s->name() + "'");
    }

    // if this type description doesn't match what we expect, we have to reject the file for this reader
    if (!ty::equivModOffset(ty, s->type())) {
      throw std::runtime_error(
        "Trace file defines series '" + name + "' with inconsistent type:\n" +
        "  Expected: " + ty::show(s->type()) + "\n" +
        "  Actual:   " + ty::show(ty)
      );
    }

    // all good
    return s;
  }

  // read up to a non-control constructor ID
  // (this might mean also reading and internalizing any number of intermediate constructor definitions)
  bool maybeReadCtor(ctorid* c) {
    while (true) {
      *c = 0;
      try {
        readBytes(reinterpret_cast<uint8_t*>(c), this->ctorsz);
      } catch (...) {
        // couldn't read past EOF, so we're done
        return false;
      }
      if (*c != 0) {
        // take 1 from the recorded constructor to discount the "new constructor" constructor at id 0
        *c -= 1;
        return true;
      }

      // are we just resetting the symbol table?
      // if so, just reset it and continue trying to read an event constructor
      if (this->allowSymbolTableResets && readByte() == 1) {
        this->symbol_by_id.clear();
        this->symbolsz = 1;
        continue;
      }

      // read the new constructor name
      uint16_t nsz = 0;
      readBytes(reinterpret_cast<uint8_t*>(&nsz), sizeof(nsz));
      std::string n;
      n.resize(nsz);
      readBytes(reinterpret_cast<uint8_t*>(&n.front()), nsz);

      // read the new constructor type
      uint32_t tlen = 0;
      readBytes(reinterpret_cast<uint8_t*>(&tlen), sizeof(tlen));
      ty::bytes tdefbs;
      tdefbs.resize(tlen);
      readBytes(tdefbs.data(), tlen);
      ty::desc tdef = ty::decode(tdefbs);

      // install the reader for this constructor ID
      rseriesi* s = makeReaderAt(this->ctorc, n, tdef);
      if (this->series_refs.size() <= this->ctorc) {
        this->series_refs.resize(this->ctorc+1);
      }
      this->series_refs[this->ctorc] = s;

      // update the count and size of constructor codes (we've added a new constructor case)
      ++this->ctorc;
      if (this->ctorsz < sizeof(ctorid) && this->ctorc >= (1UL << size_t(this->ctorsz*8UL))) {
        ++this->ctorsz;
      }
    }
  }

  void readNextBlock() {
    this->buffer_pos = 0;
    this->buffer.resize(this->buffer_sz);

    ssize_t r = ::read(this->fd, this->buffer.data(), this->buffer_sz);
    if (r < 0) {
      raiseSysError("Error while reading trace file", this->path);
    }
    this->buffer.resize(static_cast<size_t>(r));
  }
};

// simple test to determine if a file is a trace file (without having to actually decode it)
inline bool canReadFile(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
  if (fd < 0) {
    return false;
  }
  constexpr size_t prefix_len = sizeof(HTRACE_MAGIC_PREFIX) - 1;
  char prefix_buf[prefix_len + 1];
  if (::read(fd, prefix_buf, prefix_len) != ssize_t(prefix_len)) {
    ::close(fd);
    return false;
  }
  if (strncmp(prefix_buf, HTRACE_MAGIC_PREFIX, prefix_len) != 0) {
    ::close(fd);
    return false;
  }
  uint32_t expected_ctorsz=0;
  if (::read(fd, reinterpret_cast<uint8_t*>(&expected_ctorsz), sizeof(expected_ctorsz)) != sizeof(expected_ctorsz)) {
    ::close(fd);
    return false;
  }
  ::close(fd);
  return true;
}

// type directed storage into and out of trace files
template <typename T, typename P = void>
  struct store {
  };

// test whether one or a sequence of types have 'store<T>' specializations
// (a poor man's qualified types)
template <typename T, typename P = void>
struct has_store { static const bool value = false; };
template <typename T>
struct has_store<T, typename valid_type<decltype(store<T>::write(nullptr,*reinterpret_cast<const T*>(0xdeadbeef)))>::type> { static const bool value = true; };

template <typename ... Ts>
struct all_have_store { static const bool value = true; };
template <typename T, typename ... Ts>
struct all_have_store<T, Ts...> { static const bool value = has_store<T>::value && all_have_store<Ts...>::value; };

// the unit type stores trivially
template <>
  struct store<unit> {
    static ty::desc type() { return ty::prim("unit"); }
    static void write(writer*, unit) { }
    static void read (reader*, unit*) { }
  };

// primitives
#define PRIV_HTRACE_DEFINE_PRIMTY_WTY(T, tdesc) \
  template <> \
    struct store<T> { \
      static ty::desc type ()                { return tdesc; } \
      static void     write(writer* w, T  x) { w->writeBytes(reinterpret_cast<const uint8_t*>(&x), sizeof(T)); } \
      static void     read (reader* r, T* x) { r->readBytes(reinterpret_cast<uint8_t*>(x), sizeof(T)); } \
    }
#define PRIV_HTRACE_DEFINE_PRIMTY(T, n) PRIV_HTRACE_DEFINE_PRIMTY_WTY(T, (ty::prim(n)))
#define PRIV_HTRACE_DEFINE_PRIMTY_W(T, n, r) PRIV_HTRACE_DEFINE_PRIMTY_WTY(T, (ty::prim(n, ty::prim(r))))

PRIV_HTRACE_DEFINE_PRIMTY(bool,     "bool");
PRIV_HTRACE_DEFINE_PRIMTY(uint8_t,  "byte");
PRIV_HTRACE_DEFINE_PRIMTY(char,     "char");
PRIV_HTRACE_DEFINE_PRIMTY(int16_t,  "short");
PRIV_HTRACE_DEFINE_PRIMTY(int32_t,  "int");
PRIV_HTRACE_DEFINE_PRIMTY(int64_t,  "long");
PRIV_HTRACE_DEFINE_PRIMTY_W(uint16_t, "ushort", "short");
PRIV_HTRACE_DEFINE_PRIMTY_W(uint32_t, "uint", "int");
PRIV_HTRACE_DEFINE_PRIMTY_W(uint64_t, "ulong", "long");
#if defined(__APPLE__) && defined(__MACH__)
PRIV_HTRACE_DEFINE_PRIMTY_W(size_t, "ulong", "long");
#endif
PRIV_HTRACE_DEFINE_PRIMTY(__int128, "int128");
PRIV_HTRACE_DEFINE_PRIMTY(float,    "float");
PRIV_HTRACE_DEFINE_PRIMTY(double,   "double");

// write fixed-length arrays
template <typename T, size_t N>
  struct store<T[N], typename tbool<has_store<T>::value>::type> {
    static ty::desc type() { return ty::array(store<T>::type(), ty::nat(N)); }
    static void write(writer* w, const T (&x)[N]) {
      for (size_t i = 0; i < N; ++i) {
        store<T>::write(w, x[i]);
      }
    }
    static void read(reader* r, T (&x)[N]) {
      for (size_t i = 0; i < N; ++i) {
        store<T>::read(r, &x[i]);
      }
    }
  };
template <typename T, size_t N>
  struct store<std::array<T, N>, typename tbool<has_store<T>::value>::type> {
    static ty::desc type() { return ty::array(store<T>::type(), ty::nat(N)); }
    static void write(writer* w, const std::array<T, N>& x) {
      for (size_t i = 0; i < N; ++i) {
        store<T>::write(w, x[i]);
      }
    }
    static void read(reader* r, std::array<T, N>* x) {
      for (size_t i = 0; i < N; ++i) {
        store<T>::read(r, &((*x)[i]));
      }
    }
  };

// store strings
template <>
  struct store<char*> {
    static ty::desc type() { return ty::array(ty::prim("char")); }
    static void write(writer* w, const char* x, size_t n) {
      store<size_t>::write(w, n);
      w->writeBytes(reinterpret_cast<const uint8_t*>(x), n);
    }
    static void write(writer* w, const char* x) {
      write(w, x, strlen(x));
    }
    // read intentionally left out (unsafe)
  };
template <> struct store<const char*> : public store<char*> { };
template <>
  struct store<std::string> {
    static ty::desc type() { return store<const char*>::type(); }
    static void write(writer* w, const std::string& x) {
      store<const char*>::write(w, x.data(), x.size());
    }
    static void read(reader* r, std::string* x) {
      size_t n=0;
      store<size_t>::read(r, &n);
      x->resize(n);
      if (n > 0) {
        r->readBytes(reinterpret_cast<uint8_t*>(&x->front()), n);
      }
    }
  };

// store vectors
template <typename T>
  struct store<std::vector<T>, typename tbool<has_store<T>::value>::type> {
    static ty::desc type() { return ty::array(store<T>::type()); }
    static void write(writer* w, const std::vector<T>& xs) {
      store<size_t>::write(w, xs.size());
      for (const auto& x : xs) {
        store<T>::write(w, x);
      }
    }
    static void read(reader* r, std::vector<T>* xs) {
      size_t n=0;
      store<size_t>::read(r, &n);
      xs->resize(n);
      for (size_t i = 0; i < n; ++i) {
        store<T>::read(r, &(*xs)[i]);
      }
    }
  };

// store sets (as sequences)
template <typename CT, typename T>
  struct StoreSet {
    static ty::desc type() { return ty::array(store<T>::type()); }
    static void write(writer* w, const CT& xs) {
      store<size_t>::write(w, xs.size());
      for (const auto& x : xs) {
        store<T>::write(w, x);
      }
    }
    static void read(reader* r, CT* xs) {
      size_t n=0;
      store<size_t>::read(r, &n);
      for (size_t i = 0; i < n; ++i) {
        T x;
        store<T>::read(r, &x);
        xs->insert(x);
      }
    }
  };
template <typename T> struct store<std::set<T>, typename tbool<has_store<T>::value>::type> : public StoreSet<std::set<T>, T> { };
template <typename T> struct store<std::unordered_set<T>, typename tbool<has_store<T>::value>::type> : public StoreSet<std::unordered_set<T>, T> { };

// store pairs
template <typename U, typename V>
  struct store<std::pair<U,V>, typename tbool<has_store<U>::value && has_store<V>::value>::type> {
    static ty::desc type() { return ty::tup(-1, store<U>::type(), -1, store<V>::type()); }
    static void write(writer* w, const std::pair<U,V>& x) {
      store<U>::write(w, x.first);
      store<V>::write(w, x.second);
    }
    static void read(reader* r, std::pair<U,V>* x) {
      store<U>::read(r, &x->first);
      store<V>::read(r, &x->second);
    }
  };

// store tuples
template <size_t i, size_t n, typename ... Ts>
  struct TupleStore {
    typedef typename nth<i, Ts...>::type H;
    typedef TupleStore<i+1, n, Ts...>    Recurse;

    static void type(ty::StructFields* fs) {
      fs->push_back(ty::StructField(".f" + string::from(i), -1, store<H>::type()));
      Recurse::type(fs);
    }
    static void write(writer* w, const tuple<Ts...>& x) {
      store<H>::write(w, x.template at<i>());
      Recurse::write(w, x);
    }
    static void read(reader* r, tuple<Ts...>* x) {
      store<H>::read(r, &x->template at<i>());
      Recurse::read(r, x);
    }
  };
template <size_t n, typename ... Ts>
  struct TupleStore<n, n, Ts...> {
    static void type(ty::StructFields*) { }
    static void write(writer*, const tuple<Ts...>&) { }
    static void read(reader*, tuple<Ts...>*) { }
  };
template <typename ... Ts>
  struct store<tuple<Ts...>, typename tbool<all_have_store<Ts...>::value>::type> {
    typedef TupleStore<0, sizeof...(Ts), Ts...> Store;

    static ty::desc type() {
      ty::StructFields fs;
      Store::type(&fs);
      return ty::record(fs);
    }
    static void write(writer* w, const tuple<Ts...>& x) {
      Store::write(w, x);
    }
    static void read(reader* r, tuple<Ts...>* x) {
      Store::read(r, x);
    }
  };

// store (user) reflective structs
struct DescStructF {
  ty::StructFields* fs;
  DescStructF(ty::StructFields* fs) : fs(fs) { }
  template <typename T>
    void visit(const char* fname) {
      this->fs->push_back(ty::StructField(fname, -1, store<T>::type()));
    }
};
template <typename T>
  struct store<T, typename tbool<T::is_hmeta_struct && has_store<typename T::as_tuple_type>::value>::type> {
    typedef typename T::as_tuple_type TT;
    static ty::desc type() {
      ty::StructFields fs;
      DescStructF df(&fs);
      T::meta(df);
      return ty::record(fs);
    }
    static void write(writer* w, const T& x) {
      store<TT>::write(w, *reinterpret_cast<const TT*>(&x));
    }
    static void read(reader* r, T* x) {
      store<TT>::read(r, reinterpret_cast<TT*>(x));
    }
  };

// store reflective enums
template <size_t x, size_t lb, size_t ub, typename P = void>
  struct InRange { };
template <size_t x, size_t lb, size_t ub>
  struct InRange<x, lb, ub, typename tbool<lb <= x && x < ub>::type> { typedef void type; };

template <size_t tags, typename P = void>
  struct EnumTagSize { typedef uint64_t type; }; // support enums with up to 2^64 cases
template <size_t tags>
  struct EnumTagSize<tags, typename InRange<tags, 0UL,       1UL << 8UL >::type> { typedef uint8_t type; };
template <size_t tags>
  struct EnumTagSize<tags, typename InRange<tags, 1UL<<8UL,  1UL << 16UL>::type> { typedef uint16_t type; };
template <size_t tags>
  struct EnumTagSize<tags, typename InRange<tags, 1UL<<16UL, 1UL << 32UL>::type> { typedef uint32_t type; };

template <typename T>
  struct store<T, typename tbool<T::is_hmeta_enum>::type> {
    static ty::desc type() {
      return ty::enumdef(store<uint32_t>::type(), T::meta()); // not necessarily stored in 4 bytes, but for type descriptions here it's fine
    }
    typedef typename EnumTagSize<T::ctorCount>::type tag_type;
    static void write(writer* w, T x) {
      store<tag_type>::write(w, static_cast<tag_type>(T::toOrd(x)));
    }
    static void read(reader* r, T* x) {
      tag_type t=0;
      store<tag_type>::read(r, &t);
      *x = T::fromOrd(static_cast<typename T::rep_t>(t));
    }
  };

// store variants
template <size_t i, size_t n, typename ... Ts>
  struct StoreVariantDesc {
    typedef typename nth<i, Ts...>::type    H;
    typedef StoreVariantDesc<i+1, n, Ts...> Recurse;

    static void type(ty::VariantCtors* cs) {
      cs->push_back(ty::VariantCtor(".f" + string::from(i), static_cast<int>(i), store<H>::type()));
      Recurse::type(cs);
    }
  };
template <size_t n, typename ... Ts>
  struct StoreVariantDesc<n, n, Ts...> {
    static void type(ty::VariantCtors*) { }
  };

template <size_t tag, typename T, typename M>
  struct VariantWrite { static void fn(T* vd, writer* w) { store<T>::write(w, *vd); } };
template <size_t tag, typename T, typename M>
  struct VariantRead {
    static void fn(T* vd, reader* r) {
      new (vd) T();
      store<T>::read(r, vd);
    }
  };
template <typename ... Ts>
  struct store<variant<Ts...>, typename tbool<all_have_store<Ts...>::value>::type> {
    typedef StoreVariantDesc<0, sizeof...(Ts), Ts...> Reflect;
    typedef typename EnumTagSize<sizeof...(Ts)>::type tag_type;

    static ty::desc type() {
      ty::VariantCtors cs;
      Reflect::type(&cs);
      return ty::variant(cs);
    }
    static void write(writer* w, const variant<Ts...>& x) {
      store<tag_type>::write(w, static_cast<tag_type>(x.unsafeTag()));
      x.template apply<void, VariantWrite, void, writer*>(w);
    }
    static void read(reader* r, variant<Ts...>* x) {
      tag_type tt=0;
      store<tag_type>::read(r, &tt);
      x->unsafeTag() = tt;
      variantApp<void, VariantRead, void, tuple<Ts...>, reader*>::apply(x->unsafeTag(), x->unsafePayload(), r);
    }
  };

// store (user) reflective variants
struct DescVariantF {
  ty::VariantCtors* ctors;
  DescVariantF(ty::VariantCtors* ctors) : ctors(ctors) { }
  template <typename T>
    void ctor(const char* n, int id) {
      this->ctors->push_back(ty::VariantCtor(n, id, store<T>::type()));
    }
};
template <typename T>
  struct store<T, typename tbool<T::is_hmeta_variant && has_store<typename T::as_variant_type>::value>::type> {
    typedef typename T::as_variant_type VT;
    static ty::desc type() {
      ty::VariantCtors cs;
      DescVariantF f(&cs);
      T::meta(f);
      return ty::variant(cs);
    }
    static void write(writer* w, const T& x) { store<VT>::write(w, *reinterpret_cast<const VT*>(&x)); }
    static void read (reader* r,       T* x) { store<VT>::read (r,  reinterpret_cast<VT*>(x)); }
  };

// store generic structs/variants carrying type-level field names
// (if the underlying type isn't tuple or variant, the names are ignored)
template <typename NameList, typename T>
struct store<withNames<NameList, T>, typename tbool<has_store<T>::value>::type> {
  static ty::desc type() {
    auto ns = lowerStringList<NameList>();

    // try to add names into this type
    ty::desc u = store<T>::type();
    if (const auto* ps = unroll(u).record()) {
      ty::StructFields fs;
      for (size_t i = 0; i < ps->fields.size(); ++i) {
        const auto& f = ps->fields[i];
        if (i < ns.size()) {
          auto n = ns[i].empty() ? ".f"+string::from(i) : ns[i];
          fs.push_back(ty::StructField(n, f.at<1>(), f.at<2>()));
        } else {
          fs.push_back(f);
        }
      }
      return ty::record(fs);
    } else if (const auto* pv = unroll(u).variant()) {
      ty::VariantCtors cs;
      for (size_t i = 0; i < pv->ctors.size(); ++i) {
        const auto& c = pv->ctors[i];
        if (i < ns.size()) {
          auto n = ns[i].empty() ? ".f"+string::from(i) : ns[i];
          cs.push_back(ty::VariantCtor(n, c.at<1>(), c.at<2>()));
        } else {
          cs.push_back(c);
        }
      }
      return ty::variant(cs);
    } else {
      return u;
    }
  }
  static void write(writer* w, const withNames<NameList, T>& s) { store<T>::write(w, s.value); }
  static void read (reader* r,       withNames<NameList, T>* s) { store<T>::read (r, &s->value); }
};

// store opaque type aliases
template <typename T>
  struct store<T, typename tbool<T::is_hmeta_alias && has_store<typename T::type>::value>::type> {
    typedef typename T::type RT;
    static ty::desc type() { return ty::prim(T::name(), store<RT>::type()); }
    static void write(writer* w, const T& x) {
      store<RT>::write(w, x.value);
    }
    static void read(reader* r, T* x) {
      store<RT>::read(r, &x->value);
    }
  };

// store chrono times
template <>
  struct store<std::chrono::system_clock::time_point> {
    static ty::desc type() { return ty::prim("datetime", store<size_t>::type()); }
    static void write(writer* w, const std::chrono::system_clock::time_point& t) {
      store<size_t>::write(w, std::chrono::duration_cast<std::chrono::microseconds>(t.time_since_epoch()).count());
    }
    static void read(reader* r, std::chrono::system_clock::time_point* t) {
      size_t ut=0;
      store<size_t>::read(r, &ut);
      *t = std::chrono::system_clock::time_point(std::chrono::microseconds(ut));
    }
  };

// store chrono durations
template <typename Rep, std::intmax_t Numerator, std::intmax_t Denominator>
  struct store<std::chrono::duration<Rep, std::ratio<Numerator, Denominator>>> {
    static ty::desc type() {
      // cduration Rep Numerator Denominator
      return ty::appc(ty::prim("cduration", ty::fnc({"rep", "n", "d"}, ty::var("rep"))), {store<Rep>::type(), ty::nat(Numerator), ty::nat(Denominator)});
    }

    typedef std::chrono::duration<Rep, std::ratio<Numerator, Denominator>> CT;
    static void write(writer* w, const CT& dt) {
      store<Rep>::write(w, dt.count());
    }
    static void read(reader* r, CT* dt) {
      Rep udt = Rep();
      store<Rep>::read(r, &udt);
      *dt = CT(udt);
    }
  };
template <typename Rep>
  struct store<ChronoDuration<Rep>> {
    // we can't ask for a type description for this type
    // its n/d parameters are unknown statically and must have been dealt with elsewhere

    static void write(writer* w, const ChronoDuration<Rep>& dt) {
      store<Rep>::write(w, dt.value);
    }
    static void read(reader* r, ChronoDuration<Rep>* dt) {
      store<Rep>::read(r, &dt->value);
    }
  };

// store recursive types
template <typename T>
  struct store<RolledRec<T>> {
    static ty::desc type() {
      return withRecursive(ty::var("x"), []() { return ty::recursive("x", store<typename T::Unrolled>::type()); });
    }
    static void write(writer* w, const RolledRec<T>& x) {
      store<typename T::Unrolled>::write(w, x.rolled->unrolled);
    }
    static void read(reader* r, RolledRec<T>* x) {
      typedef typename T::Unrolled DT;
      DT ux;
      store<DT>::read(r, &ux);
      *x = RolledRec<T>::roll(ux);
    }
  };

// allow serialization of any closed type freely generated by supported types
template <typename T>
wseries<T>::wseries(writer* w, ctorid cid) : wseriesi(cid), w(w) { }
template <typename T>
ty::desc wseries<T>::type() { return store<T>::type(); }

template <typename T>
void wseries<T>::operator()(datetime_t ts, const T& x) {
  this->w->writeCtor(id());
  this->w->writeTimestamp(ts.value);
  store<T>::write(this->w, x);
  this->w->flushAtLimit();
}

template <typename U>
dwseries<U>::dwseries(writer* w, ctorid id, const WriteF& writeF) : wseriesi(id), w(w), writeF(writeF) {
}
template <typename U>
void dwseries<U>::operator()(datetime_t ts, const U& u) {
  this->w->writeCtor(id());
  this->w->writeTimestamp(ts.value);
  this->writeF(this->w, u);
  this->w->flushAtLimit();
}

// allow deserialization of any closed type freely generated by supported types
template <typename T>
rseries<T>::rseries(const std::string& name, const std::function<void(datetime_t, const T&)>& handler) : rseriesi(name), handler(handler) { }
template <typename T>
ty::desc rseries<T>::type() const { return store<T>::type(); }

template <typename T>
void rseries<T>::read(reader* r) {
  uint64_t ts = r->readTimestamp();
  T t;
  store<T>::read(r, &t);
  this->handler(datetime_t(ts), t);
}

// supported types can be mapped to type descriptions through storage types
template <typename T>
  ty::desc descSeriesType() {
    return store<T>::type();
  }

}
END_HLOG_NAMESPACE

#endif

