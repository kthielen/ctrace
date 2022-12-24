/*
 * ctrace : variant of trace.h with entropy coding (an optional storage backend to produce much smaller files)
 *
 * This implementation of entropy coding trace data reduces to arithmetic encoding, using unique models per-type.
 */

#ifndef HOBBES_CTRACE_H_INCLUDED
#define HOBBES_CTRACE_H_INCLUDED

#define _XOPEN_SOURCE
#include "reflect.h"
#include <algorithm>
#include <array>
#include <numeric>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <functional>
#include <sstream>
#include <memory>

#include <type_traits>
#include <limits>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <limits.h>
#include <signal.h>
#include <poll.h>
#include <ucontext.h>

BEGIN_HLOG_NAMESPACE
namespace ctrace {

// to save some space, we cut some sizes down where we assume we can get away with it
// but just to be sure, let's at least fail loudly prior to doing the wrong thing
template <typename U, typename T>
inline U truncate(T t, const char* explain = "truncate failure") {
  if (t >= std::numeric_limits<U>::max()) {
    throw std::runtime_error(explain);
  }
  return static_cast<U>(t);
}

// optionally sanity check encoded array lengths to at least avoid a messy OOM
inline size_t maxArrLen(size_t upd = 0) {
  static size_t maxLen = 0;
  if (maxLen == 0) {
    maxLen = 100000000;
    if (const char* maxLenStr = std::getenv("HLOG_MAX_READ_ARR_LEN")) {
      auto envMaxLen = atoll(maxLenStr);
      if (envMaxLen > 0) {
        maxLen = envMaxLen;
      }
    }
  }
  if (upd) {
    size_t oldv = maxLen;
    maxLen = upd;
    return oldv;
  } else {
    return maxLen;
  }
}
template <typename T>
inline T checkArrLen(T len) {
  if (len > maxArrLen()) {
    std::ostringstream ss;
    ss << "Array length sanity check failure: " << len << " > " << maxArrLen();
    throw std::runtime_error(ss.str());
  }
  return len;
}

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

/*******************************************************
 *
 * arithmetic encoding/decoding, to pack values down to the limit of entropy
 *
 *******************************************************/

// output data will be "segmented" so that readers see it as a sequence of independent partitions (and can decode them in parallel)
// to make this work, segments have to be aligned at byte boundaries, and predictions have to be reset so that no initial knowledge
// is assumed (other than the prior knowledge of which events/types can be recorded)
//
// each segment begins with a header encoded without bias (so it's read directly as raw bytes)
struct SegmentHeader {
  typedef std::vector<uint8_t> Bitset;

  uint32_t toNextSegment = 0;    // an offset from the current byte position (where this segment begins) to the next segment
  uint32_t eventCount = 0;       // count of events stored in the segment
  Bitset   eventsRecorded = { }; // flags indicating whether an event (by index) appears at least once in the segment
};
inline size_t encodedSize(const SegmentHeader& s) {
  return sizeof(s.toNextSegment) +
         sizeof(s.eventCount) +
         sizeof(uint32_t) +
         s.eventsRecorded.size();
}
inline void writeHeader(std::vector<uint8_t>* bs, const SegmentHeader& s) {
  auto next = reinterpret_cast<const uint8_t*>(&s.toNextSegment);
  bs->insert(bs->end(), next, next + sizeof(s.toNextSegment));
  auto eventC = reinterpret_cast<const uint8_t*>(&s.eventCount);
  bs->insert(bs->end(), eventC, eventC + sizeof(s.eventCount));
  auto eventBC = uint32_t(s.eventsRecorded.size());
  assert(size_t(eventBC) == s.eventsRecorded.size());
  auto bc = reinterpret_cast<const uint8_t*>(&eventBC);
  bs->insert(bs->end(), bc, bc + sizeof(eventBC));
  bs->insert(bs->end(), s.eventsRecorded.begin(), s.eventsRecorded.end());
}
inline void writeHeader(int fd, const std::string& path, const SegmentHeader& s) {
  std::vector<uint8_t> bs;
  writeHeader(&bs, s);
  if (::write(fd, bs.data(), bs.size()) != ssize_t(bs.size())) {
    raiseSysError("Failed to write segment header to file", path);
  }
}
inline void readHeader(SegmentHeader* s, const std::function<void(uint8_t*, size_t)>& readF) {
  readF(reinterpret_cast<uint8_t*>(&s->toNextSegment), sizeof(s->toNextSegment));
  readF(reinterpret_cast<uint8_t*>(&s->eventCount), sizeof(s->eventCount));
  uint32_t events = 0;
  readF(reinterpret_cast<uint8_t*>(&events), sizeof(events));
  s->eventsRecorded.resize(size_t(events));
  readF(s->eventsRecorded.data(), s->eventsRecorded.size());
}

// quickly batch count bits (to take large steps in arith coding when possible)
#define PRIV_HCFREGION_COUNTLZ(x)  __builtin_clz((x))
#define PRIV_HCFREGION_LIKELY(x)   __builtin_expect((x),1)
#define PRIV_HCFREGION_UNLIKELY(x) __builtin_expect((x),0)

// pow2Ceil(x) is the smallest x', such that x' is a power of 2 and x' >= x
template <typename T>
inline T pow2Ceil(T x) {
  if (x == 0) {
    return 1;
  } else {
    T xp = 1 << ((sizeof(T) * 8) - PRIV_HCFREGION_COUNTLZ(x) - 1);
    return xp == x ? x : (xp << 1);
  }
}

// the least number of bytes necessary to represent the given value
template <typename T>
uint8_t leastBytesNeeded(T x) {
  uint8_t r = 0;
  do {
    x >>= 8;
    ++r;
  }
  while (x > 0);
  return r;
}

// take the upper 'bitc' bits of the bitwise reversal of 'bits'
// (this is useful in subsequent encoding and decoding steps that will take multi-bit steps through arithmetic coder ranges)
inline uint8_t reverseBits(uint8_t bits, uint8_t bitc) {
  uint8_t result =
    uint8_t((bits & 1) << 7) |
    uint8_t((bits & 2) << 5) |
    uint8_t((bits & 4) << 3) |
    uint8_t((bits & 8) << 1) |
    uint8_t((bits >> 1) & 8) |
    uint8_t((bits >> 3) & 4) |
    uint8_t((bits >> 5) & 2) |
    uint8_t(bits >> 7);
  return uint8_t(result >> (8 - bitc));
}

// make a lookup table of nybble shifts for arithmetic coding
// this allows us to take 4 bits off the top of the low and high range of arithmetic coder state
// for each such pairing of low/high nybbles the result is (bit_count, bits_output)
inline uint8_t fromNybs(uint8_t low, uint8_t high) { return uint8_t(low | (high << 4)); }
typedef uint8_t nybshift_t;
inline uint8_t nsCount(nybshift_t x) { return uint8_t(x & 0x0f); }
inline uint8_t nsBits (nybshift_t x) { return uint8_t(x >> 4); }

inline std::array<nybshift_t, 256> makeNybShifts() {
  std::array<nybshift_t, 256> result;

  for (uint8_t low = 0; low < (1 << 4); ++low) {
    for (uint8_t high = 0; high < (1 << 4); ++high) {
      uint8_t bits = 0;
      uint8_t bitc = 0;
      for (uint8_t i = 0; i < 4; ++i) {
        bool l = (low  & (1 << (3 - i))) != 0;
        bool h = (high & (1 << (3 - i))) != 0;

        if (l != h) {
          // at this point we can't shift bits out of either low or high, so we're done
          break;
        }

        // either low's most significant bit has crossed to 1 (in which case we write a 1)
        // or else high's most significant bit has crossed to 0 (in which case we write a 0)
        bits = uint8_t((bits << 1) | (l ? 1 : 0));
        ++bitc;
      }
      // reverse the bit sequence to match insertion order
      result[fromNybs(low, high)] = fromNybs(bitc, reverseBits(bits, bitc));
    }
  }

  return result;
}

// parameters for arithmetic coding and modeling
struct arithn {
  typedef uint32_t code;
  typedef uint32_t freq;

  static constexpr uint32_t cbits    = 16;
  static constexpr code     cmax     = (code(1) << cbits) - code(1);
  static constexpr code     cfourth  = code(1) << (cbits-2);
  static constexpr code     chalf    = 2*cfourth;
  static constexpr code     c3fourth = 3*cfourth;

  // for symbol frequency counting
  // set the adjustment period (when prior stats are converted to future expectations)
  static constexpr uint32_t fbits        = 14;
  static constexpr freq     fmax         = (freq(1) << fbits) - freq(1);
  static constexpr freq     fmax_pre_esc = fmax - 1;
};

// shift up to 4 bits out of a range low/high
// ('1' where the upper bit of low is 1, '0' where the upper bit of high is 0)
inline nybshift_t rangeShiftNybble(arithn::code low, arithn::code high) {
  static thread_local std::array<nybshift_t, 256> shifts = makeNybShifts();
  uint8_t highNyb = uint8_t(high >> (arithn::cbits - 4));
  uint8_t lowNyb  = uint8_t(low  >> (arithn::cbits - 4));
  return shifts[fromNybs(lowNyb, highNyb)];
}

// an arithmetic encoded output buffer
// writes to a byte buffer, but minimizing bit writes according to external models
class AEOutput {
public:
  std::vector<uint8_t> buffer; // completed bytes, ready to commit (write to file)

  // widen the encodable range if our range is narrowing without moving upper bits
  // (if we didn't widen the range, this would reduce/remove our ability to distinguish symbols)
  bool widenRangeWithRoll() {
    if (this->low >= arithn::cfourth && this->high < arithn::c3fourth) {
      ++this->roll;
      this->low  -= arithn::cfourth;
      this->high -= arithn::cfourth;

      this->low  = (this->low << 1) & arithn::cmax;
      this->high = ((this->high << 1) | 1) & arithn::cmax;
      return true;
    }
    return false;
  }

  // shift upper bits out of low and high ends of range, assuming we're not accumulating a roll
  // (if the encoder range narrows enough that we risk precision loss, we'll start accumulating rolls)
  void rescaleNoRoll() {
    while (true) {
      nybshift_t ns = rangeShiftNybble(this->low, this->high);
      uint8_t    bc = nsCount(ns);

      if (PRIV_HCFREGION_LIKELY(bc > 0)) {
        putbits(nsBits(ns), bc);
        this->low  = (this->low << bc) & arithn::cmax;
        this->high = ((this->high << bc) | ((1<<bc)-1)) & arithn::cmax;
      } else if (widenRangeWithRoll()) {
        rescaleRoll();
        return;
      } else {
        return;
      }
    }
  }

  // shift bits out of low/high ends of the range accounting for rolls
  void rescaleRoll() {
    while (true) {
      nybshift_t ns = rangeShiftNybble(this->low, this->high);
      uint8_t    bc = nsCount(ns);

      if (PRIV_HCFREGION_LIKELY(bc > 0)) {
        uint8_t bs = nsBits(ns);
        if ((bs & 1) != 0) {
          put1roll(this->roll);
        } else {
          put0roll(this->roll);
        }
        this->roll = 0;
        putbits(uint8_t(bs >> 1), uint8_t(bc - 1));
        this->low  = (this->low << bc) & arithn::cmax;
        this->high = ((this->high << bc) | ((1<<bc)-1)) & arithn::cmax;
        rescaleNoRoll();
      } else if (!widenRangeWithRoll()) {
        return;
      }
    }
  }

  // encode a symbol as a partition low/high out of an interval
  // e.g. a bool value of 'true' with false:25% and true:75% could be written with:
  //                write(25, 100, 100)
  //
  void write(arithn::code clow, arithn::code chigh, arithn::code cinterval) {
    assert(clow < chigh);
    assert(cinterval > 0);
    assert(cinterval <= arithn::fmax);
    assert((chigh - clow) <= cinterval);
    assert(chigh <= cinterval);

    arithn::code r = this->high - this->low + 1;

    this->high = this->low + (r * chigh / cinterval) - 1;
    this->low  = this->low + (r * clow  / cinterval);

    assert(this->low <= this->high);

    // we need to account for overflow here and shift out bits as our range narrows
    // if high's most significant bit (MSB) goes from 1 to 0, we can shift out a 0
    // likewise if low's MSB goes from 0 to 1, we can shift out a 1
    // and we need to account for the case of near convergence where range precision is lost (accumulating a "roll" if the convergence is 01111... or 10000...)
    if (this->roll > 0) {
      rescaleRoll();
    } else {
      rescaleNoRoll();
    }
  }

  // force all state to output (to shut down further communication)
  void complete() {
    // flush interval state
    if (this->low < arithn::cfourth) {
      put0roll(this->roll+1);
    } else {
      put1roll(this->roll+1);
    }
    this->roll=0;

    // add a landing pad for the terminal statement
    put0roll(32);

    // flush the bit buffer up to its index
    if (this->bitidx > 0) {
      this->buffer.push_back(this->bits[0]);
      uint8_t k=1;
      while (this->bitidx > 8) {
        this->bitidx = uint8_t(this->bitidx - 8);
        this->buffer.push_back(this->bits[k++]);
      }
    }
    memset(this->bits, 0, sizeof(this->bits));
    this->bitidx = 0;
  }

  void startNextSegment(const SegmentHeader& header) {
    // flush encoder state
    if (this->low < arithn::cfourth) {
      put0roll(this->roll+1);
    } else {
      put1roll(this->roll+1);
    }
    if (this->bitidx > 0) {
      this->buffer.push_back(this->bits[0]);
      uint8_t k=1;
      while (this->bitidx > 8) {
        this->bitidx = uint8_t(this->bitidx - 8);
        this->buffer.push_back(this->bits[k++]);
      }
    }
    memset(this->bits, 0, sizeof(this->bits));
    this->bitidx = 0;
    this->buffer.push_back(0);
    this->buffer.push_back(0);

    writeHeader(&this->buffer, header);

    // now begin a new encoder state at the new position
    this->low  = 0;
    this->high = arithn::cmax;
    this->roll = 0;
  }
private:
  arithn::code low  = 0;            // current state of low end of range (implicit 0 bits follow)
  arithn::code high = arithn::cmax; // current state of high end of range (implicit 1 bits follow)
  uint32_t     roll = 0;            // count of accumulated roll decisions (to avoid precision loss when range narrows before low/high cross their upper bit boundaries)

  uint8_t bits[8] = {0,0,0,0,0,0,0,0}; // bit buffer to accumulate updates prior to pushing them out to 'buffer'
  uint8_t bitidx  = 0;                 // bit index into 'bits'

  void flushbits() {
    this->buffer.insert(this->buffer.end(), this->bits, this->bits + sizeof(this->bits));
    memset(this->bits, 0, sizeof(this->bits));
    this->bitidx = 0;
  }

  void putbits(uint8_t v, uint8_t bc) {
    uint32_t byi       = this->bitidx>>3;     // floor(bitindex/8), the head byte index in 'bits'
    uint8_t  usedBitc  = this->bitidx&0x07;   // the low 3 bits of the bit index, where in the head byte to write
    uint8_t  availBitc = uint8_t(8-usedBitc); // the 'free space' in the head byte (prior to writing 'bc' bits)

    this->bits[byi] = uint8_t(this->bits[byi] | uint8_t(v << usedBitc)); // should be 'bits[byi] |= v << usedBitc' but this is what repeated truncation warn/error messages came down to
    if (bc > availBitc) {
      ++byi;

      if (byi == sizeof(this->bits)) {
        flushbits();
        this->bits[0] = uint8_t(this->bits[0] | uint8_t(v >> availBitc));
        this->bitidx = uint8_t(bc-availBitc);
      } else {
        this->bits[byi] = uint8_t(this->bits[byi] | uint8_t(v >> availBitc));
        this->bitidx = uint8_t(this->bitidx + bc);
      }
    } else {
      this->bitidx = uint8_t(this->bitidx + bc);
      if (this->bitidx == 8*sizeof(this->bits)) {
        flushbits();
      }
    }
  }
  void putzeroes(arithn::code bitc) {
    // advance the bit index bitc times (unwritten bits are assumed to be 0)
    // flush bits as many times as necessary to bring the new index in range
    static const uint32_t lim = 8*sizeof(this->bits);
    arithn::code newc = arithn::code(this->bitidx) + bitc;
    while (newc >= lim) {
      flushbits();
      newc -= lim;
    }
    this->bitidx = uint8_t(newc);
  }
  void putones(arithn::code bitc) {
    // output 1s for the remainder of bitc/8
    uint8_t odds = uint8_t(bitc&0x07);
    putbits(uint8_t((1<<odds)-1), odds);
    bitc -= odds;
    // output blocks of 8 1s, floor(bitc/8) times
    while (bitc > 0) {
      putbits(0xFF, 8);
      bitc -= 8;
    }
  }
  void put0roll(uint32_t roll) {
    switch (roll) {
    case 0: putbits(  0, 1); break; // 0
    case 1: putbits(  2, 2); break; // 01
    case 2: putbits(  6, 3); break; // 011
    case 3: putbits( 14, 4); break; // 0111
    case 4: putbits( 30, 5); break; // 01111
    case 5: putbits( 62, 6); break; // 011111
    case 6: putbits(126, 7); break; // 0111111
    case 7: putbits(254, 8); break; // 01111111
    default:
      putbits(254, 8);
      putones(roll-7);
      break;
    }
  }
  void put1roll(uint32_t roll) {
    switch (roll) {
    case 0: putbits(1, 1); break; // 1
    case 1: putbits(1, 2); break; // 10
    case 2: putbits(1, 3); break; // 100
    case 3: putbits(1, 4); break; // 1000
    case 4: putbits(1, 5); break; // 10000
    case 5: putbits(1, 6); break; // 100000
    case 6: putbits(1, 7); break; // 1000000
    case 7: putbits(1, 8); break; // 10000000
    default:
      putbits(1, 8);
      putzeroes(roll-7);
      break;
    }
  }
};

// provide an escape path for reads that hit EOF prematurely
struct input_at_eof {
};

// an input buffer for arithmetic encoded data
class AEInput {
public:
  // a user function to supply bytes from an input source
  // returns false when the source has terminated
  typedef std::function<bool(std::vector<uint8_t>*)> ReadBytesF;
  AEInput(const ReadBytesF& readBytes) : readBytes(readBytes) {
    // init start of message bits
    assert(arithn::cbits % 8 == 0);
    for (arithn::code k = 0; k < arithn::cbits; k += 8) {
      this->value = (this->value << 8) | getbits(8);
    }
  }

  bool eof() const {
    // we're at EOF when there are no more bit batches to load
    // and we've read all of the last batch
    return this->atEOF && this->bitidx == this->bitc;
  }

  // update local state assuming that the given sub-range of an interval was read
  // (mirrors updates in the writer for the same model)
  void shift(arithn::code clow, arithn::code chigh, arithn::code cinterval) {
    assert(chigh > clow);
    assert(cinterval > 0);
    assert(cinterval <= arithn::fmax);
    assert((chigh - clow) <= cinterval);
    assert(chigh <= cinterval);

    arithn::code r = this->high - this->low + 1;

    this->high = this->low + (r * chigh / cinterval) - 1;
    this->low  = this->low + (r * clow  / cinterval);

    assert(this->low <= this->high);

    while (true) {
      nybshift_t ns = rangeShiftNybble(this->low, this->high);
      uint8_t    bc = nsCount(ns);

      if (bc == 0) {
        if (this->low >= arithn::cfourth && this->high < arithn::c3fourth) {
          this->low   -= arithn::cfourth;
          this->high  -= arithn::cfourth;
          this->value -= arithn::cfourth;
          bc = 1;
        } else {
          break;
        }
      }

      this->low   =  (this->low   << bc)                & arithn::cmax;
      this->high  = ((this->high  << bc) | ((1<<bc)-1)) & arithn::cmax;
      this->value = ((this->value << bc) | getbits(bc)) & arithn::cmax;

      assert(this->low <= this->value);
      assert(this->value <= this->high);
    }
  }

  // normalize the current value for the given interval
  // (this shifts it to a local model space where we can decide what symbol is represented)
  arithn::code svalue(arithn::code cinterval) const {
    arithn::code r = this->high - this->low + 1;
    return ((this->value - this->low + 1) * cinterval - 1) / r;
  }

  // how many bits have we read so far?
  size_t bitsRead() const {
    return this->accBitsRead + this->bitidx;
  }

  void startNextSegment(SegmentHeader* header) {
    // flush bits to terminate the segment
    size_t db = this->bitidx % 8;
    if (db < 7) {
      getbits(uint8_t(8 - db));
    } else {
      getbits(uint8_t(1));
      getbits(uint8_t(8));
    }

    // read the segment header
    readHeader(header,
      [this](uint8_t* bs, size_t c) {
        for (; c != 0; --c) {
          *(bs++) = reverseBits(this->getbits(uint8_t(8)), 8);
        }
      }
    );

    // reset the decoder state and prepare to read the next segment from scratch
    this->low   = 0;
    this->high  = arithn::cmax;
    this->value = 0;
    for (arithn::code k = 0; k < arithn::cbits; k += 8) {
      this->value = (this->value << 8) | getbits(8);
    }
  }

  // reset the decoder state, should only be done after flushing model state and jumping to a new segment
  void reset() {
    this->bitidx = this->bitc = 0;
    this->bits.clear();
    this->atEOF = false;

    this->low   = 0;
    this->high  = arithn::cmax;
    this->value = 0;
    for (arithn::code k = 0; k < arithn::cbits; k += 8) {
      this->value = (this->value << 8) | getbits(8);
    }
  }
private:
  ReadBytesF           readBytes;
  std::vector<uint8_t> bits;                // a buffered window in the total bitstream
  size_t               bitidx      = 0;     // current read position (as bit index) in the total bitstream
  size_t               bitc        = 0;     // total count of bits in the window (just bits.size()*8)
  bool                 atEOF       = false; // have we reached the end of the stream?
  size_t               accBitsRead = 0;     // keep track of how many bits have been read in all (accBitsRead+bitidx=total bits read)

  // load some more bits, error if reading past EOF
  void loadBytes() {
    if (this->atEOF) {
      // nothing but nonsense past EOF
      throw input_at_eof();
    } else {
      this->accBitsRead += this->bitc;
      this->atEOF        = !this->readBytes(&this->bits);
      this->bitidx       = 0;
      this->bitc         = this->bits.size() * 8;
    }
  }

  // read and consume one bit out of the stream
  bool getbit() {
    // if we've finished a batch, load a fresh one first
    if (this->bitidx == this->bitc) {
      loadBytes();
    }

    // now just get out the next bit
    bool r = this->bits[this->bitidx>>3] & (1 << (this->bitidx&0x7));
    ++this->bitidx;
    return r;
  }

  // read and consume <=8 bits out of the stream
  uint8_t getbits(uint8_t bc) {
    if (bc == 1) return getbit() ? 1 : 0;

    assert(bc > 0 && bc <= 8);
    static const uint8_t selBits[] = { 0x0, 0x1, 0x3, 0x7, 0xf, 0x1f, 0x3f, 0x7f, 0xff }; // (1 << bits) - 1, ie: a bit vector to select bc bits, for all bc

    // if we've just finished a batch, load a fresh one first
    if (this->bitidx == this->bitc) {
      loadBytes();
    }

    auto    byteIdx   = this->bitidx >> 3;
    uint8_t curByte   = this->bits[byteIdx];
    uint8_t usedBits  = uint8_t(this->bitidx & 0x7);
    uint8_t availBits = uint8_t(8 - usedBits);

    if (availBits >= bc) {
      // best case, we have the bits to read already!
      this->bitidx += size_t(bc);
      return reverseBits(uint8_t(curByte >> usedBits) & selBits[bc], bc);
    } else {
      // we have a torn read across bytes
      // take what's left in the current byte, then the remainder out of the next byte
      // if there is no next byte, we have to load more bytes first
      uint8_t result     = uint8_t(curByte >> usedBits);
      uint8_t bitsToTake = uint8_t(bc - availBits);

      if (PRIV_HCFREGION_LIKELY(++byteIdx < this->bits.size())) {
        curByte = this->bits[byteIdx];
        this->bitidx += size_t(bc);
      } else {
        loadBytes();
        curByte = this->bits[0];
        this->bitidx = size_t(bitsToTake);
      }

      return reverseBits(uint8_t(result | ((curByte & selBits[bitsToTake]) << availBits)), bc);
    }
  }

  // arithmetic encoder state
  arithn::code low   = 0;             // the current low end of the encoder range (implicit 0s follow)
  arithn::code high  = arithn::cmax;  // the current high end of the encoder range (implicit 1s follow)
  arithn::code value = 0;             // the current head bits of the actual encoded value (the next bits of the message, in the low/high range)
};

/*******************************************************
 *
 * value modeling : count statistics and map past observations to future expectations on a per-type basis (treating values independently)
 *
 *******************************************************/

class BoolModel {
public:
  void write(AEOutput* out, bool s) {
    if (s) {
      out->write(0, this->tc, this->c);
      addTrue();
    } else {
      out->write(this->tc, this->c, this->c);
      addFalse();
    }
  }
  void read(AEInput* in, bool* s) {
    *s = read(in);
  }
  bool read(AEInput* in) {
    if (in->svalue(this->c) < this->tc) {
      in->shift(0, this->tc, this->c);
      addTrue();
      return true;
    } else {
      in->shift(this->tc, this->c, this->c);
      addFalse();
      return false;
    }
  }
  void reset() {
    this->tc = 1;
    this->c  = 2;
  }
  tym::Ptr<size_t> memoryUsed() const {
    return tym::prim<size_t>(sizeof(*this));
  }
private:
  arithn::freq tc = 1;
  arithn::freq c  = 2;

  void rescale() {
    static constexpr arithn::freq ReflectPeriod = arithn::fmax_pre_esc;
    if (PRIV_HCFREGION_UNLIKELY(this->c == ReflectPeriod)) {
      this->tc = this->tc >> 1;
      this->c  = this->c  >> 1;
      if (this->tc == 0) {
        this->tc = 1;
      }
    }
  }
  void addTrue()  { rescale(); ++this->tc; ++this->c; }
  void addFalse() { rescale();             ++this->c; }
};

// try to minimize memory usage in arrays (where we know that sizes will be small)
// (the only reason that we use this instead of std::vector is its smaller memory footprint)
template <typename T>
class __attribute__((__packed__)) small_array {
public:
  small_array() : sz(0), vs(nullptr) { }
  small_array(const small_array<T>& rhs) : sz(rhs.sz), vs(copy(rhs.sz, rhs.vs, rhs.sz)) {
  }
  ~small_array() { clear(); }
  size_t size() const { return size_t(this->sz); }
  void emplace_back(T&& t) {
    uint8_t nsz = uint8_t(this->sz + 1);
    T* nvs = copy(nsz, this->vs, this->sz);
    nvs[this->sz] = std::move(t);
    delete[] this->vs;
    this->vs = nvs;
    this->sz = nsz;
  }
  void push_back(T t) {
    uint8_t nsz = uint8_t(this->sz + 1);
    T* nvs = copy(nsz, this->vs, this->sz);
    nvs[this->sz] = t;
    delete[] this->vs;
    this->vs = nvs;
    this->sz = nsz;
  }
  T& operator[](size_t i) {
    return this->vs[i];
  }
  const T& operator[](size_t i) const {
    return this->vs[i];
  }
  small_array<T>& operator=(const small_array<T>& rhs) {
    if (&rhs == this) return *this;
    delete[] this->vs;
    this->sz = rhs.sz;
    this->vs = copy(this->sz, rhs.vs, this->sz);
    return *this;
  }
  void clear() {
    delete[] this->vs;
    this->vs = nullptr;
    this->sz = 0;
  }
  void resize(uint8_t nsz) {
    if (nsz <= sz) {
      this->sz = nsz;
    } else {
      auto nvs = copy(nsz, this->vs, this->sz);
      clear();
      this->vs = nvs;
      this->sz = nsz;
    }
  }
private:
  uint8_t sz = 0;
  T*      vs = nullptr;

  static T* copy(uint8_t allocsz, T* rhs, uint8_t sz) {
    T* n = new T[allocsz];
    for (uint8_t i = 0; i < sz; ++i) {
      n[i] = std::move(rhs[i]);
    }
    return n;
  }
};

// a simple small vector for sparse counts (for cumulative sums in frequency counts to come)
template <typename T>
struct small_sparse_vector {
public:
  T get(uint16_t i) const {
    if (i < this->minIndex) {
      return T();
    } else {
      i = uint16_t(i - this->minIndex);
      return (i < this->xs.size()) ? this->xs[i] : T();
    }
  }
  void incr(uint16_t i, T dx) {
    if (i < this->minIndex) {
      std::vector<T> nxs;
      nxs.resize(this->minIndex - i);
      nxs[0] = dx;
      for (size_t j = 1; j < nxs.size(); ++j) {
        nxs[j] = 0;
      }
      this->xs.insert(this->xs.begin(), nxs.begin(), nxs.end());
      this->minIndex = i;
    } else {
      i = uint16_t(i - this->minIndex);
      if (i >= this->xs.size()) {
        this->xs.resize(i + 1);
      }
      this->xs[i] = T(this->xs[i] + dx);
    }
  }
  void rescale(T truncBits = 1) {
    for (auto& x : this->xs) {
      x = T(x >> truncBits);
    }
  }
  void resize(uint16_t size) {
    if (size < this->logicalSize) {
      auto diff = uint16_t(this->logicalSize - size);
      if (diff >= this->xs.size()) {
        this->xs.clear();
      } else {
        this->xs.resize(this->xs.size() - diff);
      }
    }
    this->logicalSize = size;
  }
  uint16_t size() const {
    return this->logicalSize;
  }
  bool anyEq(T x) const {
    if (x == T() && this->xs.size() < this->logicalSize) {
      return true;
    }
    for (auto tx : this->xs) {
      if (tx == x) {
        return true;
      }
    }
    return false;
  }
  size_t memoryUsed() const {
    return sizeof(*this) + sizeof(T) * this->xs.size();
  }
private:
  uint16_t minIndex = 0;
  uint16_t logicalSize = 0;
  std::vector<T> xs;
};

// maintain a set of cumulative frequencies
// so that lookup and addition are fast enough to do continually
//
// internally this is represented in a "fenwick tree", making cumsum queries and updates O(log n)
// logically, the cumsum up to a symbol is found in the tree by following the path from the symbol to the root (adding as we go)
// however, we don't explicitly construct this tree
// instead, we say that the "parent" of a symbol is another symbol -- just setting the least significant bit to 0
// this way there are O(log n) parents (e.g. 11111111b -> 11111110b -> 11111100b -> ... 00000000b)
//
// when we want to update the frequency of a specific symbol (rather than query the cumsum up to that symbol),
// then the situation is reversed -- rather than adding all of the sums up to the given index,
// then we need to add to the sums _after_ the given index
//
// to find the indexes for successor nodes, we just add the least significant bit to the symbol until we pass the number of symbols
// (e.g. 00001010 -> 00001100 -> 00010000 -> 00100000)
template <typename Symbol = uint16_t, typename Count = arithn::freq, typename ExpandedCode = arithn::freq>
class CumFreqs {
public:
  CumFreqs() {
    this->counts.resize(1);
  }
  void incr(Symbol symbol, Count dc = 1) {
    ++symbol;
    for (; symbol < this->counts.size(); symbol = next(symbol)) {
      this->counts.incr(symbol, dc);
    }
    this->tot = Count(this->tot + dc);
  }
  Count cfreqAt(Symbol symbol) const {
    if (symbol == 0) return 0;
    arithn::freq s = 0;
    for (; symbol != 0; symbol = parent(symbol)) {
      s += this->counts.get(symbol);
    }
    return Count(s);
  }
  bool range(Symbol symbol, ExpandedCode* low, ExpandedCode* high) const {
    if (symbol == 0) {
      *low = 0;
      *high = cfreqAt(Symbol(symbol + 1));
    } else {
      // *low = cfreqAt(symbol); *high = cfreqAt(symbol + 1);
      Symbol nsym = Symbol(symbol + 1);
      Count diff = 0;
      for (; nsym > symbol; nsym = parent(nsym)) {
        diff = Count(diff + Count(this->counts.get(nsym)));
      }
      Count symsum = 0;
      for (; symbol > nsym; symbol = parent(symbol)) {
        symsum = Count(symsum + Count(this->counts.get(symbol)));
      }
      diff = Count(diff - symsum);
      for (; symbol != 0; symbol = parent(symbol)) {
        symsum = Count(symsum + Count(this->counts.get(symbol)));
      }
      *low = symsum;
      *high = symsum + diff;
    }
    return *low < *high;
  }
  void find(ExpandedCode value, Symbol* symbol, ExpandedCode* low, ExpandedCode* high) const {
    if (this->counts.size() > 0) {
      Symbol i = 0, j = Symbol(pow2Ceil(this->counts.size() - 1));
      for (; j > 0; j = Symbol(j >> 1)) {
        auto k = Symbol(i + j);
        if (k < this->counts.size()) {
          auto v = this->counts.get(k);
          if (v <= value) {
            value -= v;
            i = k;
          }
        }
      }
      *symbol = i;
      range(i, low, high);
    } else {
      *symbol = 0;
      *low    = 0;
      *high   = 0;
    }
  }
  void resize(Symbol c) {
    this->counts.resize(Symbol(c + 1));
    this->tot = cfreqAt(size());
  }
  Symbol size() const {
    return Symbol(this->counts.size() - 1);
  }
  Count sum() const {
    return this->tot;
  }
  void rescale(Count truncBits = 1) {
    this->counts.rescale(truncBits);
    this->tot = cfreqAt(size());
  }
  size_t memoryUsed() const {
    return sizeof(*this) + this->counts.memoryUsed();
  }
  bool anyZero() const {
    return this->counts.anyEq(0);
  }
private:
  small_sparse_vector<Count> counts;
  Count tot = 0;

  Symbol parent(Symbol i) const {
    // to get the parent index, clear the least significant bit
    // e.g. parent(22) = 20, because:
    //     10110b - (10110b & 01010b) = 10110b - 00010b = 10100b = 20
    return Symbol(i - (i & (-i)));
  }
  Symbol next(Symbol i) const {
    // to get the next index, increment the least significant bit
    // e.g. next(22) = 24, because:
    //     10110b + (10110b & 01010b) = 10110b + 00010b = 11000b = 24
    return Symbol(i + (i & (-i)));
  }
};

// normalize a sequence of frequencies to be safely in bounds for arithmetic coding
inline arithn::freq sumFreq(const std::vector<arithn::freq>& freqs) {
  arithn::freq r = 0;
  for (auto freq : freqs) {
    r += freq;
  }
  return r;
}
inline std::vector<arithn::freq> normalizeFreqs(std::vector<arithn::freq> freqs) {
  if (freqs.size() > arithn::fmax) {
    freqs.resize(arithn::fmax);
  }
  arithn::freq sum = sumFreq(freqs);
  while (sum > arithn::fmax) {
    for (auto& freq : freqs) {
      uint64_t sfreq = (uint64_t(freq) * uint64_t(arithn::fmax)) / uint64_t(sum);
      freq = std::max<arithn::freq>(1, arithn::freq(sfreq));
    }
    sum = sumFreq(freqs);
  }
  return freqs;
}

template <typename SymT>
class VarSetModel {
public:
  typedef uint32_t index_t;
  typedef uint16_t symbol;

  VarSetModel(size_t initialSize = 1) : initialSize(initialSize > 0 ? initialSize : 1) {
    reset();
  }
  VarSetModel(const std::vector<arithn::freq>& symbolFreqs) {
    auto fs = normalizeFreqs(symbolFreqs);
    this->freqs.resize(fs.size());
    for (symbol s = 0; s < symbol(fs.size()); ++s) {
      this->freqs.incr(s, fs[s]);
    }
    maybeEncodeEscape();
  }

  // add a new symbol to the set
  void extendSet() {
    symbol s = symbol(this->freqs.size());
    this->freqs.resize(s + 1);
    this->freqs.incr(s);
  }

  void write(AEOutput* out, SymT s) {
    auto c = static_cast<symbol>(s);
    assert(c != esc());
    arithn::code clow, chigh;
    if (this->freqs.range(c, &clow, &chigh)) {
      out->write(clow, chigh, this->freqs.sum());
    } else {
      this->freqs.range(esc(), &clow, &chigh);
      out->write(clow, chigh, this->freqs.sum());
      out->write(static_cast<arithn::code>(c), static_cast<arithn::code>(c)+1, escRange());
    }
    add(c);
  }

  void read(AEInput* in, SymT* s) {
    symbol c=0;
    arithn::code clow=0, chigh=0;
    this->freqs.find(in->svalue(this->freqs.sum()), &c, &clow, &chigh);
    in->shift(clow, chigh, this->freqs.sum());

    if (c == esc()) {
      c = static_cast<symbol>(in->svalue(escRange()));
      in->shift(arithn::code(c), arithn::code(c+1), escRange());
    }

    *s = static_cast<SymT>(c);
    add(c);
  }

  void reset() {
    this->freqs = CumFreqsT();
    this->freqs.resize(symbol(this->initialSize));
    this->hasEsc = false;
    maybeEncodeEscape();
  }

  tym::Ptr<size_t> memoryUsed() const {
    return tym::prim<size_t>(this->freqs.memoryUsed());
  }
private:
  typedef uint16_t count_t;
  typedef CumFreqs<uint16_t, count_t, arithn::freq> CumFreqsT;
  CumFreqsT freqs;
  size_t initialSize = 1;
  static const arithn::freq rebuildAt = arithn::fmax_pre_esc;

  void add(symbol s) {
    if (PRIV_HCFREGION_UNLIKELY(this->freqs.sum() >= rebuildAt)) {
      this->freqs.rescale();
      maybeEncodeEscape();
      if (this->hasEsc) {
        this->freqs.incr(esc());
      }
    }
    this->freqs.incr(s);
  }

  // escape symbol handling (used iff some valid symbols currently have probability 0)
  bool hasEsc = false;

  symbol esc() const {
    return symbol(this->freqs.size() - 1);
  }
  uint32_t escRange() const {
    return uint32_t(this->freqs.size() - (this->hasEsc ? 1 : 0));
  }
  void maybeEncodeEscape() {
    bool hasZeroes = this->freqs.anyZero();
    if (this->hasEsc && !hasZeroes) {
      this->freqs.resize(uint16_t(this->freqs.size() - 1));
      this->hasEsc = false;
    } else if (!this->hasEsc && hasZeroes) {
      this->freqs.resize(uint16_t(this->freqs.size() + 1));
      this->freqs.incr(esc());
      this->hasEsc = true;
    }
  }
};

template <typename SymT, uint16_t MaxSymbol>
class FinSetModel : public VarSetModel<SymT> {
public:
  FinSetModel() : VarSetModel<SymT>(MaxSymbol + 1) {
  }
};

typedef FinSetModel<uint8_t, 0xff> ByteModel;

// a "counter" model of a numeric value by its representation as independent bytes
// not a great way to encode a value, but adequate as a last resort
template <typename SymT>
class ByteSeqModel {
public:
  ByteSeqModel() {
  }
  void write(AEOutput* out, SymT s) {
    auto b = reinterpret_cast<const uint8_t*>(&s);
    for (uint8_t i = 0; i < sizeof(SymT); ++i) {
      this->ms[i].write(out, *b);
      ++b;
    }
  }
  void read(AEInput* in, SymT* s) {
    auto b = reinterpret_cast<uint8_t*>(s);
    for (uint8_t i = 0; i < sizeof(SymT); ++i) {
      this->ms[i].read(in, b);
      ++b;
    }
  }
  void reset() {
    for (uint8_t i = 0; i < sizeof(SymT); ++i) {
      this->ms[i].reset();
    }
  }
  tym::Ptr<size_t> memoryUsed() const {
    tym::Ptrs<size_t> ps;
    for (uint8_t i = 0; i < sizeof(SymT); ++i) {
      ps.push_back(this->ms[i].memoryUsed());
    }
    return tym::tup(ps);
  }
private:
  typedef std::array<ByteModel, sizeof(SymT)> Ms;
  Ms ms;
};

// take care of the boilerplate of holding on to per-value counts and producing a score
//   * what we're trying to do here is count how many of each value is observed (up to a maximum number of values)
//   * we should see that sequences with few high frequency values are scored higher than ones with many low frequency values
template <typename T, uint8_t MaxSamples = 64>
class ScoredSequence {
public:
  void observe(T x) {
    if (!incrementCount(x)) {
      // if we didn't find this value to increment,
      // then we have to insert it at the end (with the lowest frequency)
      if (this->valueCount < MaxSamples) {
        uint8_t i = this->valueCount;
        this->values[i] = x;
        this->freqs[i] = 1;
        ++this->valueCount;
      } else {
        this->values.back() = x;
        this->freqs.back() = 1;
      }
    }
  }

  size_t score() const {
    if (this->valueCount > 1) {
      size_t ssum = 0;
      for (uint8_t i = 0; i < this->valueCount; ++i) {
        ssum += this->freqs[i] * this->freqs[i];
      }
      return ssum / size_t(this->valueCount);
    } else {
      // somehow we observed 1 or 0 values,
      // in which case this model can be expected to compress values to ~0 bits
      // so you can't get better than that
      return std::numeric_limits<size_t>::max();
    }
  }
  void reset() {
    for (uint8_t i = 0; i < this->values.size(); ++i) {
      this->values[i] = T();
      this->freqs[i] = 0;
    }
    this->valueCount = 0;
  }

  std::array<T,        MaxSamples> values;
  std::array<uint32_t, MaxSamples> freqs;
  uint8_t                          valueCount = 0;

  bool incrementCount(T x) {
    for (uint8_t i = 0; i < this->valueCount; ++i) {
      if (x == this->values[i]) {
        ++this->freqs[i];

        // while we're incrementing this value, should we swap it with its predecessor?
        // (this will also keep the sequence in descending order)
        if (i > 0 && this->freqs[i - 1] < this->freqs[i]) {
          uint8_t j = uint8_t(i - 1);
          std::swap(this->values[i], this->values[j]);
          std::swap(this->freqs[i], this->freqs[j]);
        }

        return true;
      }
    }
    return false;
  }
};

// naively score byte sequences as independent series
template <typename T, uint8_t MaxSamples = 64>
class ScoredByteSeqModel {
public:
  void observe(T x) {
    for (uint8_t i = 0; i < sizeof(T); ++i) {
      this->scores[i].observe(reinterpret_cast<const uint8_t*>(&x)[i]);
    }
  }
  size_t score() const {
    size_t s = 0;
    for (uint8_t i = 0; i < sizeof(T); ++i) {
      s += this->scores[i].score();
    }
    return s;
  }
  void reset() {
    for (uint8_t i = 0; i < sizeof(T); ++i) {
      this->scores[i].reset();
      this->ms[i].reset();
    }
  }
  void chosen() {
  }

  template <typename DefaultModel>
  void write(AEOutput* out, T s, DefaultModel&) {
    auto b = reinterpret_cast<const uint8_t*>(&s);
    for (uint8_t i = 0; i < sizeof(T); ++i) {
      this->ms[i].write(out, *b);
      ++b;
    }
  }
  template <typename DefaultModel>
  void read(AEInput* in, T* s, DefaultModel&) {
    auto b = reinterpret_cast<uint8_t*>(s);
    for (uint8_t i = 0; i < sizeof(T); ++i) {
      this->ms[i].read(in, b);
      ++b;
    }
  }
  tym::Ptr<size_t> memoryUsed() const {
    tym::Ptrs<size_t> ps;
    for (uint8_t i = 0; i < sizeof(T); ++i) {
      ps.push_back(this->ms[i].memoryUsed());
    }
    return tym::tup(ps);
  }
private:
  typedef ScoredSequence<uint8_t, uint8_t(MaxSamples / sizeof(T))> ByteScore;
  typedef std::array<ByteScore, sizeof(T)> Scores;
  Scores scores;

  typedef std::array<ByteModel, sizeof(T)> Ms;
  Ms ms;
};

// a scored model that just internalizes typical values
template <typename T, uint8_t MaxSamples = 64>
class InternValueModel : public ScoredSequence<T, MaxSamples> {
public:
  template <typename DefaultModel>
  void write(AEOutput* out, T x, DefaultModel& def) {
    if (auto s = findID(x)) {
      this->id.write(out, s);
    } else {
      this->id.write(out, 0);
      def.write(out, x, def);
      add(x);
    }
  }
  template <typename DefaultModel>
  void read(AEInput* in, T* x, DefaultModel& def) {
    uint8_t s = 0;
    this->id.read(in, &s);
    if (s > 0) {
      *x = fromID(s);
    } else {
      def.read(in, x, def);
      add(*x);
    }
  }
  void chosen() {
    // copy observed values to our local state (which can be larger)
    initValues(&this->values[0], this->valueCount);
  }
  void reset() {
    ScoredSequence<T, MaxSamples>::reset();
    this->id.reset();
    this->internedValues = small_array<T>();
  }
private:
  ByteModel id; // 0 iff an out of bounds value, else an ID up to 255 for such a value

  // keep a local array of typical values, which we've interned here
  small_array<T> internedValues;

  void initValues(T* values, size_t sz) {
    if (sz < this->internedValues.size()) {
      // if these new values are already a prefix of our current expectations,
      // then don't bother re-initializing since the effect would just be to truncate
      for (uint8_t i = 0; i < this->internedValues.size(); ++i) {
        if (size_t(i) == sz) {
          // we got this far, everything in the input is already accounted for
          return;
        } else if (values[i] != this->internedValues[i]) {
          // there's some value we don't know or is at a different priority
          // allow the reset to proceed
          break;
        }
      }
    }

    auto len = uint8_t(std::min<size_t>(255, sz));
    this->internedValues.resize(len);
    for (uint8_t i = 0; i < len; ++i) {
      this->internedValues[i] = values[i];
    }
  }
  uint8_t findID(T x) const {
    for (uint8_t i = 0; i < this->internedValues.size(); ++i) {
      if (x == this->internedValues[i]) {
        return uint8_t(i + 1);
      }
    }
    return 0;
  }
  T fromID(uint8_t s) const {
    return this->internedValues[s - 1];
  }
  void add(T x) {
    if (this->internedValues.size() == 255) {
      // can't add any more, drop it
    } else {
      this->internedValues.push_back(x);
    }
  }
};

// a scored model that computes self-differences in time
template <typename DiffModel, typename T, uint8_t MaxSamples = 64>
class DiffValueModel {
public:
  void observe(T x) {
    this->s.observe(T(x - this->obvLastValue));
    this->obvLastValue = x;
  }
  size_t score() const {
    return this->s.score();
  }
  void reset() {
    this->s.reset();
    this->lastValue = T();
    this->obvLastValue = T();
    this->m.reset();
  }
  void chosen() {
    this->lastValue = this->obvLastValue;
  }

  template <typename DefaultModel>
  void write(AEOutput* out, T x, DefaultModel&) {
    write(out, x);
  }
  template <typename DefaultModel>
  void read(AEInput* in, T* x, DefaultModel&) {
    *x = read(in);
  }

  // a convenience for code using this model outside an ensemble
  void write(AEOutput* out, T x) {
    m.write(out, T(x - this->lastValue));
    this->lastValue = x;
  }
  T read(AEInput* in) {
    T diff = T();
    m.read(in, &diff);
    this->lastValue = T(this->lastValue + diff);
    return this->lastValue;
  }
  tym::Ptr<size_t> memoryUsed() const {
    return tym::prim<size_t>(sizeof(*this));
  }
private:
  ScoredSequence<T, MaxSamples> s;
  T lastValue = T();
  T obvLastValue = T();
  DiffModel m;
};

// a 'model selection' model, which takes any number of models (plus a default model)
// and uses whichever one scores observations highest to actually do the encoding
template <size_t Index, size_t Len, typename T, typename DefaultModel, typename ... Models>
struct ModelSet {
  typedef ModelSet<Index + 1, Len, T, DefaultModel, Models...> Next;

  // write the given value through the selected model
  // (or the default model if index is out of bounds)
  static void write(tuple<Models...>& ms, uint16_t index, DefaultModel& dm, AEOutput* out, T x) {
    if (index == Index) {
      ms.template at<Index>().write(out, x, dm);
    } else {
      Next::write(ms, index, dm, out, x);
    }
  }
  // read a value through the selected model
  // (or the default model if index is out of bounds)
  static void read(tuple<Models...>& ms, uint16_t index, DefaultModel& dm, AEInput* in, T* x) {
    if (index == Index) {
      ms.template at<Index>().read(in, x, dm);
    } else {
      Next::read(ms, index, dm, in, x);
    }
  }

  // observe a sample of the value type (this should change the self score kept by each model)
  static void observe(tuple<Models...>& ms, T x) {
    ms.template at<Index>().observe(x);
    Next::observe(ms, x);
  }
  // sample the score for each model
  static void score(const tuple<Models...>& ms, std::array<size_t, Len>* scores) {
    (*scores)[Index] = ms.template at<Index>().score();
    Next::score(ms, scores);
  }
  static size_t scoreAt(const tuple<Models...>& ms, size_t index, const DefaultModel& dm) {
    if (index == 0) {
      return ms.template at<Index>().score();
    } else {
      return Next::scoreAt(ms, index - 1, dm);
    }
  }
  // reset scores across all models
  static void reset(tuple<Models...>& ms, DefaultModel& dm) {
    ms.template at<Index>().reset();
    Next::reset(ms, dm);
  }
  // notify a model that it has been chosen (to allow it to internalize its observations if necessary)
  static void chosen(tuple<Models...>& ms, uint16_t index) {
    if (index == Index) {
      ms.template at<Index>().chosen();
    } else {
      Next::chosen(ms, index);
    }
  }

  // just for debugging, query the amount of memory used for each model
  static void memoryUsed(const tuple<Models...>& ms, std::vector<tym::Ptr<size_t>>* fields) {
    fields->push_back(ms.template at<Index>().memoryUsed());
    Next::memoryUsed(ms, fields);
  }
  static tym::Ptr<size_t> memoryUsed(const tuple<Models...>& ms) {
    std::vector<tym::Ptr<size_t>> fields;
    memoryUsed(ms, &fields);
    return tym::tup<size_t>(fields);
  }
};
template <size_t Len, typename T, typename DefaultModel, typename ... Models>
struct ModelSet<Len, Len, T, DefaultModel, Models...> {
  static void             write(tuple<Models...>&, uint16_t, DefaultModel& m, AEOutput* out, T x) { m.write(out, x, m); }
  static void             read(tuple<Models...>&, uint16_t, DefaultModel& m, AEInput* in, T* x)   { m.read(in, x, m); }
  static void             observe(tuple<Models...>&, T)                                           { }
  static void             score(const tuple<Models...>&, std::array<size_t, Len>*)                { }
  static size_t           scoreAt(const tuple<Models...>&, size_t, const DefaultModel& m)         { return m.score(); }
  static void             reset(tuple<Models...>&, DefaultModel& m)                               { m.reset(); }
  static void             chosen(tuple<Models...>&, uint16_t)                                     { }
  static void             memoryUsed(const tuple<Models...>&, std::vector<tym::Ptr<size_t>>*)     { }
  static tym::Ptr<size_t> memoryUsed(const tuple<Models...>&)                                     { return tym::unit<size_t>(); }
};

template <typename Config, typename DefaultModel, typename ... Models>
class EnsembleModel {
public:
  typedef typename Config::value_type T;

  void write(AEOutput* out, T x) {
    MSet::write(this->models, this->modelIndex, this->defaultModel, out, x);
    observe(x);
  }
  void read(AEInput* in, T* x) {
    MSet::read(this->models, this->modelIndex, this->defaultModel, in, x);
    observe(*x);
  }
  size_t score() const {
    return MSet::scoreAt(this->models, this->modelIndex, this->defaultModel);
  }
  void reset() {
    MSet::reset(this->models, this->defaultModel);
    this->samples = 0;
    this->modelIndex = sizeof...(Models);
  }
  tym::Ptr<size_t> memoryUsed() const {
    return
      tym::rec<size_t>({
        {"default",   this->defaultModel.memoryUsed()},
        {"optmodels", MSet::memoryUsed(this->models)}
      });
  }
private:
  typedef ModelSet<0, sizeof...(Models), T, DefaultModel, Models...> MSet;

  DefaultModel     defaultModel;
  tuple<Models...> models;
  uint16_t         samples = 0;
  uint16_t         modelIndex = sizeof...(Models);

  void observe(T x) {
    this->defaultModel.observe(x);
    MSet::observe(this->models, x);
    if (++this->samples == Config::observation_period) {
      // now that we've read enough samples to consider selecting a non-default model
      // choose the model giving the highest score (if that score is higher than the default)
      this->samples = 0;
      std::array<size_t, sizeof...(Models)> scores;
      MSet::score(this->models, &scores);

      size_t maxScore = this->defaultModel.score();
      this->modelIndex = sizeof...(Models);
      for (uint16_t i = 0; i < uint16_t(sizeof...(Models)); ++i) {
        if (maxScore < scores[i]) {
          maxScore = scores[i];
          this->modelIndex = i;
          MSet::chosen(this->models, this->modelIndex);
        }
      }
    }
  }
};

// encode whole numbers by heuristically selecting an effective lower-dimensional representation, falling back on a byte-wise encoding only if necessary
template <typename IntegerT>
struct WholeNumberConfig {
  typedef IntegerT value_type;
  static const uint16_t observation_period = 256;
};

template <typename IntegerT>
class WholeNumberModel : public EnsembleModel<WholeNumberConfig<IntegerT>, ScoredByteSeqModel<IntegerT>, InternValueModel<IntegerT>, DiffValueModel<ByteSeqModel<IntegerT>, IntegerT>> {
public:
  tym::Ptr<size_t> memoryUsed() const {
    return
      tym::rec<size_t>({
        { "local", tym::prim<size_t>(sizeof(*this)) }
      });
  }
};

// dictionary-code substring values
// (expecting redundancy between/within strings)
struct LZWNode {
  typedef uint32_t Code;
  Code code = 0;
  char c    = '\0';

  typedef small_array<LZWNode> SuccessorsV;
  typedef std::unique_ptr<SuccessorsV> Successors;
  Successors successors = Successors(new SuccessorsV());

  // follow 'str' as far as possible from this node, returning the suffix where we stopped
  // output:
  //   foundCode - the ID of the followed path (readers keep a dictionary to map these back to strings)
  //   nextCode  - the variable to increment when allocating a new code
  const char* follow(const char* str, Code* foundCode, Code* nextCode) {
    *foundCode = this->code;
    auto* node = this;
    for (; *str != '\0'; ++str) {
      char strc = *str;

      // try to find this char under the current node
      size_t childi = 0;
      size_t childn = node->successors->size();

      for (; childi < childn; ++childi) {
        auto* nnode = &(*node->successors)[childi];
        if (nnode->c == strc) {
          // we have a match at this char, so we can at least return the path up to this point
          // (but we will continue the outer loop and see if we can follow more chars from here)
          *foundCode = nnode->code;
          node = nnode;
          break;
        }
      }

      // if we couldn't find this char, we've reached the end of the longest path we can code for this string
      // but take a tentative step for next time by making a new branch for this char
      if (childi == childn) {
        // the dictionary implicitly holds length-1 branches from the root for each single char
        // so we have a special case here to avoid allocating a new code if we're just forced to reify the char node
        Code ncode = 0;
        if (node->code == 0) {
          ncode = static_cast<Code>(static_cast<uint8_t>(strc));
        } else {
          ncode = *nextCode;
          ++*nextCode;
        }
        assert(ncode < *nextCode);

        LZWNode n;
        n.code = ncode;
        n.c = strc;
        node->successors->emplace_back(std::move(n));
        break;
      }
    }
    return str;
  }
};
class StringModel {
public:
  void write(AEOutput* o, const char* v) {
    while (*v != '\0') {
      Code c = 0;
      v = this->wdict.follow(v, &c, &this->nextCode);
      if (c > 0) {
        this->code.write(o, c);
      }
    }
    this->code.write(o, 0);
  }
  void write(AEOutput* o, const std::string& v) {
    write(o, v.c_str());
  }
  void read(AEInput* i, std::string* v) {
    std::ostringstream ss;
    Code substr = 0;
    Code prevsubstr = 0;
    size_t len = 0;
    while (true) {
      this->code.read(i, &substr);

      bool inferredExtension = false; // handles the degenerate case where the encoder is one step ahead
      if (substr == 0) {
        break;
      } else if (substr < 256) {
        ss << static_cast<char>(substr);
        len += 1;
      } else {
        Code rdictidx = substr - 256;
        if (rdictidx == this->rdict.size()) {
          // this happens when there is repetition like 'abababa'
          // the encoder is one step ahead in this case, but we can
          // infer that it must have added the previous symbol plus its first char
          assert(prevsubstr > 0);
          rdictExtend(prevsubstr, prevsubstr);
          inferredExtension = true;
        }
        assert(rdictidx < this->rdict.size());
        ss << this->rdict[rdictidx];
        len += this->rdict[rdictidx].size();
      }
      if (prevsubstr > 0 && !inferredExtension) {
        rdictExtend(prevsubstr, substr);
      }
      prevsubstr = substr;
      checkArrLen(len);
    }
    *v = ss.str();
  }
  void reset() {
    this->wdict = LZWNode();
    this->nextCode = 256;
    this->code = ByteSeqModel<Code>();
    this->rdict.clear();
  }
  tym::Ptr<size_t> memoryUsed() const {
    size_t r = sizeof(LZWNode)*(this->nextCode-256) + sizeof(this->rdict);
    for (const auto& s : this->rdict) {
      r += sizeof(s) + s.size();
    }
    return
      tym::rec<size_t>({
        { "dictionary", tym::prim<size_t>(r) },
        { "code", this->code.memoryUsed() }
      });
  }
private:
  typedef LZWNode::Code Code;
  LZWNode wdict;
  Code    nextCode = 256; // assume that codes 0-255 code for length-1 strings with the same char code (e.g.: 32=" ")

  ByteSeqModel<Code> code;

  std::vector<std::string> rdict;
  void rdictExtend(Code prev, Code next) {
    this->rdict.push_back(codedString(prev));
    this->rdict.back() += firstCharOf(next);
  }
  const std::string& codedString(Code c) {
    static std::vector<std::string> singleChars = singleCharStrings();
    if (c < 256) {
      return singleChars[c];
    } else {
      c -= 256;
      assert(c < this->rdict.size());
      return this->rdict[c];
    }
  }
  static std::vector<std::string> singleCharStrings() {
    std::vector<std::string> result;
    result.resize(256);
    for (size_t i = 1; i < result.size(); ++i) {
      result[i] = std::string(1, static_cast<char>(i));
    }
    return result;
  }
  char firstCharOf(Code c) {
    if (c < 256) {
      return static_cast<char>(c);
    } else {
      c -= 256;
      assert(c < this->rdict.size());
      assert(this->rdict[c].size() > 0);
      return this->rdict[c][0];
    }
  }
};

// when recording a numeric value, there may be some number of previously recorded values of the same type, which could be correlated
// if a correlation is observed, switch to encoding differences with it
template <typename T, T defaultValue, template <typename U> class EscModel, uint16_t ReflectionPeriod = (1 << 8)>
class LinkedIntValueModel {
public:
  // disallow copying because we rely on unique persistent value pointers per linked-value model instance
  LinkedIntValueModel() = default;
  LinkedIntValueModel(const LinkedIntValueModel&) = delete;
  LinkedIntValueModel& operator=(const LinkedIntValueModel&) const = delete;

  void link(const LinkedIntValueModel* predecessor) {
    for (auto p : this->predecessors) {
      if (p == &predecessor->lastValue) {
        // this is dumb, we're already tracking this predecessor
        return;
      }
    }
    this->predecessors.push_back(&predecessor->lastValue);
    this->matchCounts.push_back(0);
    this->narrowingDiffSums.push_back(0);
  }

  void write(AEOutput* o, T x) {
    if (this->copyPredecessor < this->predecessors.size()) {
      // we expect an exact copy from this predecessor
      // first check for it, and if it matches then encode the match
      // else encode that it didn't match, and then encode the actual value
      if (x == *this->predecessors[this->copyPredecessor]) {
        this->correlated.write(o, true);
      } else {
        this->correlated.write(o, false);
        this->m.write(o, x);
      }
    } else if (this->diffPredecessor < this->predecessors.size()) {
      ST d = ST(ST(x) - ST(*this->predecessors[this->diffPredecessor]));
      this->diffM.write(o, d);
    } else {
      // no correlation expected, just encode the value as-is
      this->m.write(o, x);
    }

    // note that this value was just observed, it might affect correlation statistics
    observe(x);
  }
  void read(AEInput* i, T* x) {
    if (this->copyPredecessor < this->predecessors.size()) {
      // we expect a correlation,
      // first check for it, and if it matches then we've got it
      // else if it didn't match, then read the value
      if (this->correlated.read(i)) {
        *x = *this->predecessors[this->copyPredecessor];
      } else {
        this->m.read(i, x);
      }
    } else if (this->diffPredecessor < this->predecessors.size()) {
      ST diff = ST();
      this->diffM.read(i, &diff);
      *x = T(diff + ST(*this->predecessors[this->diffPredecessor]));
    } else {
      // no correlation expected, just read the value as-is
      this->m.read(i, x);
    }

    // note that this value was just observed, it might affect correlation statistics
    observe(*x);
  }
  void reset() {
    this->m                 = EscModel<T>();
    this->diffM             = EscModel<ST>();
    this->lastValue         = defaultValue;
    this->predecessors      = Predecessors();
    this->matchCounts       = MatchCounts();
    this->narrowingDiffSums = NarrowingDiffSums();
    this->copyPredecessor   = uint16_t(-1);
    this->diffPredecessor   = uint16_t(-1);
    this->observations      = 0;

    this->correlated.reset();
  }
  tym::Ptr<size_t> memoryUsed() const {
    return
      tym::rec<size_t>({
        { "local", tym::prim<size_t>(sizeof(*this) - (sizeof(this->m) + sizeof(this->diffM) + sizeof(this->correlated))) },
        { "escape", this->m.memoryUsed() },
        { "diff", this->diffM.memoryUsed() },
        { "corr", this->correlated.memoryUsed() },
        { "preds", tym::prim<size_t>(sizeof(decltype(this->predecessors[0]))*this->predecessors.size()) },
        { "matches", tym::prim<size_t>(sizeof(decltype(this->matchCounts[0]))*this->matchCounts.size()) },
        { "ndiffs", tym::prim<size_t>(sizeof(decltype(this->narrowingDiffSums[0]))*this->narrowingDiffSums.size()) }
      });
  }
private:
  typedef typename std::make_signed<T>::type ST;
  typedef std::vector<const T*>              Predecessors;
  typedef std::vector<uint16_t>              MatchCounts;
  typedef std::vector<T>                     NarrowingDiffSums;

  EscModel<T>       m;                              // encode the value by itself, when no correlation is expected
  EscModel<ST>      diffM;                          // encode the value as a difference from its maximally correlated predecessor
  BoolModel         correlated;                     // if a correlation is predicted, encodes whether the correlation matches
  T                 lastValue = defaultValue;       // the last value recorded by this model (predicts downstream linked values)
  Predecessors      predecessors;                   // all of the preceding linked values under consideration to predict this one
  MatchCounts       matchCounts;                    // how many times each predecessor has matched this value
  NarrowingDiffSums narrowingDiffSums;              // how much each predecessor differs from this value over the sampling period
  uint16_t          copyPredecessor = uint16_t(-1); // initially we guess no perfect link
  uint16_t          diffPredecessor = uint16_t(-1); // initially we guess no diff link
  uint16_t          observations = 0;               // keep track of how many observations we've made (every period we will consider whether to correlate)

  void observe(T x) {
    // check for correlations with predecessors
    for (size_t i = 0; i < this->predecessors.size(); ++i) {
      this->matchCounts[i] = uint16_t(this->matchCounts[i] + (x == *this->predecessors[i] ? 1 : 0));

      ST d = ST(ST(x) - ST(*this->predecessors[i]));
      this->narrowingDiffSums[i] = T(this->narrowingDiffSums[i] + T(d < 0 ? (d*-1) : d));
    }

    // if at the review period, internalize observations (possibly pick a correlated predecessor)
    if (++this->observations == ReflectionPeriod) {
      if (!this->predecessors.empty()) {
        size_t mc = maxIndex(this->matchCounts);
        if (this->matchCounts[mc] >= ReflectionPeriod / 2) {
          // heuristically decide that if we're correlated to a value >=50% of the time, it's worth expecting
          this->copyPredecessor = uint16_t(mc);
        } else {
          this->copyPredecessor = uint16_t(-1);
          // we can't eliminate the data with an exact copy, try for a narrowing difference instead
          size_t dc = minIndex(this->narrowingDiffSums);
          static const T maxPredDiff = std::numeric_limits<T>::max() >> (sizeof(T) * 8 / 2);
          if (this->narrowingDiffSums[dc] < maxPredDiff) {
            this->diffPredecessor = uint16_t(dc);
          } else {
            this->diffPredecessor = uint16_t(-1);
          }
        }
        // reset counts/sums for the next observation period
        std::fill(this->matchCounts.begin(), this->matchCounts.end(), 0);
        std::fill(this->narrowingDiffSums.begin(), this->narrowingDiffSums.end(), 0);
      }
      this->observations = 0;
    }

    // mark the last recorded value for successors linked downstream
    this->lastValue = x;
  }

  // move somewhere more generic?
  template <typename X>
  static size_t maxIndex(const std::vector<X>& xs) {
    if (xs.size() == 0) return 0;

    size_t midx = 0;
    X m = xs[0];
    for (size_t i = 1; i < xs.size(); ++i) {
      if (m < xs[i]) {
        m = xs[i];
        midx = i;
      }
    }
    return midx;
  }
  template <typename X>
  static size_t minIndex(const std::vector<X>& xs) {
    if (xs.size() == 0) return 0;

    size_t midx = 0;
    X m = xs[0];
    for (size_t i = 1; i < xs.size(); ++i) {
      if (xs[i] < m) {
        m = xs[i];
        midx = i;
      }
    }
    return midx;
  }
};

/*******************************************************
 *
 * model higher level types:
 *   IdentifierSet          : a finite set of IDs that can grow, encode minimally (used for ctor IDs, symbol IDs, ...)
 *   ScopedIdentifierSetRef : a reduced scope for a value out of an IdentifierSet (e.g. prediction of one ctor ID after another has been recorded)
 *   SymbolSet              : a set of interned strings (writing IDs to avoid repeating long strings)
 *   TimestampSet           : timestamps likely written in increasing order (so can encode as small differences)
 *   TypeSet                : type descriptions packed more efficiently than default encoding
 *   UserIDModel            : a finite set of IDs for use by users
 *
 *******************************************************/

// write values in a monotonically increasing counter
// (used for log statement constructor IDs)
class IdentifierSet {
public:
  uint32_t fresh() {
    auto r = this->count;
    ++this->count;
    return r;
  }
  void reset(uint32_t c) {
    this->count = c;
  }
  uint32_t size() const {
    return this->count;
  }
  void write(AEOutput* o, uint32_t c) {
    this->code.write(o, c);
  }
  uint32_t read(AEInput* i) {
    uint32_t r=0;
    this->code.read(i, &r);
    return r;
  }
  tym::Ptr<size_t> memoryUsed() const {
    return this->code.memoryUsed();
  }
private:
  uint32_t                   count = 0;
  WholeNumberModel<uint32_t> code;
};

// write identifier values out of a (dynamic) universe in a nested scope
// this will be used to predict log statement constructors given the previous recorded log statement
// (the type documents intent but otherwise stores identically to a regular whole number model)
class ScopedIdentifierSetRef {
public:
  ScopedIdentifierSetRef() {
  }
  void write(AEOutput* o, uint32_t id) {
    this->code.write(o, id);
  }
  uint32_t read(AEInput* i) {
    uint32_t r = 0;
    this->code.read(i, &r);
    return r;
  }
  void reset() {
    this->code.reset();
  }
  tym::Ptr<size_t> memoryUsed() const {
    return this->code.memoryUsed();
  }
private:
  WholeNumberModel<uint32_t> code;
};

// uncompressed files have a concept of "symbol" to reduce storage cost by not duplicating string contents
// strings are LZW-coded in compressed files, so encoding as strings will basically have the same effect
// (but going through a symbol type will merge all of those statistics together, which will be helpful if
// files wind up storing the same string across many log statements, but otherwise might hurt)
class SymbolSet {
public:
  void write(AEOutput* o, const std::string& v) {
    this->stringM.write(o, v);
  }
  void read(AEInput* i, std::string* v) {
    this->stringM.read(i, v);
  }
  void reset() {
    this->stringM.reset();
  }
  tym::Ptr<size_t> memoryUsed() const {
    return this->stringM.memoryUsed();
  }
private:
  StringModel stringM;
};

// write timestamps assuming they typically increase but can have (likely small) inversions
typedef DiffValueModel<ByteSeqModel<uint64_t>, uint64_t> TimestampSet;

// write type descriptions more efficiently than blind byte copying
// type descriptions are logically just a basic recursive variant type,
// it might be worth having a generic representation of these too (most of this code is boilerplate)
class TypeSet {
public:
  void write(AEOutput* o, const ty::desc& t) {
    // don't encode the same type description twice
    // (due to structural type representation, we expect a lot of repeats)
    auto tbs = ty::encoding(t);
    auto c   = this->internedTypeIDs.find(tbs);

    if (c != this->internedTypeIDs.end()) {
      this->ids.write(o, c->second + 1);
      return;
    }
    this->ids.write(o, 0);

    auto id = this->ids.fresh();
    this->internedTypeIDs[tbs] = id;
    this->internedTypes.push_back(t);

    // as a variant, write tag, then depending on tag, write corresponding payload
    unroll(t).caseOf<void>({
      .nat = [&](const ty::Nat& x) {
        write_tctor(o, PRIV_HPPF_TYCTOR_SIZE);
        this->counts.write(o, truncate<uint32_t>(x.x));
      },
      .prim = [&](const ty::Prim& x) {
        write_tctor(o, PRIV_HPPF_TYCTOR_PRIM);
        this->symbols.write(o, x.n);
        if (const auto* rep = some(x.rep)) {
          this->primHasRep.write(o, true);
          write(o, *rep);
        } else {
          this->primHasRep.write(o, false);
        }
      },
      .var = [&](const ty::Var& x) {
        write_tctor(o, PRIV_HPPF_TYCTOR_TVAR);
        this->symbols.write(o, x.n);
      },
      .farr = [&](const ty::FArr& x) {
        write_tctor(o, PRIV_HPPF_TYCTOR_FIXEDARR);
        write(o, x.t);
        write(o, x.len);
      },
      .arr = [&](const ty::Arr& x) {
        write_tctor(o, PRIV_HPPF_TYCTOR_ARR);
        write(o, x.t);
      },
      .variant = [&](const ty::Variant& x) {
        write_tctor(o, PRIV_HPPF_TYCTOR_VARIANT);
        this->counts.write(o, truncate<uint32_t>(x.ctors.size()));
        for (const auto& ctor : x.ctors) {
          this->symbols.write(o, ctor.at<0>());
          this->counts.write(o, ctor.at<1>());
          write(o, ctor.at<2>());
        }
      },
      .record = [&](const ty::Struct& x) {
        write_tctor(o, PRIV_HPPF_TYCTOR_STRUCT);
        this->counts.write(o, truncate<uint32_t>(x.fields.size()));
        for (const auto& field : x.fields) {
          this->symbols.write(o, field.at<0>());
          this->counts.write(o, uint32_t(field.at<1>()));
          write(o, field.at<2>());
        }
      },
      .recursive = [&](const ty::Recursive& x) {
        write_tctor(o, PRIV_HPPF_TYCTOR_RECURSIVE);
        this->symbols.write(o, x.x);
        write(o, x.t);
      },
      .fn = [&](const ty::Fn& x) {
        write_tctor(o, PRIV_HPPF_TYCTOR_TABS);
        this->counts.write(o, truncate<uint32_t>(x.args.size()));
        for (const auto& argn : x.args) {
          this->symbols.write(o, argn);
        }
        write(o, x.t);
      },
      .app = [&](const ty::App& x) {
        write_tctor(o, PRIV_HPPF_TYCTOR_TAPP);
        write(o, x.f);
        this->counts.write(o, truncate<uint32_t>(x.args.size()));
        for (const auto& arg : x.args) {
          write(o, arg);
        }
      }
    });
  }
  ty::desc read(AEInput* i) {
    // is this a type description we've already read?
    uint32_t c = this->ids.read(i);

    if (c > 0) {
      assert(c <= this->internedTypes.size());
      return this->internedTypes[c - 1];
    }

    auto id = this->ids.fresh();

    // it's a new type description
    ty::desc t;

    switch (read_tctor(i)) {
    case PRIV_HPPF_TYCTOR_PRIM: {
      std::string n;
      this->symbols.read(i, &n);
      bool rep=false;
      this->primHasRep.read(i, &rep);
      if (rep) {
        t = ty::prim(n, read(i));
      } else {
        t = ty::prim(n);
      }
      break;
    }
    case PRIV_HPPF_TYCTOR_TVAR: {
      std::string n;
      this->symbols.read(i, &n);
      t = ty::var(n);
      break;
    }
    case PRIV_HPPF_TYCTOR_FIXEDARR: {
      auto a = read(i);
      auto n = read(i);
      t = ty::array(a, n);
      break;
    }
    case PRIV_HPPF_TYCTOR_ARR: {
      auto a = read(i);
      t = ty::array(a);
      break;
    }
    case PRIV_HPPF_TYCTOR_VARIANT: {
      uint32_t n=0;
      this->counts.read(i, &n);

      ty::VariantCtors ctors;
      for (uint32_t k = 0; k < n; ++k) {
        std::string n;
        this->symbols.read(i, &n);
        uint32_t c=0;
        this->counts.read(i, &c);
        auto ft = read(i);
        ctors.push_back(ty::VariantCtor(n, c, ft));
      }
      t = ty::variant(ctors);
      break;
    }
    case PRIV_HPPF_TYCTOR_STRUCT: {
      uint32_t n=0;
      this->counts.read(i, &n);

      ty::StructFields fields;
      for (uint32_t k = 0; k < n; ++k) {
        std::string fn;
        this->symbols.read(i, &fn);
        uint32_t o=0;
        this->counts.read(i, &o);
        auto ft = read(i);
        fields.push_back(ty::StructField(fn, int(o), ft));
      }
      t = ty::record(fields);
      break;
    }
    case PRIV_HPPF_TYCTOR_SIZE: {
      uint32_t n=0;
      this->counts.read(i, &n);
      t = ty::nat(n);
      break;
    }
    case PRIV_HPPF_TYCTOR_TAPP: {
      auto f = read(i);

      uint32_t n=0;
      this->counts.read(i, &n);
      std::vector<ty::desc> args;
      for (uint32_t k = 0; k < n; ++k) {
        args.push_back(read(i));
      }
      t = ty::appc(f, args);
      break;
    }
    case PRIV_HPPF_TYCTOR_RECURSIVE: {
      std::string n;
      this->symbols.read(i, &n);

      auto b = read(i);

      t = ty::recursive(n, b);
      break;
    }
    case PRIV_HPPF_TYCTOR_TABS: {
      uint32_t n=0;
      this->counts.read(i, &n);
      std::vector<std::string> args;
      for (uint32_t k = 0; k < n; ++k) {
        std::string a;
        this->symbols.read(i, &a);
        args.push_back(a);
      }
      auto b = read(i);

      t = ty::fnc(args, b);
      break;
    }
    default:
      assert(false && "Invalid type description, internal error");
      break;
    }

    if (this->internedTypes.size() <= id) {
      this->internedTypes.resize(id + 1);
    }
    this->internedTypes[id] = t;
    return t;
  }
  void reset() {
    this->ids.reset(0);
    this->internedTypeIDs.clear();
    this->internedTypes.clear();
    this->ctor.reset();
    this->symbols.reset();
    this->primHasRep.reset();
    this->counts.reset();
  }
  tym::Ptr<size_t> memoryUsed() const {
    return
      tym::rec<size_t>({
        { "locals",         tym::prim<size_t>(sizeof(*this) - (sizeof(this->ids) + sizeof(this->ctor))) },
        { "ids",            this->ids.memoryUsed() },
        { "ctor",           this->ctor.memoryUsed() },
        { "interned-types", tym::prim<size_t>(sizeof(decltype(*this->internedTypes.begin()))*this->internedTypes.size()) },
        { "interned-ids",   tym::prim<size_t>(sizeof(decltype(*this->internedTypeIDs.begin()))*this->internedTypeIDs.size()) }
      });
  }
private:
  IdentifierSet                            ids;
  std::map<std::vector<uint8_t>, uint32_t> internedTypeIDs;
  std::vector<ty::desc>                    internedTypes;

  FinSetModel<uint8_t, 0x9> ctor;
  const std::vector<uint32_t> packedCtorIDs() {
    static const std::vector<uint32_t> vs = {
      PRIV_HPPF_TYCTOR_PRIM, PRIV_HPPF_TYCTOR_TVAR,    PRIV_HPPF_TYCTOR_FIXEDARR,
      PRIV_HPPF_TYCTOR_ARR,  PRIV_HPPF_TYCTOR_VARIANT, PRIV_HPPF_TYCTOR_STRUCT,
      PRIV_HPPF_TYCTOR_SIZE, PRIV_HPPF_TYCTOR_TAPP,    PRIV_HPPF_TYCTOR_RECURSIVE,
      PRIV_HPPF_TYCTOR_TABS
    };
    return vs;
  }

  void write_tctor(AEOutput* o, uint32_t c) {
    for (uint8_t i = 0; i < packedCtorIDs().size(); ++i) {
      if (c == packedCtorIDs()[i]) {
        this->ctor.write(o, i);
        return;
      }
    }
    throw std::runtime_error("Invalid type code, internal error");
  }
  uint32_t read_tctor(AEInput* i) {
    uint8_t c=0;
    this->ctor.read(i, &c);
    if (c < packedCtorIDs().size()) {
      return this->packedCtorIDs()[c];
    }
    throw std::runtime_error("Read invalid type code, internal error");
  }

  // intern all strings in type descriptions (type names, field names, constructor names, variable names, etc)
  SymbolSet symbols;

  // primitive types add a bool to indicate whether they close over a representation type
  BoolModel primHasRep;

  // structs/variants/fncalls have counts
  WholeNumberModel<uint32_t> counts;
};

// user-defined identifiers
class UserIDModel {
public:
  void write(AEOutput* o, uint32_t c) {
    if (c == this->last) {
      this->repeat.write(o, true);
    } else {
      this->repeat.write(o, false);
      this->code.write(o, c);
      this->last = c;
    }
  }
  uint32_t read(AEInput* i) {
    bool rep = false;
    this->repeat.read(i, &rep);
    if (!rep) {
      uint32_t id = 0;
      this->code.read(i, &id);
      this->last = id;
    }
    return this->last;
  }
  void reset() {
    this->repeat.reset();
    this->code.reset();
    this->last = 0;
  }
  tym::Ptr<size_t> memoryUsed() const {
    return this->code.memoryUsed();
  }
private:
  BoolModel repeat;
  WholeNumberModel<uint32_t> code;
  uint32_t last = 0;
};

/*******************************************************
 *
 * trace wrapper around modeling and entropy coding (mostly identical to trace I/O without entropy coding)
 *
 *******************************************************/

class writer;
class reader;

// keep track of a fixed window of updating values
// (used to link predecessor to successor events up to a given distance)
template <typename T>
class small_ring_buffer {
public:
  small_ring_buffer(uint8_t len, const T& def = T()) : len(len), widx(0), def(def) {
    for (uint8_t i = 0; i < this->len; ++i) {
      this->buffer[i] = def;
    }
  }
  size_t size() const {
    return size_t(this->len);
  }
  void push(const T& t) {
    if (this->len > 0) {
      this->buffer[this->widx] = t;
      this->widx = uint8_t((uint16_t(this->widx) + 1) % uint16_t(this->len));
    }
  }
  T& operator[](size_t i) {
    return this->buffer[absToLocal(i)];
  }
  const T& operator[](size_t i) const {
    return this->buffer[absToLocal(i)];
  }
  uint8_t next_index() const {
    return this->widx;
  }
  void clear() {
    this->widx = 0;
    for (uint8_t i = 0; i < this->len; ++i) {
      this->buffer[i] = this->def;
    }
  }
private:
  uint8_t len  = 0;
  uint8_t widx = 0;
  T def = T();
  T buffer[256];

  size_t absToLocal(size_t i) const {
    if (this->len == 0) return 0; // anything makes sense out of bounds

    if (i < size_t(this->widx)) {
      return size_t(this->widx) - 1 - i;
    } else {
      return size_t(this->len) - 1 - (i - size_t(this->widx));
    }
  }
};

// a single constructor out of the (implicitly defined) variant for all trace values
// (variant values have a unique constructor ID in the range 1..2^32-1)
typedef uint32_t ctorid;

// eventually we will define a set of model types, a small subset of which will be primitives
// at this point we don't know what those types look like, but we know that they are identified by
// type descriptions, and we will want to link them together (in a network representing the order
// of events recorded in a file)
typedef std::pair<ty::desc, std::vector<void*>> PrimModelsByType;
typedef std::vector<PrimModelsByType> PrimModelsByTypes;

inline void addConnections(PrimModelsByTypes* p, const ty::desc& t, const std::vector<void*>& cs) {
  for (auto& m : *p) {
    if (ty::equivModOffset(t, m.first)) {
      m.second.insert(m.second.end(), cs.begin(), cs.end());
      return;
    }
  }
  p->emplace_back(t, cs);
}
inline const std::vector<void*>& connectionsByType(const ty::desc& t, const PrimModelsByTypes& p) {
  for (const auto& m : p) {
    if (ty::equivModOffset(t, m.first)) {
      return m.second;
    }
  }
  static std::vector<void*> empty;
  return empty;
}

// type-annotated writers for individual constructors
// all wseries<T> are subtypes of a type-erased view wseriesi
class wseriesi {
public:
  wseriesi(ctorid cid) : cid(cid) { }
  virtual ~wseriesi() { }
  ctorid id() const { return this->cid; }

  void follows(wseriesi* pred) {
    if (this->knownPredecessorIDs.insert(pred->id()).second) {
      // this is a new predecessor link
      PrimModelsByTypes connections;
      pred->extractModels(&connections);
      this->linkPredModels(connections);
    }
  }

  // forget predecessor series and model state
  // (this is necessary to start segments fresh so that readers can jump to them)
  void resetState() {
    this->knownPredecessorIDs.clear();
    this->reset();
  }

  // for debugging, describe how much memory is used by the model for this series
  virtual tym::Ptr<size_t> memoryUsed() const = 0;
protected:
  // extract primitive models for linkage to successor series
  virtual void extractModels(PrimModelsByTypes*) = 0;

  // link predecessor models into stored models
  virtual void linkPredModels(const PrimModelsByTypes&) = 0;

  // reset statistics used for encoding this data
  // (this will be necessary to periodically reset and link to interior partitions of data)
  virtual void reset() = 0;
private:
  ctorid cid;
  std::set<ctorid> knownPredecessorIDs; // keep track of which series are predecessors (and prevent multiple insertions of the same series)
};
template <typename T>
class wseries;
template <typename T>
wseries<T>* makeSeriesWriter(writer*, ctorid);

template <typename U>
struct DynamicWriterImpl {
  virtual void write(writer*, const U&) = 0;
  virtual void extractModels(PrimModelsByTypes*) = 0;
  virtual void linkPredModels(const PrimModelsByTypes&) = 0;
  virtual void reset() = 0;
};
template <typename U>
using DynamicWriterImplPtr = std::shared_ptr<DynamicWriterImpl<U>>;
template <typename U>
class dwseries;
template <typename U>
dwseries<U>* makeDynamicSeriesWriter(writer*, ctorid, const DynamicWriterImplPtr<U>&);

// describe the type T (to persist or check a stored type description)
template <typename T>
ty::desc descSeriesType();

// write a trace file
//
//  to write a trace file into 'filename.log', construct a variable like:
//    ctrace::writer w("filename.log");
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
//  specialization of 'ctrace::model<T>' defined (usually users will not need to specialize
//  this to get serialization, but it's an option)
//
//  the buffer is flushed after a write if the buffer size is beyond the configured limit
//
//  to force a buffer flush, just call:
//    w.flush();
//
#define HCTRACE_MAGIC_PREFIX "HCTRACE9"
#define HCTRACE_HEADER_LEN   /*magic*/(sizeof(HCTRACE_MAGIC_PREFIX) - 1) + /*ctor count*/sizeof(uint32_t) + /*lookback*/sizeof(uint8_t)

struct WriterConfig {
  uint8_t  lookback     = 5;                // how many predecessor events in the past should predict successor events?
  uint64_t bufferLimit  = 32 * 4096;        // how large should the buffered data be before we force a flush to disk?
  uint64_t minBatchSize = 10 * 1024 * 1024; // how many bytes do we want in each batch of compressed data? (this determines how much reading can be done in parallel)
};

class writer {
public:
  writer(const std::string& path, const WriterConfig& config = WriterConfig())
  : path(path)
  , config(config)
  , previousCtors(config.lookback)
  {
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

    // write the magic prefix so that readers can identify this file correctly
    static const char   prefix[]  = HCTRACE_MAGIC_PREFIX;
    static const size_t prefixLen = sizeof(prefix) - 1; // ignore the '\0' char

    if (::write(this->fd, prefix, prefixLen) != ssize_t(prefixLen)) {
      raiseSysError("Failed to write prefix to trace file", path);
    }

    // initially start with 0 constructors defined
    // (this is a special area in the file header that will be updated with each new constructor)
    updateConstructorCount(0);

    // record the configured lookback
    if (::write(this->fd, reinterpret_cast<const uint8_t*>(&config.lookback), sizeof(config.lookback)) != ssize_t(sizeof(config.lookback))) {
      raiseSysError("Failed to write lookback to trace file", path);
    }

    // record the first segment header (initially null)
    this->prevSegmentHeaderPos = currentFilePos();
    writeHeader(this->fd, path, this->segment);

    // and predict from the 0th constructor initially
    this->constructorFromPrevious.emplace_back();
  }
  ~writer() {
    for (auto* p : this->seriesRefs) {
      delete p;
    }
    this->seriesRefs.clear();

    hangup();
    ::close(this->fd);
  }

  const std::string& filePath() const {
    return this->path;
  }

  void hangup() {
    terminateStream();
    this->output.complete();
    flush(false);
    for (size_t i = 0; i < this->segment.eventsRecorded.size(); ++i) {
      this->segment.eventsRecorded[i] = this->nextSegment.eventsRecorded[i];
    }
    updatePreviousSegmentHeader(this->prevSegmentHeaderPos, this->segment);
    ::fsync(this->fd);
  }

  void flush(bool resetAtBatchSize = true) {
    const auto& buf = this->output.buffer;
    size_t k = 0;
    while (k < buf.size()) {
      auto n = ::write(this->fd, buf.data() + k, buf.size() - k);
      if (n < 0 || (n == 0 && errno != 0)) {
        raiseSysError("Failed to write to file", this->path);
      }
      k += n;
    }
    this->output.buffer.clear();

    // start a new segment if we get up to the given limit
    this->batchBytesWritten += k;
    if (resetAtBatchSize && this->batchBytesWritten >= this->config.minBatchSize) {
      // close this segment in the output stream and start a new one
      resetStream();
      auto predictNextSegment = this->nextSegment;
      for (uint8_t& bits : predictNextSegment.eventsRecorded) {
        bits = 0xFF;
      }
      this->output.startNextSegment(predictNextSegment);
      auto nextSegmentStart = currentOutputPos() - encodedSize(predictNextSegment);

      // now that the whole previous segment is closed
      // update the previous segment header to point to this segment,
      // and to include the counts and event bits we've observed
      assert(this->segment.eventsRecorded.size() <= this->nextSegment.eventsRecorded.size());
      for (size_t i = 0; i < this->segment.eventsRecorded.size(); ++i) {
        this->segment.eventsRecorded[i] = this->nextSegment.eventsRecorded[i];
      }
      updatePreviousSegmentHeader(nextSegmentStart, this->segment);
      this->segment = this->nextSegment;
      for (size_t i = 0; i < this->segment.eventsRecorded.size(); ++i) {
        this->segment.eventsRecorded[i] = 0;
        this->nextSegment.eventsRecorded[i] = 0;
      }

      // we're now in a new segment, reset statistics to encode from scratch
      reset();
      this->batchBytesWritten = 0;
      ++this->segmentCount;
    }
  }
  void flushAtLimit() {
    if (this->output.buffer.size() >= this->config.bufferLimit) {
      flush();
    }
  }
  size_t bufferLimit() const { return this->config.bufferLimit; }
  void bufferLimit(size_t x) { this->config.bufferLimit = x; flushAtLimit(); }

  // the next available constructor ID
  // it gets ID of size+1 because 0 is reserved for the "new series" constructor
  // (typically not a user API)
  ctorid nextConstructorID() const {
    return ctorid(this->seriesRefs.size() + 1);
  }

  // allocate an object to write just one constructor value
  // (optional size argument just for compatibility with fregion::writer interface)
  template <typename T>
    wseries<T>& series(const std::string& name, size_t unusedSegSize = 0) {
      // allocate this writer and register it with encoders
      // it gets ID of size+1 because 0 is reserved for the "new series" constructor
      auto* p = makeSeriesWriter<T>(this, nextConstructorID());
      registerSeries(name, descSeriesType<T>(), p);
      return *p;
    }

  // allocate an object to write a "dynamically typed" value
  // (the type description must be provided at the point of allocation here)
  // it will work similarly to a statically typed series,
  // but its input can be represented by a homogenous universal type
  template <typename U>
    dwseries<U>& dynamicSeries(const std::string& name, const ty::desc& ty, const DynamicWriterImplPtr<U>& impl) {
      auto* p = makeDynamicSeriesWriter(this, nextConstructorID(), impl);
      registerSeries(name, ty, p);
      return *p;
    }

  // register a generic series writer (not a user API, use series<T>(...) instead)
  void registerSeries(const std::string& name, const ty::desc& ty, wseriesi* p) {
    if (p->id() != nextConstructorID()) {
      throw std::runtime_error("Internal error, write series must be registered in order");
    }

    // write out this type-registration message
    defineNewSeries();
    this->constructors.fresh();
    this->constructorFromPrevious.emplace_back();
    this->symbols.write(&this->output, name);
    this->types.write(&this->output, ty);

    // add the writer for this constructor to the total sequence
    this->seriesRefs.push_back(p);
    this->seriesNames.push_back(name);

    // leave a hint to readers that need to know how many constructors are defined in total
    updateConstructorCount(ctorid(this->seriesRefs.size()));

    // and make space for this event in segment headers
    if (p->id() >= 8 * this->nextSegment.eventsRecorded.size()) {
      size_t reqBitCount = p->id() + 1;
      size_t reqByteCount = (reqBitCount / 8) + (reqBitCount % 8 == 0 ? 0 : 1);
      this->nextSegment.eventsRecorded.resize(reqByteCount);
    }
  }

  // mark the start of a new payload by writing as many bits as are necessary now for a constructor (not a user API)
  // also (if writing a series ID) connect the predecessor series to the new series, in case values from previous series can predict values in successors
  void writeCtor(ctorid id) {
    this->constructorFromPrevious[this->previousCtors[0]].write(&this->output, id);
    if (id) {
      for (size_t i = 0; i < this->previousCtors.size(); ++i) {
        if (auto pid = this->previousCtors[i]) {
          this->seriesRefs[id - 1]->follows(this->seriesRefs[pid - 1]);
        }
      }

      // for the segment header, remember that this event happened at least once
      ctorid bid = id - 1;
      uint8_t& bflags = this->nextSegment.eventsRecorded[bid >> 3];
      bflags = uint8_t(bflags | uint8_t(1 << (bid & 7)));
      ++this->segment.eventCount;
    }
    this->previousCtors.push(id);
  }

  // write a symbol (avoid repeating expensive variable-length strings) (not a user API)
  void writeSymbol(const std::string& x) {
    this->symbols.write(&this->output, x);
  }

  // write the current timestamp (must be monotonic between calls) (not a user API)
  void writeTimestamp(uint64_t ts) {
    this->timestamps.write(&this->output, ts);
  }

  // write an identifier from a given universe ID
  // (at file scope so that statistics can be shared across series)
  void writeID(size_t u, uint32_t id) {
    if (this->identifiers.size() <= u) {
      this->identifiers.resize(u + 1);
    }
    this->identifiers[u].write(out(), id);
  }

  // access the encoding output stream (not a user API)
  AEOutput* out() { return &this->output; }

  // generally only useful for debugging
  const TypeSet& typeSet() const { return this->types; }
  const TimestampSet& timestampSet() const { return this->timestamps; }
  const SymbolSet& symbolSet() const { return this->symbols; }
  const std::vector<std::string>& internalSeriesNames() const { return this->seriesNames; }
  const std::vector<wseriesi*>& internalSeriesRefs() const { return this->seriesRefs; }
  size_t internalSegmentCount() const { return this->segmentCount; }
private:
  std::string   path;                     // the output path for our file
  int           fd = -1;                  // the file descriptor for output
  WriterConfig  config;                   // configuration parameters for prediction and buffer management
  AEOutput      output;                   // entropy coded output buffer
  size_t        batchBytesWritten = 0;    // how many bytes have we written to the output within the current batch?
  uint64_t      prevSegmentHeaderPos = 0; // the previous position waiting for a segment link (we have to patch this when we make a new segment)
  size_t        segmentCount = 1;         // how many segments have we created in this file (default is 1 for the initial segment)
  SegmentHeader segment;                  // header information for the current segment that we're writing into
  SegmentHeader nextSegment;              // information for the next segment to write (event bitmasks have to go here since events can be introduced intrasegment)

  // update the count of constructors defined in the file
  // (this should be rare enough in long runs that the expensive seek/reset on the file is amortized out)
  void updateConstructorCount(uint32_t c) {
    constexpr off_t count_start = sizeof(HCTRACE_MAGIC_PREFIX) - 1;
    if (::lseek(this->fd, count_start, SEEK_SET) == static_cast<off_t>(-1)) {
      raiseSysError("Failed to seek in trace to update constructor count", this->path);
    }
    if (::write(this->fd, reinterpret_cast<const uint8_t*>(&c), sizeof(c)) != sizeof(c)) {
      raiseSysError("Failed to write constructor count to trace file header", this->path);
    }
    if (::lseek(this->fd, 0, SEEK_END) == static_cast<off_t>(-1)) {
      raiseSysError("Failed to reset write position in trace after updating constructor count", this->path);
    }
  }

  // maintain internal links and summary details between segments in the file
  void updatePreviousSegmentHeader(uint64_t newSegmentStart, SegmentHeader header) {
    header.toNextSegment = uint32_t(newSegmentStart - this->prevSegmentHeaderPos);
    assert(this->prevSegmentHeaderPos + header.toNextSegment == newSegmentStart);

    if (::lseek(this->fd, this->prevSegmentHeaderPos, SEEK_SET) == static_cast<off_t>(-1)) {
      raiseSysError("Failed to seek to previous segment link for update", this->path);
    }
    writeHeader(this->fd, this->path, header);
    if (::lseek(this->fd, 0, SEEK_END) == static_cast<off_t>(-1)) {
      raiseSysError("Failed to reset write position in trace after updating segment header", this->path);
    }
    this->prevSegmentHeaderPos = newSegmentStart;
  }
  uint64_t currentFilePos() const {
    off_t offset = ::lseek(this->fd, 0, SEEK_CUR);
    if (offset == static_cast<off_t>(-1)) {
      raiseSysError("Failed to query current file position", this->path);
    }
    return uint64_t(offset);
  }
  uint64_t currentOutputPos() {
    return currentFilePos() + out()->buffer.size();
  }

  // basic control sequences
  // * define a new series (registering a new log statement)
  void defineNewSeries() { writeCtor(0); this->commandType.write(&this->output, 0); }
  // * or record a stream reset (a point where readers can jump to and start decoding with 0 knowledge)
  void resetStream()     { writeCtor(0); this->commandType.write(&this->output, 1); }
  // * or terminate the stream (anything after this will be ignored)
  void terminateStream() { writeCtor(0); this->commandType.write(&this->output, 2); }

  FinSetModel<uint8_t, 0x2>           commandType;             // indicates how to interpret a stream command (new series, reset, hangup)
  IdentifierSet                       constructors;            // keep track of constructor IDs for log statements
  small_ring_buffer<uint32_t>         previousCtors;           // the previously recorded constructor ID
  std::vector<ScopedIdentifierSetRef> constructorFromPrevious; // given the previous constructor (by index), predict the next likely constructor
  TypeSet                             types;                   // keep track of type descriptions (per constructor)
  SymbolSet                           symbols;                 // keep track of internalized symbols
  TimestampSet                        timestamps;              // keep track of timestamps
  std::vector<UserIDModel>            identifiers;             // allow any number of (disjoint) user-defined identifier sets

  // objects allocated to write event values by constructor
  typedef std::vector<wseriesi*> SeriesRefs;
  SeriesRefs seriesRefs;

  typedef std::vector<std::string> SeriesNames;
  SeriesNames seriesNames;

  // reset statistics everywhere to start from scratch, as if recording a fresh file
  // this will let parallel readers jump to different points and decompress immediate
  void reset() {
    this->commandType.reset();
    this->previousCtors.clear();
    this->types.reset();
    this->symbols.reset();
    this->timestamps.reset();
    for (auto& s : this->identifiers) {
      s.reset();
    }
    for (auto& s : this->constructorFromPrevious) {
      s.reset();
    }
    for (auto* s : this->seriesRefs) {
      s->resetState();
    }
  }
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

  void follows(rseriesi* pred) {
    if (this->knownPredecessorIDs.insert(pred->name()).second) {
      // this is a new predecessor link
      PrimModelsByTypes connections;
      pred->extractModels(&connections);
      this->linkPredModels(connections);
    }
  }

  // reset correlations and statistics used for encoding this data
  // (this will be necessary to periodically reset and link to interior partitions of data)
  void resetState() {
    this->knownPredecessorIDs.clear();
    this->reset();
  }
protected:
  friend class reader;

  // extract primitive models for linkage to successor series
  virtual void extractModels(PrimModelsByTypes*) = 0;

  // link predecessor models into stored models
  virtual void linkPredModels(const PrimModelsByTypes&) = 0;

  // reset model-specific statistics
  virtual void reset() = 0;

  // clone the series reader (for use in forked readers)
  virtual rseriesi* fork() = 0;
private:
  std::string sname;
  std::set<std::string> knownPredecessorIDs; // keep track of which series are predecessors (and prevent multiple insertions of the same series)
};
template <typename F>
rseriesi* makeSeriesReader(const std::string&, const F&);

// optionally block while we're waiting for more data to come in
class FileWatch {
public:
  explicit FileWatch(std::string) {
    throw std::runtime_error("FileWatch disabled for this test");
  }
  ~FileWatch() {
  }
  void wait() {
  }
};

// a basic interface for reading data out of a trace file
// (normally this will just defer to the filesystem, but we can also use this to read from e.g. network sources)
HLOG_DEFINE_STRUCT(
  CoroutineDef,
  (ucontext_t*, scheduler),
  (ucontext_t*, reader)
);
HLOG_DEFINE_VARIANT(
  AtEOF,
  (Hangup,  unit),
  (Wait,    unit),
  (Suspend, CoroutineDef)
);
struct ReadTraceData {
  typedef std::shared_ptr<ReadTraceData> Ptr;
  typedef std::vector<Ptr> Ptrs;
  virtual ~ReadTraceData() = default;

  // how many different event types will be defined in this trace data?
  virtual uint32_t expectedSeriesCount() = 0;

  // how many previous events prior to the current event can inform predictions of the current event?
  virtual uint8_t seriesLookback() = 0;

  // read some bytes from the current position
  virtual bool readBytes(std::vector<uint8_t>*) = 0;
  virtual void readExactBytes(uint8_t*, size_t) = 0;

  // (optional) get the current segment header
  virtual SegmentHeader currentSegmentHeader() = 0;

  // (optional) the position of the next segment (the second segment, after the default initial segment)
  //            may be 0 if there is no next segment
  virtual uint64_t nextSegmentPos() = 0;

  // (optional) every segment position starting from (and including) a given position
  virtual std::vector<uint64_t> successorSegmentPositions(uint64_t) = 0;
  virtual std::vector<SegmentHeader> successorSegmentHeaders(uint64_t) = 0;

  // (optional) seek to a segment by position
  virtual void seekToSegmentStart(uint64_t, SegmentHeader*) = 0;

  // (optional) fork a reader across a sequence of segment positions
  virtual Ptrs forkAcrossSegments(const std::vector<uint64_t>&, std::vector<SegmentHeader>*) = 0;

  // (optional) allow overriding of EOF handling behavior
  virtual AtEOF behaviorAtEOF() const = 0;
  virtual void setBehaviorAtEOF(const AtEOF&) = 0;
};
typedef ReadTraceData::Ptr ReadTraceDataPtr;
typedef ReadTraceData::Ptrs ReadTraceDataPtrs;

class ReadTraceDataFromFile : public ReadTraceData {
public:
  ReadTraceDataFromFile(const std::string& path, AtEOF atEOF, size_t bufferSz) : path(path), atEOF(atEOF), bufferSz(bufferSz) {
    this->fd = openFile(path, &this->ctors, &this->lookback, &this->header, &this->nextSeg);
  }
  ~ReadTraceDataFromFile() override {
    ::close(fd);
  }
  uint32_t expectedSeriesCount() override {
    return this->ctors;
  }
  uint8_t seriesLookback() override {
    return this->lookback;
  }
  bool readBytes(int fd, std::vector<uint8_t>* bs) {
    return
      this->atEOF.caseOf<bool>({
        .Hangup = [&](unit) {
          // we don't want to wait for more data
          // if we don't read a full buffer, we're at EOF
          readToBufferSize(fd, bs);
          return bs->size() == this->bufferSz;
        },
        .Wait = [&](unit) {
          // we'd like to tail this file, so if we can't
          // read 1 byte, wait until something comes in
          do {
            readToBufferSize(fd, bs);
            if (bs->size() == 0) {
              waitForFileUpdate();
            }
          }
          while (bs->size() == 0);
          return true;
        },
        .Suspend = [&](CoroutineDef cor) -> bool {
          // we'd like to tail this file, but the user
          // has something else to do, so instead of
          // blocking, just jump to their context
          do {
            readToBufferSize(fd, bs);
            if (bs->size() == 0) {
              if (swapcontext(cor.reader, cor.scheduler) == -1) {
                throw std::runtime_error("Failed to resume scheduler fiber while suspending reading of '" + this->path + "'");
              }
            }
          }
          while (bs->size() == 0);
          return true;
        }
      });
  }
  bool readBytes(std::vector<uint8_t>* bs) override {
    return readBytes(this->fd, bs);
  }
  void readExactBytes(int fd, uint8_t* bs, size_t c) {
    while (c > 0) {
      auto dc = readInto(fd, bs, c);
      if (dc == 0) {
        if (this->atEOF == AtEOF::Hangup(unit())) {
          throw std::runtime_error("Unexpected EOF in '" + this->path + "'");
        } else {
          waitForFileUpdate();
        }
      }
      bs += dc;
      c -= dc;
    }
  }
  void readExactBytes(uint8_t* bs, size_t c) override {
    readExactBytes(this->fd, bs, c);
  }

  SegmentHeader currentSegmentHeader() override {
    return this->header;
  }
  uint64_t nextSegmentPos() override {
    return this->nextSeg;
  }

  std::vector<uint64_t> successorSegmentPositions(uint64_t seg) override {
    std::vector<uint64_t> result;
    for (const auto& header : successorSegmentHeaders(seg)) {
      result.push_back(seg);
      seg += header.toNextSegment;
    }
    return result;
  }
  std::vector<SegmentHeader> successorSegmentHeaders(uint64_t seg) override {
    int tfd = ::open(this->path.c_str(), O_RDONLY, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
    if (tfd < 0) {
      raiseSysError("Failed to re-open file for reading", this->path);
    }
    auto readExactBytes =
      [&](uint8_t* bs, size_t c) {
        auto r = ::read(tfd, reinterpret_cast<uint8_t*>(bs), c);
        if (r != ssize_t(c)) {
          ::close(fd);
          raiseSysError(r == 0 ? "Unexpected EOF in trace file" : "Failed to read from trace file", path);
        }
      };
    try {
      std::vector<SegmentHeader> result;
      while (seg != 0) {
        if (::lseek(tfd, seg, SEEK_SET) == static_cast<off_t>(-1)) {
          raiseSysError("Failed to seek in trace to read segment link", this->path);
        }
        SegmentHeader header;
        readHeader(&header, readExactBytes);
        result.push_back(header);
        uint64_t nextSeg = seg + uint64_t(header.toNextSegment);
        if (nextSeg < seg) {
          throw std::runtime_error("Consistency error in trace file (corrupted file?): expecting forward-only segments, but found segment at " + string::from(seg) + " pointing to " + string::from(nextSeg));
        } else if (header.toNextSegment == 0) {
          break;
        } else {
          seg = nextSeg;
        }
      }
      ::close(tfd);
      return result;
    } catch (...) {
      ::close(tfd);
      throw;
    }
  }

  void seekToSegmentStart(uint64_t seg, SegmentHeader* header) override {
    if (::lseek(this->fd, seg, SEEK_SET) == static_cast<off_t>(-1)) {
      raiseSysError("Failed to seek in trace to segment link", this->path);
    }
    readHeader(&this->header, [this](uint8_t* bs, size_t c) { this->readExactBytes(bs, c); });
    if (header) { *header = this->header; }
  }

  ReadTraceDataPtrs forkAcrossSegments(const std::vector<uint64_t>& segs, std::vector<SegmentHeader>* headers) override {
    ReadTraceDataPtrs result;
    headers->resize(segs.size());
    for (size_t i = 0; i < segs.size(); ++i) {
      result.push_back(ReadTraceDataPtr(new ReadTraceDataFromFile(this->path, AtEOF::Hangup(unit()), this->bufferSz)));
      result.back()->seekToSegmentStart(segs[i], &(*headers)[i]);
    }
    return result;
  }

  AtEOF behaviorAtEOF() const override { return this->atEOF; }
  void setBehaviorAtEOF(const AtEOF& b) override { this->atEOF = b; }
private:
  std::string   path;
  int           fd       = -1;
  AtEOF         atEOF    = AtEOF::Hangup(unit());
  size_t        bufferSz =  0;
  uint32_t      ctors    =  0;
  uint8_t       lookback =  1;
  SegmentHeader header;
  uint64_t      nextSeg  =  0;

  size_t readInto(int fd, uint8_t* bs, size_t c) {
    ssize_t r = ::read(fd, bs, c);
    if (r < 0) {
      raiseSysError("Error while reading ctrace file", this->path);
    }
    return size_t(r);
  }
  void readToBufferSize(int fd, std::vector<uint8_t>* bs) {
    bs->resize(this->bufferSz);
    bs->resize(readInto(fd, bs->data(), bs->size()));
  }

  std::unique_ptr<FileWatch> fileWatch;
  void waitForFileUpdate() {
    if (!this->fileWatch) {
      this->fileWatch = std::unique_ptr<FileWatch>(new FileWatch(this->path));
    }
    this->fileWatch->wait();
  }

  // standard way to read a ctrace file -- open, validate, read header
  // after this part is done, all bits read out of the file are entropy-coded values
  int openFile(const std::string& path, uint32_t* expectedCtors, uint8_t* lookback, SegmentHeader* header, uint64_t* nextSeg) {
    // open the file for reading
    int fd = open(path.c_str(), O_RDONLY, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
    if (fd < 0) {
      raiseSysError("Failed to open file for reading", path);
    }

    auto readExactBytes =
      [fd, this](uint8_t* bs, size_t c) {
        this->readExactBytes(fd, bs, c);
      };

    // read the magic prefix (else it's not a valid trace file)
    constexpr size_t prefixLen = sizeof(HCTRACE_MAGIC_PREFIX) - 1;
    char prefix[prefixLen + 1];

    readExactBytes(reinterpret_cast<uint8_t*>(prefix), prefixLen);
    if (strncmp(prefix, HCTRACE_MAGIC_PREFIX, prefixLen) != 0) {
      // not sure what this is, it's not a compressed trace file
      ::close(fd);
      throw std::runtime_error("Not a valid trace file: " + path);
    }

    // read the expected constructor count (in case we need to know how far to read for all valid definitions)
    readExactBytes(reinterpret_cast<uint8_t*>(expectedCtors), sizeof(*expectedCtors));

    // read the configured lookback (how far back to predict successor events from predecessor events)
    readExactBytes(reinterpret_cast<uint8_t*>(lookback), sizeof(*lookback));

    // read the first segment header, just use it to get the offset to the next segment
    off_t offset = ::lseek(fd, 0, SEEK_CUR);
    if (offset == static_cast<off_t>(-1)) {
      raiseSysError("Failed to query current file position", path);
    }
    assert(offset == HCTRACE_HEADER_LEN);

    readHeader(header, readExactBytes);
    if (header->toNextSegment == 0) {
      *nextSeg = 0;
    } else {
      *nextSeg = uint64_t(offset) + uint64_t(header->toNextSegment);
    }

    // done, we're ready to continue reading
    return fd;
  }
};

// read a trace file
//
//  to read a trace file from 'filename.log', just construct a variable like:
//    ctrace::reader r("filename.log");
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
//  specialization of 'ctrace::model<T>' defined (usually users will not need to specialize
//  this to get serialization, but it's an option)
//
class reader {
public:
  reader(const ReadTraceDataPtr& freader)
  : freader(freader),
    expectedCtors(freader->expectedSeriesCount()),
    input([freader](std::vector<uint8_t>* b) -> bool { return freader->readBytes(b); }),
    nextSegmentPos(freader->nextSegmentPos()),
    previousCtors(freader->seriesLookback())
  {
    // predict from the 0th constructor initially
    this->constructorFromPrevious.emplace_back();

    // copy the current segment header
    this->segment = freader->currentSegmentHeader();
  }
  // start a reader from a file
  // rewrite the EOF behavior after initializing the file to avoid dealing with coroutine resumption in initialization
  reader(const std::string& path, AtEOF atEOF = AtEOF::Hangup(unit()), size_t bufferSz = 32*4096)
  : reader(ReadTraceDataPtr(new ReadTraceDataFromFile(path, atEOF.Hangup() ? atEOF : AtEOF::Wait(unit()), bufferSz)))
  {
    this->freader->setBehaviorAtEOF(atEOF);
  }
  virtual ~reader() {
    for (auto* r : this->seriesRefs) {
      delete r;
    }
    this->seriesRefs.clear();
  }

  // how many constructor definitions are predicted by this file's header?
  uint32_t expectedConstructorCount() const { return this->expectedCtors; }

  // how many constructor definitions have we counted so far?
  uint32_t constructorsDefined() const { return this->constructors.size(); }

  // given the current reader state (which may be many segments in)
  // get segment header information
  const SegmentHeader& segmentHeader() const { return this->segment; }
  uint64_t nextSegmentPosition() const { return this->nextSegmentPos; }
  const SegmentHeader::Bitset& eventsInSegment() const { return this->segment.eventsRecorded; }
  std::vector<uint64_t> nextSegmentPositions() const { return this->freader->successorSegmentPositions(nextSegmentPosition()); }
  std::vector<SegmentHeader> nextSegmentHeaders() const { return this->freader->successorSegmentHeaders(nextSegmentPosition()); }

  std::set<std::string> eventNamesSet(const SegmentHeader::Bitset& bitset) {
    std::set<std::string> result;
    for (size_t i = 0; i < this->seriesRefs.size(); ++i) {
      size_t  b    = i >> 3;
      uint8_t bits = b < bitset.size() ? bitset[b] : 0;
      if (((bits >> (i & 7)) & 1) == 1) {
        result.insert(this->seriesRefs[i]->name());
      }
    }
    return result;
  }

  void seekToSegmentAt(uint64_t seg) {
    // seek to the new file position and load the successor link
    this->freader->seekToSegmentStart(seg, &this->segment);
    this->nextSegmentPos = seg + this->segment.toNextSegment;

    // reset decoder and model state
    this->input.reset();
    reset();
  }
  void seekToInitialSegment() {
    seekToSegmentAt(HCTRACE_HEADER_LEN);
  }

  std::vector<std::shared_ptr<reader>> forkAcrossSegments(const std::vector<uint64_t>& segs) {
    // we can't fork until we've read the type definitions for every event
    while (constructorsDefined() < expectedConstructorCount()) {
      if (!step()) {
        return { };
      }
    }

    // now at least we're certain that all event type definitions are known
    // we can fork this reader state at each requested segment
    std::vector<std::shared_ptr<reader>> result;
    std::vector<SegmentHeader> headers;
    auto forkedReaders = this->freader->forkAcrossSegments(segs, &headers);
    for (size_t i = 0; i < forkedReaders.size(); ++i) {
      const auto& fr = forkedReaders[i];
      SeriesRefs frefs;
      for (auto ref : this->seriesRefs) {
        frefs.push_back(ref->fork());
      }

      result.push_back(fork(fr, frefs));
      auto& f = result.back();
      f->constructorFromPrevious = this->constructorFromPrevious;
      f->seriesRefs = frefs;
      f->segment = headers[i];
      f->reset();
      f->forked = true;
    }
    this->forked = true;
    return result;
  }
  bool isForked() const { return this->forked; }

  // add a handler for another constructor (matched in order with the write side)
  template <typename F>
  void series(const std::string& n, const F& f) {
    this->seriesRefs.push_back(makeSeriesReader(n, f));
  }

  // try to find the series with a given name
  // (raises an error if the name isn't given to any series)
  rseriesi* seriesWithName(const std::string& n) const {
    for (size_t i = 0; i < this->seriesRefs.size(); ++i) {
      if (this->seriesRefs[i]->name() == n) {
        return this->seriesRefs[i];
      }
    }
    std::ostringstream err;
    err << "No such series ";
    if (this->seriesRefs.size() < this->expectedConstructorCount()) {
      err << "yet ";
    }
    err << "defined: '" << n << "'";
    throw std::runtime_error(err.str());
  }

  // read and process a single constructor
  bool step() {
    try {
      ctorid c = 0;
      if (this->atEOF || !maybeReadCtor(&c)) { this->atEOF = true; return false; }
      if (this->seriesRefs.size() <= c) {
        throw std::runtime_error("Invalid constructor for trace file (corruption?): " + string::from(c));
      }
      this->seriesRefs[c]->read(this);
      return true;
    } catch (input_at_eof&) {
      this->atEOF = true;
      return false;
    }
  }

  // read and process the file data until EOF
  void run() {
    while (step());
  }

  // read a symbol (avoids repeating expensive variable-length strings)
  std::string readSymbol() {
    std::string s;
    this->symbols.read(&this->input, &s);
    return s;
  }

  // read a timestamp encoded as a relative diff (see writer::writeTimestamp)
  uint64_t readTimestamp() {
    return this->timestamps.read(&this->input);
  }

  // read an identifier from a given universe ID
  // (at file scope so that statistics can be shared across series)
  uint32_t readID(size_t u) {
    if (this->identifiers.size() <= u) {
      this->identifiers.resize(u + 1);
    }
    return this->identifiers[u].read(in());
  }

  // access the input stream (only meant for rseries<T> implementation, better not to treat as a user API)
  AEInput* in() { return &this->input; }

  // for trace analysis / debugging, provide the number of bits for the last ctor read
  size_t lastCtorBits() const { return this->lastcsz; }
protected:
  ReadTraceDataPtr freader;
  uint32_t         expectedCtors = 0;
  bool             forked = false;
  size_t           lastcsz = 0;
  AEInput          input;
  SegmentHeader    segment;
  uint64_t         nextSegmentPos = 0;
  bool             atEOF = false;

  FinSetModel<uint8_t, 0x2>           commandType;             // indicates how to interpret a stream command (new series, reset, hangup)
  IdentifierSet                       constructors;            // keep track of constructor IDs for log statements
  small_ring_buffer<uint32_t>         previousCtors;           // the previously recorded constructor ID
  std::vector<ScopedIdentifierSetRef> constructorFromPrevious; // given the previous constructor (by index), predict the next likely constructor
  TypeSet                             types;                   // keep track of type descriptions (per constructor)
  SymbolSet                           symbols;                 // keep track of internalized symbols
  TimestampSet                        timestamps;              // keep track of timestamps
  std::vector<UserIDModel>            identifiers;             // allow any number of (disjoint) user-defined identifier sets

  // keep track of all registered series readers (assumed closed)
  typedef std::vector<rseriesi*> SeriesRefs;
  SeriesRefs seriesRefs;

  // fork a reader at an internal segment, potentially for concurrent evaluation
  virtual std::shared_ptr<reader> fork(const ReadTraceDataPtr& fr, const SeriesRefs&) {
    return std::shared_ptr<reader>(new reader(fr));
  }

  // reset statistics everywhere to start from scratch, as if recording a fresh file
  // this will let parallel readers jump to different points and decompress immediate
  void reset() {
    this->commandType.reset();
    this->previousCtors.clear();
    this->types.reset();
    this->symbols.reset();
    this->timestamps.reset();
    for (auto& s : this->identifiers) {
      s.reset();
    }
    for (auto& s : this->constructorFromPrevious) {
      s.reset();
    }
    for (auto* s : this->seriesRefs) {
      s->resetState();
    }
  }

  // by default, we assume all readers are installed with correct types up front
  // so this step only performs validation (to ensure that installed readers have the right name and type)
  //
  // but a reasonable implementation could be given to _derive_ readers from stored name and type
  virtual rseriesi* makeReaderAt(uint32_t i, const std::string& name, const ty::desc& ty) {
    // we're at a new constructor definition
    // do we actually have one registered here?
    if (this->seriesRefs.size() <= i) {
      throw std::runtime_error("Trace file has registered series beyond the expected set");
    }
    rseriesi* s = this->seriesRefs[i];

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
    while (!this->input.eof()) {
      // try to read the next constructor ID, considering the previously-read constructor
      // (some events are more or less likely depending on previous events)
      auto s0 = in()->bitsRead();
      *c = this->constructorFromPrevious[this->previousCtors[0]].read(&this->input);
      this->lastcsz = in()->bitsRead() - s0;

      // data from previous events can also predict data in subsequent events
      if (*c) {
        for (size_t i = 0; i < this->previousCtors.size(); ++i) {
          if (auto pid = this->previousCtors[i]) {
            this->seriesRefs[*c - 1]->follows(this->seriesRefs[pid - 1]);
          }
        }
      }

      // remember this event to predict future events
      this->previousCtors.push(*c);

      // if this is an event rather than a control message, we're done
      if (*c != 0) {
        *c -= 1;
        return true;
      }

      // now we can get a command indicating one of three possibilities:
      //  * a new event registration
      //  * a stats-reset event (a point where parallel readers could safely jump to)
      //  * a soft hangup (which means we'll end the stream)
      uint8_t cmd = 0;
      this->commandType.read(&this->input, &cmd);
      if (cmd == 1) {
        // we're at the end of a segment, detach if we're forked (other reader instances will be expected to handle data for other segments)
        if (this->forked) {
          return false;
        }

        // otherwise switch to reading the next segment
        // reset statistics (and pick up the link to the next segment while doing it)
        in()->startNextSegment(&this->segment);
        this->nextSegmentPos += this->segment.toNextSegment;
        reset();
      } else if (cmd == 2) {
        // hangup
        return false;
      } else {
        // read the new constructor name and type
        std::string n;
        this->symbols.read(&this->input, &n);
        ty::desc tdef = this->types.read(&this->input);

        // install the reader for this constructor ID
        auto id = this->constructors.size();
        rseriesi* s = makeReaderAt(id, n, tdef);
        if (this->seriesRefs.size() <= id) {
          this->seriesRefs.resize(id+1);
        }
        this->seriesRefs[id] = s;

        // update the count and size of constructor codes (we've added a new constructor case)
        this->constructors.fresh();
        this->constructorFromPrevious.emplace_back();
      }
    }
    return false;
  }
};

// simple test to determine if a file is a compressed trace file (without having to actually decode it)
inline bool canReadFile(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
  if (fd < 0) {
    return false;
  }
  constexpr size_t prefix_len = sizeof(HCTRACE_MAGIC_PREFIX) - 1;
  char prefix_buf[prefix_len + 1];
  if (::read(fd, prefix_buf, prefix_len) != ssize_t(prefix_len)) {
    ::close(fd);
    return false;
  }
  if (strncmp(prefix_buf, HCTRACE_MAGIC_PREFIX, prefix_len) != 0) {
    ::close(fd);
    return false;
  }
  uint32_t expectedCtors=0;
  bool result = ::read(fd, reinterpret_cast<uint8_t*>(&expectedCtors), sizeof(expectedCtors)) == sizeof(expectedCtors);
  ::close(fd);
  return result;
}

/*******************************************************
 *
 * model<T> : decide what model to use to efficiently encode/decode values on a per-type basis
 *
 *******************************************************/

template <typename T, typename P = void>
struct model { };

// test whether one or a sequence of types have 'model<T>' specializations
// (a poor man's qualified types)
template <typename T, typename P = void>
struct has_model { static const bool value = false; };
template <typename T>
struct has_model<T, typename valid_type<decltype(model<T>::type())>::type> { static const bool value = true; };

template <typename ... Ts>
struct all_have_model { static const bool value = true; };
template <typename T, typename ... Ts>
struct all_have_model<T, Ts...> { static const bool value = has_model<T>::value && all_have_model<Ts...>::value; };

// map model type constructor over a type sequence
template <typename T>
struct ModelOf {
  typedef model<T> type;
};

// unit values already encode in 0 bits, so nothing special is required to observe bias
template <>
struct model<unit> {
  static ty::desc type() { return ty::prim("unit"); }
  void write(writer*, unit) { }
  void read(reader*, unit*) { }
  void reset() { }
  void connections(PrimModelsByTypes*) { }
  void bind(const PrimModelsByTypes&) { }
  tym::Ptr<size_t> memoryUsed() const { return tym::prim<size_t>(0); }
};

// bool has a special case already
template <>
struct model<bool> {
  static ty::desc type() { return ty::prim("bool"); }
  BoolModel m;
  void write(writer* w, bool s) { m.write(w->out(), s); }
  void read(reader* r, bool* s) { m.read(r->in(), s); }
  void reset() { m.reset(); }
  void connections(PrimModelsByTypes*) { }
  void bind(const PrimModelsByTypes&) { }
  tym::Ptr<size_t> memoryUsed() const { return this->m.memoryUsed(); }
};

// whole number types can encode with the whole number model
template <typename Name, typename T, bool inferName = true>
struct mednum_model {
  static ty::desc type() { return (!inferName || std::is_signed<T>::value) ? ty::prim(Name::str()) : ty::prim(std::string("u") + Name::str(), ty::prim(Name::str())); }
  typedef LinkedIntValueModel<T, 0, WholeNumberModel> M;
  M m;
  void write(writer* w, T v) { m.write(w->out(), v); }
  void read(reader* r, T* v) { m.read(r->in(), v); }
  void reset() { m.reset(); }
  void connections(PrimModelsByTypes* p) {
    addConnections(p, type(), {&this->m});
  }
  void bind(const PrimModelsByTypes& p) {
    for (auto* pred : connectionsByType(type(), p)) {
      this->m.link(reinterpret_cast<M*>(pred));
    }
  }
  tym::Ptr<size_t> memoryUsed() const { return this->m.memoryUsed(); }
};
template <> struct model<uint8_t>  : public mednum_model<HPPF_TSTR_8("byte"),  uint8_t,  false> { };
template <> struct model<char>     : public mednum_model<HPPF_TSTR_8("char"),  char,     false> { };
template <> struct model<int16_t>  : public mednum_model<HPPF_TSTR_8("short"), int16_t,  true>  { };
template <> struct model<uint16_t> : public mednum_model<HPPF_TSTR_8("short"), uint16_t, true>  { };
template <> struct model<int32_t>  : public mednum_model<HPPF_TSTR_8("int"),   int32_t,  true>  { };
template <> struct model<uint32_t> : public mednum_model<HPPF_TSTR_8("int"),   uint32_t, true>  { };
template <> struct model<int64_t>  : public mednum_model<HPPF_TSTR_8("long"),  int64_t,  true>  { };
template <> struct model<uint64_t> : public mednum_model<HPPF_TSTR_8("long"),  uint64_t, true>  { };

template <typename T, typename U>
inline const T* alias_cast(const U* u) {
  return reinterpret_cast<const T*>(reinterpret_cast<const char*>(u));
}
template <typename T, typename U>
inline T* alias_cast(U* u) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(u));
}

// 32-bit floats are like packed structures with three fields: bool neg, byte exp, and uint22 fract
template <>
struct model<float> {
  model<bool>                sign;   // bit 31
  model<uint8_t>             exp;    // bits 23-30
  model<uint16_t>            fract1; // bits 6-22
  FinSetModel<uint8_t, 0x3f> fract0; // bits 0-5

  void write(writer* w, float v) {
    uint32_t x = *alias_cast<uint32_t>(&v);
    this->fract0.write(w->out(), uint8_t(x & 0x3f));
    x >>= 7;
    this->fract1.write(w, uint16_t(x));
    x >>= 16;
    this->exp.write(w, uint8_t(x));
    x >>= 8;
    this->sign.write(w, x ? true : false);
  }
  void read(reader* r, float* v) {
    uint8_t  f0, e;
    uint16_t f1;
    bool     s;

    this->fract0.read(r->in(), &f0);
    this->fract1.read(r, &f1);
    this->exp.read(r, &e);
    this->sign.read(r, &s);

    uint32_t x = (uint32_t(s ? 1 : 0) << 31) |
                 (uint32_t(e) << 23) |
                 (uint32_t(f1) << 7) |
                 uint32_t(f0);

    *v = *alias_cast<float>(&x);
  }
  void reset() {
    this->sign.reset();
    this->exp.reset();
    this->fract1.reset();
    this->fract0.reset();
  }

  static ty::desc type() { return ty::prim("float"); }
  void connections(PrimModelsByTypes* p) {
    addConnections(p, type(), {this});
  }
  void bind(const PrimModelsByTypes& p) {
    for (auto* pred : connectionsByType(type(), p)) {
      auto* rhs = reinterpret_cast<model<float>*>(pred);
      this->exp.m.link(&rhs->exp.m);
      this->fract1.m.link(&rhs->fract1.m);
    }
  }
  tym::Ptr<size_t> memoryUsed() const {
    return
      tym::rec<size_t>({
        { "sign",   this->sign.memoryUsed() },
        { "exp",    this->exp.memoryUsed() },
        { "fract1", this->fract1.memoryUsed() },
        { "fract0", this->fract0.memoryUsed() }
      });
  }
};

// 64-bit floats are just like 32-bit floats, with fields widened a little
template <>
struct model<double> {
  model<bool>                sign;   // bit 63
  model<uint8_t>             exp1;   // bits 55-62
  FinSetModel<uint8_t, 0x07> exp0;   // bits 52-54
  model<uint32_t>            fract2; // bits 20-51
  model<uint16_t>            fract1; // bits 4-19
  FinSetModel<uint8_t, 0x0f> fract0; // bits 0-3

  void write(writer* w, double v) {
    uint64_t x = *alias_cast<uint64_t>(&v);
    this->fract0.write(w->out(), uint8_t(x & 0x0f));
    x >>= 4;
    this->fract1.write(w, uint16_t(x));
    x >>= 16;
    this->fract2.write(w, uint32_t(x));
    x >>= 32;
    this->exp0.write(w->out(), uint8_t(x & 0x07));
    x >>= 3;
    this->exp1.write(w, uint8_t(x));
    x >>= 8;
    this->sign.write(w, x ? true : false);
  }
  void read(reader* r, double* v) {
    uint8_t  f0, e0, e1;
    uint16_t f1;
    uint32_t f2;
    bool     s;

    this->fract0.read(r->in(), &f0);
    this->fract1.read(r, &f1);
    this->fract2.read(r, &f2);
    this->exp0.read(r->in(), &e0);
    this->exp1.read(r, &e1);
    this->sign.read(r, &s);

    uint64_t x = (uint64_t(s ? 1 : 0) << 63) |
                 (uint64_t(e1) << 55) |
                 (uint64_t(e0) << 52) |
                 (uint64_t(f2) << 20) |
                 (uint64_t(f1) << 4) |
                 uint64_t(f0);

    *v = *alias_cast<double>(&x);
  }
  void reset() {
    this->sign.reset();
    this->exp1.reset();
    this->exp0.reset();
    this->fract2.reset();
    this->fract1.reset();
    this->fract0.reset();
  }

  static ty::desc type() { return ty::prim("double"); }
  void connections(PrimModelsByTypes* p) {
    addConnections(p, type(), {this});
  }
  void bind(const PrimModelsByTypes& p) {
    for (auto* pred : connectionsByType(type(), p)) {
      auto* rhs = reinterpret_cast<model<double>*>(pred);
      this->exp1.m.link(&rhs->exp1.m);
      this->fract2.m.link(&rhs->fract2.m);
      this->fract1.m.link(&rhs->fract1.m);
    }
  }
  tym::Ptr<size_t> memoryUsed() const {
    return
      tym::rec<size_t>({
        { "sign",   this->sign.memoryUsed() },
        { "exp1",   this->exp1.memoryUsed() },
        { "exp0",   this->exp0.memoryUsed() },
        { "fract2", this->fract2.memoryUsed() },
        { "fract1", this->fract1.memoryUsed() },
        { "fract0", this->fract0.memoryUsed() }
      });
  }
};

// write fixed-length arrays
// to avoid excess memory use, one model is used for all array elements
template <typename T, size_t N>
struct array_model {
  static ty::desc type() { return ty::array(model<T>::type(), ty::nat(N)); }
  typedef model<T> M;
  M m;
  void array_write(writer* w, const T* v) {
    for (size_t i = 0; i < N; ++i) {
      m.write(w, v[i]);
    }
  }
  void array_read(reader* r, T* v) {
    for (size_t k = 0; k < N; ++k) {
      m.read(r, &v[k]);
    }
  }
  void reset() {
    m.reset();
  }
  void connections(PrimModelsByTypes* p) { this->m.connections(p); }
  void bind(const PrimModelsByTypes& p) { this->m.bind(p); }
  tym::Ptr<size_t> memoryUsed() const { return this->m.memoryUsed(); }
};
template <typename T, size_t N>
  struct model<T[N], typename tbool<has_model<T>::value>::type> : public array_model<T, N> {
    void write(writer* w, const T (&v)[N]) {
      this->array_write(w, v);
    }
    void read(reader* r, T (&v)[N]) {
      this->array_read(r, v);
    }
  };
template <typename T, size_t N>
  struct model<std::array<T, N>, typename tbool<has_model<T>::value>::type> : public array_model<T, N> {
    void write(writer* w, const std::array<T, N>& v) {
      this->array_write(w, v.data());
    }
    void read(reader* r, std::array<T, N>* v) {
      this->array_read(r, v->data());
    }
  };

// write strings as dependent pair types (n:nat, char[n])
// to avoid excess memory use, one model is used for all array elements
template <>
  struct model<char*> {
    static ty::desc type() { return ty::array(ty::prim("char")); }
    StringModel m;
    void write(writer* w, const char* v) {
      m.write(w->out(), v);
    }
    void reset() {
      m.reset();
    }
    // read intentionally left out (unsafe)
    void connections(PrimModelsByTypes*) { }
    void bind(const PrimModelsByTypes&) { }
    tym::Ptr<size_t> memoryUsed() const { return this->m.memoryUsed(); }
  };
template <> struct model<const char*> : public model<char*> { };
template <>
  struct model<std::string> {
    static ty::desc type() { return ty::array(ty::prim("char")); }
    StringModel m;
    void write(writer* w, const std::string& v) {
      m.write(w->out(), v);
    }
    void read(reader* r, std::string* v) {
      m.read(r->in(), v);
    }
    void reset() {
      m.reset();
    }
    void connections(PrimModelsByTypes*) { }
    void bind(const PrimModelsByTypes&) { }
    tym::Ptr<size_t> memoryUsed() const { return this->m.memoryUsed(); }
  };

// write vectors as dependent pair types (n:nat, T[n]) similarly to strings
template <typename T>
  struct model<std::vector<T>, typename tbool<has_model<T>::value>::type> {
    static ty::desc type() { return ty::array(model<T>::type()); }
    typedef std::pair<model<size_t>, model<T>> M;
    M m;
    void write(writer* w, const std::vector<T>& vs) {
      m.first.write(w, vs.size());
      for (const auto& v : vs) {
        m.second.write(w, v);
      }
    }
    void read(reader* r, std::vector<T>* vs) {
      size_t n=0;
      m.first.read(r, &n);
      vs->resize(checkArrLen(n));
      for (size_t k = 0; k < n; ++k) {
        m.second.read(r, &(*vs)[k]);
      }
    }
    void reset() {
      this->m.first.reset();
      this->m.second.reset();
    }
    void connections(PrimModelsByTypes* p) {
      this->m.first.connections(p);
      this->m.second.connections(p);
    }
    void bind(const PrimModelsByTypes& p) {
      this->m.first.bind(p);
      this->m.second.bind(p);
    }
    tym::Ptr<size_t> memoryUsed() const {
      return
        tym::rec<size_t>({
          {"len",  this->m.first.memoryUsed() },
          {"elem", this->m.second.memoryUsed() }
        });
    }
  };

// write sets (as variable-length arrays)
template <typename CT, typename T>
  struct SetModel {
    static ty::desc type() { return ty::array(model<T>::type()); }
    typedef std::pair<model<size_t>, model<T>> M;
    M m;
    void write(writer* w, const CT& vs) {
      m.first.write(w, vs.size());
      for (const auto& v : vs) {
        m.second.write(w, v);
      }
    }
    void read(reader* r, CT* vs) {
      size_t n=0;
      m.first.read(r, &n);
      checkArrLen(n);
      for (size_t k = 0; k < n; ++k) {
        T v;
        m.second.read(r, &v);
        vs->insert(v);
      }
    }
    void reset() {
      this->m.first.reset();
      this->m.second.reset();
    }
    void connections(PrimModelsByTypes* p) {
      this->m.first.connections(p);
      this->m.second.connections(p);
    }
    void bind(const PrimModelsByTypes& p) {
      this->m.first.bind(p);
      this->m.second.bind(p);
    }
    tym::Ptr<size_t> memoryUsed() const {
      return
        tym::rec<size_t>({
          {"len",  this->m.first.memoryUsed() },
          {"elem", this->m.second.memoryUsed() }
        });
    }
  };
template <typename T> struct model<std::set<T>, typename tbool<has_model<T>::value>::type> : public SetModel<std::set<T>, T> { };
template <typename T> struct model<std::unordered_set<T>, typename tbool<has_model<T>::value>::type> : public SetModel<std::unordered_set<T>, T> { };

// write pairs
template <typename U, typename V>
  struct model<std::pair<U,V>, typename tbool<has_model<U>::value && has_model<V>::value>::type> {
    static ty::desc type() { return ty::tuple({ model<U>::type(), model<V>::type() }); }
    typedef std::pair<model<U>, model<V>> M;
    M m;
    void write(writer* w, const std::pair<U,V>& v) {
      this->m.first.write(w, v.first);
      this->m.second.write(w, v.second);
    }
    void read(reader* r, std::pair<U,V>* v) {
      this->m.first.read(r, &v->first);
      this->m.second.read(r, &v->second);
    }
    void reset() {
      this->m.first.reset();
      this->m.second.reset();
    }
    void connections(PrimModelsByTypes* p) {
      this->m.first.connections(p);
      this->m.second.connections(p);
    }
    void bind(const PrimModelsByTypes& p) {
      this->m.first.bind(p);
      this->m.second.bind(p);
    }
    tym::Ptr<size_t> memoryUsed() const { return tym::tup<size_t>({ this->m.first.memoryUsed(), this->m.second.memoryUsed() }); }
  };

// write tuples
template <size_t k, size_t n, typename ... Ts>
  struct TupleModel {
    typedef typename fmap<ModelOf, tuple<Ts...>>::type M;

    typedef typename nth<k, Ts...>::type H;
    typedef TupleModel<k+1, n, Ts...>    Recurse;

    static void type(ty::StructFields* fs) {
      fs->push_back(ty::StructField(".f" + string::from(k), -1, model<H>::type()));
      Recurse::type(fs);
    }
    static void selfBind(M& m) {
      PrimModelsByTypes p;
      m.template at<k>().connections(&p);
      Recurse::bind(p, m);
      Recurse::selfBind(m);
    }
    static void write(writer* w, M& m, const tuple<Ts...>& v) {
      m.template at<k>().write(w, v.template at<k>());
      Recurse::write(w, m, v);
    }
    static void read(reader* r, M& m, tuple<Ts...>* v) {
      m.template at<k>().read(r, &v->template at<k>());
      Recurse::read(r, m, v);
    }
    static void reset(M& m) {
      m.template at<k>().reset();
      Recurse::reset(m);
    }
    static void connections(PrimModelsByTypes* p, M& m) {
      m.template at<k>().connections(p);
      Recurse::connections(p, m);
    }
    static void bind(const PrimModelsByTypes& p, M& m) {
      m.template at<k>().bind(p);
      Recurse::bind(p, m);
    }
    static void memoryUsed(std::vector<tym::Ptr<size_t>>* szs, const M& m) {
      szs->push_back(m.template at<k>().memoryUsed());
      Recurse::memoryUsed(szs, m);
    }
  };
template <size_t n, typename ... Ts>
  struct TupleModel<n, n, Ts...> {
    typedef typename fmap<ModelOf, tuple<Ts...>>::type M;

    static void type(ty::StructFields*) { }
    static void selfBind(M&) { }
    static void write(writer*, M&, const tuple<Ts...>&) { }
    static void read(reader*, M&, tuple<Ts...>*) { }
    static void reset(M&) { }
    static void connections(PrimModelsByTypes*, M&) { }
    static void bind(const PrimModelsByTypes&, M&) { }
    static void memoryUsed(std::vector<tym::Ptr<size_t>>*, const M&) { }
  };
template <typename ... Ts>
  struct model<tuple<Ts...>, typename tbool<all_have_model<Ts...>::value>::type> {
    typedef TupleModel<0, sizeof...(Ts), Ts...> TM;
    typedef typename TM::M M;
    M m;
    model() {
      // predict successor fields from predecessor fields
      TM::selfBind(this->m);
    }

    static ty::desc type() {
      ty::StructFields fs;
      TM::type(&fs);
      return ty::record(fs);
    }
    void write(writer* w, const tuple<Ts...>& v) {
      TM::write(w, this->m, v);
    }
    void read(reader* r, tuple<Ts...>* v) {
      TM::read(r, this->m, v);
    }
    void reset() {
      TM::reset(this->m);
    }
    void connections(PrimModelsByTypes* p) {
      TM::connections(p, this->m);
    }
    void bind(const PrimModelsByTypes& p) {
      TM::bind(p, this->m);
    }
    tym::Ptr<size_t> memoryUsed() const {
      std::vector<tym::Ptr<size_t>> fs;
      TM::memoryUsed(&fs, this->m);
      return tym::tup(fs);
    }
  };

// write (user) reflective structs
struct DescStructF {
  ty::StructFields* fs;
  DescStructF(ty::StructFields* fs) : fs(fs) { }
  template <typename T>
    void visit(const char* fname) {
      this->fs->push_back(ty::StructField(fname, -1, model<T>::type()));
    }
};
template <typename T>
  struct model<T, typename tbool<T::is_hmeta_struct && has_model<typename T::as_tuple_type>::value>::type> {
    static ty::desc type() {
      ty::StructFields fs;
      DescStructF df(&fs);
      T::meta(df);
      return ty::record(fs);
    }
    typedef typename T::as_tuple_type TT;
    typedef model<TT> M;
    M m;

    void write(writer* w, const T& v) {
      this->m.write(w, *reinterpret_cast<const TT*>(&v));
    }
    void read(reader* r, T* v) {
      this->m.read(r, reinterpret_cast<TT*>(v));
    }
    void reset() {
      this->m.reset();
    }
    void connections(PrimModelsByTypes* p) {
      this->m.connections(p);
    }
    void bind(const PrimModelsByTypes& p) {
      this->m.bind(p);
    }
    tym::Ptr<size_t> memoryUsed() const {
      std::vector<tym::Ptr<size_t>> fs;
      M::TM::memoryUsed(&fs, this->m.m);

      ty::StructFields tfs;
      DescStructF df(&tfs);
      T::meta(df);

      tym::RecordTM<size_t>::Fields sfs;
      for (size_t i = 0; i < std::min<size_t>(fs.size(), tfs.size()); ++i) {
        sfs.push_back({ tfs[i].at<0>(), fs[i] });
      }
      return tym::rec<size_t>(sfs);
    }
  };

// write reflective enums
template <typename T>
  struct model<T, typename tbool<T::is_hmeta_enum>::type> {
    static ty::desc type() {
      return ty::enumdef(model<uint32_t>::type(), T::meta());
    }

    typedef FinSetModel<uint16_t, T::ctorCount - 1> M;
    M m;

    void write(writer* w, T v) {
      assert(T::toOrd(v) == static_cast<uint16_t>(T::toOrd(v)));
      m.write(w->out(), static_cast<uint16_t>(T::toOrd(v)));
    }
    void read(reader* r, T* v) {
      uint16_t c=0;
      m.read(r->in(), &c);
      *v = T::fromOrd(static_cast<typename T::rep_t>(c));
    }
    void reset() {
      this->m.reset();
    }
    void connections(PrimModelsByTypes*) { }
    void bind(const PrimModelsByTypes&) { }
    tym::Ptr<size_t> memoryUsed() const { return this->m.memoryUsed(); }
  };

// write variants
template <size_t i, size_t n, typename ... Ts>
  struct VariantModelDesc {
    typedef typename nth<i, Ts...>::type    H;
    typedef VariantModelDesc<i+1, n, Ts...> Recurse;

    static void type(ty::VariantCtors* cs) {
      cs->push_back(ty::VariantCtor(".f" + string::from(i), static_cast<int>(i), model<H>::type()));
      Recurse::type(cs);
    }
  };
template <size_t n, typename ... Ts>
  struct VariantModelDesc<n, n, Ts...> {
    static void type(ty::VariantCtors*) { }
  };

template <typename Model>
struct VariantWrite {
  template <size_t tag, typename T, typename Z>
  struct App {
    static void fn(T* vd, writer* w, Model* m) {
      m->template at<tag>().write(w, *vd);
    }
  };
  template <typename ... Ts>
  static void write(writer* w, const variant<Ts...>& v, Model* m) {
    v.template apply<void, App, void, writer*>(w, m);
  }
};
template <typename Model>
struct VariantRead {
  template <size_t tag, typename T, typename Z>
  struct App {
    static void fn(T* vd, reader* r, Model* m) {
      new (vd) T();
      m->template at<tag>().read(r, vd);
    }
  };
  template <typename ... Ts>
  static void read(reader* r, uint16_t tag, variant<Ts...>* v, Model* m) {
    variantApp<void, App, void, tuple<Ts...>, reader*, Model*>::apply(tag, v->unsafePayload(), r, m);
  }
};
template <typename ... Ts>
  struct model<variant<Ts...>, typename tbool<all_have_model<Ts...>::value>::type> {
    typedef VariantModelDesc<0, sizeof...(Ts), Ts...> Reflect;
    static ty::desc type() {
      ty::VariantCtors cs;
      Reflect::type(&cs);
      return ty::variant(cs);
    }
    // variant of N constructors, model of tag N, tuple of models for each constructor
    static_assert(sizeof...(Ts) > 0 && sizeof...(Ts) < (1UL << 16UL), "Variant model requires at least one constructor, and fewer than 2^16 constructors.");
    typedef FinSetModel<uint16_t, uint16_t(sizeof...(Ts) - 1)> TagModel;
    typedef TupleModel<0, sizeof...(Ts), Ts...> TM;
    typedef typename TM::M ValueModels;
    typedef std::pair<TagModel, ValueModels> M;
    M m;

    void write(writer* w, const variant<Ts...>& v) {
      this->m.first.write(w->out(), static_cast<uint16_t>(v.unsafeTag()));
      VariantWrite<ValueModels>::write(w, v, &m.second);
    }
    void read(reader* r, variant<Ts...>* v) {
      uint16_t t=0;
      this->m.first.read(r->in(), &t);
      v->unsafeTag() = t;
      VariantRead<ValueModels>::read(r, t, v, &m.second);
    }
    void reset() {
      this->m.first.reset();
      TM::reset(this->m.second);
    }
    void connections(PrimModelsByTypes* p) {
      TM::connections(p, this->m.second);
    }
    void bind(const PrimModelsByTypes& p) {
      TM::bind(p, this->m.second);
    }
    tym::Ptr<size_t> memoryUsed() const {
      std::vector<tym::Ptr<size_t>> fs;
      TM::memoryUsed(&fs, this->m.second);
      return
        tym::rec<size_t>({
          { "tag",     this->m.first.memoryUsed() },
          { "payload", tym::tup(fs) }
        });
    }
  };

// write (user) reflective variants
struct DescVariantF {
  ty::VariantCtors* ctors;
  DescVariantF(ty::VariantCtors* ctors) : ctors(ctors) { }
  template <typename T>
    void ctor(const char* n, int id) {
      this->ctors->push_back(ty::VariantCtor(n, id, model<T>::type()));
    }
};
template <typename T>
  struct model<T, typename tbool<T::is_hmeta_variant && has_model<typename T::as_variant_type>::value>::type> {
    static ty::desc type() {
      ty::VariantCtors cs;
      DescVariantF f(&cs);
      T::meta(f);
      return ty::variant(cs);
    }
    typedef typename T::as_variant_type VT;
    typedef model<VT> M;
    M m;
    void write(writer* w, const T& v) { this->m.write(w, *reinterpret_cast<const VT*>(&v)); }
    void read (reader* r,       T* v) { this->m.read (r,  reinterpret_cast<VT*>(v)); }
    void reset() { this->m.reset(); }
    void connections(PrimModelsByTypes* p) { this->m.connections(p); }
    void bind(const PrimModelsByTypes& p) { this->m.bind(p); }
    tym::Ptr<size_t> memoryUsed() const { return this->m.memoryUsed(); }
  };

// write generic structs/variants carrying type-level field names
// (if the underlying type isn't tuple or variant, the names are ignored)
template <typename NameList, typename T>
struct model<withNames<NameList, T>, typename tbool<has_model<T>::value>::type> {
  static ty::desc type() {
    auto ns = lowerStringList<NameList>();

    // try to add names into this type
    ty::desc u = model<T>::type();
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
  typedef model<T> M;
  M m;
  void write(writer* w, const withNames<NameList, T>& v) { this->m.write(w, v.value); }
  void read (reader* r,       withNames<NameList, T>* v) { this->m.read (r, &v->value); }
  void reset() { this->m.reset(); }

  void connections(PrimModelsByTypes* p) {
    this->m.connections(p);
  }
  void bind(const PrimModelsByTypes& p) {
    this->m.bind(p);
  }
  tym::Ptr<size_t> memoryUsed() const {
    return this->m.memoryUsed();
  }
};

// write opaque type aliases
template <typename T>
  struct model<T, typename tbool<T::is_hmeta_alias && has_model<typename T::type>::value>::type> {
    typedef typename T::type RT;
    static ty::desc type() { return ty::prim(T::name(), model<RT>::type()); }

    typedef model<RT> M;
    M m;

    void write(writer* w, const T& v) {
      this->m.write(w, v.value);
    }
    void read(reader* r, T* v) {
      this->m.read(r, &v->value);
    }
    void reset() {
      this->m.reset();
    }
    tym::Ptr<size_t> memoryUsed() const {
      return this->m.memoryUsed();
    }

    // connections between models for aliased types work just like for underlying types
    // except that the alias types are treated as distinct types, so should only be connected
    // to their identical typed connections (e.g. a "datetime" shouldn't be connected to
    // a previous "long" value, even though the representation for a "datetime" is "long",
    // it should only be connected to other "datetime" values)
    void connections(PrimModelsByTypes* p) {
      PrimModelsByTypes local;
      this->m.connections(&local);
      addConnections(p, type(), connectionsByType(M::type(), local));
    }
    void bind(const PrimModelsByTypes& p) {
      PrimModelsByTypes local;
      addConnections(&local, M::type(), connectionsByType(type(), p));
      this->m.bind(local);
    }
  };

// write chrono times
template <>
  struct model<std::chrono::system_clock::time_point> {
    typedef model<datetime_t> M;
    M m;
    static ty::desc type() { return M::type(); }
    void write(writer* w, const std::chrono::system_clock::time_point& v) {
      this->m.write(w, datetime_t(std::chrono::duration_cast<std::chrono::microseconds>(v.time_since_epoch()).count()));
    }
    void read(reader* r, std::chrono::system_clock::time_point* v) {
      datetime_t ut;
      this->m.read(r, &ut);
      *v = std::chrono::system_clock::time_point(std::chrono::microseconds(ut.value));
    }
    void reset() {
      this->m.reset();
    }
    void connections(PrimModelsByTypes* p) { this->m.connections(p); }
    void bind(const PrimModelsByTypes& p) { this->m.bind(p); }
    tym::Ptr<size_t> memoryUsed() const { return this->m.memoryUsed(); }
  };

// write chrono durations
template <typename Rep, std::intmax_t Numerator, std::intmax_t Denominator>
  struct model<std::chrono::duration<Rep, std::ratio<Numerator, Denominator>>> {
    static ty::desc type() {
      // cduration Rep Numerator Denominator
      return ty::appc(ty::prim("cduration", ty::fnc({"rep", "n", "d"}, ty::var("rep"))), {model<Rep>::type(), ty::nat(Numerator), ty::nat(Denominator)});
    }

    typedef std::chrono::duration<Rep, std::ratio<Numerator, Denominator>> CT;
    typedef model<Rep> M;
    M m;
    void write(writer* w, const CT& v) {
      this->m.write(w, v.count());
    }
    void read(reader* r, CT* v) {
      Rep uv;
      this->m.read(r, &uv);
      *v = CT(uv);
    }
    void reset() {
      this->m.reset();
    }
    void connections(PrimModelsByTypes* p) {
      PrimModelsByTypes local;
      this->m.connections(&local);
      addConnections(p, type(), connectionsByType(M::type(), local));
    }
    void bind(const PrimModelsByTypes& p) {
      PrimModelsByTypes local;
      addConnections(&local, M::type(), connectionsByType(type(), p));
      this->m.bind(local);
    }
    tym::Ptr<size_t> memoryUsed() const { return this->m.memoryUsed(); }
  };

// write recursive types
// this definition gets a little circuitous due to the nature of data recursion
//
// basically, at any point we have to deal with an indefinite unrolling, we hold on to
// a little bit of state during construction to distinguish the top level from the
// nested levels (and make sure that nested levels just refer back to the top level)
template <typename T>
  struct model<RolledRec<T>> {
    static ty::desc type() {
      return withRecursive(ty::var("x"), []() { return ty::recursive("x", model<typename T::Unrolled>::type()); });
    }
    typedef std::shared_ptr<model<typename T::Unrolled>> M;
    M m = M();
    model<RolledRec<T>>* parent = nullptr;
    model() {
      this->parent =
        withRecursive(this,
          [this]() {
            this->m = M(new model<typename T::Unrolled>());
            return this;
          }
        );
    }
    void write(writer* w, const RolledRec<T>& x) {
      this->parent->m->write(w, x.rolled->unrolled);
    }
    void read(reader* r, RolledRec<T>* x) {
      typedef typename T::Unrolled DT;
      DT ux;
      this->parent->m->read(r, &ux);
      *x = RolledRec<T>::roll(ux);
    }
    void reset() {
      if (this->parent == this) {
        this->m->reset();
      }
    }
    void connections(PrimModelsByTypes* p) {
      if (this->parent == this) {
        this->m->connections(p);
      }
    }
    void bind(const PrimModelsByTypes& p) {
      if (this->parent == this) {
        this->m->bind(p);
      }
    }
    tym::Ptr<size_t> memoryUsed() const {
      return (this->parent == this) ? this->m->memoryUsed() : tym::unit<size_t>();
    }
  };

// write 'model-dependent pairs', similar to a sigma type except the _model_ of the second value is dependent on the first value
// this is useful if a user knows that the statistical properties of some data are highly dependent on some other data
// (as e.g. market data entropy greatly reduces when dependent on symbol)
template <typename U, typename V>
struct model_dependent_pair {
  U first;
  V second;

  model_dependent_pair() : first(), second() { }
  model_dependent_pair(const model_dependent_pair<U, V>& p) : first(p.first), second(p.second) { }
  model_dependent_pair(const U& first, const V& second) : first(first), second(second) { }
  model_dependent_pair<U, V>& operator=(const model_dependent_pair<U, V>& rhs) { this->first = rhs.first; this->second = rhs.second; return *this; }

  bool operator==(const model_dependent_pair<U, V>& rhs) const {
    return this->first == rhs.first && this->second == rhs.second;
  }
};
template <typename U, typename V>
  struct model<model_dependent_pair<U, V>> {
    static ty::desc type() {
      // model_dependent_pair U V
      // (the reader has the U and V types, which it can use to correctly decode this data)
      return ty::appc(ty::prim("model_dependent_pair", ty::fnc({"u", "v"}, ty::prim("undefined-representation"))), {model<U>::type(), model<V>::type()});
    }

    model<U> first_model;
    std::unordered_map<U, model<V>> second_model;

    void write(writer* w, const model_dependent_pair<U, V>& p) {
      this->first_model.write(w, p.first);
      this->second_model[p.first].write(w, p.second);
    }
    void read(reader* r, model_dependent_pair<U, V>* p) {
      this->first_model.read(r, &p->first);
      this->second_model[p->first].read(r, &p->second);
    }
    void reset() {
      this->first_model.reset();
      this->second_model.clear();
    }

    void connections(PrimModelsByTypes* p) { this->first_model.connections(p); }
    void bind(const PrimModelsByTypes& p) { this->first_model.bind(p); }

    tym::Ptr<size_t> memoryUsed() const {
      auto v = tym::unit<size_t>();
      if (!this->second_model.empty()) {
        auto i = this->second_model.begin();
        v = i->second.memoryUsed();
        ++i;
        for (; i != this->second_model.end(); ++i) {
          v = tym::add<size_t>(v, i->second.memoryUsed());
        }
      }
      return
        tym::rec<size_t>({
          { "depkey", this->first_model.memoryUsed() },
          { "value",  v }
        });
    }
  };

// write 'identifiers', which may be used often between many different sequences (and should share statistics between them)
// identifiers are indexed by a "universe ID" so that many disjoint identifiers can be used
template <size_t U>
  struct identifier {
    uint32_t value;
    identifier() : value(0) { }
    identifier(uint32_t value) : value(value) { }
    bool operator==(const identifier<U>& rhs) const { return this->value == rhs.value; }
    bool operator <(const identifier<U>& rhs) const { return this->value < rhs.value; }
  };
template <size_t U>
  struct model<identifier<U>> {
    static ty::desc type() {
      // identifier U
      // readers can decode this identifier knowing its universe ID
      return ty::appc(ty::prim("identifier", ty::fnc({"u"}, ty::prim("undefined-representation"))), {ty::nat(U)});
    }

    void write(writer* w, const identifier<U>& p) {
      w->writeID(U, p.value);
    }
    void read(reader* r, identifier<U>* p) {
      p->value = r->readID(U);
    }
    void reset() {
    }

    void connections(PrimModelsByTypes* p) { }
    void bind(const PrimModelsByTypes& p) { }

    tym::Ptr<size_t> memoryUsed() const {
      return tym::unit<size_t>();
    }
  };

// supported types can be mapped to type descriptions through encoding models
template <typename T>
  ty::desc descSeriesType() {
    return model<T>::type();
  }

// allow serialization of any closed type freely generated by supported types
template <typename T>
class wseries : public wseriesi {
public:
  wseries(writer* w, ctorid id) : wseriesi(id), w(w) { }
  void operator()(datetime_t ts, const T& v) {
    this->w->writeCtor(id());
    this->w->writeTimestamp(ts.value);
    this->m.write(this->w, v);
    this->w->flushAtLimit();
  }
  tym::Ptr<size_t> memoryUsed() const override {
    return this->m.memoryUsed();
  }
protected:
  void extractModels(PrimModelsByTypes* pmodels) override {
    this->m.connections(pmodels);
  }
  void linkPredModels(const PrimModelsByTypes& pmodels) override {
    this->m.bind(pmodels);
  }
  void reset() override {
    this->m.reset();
  }
private:
  writer* w;
  model<T> m;
};
template <typename T>
  wseries<T>* makeSeriesWriter(writer* w, ctorid id) {
    return new wseries<T>(w, id);
  }

template <typename U>
class dwseries : public wseriesi {
public:
  dwseries(writer* w, ctorid id, const DynamicWriterImplPtr<U>& impl) : wseriesi(id), w(w), impl(impl) { }
  void operator()(datetime_t ts, const U& v) {
    this->w->writeCtor(id());
    this->w->writeTimestamp(ts.value);
    this->impl->write(this->w, v);
    this->w->flushAtLimit();
  }
  tym::Ptr<size_t> memoryUsed() const override {
    // TODO: maybe incorporate this in the impl interface
    //       for now, just lie and say we don't use any memory
    return tym::unit<size_t>();
  }
protected:
  void extractModels(PrimModelsByTypes* pmodels) override {
    this->impl->extractModels(pmodels);
  }
  void linkPredModels(const PrimModelsByTypes& pmodels) override {
    this->impl->linkPredModels(pmodels);
  }
  void reset() override {
    this->impl->reset();
  }
private:
  writer* w;
  DynamicWriterImplPtr<U> impl;
};
template <typename U>
dwseries<U>* makeDynamicSeriesWriter(writer* w, ctorid id, const DynamicWriterImplPtr<U>& impl) {
  return new dwseries<U>(w, id, impl);
}

// allow deserialization of any closed type freely generated by supported types
template <typename T>
class rseries : public rseriesi {
public:
  rseries(const std::string& n, const std::function<void(datetime_t, const T&)>& handler) : rseriesi(n), handler(handler) {
  }
  ty::desc type() const override { return descSeriesType<T>(); }
  void read(reader* r) override {
    datetime_t t = r->readTimestamp();
    T v;
    m.read(r, &v);
    this->handler(t, v);
  }
protected:
  void extractModels(PrimModelsByTypes* pmodels) override {
    this->m.connections(pmodels);
  }
  void linkPredModels(const PrimModelsByTypes& pmodels) override {
    this->m.bind(pmodels);
  }
  void reset() override {
    this->m.reset();
  }
  rseriesi* fork() override {
    return new rseries<T>(name(), this->handler);
  }
private:
  std::function<void(datetime_t, const T&)> handler;
  model<T> m;
};
template <typename F>
rseriesi* makeSeriesReader(const std::string& n, const F& f) {
  typedef typename function_traits<F>::argl Argl;
  typedef typename tupType<1, Argl>::type CArg0R;
  typedef typename std::remove_reference<CArg0R>::type CArg0;
  typedef typename std::remove_const<CArg0>::type T;
  return new rseries<T>(n, typename function_traits<F>::fn(f));
}

}
END_HLOG_NAMESPACE

// define simple hash functions for ctrace lib types
namespace std {
  template <size_t U>
  struct hash<HLOG_NS::ctrace::identifier<U>> {
    hash<uint32_t> u32h;
    size_t operator()(const HLOG_NS::ctrace::identifier<U>& id) const noexcept {
      return this->u32h(id.value);
    }
  };
}

#endif

