/*
 * reflect : common definitions to help introduce zero-overhead reflective types (consistent with algebraic data types in hobbes) and write their meta-data descriptions
 *
 * A macro 'HPPF_DEFINE_STRUCT(T, (T0, fn0), ... (TN, fnN))' is included to be equivalent to '{fn0:T0, ..., fnN:TN}'
 * A macro 'HPPF_DEFINE_VARIANT(T, (cn0, T0), ... (cnN, TN))' is included to be equivalent to '|cn0:T0, ..., cnN:TN|'
 * A macro 'HPPF_DEFINE_ENUM(T, (cn0), ... (cnN))' is included to be equivalent to '|cn0:(), ..., cnN:()|'
 * A type 'tuple<T0, ..., TN>' is included to be equivalent to '(T0 * ... * TN)' and have standard memory layout
 * A type 'variant<T0, ..., TN>' is included to be equivalent to '(T0 + ... + TN)' and have standard memory layout
 *
 * functions are also included to read and write serialized type descriptions
 *
 * For the convenience of users who are only interested in logging, the HPPF_DEFINE_* macros also have HLOG_DEFINE_* aliases
 */

#ifndef HOBBES_REFLECT_H_INCLUDED
#define HOBBES_REFLECT_H_INCLUDED

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <stack>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <cxxabi.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <fcntl.h>
#include <unistd.h>

// macOS doesn't support fallocate
inline int posix_fallocate(int fd, off_t o, off_t dsz) {
  return ftruncate(fd, o+dsz);
}
#endif

#define HLOG_LIB_VERSION_NAME version_1_0_5
#define HLOG_NS ::hlog::HLOG_LIB_VERSION_NAME
#define BEGIN_HLOG_NAMESPACE namespace hlog { namespace HLOG_LIB_VERSION_NAME {
#define END_HLOG_NAMESPACE }}

BEGIN_HLOG_NAMESPACE

// basic string utilities used here
namespace string {
  template <typename T>
    inline std::string from(const T& x) {
      std::ostringstream ss;
      ss << x;
      return ss.str();
    }
  template <typename T>
    inline T to(const std::string& x) {
      std::stringstream ss(x);
      T r = T();
      ss >> r;
      return r;
    }
  inline std::string lcase(const std::string& x) {
    std::ostringstream ss;
    for (char c : x) {
      if (c >= 'A' && c <= 'Z') {
        ss << char((c - 'A') + 'a');
      } else {
        ss << c;
      }
    }
    return ss.str();
  }

  // pretty-print C++ type names
  template <typename T>
    inline std::string demangle() {
      if (const char* tn = typeid(T).name()) {
        int s = 0;
        if (char* dmn = abi::__cxa_demangle(tn, 0, 0, &s)) {
          std::string r(dmn);
          free(dmn);
          return r;
        } else {
          return std::string(tn);
        }
      } else {
        return "";
      }
    }

  // tokenize strings
  typedef std::vector<std::string> seq;
  inline seq csplit(const std::string& str, const std::string& pivot) {
    typedef std::string::size_type size_type;

    seq       ret;
    size_type mp = 0;

    while (mp != std::string::npos) {
      size_type nmp = str.find(pivot, mp);

      if (nmp == std::string::npos) {
        ret.push_back(str.substr(mp, str.size() - mp));
        mp = nmp;
      } else {
        ret.push_back(str.substr(mp, nmp - mp));
        mp = nmp + pivot.size();
      }
    }

    return ret;
  }

  typedef std::pair<std::string, std::string> pair;
  inline pair lsplit(const std::string& s, const std::string& ss) {
    size_t p = s.find(ss);
    if (p == std::string::npos) {
      return pair(s, "");
    } else {
      return pair(s.substr(0, p), s.substr(p + ss.size(), s.size()));
    }
  }
  inline pair rsplit(const std::string& s, const std::string& ss) {
    size_t p = s.rfind(ss);
    if (p == std::string::npos) {
      return pair("", s);
    } else {
      return pair(s.substr(0, p), s.substr(p + ss.size(), s.size()));
    }
  }

  inline std::string cdelim(const seq& ss, const std::string& d) {
    if (ss.size() > 0) {
      std::ostringstream x;
      x << ss[0];
      for (size_t i = 1; i < ss.size(); ++i) {
        x << d << ss[i];
      }
      return x.str();
    } else {
      return "";
    }
  }

  inline void printTable(std::ostream& out, const std::vector<std::vector<std::string>>& cs, bool printHeader = true) {
    if (cs.empty()) return;
    for (const auto& c : cs) { if (c.empty()) return; }

    // determine column padding
    std::vector<size_t> widths;
    for (const auto& c : cs) {
      size_t w = 0;
      for (const auto& s : c) {
        w = std::max<size_t>(w, s.size());
      }
      widths.push_back(w);
    }

    if (printHeader) {
      // print headers
      for (size_t c = 0; c < cs.size(); ++c) {
        const auto& s = cs[c][0];
        out << (c?" ":"") << (c+1<cs.size() ? std::string(widths[c] - s.size(), ' ') : "") << s;
      }
      out << "\n";
      for (size_t c = 0; c < cs.size(); ++c) {
        out << (c?" ":"") << std::string(widths[c], '-');
      }
      out << "\n";
    }

    // determine row count
    size_t rc = cs[0].size();
    for (const auto& c : cs) { rc = std::min<size_t>(rc, c.size()); }

    // print rows
    for (size_t r = printHeader ? 1 : 0; r < rc; ++r) {
      for (size_t c = 0; c < cs.size(); ++c) {
        const auto& s = cs[c][r];
        out << (c?" ":"") << (c+1<cs.size() ? std::string(widths[c] - s.size(), ' ') : "") << s;
      }
      out << "\n";
    }
  }
  inline std::string showTable(const std::vector<std::vector<std::string>>& cs, bool showHeader = true) {
    std::ostringstream ss;
    printTable(ss, cs, showHeader);
    return ss.str();
  }
}

// align a value to a boundary
template <typename T>
inline T align(T x, T m) {
  if (m == 0 || (x % m) == 0) {
    return x;
  } else {
    return (1 + (x / m)) * m;
  }
}
template <typename T>
inline T* palign(T* x, size_t m) {
  return reinterpret_cast<T*>(align<uint64_t>(reinterpret_cast<uint64_t>(x), m));
}

typedef __int128 int128_t;

// generic structural hashing without clashing with std::hash
template <typename T, typename P = void>
struct GenericHash {
  std::hash<T> h;
  size_t operator()(T x) const noexcept {
    return this->h(x);
  }
};
template <typename ... Ts>
inline size_t GHash(size_t h, Ts...) {
  return h;
}
template <typename T, typename ... Ts>
inline size_t GHash(size_t h, const T& t, const Ts&... ts) {
  GenericHash<T> th;
  h ^= th(t) + 0x9e3779b9 + (h<<6) + (h>>2);
  return GHash(h, ts...);
}
template <typename U, typename V>
struct GenericHash<std::pair<U, V>> {
  size_t operator()(const std::pair<U, V>& p) const noexcept {
    return GHash(0, p.first, p.second);
  }
};
template <typename T>
struct GenericHash<std::vector<T>> {
  size_t operator()(const std::vector<T>& xs) const noexcept {
    auto h = GHash(0, xs.size());
    for (const auto& x : xs) {
      h = GHash(h, x);
    }
    return h;
  }
};
template <typename T, size_t N>
struct GenericHash<std::array<T, N>> {
  size_t operator()(const std::array<T, N>& xs) const noexcept {
    size_t h = 0;
    for (const auto& x : xs) {
      h = GHash(h, x);
    }
    return h;
  }
};

// very basic macro metaprogramming
#define PRIV_HPPF_FIRST(a, ...) a
#define PRIV_HPPF_SECOND(a, b, ...) b
#define PRIV_HPPF_JOIN(a,b) a ## b
#define PRIV_HPPF_IS_NEGATE(...) PRIV_HPPF_SECOND(__VA_ARGS__, 0)
#define PRIV_HPPF_NOT(x) PRIV_HPPF_IS_NEGATE(PRIV_HPPF_JOIN(PRIV_HPPF_SNOT_, x))
#define PRIV_HPPF_SNOT_0 NEGATE, 1
#define PRIV_HPPF_BOOL(x) PRIV_HPPF_NOT(PRIV_HPPF_NOT(x))
#define PRIV_HPPF_IF_ELSE(condition) PRIV_HPPF_SIF_ELSE(PRIV_HPPF_BOOL(condition))
#define PRIV_HPPF_SIF_ELSE(condition) PRIV_HPPF_JOIN(PRIV_HPPF_SIF_, condition)
#define PRIV_HPPF_SIF_1(...) __VA_ARGS__ PRIV_HPPF_SIF_1_ELSE
#define PRIV_HPPF_SIF_0(...)             PRIV_HPPF_SIF_0_ELSE
#define PRIV_HPPF_SIF_1_ELSE(...)
#define PRIV_HPPF_SIF_0_ELSE(...) __VA_ARGS__
#define PRIV_HPPF_EMPTY()
#define PRIV_HPPF_EVAL(...) PRIV_HPPF_EVAL256(__VA_ARGS__)
#define PRIV_HPPF_EVAL256(...) PRIV_HPPF_EVAL128(PRIV_HPPF_EVAL128(__VA_ARGS__))
#define PRIV_HPPF_EVAL128(...) PRIV_HPPF_EVAL64(PRIV_HPPF_EVAL64(__VA_ARGS__))
#define PRIV_HPPF_EVAL64(...) PRIV_HPPF_EVAL32(PRIV_HPPF_EVAL32(__VA_ARGS__))
#define PRIV_HPPF_EVAL32(...) PRIV_HPPF_EVAL16(PRIV_HPPF_EVAL16(__VA_ARGS__))
#define PRIV_HPPF_EVAL16(...) PRIV_HPPF_EVAL8(PRIV_HPPF_EVAL8(__VA_ARGS__))
#define PRIV_HPPF_EVAL8(...) PRIV_HPPF_EVAL4(PRIV_HPPF_EVAL4(__VA_ARGS__))
#define PRIV_HPPF_EVAL4(...) PRIV_HPPF_EVAL2(PRIV_HPPF_EVAL2(__VA_ARGS__))
#define PRIV_HPPF_EVAL2(...) PRIV_HPPF_EVAL1(PRIV_HPPF_EVAL1(__VA_ARGS__))
#define PRIV_HPPF_EVAL1(...) __VA_ARGS__
#define PRIV_HPPF_DEFER2(m) m PRIV_HPPF_EMPTY PRIV_HPPF_EMPTY()()
#define PRIV_HPPF_HAS_PARGS(...) PRIV_HPPF_BOOL(PRIV_HPPF_FIRST(PRIV_HPPF_SEOAP_ __VA_ARGS__)())
#define PRIV_HPPF_SEOAP_(...) PRIV_HPPF_BOOL(PRIV_HPPF_FIRST(PRIV_HPPF_SEOA_ __VA_ARGS__)())
#define PRIV_HPPF_SEOA_() 0
#define PRIV_HPPF_MAP(f, VS...) PRIV_HPPF_EVAL(PRIV_HPPF_MAPP(f, VS))
#define PRIV_HPPF_MAPP(f, H, T...)        \
  f H                                 \
  PRIV_HPPF_IF_ELSE(PRIV_HPPF_HAS_PARGS(T))(  \
    PRIV_HPPF_DEFER2(PRIV_HPPF_SMAPP)()(f, T) \
  )(                                  \
  )
#define PRIV_HPPF_SMAPP() PRIV_HPPF_MAPP
#define PRIV_HPPF_MAPL(f, VS...) PRIV_HPPF_EVAL(PRIV_HPPF_MAPLP(f, VS))
#define PRIV_HPPF_MAPLP(f, H, T...)              \
  f(H)                                             \
  PRIV_HPPF_IF_ELSE(PRIV_HPPF_HAS_PARGS(T))(   \
    PRIV_HPPF_DEFER2(PRIV_HPPF_SMAPLP)()(f, T) \
  )(                                               \
  )
#define PRIV_HPPF_SMAPLP() PRIV_HPPF_MAPLP

#define HPPF_MAP(f, VS...) PRIV_HPPF_EVAL(PRIV_HPPF_MAPP(f, VS))

// basic compile-time strings
template <typename ... CS>
constexpr size_t packedValueAt(size_t i, CS...) {
  return 0;
}
template <typename ... CS>
constexpr size_t packedValueAt(size_t i, size_t c, CS... cs) {
  return (i == 0) ? c : packedValueAt(i-1, cs...);
}
constexpr size_t nonZeroCharCount(size_t cb) {
  return (((cb>>(0*8))&0xff)!=0?1:0) +
         (((cb>>(1*8))&0xff)!=0?1:0) +
         (((cb>>(2*8))&0xff)!=0?1:0) +
         (((cb>>(3*8))&0xff)!=0?1:0) +
         (((cb>>(4*8))&0xff)!=0?1:0) +
         (((cb>>(5*8))&0xff)!=0?1:0) +
         (((cb>>(6*8))&0xff)!=0?1:0) +
         (((cb>>(7*8))&0xff)!=0?1:0);
}

template <size_t... cbs>
struct PackedLen {
  static constexpr size_t value = 0;
};
template <size_t cb, size_t... cbs>
struct PackedLen<cb, cbs...> {
  static constexpr size_t value = nonZeroCharCount(cb) + PackedLen<cbs...>::value;
};

template <size_t N>
  constexpr char at(size_t i, const char (&s)[N]) {
    return (i < N) ? s[i] : '\0';
  }
template <size_t N>
  constexpr size_t at8S(size_t i, size_t k, const char (&s)[N]) {
    return (k==8) ? 0 : ((static_cast<size_t>(at(i+k,s))<<(8*k))|at8S(i,k+1,s));
  }
template <size_t N>
  constexpr size_t at8(size_t i, const char (&s)[N]) {
    return at8S(i, 0, s);
  }
template <size_t... pcs>
  struct strpack {
    static constexpr char at(size_t i) {
      return char(packedValueAt(i/8, pcs...) >> (8*(i%8)));
    }
    static constexpr size_t len = PackedLen<pcs...>::value;
    static const char* str() {
      constexpr size_t msg[] = {pcs...};
      static_assert(((msg[(sizeof(msg)/sizeof(msg[0]))-1]) & 0xFF00000000000000) == 0, "compile-time string larger than internal max limit (this limit can be bumped in hlog/reflect.h)");

      static const size_t smsg[] = {pcs...};
      return reinterpret_cast<const char*>(smsg);
    }
  };
template <size_t, typename T>
  struct strpack_cons { };
template <size_t p, size_t... pcs>
  struct strpack_cons<p, strpack<pcs...>> { typedef strpack<p, pcs...> type; };
template <typename T>
  struct trim { };
template <>
  struct trim<strpack<>> {
    typedef strpack<> type;
  };
template <size_t... pcs>
  struct trim<strpack<0, pcs...>> {
    typedef strpack<0> type;
  };
template <size_t p, size_t... pcs>
  struct trim<strpack<p, pcs...>> {
    typedef typename strpack_cons<p, typename trim<strpack<pcs...>>::type>::type type;
  };
#define PRIV_HPPF_TSTR32(i,s)   HLOG_NS::at8(i,s),HLOG_NS::at8(i+8,s),HLOG_NS::at8(i+16,s),HLOG_NS::at8(i+24,s)
#define PRIV_HPPF_TSTR64(i,s)   PRIV_HPPF_TSTR32(i,s),PRIV_HPPF_TSTR32(i+32,s)
#define PRIV_HPPF_TSTR128(i,s)  PRIV_HPPF_TSTR64(i,s),PRIV_HPPF_TSTR64(i+64,s)
#define PRIV_HPPF_TSTR256(i,s)  PRIV_HPPF_TSTR128(i,s),PRIV_HPPF_TSTR128(i+128,s)
#define PRIV_HPPF_TSTR512(i,s)  PRIV_HPPF_TSTR256(i,s),PRIV_HPPF_TSTR256(i+256,s)
#define PRIV_HPPF_TSTR1024(i,s) PRIV_HPPF_TSTR512(i,s),PRIV_HPPF_TSTR512(i+512,s)

#define HPPF_TSTR_8(s)  HLOG_NS::trim<HLOG_NS::strpack<HLOG_NS::at8(0,s)>>::type
#define HPPF_TSTR_16(s) HLOG_NS::trim<HLOG_NS::strpack<HLOG_NS::at8(0,s), HLOG_NS::at8(8,s)>>::type
#define HPPF_TSTR_32(s) HLOG_NS::trim<HLOG_NS::strpack<PRIV_HPPF_TSTR32(0,s)>>::type

#define HPPF_TINY_TSTR(s)  HLOG_NS::trim<HLOG_NS::strpack<PRIV_HPPF_TSTR32(0,s)>>::type
#define HPPF_SMALL_TSTR(s) HLOG_NS::trim<HLOG_NS::strpack<PRIV_HPPF_TSTR64(0,s)>>::type
#define HPPF_LARGE_TSTR(s) HLOG_NS::trim<HLOG_NS::strpack<PRIV_HPPF_TSTR1024(0,s)>>::type
#define HPPF_TSTR(s)       HPPF_LARGE_TSTR(s)

// guard to detect only valid types
template <typename T> struct valid_type { typedef void type; };

// value-level bool -> type-level bool
template <bool f> struct tbool { };
template <>       struct tbool<true> { typedef void type; };

// test to see if a type(s) can be printed
// (we'll use this to derive print instances for reflective composite types)
template <typename T, class = decltype(std::declval<std::ostream&>() << std::declval<T>())>
  std::true_type has_shl_ostream_defined(const T&);
std::false_type has_shl_ostream_defined(...);
template <typename T>
  struct can_show_type {
    static const bool value = decltype(has_shl_ostream_defined(std::declval<T>()))::value;
  };
template <typename ... Ts>
  struct can_show_types { static const bool value = true; };
template <typename T, typename ... Ts>
  struct can_show_types<T, Ts...> { static const bool value = can_show_type<T>::value && can_show_types<Ts...>::value; };

// a standard layout tuple type
constexpr size_t alignTo(size_t x, size_t a) {
  return (x % a == 0) ? x : (a * (1 + x/a));
}
constexpr size_t szmax(size_t x, size_t y) {
  return (x > y) ? x : y;
}

template <size_t index, size_t base, typename ... Fields>
  struct offsetInfo {
    static const size_t offset       = base;
    static const size_t maxAlignment = 1;
    static const size_t size         = 0;
    static const bool   packed       = true; // but technically unit can't be memcopied

    static void defInit (uint8_t*) { }
    static void initFrom(uint8_t*, const Fields&...) { }
    static void copyFrom(uint8_t*, const uint8_t*) { }
    static void destroy (uint8_t*) { }
    static bool eq      (const uint8_t*, const uint8_t*) { return true; } // all units are equal
  };
template <size_t index, size_t base, typename Field, typename ... Fields>
  struct offsetInfo<index, base, Field, Fields...> {
    static const size_t offset = alignTo(base, alignof(Field));

    typedef offsetInfo<index+1, offset+sizeof(Field), Fields...> tail;

    static const size_t maxAlignment = szmax(alignof(Field), tail::maxAlignment);
    static const size_t size         = (offset-base) + sizeof(Field) + tail::size;
    static const bool   packed       = (offset == base) && tail::packed;

    static void defInit(uint8_t* b) {
      new (b+offset) Field();
      tail::defInit(b);
    }
    static void initFrom(uint8_t* b, const Field& f, const Fields&... fs) {
      new (b+offset) Field(f);
      tail::initFrom(b, fs...);
    }
    static void copyFrom(uint8_t* lhs, const uint8_t* rhs) {
      new (lhs+offset) Field(*reinterpret_cast<const Field*>(rhs+offset));
      tail::copyFrom(lhs, rhs);
    }
    static void destroy(uint8_t* b) {
      reinterpret_cast<Field*>(b+offset)->~Field();
      tail::destroy(b);
    }
    static bool eq(const uint8_t* lhs, const uint8_t* rhs) {
      return (*reinterpret_cast<const Field*>(lhs+offset) == *reinterpret_cast<const Field*>(rhs+offset)) ? tail::eq(lhs,rhs) : false;
    }
  };

template <size_t n, typename Offs>
  struct offsetAt : public offsetAt<n-1, typename Offs::tail> { };
template <typename Offs>
  struct offsetAt<0, Offs> { static const size_t value = Offs::offset; };

template <size_t, typename ... Fields>
  struct nth { };
template <typename Field, typename ... Fields>
  struct nth<0, Field, Fields...> { typedef Field type; };
template <size_t n, typename Field, typename ... Fields>
  struct nth<n, Field, Fields...> : nth<n-1, Fields...> { };

template <typename ... Fields>
  struct __attribute__((aligned(offsetInfo<0, 0, Fields...>::maxAlignment))) tuple {
    typedef offsetInfo<0, 0, Fields...> offs;
    static const size_t alignment = offs::maxAlignment;
    static const size_t size      = alignTo(offs::size, offs::maxAlignment);
    static const bool   packed    = (size == offs::size) && offs::packed;
    static const size_t count     = sizeof...(Fields);
    uint8_t buffer[size];

    tuple() {
      offs::defInit(this->buffer);
    }
    tuple(const Fields&... fs) {
      offs::initFrom(this->buffer, fs...);
    }
    tuple(const tuple<Fields...>& rhs) {
      offs::copyFrom(this->buffer, rhs.buffer);
    }
    ~tuple() {
      offs::destroy(this->buffer);
    }
    tuple<Fields...>& operator=(const tuple<Fields...>& rhs) {
      if (this != &rhs) {
        offs::destroy(this->buffer);
        offs::copyFrom(this->buffer, rhs.buffer);
      }
      return *this;
    }

    template <size_t k>
      typename nth<k, Fields...>::type* atP() {
        return reinterpret_cast<typename nth<k, Fields...>::type*>(this->buffer + offsetAt<k, offs>::value);
      }
    template <size_t k>
      typename nth<k, Fields...>::type& at() { return *atP<k>(); }
    template <size_t k>
      const typename nth<k, Fields...>::type* atP() const {
        return reinterpret_cast<const typename nth<k, Fields...>::type*>(this->buffer + offsetAt<k, offs>::value);
      }
    template <size_t k>
      const typename nth<k, Fields...>::type& at() const { return *atP<k>(); }
  };
template <>
  struct tuple<> {
    static const size_t alignment = 1;
    static const size_t size      = 0;

    tuple() { }
    tuple(const tuple<>&) { }
    ~tuple() { }

    tuple<>& operator=(const tuple<>&) { return *this; }
  };

template <typename ... Fields>
  inline bool operator==(const tuple<Fields...>& lhs, const tuple<Fields...>& rhs) {
    return offsetInfo<0, 0, Fields...>::eq(lhs.buffer, rhs.buffer);
  }
template <typename ... Fields>
  inline bool operator!=(const tuple<Fields...>& lhs, const tuple<Fields...>& rhs) {
    return !(lhs == rhs);
  }

// hash a tuple
template <size_t i, size_t n, typename ... Ts>
struct HashTuple {
  static size_t hash(size_t h, const tuple<Ts...>& v) {
    return HashTuple<i+1, n, Ts...>::hash(GHash(h, v.template at<i>()), v);
  }
};
template <size_t n, typename ... Ts>
struct HashTuple<n, n, Ts...> {
  static size_t hash(size_t h, const tuple<Ts...>&) { return h; }
};
template <typename ... Ts>
struct GenericHash<tuple<Ts...>> {
  size_t operator()(const tuple<Ts...>& v) const noexcept {
    return HashTuple<0, sizeof...(Ts), Ts...>::hash(0, v);
  }
};

template <size_t i, typename T>
  struct tupType { };
template <size_t i, typename ... Ts>
  struct tupType<i, tuple<Ts...>> { typedef typename nth<i, Ts...>::type type; };

template <typename T, size_t i, size_t e, typename ... Ts>
struct FindFirstIndex {
  static const size_t value = std::is_same<T, typename nth<i, Ts...>::type>::value ? i : FindFirstIndex<T, i+1, e, Ts...>::value;
};
template <typename T, size_t e, typename ... Ts>
struct FindFirstIndex<T, e, e, Ts...> {
  static const size_t value = e;
};
template <typename T, typename X>
struct FirstIndexOf { };
template <typename T, typename ... Ts>
struct FirstIndexOf<T, tuple<Ts...>> {
  typedef FindFirstIndex<T, 0, sizeof...(Ts), Ts...> Found;
  static const size_t value = Found::value;
  static const bool exists = Found::value < sizeof...(Ts);
};

// map a function over a tuple's types (useful for structurally-derived types)
template <typename ... Ts>
  struct concatT { };
template <typename ... Ts1, typename ... Ts2, typename ... Rest>
  struct concatT<tuple<Ts1...>, tuple<Ts2...>, Rest...> {
    typedef typename concatT<tuple<Ts1..., Ts2...>, Rest...>::type type;
  };
template <typename ... Ts>
  struct concatT<tuple<Ts...>> {
    typedef tuple<Ts...> type;
  };

template <template <typename T> class F, typename X>
  struct fmap { };
template <template <typename T> class F, typename H, typename ... Rest>
  struct fmap<F, tuple<H, Rest...>> {
    typedef typename concatT<tuple<typename F<H>::type>, typename fmap<F, tuple<Rest...>>::type>::type type;
  };
template <template <typename T> class F, typename H>
  struct fmap<F, tuple<H>> {
    typedef tuple<typename F<H>::type> type;
  };
template <template <typename T> class F>
  struct fmap<F, tuple<>> {
    typedef tuple<> type;
  };

template <template <typename ... Args> class F, typename ... Bindings>
struct fcurry {
  template <typename ... Remainder>
  struct ApplyRemainder {
    using type = F<Bindings..., Remainder...>;
  };
  template <typename ... Remainder>
  using type = typename ApplyRemainder<Remainder...>::type;
};

// show a tuple iff its fields can be shown
template <size_t i, size_t n, typename ... Ts>
struct ShowTuple {
  static void show(std::ostream& o, const tuple<Ts...>& v) {
    if (i > 0) o << ", ";
    o << v.template at<i>();
    ShowTuple<i+1, n, Ts...>::show(o, v);
  }
};
template <size_t n, typename ... Ts>
struct ShowTuple<n, n, Ts...> {
  static void show(std::ostream&, const tuple<Ts...>&) { }
};
template <typename ... Ts>
inline typename std::enable_if<can_show_types<Ts...>::value, std::ostream&>::type operator<<(std::ostream& o, const tuple<Ts...>& v) {
  o << "(";
  ShowTuple<0, tuple<Ts...>::count, Ts...>::show(o, v);
  o << ")";
  return o;
}

// support arithmetic on tuples
// this huge mess with macros and template specialization is just compensation for the poor support for type classes in C++
// in a PL like hobbes that gives a damn about reasoning about/with types, it'd look like:
//
//   instance Add () () ()                                                                      # the unit type where tuple addition bottoms out
//   instance (xs=(x*xs'), ys=(y*ys'), Add x y z, Add xs' ys' zs', (z*zs')=zs) => Add xs ys zs  # recursive tuple decomposition and inference at once
//
// oh well, sometimes we use the tools we're told to use and try to make the best of it
#define PRIV_HPPF_DEF_TUPLE_ARITH_OP(NAME, OP) \
  template <typename U, typename V, class = decltype(std::declval<U>() OP std::declval<V>())> \
  std::true_type has_##NAME##_defined(const U&, const V&); \
  std::false_type has_##NAME##_defined(...); \
  template <typename U, typename V> \
  struct can_##NAME { static const bool value = decltype(has_##NAME##_defined(std::declval<U>(), std::declval<V>()))::value; }; \
  template <typename U, typename V, bool _ = can_##NAME<U, V>::value> \
  struct NAME { typedef decltype(std::declval<U>() + std::declval<V>()) type; static type run(U u, V v) { return u + v; } }; \
  template <typename U, typename V> \
  struct NAME<U, V, false> { }; \
  \
  template <typename X, typename Y> \
  struct can_##NAME<tuple<X>, tuple<Y>> { \
    static const bool value = can_##NAME<X, Y>::value; \
  }; \
  template <typename X, typename ... Xs, typename Y, typename ... Ys> \
  struct can_##NAME<tuple<X, Xs...>, tuple<Y, Ys...>> { \
    static const bool value = can_##NAME<X, Y>::value && can_##NAME<tuple<Xs...>, tuple<Ys...>>::value; \
  }; \
  template <size_t i, size_t n, typename Xs, typename Ys, typename Rs> \
  struct NAME##Into { }; \
  template <size_t i, size_t n, typename ... Xs, typename ... Ys, typename ... Rs> \
  struct NAME##Into<i, n, tuple<Xs...>, tuple<Ys...>, tuple<Rs...>> { \
    typedef NAME##Into<i+1, n, tuple<Xs...>, tuple<Ys...>, tuple<Rs...>> Next; \
    static void run(const tuple<Xs...>& xs, const tuple<Ys...>& ys, tuple<Rs...>* rs) { \
      rs->template at<i>() = xs.template at<i>() OP ys.template at<i>(); \
      Next::run(xs, ys, rs); \
    } \
  }; \
  template <size_t n, typename ... Xs, typename ... Ys, typename ... Rs> \
  struct NAME##Into<n, n, tuple<Xs...>, tuple<Ys...>, tuple<Rs...>> { \
    static void run(const tuple<Xs...>&, const tuple<Ys...>&, tuple<Rs...>*) { \
    } \
  }; \
  template <typename X, typename Y> \
  struct NAME<tuple<X>, tuple<Y>, true> { \
    typedef typename NAME<X, Y>::type head_type; \
    typedef tuple<head_type> type; \
    static type run(const tuple<X>& xs, const tuple<Y>& ys) { return type(xs.template at<0>() OP ys.template at<0>()); } \
  }; \
  template <typename X, typename ... Xs, typename Y, typename ... Ys> \
  struct NAME<tuple<X, Xs...>, tuple<Y, Ys...>, true> { \
    typedef typename NAME<X, Y>::type head_type; \
    typedef typename NAME<tuple<Xs...>, tuple<Ys...>>::type tail_type; \
    typedef typename concatT<tuple<head_type>, tail_type>::type type; \
    static type run(const tuple<X, Xs...>& xs, const tuple<Y, Ys...>& ys) { \
      type result; \
      NAME##Into<0, type::count, tuple<X, Xs...>, tuple<Y, Ys...>, type>::run(xs, ys, &result); \
      return result; \
    } \
  }; \
  template <typename ... Xs, typename ... Ys> \
    inline typename std::enable_if<can_##NAME<tuple<Xs...>, tuple<Ys...>>::value, typename NAME<tuple<Xs...>, tuple<Ys...>>::type>::type operator OP(const tuple<Xs...>& xs, const tuple<Ys...>& ys) { \
      return NAME<tuple<Xs...>, tuple<Ys...>>::run(xs, ys); \
    }

PRIV_HPPF_DEF_TUPLE_ARITH_OP(add, +);
PRIV_HPPF_DEF_TUPLE_ARITH_OP(sub, -);
PRIV_HPPF_DEF_TUPLE_ARITH_OP(mul, *);
PRIV_HPPF_DEF_TUPLE_ARITH_OP(div, /);

template <size_t i, size_t n, typename X, typename Y> struct OrdEq { };
template <size_t i, size_t n, typename ... Xs, typename ... Ys>
struct OrdEq<i, n, tuple<Xs...>, tuple<Ys...>> {
  typedef OrdEq<i+1, n, tuple<Xs...>, tuple<Ys...>> Next;
  static bool eq (const tuple<Xs...>& xs, const tuple<Ys...>& ys) { return xs.template at<i>() == ys.template at<i>() && Next::eq(xs, ys); }
  static bool lt (const tuple<Xs...>& xs, const tuple<Ys...>& ys) { return xs.template at<i>() <  ys.template at<i>() || (xs.template at<i>() == ys.template at<i>() && Next::lt(xs, ys)); }
  static bool lte(const tuple<Xs...>& xs, const tuple<Ys...>& ys) { return xs.template at<i>() <= ys.template at<i>() || Next::lte(xs, ys); }
  static bool gt (const tuple<Xs...>& xs, const tuple<Ys...>& ys) { return xs.template at<i>() >  ys.template at<i>() || (xs.template at<i>() == ys.template at<i>() && Next::gt(xs, ys)); }
  static bool gte(const tuple<Xs...>& xs, const tuple<Ys...>& ys) { return xs.template at<i>() >= ys.template at<i>() || Next::gte(xs, ys); }
};
template <size_t n, typename ... Xs, typename ... Ys>
struct OrdEq<n, n, tuple<Xs...>, tuple<Ys...>> {
  static bool eq (const tuple<Xs...>&, const tuple<Ys...>&) { return true;  } // all unit values are equal
  static bool lt (const tuple<Xs...>&, const tuple<Ys...>&) { return false; } // no unit value is strictly less than any other
  static bool lte(const tuple<Xs...>&, const tuple<Ys...>&) { return true;  } // all unit values are equal
  static bool gt (const tuple<Xs...>&, const tuple<Ys...>&) { return false; } // no unit value is strictly greater than any other
  static bool gte(const tuple<Xs...>&, const tuple<Ys...>&) { return true;  } // all unit values are equal
};
template <typename ... Xs, typename ... Ys>
inline typename std::enable_if<sizeof...(Xs)==sizeof...(Ys), bool>::type operator==(const tuple<Xs...>& xs, const tuple<Ys...>& ys) {
  return OrdEq<0, sizeof...(Xs), tuple<Xs...>, tuple<Ys...>>::eq(xs, ys);
}
template <typename ... Xs, typename ... Ys>
inline typename std::enable_if<sizeof...(Xs)==sizeof...(Ys), bool>::type operator<(const tuple<Xs...>& xs, const tuple<Ys...>& ys) {
  return OrdEq<0, sizeof...(Xs), tuple<Xs...>, tuple<Ys...>>::lt(xs, ys);
}
template <typename ... Xs, typename ... Ys>
inline typename std::enable_if<sizeof...(Xs)==sizeof...(Ys), bool>::type operator<=(const tuple<Xs...>& xs, const tuple<Ys...>& ys) {
  return OrdEq<0, sizeof...(Xs), tuple<Xs...>, tuple<Ys...>>::lte(xs, ys);
}
template <typename ... Xs, typename ... Ys>
inline typename std::enable_if<sizeof...(Xs)==sizeof...(Ys), bool>::type operator>(const tuple<Xs...>& xs, const tuple<Ys...>& ys) {
  return OrdEq<0, sizeof...(Xs), tuple<Xs...>, tuple<Ys...>>::gt(xs, ys);
}
template <typename ... Xs, typename ... Ys>
inline typename std::enable_if<sizeof...(Xs)==sizeof...(Ys), bool>::type operator>=(const tuple<Xs...>& xs, const tuple<Ys...>& ys) {
  return OrdEq<0, sizeof...(Xs), tuple<Xs...>, tuple<Ys...>>::gte(xs, ys);
}

// lower a tuple of compile-time strings to a value-level list of strings
template <typename T>
struct LowerStrings { static void init(std::vector<std::string>*) { } };
template <typename Str, typename ... Strs>
struct LowerStrings<tuple<Str, Strs...>> {
  static void init(std::vector<std::string>* r) {
    r->push_back(Str::str());
    LowerStrings<tuple<Strs...>>::init(r);
  }
};
template <typename Strs>
inline std::vector<std::string> lowerStringList() {
  std::vector<std::string> r;
  LowerStrings<Strs>::init(&r);
  return r;
}

// all types have an identity
template <typename T>
inline T id(T x) { return x; }

// the trivially true proposition -- ie: C's "void" with its one value constructible
struct unit {
  unit() { }
  bool operator==(const unit&) const { return true; }
  bool operator< (const unit&) const { return false; }
};
inline std::ostream& operator<<(std::ostream& o, const unit&) { o << "()"; return o; }
template <> struct GenericHash<unit> { size_t operator()(unit) const noexcept { return 0; } };

// drop the first type from a sequence to determine a tuple type
// (useful with these macro expansions where we can't distinguish first and rest values)
template <typename ... Ts>             struct tupleTail           {                            };
template <typename T, typename ... Ts> struct tupleTail<T, Ts...> { typedef tuple<Ts...> type; };

template <typename T, typename ... Ts>
inline typename tupleTail<T, Ts...>::type entupleTail(const T&, const Ts&... vs) {
  return tuple<Ts...>(vs...);
}

// reflective structs
#define PRIV_HPPF_STRUCT_FIELD(t, n) t n;
#define PRIV_HPPF_STRUCT_FIELD_VISIT(t, n) v.template visit<t>(#n);
#define PRIV_HPPF_STRUCT_FIELD_EQ(t, n) && this->n == rhs.n
#define PRIV_HPPF_STRUCT_FIELD_TYARGL(t, n) , t
#define PRIV_HPPF_STRUCT_FIELD_NAME_T(_, n) , HPPF_SMALL_TSTR(#n)
#define PRIV_HPPF_STRUCT_FIELD_NLOOKUP(t, n) if (o == idx) { return #n; } ++o;
#define PRIV_HPPF_STRUCT_FIELD_SHOW(_, n) ss << ", " << #n << "=" << v.n;

#ifdef HPPF_SKIP_LIFTED_LABELS
#define PRIV_HPPF_DEFINE_STRUCT_LIFTED_LABELS(FIELDS...)
#else
#define PRIV_HPPF_DEFINE_STRUCT_LIFTED_LABELS(FIELDS...) typedef typename HLOG_NS::tupleTail<int PRIV_HPPF_MAP(PRIV_HPPF_STRUCT_FIELD_NAME_T, FIELDS)>::type field_name_list;
#endif

#define HPPF_DEFINE_STRUCT(T, FIELDS...) \
  struct T { \
    PRIV_HPPF_MAP(PRIV_HPPF_STRUCT_FIELD, FIELDS) /* struct fields */ \
    static const bool __attribute__((unused)) is_hmeta_struct = true; /* identify this type as a struct */ \
    typedef typename HLOG_NS::tupleTail<int PRIV_HPPF_MAP(PRIV_HPPF_STRUCT_FIELD_TYARGL, FIELDS)>::type as_tuple_type; \
    PRIV_HPPF_DEFINE_STRUCT_LIFTED_LABELS(FIELDS) \
    static std::string _hmeta_struct_type_name() { return #T; } \
    template <typename V> \
      static void meta(V& v) { \
        PRIV_HPPF_MAP(PRIV_HPPF_STRUCT_FIELD_VISIT, FIELDS) \
      } \
    template <size_t idx> \
      static const char* _hmeta_field_name() { \
        size_t o = 0; \
        PRIV_HPPF_MAP(PRIV_HPPF_STRUCT_FIELD_NLOOKUP, FIELDS) \
        return "???"; \
      } \
    template <typename X> \
      inline typename std::enable_if<std::is_base_of<X, T>::value, bool>::type operator==(const X& rhs) const { \
        return true PRIV_HPPF_MAP(PRIV_HPPF_STRUCT_FIELD_EQ, FIELDS); \
      } \
    template <typename X> \
      inline typename std::enable_if<std::is_base_of<X, T>::value, bool>::type operator!=(const X& rhs) const { \
        return !(*this == rhs); \
      } \
    template <typename X> \
      friend inline typename std::enable_if<std::is_base_of<X,T>::value && HLOG_NS::can_show_type<typename T::as_tuple_type>::value,std::ostream&>::type operator<<(std::ostream& o, const X& v) { \
        using namespace HLOG_NS; \
        std::ostringstream ss; \
        PRIV_HPPF_MAP(PRIV_HPPF_STRUCT_FIELD_SHOW, FIELDS); \
        o << "{ " << ss.str().substr(2) << " }"; \
        return o; \
      } \
      static const bool __attribute__((unused)) _h_has_type_name = true; \
      static std::string _h_type_name() { return #T; } \
  }

// hash reflective structs (piggybacking on the equivalent definition for tuples)
template <typename T>
struct GenericHash<T, typename tbool<T::is_hmeta_struct>::type> {
  size_t operator()(const T& x) const noexcept {
    GenericHash<typename T::as_tuple_type> sh;
    return sh(*reinterpret_cast<const typename T::as_tuple_type*>(&x));
  }
};

// lift self arithmetic on reflective structs (just piggyback on what we already did for standard layout tuples)
#define PRIV_HPPF_DEF_STRUCT_ARITH_OP(NAME, OP) \
  template <typename X> \
    inline typename std::enable_if<X::is_hmeta_struct && can_##NAME<typename X::as_tuple_type, typename X::as_tuple_type>::value, X>::type operator OP(const X& x, const X& y) { \
      auto result = NAME<typename X::as_tuple_type, typename X::as_tuple_type>::run(*reinterpret_cast<const typename X::as_tuple_type*>(&x), *reinterpret_cast<const typename X::as_tuple_type*>(&y)); \
      return *reinterpret_cast<const X*>(&result); \
    }

PRIV_HPPF_DEF_STRUCT_ARITH_OP(add, +);
PRIV_HPPF_DEF_STRUCT_ARITH_OP(sub, -);
PRIV_HPPF_DEF_STRUCT_ARITH_OP(mul, *);
PRIV_HPPF_DEF_STRUCT_ARITH_OP(div, /);

// reflective enumerations
#define PRIV_HPPF_ENUM_CTOR_DEF(n) n ,
#define PRIV_HPPF_ENUM_CTOR_CTOR(n) static SelfT n() { return SelfT(SelfT::Enum::n); }
#define PRIV_HPPF_ENUM_META(n) m.push_back(std::pair<std::string,uint32_t>(#n, static_cast<uint32_t>(SelfT::Enum::n)));
#define PRIV_HPPF_ENUM_SHOW(n) case SelfT::Enum::n : return "|" #n "|";
#define PRIV_HPPF_ENUM_CTOR_COUNT(n) +1

#define HPPF_DEFINE_PACKED_ENUM(T, R, CTORS...) \
  struct T { \
    static const bool is_hmeta_enum = true; \
    static const uint32_t ctorCount = 0 PRIV_HPPF_MAP(PRIV_HPPF_ENUM_CTOR_COUNT, CTORS); \
    typedef R rep_t; \
    enum class Enum : R { \
      PRIV_HPPF_MAP(PRIV_HPPF_ENUM_CTOR_DEF, CTORS) \
    }; \
    Enum value; \
    T() : value() { } \
    T(Enum v) : value(v) { } \
    T(R r) : value(static_cast<Enum>(r)) { } \
    T& operator=(Enum v) { this->value = v; return *this; } \
    operator Enum() const { return this->value; } \
    operator R() const { return static_cast<R>(this->value); } \
    typedef T SelfT; \
    PRIV_HPPF_MAP(PRIV_HPPF_ENUM_CTOR_CTOR, CTORS) \
    typedef std::vector<std::pair<std::string,R>> MetaSeq; \
    static MetaSeq meta() { \
      MetaSeq m; \
      PRIV_HPPF_MAP(PRIV_HPPF_ENUM_META, CTORS); \
      return m; \
    } \
    static R toOrd(Enum x) { return static_cast<R>(x); } \
    static Enum fromOrd(R x) { return static_cast<Enum>(x); } \
    bool operator==(const T& rhs) const { return this->value == rhs.value; } \
    bool operator==(Enum rhs) const { return this->value == rhs; } \
    bool operator!=(const T& rhs) const { return this->value != rhs.value; } \
    bool operator!=(Enum rhs) const { return this->value != rhs; } \
    bool operator<(const T& rhs) const { return this->value < rhs.value; } \
    bool operator<(Enum rhs) const { return this->value < rhs; } \
    std::string show() const { switch (this->value) { PRIV_HPPF_MAP(PRIV_HPPF_ENUM_SHOW, CTORS); default: return "?value=" + HLOG_NS::string::from(uint64_t(rep_t(this->value))) + "?"; } } \
    friend inline std::ostream& operator<<(std::ostream& o, const T& v) { o << v.show(); return o; } \
  }

#define HPPF_DEFINE_ENUM(T, CTORS...) HPPF_DEFINE_PACKED_ENUM(T, uint32_t, ## CTORS)

// reflective enums with constructor values
#define PRIV_HPPF_ENUM_V_CTOR_DEF(n, v) n = v,
#define PRIV_HPPF_ENUM_V_CTOR_CTOR(n, _) static SelfT n() { return SelfT(SelfT::Enum::n); }
#define PRIV_HPPF_ENUM_V_META(n, _) m.push_back(std::pair<std::string,uint32_t>(#n, static_cast<uint32_t>(SelfT::Enum::n)));
#define PRIV_HPPF_ENUM_V_SHOW(n, _) case SelfT::Enum::n : return "|" #n "|";
#define PRIV_HPPF_ENUM_V_CTOR_COUNT(n, _) +1

#define PRIV_HPPF_ENUM_V_ORD_CTOR_DEF(n, _) ord_##n ,
#define PRIV_HPPF_ENUM_V_TO_ORD(n, _) case SelfT::Enum::n: return static_cast<SelfT::rep_t>(SelfT::EnumOrd::ord_##n) - 1;
#define PRIV_HPPF_ENUM_V_FROM_ORD(n, _) , SelfT::Enum::n

#define HPPF_DEFINE_PACKED_ENUM_V(T, R, CTORS...) \
  struct T { \
    static const bool is_hmeta_enum = true; \
    static const uint32_t ctorCount = 0 PRIV_HPPF_MAP(PRIV_HPPF_ENUM_V_CTOR_COUNT, CTORS); \
    typedef R rep_t; \
    enum class Enum : R { \
      PRIV_HPPF_MAP(PRIV_HPPF_ENUM_V_CTOR_DEF, CTORS) \
    }; \
    Enum value; \
    enum class EnumOrd : R { \
      __zero__ = 0, \
      PRIV_HPPF_MAP(PRIV_HPPF_ENUM_V_ORD_CTOR_DEF, CTORS) \
    }; \
    T() : value() { } \
    T(Enum v) : value(v) { } \
    T(R r) : value(static_cast<Enum>(r)) { } \
    T& operator=(Enum v) { this->value = v; return *this; } \
    operator Enum() const { return this->value; } \
    operator R() const { return static_cast<R>(this->value); } \
    typedef T SelfT; \
    PRIV_HPPF_MAP(PRIV_HPPF_ENUM_V_CTOR_CTOR, CTORS) \
    typedef std::vector<std::pair<std::string,R>> MetaSeq; \
    static MetaSeq meta() { \
      MetaSeq m; \
      PRIV_HPPF_MAP(PRIV_HPPF_ENUM_V_META, CTORS); \
      return m; \
    } \
    static rep_t toOrd(Enum x) { switch (x) { PRIV_HPPF_MAP(PRIV_HPPF_ENUM_V_TO_ORD, CTORS) default: return -1; } } \
    static Enum fromOrd(R x) { static const Enum vs[] = { Enum() PRIV_HPPF_MAP(PRIV_HPPF_ENUM_V_FROM_ORD, CTORS) }; ++x; return x < (sizeof(vs)/sizeof(vs[0]))? vs[x] : Enum(); } \
    bool operator==(const T& rhs) const { return this->value == rhs.value; } \
    bool operator==(Enum rhs) const { return this->value == rhs; } \
    bool operator!=(const T& rhs) const { return this->value != rhs.value; } \
    bool operator!=(Enum rhs) const { return this->value != rhs; } \
    bool operator<(const T& rhs) const { return this->value < rhs.value; } \
    bool operator<(Enum rhs) const { return this->value < rhs; } \
    std::string show() const { switch (this->value) { PRIV_HPPF_MAP(PRIV_HPPF_ENUM_V_SHOW, CTORS); default: return "?value=" + HLOG_NS::string::from(uint64_t(rep_t(this->value))) + "?"; } } \
    friend inline std::ostream& operator<<(std::ostream& o, const T& v) { o << v.show(); return o; } \
  }

// generic hashing for enums
template <typename T>
struct GenericHash<T, typename tbool<T::is_hmeta_enum>::type> {
  size_t operator()(const T& x) const noexcept {
    GenericHash<typename T::rep_t> rh;
    return rh(x);
  }
};

#define HPPF_DEFINE_ENUM_V(T, CTORS...) HPPF_DEFINE_PACKED_ENUM_V(T, uint32_t, ## CTORS)

// standard layout sum type with efficient dispatch
template <bool f, typename T, typename F>
  struct TIfF { };
template <typename T, typename F>
  struct TIfF<true, T, F> { typedef T type; };
template <typename T, typename F>
  struct TIfF<false, T, F> { typedef F type; };
template <typename T>
  struct TSizeOfF { static const size_t value = sizeof(T); };
template <typename T>
  struct TAlignOfF { static const size_t value = alignof(T); };
template <template <class> class SzP, typename T0, typename ... Ts>
  struct maximum { static const size_t value = SzP<T0>::value; typedef T0 type; };
template <template <class> class SzP, typename T0, typename T1, typename ... Ts>
  struct maximum<SzP, T0, T1, Ts...> : public maximum<SzP, typename TIfF<SzP<T1>::value < SzP<T0>::value, T0, T1>::type, Ts...> { };

template <typename T, typename ... Ctors>
  struct CtorIndexOf { static const uint32_t value = 0; };
template <typename T, typename ... Ctors>
  struct CtorIndexOf<T, T, Ctors...> { static const uint32_t value = 0; };
template <typename T, typename Ctor, typename ... Ctors>
  struct CtorIndexOf<T, Ctor, Ctors...> { static const uint32_t value = 1 + CtorIndexOf<T, Ctors...>::value; };
template <typename T, typename ... Ts>
  struct First { typedef T type; };

template <size_t i, typename R, template <size_t,class,class> class F, typename U, typename A, typename ... Ctors>
struct variantAppInit {
};
template <size_t i, typename R, template <size_t,class,class> class F, typename U, typename ... Args, typename ... Ctors>
struct variantAppInit<i, R, F, U, tuple<Args...>, Ctors...> {
  typedef R (*PF)(void*, Args...);
  static bool init(PF*) { return true; }
};
template <size_t i, typename R, template <size_t,class,class> class F, typename U, typename ... Args, typename Ctor, typename ... Ctors>
struct variantAppInit<i, R, F, U, tuple<Args...>, Ctor, Ctors...> {
  typedef R (*PF)(void*, Args...);
  static bool init(PF* pfs) {
    typedef R (*TPF)(Ctor*, Args...);
    TPF tpf = &F<i, Ctor, U>::fn;
    pfs[i] = reinterpret_cast<PF>(tpf);
    return variantAppInit<i+1, R, F, U, tuple<Args...>, Ctors...>::init(pfs);
  }
};

template <typename R, template <size_t,class,class> class F, typename U, typename V, typename ... Args>
struct variantApp {
};
template <template <size_t,class,class> class F, typename U, typename ... Ctors, typename ... Args>
struct variantApp<void, F, U, tuple<Ctors...>, Args...> {
  typedef void (*PF)(void*, Args...);
  static PF* fns() {
    static PF fvec[sizeof...(Ctors)];
    static bool init = variantAppInit<0, void, F, U, tuple<Args...>, Ctors...>::init(fvec);
    return fvec;
    if (init) return fvec; // <-- pointless, but prevents an unused variable error
  }

  static void apply(uint32_t id, void* payload, Args... args) {
    fns()[id](payload, args...);
  }
};

template <typename R, template <size_t,class,class> class F, typename U, typename ... Ctors, typename ... Args>
struct variantApp<R, F, U, tuple<Ctors...>, Args...> {
  typedef R (*PF)(void*, Args...);
  static PF* fns() {
    static PF fvec[sizeof...(Ctors)];
    static bool init = variantAppInit<0, R, F, U, tuple<Args...>, Ctors...>::init(fvec);
    return fvec;
    if (init) return fvec; // <-- pointless, but prevents an unused variable error
  }

  static R apply(uint32_t id, void* payload, Args... args) {
    return fns()[id](payload, args...);
  }
};

template <size_t, typename T, typename U>
struct variantPayloadCtor {
  static void fn(T* p, const void* rhsp) {
    new (p) T(*reinterpret_cast<const T*>(rhsp));
  }
};
template <size_t, typename T, typename U>
struct variantPayloadDtor {
  static void fn(T* p) {
    p->~T();
  }
};
template <size_t, typename T, typename U>
struct variantPayloadEq {
  static bool fn(T* lhs, const void* rhs) {
    return *lhs == *reinterpret_cast<const T*>(rhs);
  }
};

template <typename ... Ctors>
  class __attribute__((aligned(alignof(typename maximum<TAlignOfF, Ctors...>::type)))) variant {
  public:
    static const size_t count = sizeof...(Ctors);
    static_assert(count > 0, "Empty variants are impossible to construct");

    static const size_t alignment = alignof(typename maximum<TAlignOfF, Ctors...>::type);

    typedef variantApp<void, variantPayloadCtor, void, tuple<Ctors...>, const void*> VCCtor;
    typedef variantApp<void, variantPayloadDtor, void, tuple<Ctors...>>              VDtor;

    variant() : tag(0) {
      new (this->storage) typename First<Ctors...>::type();
    }
    template <typename T>
      variant(const T& t) : tag(CtorIndexOf<T, Ctors...>::value) {
        static_assert(CtorIndexOf<T, Ctors...>::value < sizeof...(Ctors), "Constructor type isn't part of variant");
        new (this->storage) T(t);
      }
    variant(const variant<Ctors...>& rhs) : tag(rhs.tag) {
      VCCtor::apply(this->tag, this->storage, rhs.storage);
    }
    ~variant() {
      VDtor::apply(this->tag, this->storage);
    }
    variant<Ctors...>& operator=(const variant<Ctors...>& rhs) {
      if (this != &rhs) {
        VDtor::apply(this->tag, this->storage);
        this->tag = rhs.tag;
        VCCtor::apply(this->tag, this->storage, rhs.storage);
      }
      return *this;
    }

    template <typename T>
      T* get() { return findByCtor<T>(); }
    template <typename T>
      const T* get() const { return findByCtor<T>(); }

    template <size_t N>
      typename nth<N, Ctors...>::type* constructor() { return size_t(this->tag) == N ? reinterpret_cast<typename nth<N, Ctors...>::type*>(this->storage) : nullptr; }
    template <size_t N>
      const typename nth<N, Ctors...>::type* constructor() const { return size_t(this->tag) == N ? reinterpret_cast<const typename nth<N, Ctors...>::type*>(this->storage) : nullptr; }

    template <typename R, template <size_t,class,class> class F, typename U, typename ... Args>
      R apply(Args... args) {
        return variantApp<R, F, U, tuple<Ctors...>, Args...>::apply(this->tag, this->storage, args...);
      }
    template <typename R, template <size_t,class,class> class F, typename U, typename ... Args>
      R apply(Args... args) const {
        return variantApp<R, F, U, tuple<Ctors...>, Args...>::apply(this->tag, const_cast<void*>(reinterpret_cast<const void*>(this->storage)), args...);
      }
  public:
    const uint32_t& unsafeTag() const     { return this->tag; }
    uint32_t&       unsafeTag()           { return this->tag; }
    const void*     unsafePayload() const { return this->storage; }
    void*           unsafePayload()       { return this->storage; }
  private:
    uint32_t tag;
    union {
      char storage[maximum<TSizeOfF, Ctors...>::value];
      typename maximum<TAlignOfF, Ctors...>::type maxAlignedT;
    };

    template <typename T>
      T* findByCtor() const {
        static_assert(CtorIndexOf<T, Ctors...>::value < sizeof...(Ctors), "Constructor type isn't part of variant");

        if (this->tag == CtorIndexOf<T, Ctors...>::value) {
          return const_cast<T*>(reinterpret_cast<const T*>(this->storage));
        } else {
          return 0;
        }
      }
  };
template <typename T, typename ... Ctors>
T* get(variant<Ctors...>& v) { return v.template get<T>(); }
template <typename T, typename ... Ctors>
const T* get(const variant<Ctors...>& v) { return v.template get<T>(); }

template <typename ... Ctors>
inline bool operator==(const variant<Ctors...>& lhs, const variant<Ctors...>& rhs) {
  return lhs.unsafeTag() == rhs.unsafeTag() &&
         variantApp<bool, variantPayloadEq, void, tuple<Ctors...>, const void*>::apply(lhs.unsafeTag(), const_cast<void*>(reinterpret_cast<const void*>(lhs.unsafePayload())), reinterpret_cast<const void*>(rhs.unsafePayload()));
}
template <typename ... Ctors>
inline bool operator!=(const variant<Ctors...>& lhs, const variant<Ctors...>& rhs) {
  return !(lhs == rhs);
}

template <size_t, typename T, typename U>
struct VariantHash {
  static size_t fn(T* p, size_t h) {
    return GHash(h, *p);
  }
};
template <typename ... Ctors>
struct GenericHash<variant<Ctors...>> {
  size_t operator()(const variant<Ctors...>& v) const noexcept {
    return v.template apply<size_t, VariantHash, void>(GHash(0, v.unsafeTag()));
  }
};

template <typename ... Ts>             struct variantTail           {                              };
template <typename T, typename ... Ts> struct variantTail<T, Ts...> { typedef variant<Ts...> type; };

template <typename T>
  struct toVariant { };
template <typename ... Ts>
  struct toVariant<tuple<Ts...>> { typedef variant<Ts...> type; };

template <typename T>
  struct toTuple { };
template <typename ... Ts>
  struct toTuple<variant<Ts...>> { typedef tuple<Ts...> type; };

template <typename T>
using optional = variant<unit, T>;
template <typename T>
const T* some(const optional<T>& x) {
  return x.template constructor<1>();
}

// show a variant iff its constructors can be shown
template <size_t, typename T, typename U>
  struct ShowVariantPayload {
    static void fn(T* x, std::ostream* out) {
      *out << *x;
    }
  };

template <typename ... Ts>
  inline typename std::enable_if<0<sizeof...(Ts) && can_show_types<Ts...>::value, std::ostream&>::type operator<<(std::ostream& o, const variant<Ts...>& cv) {
    variant<Ts...>& v = *const_cast<variant<Ts...>*>(&cv);
    o << "|" << v.unsafeTag() << "=";
    variantApp<void, ShowVariantPayload, void, tuple<Ts...>, std::ostream*>::apply(v.unsafeTag(), const_cast<void*>(reinterpret_cast<const void*>(v.unsafePayload())), &o);
    o << "|";
    return o;
  }

// a function type blocking null construction
// (we will want to immediately use this for generic variant deconstruction)
template <typename Sig>
struct NonNullFunction {
  using Rep = std::function<Sig>;
  Rep f;
  template <typename F>
  NonNullFunction(const F& f) : f(f) { }
};

// reflective variants
#define PRIV_HPPF_VARIANT_TYARGL(_, t)      , t
#define PRIV_HPPF_VARIANT_CTOR(n, t)        static SelfT n(t const& x) { SelfT r; r.tag = Enum::tag_##n; new (&r.n##_data) t(x); return r; }
#define PRIV_HPPF_VARIANT_DTOR(n, t)        t const* n() const { return this->tag == Enum::tag_##n ? &this->n##_data : nullptr; }
#define PRIV_HPPF_VARIANT_PCOPY(n, t)       case Enum::tag_##n: new (&this->n##_data) t(rhs.n##_data); break;
#define PRIV_HPPF_VARIANT_GINIT(n, t)       case Enum::tag_##n: new (&this->n##_data) t(); v.template init<t>(&this->n##_data); break;
#define PRIV_HPPF_VARIANT_PDESTROY(n, t)    case Enum::tag_##n: { typedef t PRIV_DT; reinterpret_cast<PRIV_DT*>(&this->n##_data)->~PRIV_DT(); } break;
#define PRIV_HPPF_VARIANT_CTOR_TAG(n, t)    tag_##n,
#define PRIV_HPPF_VARIANT_CTOR_DATA(n, t)   t n##_data;
#define PRIV_HPPF_VARIANT_CTOR_COUNT(_,__)   +1
#define PRIV_HPPF_VARIANT_EQCASE(n, _)      case Enum::tag_##n: return (this->n##_data == rhs.n##_data);
#define PRIV_HPPF_VARIANT_META(n, t)        v.template ctor<t>(#n, static_cast<uint32_t>(Enum::tag_##n));
#define PRIV_HPPF_VARIANT_VCASE(n, _)       case Enum::tag_##n: return v. n (this->n##_data);
#define PRIV_HPPF_VARIANT_GVCASE(n, t)      case Enum::tag_##n: return v.template visit<t>(#n, static_cast<uint32_t>(Enum::tag_##n), this->n##_data);
#define PRIV_HPPF_VARIANT_SHOW(n, t)        case X::Enum::tag_##n: o << "|" << #n << "=" << v.n##_data << "|"; break;
#define PRIV_HPPF_VARIANT_HMCTOR_NAME(n, _) case Enum::tag_##n: return #n;
#define PRIV_HPPF_VARIANT_VAPP_DECL(n, t)   HLOG_NS::NonNullFunction<R(t const&)> n;
#define PRIV_HPPF_VARIANT_VAPP_APP(n, t)    case Enum::tag_##n: return f.n.f(this->n##_data);
#define PRIV_HPPF_VARIANT_ID_DEF(n, _)      n,

#define PRIV_HPPF_VARIANT_CTOR_NAME_T(n, _) , HPPF_SMALL_TSTR(#n)
#ifdef HPPF_SKIP_LIFTED_LABELS
#define PRIV_HPPF_DEFINE_VARIANT_LIFTED_LABELS(CTORS...)
#else
#define PRIV_HPPF_DEFINE_VARIANT_LIFTED_LABELS(CTORS...) typedef typename HLOG_NS::tupleTail<int PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_CTOR_NAME_T, CTORS)>::type ctor_name_list;
#endif

#define HPPF_DEFINE_VARIANT(T, CTORS...) \
  struct T { \
    static const bool __attribute__((unused)) is_hmeta_variant = true; \
    typedef T SelfT; \
    static const uint32_t __attribute__((unused)) ctorCount = 0 PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_CTOR_COUNT, CTORS); \
    typedef typename HLOG_NS::variantTail<int PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_TYARGL, CTORS)>::type as_variant_type; \
    PRIV_HPPF_DEFINE_VARIANT_LIFTED_LABELS(CTORS) \
    T() : tag(Enum::COUNT) { } \
    PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_CTOR, CTORS) \
    PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_DTOR, CTORS) \
    T(const T& rhs) : tag(rhs.tag) { \
      switch (this->tag) { \
      PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_PCOPY, CTORS) \
      default: break; \
      } \
    } \
    ~T() { \
      switch (this->tag) { \
      PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_PDESTROY, CTORS) \
      default: break; \
      } \
    } \
    T& operator=(const T& rhs) { \
      if (this == &rhs) return *this; \
      switch (this->tag) { \
      PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_PDESTROY, CTORS) \
      default: break; \
      } \
      this->tag = rhs.tag; \
      switch (this->tag) { \
      PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_PCOPY, CTORS) \
      default: break; \
      } \
      return *this; \
    } \
    template <typename V> \
      static void meta(V& v) { \
        PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_META, CTORS) \
      } \
    template <typename R> \
    struct VariantApp { \
      PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_VAPP_DECL, CTORS) \
    }; \
    template <typename R> \
    R caseOf(const VariantApp<R>& f) const { \
      switch (this->tag) { \
      default: \
      PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_VAPP_APP, CTORS) \
      } \
    } \
    template <typename V> \
      void gvisit(V& v) const { \
        switch (this->tag) { \
        PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_GVCASE, CTORS) \
        default: break; \
        } \
      } \
    template <typename V> \
      void ginit(uint32_t t, V& v) { \
        switch (this->tag) { \
        PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_PDESTROY, CTORS) \
        default: break; \
        } \
        this->tag = static_cast<Enum>(t); \
        switch (this->tag) { \
        PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_GINIT, CTORS) \
        default: break; \
        } \
      } \
    template <size_t i> \
      static const char* _hmeta_ctor_name() { \
        static_assert(i < as_variant_type::count, "Invalid constructor index in variant"); \
        switch (static_cast<Enum>(i)) { \
        PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_HMCTOR_NAME, CTORS) \
        default: return "impossible"; \
        } \
      } \
    template <size_t i> \
      static uint32_t _hmeta_ctor_id() { \
        static_assert(i < as_variant_type::count, "Invalid constructor index in variant"); \
        return static_cast<uint32_t>(i); \
      } \
    template <typename X> \
      inline typename std::enable_if<std::is_base_of<X, T>::value, bool>::type operator==(const X& rhs) const { \
        if (this->tag != rhs.tag) { \
          return false; \
        } else { \
          switch (this->tag) { \
          PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_EQCASE, CTORS) \
          default: return false; \
          } \
        } \
      } \
    template <typename X> \
      inline typename std::enable_if<std::is_base_of<X, T>::value, bool>::type operator!=(const X& rhs) const { \
        return !(*this == rhs); \
      } \
    enum class ID : uint32_t { \
      PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_ID_DEF, CTORS) \
    }; \
    uint32_t constructorID() const { return uint32_t(this->tag); } \
  private: \
    enum class Enum : uint32_t { \
      PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_CTOR_TAG, CTORS) \
      COUNT \
    }; \
    Enum tag; \
    union { \
      char data[1]; \
      PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_CTOR_DATA, CTORS) \
    }; \
  public: \
    template <typename X> \
      friend inline typename std::enable_if<std::is_base_of<X,T>::value && HLOG_NS::can_show_type<T::as_variant_type>::value,std::ostream&>::type operator<<(std::ostream& o, const X& v) { \
        switch (v.tag) { \
        PRIV_HPPF_MAP(PRIV_HPPF_VARIANT_SHOW, CTORS) \
        default: o << "?value=" << HLOG_NS::string::from(uint32_t(v.tag)) << "?"; break; \
        } \
        return o; \
      } \
    static const bool __attribute__((unused)) _h_has_type_name = true; \
    static std::string _h_type_name() { return #T; } \
  }

template <typename T>
struct GenericHash<T, typename tbool<T::is_hmeta_variant>::type> {
  size_t operator()(const T& v) const noexcept {
    GenericHash<typename T::as_variant_type> vh;
    return vh(*reinterpret_cast<const typename T::as_variant_type*>(&v));
  }
};

// a type carrying field or constructor names at the type level
// (map into generic tuple or variant types without forgetting source field names)
template <typename NameList, typename T>
struct __attribute__((aligned(alignof(T)))) withNames {
  typedef NameList name_list;
  typedef T        value_type;

  T value;
};
template <typename NameList, typename T, typename U>
inline bool operator==(const withNames<NameList, T>& lhs, const U& rhs) {
  return lhs.value == rhs;
}
template <typename NameList, typename T, typename U>
inline bool operator==(const U& lhs, const withNames<NameList, T>& rhs) {
  return lhs == rhs.value;
}
template <typename NameList, typename T>
struct GenericHash<withNames<NameList, T>> {
  size_t operator==(const withNames<NameList, T>& x) {
    return GHash(0, x.value);
  }
};

// show a withNames tuple (as a struct) iff its fields can be shown
template <size_t i, size_t n, typename ... Ts>
  struct ShowTupleWithNames {
    static void show(const std::vector<std::string>& names, std::ostream& o, const tuple<Ts...>& v) {
      if (i > 0) o << ", ";
      if (i < names.size()) {
        o << names[i];
      } else {
        o << "???";
      }
      o << "=" << v.template at<i>();
      ShowTupleWithNames<i+1, n, Ts...>::show(names, o, v);
    }
  };
template <size_t n, typename ... Ts>
  struct ShowTupleWithNames<n, n, Ts...> {
    static void show(const std::vector<std::string>&, std::ostream&, const tuple<Ts...>&) { }
  };

template <typename NameList, typename ... Ts>
  inline typename std::enable_if<can_show_types<Ts...>::value, std::ostream&>::type operator<<(std::ostream& o, const withNames<NameList, tuple<Ts...>>& v) {
    static std::vector<std::string> names = lowerStringList<NameList>();
    o << "{";
    ShowTupleWithNames<0, sizeof...(Ts), Ts...>::show(names, o, v.value);
    return o << "}";
  }

// show a withNames variant iff its constructors can be shown
template <typename NameList, typename ... Ts>
  inline typename std::enable_if<0<sizeof...(Ts) && can_show_types<Ts...>::value, std::ostream&>::type operator<<(std::ostream& o, const withNames<NameList, variant<Ts...>>& cv) {
    static std::vector<std::string> names = lowerStringList<NameList>();
    variant<Ts...>& v = *const_cast<variant<Ts...>*>(&cv.value);
    auto tid = v.unsafeTag();
    if (tid < names.size()) {
      o << "|" << names[tid] << "=";
    } else {
      o << "|" << tid << "=";
    }
    variantApp<void, ShowVariantPayload, void, tuple<Ts...>, std::ostream*>::apply(v.unsafeTag(), const_cast<void*>(reinterpret_cast<const void*>(v.unsafePayload())), &o);
    return o << "|";
  }

// arrays with dynamically-determined length
template <typename T>
  struct array {
    typedef array<T>* ptr;

    size_t size;
    T      data[1]; // unknown length, bytes inline with this structure

    const T& operator[](size_t i) const { return this->data[i]; }
          T& operator[](size_t i)       { return this->data[i]; }
  };
template <typename T> const T* begin(const array<T>* d) { return d->data; }
template <typename T> const T* end  (const array<T>* d) { return d->data + d->size; }
template <typename T>       T* begin(      array<T>* d) { return d->data; }
template <typename T>       T* end  (      array<T>* d) { return d->data + d->size; }

template <typename T> const T* begin(const array<T>& d) { return d.data; }
template <typename T> const T* end  (const array<T>& d) { return d.data + d.size; }
template <typename T>       T* begin(      array<T>& d) { return d.data; }
template <typename T>       T* end  (      array<T>& d) { return d.data + d.size; }

// arrays with dynamically-determined length but with a fixed-capacity
template <typename T, size_t N>
  struct carray {
    size_t size;
    T      data[N];

    const T& operator[](size_t i) const { return this->data[i]; }
          T& operator[](size_t i)       { return this->data[i]; }
  };
template <size_t N>
  struct carray<char,N> {
    size_t size;
    char   data[N];

    const char& operator[](size_t i) const { return this->data[i]; }
          char& operator[](size_t i)       { return this->data[i]; }

    carray<char,N>& operator=(const char* s) {
      size_t n = std::min<size_t>(N, strlen(s));
      memcpy(this->data, s, n);
      this->size = n;
      return *this;
    }
  };

template <typename T, size_t N> const T* begin(const carray<T,N>* d) { return d->data; }
template <typename T, size_t N> const T* end  (const carray<T,N>* d) { return d->data + d->size; }
template <typename T, size_t N>       T* begin(      carray<T,N>* d) { return d->data; }
template <typename T, size_t N>       T* end  (      carray<T,N>* d) { return d->data + d->size; }

template <typename T, size_t N> const T* begin(const carray<T,N>& d) { return d.data; }
template <typename T, size_t N> const T* end  (const carray<T,N>& d) { return d.data + d.size; }
template <typename T, size_t N>       T* begin(      carray<T,N>& d) { return d.data; }
template <typename T, size_t N>       T* end  (      carray<T,N>& d) { return d.data + d.size; }

// define opaque type aliases (inherit from 'alias', add a static 'name()' function)
template <typename RepTy>
struct alias {
  typedef alias<RepTy> SelfT;

  static const bool is_hmeta_alias = true;
  typedef RepTy type;
  RepTy value;

  // allow quiet truncation down to representation if necessary
  // (NOTE: after truncation, we've lost the distinction that this alias makes!)
  operator type() const { return this->value; }

  alias() : value() { }
  alias(const RepTy& x) : value(x) { }
  constexpr alias(const SelfT&) = default;
  alias& operator=(const SelfT&) = default;
  bool operator==(const SelfT& rhs) const { return this->value == rhs.value; }
  bool operator==(const RepTy& rhs) const { return this->value == rhs; }
};
template <typename RepTy>
struct GenericHash<alias<RepTy>> {
  size_t operator()(const alias<RepTy>& x) const noexcept {
    return GHash(0, x.value);
  }
};

#define HPPF_DEFINE_TYPE_ALIAS_AS(PRIV_ATY, N, PRIV_REPTY) \
  struct PRIV_ATY : public HLOG_NS::alias<PRIV_REPTY> { \
    using HLOG_NS::alias<PRIV_REPTY>::alias; \
    static const char* name() { return #N; } \
  }

#define HPPF_DEFINE_TYPE_ALIAS(PRIV_ATY, PRIV_REPTY) HPPF_DEFINE_TYPE_ALIAS_AS(PRIV_ATY, PRIV_ATY, PRIV_REPTY)

HPPF_DEFINE_TYPE_ALIAS_AS(datetime_t, datetime, int64_t);

#if !defined(BUILD_OSX) or defined(CLOCK_REALTIME)
inline datetime_t now() {
  timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) == 0) {
    return datetime_t((ts.tv_sec * 1000000000L + ts.tv_nsec) / 1000L);
  } else {
    return datetime_t(0);
  }
}
#else
inline datetime_t now() {
  struct timeval t;
  if (gettimeofday(&t, 0) == 0) {
    return datetime_t((t.tv_sec * 1000000L) + t.tv_usec);
  } else {
    return datetime_t(0);
  }
}
#endif

HPPF_DEFINE_TYPE_ALIAS_AS(timespan_t, timespan, int64_t);

inline timespan_t lookupLocalTZ() {
  time_t t = time(nullptr);
  struct tm lt;
  memset(&lt, 0, sizeof(lt));
  localtime_r(&t, &lt);
  return timespan_t(lt.tm_gmtoff * 1000L * 1000L);
}
inline timespan_t localTimeZoneOffset() {
  static thread_local timespan_t tz = lookupLocalTZ();
  return tz;
}

// structural recursive types
// these mirror the iso-recursive types sometimes written 'mu X.F(X)' where 'X' represents a point of recursion
// instead of a new type constructor form, users can just use a template argument to represent recursion points
// and the total type is defined as the fixed point of self-application into that argument
//
// ultimately recursive types must be represented in memory with pointers, otherwise they would require an
// infinite amount of storage space (realizing all possible recursive substitutions)
template <typename T>
struct RolledRec {
  typedef T type;
  typedef std::shared_ptr<T> Ptr;
  Ptr rolled;
  template <typename X>
  static RolledRec<T> roll(const X& x) { return T::roll(typename T::Unrolled(x)); }
  inline operator bool() const { return this->rolled.get() != nullptr; }
};
template <template <typename T> class K>
struct fix {
  typedef RolledRec<fix<K>> Rolled;
  typedef typename Rolled::Ptr Ptr;
  typedef K<Rolled> Unrolled;
  Unrolled unrolled;
  static Rolled roll(const Unrolled& x) { return Rolled { .rolled = Ptr(new fix<K> { .unrolled = x }) }; }
};
template <template <typename T> class K>
using recursive = typename fix<K>::Rolled;
template <typename T>
using Unroll = typename T::type::Unrolled;
template <typename T>
inline const Unroll<T>& unroll(const T& t) { return t.rolled->unrolled; }

template <typename T>
  inline std::ostream& operator<<(std::ostream& o, const RolledRec<T>& x) {
    return o << x.rolled->unrolled;
  }
template <typename T>
inline bool operator==(const RolledRec<T>& lhs, const RolledRec<T>& rhs) {
  return lhs.rolled->unrolled == rhs.rolled->unrolled;
}
template <typename T>
inline bool operator!=(const RolledRec<T>& lhs, const RolledRec<T>& rhs) {
  return !(lhs == rhs);
}
template <typename T>
inline bool operator<(const RolledRec<T>& lhs, const RolledRec<T>& rhs) {
  return lhs.rolled->unrolled < rhs.rolled->unrolled;
}

template <typename T>
struct GenericHash<RolledRec<T>> {
  size_t operator()(const RolledRec<T>& x) const noexcept {
    GenericHash<Unroll<RolledRec<T>>> xh;
    return xh(unroll(x));
  }
};

// useful when deconstructing recursive types,
// (an induction principle for recursive types -- avoid infinite loops),
//
// given an assumption for the recursive case,
// and a function to compute the top-level recursive case
//
// produce a total value
template <typename T, typename F>
inline T withRecursive(const T& base, const F& fn) {
  static thread_local const T* nested = nullptr;
  if (nested != nullptr) {
    return *nested;
  }
  try {
    nested = &base;
    auto result = fn();
    nested = nullptr;
    return result;
  } catch (...) {
    nested = nullptr;
    throw;
  }
}

// type-level rewriting
// TODO: cover more type families?  these are at least the typical ones
template <typename From, typename To, typename Def, typename P = void>
struct rewrite { typedef Def type; };
template <typename From, typename To>
struct rewrite<From, To, From> { typedef To type; };

template <typename From, typename To, typename T>
struct rewrite<From, To, std::vector<T>> {
  typedef typename rewrite<From, To, T>::type sT;
  typedef std::vector<sT> type;
};
template <typename From, typename To, typename U, typename V>
struct rewrite<From, To, std::pair<U, V>> {
  typedef typename rewrite<From, To, U>::type sU;
  typedef typename rewrite<From, To, V>::type sV;
  typedef std::pair<sU, sV> type;
};
template <typename From, typename To, typename K, typename V>
struct rewrite<From, To, std::map<K, V>> {
  typedef typename rewrite<From, To, K>::type sK;
  typedef typename rewrite<From, To, V>::type sV;
  typedef std::map<sK, sV> type;
};

template <typename From, typename To>
struct rewriteMap {
  template <typename T>
  struct F {
    typedef typename rewrite<From, To, T>::type type;
  };
};
template <typename From, typename To, typename ... Ts>
struct rewrite<From, To, tuple<Ts...>> {
  typedef typename fmap<rewriteMap<From, To>::template F, tuple<Ts...>>::type type;
};

template <typename T>
struct rewriteToVariant { };
template <typename ... Ts>
struct rewriteToVariant<tuple<Ts...>> { typedef variant<Ts...> type; };
template <typename From, typename To, typename ... Ts>
struct rewrite<From, To, variant<Ts...>> {
  typedef typename rewriteToVariant<typename rewrite<From, To, tuple<Ts...>>::type>::type type;
};

template <typename From, typename To, typename Names, typename T>
struct rewrite<From, To, withNames<Names, T>> {
  typedef withNames<Names, typename rewrite<From, To, T>::type> type;
};

template <typename From, typename To, typename Def>
struct rewrite<From, To, Def, typename tbool<Def::is_hmeta_struct && std::is_same<typename Def::as_tuple_type, typename rewrite<From, To, typename Def::as_tuple_type>::type>::value>::type> {
  typedef Def type;
};
template <typename From, typename To, typename Def>
struct rewrite<From, To, Def, typename tbool<Def::is_hmeta_struct && !std::is_same<typename Def::as_tuple_type, typename rewrite<From, To, typename Def::as_tuple_type>::type>::value>::type> {
  typedef withNames<typename Def::field_name_list, typename rewrite<From, To, typename Def::as_tuple_type>::type> type;
};

template <typename From, typename To, typename Def>
struct rewrite<From, To, Def, typename tbool<Def::is_hmeta_variant && std::is_same<typename Def::as_variant_type, typename rewrite<From, To, typename Def::as_variant_type>::type>::value>::type> {
  typedef Def type;
};
template <typename From, typename To, typename Def>
struct rewrite<From, To, Def, typename tbool<Def::is_hmeta_variant && !std::is_same<typename Def::as_variant_type, typename rewrite<From, To, typename Def::as_variant_type>::type>::value>::type> {
  typedef withNames<typename Def::ctor_name_list, typename rewrite<From, To, typename Def::as_variant_type>::type> type;
};

// rewriting in recursive types, trickier than you might expect!
//
// the idea here is to:
//   * unroll the recursive type,
//   * replace recursion points with a "plug" (to avoid recursive rewriting)
//   * rewrite everything else, leaving the plug(s) as-is
//   * replace the plugs with a free type parameter in the rewritten unrolled type
//   * retie the recursive knot with the rewritten unrolled type with a type parameter
struct recursive_plug { };
template <typename T, typename P = void>
struct plugToTypeConstructor {
  template <typename X>
  using type = T;
};
template <>
struct plugToTypeConstructor<recursive_plug> {
  template <typename X>
  using type = X;
};
template <typename T>
struct plugToTypeConstructor<std::vector<T>> {
  template <typename X>
  using type = std::vector<typename plugToTypeConstructor<T>::template type<X>>;
};
template <typename U, typename V>
struct plugToTypeConstructor<std::pair<U, V>> {
  template <typename X>
  using type = std::pair<typename plugToTypeConstructor<U>::template type<X>, typename plugToTypeConstructor<V>::template type<X>>;
};
template <typename K, typename V>
struct plugToTypeConstructor<std::map<K, V>> {
  template <typename X>
  using type = std::map<typename plugToTypeConstructor<K>::template type<X>, typename plugToTypeConstructor<V>::template type<X>>;
};
template <typename X>
struct flipPlug {
  template <typename T>
  struct TC {
    typedef typename plugToTypeConstructor<T>::template type<X> type;
  };
};
template <typename ... Ts>
struct plugToTypeConstructor<variant<Ts...>> {
  template <typename X>
  using type = typename toVariant<typename fmap<flipPlug<X>::template TC, tuple<Ts...>>::type>::type;
};
template <typename ... Ts>
struct plugToTypeConstructor<tuple<Ts...>> {
  template <typename X>
  using type = typename fmap<flipPlug<X>::template TC, tuple<Ts...>>::type;
};
template <typename Names, typename T>
struct plugToTypeConstructor<withNames<Names, T>> {
  template <typename X>
  using type = withNames<Names, typename plugToTypeConstructor<T>::template type<X>>;
};
template <typename T>
struct plugToTypeConstructor<T, typename tbool<T::is_hmeta_variant>::type> {
  template <typename X>
  using type = typename plugToTypeConstructor<withNames<typename T::ctor_name_list, typename T::as_variant_type>>::template type<X>;
};
template <typename T>
struct plugToTypeConstructor<T, typename tbool<T::is_hmeta_struct>::type> {
  template <typename X>
  using type = typename plugToTypeConstructor<withNames<typename T::field_name_list, typename T::as_tuple_type>>::template type<X>;
};
template <typename From, typename To, typename X>
struct rewrite<From, To, RolledRec<X>, typename tbool<!std::is_same<From, RolledRec<X>>::value>::type> {
  typedef typename rewrite<RolledRec<X>, recursive_plug, Unroll<RolledRec<X>>>::type UT;
  typedef typename rewrite<From, To, UT>::type sUT;
  typedef recursive<plugToTypeConstructor<sUT>::template type> type;
};

// infer attributes of function types, normalize function types into std::function
template <typename F, typename P = void>
struct function_traits {
};
template <typename R, typename C, typename ... Args>
struct function_traits<R (C::*)(Args...) const> {
  typedef R result;
  typedef tuple<Args...> argl;
  typedef std::function<R (Args...)> fn;
};
template <typename R, typename C, typename ... Args>
struct function_traits<R (C::*)(Args...)> {
  typedef R result;
  typedef tuple<Args...> argl;
  typedef std::function<R (Args...)> fn;
};
template <typename R, typename ... Args>
struct function_traits<R (*)(Args...)> {
  typedef R result;
  typedef tuple<Args...> argl;
  typedef std::function<R (Args...)> fn;
};
template <typename R, typename ... Args>
struct function_traits<R(Args...)> {
  typedef R result;
  typedef tuple<Args...> argl;
  typedef std::function<R (Args...)> fn;
};

// infer function types specifically for function-like objects (e.g. lambdas)
template <typename F, class = decltype(&F::operator())>
std::true_type has_fncall_defined(const F&);
std::false_type has_fncall_defined(...);
template <typename T>
struct can_call_type {
  static const bool value = decltype(has_fncall_defined(std::declval<T>()))::value;
};
template <typename F>
struct function_traits<F, typename tbool<can_call_type<F>::value>::type> {
  typedef function_traits<decltype(&F::operator())> Rep;
  typedef typename Rep::result result;
  typedef typename Rep::argl   argl;
  typedef typename Rep::fn     fn;
};

// get the i-th argument type out of a function, strip any const& on it if applicable
template <typename F, size_t i>
using ArgValueType = typename std::remove_const<typename std::remove_reference<typename tupType<i, typename function_traits<F>::argl>::type>::type>::type;

// get the return type from a function
template <typename F>
struct ResultT {
  typedef typename function_traits<F>::result type;
};
template <typename F>
using Result = typename ResultT<F>::type;

/***************************************************
 *
 * type description/encoding
 *
 ***************************************************/

namespace ty {

// a type for type descriptions
// this type is itself a recursive sum of product types
// (covering a large space of algebraic types)
template <typename T>
using VariantCtorDef = tuple<std::string, uint32_t, T>;
template <typename T>
using VariantCtorDefs = std::vector<VariantCtorDef<T>>;
template <typename T>
using StructFieldDef = tuple<std::string, int, T>;
template <typename T>
using StructFieldDefs = std::vector<StructFieldDef<T>>;

HPPF_DEFINE_STRUCT(
  NatDef,
  (size_t, x)
);
template <typename T>
HPPF_DEFINE_STRUCT(
  PrimDef,
  (std::string, n),  // the name of this type (assumed primitive)
  (optional<T>, rep) // "hidden" representation (may be null if actually primitive)
);
HPPF_DEFINE_STRUCT(
  VarDef,
  (std::string, n)
);
template <typename T>
HPPF_DEFINE_STRUCT(
  FArrDef,
  (T, t),  // the array element type
  (T, len) // the array length
);
template <typename T>
HPPF_DEFINE_STRUCT(
  ArrDef,
  (T, t) // the array element type
);
template <typename T>
HPPF_DEFINE_STRUCT(
  VariantDef,
  (VariantCtorDefs<T>, ctors)
);
template <typename T>
HPPF_DEFINE_STRUCT(
  StructDef,
  (StructFieldDefs<T>, fields)
);
template <typename T>
HPPF_DEFINE_STRUCT(
  RecursiveDef,
  (std::string, x),
  (T,           t)
);
template <typename T>
HPPF_DEFINE_STRUCT(
  FnDef,
  (std::vector<std::string>, args), // names for expected arguments
  (T,                        t)     // function body, assuming input arguments
);
template <typename T>
HPPF_DEFINE_STRUCT(
  AppDef,
  (T,              f),   // the type-level function being applied
  (std::vector<T>, args)
);

// the recursive type for type descriptions
template <typename T>
HPPF_DEFINE_VARIANT(
  RecType,
  (nat,       NatDef),
  (prim,      PrimDef<T>),
  (var,       VarDef),
  (farr,      FArrDef<T>),
  (arr,       ArrDef<T>),
  (variant,   VariantDef<T>),
  (record,    StructDef<T>),
  (recursive, RecursiveDef<T>),
  (fn,        FnDef<T>),
  (app,       AppDef<T>)
);

// tie the recursive type together
// (and add some shorthand to avoid some template noise)
using desc      = recursive<RecType>;
using descs     = std::vector<desc>;
using Nat       = NatDef;
using Prim      = PrimDef<desc>;
using Var       = VarDef;
using FArr      = FArrDef<desc>;
using Arr       = ArrDef<desc>;
using Variant   = VariantDef<desc>;
using Struct    = StructDef<desc>;
using Recursive = RecursiveDef<desc>;
using Fn        = FnDef<desc>;
using App       = AppDef<desc>;

using VariantCtor  = VariantCtorDef<desc>;
using VariantCtors = VariantCtorDefs<desc>;
using StructField  = StructFieldDef<desc>;
using StructFields = StructFieldDefs<desc>;

// simple shorthand for type description constructors
inline desc nat(size_t x) { return desc::roll(Unroll<desc>::nat({.x=x})); }

inline desc prim(const std::string& n)                            { return desc::roll(Unroll<desc>::prim({.n=n, .rep=unit{}})); }
inline desc prim(const std::string& n, const desc& rep)           { return desc::roll(Unroll<desc>::prim({.n=n, .rep=rep})); }
inline desc prim(const std::string& n, const optional<desc>& rep) { return desc::roll(Unroll<desc>::prim({.n=n, .rep=rep})); }

inline bool isPrim(const desc& t, const std::string& n) {
  if (const auto* p = unroll(t).prim()) {
    return p->n == n;
  }
  return false;
}
inline bool isPrim(const optional<desc>& mt, const std::string& n) {
  if (const auto* t = some(mt)) {
    return isPrim(*t, n);
  }
  return false;
}

inline desc var(const std::string& n) { return desc::roll(Unroll<desc>::var({.n=n})); }

inline desc array(const desc& t, const desc& len) { return desc::roll(Unroll<desc>::farr({.t=t, .len=len})); }

inline desc array(const desc& t) { return desc::roll(Unroll<desc>::arr({.t=t})); }

template <typename T>
inline desc variant(const VariantCtorDefs<T>& ctors) { return desc::roll(Unroll<desc>::variant({.ctors=ctors})); }

inline desc either(const descs& tys) {
  VariantCtors cs;
  for (size_t i = 0; i < tys.size(); ++i) {
    cs.push_back({ ".f" + string::from(i), uint32_t(i), tys[i] });
  }
  return variant(cs);
}

template <typename ... Tys>
  struct sumAcc { static void acc(std::vector<desc>*, const Tys& ...) { } };
template <typename ... Tys>
  struct sumAcc<desc, Tys...> { static void acc(std::vector<desc>* ts, const desc& t, const Tys& ... tt) { ts->push_back(t); sumAcc<Tys...>::acc(ts, tt...); } };

template <typename ... Tys>
  inline desc sum(const Tys&... tt) {
    std::vector<desc> ts;
    sumAcc<Tys...>::acc(&ts, tt...);

    VariantCtors cs;
    for (size_t i = 0; i < ts.size(); ++i) {
      std::ostringstream n;
      n << ".f" << i;
      cs.push_back(VariantCtor(n.str(), static_cast<uint32_t>(i), ts[i]));
    }
    return variant(cs);
  }

template <typename T>
inline desc record(const StructFieldDefs<T>& fields) { return fields.size() == 0 ? prim("unit") : desc::roll(Unroll<desc>::record({.fields=fields})); }

inline desc tuple(const descs& tys) {
  StructFields fs;
  for (size_t i = 0; i < tys.size(); ++i) {
    fs.push_back({ ".f" + string::from(i), -1, tys[i] });
  }
  return record(fs);
}

template <typename ... Tys>
  struct tupAcc { static void acc(StructFields*) { } };
template <typename ... Tys>
  struct tupAcc<int, desc, Tys...> {
    static void acc(StructFields* fs, const int& o, const desc& t, const Tys& ... tt) {
      fs->push_back({"", o, t}); tupAcc<Tys...>::acc(fs, tt...);
    }
  };
template <typename ... Tys>
  struct tupAcc<size_t, desc, Tys...> {
    static void acc(StructFields* fs, const size_t& o, const desc& t, const Tys& ... tt) {
      fs->push_back({"", int(o), t}); tupAcc<Tys...>::acc(fs, tt...);
    }
  };
template <typename ... Tys>
  desc tup(const Tys& ... tt) {
    StructFields fs;
    tupAcc<Tys...>::acc(&fs, tt...);
    for (size_t i = 0; i < fs.size(); ++i) {
      std::ostringstream n;
      n << ".f" << i;
      fs[i].at<0>() = n.str();
    }
    return record(fs);
  }

template <typename ... Tys>
  struct recAcc { static void acc(StructFields*) { } };
template <size_t N, typename ... Tys>
  struct recAcc<char[N], int, desc, Tys...> {
    static void acc(StructFields* fs, const char (&n)[N], const int& o, const desc& t, const Tys& ... tt) {
      fs->push_back({n, o, t}); recAcc<Tys...>::acc(fs, tt...);
    }
  };
template <size_t N, typename ... Tys>
  struct recAcc<char[N], size_t, desc, Tys...> {
    static void acc(StructFields* fs, const char (&n)[N], const size_t& o, const desc& t, const Tys& ... tt) {
      fs->push_back({n, int(o), t}); recAcc<Tys...>::acc(fs, tt...);
    }
  };
template <typename ... Tys>
  desc rec(const Tys& ... tt) {
    StructFields fs;
    recAcc<Tys...>::acc(&fs, tt...);
    return record(fs);
  }

inline desc recursive(const std::string& x, const desc& t) {
  return desc::roll(Unroll<desc>::recursive({.x=x, .t=t}));
}

inline desc fnc(const std::vector<std::string>& args, const desc& t) {
  return desc::roll(Unroll<desc>::fn({.args=args, .t=t}));
}

template <typename ... Tys>
  struct fnAcc { static desc acc(std::vector<std::string>*) { throw std::runtime_error("fn must define arg list and body"); } };
template <>
  struct fnAcc<desc> { static desc acc(std::vector<std::string>* args, const desc& t) { return fnc(*args, t); } };
template <size_t N, typename ... Tys>
  struct fnAcc<char[N], Tys...> { static desc acc(std::vector<std::string>* args, const char (&n)[N], const Tys& ... tt) { args->push_back(n); return fnAcc<Tys...>::acc(args, tt...); } };

template <typename ... Tys>
  inline desc fn(const Tys& ... tt) { std::vector<std::string> args; return fnAcc<Tys...>::acc(&args, tt...); }

inline desc appc(const desc& f, const std::vector<desc>& args) { return desc::roll(Unroll<desc>::app({.f=f, .args=args})); }

template <typename ... Tys>
  struct appAcc { static void acc(std::vector<desc>*) { } };
template <typename ... Tys>
  struct appAcc<desc, Tys...> { static void acc(std::vector<desc>* ds, const desc& d, const Tys& ... tt) { ds->push_back(d); appAcc<Tys...>::acc(ds, tt...); } };

template <typename ... Tys>
  inline desc app(const desc& f, const Tys& ... tt) {
    std::vector<desc> ds;
    appAcc<Tys...>::acc(&ds, tt...);
    return appc(f, ds);
  }

template <typename R>
inline desc enumdef(const desc& rep, const std::vector<std::pair<std::string, R>>& eds) {
  VariantCtors ctors;
  for (const auto& ed : eds) {
    ctors.push_back(VariantCtor(ed.first, ed.second, prim("unit")));
  }
  if (sizeof(R) == sizeof(uint32_t)) {
    return variant(ctors);
  } else {
    return app(prim("penum", fn("r", "v", var("r"))), rep, variant(ctors));
  }
}

inline desc fileRef(const desc& t) {
  return app(prim("fileref", fn("x", prim("long"))), t);
}
inline optional<desc> maybeFileDeref(const desc& t) {
  if (const auto* app = unroll(t).app()) {
    if (const auto* fn = unroll(app->f).prim()) {
      if (app->args.size() > 0) {
        if (fn->n == "fileref") {
          return app->args[0];
        }
      }
    }
  }
  return unit{};
}

inline desc list(const desc& t) {
  // list t = mu x.()+(t*x)
  return recursive("x", sum(prim("unit"), tup(-1, t, -1, var("x"))));
}

template <typename U, typename V>
inline std::function<V(const U&)> constFn(const V& v) { return [v](const U&) { return v; }; }

// remove file ref types (recursively)
// (this reverses the storage view of all types)
inline desc elimFileRefs(const desc& t) {
  return
    unroll(t).caseOf<desc>({
      .nat  = constFn<Nat>(t),
      .prim = constFn<Prim>(t),
      .var  = constFn<Var>(t),
      .farr = [](const FArr& fa) { return array(elimFileRefs(fa.t), elimFileRefs(fa.len)); },
      .arr  = [](const Arr& a) { return array(elimFileRefs(a.t)); },
      .variant =
        [](const Variant& v) {
          VariantCtors ctors;
          for (const auto& ctor : v.ctors) {
            ctors.push_back(VariantCtor(ctor.at<0>(), ctor.at<1>(), elimFileRefs(ctor.at<2>())));
          }
          return variant(ctors);
        },
      .record =
        [](const Struct& s) {
          StructFields fields;
          for (const auto& field : s.fields) {
            fields.push_back(StructField(field.at<0>(), -1, elimFileRefs(field.at<2>())));
          }
          return record(fields);
        },
      .recursive = [](const Recursive& r) { return recursive(r.x, elimFileRefs(r.t)); },
      .fn = [](const Fn& f) { return fnc(f.args, elimFileRefs(f.t)); },

      // the actual content of this function
      // (stripping file refs out type descriptions)
      .app =
        [](const App& app) {
          if (const auto* fn = unroll(app.f).prim()) {
            if (fn->n == "fileref") {
              if (app.args.size() > 0) {
                return elimFileRefs(app.args[0]);
              }
            }
          }
          std::vector<desc> args;
          for (const auto& arg : app.args) {
            args.push_back(elimFileRefs(arg));
          }
          return appc(elimFileRefs(app.f), args);
        }
    });
}

// encode a type description as a byte array (for storage/transmission)
typedef std::vector<uint8_t> bytes;

#define PRIV_HPPF_TYCTOR_PRIM      (static_cast<uint32_t>(0))
#define PRIV_HPPF_TYCTOR_TVAR      (static_cast<uint32_t>(2))
#define PRIV_HPPF_TYCTOR_FIXEDARR  (static_cast<uint32_t>(4))
#define PRIV_HPPF_TYCTOR_ARR       (static_cast<uint32_t>(5))
#define PRIV_HPPF_TYCTOR_VARIANT   (static_cast<uint32_t>(6))
#define PRIV_HPPF_TYCTOR_STRUCT    (static_cast<uint32_t>(7))
#define PRIV_HPPF_TYCTOR_SIZE      (static_cast<uint32_t>(11))
#define PRIV_HPPF_TYCTOR_TAPP      (static_cast<uint32_t>(12))
#define PRIV_HPPF_TYCTOR_RECURSIVE (static_cast<uint32_t>(13))
#define PRIV_HPPF_TYCTOR_TABS      (static_cast<uint32_t>(15))

template <typename T>
  void w(const T& x, bytes* out) {
    out->insert(out->end(), reinterpret_cast<const uint8_t*>(&x), reinterpret_cast<const uint8_t*>(&x) + sizeof(x));
  }
inline void ws(const char* x, bytes* out) {
  size_t n = strlen(x);
  w(n, out);
  out->insert(out->end(), x, x + n);
}
inline void ws(const std::string& x, bytes* out) {
  w(static_cast<size_t>(x.size()), out);
  out->insert(out->end(), x.begin(), x.end());
}
inline void ws(const bytes& x, bytes* out) {
  w(static_cast<size_t>(x.size()), out);
  out->insert(out->end(), x.begin(), x.end());
}

inline void encode(const desc& t, bytes* o) {
  unroll(t).caseOf<void>({
    .nat = [&](const Nat& n) {
      w(PRIV_HPPF_TYCTOR_SIZE, o);
      w(n.x, o);
    },
    .prim = [&](const Prim& p) {
      w(PRIV_HPPF_TYCTOR_PRIM, o);
      ws(p.n, o);
      if (const auto* rep = some(p.rep)) {
        w(true, o);
        encode(*rep, o);
      } else {
        w(false, o);
      }
    },
    .var = [&](const Var& v) {
      w(PRIV_HPPF_TYCTOR_TVAR, o);
      ws(v.n, o);
    },
    .farr = [&](const FArr& fa) {
      w(PRIV_HPPF_TYCTOR_FIXEDARR, o);
      encode(fa.t, o);
      encode(fa.len, o);
    },
    .arr = [&](const Arr& a) {
      w(PRIV_HPPF_TYCTOR_ARR, o);
      encode(a.t, o);
    },
    .variant = [&](const Variant& v) {
      w(PRIV_HPPF_TYCTOR_VARIANT, o);
      w(size_t(v.ctors.size()), o);
      for (const auto& ctor : v.ctors) {
        ws(ctor.at<0>(), o);
        w(uint32_t(ctor.at<1>()), o);
        encode(ctor.at<2>(), o);
      }
    },
    .record = [&](const Struct& s) {
      w(PRIV_HPPF_TYCTOR_STRUCT, o);
      w(size_t(s.fields.size()), o);
      for (const auto& field : s.fields) {
        ws(field.at<0>(), o);
        w(static_cast<uint32_t>(field.at<1>()), o);
        encode(field.at<2>(), o);
      }
    },
    .recursive = [&](const Recursive& r) {
      w(PRIV_HPPF_TYCTOR_RECURSIVE, o);
      ws(r.x, o);
      encode(r.t, o);
    },
    .fn = [&](const Fn& f) {
      w(PRIV_HPPF_TYCTOR_TABS, o);
      w(size_t(f.args.size()), o);
      for (const auto& arg : f.args) {
        ws(arg, o);
      }
      encode(f.t, o);
    },
    .app = [&](const App& app) {
      w(PRIV_HPPF_TYCTOR_TAPP, o);
      encode(app.f, o);
      w(size_t(app.args.size()), o);
      for (const auto& arg : app.args) {
        encode(arg, o);
      }
    }
  });
}

inline bytes encoding(const desc& t) {
  bytes r;
  encode(t, &r);
  return r;
}

inline bool operator==(const desc& t0, const desc& t1) {
  return encoding(t0) == encoding(t1);
}
inline bool operator!=(const desc& t0, const desc& t1) {
  return !(t0 == t1);
}

// decode a type description from a byte array generated by 'encode'
template <typename T>
  T readValue(const bytes& bs, size_t* i) {
    assert((bs.size() >= (*i + sizeof(T))) && "Invalid type encoding, expected data not available");
    T result;
    memcpy(&result, reinterpret_cast<const T*>(&bs[*i]), sizeof(T));
    *i += sizeof(T);
    return result;
  }
inline std::string readString(const bytes& bs, size_t* i) {
  auto sz = readValue<size_t>(bs, i);
  assert((bs.size() >= (*i + sz)) && "Invalid type encoding, expected string data not available");
  const char* s = reinterpret_cast<const char*>(&bs[*i]);
  std::string result(s, s + sz);
  *i += sz;
  return result;
}

inline desc decodeFrom(const bytes& bs, size_t* i) {
  switch (readValue<uint32_t>(bs, i)) {
  case PRIV_HPPF_TYCTOR_PRIM: {
      std::string n = readString(bs, i);
      if (readValue<bool>(bs, i)) {
        return prim(n, decodeFrom(bs, i));
      } else {
        return prim(n);
      }
    }

  case PRIV_HPPF_TYCTOR_TVAR: {
      std::string n = readString(bs, i);
      return var(n);
    }

  case PRIV_HPPF_TYCTOR_FIXEDARR: {
      desc t = decodeFrom(bs, i);
      desc n = decodeFrom(bs, i);
      return array(t, n);
    }

  case PRIV_HPPF_TYCTOR_ARR: {
      return array(decodeFrom(bs, i));
    }

  case PRIV_HPPF_TYCTOR_VARIANT: {
      size_t n = readValue<size_t>(bs, i);
      VariantCtors cs;
      for (size_t k = 0; k < n; ++k) {
        std::string n  = readString(bs, i);
        int         id = readValue<int>(bs, i);
        desc        t  = decodeFrom(bs, i);

        cs.push_back(VariantCtor(n, id, t));
      }
      return variant(cs);
    }

  case PRIV_HPPF_TYCTOR_STRUCT: {
      size_t count = readValue<size_t>(bs, i);
      StructFields fs;
      for (size_t k = 0; k < count; ++k) {
        std::string n = readString(bs, i);
        int         o = readValue<int>(bs, i);
        desc        t = decodeFrom(bs, i);

        fs.push_back(StructField(n, o, t));
      }
      return record(fs);
    }

  case PRIV_HPPF_TYCTOR_SIZE: {
      return nat(readValue<size_t>(bs, i));
    }

  case PRIV_HPPF_TYCTOR_TAPP: {
      desc   f = decodeFrom(bs, i);
      size_t n = readValue<size_t>(bs, i);

      std::vector<desc> args;
      for (size_t k = 0; k < n; ++k) {
        args.push_back(decodeFrom(bs, i));
      }

      return appc(f, args);
    }

  case PRIV_HPPF_TYCTOR_RECURSIVE: {
      std::string n = readString(bs, i);
      desc        t = decodeFrom(bs, i);

      return recursive(n, t);
    }

  case PRIV_HPPF_TYCTOR_TABS: {
      size_t   n = readValue<size_t>(bs, i);
      std::vector<std::string> args;
      for (size_t k = 0; k < n; ++k) {
        args.push_back(readString(bs, i));
      }
      desc t = decodeFrom(bs, i);

      return fnc(args, t);
    }

  default:
    assert(false && "Invalid type description, internal error");
    throw std::runtime_error("Invalid type description");
  }
}

inline desc decode(const bytes& bs) {
  size_t i = 0;
  return decodeFrom(bs, &i);
}

inline void print(std::ostream& o, const bytes& bs) {
  o << "0x";
  for (auto b : bs) {
    static const char nybs[] = "0123456789abcdef";
    o << nybs[(b>>4)%16];
    o << nybs[(b&0x0f)%16];
  }
}

inline bool isUnit(const desc& t) {
  if (const auto* pn = unroll(t).prim()) {
    return pn->n == "unit";
  } else if (const auto* s = unroll(t).record()) {
    return s->fields.empty();
  }
  return false;
}
inline bool isTuple(const Struct& s) {
  return s.fields.size() > 0 && s.fields[0].at<0>().size() > 0 && s.fields[0].at<0>()[0] == '.';
}
inline bool isTuple(const desc& t) {
  if (const auto* s = unroll(t).record()) {
    return isTuple(*s);
  }
  return false;
}
inline bool isEnum(const Variant& v) {
  for (const auto& c : v.ctors) {
    if (!isUnit(c.at<2>())) {
      return false;
    }
  }
  return true;
}
inline bool isEnum(const desc& t) {
  if (const auto* v = unroll(t).variant()) {
    return isEnum(*v);
  }
  return false;
}

// describe a type description for human consumption
inline void describe(const desc& t, std::ostream& o) {
  unroll(t).caseOf<void>({
    .nat = [&](const Nat& n) {
      o << n.x;
    },
    .prim = [&](const Prim& p) {
      o << (p.n != "unit" ? p.n : "()");
    },
    .var = [&](const Var& v) {
      o << v.n;
    },
    .farr = [&](const FArr& fa) {
      o << "[:";
      describe(fa.t, o);
      o << "|";
      describe(fa.len, o); o << ":]";
    },
    .arr = [&](const Arr& a) {
      o << "[";
      describe(a.t, o);
      o << "]";
    },
    .variant = [&](const Variant& v) {
      if (v.ctors.size() == 0) {
        o << "void";
      } else {
        // show as sum or variant
        if (v.ctors[0].at<0>().substr(0,1) == ".") {
          o << "(";
          describe(v.ctors[0].at<2>(), o);
          for (size_t i = 1; i < v.ctors.size(); ++i) {
            o << "+";
            describe(v.ctors[i].at<2>(), o);
          }
          o << ")";
        } else {
          o << "|" << v.ctors[0].at<0>();
          if (!isUnit(v.ctors[0].at<2>())) {
            o << ":";
            describe(v.ctors[0].at<2>(), o);
          }
          for (size_t i = 1; i < v.ctors.size(); ++i) {
            o << ", " << v.ctors[i].at<0>();
            if (!isUnit(v.ctors[i].at<2>())) {
              o << ":";
              describe(v.ctors[i].at<2>(), o);
            }
          }
          o << "|";
        }
      }
    },
    .record = [&](const Struct& s) {
      if (s.fields.size() == 0) {
        o << "()";
      } else {
        // show as product or record
        if (s.fields[0].at<0>().substr(0,1) == ".") {
          o << "(";
          describe(s.fields[0].at<2>(), o);
          for (size_t i = 1; i < s.fields.size(); ++i) {
            o << "*";
            describe(s.fields[i].at<2>(), o);
          }
          o << ")";
        } else {
          o << "{" << s.fields[0].at<0>() << ":";
          describe(s.fields[0].at<2>(), o);
          for (size_t i = 1; i < s.fields.size(); ++i) {
            o << ", " << s.fields[i].at<0>() << ":";
            describe(s.fields[i].at<2>(), o);
          }
          o << "}";
        }
      }
    },
    .recursive = [&](const Recursive& r) {
      o << "^" << r.x << ".";
      describe(r.t, o);
    },
    .fn = [&](const Fn& f) {
      o << "\\";
      if (f.args.size() > 0) {
        o << f.args[0];
        for (size_t i = 1; i < f.args.size(); ++i) {
          o << " " << f.args[i];
        }
        o << ".";
        describe(f.t, o);
      }
    },
    .app = [&](const App& app) {
      // show file ref types in a special way
      if (const auto* fn = unroll(app.f).prim()) {
        if (fn->n == "fileref" && app.args.size() > 0) {
          describe(app.args[0], o);
          o << "@?";
          return;
        }
      }

      describe(app.f, o);
      o << "(";
      if (app.args.size() > 0) {
        describe(app.args[0], o);
        for (size_t i = 1; i < app.args.size(); ++i) {
          o << ", ";
          describe(app.args[i], o);
        }
      }
      o << ")";
    }
  });
}

inline std::string show(const desc& t) {
  std::ostringstream ss;
  describe(t, ss);
  return ss.str();
}

// type description equivalence modulo offset in record types (allows undefined offsets)
inline bool equivModOffset(const desc& t0, const desc& t1) {
  return
    unroll(t0).caseOf<bool>({
      .nat = [&](const Nat& lhs) {
        if (const auto* rhs = unroll(t1).nat()) {
          return lhs.x == rhs->x;
        }
        return false;
      },
      .prim = [&](const Prim& lhs) {
        if (const auto* rhs = unroll(t1).prim()) {
          return lhs.n == rhs->n;
        }
        return false;
      },
      .var = [&](const Var& lhs) {
        if (const auto* rhs = unroll(t1).var()) {
          return lhs.n == rhs->n;
        }
        return false;
      },
      .farr = [&](const FArr& lhs) {
        if (const auto* rhs = unroll(t1).farr()) {
          return equivModOffset(lhs.t, rhs->t) && equivModOffset(lhs.len, rhs->len);
        }
        return false;
      },
      .arr = [&](const Arr& lhs) {
        if (const auto* rhs = unroll(t1).arr()) {
          return equivModOffset(lhs.t, rhs->t);
        }
        return false;
      },
      .variant = [&](const Variant& lhs) {
        if (const auto* rhs = unroll(t1).variant()) {
          if (lhs.ctors.size() == rhs->ctors.size()) {
            for (size_t i = 0; i < lhs.ctors.size(); ++i) {
              if (
                lhs.ctors[i].at<0>() != rhs->ctors[i].at<0>() ||
                lhs.ctors[i].at<1>() != rhs->ctors[i].at<1>() ||
                !equivModOffset(lhs.ctors[i].at<2>(), rhs->ctors[i].at<2>())
              ) {
                return false;
              }
            }
            return true;
          }
        }
        return false;
      },
      .record = [&](const Struct& lhs) {
        if (const auto* rhs = unroll(t1).record()) {
          if (lhs.fields.size() == rhs->fields.size()) {
            for (size_t i = 0; i < lhs.fields.size(); ++i) {
              if (
                lhs.fields[i].at<0>() != rhs->fields[i].at<0>() ||
                !equivModOffset(lhs.fields[i].at<2>(), rhs->fields[i].at<2>())
              ) {
                return false;
              }
            }
            return true;
          }
        }
        return false;
      },
      .recursive = [&](const Recursive& lhs) {
        if (const auto* rhs = unroll(t1).recursive()) {
          return equivModOffset(lhs.t, rhs->t);
        }
        return false;
      },
      .fn = [&](const Fn& lhs) {
        if (const auto* rhs = unroll(t1).fn()) {
          return lhs.args.size() == rhs->args.size() && equivModOffset(lhs.t, rhs->t);
        }
        return false;
      },
      .app = [&](const App& lhs) {
        if (const auto* rhs = unroll(t1).app()) {
          if (lhs.args.size() == rhs->args.size()) {
            for (size_t i = 0; i < lhs.args.size(); ++i) {
              if (!equivModOffset(lhs.args[i], rhs->args[i])) {
                return false;
              }
            }
            return equivModOffset(lhs.f, rhs->f);
          }
        }
        return false;
      }
    });
}

typedef std::map<std::string, desc> TyVarEnv;

// substitute types for variables
// TODO: avoid capture
inline desc substitute(const TyVarEnv& tenv, const desc& t) {
  return
    unroll(t).caseOf<desc>({
      .nat = constFn<Nat>(t),
      .prim = [&](const Prim& x) {
        if (const auto* rep = some(x.rep)) {
          return prim(x.n, substitute(tenv, *rep));
        }
        return t;
      },
      .var = [&](const Var& x) {
        auto k = tenv.find(x.n);
        return (k == tenv.end()) ? t : k->second;
      },
      .farr = [&](const FArr& x) { return array(substitute(tenv, x.t), substitute(tenv, x.len)); },
      .arr = [&](const Arr& x) { return array(substitute(tenv, x.t)); },
      .variant =
        [&](const Variant& x) {
          VariantCtors ctors;
          for (const auto& ctor : x.ctors) {
            ctors.push_back(VariantCtor(ctor.at<0>(), ctor.at<1>(), substitute(tenv, ctor.at<2>())));
          }
          return variant(ctors);
        },
      .record =
        [&](const Struct& x) {
          StructFields fields;
          for (const auto& field : x.fields) {
            fields.push_back(StructField(field.at<0>(), -1, substitute(tenv, field.at<2>())));
          }
          return record(fields);
        },
      .recursive = [&](const Recursive& x) {
        if (tenv.count(x.x)) {
          TyVarEnv ntenv = tenv;
          ntenv.erase(ntenv.find(x.x));
          return recursive(x.x, substitute(ntenv, x.t));
        } else {
          return recursive(x.x, substitute(tenv, x.t));
        }
      },
      .fn = [&](const Fn& x) {
        TyVarEnv ntenv = tenv;
        for (const auto& arg : x.args) {
          auto k = ntenv.find(arg);
          if (k != ntenv.end()) {
            ntenv.erase(k);
          }
        }
        return fnc(x.args, substitute(ntenv, x.t));
      },
      .app = [&](const App& x) {
        std::vector<desc> args;
        for (const auto& arg : x.args) {
          args.push_back(substitute(tenv, arg));
        }
        return appc(substitute(tenv, x.f), args);
      }
    });
}

// apply a type -> type function to some arguments
inline desc apply(const desc& fn, const std::vector<desc>& args) {
  auto f = fn;
  // expect to be able to get a function out of 'fn'
  while (unroll(f).fn() == nullptr) {
    if (const auto* p = unroll(f).prim()) {
      if (const auto* rep = some(p->rep)) {
        f = *rep;
        continue;
      }
    }
    throw std::runtime_error("Not a type function: " + show(fn));
  }
  const auto* fd = unroll(f).fn();
  if (fd->args.size() != args.size()) {
    throw std::runtime_error("Arity mismatch in function application");
  }
  TyVarEnv tvenv;
  for (size_t i = 0; i < fd->args.size(); ++i) {
    tvenv[fd->args[i]] = args[i];
  }
  return substitute(tvenv, fd->t);
}
inline desc apply(const desc& app) {
  if (const auto* p = unroll(app).app()) {
    return apply(p->f, p->args);
  }
  throw std::runtime_error("Expected type function application: " + show(app));
}

// reduce a type description to its normalized representation
inline desc toRepresentation(const desc& t, const TyVarEnv& tenv) {
  return
    unroll(t).caseOf<desc>({
      .nat = constFn<Nat>(t),
      .prim = [&](const Prim& x) {
        if (const auto* rep = some(x.rep)) {
          return toRepresentation(*rep, tenv);
        }
        return t;
      },
      .var = [&](const Var& x) {
        auto k = tenv.find(x.n);
        return (k == tenv.end()) ? t : k->second;
      },
      .farr = [&](const FArr& x) {
        return array(toRepresentation(x.t, tenv), toRepresentation(x.len, tenv));
      },
      .arr = [&](const Arr& x) {
        return ty::array(toRepresentation(x.t, tenv));
      },
      .variant = [&](const Variant& x) {
        VariantCtors ctors;
        for (const auto& ctor : x.ctors) {
          ctors.push_back(VariantCtor(ctor.at<0>(), ctor.at<1>(), toRepresentation(ctor.at<2>(), tenv)));
        }
        return variant(ctors);
      },
      .record = [&](const Struct& x) {
        StructFields fs;
        for (const auto& field : x.fields) {
          fs.push_back(StructField(field.at<0>(), -1, toRepresentation(field.at<2>(), tenv)));
        }
        return record(fs);
      },
      .recursive = [&](const Recursive& x) {
        return recursive(x.x, toRepresentation(x.t, tenv));
      },
      .fn = constFn<Fn>(t),
      .app = [&](const App& x) {
        auto f = toRepresentation(x.f, tenv);
        const auto* apf = unroll(f).fn();
        if (apf == nullptr) {
          // can't do anything with this function
          // no further reduction is possible
          return t;
        } else if (apf->args.size() != x.args.size()) {
          throw std::runtime_error("Can't normalize type with arity mismatch in application: " + show(t));
        }

        TyVarEnv ntenv = tenv;
        for (size_t i = 0; i < apf->args.size(); ++i) {
          ntenv[apf->args[i]] = toRepresentation(x.args[i], tenv);
        }
        return toRepresentation(apf->t, ntenv);
      }
    });
}

// standard memory layout
inline size_t alignOf(const desc& t) {
  return
    unroll(t).caseOf<size_t>({
      .nat = [&](const Nat&) -> size_t {
        throw std::runtime_error("Can't determine aligment of type-level number: " + show(t));
      },
      .prim = [&](const Prim& x) {
        if (const auto* rep = some(x.rep)) {
          return alignOf(*rep);
        }
        static std::map<std::string, size_t> paligns = {
          { "unit",   1 }, { "bool", 1 }, { "char", 1 }, { "byte",  1 },
          { "short",  2 }, { "int",  4 }, { "long", 8 }, { "float", 4 },
          { "double", 8 }
        };
        auto pa = paligns.find(x.n);
        if (pa != paligns.end()) {
          return pa->second;
        } else {
          throw std::runtime_error("Can't determine alignment of unknown primitive type: " + x.n);
        }
      },
      .var = [&](const Var&) {
        return alignof(void*);
      },
      .farr = [&](const FArr& x) {
        return alignOf(x.t);
      },
      .arr = [&](const Arr&) -> size_t {
        throw std::runtime_error("Can't determine alignment of variable-length array type: " + show(t));
      },
      .variant = [&](const Variant& x) {
        if (x.ctors.size() == 0) {
          return size_t(1);
        }
        size_t maxAlign = 4; // the variant tag is an int
        for (const auto& ctor : x.ctors) {
          maxAlign = std::max<size_t>(maxAlign, alignOf(ctor.at<2>()));
        }
        return maxAlign;
      },
      .record = [&](const Struct& x) {
        size_t maxAlign = 1;
        for (const auto& field : x.fields) {
          maxAlign = std::max<size_t>(maxAlign, alignOf(field.at<2>()));
        }
        return maxAlign;
      },
      .recursive = [&](const Recursive&) {
        return alignof(void*);
      },
      .fn = [&](const Fn&) -> size_t {
        throw std::runtime_error("Can't decide alignment of type abstraction: " + show(t));
      },
      .app = [&](const App&) {
        return alignOf(toRepresentation(t, TyVarEnv()));
      }
    });
}

inline size_t sizeOf(const desc& t) {
  return
    unroll(t).caseOf<size_t>({
      .nat = [&](const Nat&) -> size_t {
        throw std::runtime_error("Can't determine size of type-level number: " + show(t));
      },
      .prim = [&](const Prim& x) {
        if (const auto* rep = some(x.rep)) {
          return sizeOf(*rep);
        } else if (x.n == "unit") {
          return size_t(0);
        } else {
          return alignOf(t);
        }
      },
      .var = [&](const Var&) {
        return sizeof(void*);
      },
      .farr = [&](const FArr& x) {
        if (const auto* len = unroll(x.len).nat()) {
          return sizeOf(x.t) * len->x;
        } else {
          throw std::runtime_error("Can't determine size of fixed array: " + show(t));
        }
      },
      .arr = [&](const Arr&) -> size_t {
        throw std::runtime_error("Can't determine size of variable-length array type: " + show(t));
      },
      .variant = [&](const Variant& x) {
        if (x.ctors.size() == 0) {
          return size_t(4);
        } else {
          size_t maxSize  = 0;
          size_t maxAlign = 4; // the variant tag is an int
          for (const auto& ctor : x.ctors) {
            maxSize  = std::max<size_t>(maxSize, sizeOf(ctor.at<2>()));
            maxAlign = std::max<size_t>(maxAlign, alignOf(ctor.at<2>()));
          }
          return alignTo(alignTo(4, maxAlign) + maxSize, maxAlign);
        }
      },
      .record = [&](const Struct& x) {
        if (x.fields.size() == 0) {
          return size_t(0);
        } else {
          size_t maxAlign = 1;
          size_t offset   = 0;
          for (const auto& field : x.fields) {
            maxAlign = std::max<size_t>(maxAlign, alignOf(field.at<2>()));
            offset   = alignTo(offset, alignOf(field.at<2>())) + sizeOf(field.at<2>());
          }
          return alignTo(offset, maxAlign);
        }
      },
      .recursive = [&](const Recursive&) {
        return sizeof(void*);
      },
      .fn = [&](const Fn&) -> size_t {
        throw std::runtime_error("Can't decide size of type abstraction: " + show(t));
      },
      .app = [&](const App&) {
        return sizeOf(toRepresentation(t, TyVarEnv()));
      }
    });
}

inline desc inferOffsets(const desc& t) {
  return
    unroll(t).caseOf<desc>({
      .nat = constFn<Nat>(t),
      .prim = [&](const Prim& x) {
        if (const auto* rep = some(x.rep)) {
          return prim(x.n, inferOffsets(*rep));
        } else {
          return t;
        }
      },
      .var = constFn<Var>(t),
      .farr = [&](const FArr& x) {
        return array(inferOffsets(x.t), inferOffsets(x.len));
      },
      .arr = [&](const Arr& x) {
        return array(inferOffsets(x.t));
      },
      .variant = [&](const Variant& x) {
        VariantCtors ctors;
        for (const auto& ctor : x.ctors) {
          ctors.push_back(VariantCtor(ctor.at<0>(), ctor.at<1>(), inferOffsets(ctor.at<2>())));
        }
        return variant(ctors);
      },
      .record = [&](const Struct& x) {
        StructFields fs;
        size_t o = 0;
        for (const auto& field : x.fields) {
          auto fty = inferOffsets(field.at<2>());
          o = (field.at<1>() >= 0) ? size_t(field.at<1>()) : alignTo(o, alignOf(fty));
          fs.push_back(StructField(field.at<0>(), int(o), fty));
          o += sizeOf(fty);
        }
        return record(fs);
      },
      .recursive = [&](const Recursive& x) {
        return withRecursive(var(x.x), [&](){ return recursive(x.x, inferOffsets(x.t)); });
      },
      .fn = constFn<Fn>(t),
      .app = [&](const App& x) {
        std::vector<desc> args;
        for (const auto& arg : x.args) {
          args.push_back(inferOffsets(arg));
        }
        return appc(inferOffsets(x.f), args);
      }
    });
}

inline desc tupleDesc(const std::vector<desc>& tys) {
  StructFields fields;
  size_t offset = 0;
  for (size_t i = 0; i < tys.size(); ++i) {
    std::ostringstream fn;
    fn << ".f" << i;
    fields.push_back(StructField(fn.str(), int(align(offset, alignOf(tys[i]))), tys[i]));
    offset += sizeOf(tys[i]);
  }
  return record(fields);
}

// produce a flattened representation of a type,
// practically this means that if the type is a struct, all nested struct fields are pulled up to the top level
// otherwise the result is the type unchanged (with an empty name)
inline desc flatView(const desc& t) {
  if (const auto* a = unroll(t).app()) {
    if (isPrim(a->f, "model_dependent_pair") && a->args.size() == 2) {
      return tup(-1, a->args[0], -1, a->args[1]);
    }
  }
  return t;
}
inline std::vector<std::pair<std::string, desc>> flattenedDef(const desc& t, const std::string& delim = "_") {
  typedef std::pair<std::string, desc> NameTy;
  std::vector<NameTy> result;
  std::stack<NameTy> nts;
  nts.push(NameTy("", t));

  while (!nts.empty()) {
    auto nt = nts.top();
    nts.pop();

    auto fvt = flatView(nt.second);
    if (const auto* structTy = unroll(fvt).record()) {
      // push fields in reverse order to get the right order on stack removal (first to last)
      for (size_t i = structTy->fields.size(); i > 0; --i) {
        std::string name = structTy->fields[i - 1].at<0>();
        ty::desc    fty  = structTy->fields[i - 1].at<2>();

        if (name.size() > 2 && name[0] == '.' && name[1] == 'f') {
          name = name.substr(2);
        }
        if (!nt.first.empty()) {
          name = nt.first + delim + name;
        }
        nts.push(NameTy(name, fty));
      }
    } else {
      // finally at a non-struct type, this must be a single column of the table for this data
      // add the column declaration, the place in the insert statement, and the object to do the insert
      result.push_back(NameTy(nt.first, nt.second));
    }
  }
  return result;
}

// rewrite a type description
inline desc rewrite(const desc& t, const std::function<optional<desc>(const desc&)>& f) {
  auto mt = f(t);
  if (const auto* rt = some(mt)) {
    return *rt;
  } else {
    return unroll(t).caseOf<desc>({
      .nat = constFn<Nat>(t),
      .prim = [&](const Prim& x) {
        if (const auto* rep = some(x.rep)) {
          return prim(x.n, rewrite(*rep, f));
        } else {
          return t;
        }
      },
      .var = constFn<Var>(t),
      .farr = [&](const FArr& x) {
        return array(rewrite(x.t, f), rewrite(x.len, f));
      },
      .arr = [&](const Arr& x) {
        return array(rewrite(x.t, f));
      },
      .variant = [&](const Variant& x) {
        VariantCtors ctors;
        for (const auto& ctor : x.ctors) {
          ctors.push_back(VariantCtor(ctor.at<0>(), ctor.at<1>(), rewrite(ctor.at<2>(), f)));
        }
        return variant(ctors);
      },
      .record = [&](const Struct& x) {
        StructFields fs;
        for (const auto& field : x.fields) {
          fs.push_back(StructField(field.at<0>(), field.at<1>(), rewrite(field.at<2>(), f)));
        }
        return record(fs);
      },
      .recursive = [&](const Recursive& x) {
        return withRecursive(var(x.x), [&](){ return recursive(x.x, rewrite(x.t, f)); });
      },
      .fn = constFn<Fn>(t),
      .app = [&](const App& x) {
        std::vector<desc> args;
        for (const auto& arg : x.args) {
          args.push_back(rewrite(arg, f));
        }
        return appc(rewrite(x.f, f), args);
      }
    });
  }
}

}

// shorthand for pulling together type-level and term-level chrono duration details
struct ChronoDef {
  std::intmax_t numerator;
  std::intmax_t denominator;
};
inline optional<ChronoDef> chronoDef(const ty::desc& t) {
  auto ap = unroll(t).app();
  if (!ap) { return unit{}; }
  if (ap->args.size() != 3) { return unit{}; }
  const auto* n = unroll(ap->args[1]).nat();
  const auto* d = unroll(ap->args[2]).nat();
  if (!n || !d) { return unit{}; }

  return ChronoDef { .numerator=std::intmax_t(n->x), .denominator=std::intmax_t(d->x) };
}
template <typename Rep>
struct ChronoDuration {
  ChronoDef def;
  Rep value;
};
template <typename Rep>
inline double toSeconds(ChronoDuration<Rep> d) {
  return double(d.value) * double(d.def.numerator) / double(d.def.denominator);
}
template <typename Rep>
inline timespan_t toTimespan(ChronoDuration<Rep> d) {
  return timespan_t(timespan_t::type((std::intmax_t(d.value) * d.def.numerator * 1000000L) / d.def.denominator));
}

/***************************************************
 *
 * type-shaped measurement (e.g. carry counts per value in a type)
 *
 ***************************************************/
namespace tym {
  template <typename T>
  struct TypeMeasure {
    virtual ~TypeMeasure() { }
  };
  template <typename T>
  using Ptr = std::shared_ptr<TypeMeasure<T>>;
  template <typename T>
  using Ptrs = std::vector<Ptr<T>>;

  template <typename T>
  struct PrimTM : public TypeMeasure<T> {
    T m;
  };
  template <typename T>
  inline Ptr<T> prim(T x) {
    auto p = new PrimTM<T>();
    p->m = x;
    return Ptr<T>(p);
  }

  template <typename T>
  struct RecordTM : public TypeMeasure<T> {
    typedef std::pair<std::string, Ptr<T>> Field;
    typedef std::vector<Field> Fields;
    Fields m;
  };
  template <typename T>
  inline Ptr<T> rec(const typename RecordTM<T>::Fields& fs) {
    auto p = new RecordTM<T>();
    p->m = fs;
    return Ptr<T>(p);
  }
  template <typename T>
  inline Ptr<T> tup(const std::vector<Ptr<T>>& fs) {
    auto p = new RecordTM<T>();
    for (size_t i = 0; i < fs.size(); ++i) {
      p->m.push_back({".f" + string::from<size_t>(i), fs[i]});
    }
    return Ptr<T>(p);
  }
  template <typename T>
  inline Ptr<T> unit() {
    return Ptr<T>(new RecordTM<T>());
  }

  template <typename T>
  struct VArrayTM : public TypeMeasure<T> {
    typedef std::pair<T, Ptr<T>> SizeAndValues;
    SizeAndValues m;
  };
  template <typename T>
  inline Ptr<T> va(T len, const Ptr<T>& v) {
    auto p = new VArrayTM<T>();
    p->m = typename VArrayTM<T>::SizeAndValues(len, v);
    return Ptr<T>(p);
  }

  template <typename T>
  struct SumTM : public TypeMeasure<T> {
    typedef std::pair<std::string, Ptr<T>> Ctor;
    typedef std::vector<Ctor> Ctors;
    typedef std::pair<T, Ctors> TagAndCtors;
    TagAndCtors m;
  };
  template <typename T>
  inline Ptr<T> sum(T tag, const typename SumTM<T>::Ctors& ctors) {
    auto p = new SumTM<T>();
    p->m = typename SumTM<T>::TagAndCtors(tag, ctors);
    return Ptr<T>(p);
  }

  // structure-preserving mapping
  template <typename To, typename From>
  inline Ptr<To> map(const std::function<To(From)>& fn, const Ptr<From>& m) {
    if (const auto* p = dynamic_cast<const PrimTM<From>*>(m.get())) {
      return prim<To>(fn(p->m));
    } else if (const auto* r = dynamic_cast<const RecordTM<From>*>(m.get())) {
      typename RecordTM<To>::Fields fs;
      for (const auto& f : r->m) {
        fs.push_back(typename RecordTM<To>::Field(f.first, map<To, From>(fn, f.second)));
      }
      return rec<To>(fs);
    } else if (const auto* a = dynamic_cast<const VArrayTM<From>*>(m.get())) {
      return va<To>(fn(a->m.first), map<To, From>(fn, a->m.second));
    } else if (const auto* s = dynamic_cast<const SumTM<From>*>(m.get())) {
      typename SumTM<To>::TagAndCtors cs;
      cs.first = fn(s->m.first);
      for (const auto& c : s->m.second) {
        cs.second.push_back(typename SumTM<To>::Ctor(c.first, map<To, From>(fn, c.second)));
      }
      return sum<To>(cs.first, cs.second);
    } else {
      return unit<To>(); // truncate if unknown constructor only
    }
  }

  // bi-structure-preserving mapping (modulo truncation)
  template <typename To, typename From1, typename From2>
  inline Ptr<To> zipWith(const std::function<To(From1,From2)>& fn, const Ptr<From1>& m1, const Ptr<From2>& m2) {
    if (const auto* p1 = dynamic_cast<const PrimTM<From1>*>(m1.get())) {
      if (const auto* p2 = dynamic_cast<const PrimTM<From2>*>(m2.get())) {
        return prim<To>(fn(p1->m, p2->m));
      }
    } else if (const auto* r1 = dynamic_cast<const RecordTM<From1>*>(m1.get())) {
      if (const auto* r2 = dynamic_cast<const RecordTM<From2>*>(m2.get())) {
        typename RecordTM<To>::Fields fs;
        auto n = std::min<size_t>(r1->m.size(), r2->m.size());
        for (size_t i = 0; i < n; ++i) {
          if (r1->m[i].first == r2->m[i].first) {
            fs.push_back(typename RecordTM<To>::Field(r1->m[i].first, zipWith<To, From1, From2>(fn, r1->m[i].second, r2->m[i].second)));
          }
        }
        return rec<To>(fs);
      }
    } else if (const auto* a1 = dynamic_cast<const VArrayTM<From1>*>(m1.get())) {
      if (const auto* a2 = dynamic_cast<const VArrayTM<From2>*>(m2.get())) {
        return va<To>(fn(a1->m.first, a2->m.first), zipWith(fn, a1->m.second, a2->m.second));
      }
    } else if (const auto* s1 = dynamic_cast<const SumTM<From1>*>(m1.get())) {
      if (const auto* s2 = dynamic_cast<const SumTM<From2>*>(m2.get())) {
        typename SumTM<To>::TagAndCtors cs;
        cs.first = fn(s1->m.first, s2->m.first);
        auto n = std::min<size_t>(s1->m.second.size(), s2->m.second.size());
        for (size_t i = 0; i < n; ++i) {
          if (s1->m.second[i].first == s2->m.second[i].first) {
            cs.second.push_back(typename SumTM<To>::Ctor(s1->m.second[i].first, zipWith<To, From1, From2>(fn, s1->m.second[i].second, s2->m.second[i].second)));
          }
        }
        return sum<To>(cs.first, cs.second);
      }
    }
    // if we got here, there was a truncation
    return unit<To>();
  }

  // destructuring map out
  template <typename S, typename T>
  inline S fold(const std::function<S(S, T)>& fn, S s, const Ptr<T>& m) {
    if (const auto* prim = dynamic_cast<const PrimTM<T>*>(m.get())) {
      return fn(s, prim->m);
    } else if (const auto* rec = dynamic_cast<const RecordTM<T>*>(m.get())) {
      for (const auto& f : rec->m) {
        s = fold<S, T>(fn, s, f.second);
      }
      return s;
    } else if (const auto* va = dynamic_cast<const VArrayTM<T>*>(m.get())) {
      return fold<S, T>(fn, fn(s, va->m.first), va->m.second);
    } else if (const auto* sum = dynamic_cast<const SumTM<T>*>(m.get())) {
      s = fn(s, sum->m.first);
      for (const auto& c : sum->m.second) {
        s = fold<S, T>(fn, s, c.second);
      }
      return s;
    } else {
      return s;
    }
  }

  // make a zero value by structure (replace every value with 0)
  template <typename T>
  inline Ptr<T> zero(const Ptr<T>& x) {
    return map<T>([](T) { return T(); }, x);
  }

  // add two measures by structure (if not the same structure, some information will be truncated)
  template <typename T>
  inline Ptr<T> add(const Ptr<T>& lhs, const Ptr<T>& rhs) {
    return zipWith<T, T, T>([](T x, T y) { return x+y; }, lhs, rhs);
  }
  template <typename T>
  inline Ptr<T> add(const std::vector<Ptr<T>>& xs) {
    if (xs.size() == 0) {
      return unit<T>();
    } else {
      auto s = xs[0];
      for (size_t i = 1; i < xs.size(); ++i) {
        s = add(s, xs[i]);
      }
      return s;
    }
  }

  // assuming an initial total and an operator "+" to combine two totals in a type measure
  // find the total over the whole type measure
  template <typename T>
  inline T total(const Ptr<T>& m, T s = T()) {
    return fold<T, T>([](T x, T y) { return x+y; }, s, m);
  }
}
END_HLOG_NAMESPACE

namespace hobbes = HLOG_NS;

// convenience aliases for users who are only interested in logging
#define HLOG_DEFINE_STRUCT(ARGS...) HPPF_DEFINE_STRUCT(ARGS)
#define HLOG_DEFINE_VARIANT(ARGS...) HPPF_DEFINE_VARIANT(ARGS)
#define HLOG_DEFINE_ENUM(ARGS...) HPPF_DEFINE_ENUM(ARGS)
#define HLOG_DEFINE_ENUM_V(ARGS...) HPPF_DEFINE_ENUM_V(ARGS)
#define HLOG_DEFINE_TYPE_ALIAS(ARGS...) HPPF_DEFINE_TYPE_ALIAS(ARGS)
#define HLOG_DEFINE_TYPE_ALIAS_AS(ARGS...) HPPF_DEFINE_TYPE_ALIAS_AS(ARGS)
#define HLOG_DEFINE_GMODEL_TYPE_ALIAS(ARGS...) HPPF_DEFINE_GMODEL_TYPE_ALIAS(ARGS)

// overloads for using custom types here with std iostreams
namespace std {
  inline std::ostream& operator<<(std::ostream& out, HLOG_NS::datetime_t dt) {
    long x = dt.value;
    int64_t s  = x / 1000000L;
    int64_t us = x % 1000000L;

    static thread_local char buf[32];
    static thread_local int64_t ls = 0;
    if (s != ls) {
      static thread_local struct tm stm;
      strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S.", localtime_r(reinterpret_cast<time_t*>(&s), &stm));
      ls = s;
    }
    return out << buf << std::setfill('0') << std::setw(6) << us;
  }
  inline std::ostream& operator<<(std::ostream& out, HLOG_NS::timespan_t ts) {
    static const long msus  = 1000;
    static const long sus   = msus * 1000;
    static const long mus   = sus * 60;
    static const long hus   = mus * 60;
    static const long dayus = hus * 24;

    uint64_t x  = labs(ts.value);
    uint64_t d  = x / dayus; x -= d*dayus;
    uint64_t h  = x / hus;   x -= h*hus;
    uint64_t m  = x / mus;   x -= m*mus;
    uint64_t s  = x / sus;   x -= s*sus;
    uint64_t ms = x / msus;  x -= ms*msus;
    uint64_t us = x;

    if (ts.value < 0) out << "-";
    if (d > 0) out << d << "d";
    if (h > 0) out << h << "h";
    if (m > 0) out << m << "m";
    if (s > 0 || (d == 0 && h == 0 && m == 0 && ms == 0 && us == 0)) {
      out << s << "s";
    }
    if (ms > 0) out << ms << "ms";
    if (us > 0) out << us << "us";

    return out;
  }
}

#endif

