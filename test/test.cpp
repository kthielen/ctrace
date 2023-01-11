
#include <hlog/trace.h>
#include <hlog/ctrace.h>

using namespace HLOG_NS;

HLOG_DEFINE_STRUCT(
  SymbolDef,
  (uint64_t,    id),
  (std::string, symbol)
);

HLOG_DEFINE_ENUM(Side, (Buy), (Sell));

HLOG_DEFINE_STRUCT(
  Quote,
  (Side,     side),
  (uint64_t, size),
  (uint64_t, price)
);
using SymbolQuote = std::pair<uint64_t, Quote>; // a quote is defined for a specific symbol
using CSymbolQuote = ctrace::model_dependent_pair<uint64_t, Quote>; // when compressing quotes, use per-symbol statistics

template <typename Output, typename QuoteType>
void generateQuotes(const std::string& path) {
  Output file(path);
  auto& symbols = file.template series<SymbolDef>("symbols");
  auto& quotes = file.template series<QuoteType>("quotes");

  // internalize some symbols
  static const std::vector<std::string> symbolNames = { "aapl", "ibm", "msft", "ms", "gs", "tsla" };
  for (size_t i = 0; i < symbolNames.size(); ++i) {
    symbols(datetime_t(0), { .id=i, .symbol=symbolNames[i] });
  }

  // write a bunch of random quotes for those symbols
  for (size_t i = 0; i < 1000000; ++i) {
    auto symbolID = i % symbolNames.size();
    auto side     = rand() % 4 == 0 ? Side::Buy() : Side::Sell();
    auto size     = symbolID * 100 + rand() % 16;
    auto price    = 1000 + symbolID * 31337;

    quotes(now(), { symbolID, { .side=side, .size=size, .price=price } });
  }
}

template <typename Input, typename QuoteType>
void printQuotes(const std::string& path) {
  Input file(path);
  std::vector<std::string> symbolNames;
  file.series("symbols", [&](datetime_t, const SymbolDef& sd) {
    if (symbolNames.size() <= sd.id) {
      symbolNames.resize(sd.id + 1);
    }
    symbolNames[sd.id] = sd.symbol;
  });
  file.series("quotes", [&](datetime_t dt, const QuoteType& qt) {
    std::cout << dt << " (" << symbolNames[qt.first] << "): " << qt.second << std::endl;
  });
  file.run();
}

int main(int argc, char** argv) {
  std::string file = "quotes.log";
  bool compressed = true;
  bool toWrite = true;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--compressed" || arg == "-c") {
      compressed = true;
    } else if (arg == "--uncompressed" || arg == "-u") {
      compressed = false;
    } else if (arg == "--write" || arg == "-w") {
      toWrite = true;
    } else if (arg == "--read" || arg == "-r") {
      toWrite = false;
    } else {
      file = arg;
    }
  }
  if (toWrite) {
    if (compressed) {
      generateQuotes<ctrace::writer, CSymbolQuote>(file);
    } else {
      generateQuotes<trace::writer, SymbolQuote>(file);
    }
  } else {
    if (compressed) {
      printQuotes<ctrace::reader, CSymbolQuote>(file);
    } else {
      printQuotes<trace::reader, SymbolQuote>(file);
    }
  }
  return 0;
}
