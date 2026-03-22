CXX      = g++
CXXFLAGS = -O3 -std=c++17 -march=native -D_GNU_SOURCE -Wall -Wextra
LDFLAGS  = -lm -lpthread -static-libgcc -static-libstdc++
SRCS     = picolm.cpp
TARGET   = picolm

PREFIX   ?= /usr/local

# Native build (x86-64 with SSE2/AVX)
native: $(TARGET)

$(TARGET): $(SRCS) model.hpp tensor.hpp quant.hpp tokenizer.hpp sampler.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS) $(LDFLAGS)

# Debug build
debug: CXXFLAGS = -std=c++17 -g -O0 -Wall -Wextra
debug: $(TARGET)

# Release with AVX2
avx2: CXXFLAGS += -mavx2 -mfma
avx2: $(TARGET)

# Install
install: $(TARGET)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/bin/

clean:
	rm -f $(TARGET) $(TARGET).exe *.obj *.o

.PHONY: native debug avx2 install clean
