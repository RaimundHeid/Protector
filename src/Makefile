MAKE=make

# Default compiler is GCC
COMPILER ?= GCC

# Detect if CC is set to icc
CC_NAME := $(shell $(CC) --version 2>&1 | head -n 1)
ifneq (,$(findstring icc,$(CC_NAME)))
    COMPILER = ICC
endif

ifeq ($(COMPILER),GCC_MINGW64)
   CC=x86_64-w64-mingw32-gcc
   CPPC=x86_64-w64-mingw32-g++
   LN=x86_64-w64-mingw32-g++
else ifeq ($(COMPILER),ICC)
   CC=icc
   CPPC=icc -w0
   LN=icc
else
   CC=gcc
   CPPC=g++
   LN=g++
endif

CLEAN_CMD=@rm -f *.o *.bak *.bin *.lrn game.* log.* 

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
    ARCH_FLAGS = -msse
else ifeq ($(UNAME_M),arm64)
    ARCH_FLAGS =
else ifeq ($(UNAME_M),aarch64)
    ARCH_FLAGS =
else
    ARCH_FLAGS =
endif

UNAME_S := $(shell uname -s)
ifeq ($(COMPILER),ICC)
    PGO_GEN_FLAGS = -prof-gen
    PGO_USE_FLAGS = -prof-use
    PGO_POST_GEN = @true
    CLEAN_PGO_DATA = rm -f *.dyn *.dpi *.dpi.lock
    LTO_FLAGS = -ipo
    COMMON_CFLAGS = -O3 -Wall -ansi -pedantic -Wno-long-long -pthread $(LTO_FLAGS) -axCORE-AVX2,CORE-AVX-1,AVX,SSE4.2,SSE4.1,SSSE3 -DNDEBUG
    COMMON_CPPFLAGS = -O3 $(LTO_FLAGS) -DNDEBUG
else
    ifeq ($(UNAME_S),Darwin)
        PGO_GEN_FLAGS = -fprofile-instr-generate
        PGO_USE_FLAGS = -fprofile-instr-use=default.profdata
        PGO_POST_GEN = xcrun llvm-profdata merge -output=default.profdata default.profraw
        CLEAN_PGO_DATA = rm -f default.profraw default.profdata
    else
        PGO_GEN_FLAGS = -fprofile-generate
        PGO_USE_FLAGS = -fprofile-use
        PGO_POST_GEN = @true
        CLEAN_PGO_DATA = rm -f *.gcda *.gcno
    endif
    LTO_FLAGS = -flto
    COMMON_CFLAGS = -O3 -Wall -std=gnu99 -pedantic -Wno-long-long -pthread $(LTO_FLAGS) $(ARCH_FLAGS) -DNDEBUG
    COMMON_CPPFLAGS = -O3 $(LTO_FLAGS) -DNDEBUG
endif

CFLAGS=$(COMMON_CFLAGS)
protectordbg: CFLAGS=-Wall -std=gnu99 -pedantic -Wno-long-long -pthread

protectorobjects = protector.o bitboard.o position.o fen.o io.o \
                   movegeneration.o matesearch.o keytable.o hash.o test.o \
                   pgn.o tools.o search.o evaluation.o coordination.o \
                   xboard.o book.o tablebase.o

protectorsources = $(protectorobjects:.o=.c) egtb.cpp

.c.o:
	$(CC) $(CFLAGS) -c $*.c

.cpp.o:
	$(CPPC) $(COMMON_CPPFLAGS) $(CPPFLAGS) -c $*.cpp

protector: $(protectorobjects) egtb.o
	$(LN) -lm -pthread $(LTO_FLAGS) -o Protector $(protectorobjects) egtb.o

protectordbg: $(protectorobjects) egtb.o
	$(LN) -lm -pthread $(LTO_FLAGS) -o Protector $(protectorobjects) egtb.o

protector_pgogen: $(protectorobjects) egtb.o
	$(LN) -lm -pthread $(LTO_FLAGS) $(PGO_GEN_FLAGS) -o Protector $(protectorobjects) egtb.o

protector_pgouse: $(protectorobjects) egtb.o
	$(LN) -lm -pthread $(LTO_FLAGS) $(PGO_USE_FLAGS) -o Protector $(protectorobjects) egtb.o

all:
	$(MAKE) clean
	$(MAKE) protector
	$(MAKE) clean

debug:
	$(MAKE) clean
	$(MAKE) protectordbg
	$(MAKE) clean

test:
	$(MAKE) debug
	./Protector -m
	./Protector -t ../pgn/problems.pgn
	./Protector -t ../pgn/wac.pgn

protectorpgo:
	$(MAKE) clean
	$(MAKE) CFLAGS='$(PGO_GEN_FLAGS) $(COMMON_CFLAGS)' COMMON_CPPFLAGS='$(PGO_GEN_FLAGS) $(COMMON_CPPFLAGS)' protector_pgogen
	LLVM_PROFILE_FILE="default.profraw" ./Protector -t ../pgn/wac.pgn
	$(PGO_POST_GEN)
	rm -f *.o
	$(MAKE) CFLAGS='$(PGO_USE_FLAGS) $(COMMON_CFLAGS)' COMMON_CPPFLAGS='$(PGO_USE_FLAGS) $(COMMON_CPPFLAGS)' protector_pgouse
	rm -f *.o

.PHONY: clean
clean:
	$(CLEAN_CMD)
	$(CLEAN_PGO_DATA)
