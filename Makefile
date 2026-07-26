# Obligatoire pour OpenBench
EXE := chess26

NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu)

all:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_GUI=OFF -DENABLE_SPSA_TUNING=ON
	cmake --build build -j$(NPROC)
	cp build/chess26 ./$(EXE)
	cp -R build/data ./data

# Compile avec l'évaluation NNUE (defaut)
nnue:
	cmake -B build-nnue -DCMAKE_BUILD_TYPE=Release -DENABLE_GUI=OFF -DENABLE_NNUE_EVAL=ON
	cmake --build build-nnue -j$(NPROC)

# Compile en HCE pur (NNUE desactive)
hce:
	cmake -B build-hce -DCMAKE_BUILD_TYPE=Release -DENABLE_GUI=OFF -DENABLE_NNUE_EVAL=OFF
	cmake --build build-hce -j$(NPROC)

# Compile avec NNUE + gperftools (CPU profiler) lie a l'executable
GPERFTOOLS_PREFIX := $(shell brew --prefix gperftools 2>/dev/null)

nnue-pfl:
	cmake -B build-nnue-pfl -DCMAKE_BUILD_TYPE=Release -DENABLE_GUI=OFF -DENABLE_NNUE_EVAL=ON \
		-DCMAKE_EXE_LINKER_FLAGS="-L$(GPERFTOOLS_PREFIX)/lib -lprofiler"
	cmake --build build-nnue-pfl -j$(NPROC)

test-nnue:
	cmake -B build-nnue -DCMAKE_BUILD_TYPE=Release -DENABLE_GUI=OFF -DENABLE_NNUE_EVAL=ON
	cmake --build build-nnue -j$(NPROC)
	ctest --test-dir build-nnue --output-on-failure

test-hce:
	cmake -B build-hce -DCMAKE_BUILD_TYPE=Release -DENABLE_GUI=OFF -DENABLE_NNUE_EVAL=OFF
	cmake --build build-hce -j$(NPROC)
	ctest --test-dir build-hce --output-on-failure

clean:
	rm -rf build build-nnue build-hce build-nnue-pfl $(EXE) data

.PHONY: all nnue hce nnue-pfl test-nnue test-hce clean