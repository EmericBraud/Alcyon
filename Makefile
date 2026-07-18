# Obligatoire pour OpenBench
EXE := chess26

NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu)

all:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_GUI=OFF
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

test-nnue:
	cmake -B build-nnue -DCMAKE_BUILD_TYPE=Release -DENABLE_GUI=OFF -DENABLE_NNUE_EVAL=ON
	cmake --build build-nnue -j$(NPROC)
	ctest --test-dir build-nnue --output-on-failure

test-hce:
	cmake -B build-hce -DCMAKE_BUILD_TYPE=Release -DENABLE_GUI=OFF -DENABLE_NNUE_EVAL=OFF
	cmake --build build-hce -j$(NPROC)
	ctest --test-dir build-hce --output-on-failure

clean:
	rm -rf build build-nnue build-hce $(EXE) data

.PHONY: all nnue hce test-nnue test-hce clean