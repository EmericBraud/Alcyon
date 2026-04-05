# Obligatoire pour OpenBench
EXE := chess26

all:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_GUI=OFF
	cmake --build build -j$(nproc)
	cp build/chess26 ./$(EXE)

clean:
	rm -rf build $(EXE)

.PHONY: all clean