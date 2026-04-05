# Obligatoire pour OpenBench
EXE := chess26

all:
	cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_GUI=OFF
	cmake --build build -j$(nproc)
	cp build/chess26 ./$(EXE)
	rm -rf ./data
	cp -a data ./data

clean:
	rm -rf build $(EXE)

.PHONY: all clean