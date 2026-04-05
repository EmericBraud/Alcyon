all:
	cmake -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j$(nproc)
	cp build/chess26 .

clean:
	rm -rf build chess26