# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -Iinclude -g

# Directories
SRC_DIR := src
BUILD_DIR := build
TARGET := chess26

# Find all .cpp files and corresponding .o files
SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# Default rule
all: clean $(TARGET) run

# Link all object files into the final executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile each .cpp into a .o inside build/
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Run target
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

cmake:
	cmake --build ./$(BUILD_DIR)
	./$(BUILD_DIR)/$(TARGET)

ctest:
	cmake --build ./$(BUILD_DIR)
	ctest --output-on-failure

.PHONY: all run clean cmake
