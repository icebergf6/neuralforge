cmake_minimum_required(VERSION 3.15)
project(LeagueOfLLMs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Compiler optimization flags
if (MSVC)
    # /O2: Maximize speed, /arch:AVX2: enable AVX2 SIMD, /W3: warnings
    add_compile_options(/O2 /arch:AVX2 /W3 /DAVX2_ENABLED)
    # Link with pthreads equivalent on MSVC (threads handled by std::thread natively)
else()
    # -O3: max optimization, -march=native: use all CPU features (AVX2 if available), -pthread: threading
    add_compile_options(-O3 -march=native -Wall -Wextra -pthread)
    add_link_options(-pthread)
endif()

# Main Engine CLI (interactive text generation)
add_executable(lengine src/main.cpp)

# Unit Tests
add_executable(test_ops tests/test_ops.cpp)
add_executable(test_tokenizer tests/test_tokenizer.cpp)

# Benchmark Performance Tool
add_executable(benchmark_perf tests/benchmark_perf.cpp)

# Link pthread on non-Windows
if (NOT MSVC AND NOT WIN32)
    target_link_libraries(lengine pthread)
    target_link_libraries(test_ops pthread)
    target_link_libraries(benchmark_perf pthread)
endif()
