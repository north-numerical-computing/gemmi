cmake_minimum_required(VERSION 3.10)

project(GEMMI VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set compiler flags based on build type
if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")

set(CMAKE_CXX_FLAGS_DEBUG "-g -Og --coverage")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-g -Og --coverage")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-O3 -flto")

# Build MEX interface.
option(BUILD_MEX "Build MEX interface" True)
if (BUILD_MEX)
    add_subdirectory("mex")
endif()

# Build documentation.
option(BUILD_DOCS "Build documentation" False)
if (BUILD_DOCS)
    add_subdirectory("docs")
endif()

# Build tests and check coverage, if required.
option(CHECK_COVERAGE "Check test coverage" False)
option(BUILD_TESTS "Run Catch2 tests" False)
if (CHECK_COVERAGE)
    set(BUILD_TESTS True)
    set(CMAKE_BUILD_TYPE Debug)
endif()
if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()