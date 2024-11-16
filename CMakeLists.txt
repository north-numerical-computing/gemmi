cmake_minimum_required(VERSION 3.10)

project(GEMMI VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set compiler flags based on build type
if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -Og -Wall -Wall -Wextra")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-g -Og -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-O3 -Wall -Wextra")

add_subdirectory("mex")
add_subdirectory("tests")