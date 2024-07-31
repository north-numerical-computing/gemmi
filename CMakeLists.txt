cmake_minimum_required(VERSION 3.0)

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

set(CMAKE_CXX_FLAGS_DEBUG "-g -pg -Wall -Wall -Wextra")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-g -pg -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-O3 -Wall -Wextra")

add_executable(example tests/main.cpp)
target_include_directories(example PRIVATE ${PROJECT_SOURCE_DIR}/include)

find_package(Matlab)
matlab_add_mex(NAME gemmi SRC ${PROJECT_SOURCE_DIR}/mex/gemmi.cpp)
add_custom_command(
        TARGET gemmi POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${PROJECT_SOURCE_DIR}/mex/gemmi.m
                ${GEMMI_BINARY_DIR}/gemmi.m)
