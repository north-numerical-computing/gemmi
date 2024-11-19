find_package(Catch2 3 REQUIRED)

add_executable(tests tests.cpp)
target_include_directories(tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(tests)

if (CHECK_COVERAGE)
    function (verbose_find_program PROG_VAR PROG_NAME)
        find_program(${PROG_VAR} ${PROG_NAME})
        if (NOT ${PROGRAM_VAR})
            message(ERROR "Cannot produce coverage report: ${PROGRAM_NAME} not found")
        endif()
    endfunction()

    verbose_find_program(GCOV gcov)
    verbose_find_program(LCOV lcov)
    verbose_find_program(GENHTML genhtml)

    if (LCOV AND GCOV AND GENHTML)
        set(COVERAGE_DIR ${PROJECT_BINARY_DIR})
        set(COVFILE ${COVERAGE_DIR}/coverage.info)
        target_compile_options(tests PRIVATE --coverage)
        target_link_options(tests PRIVATE --coverage)
        add_custom_target(coverage DEPENDS ${COVFILE})
        add_custom_command(
            OUTPUT ${COVFILE}
            COMMAND ${LCOV} -d . --zerocounters
            COMMAND ${CMAKE_CTEST_COMMAND}
            COMMAND ${LCOV} -d . --capture -o ${COVFILE}
            COMMAND ${LCOV} --remove ${COVFILE} -o ${COVFILE} '/usr/*'
            COMMAND ${LCOV} --list ${COVFILE}
            COMMAND ${GENHTML} ${COVFILE} -o ${COVERAGE_DIR}/coverage --quiet
            COMMENT "Running tests and generating coverage report"
        )
        set_directory_properties(PROPERTIES
            ADDITIONAL_CLEAN_FILES ${COVFILE}
        )
    endif()
endif()