find_package(Doxygen REQUIRED)

if(DOXYGEN_FOUND)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
        "${PROJECT_BINARY_DIR}/Doxyfile"
        COPYONLY
    )

    file(READ "${PROJECT_BINARY_DIR}/Doxyfile" DOXYFILE_CONTENT)
    string(REPLACE "@PROJECT_NAME@" "${PROJECT_NAME}" DOXYFILE_CONTENT "${DOXYFILE_CONTENT}")
    string(REPLACE "@PROJECT_VERSION@" "${PROJECT_VERSION}" DOXYFILE_CONTENT "${DOXYFILE_CONTENT}")
    string(REPLACE "@PROJECT_SOURCE_DIR@" "${PROJECT_SOURCE_DIR}" DOXYFILE_CONTENT "${DOXYFILE_CONTENT}")
    string(REPLACE "@PROJECT_BINARY_DIR@" "${PROJECT_BINARY_DIR}" DOXYFILE_CONTENT "${DOXYFILE_CONTENT}")
    file(WRITE "${PROJECT_BINARY_DIR}/Doxyfile" "${DOXYFILE_CONTENT}")

    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_BINARY_DIR}/Doxyfile"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.." # Set working directory to the project root
        COMMENT "Generating documentation with Doxygen"
        VERBATIM
    )
endif()