cmake_minimum_required(VERSION 2.8)

project(3D_SCAN)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/scan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)


add_executable(scanapp ${SOURCES})
target_link_libraries(scanapp ${OpenCV_LIBS})