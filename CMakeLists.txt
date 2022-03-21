set(CMAKE_BUILD_TYPE RELEASE)
project(epipolar)
cmake_minimum_required(VERSION 2.8)


#-------------------------------------------------------------------
# Compiler Options
#-------------------------------------------------------------------

add_definitions("-Wall -std=c++17 -O3")

if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    message(STATUS "Debug configuration")
    add_definitions("-0g")
else(CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    message(STATUS "Release configuration")
    add_definitions("-O3")
endif (CMAKE_BUILD_TYPE STREQUAL "DEBUG")

add_definitions(-D_USE_MATH_DEFINES)


#-------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
        include/
        ${OpenCV_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}
)


#-------------------------------------------------------------------
# Executables generation
#-------------------------------------------------------------------

add_executable(main_essential main_essential.cpp)
target_link_libraries(main_essential ${OpenCV_LIBS} ${EIGEN3_LIBS})

add_executable(main_homography main_homography.cpp)
target_link_libraries(main_homography ${OpenCV_LIBS} ${EIGEN3_LIBS})
