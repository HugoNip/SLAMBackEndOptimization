cmake_minimum_required(VERSION 3.16)
project(ch9)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Ceres
Find_Package(g2o REQUIRED)
Find_Package(Eigen3 REQUIRED)
Find_Package(Ceres REQUIRED)
Find_Package(Sophus REQUIRED)

SET(G2O_LIBS g2o_stuff g2o_core cxsparse)

include_directories(${CERES_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})

add_executable(bundle_adjustment_ceres bundle_adjustment_ceres.cpp)
target_link_libraries(bundle_adjustment_ceres ${CERES_LIBRARIES})