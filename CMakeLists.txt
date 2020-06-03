cmake_minimum_required(VERSION 2.8)
project(ch9)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

Find_Package(g2o REQUIRED)
Find_Package(Eigen3 REQUIRED)
Find_Package(Ceres REQUIRED)
Find_Package(Sophus REQUIRED)
# Find_Package(Csparse REQUIRED)

SET(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse)

# include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} "/usr/include/suitesparse/")

add_executable(bundle_adjustment_ceres bundle_adjustment_ceres.cpp)
add_executable(bundle_adjustment_g2o bundle_adjustment_g2o.cpp)
target_link_libraries(bundle_adjustment_ceres ${CERES_LIBRARIES})
target_link_libraries(bundle_adjustment_g2o ${G2O_LIBS})

