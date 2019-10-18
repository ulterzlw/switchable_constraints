cmake_minimum_required(VERSION 2.8.3)
project(unity-lans-gz)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

find_package(Eigen3 REQUIRED)

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(a main.cc)
target_link_libraries(a
  ${CERES_LIBRARIES} ${Sophus_LIBRARIES})

