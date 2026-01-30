cmake_minimum_required(VERSION 3.16)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type, default is Release.")

project(RegistrationFactory VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Access to Eigen matrices may cause segmentation fault due to misaligned memory with PCL
option(MARCH_NATIVE "Enable -march=native for compatibility with Eigen alignment in self-built PCL." OFF)
set(ADDITIONAL_COMPILE_OPTIONS "")
message(STATUS "${CMAKE_SYSTEM_PROCESSOR} architecture and ${CMAKE_CXX_COMPILER_ID} compiler detected.")
if(MARCH_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(ADDITIONAL_COMPILE_OPTIONS ${ADDITIONAL_COMPILE_OPTIONS} -march=native)
endif()

find_package(Eigen3 3.4 REQUIRED)
find_package(PCL 1.12 REQUIRED COMPONENTS common features registration kdtree io visualization)
find_package(nanoflann REQUIRED)
find_package(OpenMP REQUIRED)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)

if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_definitions(-D_USE_MATH_DEFINES)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

set(TRDE_HEADERS include/trde.h include/trde_node.hpp include/registration_utils.hpp include/registration_base.hpp)
set(TRDE_SOURCES src/trde.cpp)

set(GMOR_HEADERS include/gmor.h include/gmor_node.hpp include/registration_utils.hpp include/registration_base.hpp include/interval_stabbing.hpp include/sweepline.hpp include/sphere_projection.hpp)
set(GMOR_SOURCES src/gmor.cpp)

add_library(TRDE SHARED ${TRDE_HEADERS} ${TRDE_SOURCES})
target_compile_options(TRDE PUBLIC ${ADDITIONAL_COMPILE_OPTIONS})
target_link_libraries(TRDE PUBLIC Eigen3::Eigen PRIVATE OpenMP::OpenMP_CXX)

add_library(GMOR SHARED ${GMOR_HEADERS} ${GMOR_SOURCES})
target_compile_options(GMOR PUBLIC ${ADDITIONAL_COMPILE_OPTIONS})
target_link_libraries(GMOR PUBLIC Eigen3::Eigen PRIVATE OpenMP::OpenMP_CXX)

add_library(Matcher INTERFACE include/matcher.hpp)
target_include_directories(Matcher INTERFACE ${PCL_INCLUDE_DIRS})
target_link_libraries(Matcher INTERFACE ${PCL_COMMON_LIBRARIES} ${PCL_KDTREE_LIBRARIES} nanoflann::nanoflann OpenMP::OpenMP_CXX)

install(TARGETS GMOR TRDE
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Experiments
add_subdirectory(exp)

# User interface
add_executable(${PROJECT_NAME}_demo main.cpp demo_pipeline.cpp demo_pipeline.h)
target_include_directories(${PROJECT_NAME}_demo PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}_demo PRIVATE ${PROJECT_NAME} ${PCL_COMMON_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} Eigen3::Eigen glog::glog gflags OpenMP::OpenMP_CXX)
