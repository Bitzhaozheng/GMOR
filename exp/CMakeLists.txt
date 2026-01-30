cmake_minimum_required(VERSION 3.16)

add_library(${PROJECT_NAME} SHARED
    exp_3DMatch.hpp
    exp_KITTI.hpp
    utils_3DMatch.h
    utils_KITTI.h
    exp_base.hpp
    exp_config.h
    exp_io.hpp
    exp_utils.cpp
    exp_utils.h)

target_include_directories(${PROJECT_NAME} PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC glog::glog ${PCL_FEATURES_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_REGISTRATION_LIBRARIES} PRIVATE Matcher TRDE GMOR)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)
install(FILES exp_utils.h exp_config.h exp_io.hpp
    DESTINATION include
)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Install path")
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)
write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION lib/cmake/${PROJECT_NAME}
)
