CMAKE_MINIMUM_REQUIRED_VERSION(VERSION 3.13)

set(PROJECT_NAME ply2stl)

project(${PROJECT_NAME})

ser(CMAKE_CXX_STANDARD 11)

find_package(PCL 1.7 REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
    src/*.cpp
    main.cpp
)

target_link_libraries( ${PROJECT_NAME}
    ${PCL_LIBRARIES}
)  

