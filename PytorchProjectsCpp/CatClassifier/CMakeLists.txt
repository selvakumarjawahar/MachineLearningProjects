cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(cat-classifier)

find_package(Torch REQUIRED)

add_executable(
        cat-classifier
        src/main.cpp
)

target_link_libraries(cat-classifier
        PRIVATE "${TORCH_LIBRARIES}"
)

set_property(
        TARGET cat-classifier
        PROPERTY CXX_STANDARD 11
)