cmake_minimum_required(VERSION 3.18)
project(naive_attention_cuda LANGUAGES CUDA CXX)

add_executable(demo naive_attention.cu)

set_target_properties(demo PROPERTIES
    CUDA_ARCHITECTURES "80;89;90"
)

target_link_libraries(demo PRIVATE m)
