cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(naive_attention_simt LANGUAGES ASC CXX)

add_executable(demo naive_attention.asc)

target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510 --enable-simt>
)

target_link_libraries(demo PRIVATE m)
