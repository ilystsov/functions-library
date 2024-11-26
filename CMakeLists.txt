cmake_minimum_required(VERSION 3.10)
project(FunctionsTests)

add_executable(tests tests/tests.cpp)
target_link_libraries(tests /usr/lib/libgtest.a /usr/lib/libgtest_main.a)
