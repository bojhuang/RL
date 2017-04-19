cmake_minimum_required(VERSION 3.2)

project (RL CXX)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

message("build for " ${CMAKE_SYSTEM_NAME})

include_directories(utils)
include_directories(../Arcade-Learning-Environment/src)
include_directories(../Arcade-Learning-Environment/src/common)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(CMAKE_CXX_FLAGS "-std=gnu++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-pthread ${CMAKE_CXX_FLAGS}")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/MNIST
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable (RL_demo     
				${SOURCES}
				utils/fakeBMP.cpp)

# target_link_libraries(RL_demo ${OPENGLES2_gl_LIBRARY} ${SDL2_LIBRARY} ${SDL2IMAGE_LIBRARY})



