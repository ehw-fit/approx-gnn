#--project randomized --source .\src\ --opencv

cmake_minimum_required(VERSION 3.22)

project(randomized LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_UNITY_BUILD ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT ARCHITECTURE)
    set(ARCHITECTURE native)
endif()

# Compiler options
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # using GCC or Clang
    set(CMAKE_CXX_FLAGS "-Wall -Wextra ${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-unused-but-set-variable -Wno-unused-variable")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3")
    # -flto=auto
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -mavx2 -march=${ARCHITECTURE} -mtune=${ARCHITECTURE} -fassociative-math")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # using icc
    if(WIN32)
        set(CMAKE_CXX_FLAGS "/Wall ${CMAKE_CXX_FLAGS}")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /O0 /debug:full")
        if (ARCHITECTURE EQUAL native)
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O3 /MT /arch:${ARCHITECTURE} /tune:${ARCHITECTURE} /Qopt-report=1 /Qopt-report-phase=vec")
        else()
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O3 /MT -mavx2 /fp:fast=2 /QxHost /Qopt-report=1 /Qopt-report-phase=vec")
        endif()
    else()
        set(CMAKE_CXX_FLAGS "-Wall -Xlinker -lstdc++ ${CMAKE_CXX_FLAGS}")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -debug full")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mavx2 -no-prec-div -fp-model=fast=2 -march=${ARCHITECTURE} -mtune=${ARCHITECTURE} -qopt-report=1 -qopt-report-phase=vec")
        # -no-prec-div, -no-prec-sqrt -ipo -xHost
    endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # using Visual Studio C++
    set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=${ARCHITECTURE} -mtune=${ARCHITECTURE} -fassociative-math -flto=auto")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "NVHPC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc=${ACC_TARGET} -Minfo=accel -mp=multicore")
    set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -gpu=ccall,debug,pinned")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -gpu=ccall,pinned")
endif()


find_package(OpenCV REQUIRED)

file(GLOB_RECURSE SOURCES
	"${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/*.c")

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )
