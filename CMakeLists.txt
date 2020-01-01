cmake_minimum_required (VERSION 3.5)
project (ProtoN)

if ( WIN32 )
    message(FATAL_ERROR "Sorry, Windows is not supported.")
endif()

# Additional modules path for cmake
set (CMAKE_MODULE_PATH
     ${CMAKE_MODULE_PATH}
     ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(safeguards)

set(CMAKE_CXX_FLAGS_DEBUG "-std=c++14 -g -fpermissive")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -O3 -mavx -g -fpermissive")
set(CMAKE_CXX_FLAGS_RELEASEASSERT "-std=c++14 -O3 -mavx -g -fpermissive")

find_package(MKL)
if (MKL_FOUND)
	include_directories(${MKL_INCLUDE_DIRS})
	set(LINK_LIBS ${LINK_LIBS} ${MKL_LIBRARIES})
	set(HAVE_INTEL_MKL TRUE)
endif()

find_package(SILO REQUIRED)
if (SILO_FOUND)
    include_directories("${SILO_INCLUDE_DIRS}")
    set(LINK_LIBS ${LINK_LIBS} ${SILO_LIBRARIES})
    set(HAVE_SILO TRUE)
endif()

find_package(Lua)
if (LUA_FOUND)
    include_directories("${LUA_INCLUDE_DIR}")
    set(LINK_LIBS ${LINK_LIBS} ${LUA_LIBRARIES})
endif()


find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LINK_LIBS "${LINK_LIBS}" pthread)
endif()

include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")

include_directories(src)
include_directories(contrib)
add_subdirectory(apps)
add_subdirectory(tests)
add_subdirectory(output)

