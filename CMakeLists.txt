cmake_minimum_required (VERSION 3.18)
project (ProtoN)

if ( WIN32 )
    message(FATAL_ERROR "Sorry, Windows is not supported.")
endif()

# Additional modules path for cmake
set (CMAKE_MODULE_PATH
     ${CMAKE_MODULE_PATH}
     ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(safeguards)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "-g -fpermissive")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mavx -DNDEBUG -fpermissive")
set(CMAKE_CXX_FLAGS_RELEASEASSERT "-O3 -mavx -g -fpermissive")

find_package(MKL)
if (MKL_FOUND)
	include_directories(${MKL_INCLUDE_DIRS})
	set(LINK_LIBS ${LINK_LIBS} ${MKL_LIBRARIES})
	set(HAVE_INTEL_MKL TRUE)
endif()

find_package(TBB)
if (TBB_FOUND)
  include_directories(${TBB_INCLUDE_DIRS})
  set(LINK_LIBS ${LINK_LIBS} ${TBB_LIBRARIES})
  set(HAVE_INTEL_TBB TRUE)
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

find_package(TBB)
if (TBB_FOUND)
  #target_link_libraries(my_project PRIVATE tbb::tbb_static)
  include_directories(${TBB_INCLUDE_DIRS})
  set(LINK_LIBS ${LINK_LIBS} ${TBB_LIBRARIES})
  set(HAVE_INTEL_TBB TRUE)
endif()





find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LINK_LIBS "${LINK_LIBS}" pthread)
endif()

include_directories(contrib/spectra/include/)

configure_file (
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h"
  )
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${PROJECT_SOURCE_DIR}")

add_subdirectory(contrib/sol2)
<<<<<<< HEAD
set(SOL2_BUILD_LUA OFF CACHE BOOL "Override sol2 build lua option")
set(LINK_LIBS ${LINK_LIBS} sol2::sol2)

include_directories(contrib/spectra/include)
=======
set(LINK_LIBS "${LINK_LIBS}" sol2::sol2)
>>>>>>> romain_rebase_test

include_directories(src)
add_subdirectory(apps)
add_subdirectory(tests)
add_subdirectory(output)
