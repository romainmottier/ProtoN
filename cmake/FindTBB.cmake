include(FindPackageHandleStandardArgs)


option(USING_INTEL_TBB "Whether the TBB library will be linked in" OFF)
if(USING_INTEL_TBB)
	find_file(TBB_HEADER_FILE tbb.h HINTS /opt/intel/oneapi/tbb/latest/include/oneapi/ NO_DEFAULT_PATH)

	if(NOT TBB_HEADER_FILE MATCHES "tbb.h")
		set (TBB_HEADER_FILE "" CACHE PATH "Path to 'tbb.h' file")
		message( FATAL_ERROR "TBB_HEADER_FILE must contain a valid path to 'tbb.h' file!")
	else()
		get_filename_component(TBB_ROOT_DIR ${TBB_HEADER_FILE} DIRECTORY)
		get_filename_component(TBB_INCLUDE_DIRS ${TBB_ROOT_DIR} DIRECTORY)
		get_filename_component(TBB_ROOT_DIR ${TBB_INCLUDE_DIRS} DIRECTORY)
		if(APPLE)
			set(TBB_INCLUDE_DIRS ${TBB_ROOT_DIR}/include)		
			set(TBB_LIBRARIES ${TBB_ROOT_DIR}/lib/libtbb.dylib)
		else() # Linux
			set(TBB_LIBRARIES ${TBB_ROOT_DIR}/lib/intel64/gcc4.8/libtbb.so)
		endif()
	#include_directories(${TBB_INCLUDE_DIRS})
	#link_libraries(${TBB_LIBRARIES})
	endif()
	set(TBB_FOUND TRUE)
endif(USING_INTEL_TBB)


FIND_PACKAGE_HANDLE_STANDARD_ARGS(TBB DEFAULT_MSG TBB_LIBRARIES TBB_INCLUDE_DIRS)
