include(FindPackageHandleStandardArgs)


#option(USING_INTEL_TBB "Whether the TBB library will be linked in" OFF)
#if(USING_INTEL_TBB)
	find_path(TBB_INCLUDE_DIRS
    	NAMES tbb.h
    	HINTS ENV TBB_ROOT ${TBB_ROOT} ${TBB_HINTS}
    	PATH_SUFFIXES tbb)

	if (TBB_INCLUDE_DIRS)
    	message(STATUS "TBB: found include")
	endif()

	find_library(TBB_LIBRARY
    	NAMES tbb
    	HINTS ENV TBB_ROOT ${TBB_ROOT} ${TBB_HINTS}
    	PATH_SUFFIXES lib)

	if (TBB_LIBRARY)
    	message(STATUS "TBB: found library")
    	list(APPEND TBB_LIBRARIES ${TBB_LIBRARY})
	endif()

	set(TBB_FOUND TRUE)
#endif(USING_INTEL_TBB)


FIND_PACKAGE_HANDLE_STANDARD_ARGS(TBB DEFAULT_MSG TBB_LIBRARIES TBB_INCLUDE_DIRS)
