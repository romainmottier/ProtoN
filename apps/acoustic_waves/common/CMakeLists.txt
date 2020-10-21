
set(waves_sources ${waves_sources} 
	#${CMAKE_CURRENT_SOURCE_DIR}/fitted_geometry_builders.cpp
PARENT_SCOPE)

set(waves_headers ${waves_headers} 
	${CMAKE_CURRENT_SOURCE_DIR}/preprocessor.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/postprocessor.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/linear_solver.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/analytical_functions.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/newmark_hho_scheme.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/dirk_butcher_tableau.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/dirk_hho_scheme.hpp
PARENT_SCOPE)