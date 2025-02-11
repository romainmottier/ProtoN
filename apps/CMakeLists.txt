#add_subdirectory(convergence_test)
#add_subdirectory(stabilization_test)
#add_subdirectory(agglomeration_tests)
#add_subdirectory(cuthho)
#add_subdirectory(cuthho_meth2)
#add_subdirectory(cuthho_stokes)
#add_subdirectory(obstacle)

# add_subdirectory(polymesh)

#add_subdirectory(acoustic_waves)
#add_subdirectory(Unfitted_HHO)

option(BUILD_TESTS "Build tests" OFF)
if (BUILD_TESTS)
    add_subdirectory(convergence_test)
    add_subdirectory(stabilization_test)
    add_subdirectory(agglomeration_tests)
endif()

option(BUILD_CUTHHO "Build CutHHO" OFF)
if (BUILD_CUTHHO)
    add_subdirectory(cuthho)
endif()

option(BUILD_CUTHHO2 "Build CutHHO2" ON)
if (BUILD_CUTHHO2)
    add_subdirectory(cuthho_meth2)
endif()

option(BUILD_CUTHHO_STOKES "Build CutHHO for Stokes problem" ON)
if (BUILD_CUTHHO_STOKES)
    add_subdirectory(cuthho_stokes)
endif()

option(BUILD_OBSTACLE "Build obstacle application" ON)
if (BUILD_OBSTACLE)
    add_subdirectory(obstacle)
endif()

option(BUILD_POLYMESH "Build polymesh application" OFF)
if (BUILD_POLYMESH)
    add_subdirectory(polymesh)
endif()

option(BUILD_ACOUSTICS "Build acoustics application" ON)
if (BUILD_ACOUSTICS)
    add_subdirectory(acoustic_waves)
endif()

option(BUILD_POLY_EXT "Build polynomial extension applications" ON)
if (BUILD_POLY_EXT)
    add_subdirectory(Unfitted_HHO)
endif()
