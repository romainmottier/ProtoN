add_executable(fitted_acoustic_waves fitted_acoustic_waves.cpp ${waves_headers} ${waves_sources})
target_link_libraries(fitted_acoustic_waves ${LINK_LIBS})
