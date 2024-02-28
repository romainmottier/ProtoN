add_executable(fitted_elasto_acoustic_waves fitted_elasto_acoustic_waves.cpp ${waves_coupling_headers} ${waves_coupling_sources})
target_link_libraries(fitted_elasto_acoustic_waves PRIVATE ${LINK_LIBS} Spectra::Spectra) 
