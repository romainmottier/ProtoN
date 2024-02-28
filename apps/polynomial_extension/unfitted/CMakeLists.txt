add_executable(unfitted_elasto_acoustic_waves unfitted_elasto_acoustic_waves.cpp ${waves_coupling_headers} ${waves_coupling_sources})
target_link_libraries(unfitted_elasto_acoustic_waves PRIVATE ${LINK_LIBS} Spectra::Spectra)
