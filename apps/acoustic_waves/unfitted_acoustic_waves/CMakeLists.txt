add_executable(unfitted_acoustic_waves unfitted_acoustic_waves.cpp ${waves_headers} ${waves_sources})
target_link_libraries(unfitted_acoustic_waves PRIVATE ${LINK_LIBS})
