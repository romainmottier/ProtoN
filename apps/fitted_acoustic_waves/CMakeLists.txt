add_executable(fitted_acoustic_waves fitted_acoustic_waves.cpp)
target_link_libraries(fitted_acoustic_waves ${LINK_LIBS})
install(TARGETS fitted_acoustic_waves RUNTIME DESTINATION bin)
