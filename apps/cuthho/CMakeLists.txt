add_executable(cuthho_square cuthho_square.cpp)
target_link_libraries(cuthho_square PRIVATE ${LINK_LIBS} Spectra::Spectra)
install(TARGETS cuthho_square RUNTIME DESTINATION bin)
