add_executable(polymesh polymesh.cpp)
target_link_libraries(polymesh PRIVATE ${LINK_LIBS} Spectra::Spectra)
install(TARGETS polymesh RUNTIME DESTINATION bin)
