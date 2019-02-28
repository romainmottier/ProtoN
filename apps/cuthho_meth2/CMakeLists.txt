add_executable(cuthho_meth2_square cuthho_meth2_square.cpp)
target_link_libraries(cuthho_meth2_square ${LINK_LIBS})
install(TARGETS cuthho_meth2_square RUNTIME DESTINATION bin)
