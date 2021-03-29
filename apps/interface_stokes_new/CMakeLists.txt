add_executable(interface_stokes_new interface_stokes_new.cpp)
target_link_libraries(interface_stokes_new ${LINK_LIBS})
install(TARGETS interface_stokes_new RUNTIME DESTINATION bin)
