add_executable(interface_stokes interface_stokes.cpp)
target_link_libraries(interface_stokes ${LINK_LIBS})
install(TARGETS interface_stokes RUNTIME DESTINATION bin)
