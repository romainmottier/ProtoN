add_executable(taylor_green_vortex taylor_green_vortex.cpp)
target_link_libraries(taylor_green_vortex ${LINK_LIBS})
install(TARGETS taylor_green_vortex RUNTIME DESTINATION bin)
