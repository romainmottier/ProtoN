add_executable(interface interface.cpp)
target_link_libraries(interface ${LINK_LIBS})
install(TARGETS interface RUNTIME DESTINATION bin)
