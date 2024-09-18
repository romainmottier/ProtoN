add_executable(polymesh polymesh.cpp)
target_link_libraries(polymesh PRIVATE ${LINK_LIBS})
install(TARGETS polymesh RUNTIME DESTINATION bin)
