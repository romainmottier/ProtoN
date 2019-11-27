add_executable(agglomeration_tests agglomeration_tests.cpp)
target_link_libraries(agglomeration_tests ${LINK_LIBS})
install(TARGETS agglomeration_tests RUNTIME DESTINATION bin)
