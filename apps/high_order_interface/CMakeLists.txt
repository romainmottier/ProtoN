add_executable(high_order_interface high_order_interface.cpp)
target_link_libraries(high_order_interface ${LINK_LIBS})
install(TARGETS high_order_interface RUNTIME DESTINATION bin)
