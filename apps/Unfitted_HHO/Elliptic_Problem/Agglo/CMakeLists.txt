add_executable(Elliptic_Interface_Agglo Elliptic_Interface_Agglo.cpp ${laplacian_conv_tests_agglo} ${laplacian_conv_tests_agglo})
target_link_libraries(Elliptic_Interface_Agglo ${LINK_LIBS})

