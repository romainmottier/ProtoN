add_executable(Elliptic_Interface_Agglo Elliptic_Interface_Agglo.cpp ${laplacian_conv_tests_agglo} ${laplacian_conv_tests_agglo})
target_link_libraries(Elliptic_Interface_Agglo ${LINK_LIBS})

add_executable(Elliptic_Interface_Poly_Ext Elliptic_Interface_Poly_Ext.cpp ${laplacian_conv_tests_poly_ext} ${laplacian_conv_tests_poly_ext})
target_link_libraries(Elliptic_Interface_Poly_Ext ${LINK_LIBS})
