
#ifndef CutMesh_hpp
#define CutMesh_hpp

// ----- common data types ------------------------------
using RealType = double;
typedef cuthho_poly_mesh<RealType>  mesh_type;

mesh_type MeshGeneration(level_set<RealType> & level_set_function, size_t l_divs, size_t int_refsteps){
    
    mesh_init_params<RealType> mip;
    mip.Nx = 10;
    mip.Ny = 10;
    for (unsigned int i = 0; i < l_divs; i++) {
        mip.Nx *= 2;
        mip.Ny *= 2;
    }

    mesh_type msh(mip);
    detect_node_position(msh, level_set_function); 
    detect_cut_faces(msh, level_set_function); 
    detect_cut_cells(msh, level_set_function);
    detect_cell_agglo_set(msh, level_set_function);
    make_neighbors_info_cartesian(msh);
    refine_interface(msh, level_set_function, int_refsteps);
    // make_polynomial_extension(msh, level_set_function);
    
    return msh;

}


#endif
