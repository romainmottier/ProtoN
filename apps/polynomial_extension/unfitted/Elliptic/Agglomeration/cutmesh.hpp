
#ifndef CutMesh_hpp
#define CutMesh_hpp

// ----- common data types ------------------------------
using RealType = double;
typedef cuthho_poly_mesh<RealType>  mesh_type;

void CutMesh(mesh_type & msh, level_set<RealType> & level_set_function, size_t int_refsteps, bool agglomerate_Q = true);
mesh_type SquareCutMesh(level_set<RealType> & level_set_function, size_t l_divs, size_t int_refsteps = 4);

mesh_type SquareCutMesh(level_set<RealType> & level_set_function, size_t l_divs, size_t int_refsteps){
    
    mesh_init_params<RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;
    l_divs += 1;
    for (unsigned int i = 0; i < l_divs; i++) {
      mip.Nx *= 2;
      mip.Ny *= 2;
    }

    timecounter tc;

    tc.tic();
    mesh_type msh(mip);
    tc.toc();
    std::cout << bold << yellow << "         Mesh generation: " << tc << " seconds" << reset << std::endl;

    CutMesh(msh,level_set_function,int_refsteps, true);
    return msh;
}

void CutMesh(mesh_type & msh, level_set<RealType> & level_set_function, size_t int_refsteps, bool agglomerate_Q){
    
    timecounter tc;
    tc.tic();
    detect_node_position(msh, level_set_function); // ok
    detect_cut_faces(msh, level_set_function); // it could be improved
    detect_cut_cells(msh, level_set_function);
    
    if (agglomerate_Q) {
        detect_cell_agglo_set(msh, level_set_function);
        make_neighbors_info_cartesian(msh);
        refine_interface(msh, level_set_function, int_refsteps);
        make_agglomeration(msh, level_set_function);
    }else{
        refine_interface(msh, level_set_function, int_refsteps);
    }
    
    tc.toc();
    std::cout << bold << yellow << "         cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;
}



#endif
