
#ifndef CutMesh_hpp
#define CutMesh_hpp

void CutMesh(mesh_type & msh, level_set<RealType> & level_set_function, size_t int_refsteps, bool agglomerate_Q = true);

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
        // make_agglomeration(msh, level_set_function);
        make_polynomial_extension(msh, level_set_function);
    }
    else {
        refine_interface(msh, level_set_function, int_refsteps);
    }
    
    tc.toc();
    std::cout << bold << yellow << "         cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;
}

#endif
