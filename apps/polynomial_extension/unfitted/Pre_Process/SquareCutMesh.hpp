
#ifndef SquareCutMesh_hpp
#define SquareCutMesh_hpp

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

#endif
