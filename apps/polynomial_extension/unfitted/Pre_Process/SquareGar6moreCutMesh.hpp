
#ifndef SquareGar6moreCutMesh_hpp
#define SquareGar6moreCutMesh_hpp

mesh_type SquareGar6moreCutMesh(level_set<RealType> & level_set_function, size_t l_divs, size_t int_refsteps);


mesh_type SquareGar6moreCutMesh(level_set<RealType> & level_set_function, size_t l_divs, size_t int_refsteps){
    
    mesh_init_params<RealType> mip;
    mip.Nx = 3;
    mip.Ny = 3;
    mip.min_x = -1.5;
    mip.max_x =  1.5;
    mip.min_y = -1.5;
    mip.max_y =  1.5;
    
    for (unsigned int i = 0; i < l_divs; i++) {
      mip.Nx *= 2;
      mip.Ny *= 2;
    }

    timecounter tc;

    tc.tic();
    mesh_type msh(mip);
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;

    CutMesh(msh,level_set_function,int_refsteps);
    return msh;
    
}

#endif
