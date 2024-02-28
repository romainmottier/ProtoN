#ifndef HHOFirstOrderExample_hpp
#define HHOFirstOrderExample_hpp

void HHOFirstOrderExample(int argc, char **argv);

void HHOFirstOrderExample(int argc, char **argv){
    
    using RealType = double;
    size_t k_degree = 2;
    size_t n_divs   = 2;
    
    int opt;
    while ( (opt = getopt(argc, argv, "k:l:n")) != -1 )
    {
        switch(opt)
        {
            case 'k':
            {
                k_degree = atoi(optarg);
            }
                break;
            case 'l':
            {
                n_divs = atoi(optarg);
            }
                break;
            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }
    
    std::cout << bold << red << "k : " << k_degree << reset << std::endl;
    std::cout << bold << red << "l : " << n_divs << reset << std::endl;

    // The mesh in ProtoN seems like is always 2D
     mesh_init_params<RealType> mip;
     mip.Nx = 1;
     mip.Ny = 1;
    
    for (size_t i = 0; i < n_divs; i++) {
        mip.Nx *= 2;
        mip.Ny *= 2;
    }
    
    timecounter tc;
    
    // Building the cartesian mesh
    tc.tic();
    poly_mesh<RealType> msh(mip);
    tc.toc();

    std::cout << bold << cyan << "Mesh generation: " << tc << " seconds" << reset << std::endl;

    
    // Creating HHO approximation spaces and corresponding linear operator
    hho_degree_info hho_di(k_degree,k_degree);
    
    // Solving a HDG/HHO mixed problem
    auto assembler = make_assembler(msh, hho_di, true); // another assemble version
    tc.tic();
    
    // Manufactured solution
#ifdef quadratic_space_solution_Q
    
    auto exact_scal_sol_fun = [](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
        return (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();
    };
    
    auto exact_flux_sol_fun = [](const typename poly_mesh<RealType>::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> flux(2);
        flux[0] = (1 - x)*(1 - y)*y - x*(1 - y)*y;
        flux[1] = (1 - x)*x*(1 - y) - (1 - x)*x*y;
        flux[0] *=-1.0;
        flux[1] *=-1.0;
        return flux;
    };
    
    auto rhs_fun = [](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
        double x,y;
        x = pt.x();
        y = pt.y();
        return -2.0*((x - 1)*x + (y - 1)*y);
    };
    
#else
    
    auto exact_scal_sol_fun = [](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
        return std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
    };
    
    auto exact_flux_sol_fun = [](const typename poly_mesh<RealType>::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> flux(2);
        flux[0] =  M_PI*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
        flux[1] =  M_PI*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
        flux[0] *=-1.0;
        flux[1] *=-1.0;
        return flux;
    };
    
    auto rhs_fun = [](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
        double x,y;
        x = pt.x();
        y = pt.y();
        return 2.0*M_PI*M_PI*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
    };
    
#endif
    
    for (auto& cell : msh.cells)
    {
            auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
#ifdef fancy_stabilization_Q
            auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
#else
            auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
#endif
        
        auto n_rows = reconstruction_operator.second.rows();
        auto n_cols = reconstruction_operator.second.cols();
        
        auto n_s_rows = stabilization_operator.rows();
        auto n_s_cols = stabilization_operator.cols();
        
        Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
        Matrix<RealType, Dynamic, Dynamic> S_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
        S_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, n_s_rows, n_s_cols) = stabilization_operator;
        
        // Compossing objects
        Matrix<RealType, Dynamic, Dynamic> mixed_operator_loc = R_operator - S_operator;
        Matrix<RealType, Dynamic, 1> f_loc = make_mixed_rhs(msh, cell, hho_di.cell_degree(), rhs_fun);
        assembler.assemble_mixed(msh, cell, mixed_operator_loc, f_loc, exact_scal_sol_fun);
    }
    assembler.finalize();
    
    tc.tic();
    SparseLU<SparseMatrix<RealType>> analysis_t;
    analysis_t.analyzePattern(assembler.LHS);
    analysis_t.factorize(assembler.LHS);
    Matrix<RealType, Dynamic, 1> x_dof = analysis_t.solve(assembler.RHS); // new state
    tc.toc();
    
    // Computing errors
    ComputeL2ErrorTwoFields(msh, hho_di, x_dof, exact_scal_sol_fun, exact_flux_sol_fun);
    
    size_t it = 0;
    std::string silo_file_name = "scalar_mixed_";
    RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof, exact_scal_sol_fun, exact_flux_sol_fun);
    return;
    
}

#endif
