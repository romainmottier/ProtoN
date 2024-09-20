
#ifndef IHHOSecondOrder_hpp
#define IHHOSecondOrder_hpp

void IHHOSecondOrder(int argc, char **argv);

void IHHOSecondOrder(int argc, char **argv){
    
    bool render_silo_files_Q = true;
    bool render_zonal_vars_Q = false;
    using RealType = double;
    size_t k_degree = 0;
    size_t n_divs   = 0;
    
    // Final time value 1.0
    std::vector<size_t> nt_v = {10,20,40,80,160,320,640};
    std::vector<double> dt_v = {0.1,0.05,0.025,0.0125,0.00625,0.003125,0.0015625};
    int tref = 0;
    size_t nt   = nt_v[tref];
    RealType dt = dt_v[tref];
    RealType ti = 0.0;
    
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
    std::cout << bold << red << "nt : " << nt << reset << std::endl;
    std::cout << bold << red << "dt : " << dt << reset << std::endl;

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
    // Construct Mass matrix
    auto mass_assembler_p = make_assembler(msh, hho_di);
    auto mass_assembler_v = make_assembler(msh, hho_di);
    auto mass_assembler_a = make_assembler(msh, hho_di);
    tc.tic();
    // Projection for acceleration
    
    TAnalyticalFunction functions;
    functions.SetFunctionType(TAnalyticalFunction::EFunctionType::EFunctionQuadraticInSpace);
    RealType t = ti;
    auto exact_scal_sol_fun     = functions.Evaluate_u(t);
    auto exact_vel_sol_fun      = functions.Evaluate_v(t);
    auto exact_accel_sol_fun    = functions.Evaluate_a(t);
    auto exact_flux_sol_fun     = functions.Evaluate_q(t);
    
    for (auto& cell : msh.cells)
    {
        auto mass_operator   = make_mass_matrix(msh, cell, hho_di);
        auto mass_operator_a = make_cell_mass_matrix(msh, cell, hho_di);
        
        Matrix<RealType, Dynamic, 1> f_p = make_rhs(msh, cell, hho_di.cell_degree(), exact_scal_sol_fun);
        Matrix<RealType, Dynamic, 1> f_v = make_rhs(msh, cell, hho_di.cell_degree(), exact_vel_sol_fun);
        Matrix<RealType, Dynamic, 1> f_a = make_rhs(msh, cell, hho_di.cell_degree(), exact_accel_sol_fun);
        
        mass_assembler_p.assemble(msh, cell, mass_operator, f_p, exact_scal_sol_fun);
        mass_assembler_v.assemble(msh, cell, mass_operator, f_v, exact_vel_sol_fun);
        mass_assembler_a.assemble(msh, cell, mass_operator_a, f_a, exact_accel_sol_fun);
    }
    mass_assembler_p.finalize();
    mass_assembler_v.finalize();
    mass_assembler_a.finalize();
    
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial scalar, velocity and acceleration
    Matrix<RealType, Dynamic, 1> p_dof_n, v_dof_n, a_dof_n;
    
    tc.tic();
    SparseLU<SparseMatrix<RealType>> analysis;
    analysis.analyzePattern(mass_assembler_p.LHS);
    analysis.factorize(mass_assembler_p.LHS);
    p_dof_n = analysis.solve(mass_assembler_p.RHS); // Initial scalar
    v_dof_n = analysis.solve(mass_assembler_v.RHS); // Initial velocity
    a_dof_n = analysis.solve(mass_assembler_a.RHS); // Initial acceleration
    
#ifdef compute_energy_Q
        Matrix<RealType, Dynamic, 2> energy_h_values(nt+1,2);
        energy_h_values(0,0) = 0.0;
        energy_h_values(0,1) = 1.0;
        RealType energy_h0 = ComputeEnergySecondOrder(msh, hho_di, mass_assembler_a, p_dof_n, v_dof_n);
        std::cout << bold << cyan << "Initial energy computed: " << tc << " seconds" << reset << std::endl;
#endif
    
    tc.toc();
    
    if(render_silo_files_Q){
        size_t it = 0;
        std::string silo_file_name = "scalar_wave_";
        RenderSiloFileScalarField(silo_file_name, it, msh, hho_di, v_dof_n, exact_vel_sol_fun, render_zonal_vars_Q);
    }
    
    // Transient problem
    bool is_implicit_Q = true;
    
    if (is_implicit_Q) {
        
        Matrix<RealType, Dynamic, 1> a_dof_np = a_dof_n;
        
        RealType beta = 0.25;
        RealType gamma = 0.5;
        for(size_t it = 1; it <= nt; it++){
                
            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
            
            // Manufactured solution
            RealType t = dt*it+ti;
            auto exact_scal_sol_fun     = functions.Evaluate_u(t);
            auto exact_vel_sol_fun      = functions.Evaluate_v(t);
            auto exact_flux_sol_fun     = functions.Evaluate_q(t);
            auto rhs_fun      = functions.Evaluate_f(t);
            
            auto assembler = make_assembler(msh, hho_di);
            tc.tic();
            for (auto& cell : msh.cells)
            {
                auto reconstruction_operator = make_hho_laplacian(msh, cell, hho_di);
#ifdef fancy_stabilization_Q
                auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
#else
                auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
#endif
                Matrix<RealType, Dynamic, Dynamic> laplacian_loc = reconstruction_operator.second + stabilization_operator;
                Matrix<RealType, Dynamic, 1> f_loc = make_rhs(msh, cell, hho_di.cell_degree(), rhs_fun);
                assembler.assemble(msh, cell, laplacian_loc, f_loc, exact_scal_sol_fun);
            }
            assembler.finalize();
            
            
            // Compute intermediate state for scalar and rate
            p_dof_n = p_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
            v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
            Matrix<RealType, Dynamic, 1> res = assembler.LHS*p_dof_n;
            
            assembler.LHS *= beta*(dt*dt);
            assembler.LHS += mass_assembler_a.LHS;
            assembler.RHS -= res;
            tc.toc();
            std::cout << bold << cyan << "Assembly completed: " << tc << " seconds" << reset << std::endl;
                    
            tc.tic();
            SparseLU<SparseMatrix<RealType>> analysis;
            analysis.analyzePattern(assembler.LHS);
            analysis.factorize(assembler.LHS);
            a_dof_np = analysis.solve(assembler.RHS); // new acceleration
            tc.toc();

            // update scalar and rate
            p_dof_n += beta*dt*dt*a_dof_np;
            v_dof_n += gamma*dt*a_dof_np;
            a_dof_n  = a_dof_np;
        
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
            
            if(render_silo_files_Q){
                std::string silo_file_name = "scalar_wave_";
                RenderSiloFileScalarField(silo_file_name, it, msh, hho_di, v_dof_n, exact_vel_sol_fun, render_zonal_vars_Q);
            }
            
#ifdef compute_energy_Q
            {
                tc.tic();
                RealType energy_h = ComputeEnergySecondOrder(msh, hho_di, assembler, p_dof_n, v_dof_n);
                tc.toc();
                energy_h_values(it,0) = t;
                energy_h_values(it,1) = energy_h/energy_h0;
                std::cout << bold << cyan << "Energy computed: " << tc << " seconds" << reset << std::endl;
            }
#endif
            
#ifdef spatial_errors_Q
            if(it == nt){
                ComputeL2ErrorSingleField(msh, hho_di, assembler, p_dof_n, exact_scal_sol_fun, exact_flux_sol_fun);
            }
#endif
            
        }
        

        
#ifdef compute_energy_Q
    //std::cout << std::setprecision(20) << bold << cyan << "Reporting initial energy value : " << energy_h0 << reset << std::endl;
    //std::cout << std::setprecision(20) << bold << cyan << "Reporting energy values : " << energy_h_values << reset << std::endl;
#endif
        
    }
    
}

#endif

