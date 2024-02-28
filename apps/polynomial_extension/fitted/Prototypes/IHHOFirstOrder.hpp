#ifndef IHHOFirstOrder_hpp
#define IHHOFirstOrder_hpp

void IHHOFirstOrder(int argc, char **argv);

void IHHOFirstOrder(int argc, char **argv){
    
    bool render_silo_files_Q = false;
    bool render_zonal_vars_Q = false;
    using RealType = double;
    size_t k_degree = 2;
    size_t n_divs   = 2;
    
    // Final time value 1.0
    std::vector<size_t> nt_v = {10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480};
    std::vector<double> dt_v = {0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125, 0.000390625, 0.0001953125, 0.00009765625, 0.00009765625/2};
    
    int tref = 0;
    int s = 3;
    
    size_t nt       = nt_v[tref];
    RealType dt     = dt_v[tref];
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
                n_divs = atoi(optarg); //3
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

    // Projection of initial data
    
    // Creating HHO approximation spaces and corresponding linear operator
    hho_degree_info hho_di(k_degree,k_degree);
    
    // Solving a HDG/HHO mixed problem
    auto assembler = make_assembler(msh, hho_di, true); // another assemble version
    auto mass_assembler = make_assembler(msh, hho_di, true); // another assemble version
    
    TAnalyticalFunction functions;
    functions.SetFunctionType(TAnalyticalFunction::EFunctionType::EFunctionNonPolynomial);
    RealType t = 0.0;
    auto exact_scal_sol_fun     = functions.Evaluate_u(t);
    auto exact_vel_sol_fun      = functions.Evaluate_v(t);
    auto exact_accel_sol_fun    = functions.Evaluate_a(t);
    auto exact_flux_sol_fun     = functions.Evaluate_q(t);
    
    tc.tic();

    // Projecting initial state(flux and velocity)
    Matrix<RealType, Dynamic, 1> x_dof_n_m = assembler.RHS;
    {
        size_t cell_i = 0;
        for (auto& cell : msh.cells)
        {
            { /// global mass
                
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
                
                Matrix<RealType, Dynamic, Dynamic> M_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
                Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
                Matrix<RealType, Dynamic, Dynamic> M_q = R_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols);
                
                M_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols) = M_q;
                
                Matrix<RealType, Dynamic, Dynamic> v_mass_operator = make_cell_mass_matrix(msh, cell, hho_di);
                size_t cell_dof_c;
                {
                    cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                    cell_dof_c = cell_basis.size();
                }
                Matrix<RealType, Dynamic, Dynamic> M_v = v_mass_operator.block(0, 0, cell_dof_c, cell_dof_c);
                M_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, cell_dof_c, cell_dof_c) = M_v;
                
                // Compossing objects
                Matrix<RealType, Dynamic, 1> f_loc = Matrix<RealType, Dynamic, 1>::Zero(n_rows, 1);
                mass_assembler.assemble_mixed(msh, cell, M_operator, f_loc, exact_vel_sol_fun);

            }
            
            
            auto mass_flux_operator = make_flux_cell_mass_matrix(msh, cell, hho_di);
            Matrix<RealType, Dynamic, 1> f_q = make_vector_variable_rhs(msh, cell, hho_di.cell_degree()+1, exact_flux_sol_fun);
            Matrix<RealType, Dynamic, 1> dof_q = mass_flux_operator.llt().solve(f_q);
            Matrix<RealType, Dynamic, 1> dof_v = project_function(msh, cell, hho_di, exact_vel_sol_fun);
            
            size_t cell_dof;
            size_t cell_rec_dof;
            {
                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                cell_dof = cell_basis.size();

            }
            cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree()+1);
            cell_rec_dof = cell_basis.size()-1;
            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof), 0, cell_rec_dof, 1) = dof_q;
            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof)+cell_rec_dof, 0, cell_dof, 1) = dof_v.block(0, 0, cell_dof, 1);
            cell_i++;
        }
    }
    mass_assembler.finalize();
    tc.toc();
    std::cout << bold << cyan << "Initial state computed: " << tc << " seconds" << reset << std::endl;
    size_t it = 0;
    if (render_silo_files_Q) {
        std::string silo_file_name = "scalar_wave_";
        RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
    }
    
#ifdef compute_energy_Q
    Matrix<RealType, Dynamic, 2> energy_h_values(nt+1,2);
    tc.tic();
    energy_h_values(0,0) = 0.0;
    energy_h_values(0,1) = 1.0;
    RealType energy_h0 = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun);
    tc.toc();
    std::cout << bold << cyan << "Initial energy computed: " << tc << " seconds" << reset << std::endl;
#endif

    // Solving a HDG/HHO mixed problem
    Matrix<RealType, Dynamic, Dynamic> a;
    Matrix<RealType, Dynamic, 1> b;
    Matrix<RealType, Dynamic, 1> c;
    
    // DIRK(s) schemes
    bool is_sdirk_Q = true;

    if (is_sdirk_Q) {
        TDIRKSchemes::SDIRKSchemes(s, a, b, c); 
    }
    
    else {
    
        TDIRKSchemes::DIRKSchemesSS(s, a, b, c);
    }
    
    Matrix<double, Dynamic, 1> x_dof_n;
    bool optimized_Q = false;
    
            
    if (optimized_Q) {
    
        // Transient problem

#ifdef InhomogeneousQ
        ComputeInhomogeneousKGFG(assembler.LHS, assembler.RHS, msh, hho_di, assembler, t, functions);
        
#else
        ComputeKGFG(assembler.LHS, assembler.RHS, msh, hho_di, assembler, t, functions);
#endif
        TDIRKHHOAnalyses dirk_an(assembler.LHS,assembler.RHS,mass_assembler.LHS);
        
        if (is_sdirk_Q) {
            double scale = a(0,0) * dt;
            dirk_an.SetScale(scale);
            dirk_an.DecomposeMatrix();
        }
        
        for(size_t it = 1; it <= nt; it++){

            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
                   
            RealType tn = dt*(it-1)+ti;
            tc.tic();
            DIRKStepOpt(s, a, b, c, msh, hho_di, assembler, dirk_an, tn, dt, functions, x_dof_n_m, x_dof_n, is_sdirk_Q);
            tc.toc();
            
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
            
            RealType t = tn + dt;
            auto exact_vel_sol_fun = functions.Evaluate_v(t);
            auto exact_flux_sol_fun = functions.Evaluate_q(t);
            
            if (render_silo_files_Q) {
                std::string silo_file_name = "scalar_wave_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
            }
            
           
    #ifdef compute_energy_Q
                {
                    RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
                    energy_h_values(it,0) = t;
                    energy_h_values(it,1) = energy_h/energy_h0;
                }
    #endif
            
    #ifdef spatial_errors_Q
            if(it == nt){
                std::string silo_file_name = "scalar_wave_at_tf_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
                std::cout << green << "dt size = " << std::endl << dt << std::endl;
                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
            }
    #endif
            x_dof_n_m = x_dof_n;
        }
    }
    
    else {

        // Transient problem
        for(size_t it = 1; it <= nt; it++){
                
            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
            
            RealType tn = dt*(it-1)+ti;
            tc.tic();
////////////////////////////////////////////////////////////////
            DIRKStep(s, a, b, c, msh, hho_di, assembler, mass_assembler.LHS, tn, dt, functions, x_dof_n_m, x_dof_n);
            tc.toc();
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
            
            RealType t = tn + dt;
            auto exact_vel_sol_fun = functions.Evaluate_v(t);
            auto exact_flux_sol_fun = functions.Evaluate_q(t);
            
            if (render_silo_files_Q) {
                std::string silo_file_name = "scalar_wave_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
            }
            
           
    #ifdef compute_energy_Q
                {
                    RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
                    energy_h_values(it,0) = t;
                    energy_h_values(it,1) = energy_h/energy_h0;
                }
    #endif
            
    #ifdef spatial_errors_Q
            if(it == nt){
                std::string silo_file_name = "scalar_wave_at_tf_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
                std::cout << green << "dt size = " << std::endl << dt << std::endl;
                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
            }
    #endif
            x_dof_n_m = x_dof_n;
        }
    }
    
#ifdef compute_energy_Q
    std::ofstream energy_file ("IHHO_energy.txt");
    if (energy_file.is_open())
    {
      energy_file << std::setprecision(20) << bold << cyan << "Reporting initial energy value : " << energy_h0 << reset << std::endl;
      energy_file << std::setprecision(20) << bold << cyan << "Reporting energy values : " << energy_h_values << reset << std::endl;
      energy_file.close();
    }
#endif
    
}

#endif
