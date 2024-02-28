
#ifndef ICutHHOSecondOrder_hpp
#define ICutHHOSecondOrder_hpp

void ICutHHOSecondOrder(int argc, char **argv);

void ICutHHOSecondOrder(int argc, char **argv){


   ////////////////////////////////////////////////////////////////////////////
   ////////////////////////         Variables          ////////////////////////
   ////////////////////////////////////////////////////////////////////////////

    bool sc_Q            = true;   
    bool report_energy_Q = true;
    bool direct_solver_Q = true;
    
    size_t degree       = 0;
    size_t l_divs       = 4;
    size_t nt_divs      = 2;
    size_t int_refsteps = 10;
    bool   dump_debug   = false;


   ////////////////////////////////////////////////////////////////////////////
   /////////////////////////         Inputs          //////////////////////////
   ////////////////////////////////////////////////////////////////////////////

    // Simplified input
    int ch;
    while ( (ch = getopt(argc, argv, "k:l:r:n:d")) != -1 )
    {
        switch(ch)
        {   
            // Method degree
            case 'k':
               degree = atoi(optarg);
               break;
               
            // Number of cells in x and y direction
            case 'l':
               l_divs = atoi(optarg);
               break;

            // Number of interface refinement steps
            case 'r':
               int_refsteps = atoi(optarg);
               break;
               
            // Number of time refinement
            case 'n':
               nt_divs = atoi(optarg);
               break;

            // Dump debug data 
            case 'd':
               dump_debug = true;
               break;

            // Invalid argument
            case '?':
            default:
               std::cout << "wrong arguments" << std::endl;
               exit(1);
         }
     }

    argc -= optind;
    argv += optind;

   ////////////////////////////////////////////////////////////////////////////
   //////////////////////////         Mesh          ///////////////////////////
   ////////////////////////////////////////////////////////////////////////////
   
    RealType radius = 1.0/3.0;
    auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);
    mesh_type msh = SquareCutMesh(level_set_function, l_divs, int_refsteps);
    
    if (dump_debug)
    {
        dump_mesh(msh);
        output_mesh_info(msh, level_set_function);
        hho_degree_info hdi(degree+1, degree);
        PrintIntegrationRule(msh,hdi);
    }


   ////////////////////////////////////////////////////////////////////////////
   //////////////////////////      Time Control     ///////////////////////////
   ////////////////////////////////////////////////////////////////////////////    

    size_t nt = 10;
    for (unsigned int i = 0; i < nt_divs; i++) {
        nt *= 2;
    }
    
    RealType ti = 0.0;          // Initial time
    RealType tf = 1.0;          // Final time 
    RealType dt = (tf-ti)/nt;   // Time step
    RealType t = ti;          
    
    timecounter tc;
    
    // Newmark parameters
    RealType beta = 0.25;
    RealType gamma = 0.5;
    
    
   ////////////////////////////////////////////////////////////////////////////
   //////////////////////     Create static data     //////////////////////////
   ////////////////////////////////////////////////////////////////////////////    
    
    SparseMatrix<RealType> Kg, Kg_c, Mg;
    hho_degree_info hdi(degree+1, degree);
    auto test_case = make_test_case_laplacian_waves(t,msh, level_set_function);
    auto method = make_gradrec_interface_method(msh, 1.0, test_case);
    std::vector<std::pair<size_t,size_t>> cell_basis_data = create_kg_and_mg_cuthho_interface(msh, hdi, method, test_case, Kg, Mg);
    

    linear_solver<RealType> analysis;
    Kg_c = Kg;
    Kg *= beta*(dt*dt);
    Kg += Mg;
    
    
   ////////////////////////////////////////////////////////////////////////////
   /////////////////////     Static Condensation     //////////////////////////
   //////////////////////////////////////////////////////////////////////////// 
   
    if (sc_Q) {
        size_t n_dof = Kg.rows();
        size_t n_cell_dof = 0;
        for (auto &chunk : cell_basis_data) {
            n_cell_dof += chunk.second;
        }
        size_t n_face_dof = n_dof - n_cell_dof;
        analysis.set_Kg(Kg, n_face_dof);
        analysis.condense_equations_irregular_blocks(cell_basis_data);
    }else{
        analysis.set_Kg(Kg);
    }
    
    
    if (!direct_solver_Q) {
        analysis.set_iterative_solver(true);
    }
    analysis.factorize();
    
    // Projecting initial scalar, velocity and acceleration
    Matrix<RealType, Dynamic, 1> u_dof_n, v_dof_n, a_dof_n;
    
    std::ofstream enery_file("one_field_energy.txt");
    
    bool write_error_Q  = false;
    
    
    
   ////////////////////////////////////////////////////////////////////////////
   /////////////////////////     Temporal Loop     ////////////////////////////
   //////////////////////////////////////////////////////////////////////////// 
    
    for(size_t it = 1; it <= nt; it++){ // for each time step
        
        std::cout << std::endl;
        std::cout << "Time step number: " <<  it << std::endl;
        RealType t = dt*it+ti;
        auto test_case = make_test_case_laplacian_waves(t,msh, level_set_function);
        auto method = make_gradrec_interface_method(msh, 1.0, test_case);
        if (it == nt) {
            write_error_Q = true;
        }

        tc.tic();

        newmark_step_cuthho_interface(it, t, dt, beta, gamma, msh, hdi, method, test_case, u_dof_n,  v_dof_n, a_dof_n, Kg_c, analysis, write_error_Q);
        tc.toc();

        std::cout << bold << yellow << "Newmark step performed in : " << tc << " seconds" << reset << std::endl;
        
        // energy evaluation
        if(report_energy_Q){

            RealType energy_0 = 0.125;
            Matrix<RealType, Dynamic, 1> cell_mass_tested = Mg * v_dof_n;
            Matrix<RealType, 1, 1> term_1 = v_dof_n.transpose() * cell_mass_tested;
            RealType energy_h = 0.5*term_1(0,0);

            Matrix<RealType, Dynamic, 1> cell_stiff_tested = Kg_c * u_dof_n;
            Matrix<RealType, 1, 1> term_2 = u_dof_n.transpose() * cell_stiff_tested;
            energy_h += 0.5*term_2(0,0);

            energy_h /= energy_0;
            std::cout << bold << yellow << "Energy = " << energy_h << reset << std::endl;
            enery_file << std::setprecision(16) << t << " " << energy_h << std::endl;
        }
        
    }
    
    std::cout << "Number of equations : " << analysis.n_equations() << std::endl;
    std::cout << "Number of steps : " <<  nt << std::endl;
    std::cout << "Time step size : " <<  dt << std::endl;
}

#endif

