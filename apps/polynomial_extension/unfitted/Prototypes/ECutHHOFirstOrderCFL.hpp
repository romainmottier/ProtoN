
#ifndef ECutHHOFirstOrderCFL_hpp
#define ECutHHOFirstOrderCFL_hpp

void ECutHHOFirstOrderCFL(int argc, char **argv);

void ECutHHOFirstOrderCFL(int argc, char **argv){
    
    bool report_energy_Q = true;

    size_t degree           = 0;
    size_t l_divs          = 0;
    size_t nt_divs       = 0;
    size_t int_refsteps     = 4;
    bool dump_debug         = false;

    /* k <deg>:     method degree
     * l <num>:     number of cells in x and y direction
     * r <num>:     number of interface refinement steps
     * d:           dump debug data
     */

    // Simplified input
     int ch;
     while ( (ch = getopt(argc, argv, "k:l:r:n:d")) != -1 )
     {
         switch(ch)
         {
             case 'k':
                 degree = atoi(optarg);
                 break;

             case 'l':
                 l_divs = atoi(optarg);
                 break;

             case 'r':
                 int_refsteps = atoi(optarg);
                 break;
                 
             case 'n':
                 nt_divs = atoi(optarg);
                 break;

             case 'd':
                 dump_debug = true;
                 break;

             case '?':
             default:
                 std::cout << "wrong arguments" << std::endl;
                 exit(1);
         }
     }

    argc -= optind;
    argv += optind;
    
    std::vector<RealType> tf_vec;
    tf_vec = {0.5,0.4,0.3,0.2};
//    tf_vec = {0.5/4,0.4/4,0.3/4,0.2/4};

    RealType ti = 0.0;
    RealType tf = tf_vec[degree];;
    int nt_base = nt_divs;
    RealType energy_0 = 0.125;
    std::ofstream simulation_log("acoustic_two_fields_explicit_cfl.txt");
    
    for (int s = 1; s < 5; s++) {
        simulation_log << " ******************************* " << std::endl;
        simulation_log << " number of stages s =  " << s << std::endl;
        simulation_log << std::endl;
    
        for(size_t l = 0; l <= l_divs; l++){
        
            RealType radius = 1.0/3.0;
            auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);
//            RealType cy = 1.0e-15+0.5;
//            auto level_set_function = line_level_set<RealType>(cy);
            mesh_type msh = SquareCutMesh(level_set_function, l, int_refsteps);
            
            if (dump_debug)
            {
                dump_mesh(msh);
                output_mesh_info(msh, level_set_function);
            }

            size_t nt = nt_base;
            for (unsigned int i = 0; i < nt_divs; i++) {
            
                RealType dt     = (tf-ti)/nt;
                RealType t = ti;
                timecounter tc;
                
                // ERK(s) schemes
                Matrix<RealType, Dynamic, Dynamic> a;
                Matrix<RealType, Dynamic, 1> b;
                Matrix<RealType, Dynamic, 1> c;
                erk_butcher_tableau::erk_tables(s, a, b, c);
                hho_degree_info hdi(degree+1, degree);
                
                SparseMatrix<RealType> Kg, Mg;
                auto test_case = make_test_case_laplacian_waves_mixed(t,msh, level_set_function);
                auto method = make_gradrec_mixed_interface_method(msh, 1.0, test_case);
                size_t n_faces = 0;
                std::vector<std::pair<size_t,size_t>> cell_basis_data = create_mixed_kg_and_mg_cuthho_interface(msh, hdi, method, test_case, Kg, Mg, true, &n_faces);
                
                
                tc.tic();
                size_t n_face_dof, n_face_basis;
                size_t n_dof = Kg.rows();
                size_t n_cell_dof = 0;
                for (auto &chunk : cell_basis_data) {
                    n_cell_dof += chunk.second;
                }
                n_face_dof = n_dof - n_cell_dof;
                n_face_basis = face_basis<mesh_type,RealType>::size(degree);
                
                Matrix<RealType, Dynamic, 1> x_dof, rhs = Matrix<RealType, Dynamic, 1>::Zero(n_dof, 1);
                erk_hho_scheme<RealType> analysis(Kg,rhs,Mg,n_face_dof);
                analysis.Kcc_inverse_irregular_blocks(cell_basis_data);
                analysis.Sff_inverse(std::make_pair(n_faces, n_face_basis));
                tc.toc();
                std::cout << bold << cyan << "ERK analysis created: " << tc << " seconds" << reset << std::endl;
                    
                std::ofstream enery_file("e_two_fields_energy.txt");
                bool write_error_Q  = false;
                bool approx_fail_check_Q = false;

                RealType energy = energy_0;
                for(size_t it = 1; it <= nt; it++){ // for each time step
                    
                    std::cout << std::endl;
                    std::cout << "Time step number: " <<  it << std::endl;
                    RealType t = dt*it+ti;
                    auto test_case = make_test_case_laplacian_waves_mixed(t,msh, level_set_function);
                    auto method = make_gradrec_mixed_interface_method(msh, 1.0, test_case);
                    if (it == nt) {
                        write_error_Q = true;
                    }
                    
                    tc.tic();
                    erk_step_cuthho_interface_cfl(it, s, ti, dt, a, b, c, msh, hdi, method, test_case, x_dof, analysis, write_error_Q);
                    tc.toc();
                    std::cout << bold << yellow << "ERK step performed in : " << tc << " seconds" << reset << std::endl;
                    
                    // energy evaluation
                     if(report_energy_Q){

                         
                         Matrix<RealType, Dynamic, 1> cell_mass_tested = Mg * x_dof;
                         Matrix<RealType, 1, 1> term_1 = x_dof.transpose() * cell_mass_tested;
                         RealType energy_n = 0.5*term_1(0,0);
                         
                         RealType relative_energy = (energy_n - energy) / energy;
                         RealType relative_energy_0 = (energy_n - energy_0) / energy_0;
                         bool unstable_check_Q = (relative_energy > 1.0e-2) || (relative_energy_0 >= 1.0e-2);
                         if (unstable_check_Q) { // energy is increasing
                             approx_fail_check_Q = true;
                             break;
                         }
                         energy = energy_n;
        
                     }
                    
                    
                }
                
                RealType h_T = std::numeric_limits<RealType>::max();
                for (auto cell : msh.cells) {
                    
                    RealType h = diameter(msh, cell);
                    if (h < h_T) {
                        h_T = h;
                    }
                }
                
                if(approx_fail_check_Q){
                    simulation_log << std::endl;
                    simulation_log << "Simulation is unstable for :"<< std::endl;
                    simulation_log << "Number of equations : " << analysis.n_equations() << std::endl;
                    simulation_log << "Number of ERK steps =  " << s << std::endl;
                    simulation_log << "Number of time steps =  " << nt << std::endl;
                    simulation_log << "dt size =  " << dt << std::endl;
                    simulation_log << "h size =  " << h_T << std::endl;
                    simulation_log << "CFL (dt/h) =  " << dt/(h_T) << std::endl;
                    simulation_log << std::endl;
                    simulation_log.flush();
                    break;
                }else{
                    simulation_log << "Simulation is stable for :"<< std::endl;
                    simulation_log << "Number of equations : " << analysis.n_equations() << std::endl;
                    simulation_log << "Number of ERK steps =  " << s << std::endl;
                    simulation_log << "Number of time steps =  " << nt << std::endl;
                    simulation_log << "dt size =  " << dt << std::endl;
                    simulation_log << "h size =  " << h_T << std::endl;
                    simulation_log << "CFL (dt/h) =  " << dt/(h_T) << std::endl;
                    simulation_log << std::endl;
                    simulation_log.flush();
                    nt -= 5;
                    continue;
                }
                
                std::cout << "Number of equations : " << analysis.n_equations() << std::endl;
                std::cout << "Number of steps : " <<  nt << std::endl;
                std::cout << "Time step size : " <<  dt << std::endl;
                
            }
            
        }
        simulation_log << " ******************************* " << std::endl;
        simulation_log << std::endl << std::endl;
    }
    
}

#endif
