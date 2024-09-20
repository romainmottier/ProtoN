
#ifndef ECutHHOFirstOrderEigenCFL_hpp
#define ECutHHOFirstOrderEigenCFL_hpp

void ECutHHOFirstOrderEigenCFL(int argc, char **argv);

void ECutHHOFirstOrderEigenCFL(int argc, char **argv){
    
    bool report_energy_Q = true;

    size_t degree        = 0;
    size_t l_divs        = 5;
    size_t nt_divs       = 0;
    size_t int_refsteps  = 4;
    bool dump_debug      = false;

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

    RealType t = 0.0;
    int nt_base = nt_divs;
    RealType energy_0 = 0.125;
    std::ofstream simulation_log("acoustic_two_fields_explicit_cfl.txt");

    timecounter tc;
    for(size_t k = 0; k <= degree; k++){
        
        simulation_log << " ******************************* " << std::endl;
        simulation_log << " Polynomial degree =  " << k << std::endl;
        simulation_log << std::endl;
        
        for(size_t l = 4; l <= l_divs; l++){
        
            RealType radius = 1.0/3.0;
            auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);
//            RealType cy = 1.0e-15+0.5;
//            auto level_set_function = line_level_set<RealType>(cy);
            mesh_type msh = SquareCutMesh(level_set_function, l, int_refsteps);
            
            if (dump_debug)
            {
                dump_mesh(msh);
                output_mesh_info(msh, level_set_function);
                PrintAgglomeratedCells(msh);
            }
            
            hho_degree_info hdi(k+1, k);
            SparseMatrix<RealType> Kg, Mg;
            auto test_case = make_test_case_laplacian_waves_mixed(t,msh, level_set_function);
            auto method = make_gradrec_mixed_interface_method(msh, 1.0, test_case);
            size_t n_faces = 0;
            std::vector<std::pair<size_t,size_t>> cell_basis_data = create_mixed_kg_and_mg_cuthho_interface(msh, hdi, method, test_case, Kg, Mg, true, &n_faces);

            size_t n_face_dof, n_face_basis;
            size_t n_dof = Kg.rows();
            size_t n_cell_dof = 0;
            for (auto &chunk : cell_basis_data) {
                n_cell_dof += chunk.second;
            }
            n_face_dof = n_dof - n_cell_dof;
            n_face_basis = face_basis<mesh_type,RealType>::size(degree);
            
            Matrix<RealType, Dynamic, 1> rhs = Matrix<RealType, Dynamic, 1>::Zero(n_dof, 1);
            erk_hho_scheme<RealType> analysis(Kg,rhs,Mg,n_face_dof);
            
//            {
//                {
//                    MatrixXf A = Kg.toDense();
//                    MatrixXf B = Mg.toDense();
//                    GeneralizedEigenSolver<MatrixXd> ges;
//                    ges.compute(A, B, false);
//                    auto lambda_max = ges.alphas()[0]/(ges.betas()[0]);
//                    auto beta_cfl = 1.0/(lambda_max);
//                    std::cout << "Number of equations : " << Kg.rows() << std::endl;
//                    std::cout << "Largest eigenvalue : " << lambda_max << std::endl;
//                    std::cout << "Beta-CFL :  " << beta_cfl << std::endl;
//                }
//            }
            
            analysis.Kcc_inverse_irregular_blocks(cell_basis_data);
            
            RealType lambda_max = 0;
            {
                tc.tic();
//                for (int i = 0; i < n_cell_dof; i++) {
//                    for (int j = 0; j < n_cell_dof; j++) {
//                        Mg.coeffRef(i, j) = analysis.Mc_inv().coeffRef(i,j);
//                    }
//                }
                Kg = analysis.Mc_inv()*Kg;
                
                Spectra::SparseGenMatProd<RealType> op(Kg);
                Spectra::GenEigsSolver< RealType, Spectra::LARGEST_MAGN,
                                        Spectra::SparseGenMatProd<RealType> > max_eigs(&op, 1, 8);
                tc.toc();
                simulation_log << "Generalized Eigen Solver creation time: " << tc << " seconds" << std::endl;
                
                tc.tic();
                max_eigs.init();
                max_eigs.compute();
                tc.toc();
                if(max_eigs.info() == Spectra::SUCCESSFUL){
                    lambda_max = max_eigs.eigenvalues()(0).real();
                }
                simulation_log << "Generalized Eigen Solver compute time: " << tc << " seconds" << std::endl;
                
            }
            
            RealType h_T = std::numeric_limits<RealType>::max();
            for (auto cell : msh.cells) {
             
                 RealType h = diameter(msh, cell);
                 if (h < h_T) {
                     h_T = h;
                 }
            }
                

            
            auto beta_cfl = 1.0/(lambda_max);
            simulation_log << "Number of equations : " << Kg.rows() << std::endl;
            simulation_log << "Largest eigenvalue : " << lambda_max << std::endl;
            simulation_log << "l :  " << l << std::endl;
            simulation_log << "h :  " << h_T << std::endl;
            simulation_log << "Beta-CFL :  " << beta_cfl << std::endl;
            if (scaled_stab_Q) {
                simulation_log << "CFL :  " << beta_cfl/(h_T*h_T) << std::endl;
            }else{
                simulation_log << "CFL :  " << beta_cfl/(h_T) << std::endl;
            }
            
            simulation_log << std::endl;
            simulation_log.flush();
            
        }
    }
    
    simulation_log << " ******************************* " << std::endl;
    simulation_log << std::endl << std::endl;
    
}

#endif
