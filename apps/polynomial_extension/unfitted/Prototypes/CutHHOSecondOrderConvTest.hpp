
#ifndef CutHHOSecondOrderConvTest_hpp
#define CutHHOSecondOrderConvTest_hpp

void CutHHOSecondOrderConvTest(int argc, char **argv);

void CutHHOSecondOrderConvTest(int argc, char **argv){
    
  // ##################################################
  // ################################################## Simulation paramaters 
  // ##################################################

    bool direct_solver_Q = true;
    bool sc_Q = true;
    size_t degree        = 1;          // Face degree           -k
    size_t l_divs        = 2;          // Space level refinment -l
    size_t nt_divs       = 1;          // Time level refinment  -n
    size_t int_refsteps  = 4;          // Interface refinment   -r
    bool dump_debug      = false;      // Debug & Silo files    -d 

    int ch;
    while ( (ch = getopt(argc, argv, "k:l:r:n:d")) != -1 ) {
      switch(ch) {
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

    std::cout << std::endl << "   ";
    std::cout << bold << red << "CONVERGENCE TEST ON SECOND ORDER ELLIPTIC CASE " << reset;
    std::cout << std::endl << std::endl << "   ";
    std::cout << bold << red << "SIMULATION PARAMETERS : " << reset;
    std::cout << bold << cyan << std::endl;
    std::cout << "      ";
    std::cout << bold << "Polynomial degree          -k : " << degree << "     (Face unknowns)"  << std::endl;
    std::cout << "      ";
    std::cout << bold << "Space refinement level     -l : " << l_divs << std::endl;
    std::cout << "      ";
    std::cout << bold << "Time refinement level      -n : " << nt_divs << std::endl;
    std::cout << "      ";
    std::cout << bold << "Interface refinement level -r : " << int_refsteps << std::endl;
    std::cout << "      ";
    std::cout << bold << "Debug & Silo files         -d : " << dump_debug << reset << std::endl << std::endl;

  // ##################################################
  // ################################################## Level set function
  // ##################################################
    
    // Circle level set function
    RealType radius = 1.0/3.0;
    RealType xc = 0.5;
    RealType yc = 0.5;
    auto level_set_function = circle_level_set<RealType>(radius, xc, yc);

    // Line level set function
    // RealType line_y = 0.425;
    // auto level_set_function = line_level_set<RealType>(line_y);
    
  // ##################################################
  // ################################################## Space discretization
  // ##################################################

    timecounter tc;
    SparseMatrix<RealType> Kg, Mg;

    std::ofstream error_file("steady_state_two_fields_error.txt");

    // ##################################################
    // ################################################## Loop over polynomial degree
    // ##################################################

    for(size_t k = degree; k <= degree; k++){

        std::cout << std::endl; 
        std::cout << bold << red << "   Polynomial degree k : " << k << reset << std::endl;
        error_file << std::endl << "Polynomial degree k : " << k << std::endl;
        
        // Mixed order discretization
        hho_degree_info hdi(k+1, k);

        // ##################################################
        // ################################################## Loop over level of space refinement 
        // ##################################################

        for(size_t l = l_divs; l <= l_divs; l++){

            std::cout << bold << cyan << "      Space refinment level -l : " << l << reset << std::endl;

            // ##################################################
            // ################################################## Mesh generation 
            // ##################################################
            mesh_type msh = SquareCutMesh(level_set_function,l,int_refsteps);
            if (dump_debug) {
                dump_mesh(msh);
                output_mesh_info(msh, level_set_function);
            }
            auto test_case = make_test_case_laplacian_conv(msh, level_set_function);
            auto method = make_gradrec_interface_method(msh, 1.0, test_case);
            
            // std::vector<std::pair<size_t,size_t>> cell_basis_data = create_kg_and_mg_cuthho_interface(msh, hdi, method, test_case, Kg, Mg);
            std::vector<std::pair<size_t,size_t>> cell_basis_data = test_operators(msh, hdi, method, test_case, Kg, Mg);

            return;
            // // // // ##################################################
            // // // // ################################################## Static condensation
            // // // // ##################################################
            // // // linear_solver<RealType> analysis;
            // // // if (sc_Q) {
            // // //   size_t n_dof = Kg.rows();
            // // //   size_t n_cell_dof = 0;
            // // //   for (auto &chunk : cell_basis_data) {
            // // //     n_cell_dof += chunk.second;
            // // //   }
            // // //   size_t n_face_dof = n_dof - n_cell_dof;
            // // //   analysis.set_Kg(Kg, n_face_dof);
            // // //   analysis.condense_equations_irregular_blocks(cell_basis_data);
            // // // }
            // // // else {
            // // //   analysis.set_Kg(Kg);
            // // // }
            
            // // // // ##################################################
            // // // // ################################################## Solver
            // // // // ##################################################
            // // // if (direct_solver_Q) {
            // // //   analysis.set_direct_solver(true);
            // // // }
            // // // else {
            // // //   analysis.set_iterative_solver();
            // // // }
            // // // analysis.factorize();
            
            // // // // ##################################################
            // // // // ################################################## Assembly and loop over cells
            // // // // ##################################################
            // // // auto assembler = make_one_field_interface_assembler(msh, test_case.bcs_fun, hdi);
            // // // assembler.RHS.setZero(); // assuming null dirichlet data on boundary.
            // // // // Loop over cells 
            // // // for (auto& cl : msh.cells) {
            // // //   auto f = method.make_contrib_rhs(msh, cl, test_case, hdi);
            // // //   assembler.assemble_rhs(msh, cl, f);
            // // // }
            // // // Matrix<RealType, Dynamic, 1> x_dof = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows(),1);
            // // // x_dof = analysis.solve(assembler.RHS);

            // // // // ##################################################
            // // // // ################################################## postprocess
            // // // // ##################################################
            // // // error_file << "Number of equations : " << analysis.n_equations() << std::endl;
            // // // if (dump_debug) {
            // // //   std::string silo_file_name = "cut_steady_scalar_k_" + std::to_string(k) + "_";
            // // //   postprocessor<cuthho_poly_mesh<RealType>>::write_silo_one_field(silo_file_name, l, msh, hdi, assembler, x_dof, test_case.sol_fun, false);
            // // // }
            // // // postprocessor<cuthho_poly_mesh<RealType>>::compute_errors_one_field_extended(msh, hdi, assembler, x_dof, test_case.sol_fun, test_case.sol_grad, error_file);
        }
        // error_file << std::endl << std::endl;
    }
    // error_file.close();
    std::cout << std::endl;
}


#endif
