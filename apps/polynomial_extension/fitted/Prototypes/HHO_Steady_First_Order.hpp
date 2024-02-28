#ifndef HHOSteadyFirstOrder_hpp
#define HHOSteadyFirstOrder_hpp

void HHOSteadyFirstOrder(int argc, char **argv);

void HHOSteadyFirstOrder(int argc, char **argv){

  std::cout << std::endl;
  std::cout << bold << red << " HHO FOR STEADY DIFFUSION PROBLEM" << reset << std::endl;
  std::cout << std::endl;
  
  using RealType = double;   
  size_t degree  = 0;        // Poynomial degree of the method 
  size_t n_divs  = 0;        // Mesh level of refinement
  
  /////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////// Inputs
  /////////////////////////////////////////////////////////
  
  int opt;
  while ( (opt = getopt(argc, argv, "k:l:n")) != -1 ){
    switch(opt){         
    case 'k':
      {
	degree = atoi(optarg);
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
  
  std::cout << bold << cyan << "   • Input parameters" << reset << std::endl;
  std::cout << bold << green << "      ";
  std::cout << bold << "- Mesh refinement level : l = " << n_divs << std::endl;
  std::cout << "      ";
  std::cout << "- Polynomial degree of the method : k = " << degree << std::endl;
  std::cout << std::endl;
  
  
  /////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////// Convergence test
  /////////////////////////////////////////////////////////
  
  std::cout << bold << cyan;
  std::cout << "   • Convergence test for all polynomial degrees up to " << degree << " :";
  std::cout << std::endl << std::endl;
  
  // Loop over polynomial degrees 
  for(size_t k = 0; k <= degree; k++){
    
    std::cout << bold << green << "      ";
    std::cout << "- Polynomial degree : k = " << k << std::endl;
    
    // Creating HHO approximation spaces and corresponding linear operator
    // Mixed order : k+1, k / Equal Order : k, k
    hho_degree_info hho_di(k+1,k);
    
    ////////////////////////////////////////
    //////////////////////////////////////// Loop over level of mesh refinement 
    ////////////////////////////////////////
    
    for(size_t l = 0; l <= n_divs; l++){
      
      
      
      ////////////////////////////////////////////////// Mesh generation 
      std::cout << bold << yellow << "           ";
      std::cout << "* Mesh refinement : l = " << l << reset << std::endl;
      mesh_init_params<RealType> mip;
      mip.Nx = 1;
      mip.Ny = 1;
      for (size_t i = 0; i < n_divs; i++) {
	mip.Nx *= 2;
	mip.Ny *= 2;
      }
      timecounter tc;
      tc.tic();
      poly_mesh<RealType> msh(mip);
      tc.toc();
      std::cout << "                ";
      std::cout  << "Mesh generation: " << tc << " seconds" << reset << std::endl;
      
      
      
      ////////////////////////////////////////////////// Manufactured solution 
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
      
      
      
      ////////////////////////////////////////////////// Solving a HDG/HHO mixed problem
      auto assembler = make_assembler(msh, hho_di, true); 
      tc.tic();
      
      
      
      ////////////////////////////////////////////////// Loop over cells
      for (auto& cell : msh.cells){
	auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
	auto stabilization_operator  = make_hho_naive_stabilization(msh, cell, hho_di);
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
      
      ////////////////////////////////////////////////// Computing errors
      tc.tic();      
      RealType scalar_l2_error = 0.0;
      RealType flux_l2_error   = 0.0;
      size_t cell_i            = 0;

      //////////////////// Loop over cells
      for (auto& cell : msh.cells){

	if(cell_i == 0){
	  RealType h = diameter(msh, cell);
	  std::cout << "                " << "h size = " << h << std::endl;
	}
	  
	size_t cell_scal_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree());
	size_t cell_flux_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree()+1)-1;
	size_t cell_dof = cell_scal_dof + cell_flux_dof;

	// scalar evaluation : pression
	{
	  Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof+cell_flux_dof, 0, cell_scal_dof, 1);
	  Matrix<RealType, Dynamic, Dynamic> mass      = make_mass_matrix(msh, cell, hho_di.cell_degree());
	  Matrix<RealType, Dynamic, 1> rhs             = make_rhs(msh, cell, hho_di.cell_degree(), exact_scal_sol_fun, 2);
	  Matrix<RealType, Dynamic, 1> real_dofs       = mass.llt().solve(rhs);
	  Matrix<RealType, Dynamic, 1> diff            = real_dofs - scalar_cell_dof;
	  scalar_l2_error += diff.dot(mass*diff);
	}
	
	// flux evaluation : vitesse 
	auto int_rule = integrate(msh, cell, 2*(hho_di.cell_degree()+1));
	cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree()+1);
	Matrix<RealType, Dynamic, 1> flux_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_flux_dof, 1);


	//////////////////////////////////////////// Loop over integration weight
	for (auto & point_pair : int_rule) {
	  
	  RealType omega = point_pair.second;
          
	  auto t_dphi = cell_basis.eval_gradients(point_pair.first);
	  Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();
	  for (size_t i = 1; i < t_dphi.rows(); i++){
	    grad_uh = grad_uh + flux_cell_dof(i-1)*t_dphi.block(i, 0, 1, 2);
	  }
	  
	  Matrix<RealType, 1, 2> grad_u_exact = Matrix<RealType, 1, 2>::Zero();
	  grad_u_exact(0,0) =  exact_flux_sol_fun(point_pair.first)[0];
	  grad_u_exact(0,1) =  exact_flux_sol_fun(point_pair.first)[1];
	  flux_l2_error += omega * (grad_u_exact - grad_uh).dot(grad_u_exact - grad_uh);
	}
	
	cell_i++;
      }
      
      std::cout << "                " << "scalar L2-norm error = " << std::sqrt(scalar_l2_error) << std::endl;
      std::cout << "                " << "flux L2-norm error   = " << std::sqrt(flux_l2_error) << std::endl;
      tc.toc();
      std::cout << "                " << "Error completed: " << tc << " seconds" << reset << std::endl;
      
      size_t it = 0;
      std::string silo_file_name = "scalar_mixed_";
      RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof, exact_scal_sol_fun, exact_flux_sol_fun);
      
    }
    std::cout << std::endl;
  }
  
  return;
  
}


#endif
