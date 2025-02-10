/* 
///////////////////// COMPILATION
export CC=clang-15
export CXX=clang++-15
cmake ../Proton 
(verification that clang is used)
///////////////////// RUN
source /opt/intel/oneapi/setvars.sh intel64
../../../Elliptic_Interface_Poly_Ext -k 0 -l 3 -n 0 -r 0 -c 0 -s 1 -f 1
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cmath>
#include <memory>
#include <sstream>
#include <list>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/SparseExtra>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Eigen/Eigenvalues>

using namespace Eigen;

#include "core/core"
#include "core/solvers"
#include "dataio/silo_io.hpp"
#include "methods/hho"
#include "methods/cuthho"

// Common routines
#include "../../common/preprocessor.hpp"
#include "../../common/postprocessor.hpp"
#include "../../common/newmark_hho_scheme.hpp"
#include "../../common/dirk_hho_scheme.hpp"
#include "../../common/dirk_butcher_tableau.hpp"
#include "../../common/erk_hho_scheme.hpp"
#include "../../common/erk_butcher_tableau.hpp"
#include "../../common/analytical_functions.hpp"

// Agglomeration routines
#include "cutmesh.hpp"
#include "methods.hpp"
#include "test_cases.hpp"

#define scaled_stab_Q 0

// ----- common data types ------------------------------
using RealType = double;
typedef cuthho_poly_mesh<RealType>  mesh_type;

void DEBUG_CutHHOSecondOrderConvTest(int argc, char **argv);

int main(int argc, char **argv) {
    timecounter total_time;
    total_time.tic();
    
    DEBUG_CutHHOSecondOrderConvTest(argc, argv);

    total_time.toc();
    std::cout << std::endl << bold << red << "   TOTAL SIMULATION TIME: " << total_time << std::endl << std::endl;
    return 0;
}

// ./../Elliptic_Interface_Poly_Ext -k 3 -l 4 -r 0 -c 0 -s 1 -f 1 
void DEBUG_CutHHOSecondOrderConvTest(int argc, char **argv) {
    
    timecounter total_simulation_time;

    // ##################################################
    // ################################################## Simulation paramaters 
    // ##################################################
    
    size_t degree        = 1;          // Face degree           -k
    size_t l_divs        = 2;          // Space level refinment -l
    size_t nt_divs       = 1;          // Time level refinment  -n
    size_t int_refsteps  = 4;          // Interface refinment   -r
    bool dump_debug      = false;      // Debug & Silo files    -d 
    bool direct_solver_Q = true;
    bool sc_Q = true;

    int ch;
    while ( (ch = getopt(argc, argv, "k:l:n:r:c:s:f:")) != -1 ) {
        switch(ch) {
            case 'k':
                degree = atoi(optarg);
            break;
            case 'l':
                l_divs = atoi(optarg);
            break;
            case 'n':
                nt_divs = atoi(optarg);
            break;
            case 'r':
                int_refsteps = atoi(optarg);
            break;
            case 'c':
                sc_Q = atoi(optarg);
            break;
            case 's':
                direct_solver_Q = atoi(optarg);
            break;
            case 'f':
                dump_debug = atoi(optarg);
            break;
            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
            exit(1);
        }
    }

    argc -= optind;
    argv += optind;

    std::cout << std::endl << bold << red << "   CONVERGENCE TEST ON SECOND ORDER ELLIPTIC CASE - DEBUG POLYNOMIAL EXTENSION";
    std::cout << std::endl << std::endl << "   SIMULATION PARAMETERS : " << reset << bold << cyan << std::endl;
    std::cout << "      " << "Polynomial degree          -k : " << degree << "     (Face unknowns)"  << std::endl;
    std::cout << "      " << "Space refinement level     -l : " << l_divs << std::endl;
    std::cout << "      " << "Time refinement level      -n : " << nt_divs << std::endl;
    std::cout << "      " << "Interface refinement level -r : " << int_refsteps << std::endl;
    std::cout << "      " << "Static condensation        -c : " << sc_Q << std::endl;
    std::cout << "      " << "Direct solver              -s : " << direct_solver_Q << std::endl;
    std::cout << "      " << "Debug & Silo files         -f : " << dump_debug << std::endl << std::endl;
   

    // // // ################################################## Level set function
    // // HORIZONTAL LINE LEVEL SET FUNCTION: 
    // RealType line_y = 0.5015625; 
    // auto level_set_function = line_level_set<RealType>(line_y);
    // CIRCLE LEVEL SET - LEVEL SET OUTSIDE THE DOMAIN: OK
    RealType radius = 1.0/3.0; // FOR -l 0 and -l 1 ONLY TKO CELLS  
    auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);
    // // FLOWER LEVEL SET 
    // RealType radius = 1.0/3.0; // FOR -l 0 and -l 1 ONLY TKO CELLS  
    // auto level_set_function = flower_level_set<RealType>(radius, 0.5, 0.5, 12, 0.015);
    // auto level_set_function = flower_level_set<RealType>(0.31, 0.5, 0.5, 4, 0.04);
    // // SQUARE LEVEL SET 
    // auto level_set_function = square_level_set<RealType>(0.77, 0.23, 0.23, 0.77);
    // ################################################## Output files
    SparseMatrix<RealType> Kg_centered, Mg_centered;
    std::ofstream error_file_centered(                  "solution_error_file_centered.txt");
    std::ofstream grad_grad_proj_error_file_centered(   "grad_grad_proj_error_file_centered.txt");
    std::ofstream grad_proj_error_file_centered(        "grad_proj_error_file_centered.txt");
    std::ofstream stab_proj_error_file_centered(        "stab_proj_error_file_centered.txt");
    std::ofstream stab_proj_usual_error_file_centered(  "stab_proj_usual_file_centered.txt");
    std::ofstream stab_proj_cut_error_file_centered(    "stab_proj_cut_file_centered.txt");
    std::ofstream stab_proj_illdofs_error_file_centered("stab_proj_ill_dofs_file_centered.txt");
    std::ofstream grad_dofs_error_file_centered(        "grad_dofs_error_file_centered.txt");
    std::ofstream grad_grad_dofs_error_file_centered(   "grad_grad_dofs_error_file_centered.txt");
    std::ofstream stab_dofs_error_file_centered(        "stab_dofs_error_file_centered.txt");
    std::ofstream stab_dofs_usual_error_file_centered(  "stab_dofs_usual_file_centered.txt");
    std::ofstream stab_dofs_cut_error_file_centered(    "stab_dofs_cut_file_centered.txt");
    std::ofstream stab_dofs_illdofs_error_file_centered("stab_dofs_ill_dofs_file_centered.txt");
    for(size_t k = degree; k <= degree; k++) {
        std::cout << bold << red << "   Polynomial degree k : " << k << reset << std::endl;
        error_file_centered << std::endl << "Polynomial degree k : " << k << std::endl;
        grad_proj_error_file_centered << std::endl << "Polynomial degree k : " << k << std::endl;
        grad_grad_proj_error_file_centered << std::endl << "Polynomial degree k : " << k << std::endl;
        stab_proj_error_file_centered << std::endl << "Polynomial degree k : " << k << std::endl;
        stab_proj_usual_error_file_centered << std::endl << "Polynomial degree k : " << k << std::endl;
        stab_proj_cut_error_file_centered << std::endl << "Polynomial degree k : " << k << std::endl;
        stab_proj_illdofs_error_file_centered << std::endl << "Polynomial degree k : " << k << std::endl;
        grad_dofs_error_file_centered << std::endl << "Polynomial degree k : " << k << std::endl;
        grad_grad_dofs_error_file_centered << std::endl << "Polynomial degree k : " << k << std::endl;
        stab_dofs_error_file_centered << std::endl << "Polynomial degree k : " << k << std::endl;
        stab_dofs_usual_error_file_centered << std::endl << "Polynomial degree k : " << k << std::endl;
        stab_dofs_cut_error_file_centered << std::endl << "Polynomial degree k : " << k << std::endl;
        stab_dofs_illdofs_error_file_centered << std::endl << "Polynomial degree k : " << k << std::endl;
        // Mixed order discretization
        hho_degree_info hdi(k+1, k);
        // ################################################## Loop over level of space refinement             
        for(size_t l = 0; l <= l_divs; l++){
            std::cout << bold << cyan << "      Space refinment level -l : " << l << reset << std::endl;
            error_file_centered << "Space refinment level -l : " << l << std::endl;
            grad_proj_error_file_centered    << "Space refinment level -l : " << l << std::endl;
            grad_grad_proj_error_file_centered << "Space refinment level -l : " << l << std::endl;
            stab_proj_error_file_centered         << "Space refinment level -l : " << l << std::endl;
            stab_proj_usual_error_file_centered   << "Space refinment level -l : " << l << std::endl;
            stab_proj_cut_error_file_centered     << "Space refinment level -l : " << l << std::endl;
            stab_proj_illdofs_error_file_centered << "Space refinment level -l : " << l << std::endl;
            grad_dofs_error_file_centered    << "Space refinment level -l : " << l << std::endl;
            grad_grad_dofs_error_file_centered << "Space refinment level -l : " << l << std::endl;
            stab_dofs_error_file_centered         << "Space refinment level -l : " << l << std::endl;
            stab_dofs_usual_error_file_centered   << "Space refinment level -l : " << l << std::endl;
            stab_dofs_cut_error_file_centered     << "Space refinment level -l : " << l << std::endl;
            stab_dofs_illdofs_error_file_centered << "Space refinment level -l : " << l << std::endl;
            // ################################################## Mesh generation 
            mesh_type msh = SquareCutMesh(level_set_function,l,int_refsteps);
            if (dump_debug) 
                output_mesh_info(msh, level_set_function);
            // ################################################## Computation of local Stiff matrices & Assembly  
            auto test_case = make_test_case_laplacian_conv(msh, level_set_function);
            auto method = make_call_methods(msh, 1.0, test_case);
            std::vector<std::pair<size_t,size_t>> cell_basis_data = assembly_poly_extension_centered(msh, hdi, method, test_case, Kg_centered, Mg_centered);
            // ################################################## Solver 
            linear_solver<RealType> analysis;
            analysis.set_Kg(Kg_centered);
            if (direct_solver_Q) 
                analysis.set_direct_solver(true);
            else
                analysis.set_iterative_solver();
            analysis.factorize();
            // // ################################################## RHS assembly 
            auto assembler = make_one_field_interface_assembler(msh, test_case.bcs_fun, hdi);
            assembler.RHS.setZero(); 
            for (auto& cl : msh.cells) {
                auto f = method.make_contrib_rhs_centered(msh, cl, test_case, hdi);
                assembler.assemble_rhs(msh, cl, f);
            }
            // // ################################################## Solving
            Matrix<RealType, Dynamic, 1> x_dof = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows(),1);
            x_dof = analysis.solve(assembler.RHS);
            // // ################################################## Postprocess
            error_file_centered              << "Number of equations : " << analysis.n_equations() << std::endl;
            grad_proj_error_file_centered    << "Number of equations : " << analysis.n_equations() << std::endl;
            grad_grad_proj_error_file_centered << "Number of equations : " << analysis.n_equations() << std::endl;
            stab_proj_error_file_centered         << "Number of equations : " << analysis.n_equations() << std::endl;
            stab_proj_usual_error_file_centered   << "Number of equations : " << analysis.n_equations() << std::endl;
            stab_proj_cut_error_file_centered     << "Number of equations : " << analysis.n_equations() << std::endl;
            stab_proj_illdofs_error_file_centered << "Number of equations : " << analysis.n_equations() << std::endl;
            grad_dofs_error_file_centered    << "Number of equations : " << analysis.n_equations() << std::endl;
            grad_grad_dofs_error_file_centered << "Number of equations : " << analysis.n_equations() << std::endl;
            stab_dofs_error_file_centered         << "Number of equations : " << analysis.n_equations() << std::endl;
            stab_dofs_usual_error_file_centered   << "Number of equations : " << analysis.n_equations() << std::endl;
            stab_dofs_cut_error_file_centered     << "Number of equations : " << analysis.n_equations() << std::endl;
            stab_dofs_illdofs_error_file_centered << "Number of equations : " << analysis.n_equations() << std::endl;
            std::string silo_file_name_sol = "centered_sol_cut_steady_scalar_k_" + std::to_string(k)   + "_l" + std::to_string(l);
            std::string silo_file_name_proj = "centered_proj_cut_steady_scalar_k_" + std::to_string(k) + "_l" + std::to_string(l);
            if (dump_debug) {
                // DEBUGGING ON SOLUTION 
                bool test_op_on_dofs = false;      
                if (test_op_on_dofs) {
                    bool debug_grad = true;
                    if (debug_grad) {
                        // TEST GRADIENT ON DOFS 
                        auto grad_dofs = test_gradient_on_dofs_centered(msh, hdi, method, test_case, x_dof);
                        // postprocessor<cuthho_poly_mesh<RealType>>::write_silo_grad_poly_ext_centered("grad_" + silo_file_name_sol, l, msh, hdi, grad_dofs, test_case, assembler);  
                        postprocessor<cuthho_poly_mesh<RealType>>::compute_errors_grad_one_field_centered(msh, hdi, assembler, grad_dofs, test_case.sol_grad, grad_dofs_error_file_centered); 
                        std::string grad_dofs_error_file_txt = "grad_dofs_error_file_centered.txt";
                        postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(grad_dofs_error_file_txt);
                    }
                    bool debug_stab = true;
                    if (debug_stab) 
                        test_stab_on_dofs_centered(msh, hdi, method, test_case, x_dof, stab_dofs_error_file_centered, stab_dofs_usual_error_file_centered, stab_dofs_cut_error_file_centered, stab_dofs_illdofs_error_file_centered);
                    bool debug_grad_grad = true;
                    if (debug_grad_grad) {
                        auto grad_grad = test_grad_grad(msh, hdi, method, test_case);
                        auto grad_grad_dofs = x_dof.transpose() * grad_grad * x_dof;
                        postprocessor<cuthho_poly_mesh<RealType>>::compute_errors_grad_grad(msh, hdi, grad_grad_dofs, grad_grad_dofs_error_file_centered);   
                        std::string grad_grad_error_file_txt = "grad_grad_dofs_error_file_centered.txt";
                        postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad_grad(grad_grad_error_file_txt);      
                    }
                }
                bool test_op_on_proj = true;
                if (test_op_on_proj) {
                    bool debug_grad = true;
                    if (debug_grad) {
                        auto grad_dofs_proj = test_gradient_on_proj_centered(msh, hdi, method, test_case);
                        // postprocessor<cuthho_poly_mesh<RealType>>::write_silo_grad_poly_ext_centered("grad_" + silo_file_name_proj, l, msh, hdi, grad_dofs_proj, test_case, assembler);  
                        postprocessor<cuthho_poly_mesh<RealType>>::compute_errors_grad_one_field_centered(msh, hdi, assembler, grad_dofs_proj, test_case.sol_grad, grad_proj_error_file_centered);         
                        std::string grad_proj_error_file_txt = "grad_proj_error_file_centered.txt";
                        postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(grad_proj_error_file_txt);
                    }
                    bool debug_stab = true;
                    if (debug_stab) 
                        test_stab_on_proj_centered(msh, hdi, method, test_case, stab_proj_error_file_centered, stab_proj_usual_error_file_centered, stab_proj_cut_error_file_centered, stab_proj_illdofs_error_file_centered);
                    bool debug_grad_grad = true;
                    if (debug_grad_grad) {
                        Matrix<RealType, Dynamic, 1> proj_sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows(),1);
                        assembler.project_over_cells_and_faces(msh, hdi, proj_sol, test_case.sol_fun);
                        auto grad_grad = test_grad_grad(msh, hdi, method, test_case);
                        auto grad_grad_dofs = proj_sol.transpose() * grad_grad * proj_sol;
                        postprocessor<cuthho_poly_mesh<RealType>>::compute_errors_grad_grad(msh, hdi, grad_grad_dofs, grad_grad_proj_error_file_centered);   
                        std::string grad_grad_error_file_txt = "grad_grad_proj_error_file_centered.txt";
                        postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad_grad(grad_grad_error_file_txt);      
                    }
                }
                bool conditioning = false;
                if (conditioning) {    
                    std::string conditioning_file = "conditioning_k_" + std::to_string(k) + "_l_" + std::to_string(l);
                    auto conditioning = test_conditioning(msh, hdi, method, test_case);
                    postprocessor<cuthho_poly_mesh<RealType>>::write_silo_conditioning(conditioning_file, msh, hdi, conditioning, assembler);
                }
                postprocessor<cuthho_poly_mesh<RealType>>::compute_errors_one_field_centered(msh, hdi, assembler, x_dof, test_case.sol_fun, test_case.sol_grad, error_file_centered);
                // postprocessor<cuthho_poly_mesh<RealType>>::write_silo_one_field_poly_ext_centered(silo_file_name_sol, l, msh, hdi, x_dof, test_case, assembler);          
                std::string error_file_txt = "solution_error_file_centered.txt";
                postprocessor<cuthho_poly_mesh<RealType>>::write_conv_sol(error_file_txt);
            }                         
        }
        error_file_centered << std::endl << std::endl;
    }
    error_file_centered.close();
}        
