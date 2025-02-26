/* 
///////////////////// COMPILATION
export CC=clang-15
export CXX=clang++-15
cmake ../Proton 
(verification that clang is used)
///////////////////// RUN
source /opt/intel/oneapi/setvars.sh intel64
../../Elliptic_Interface_Poly_Ext -k 3 -l 4 -r 0 -c 0 -s 1 -f 1
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
#include "../../common/postprocessor.hpp"
#include "../../common/newmark_hho_scheme.hpp"
#include "../../common/dirk_hho_scheme.hpp"
#include "../../common/dirk_butcher_tableau.hpp"
#include "../../common/erk_hho_scheme.hpp"
#include "../../common/erk_butcher_tableau.hpp"
#include "../../common/analytical_functions.hpp"

// Extension routines
#include "operators.hpp"
#include "cutmesh.hpp"
#include "debug.hpp"

#define scaled_stab_Q 0

// ----- common data types ------------------------------
using RealType = double;
using VecTuple = std::vector<std::tuple<double,element_location,std::vector<double>>>;
typedef cuthho_poly_mesh<RealType>  mesh_type;

void CutHHOSecondOrderConvTest(int argc, char **argv);
void CutHHOSecondOrderConvTest_DEBUG(int argc, char **argv);

int main(int argc, char **argv) {
    // CutHHOSecondOrderConvTest(argc, argv);
    CutHHOSecondOrderConvTest_DEBUG(argc, argv);
    return 0;
}


void CutHHOSecondOrderConvTest(int argc, char **argv) {
    
    timecounter tc, tck, tcl;
    tc.tic();

    using T = double;

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
    while ( (ch = getopt(argc, argv, "k:l:n:r:c:s:v:f:")) != -1 ) {
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

    std::ofstream sim_infos("simulation_infos.txt");
    sim_infos << std::endl << "   CONVERGENCE TEST ON SECOND ORDER ELLIPTIC CASE - DEBUG POLYNOMIAL EXTENSION";
    std::cout << std::endl << bold << red << "   CONVERGENCE TEST ON SECOND ORDER ELLIPTIC CASE - DEBUG POLYNOMIAL EXTENSION";
    sim_infos << std::endl << std::endl << "   SIMULATION PARAMETERS : " << reset << std::endl;
    std::cout << std::endl << std::endl << "   SIMULATION PARAMETERS : " << reset << bold << cyan << std::endl;
    sim_infos << "   " << "Polynomial degree          -k : " << degree << "     (Face unknowns)"  << std::endl;
    std::cout << "   " << "Polynomial degree          -k : " << degree << "     (Face unknowns)"  << std::endl;
    sim_infos << "   " << "Space refinement level     -l : " << l_divs << std::endl;
    std::cout << "   " << "Space refinement level     -l : " << l_divs << std::endl;
    sim_infos << "   " << "Time refinement level      -n : " << nt_divs << std::endl;
    std::cout << "   " << "Time refinement level      -n : " << nt_divs << std::endl;
    sim_infos << "   " << "Interface refinement level -r : " << int_refsteps << std::endl;
    std::cout << "   " << "Interface refinement level -r : " << int_refsteps << std::endl;
    sim_infos << "   " << "Static condensation        -c : " << sc_Q << std::endl;
    std::cout << "   " << "Static condensation        -c : " << sc_Q << std::endl;
    sim_infos << "   " << "Direct solver              -s : " << direct_solver_Q << std::endl;
    std::cout << "   " << "Direct solver              -s : " << direct_solver_Q << std::endl;
    sim_infos << "   " << "Debug & Silo files         -f : " << dump_debug << std::endl;
    std::cout << "   " << "Debug & Silo files         -f : " << dump_debug << std::endl;

    // ##################################################
    // ################################################## Level set function
    // ##################################################

    RealType line_y = 0.5015625; 
    RealType radius = 1.0/3.0;  
    // auto level_set_function = line_level_set<RealType>(line_y);
    // auto level_set_function = square_level_set<RealType>(0.77, 0.23, 0.23, 0.77);
    auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);          
    // auto level_set_function = flower_level_set<RealType>(radius, 0.5, 0.5, 8, 0.03);            

    // ##################################################
    // ################################################## Space discretization
    // ##################################################
    
    SparseMatrix<RealType> Kg, Mg;

    // POSTPRO HHO SOLUTION
    std::string error_file_txt = "solution_error_file.txt";
    std::ofstream error_file(error_file_txt);
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_sol(error_file_txt);

    // ##################################################
    // ################################################## Loop over polynomial degree
    // ##################################################

    for(size_t k = 0; k <= degree; k++){

        tck.tic();
        std::cout << std::endl << bold << red << "   Polynomial degree k : " << k << reset << std::endl;
        error_file << std::endl << "Polynomial degree k : " << k << std::endl;
                  
        // Mixed order discretization
        hho_degree_info hdi(k+1, k);

        // ##################################################
        // ################################################## Loop over level of space refinement 
        // ##################################################

        T previous_H1 = 0.0;
        T previous_L2 = 0.0;
        T previous_h = 0.0;

        for(size_t l = 0; l <= l_divs; l++){

            tcl.tic();
            std::cout << bold << cyan << "      Space refinment level -l : " << l << reset << std::endl;
            error_file << "Space refinment level -l : " << l << std::endl;

            // ##################################################
            // ################################################## Mesh generation 
            // ##################################################

            mesh_type msh = MeshGeneration(level_set_function, l, int_refsteps);
            if (dump_debug) 
                output_mesh_info(msh, level_set_function);

            // ##################################################
            // ################################################## Computation of local Stiff matrices  
            // ##################################################

            auto test_case = make_test_case_laplacian_sin_sin(msh, level_set_function);
            auto method = make_gradrec_interface_method(msh, 1.0, test_case);

            // ##################################################
            // ################################################## Assembly  
            // ##################################################
           
            auto bcs_fun = test_case.bcs_fun;
            hho_degree_info hdi(k+1, k);
            auto assembler = make_interface_assembler(msh, bcs_fun, hdi);
            
            std::pair<VecTuple, VecTuple> Pairs = make_pair_KO_pair_OK(msh);
            // Loop on POK subcells 
            for (auto& pair : Pairs.first) { 
                auto cl = msh.cells[std::get<0>(pair)];
                auto contrib = method.make_contrib_POK(msh, pair, test_case, hdi);
                auto lc = contrib.first;
                auto f = contrib.second;
                assembler.assemble_ext(msh, pair, lc, f);  
            } 
            // Loop on PKO subcells 
            for (auto& pair : Pairs.second) {  
                auto cl = msh.cells[std::get<0>(pair)];
                auto contrib = method.make_contrib_PKO(msh, pair, test_case, hdi);
                auto lc = contrib.first;
                auto f = contrib.second;
                assembler.assemble_ext(msh, pair, lc, f);  
            } 
            assembler.finalize();
            Kg = assembler.LHS;
            
            // ##################################################
            // ################################################## Solver  
            // ##################################################
            
            linear_solver<RealType> analysis;
            analysis.set_Kg(Kg);
            if (direct_solver_Q) 
                analysis.set_direct_solver(true);
            else
                analysis.set_iterative_solver();
            analysis.factorize();

            Matrix<RealType, Dynamic, 1> x_dof = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows(),1);
            x_dof = analysis.solve(assembler.RHS);
            
            // ##################################################
            // ################################################## Postprocess  
            // ##################################################
            
            auto errors = postprocessor<cuthho_poly_mesh<RealType>>::compute_error_elliptic_second_order_poly_ext(msh, Pairs.first, hdi, assembler, x_dof, test_case.sol_fun, test_case.sol_grad, previous_h, previous_L2, previous_H1, error_file);
            previous_h  = errors[0]; 
            previous_H1 = errors[1];
            previous_L2 = errors[2];
            
            if (dump_debug) {
                bool SILO = false;
                if (SILO) {
                    std::string silo_file_name_sol = "sol_cut_steady_scalar_k_" + std::to_string(k)   + "_l" + std::to_string(l);
                    postprocessor<cuthho_poly_mesh<RealType>>::write_silo_poly_ext(silo_file_name_sol, l, msh, hdi, x_dof, test_case, assembler);  
                }
            }
            
            tcl.toc();
            std::cout << bold << yellow << "         Run l = " << l << " completed: " << tcl << " seconds" << reset << std::endl;
            
        }

        error_file << std::endl << std::endl;
        tck.toc();
        std::cout << bold << cyan << "      Run k = " << k << " completed: " << tck << " seconds" << reset << std::endl;

    }
    
    error_file.close();
    tc.toc();
    std::cout << std::endl << bold << red << "   Run completed: " << tc << " seconds" << reset << std::endl;

}


void CutHHOSecondOrderConvTest_DEBUG(int argc, char **argv) {
    
    timecounter tc, tck, tcl;
    tc.tic();

    using T = double;

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
    while ( (ch = getopt(argc, argv, "k:l:n:r:c:s:v:f:")) != -1 ) {
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

    std::ofstream sim_infos("simulation_infos.txt");
    sim_infos << std::endl << "   CONVERGENCE TEST ON SECOND ORDER ELLIPTIC CASE - DEBUG POLYNOMIAL EXTENSION";
    std::cout << std::endl << bold << red << "   CONVERGENCE TEST ON SECOND ORDER ELLIPTIC CASE - DEBUG POLYNOMIAL EXTENSION";
    sim_infos << std::endl << std::endl << "   SIMULATION PARAMETERS : " << reset << std::endl;
    std::cout << std::endl << std::endl << "   SIMULATION PARAMETERS : " << reset << bold << cyan << std::endl;
    sim_infos << "   " << "Polynomial degree          -k : " << degree << "     (Face unknowns)"  << std::endl;
    std::cout << "   " << "Polynomial degree          -k : " << degree << "     (Face unknowns)"  << std::endl;
    sim_infos << "   " << "Space refinement level     -l : " << l_divs << std::endl;
    std::cout << "   " << "Space refinement level     -l : " << l_divs << std::endl;
    sim_infos << "   " << "Time refinement level      -n : " << nt_divs << std::endl;
    std::cout << "   " << "Time refinement level      -n : " << nt_divs << std::endl;
    sim_infos << "   " << "Interface refinement level -r : " << int_refsteps << std::endl;
    std::cout << "   " << "Interface refinement level -r : " << int_refsteps << std::endl;
    sim_infos << "   " << "Static condensation        -c : " << sc_Q << std::endl;
    std::cout << "   " << "Static condensation        -c : " << sc_Q << std::endl;
    sim_infos << "   " << "Direct solver              -s : " << direct_solver_Q << std::endl;
    std::cout << "   " << "Direct solver              -s : " << direct_solver_Q << std::endl;
    sim_infos << "   " << "Debug & Silo files         -f : " << dump_debug << std::endl;
    std::cout << "   " << "Debug & Silo files         -f : " << dump_debug << std::endl;          

    // ##################################################
    // ################################################## Space discretization
    // ##################################################
    
    // POSTPRO HHO SOLUTION
    std::string error_file_txt = "solution_error_file.txt";
    std::ofstream error_file(error_file_txt);
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_sol(error_file_txt);

    // POSTPRO GRADIENT RECONSTRUCTION
    std::string grad_proj_error_file_txt = "grad_proj_error_file.txt";
    std::ofstream grad_proj_error_file(grad_proj_error_file_txt);
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(grad_proj_error_file_txt);

    // POSTPRO STABILIZATION
    std::string stab_proj_error_file_txt = "stab_proj_error_file.txt";
    std::string stab_proj_usual_file_txt = "stab_proj_usual_file.txt";
    std::string stab_proj_cut_file_txt = "stab_proj_cut_file.txt";
    std::string stab_proj_ill_dofs_file_txt = "stab_proj_ill_dofs_file.txt";
    std::ofstream stab_proj_error_file(stab_proj_error_file_txt);
    std::ofstream stab_proj_usual_error_file(stab_proj_usual_file_txt);
    std::ofstream stab_proj_cut_error_file(stab_proj_cut_file_txt);
    std::ofstream stab_proj_ill_dofs_error_file(stab_proj_ill_dofs_file_txt);

    // POSTPRO GRAD.GRAD
    std::string grad_grad_proj_error_file_txt = "grad_grad_proj_error_file.txt";
    std::ofstream grad_grad_proj_error_file(grad_grad_proj_error_file_txt);
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(grad_grad_proj_error_file_txt);

    // ##################################################
    // ################################################## Level set function
    // ##################################################

    RealType line_y = 0.5015625; 
    RealType radius = 1.0/3.0;  
    // auto level_set_function = line_level_set<RealType>(line_y);
    // auto level_set_function = square_level_set<RealType>(0.77, 0.23, 0.23, 0.77);
    auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);          
    // auto level_set_function = flower_level_set<RealType>(radius, 0.5, 0.5, 8, 0.03);  

    SparseMatrix<RealType> Kg, Mg;

    // ##################################################
    // ################################################## Loop over polynomial degree
    // ##################################################

    for(size_t k = 0; k <= degree; k++){

        tck.tic();
        std::cout << std::endl << bold << red << "   Polynomial degree k : " << k << reset << std::endl;
        error_file << std::endl << "Polynomial degree k : " << k << std::endl;
        grad_proj_error_file << std::endl << "Polynomial degree k : " << k << std::endl;
        stab_proj_error_file << "Polynomial degree k : " << k << std::endl;
        stab_proj_usual_error_file << "Polynomial degree k : " << k << std::endl;
        stab_proj_cut_error_file << "Polynomial degree k : " << k << std::endl;
        stab_proj_ill_dofs_error_file << "Polynomial degree k : " << k << std::endl;
                            
        // Mixed order discretization
        hho_degree_info hdi(k+1, k);

        // ##################################################
        // ################################################## Loop over level of space refinement 
        // ##################################################

        T previous_H1 = 0.0;
        T previous_L2 = 0.0;
        T previous_h = 0.0;

        for(size_t l = 0; l <= l_divs; l++){

            tcl.tic();
            std::cout << bold << cyan << "      Space refinment level -l : " << l << reset << std::endl;
            error_file << "Space refinment level -l : " << l << std::endl;
            grad_proj_error_file << "Space refinment level -l : " << l << std::endl;
            stab_proj_error_file << "Space refinment level -l : " << l << std::endl;
            stab_proj_usual_error_file << "Space refinment level -l : " << l << std::endl;
            stab_proj_cut_error_file << "Space refinment level -l : " << l << std::endl;
            stab_proj_ill_dofs_error_file << "Space refinment level -l : " << l << std::endl;

            // ##################################################
            // ################################################## Mesh generation 
            // ##################################################

            mesh_type msh = MeshGeneration(level_set_function, l, int_refsteps);
            if (dump_debug) 
                output_mesh_info(msh, level_set_function);

            // ##################################################
            // ################################################## Test case & Computation of local Stiff matrices  
            // ##################################################
            
            // HOMOGENEOUS WITHOUT JUMPS - SAME SOLUTION ACROSS THE INTERFACE
            // auto test_case = make_test_case_laplacian_sin_sin(msh, level_set_function);

            // (NON) HOMOGENEOUS WITHOUT JUMPS 
            auto parms = params<T>();
            parms.kappa_1 = 1.0;
            parms.kappa_2 = 10000.0;
            auto test_case = make_test_case_laplacian_contrast_6(msh, level_set_function, parms);
            
            // HOMOGENEOUS WITH JUMPS 
            // auto test_case = make_test_case_laplacian_jumps_2(msh, level_set_function); 
            // auto test_case = make_test_case_laplacian_jumps_3(msh, level_set_function); 
            
            // NON HOMOGENEOUS WITHOUT JUMPS 
            // auto parms = params<T>();
            // parms.kappa_1 = 1.0;
            // parms.kappa_2 = 1.0;
            // auto test_case = make_test_case_laplacian_contrast_2(msh, level_set_function, parms);
                        
            // NON HOMOGENEOUS WITH JUMPS 

            // ##################################################
            // ################################################## Assembly  
            // ##################################################
           
            auto method = make_gradrec_interface_method(msh, 1.0, test_case);
            auto bcs_fun = test_case.bcs_fun;
            hho_degree_info hdi(k+1, k);
            auto assembler = make_interface_assembler(msh, bcs_fun, hdi);
            
            bool DEBUG_OPERATORS = false;
            bool RUN_HHO = true;
            if (DEBUG_OPERATORS) {
                bool GRAD = true;
                bool STAB = true;
                if (GRAD) {
                    auto grad_dofs_proj = test_gradient_on_proj(msh, hdi, method, test_case);
                    postprocessor<cuthho_poly_mesh<RealType>>::compute_errors_grad_one_field(msh, hdi, assembler, grad_dofs_proj, test_case.sol_grad, grad_proj_error_file);         
                }
                if (STAB) {
                    test_stab_on_proj(msh, hdi, method, test_case, stab_proj_error_file, stab_proj_usual_error_file, stab_proj_cut_error_file, stab_proj_ill_dofs_error_file);
                    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(stab_proj_error_file_txt);
                    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(stab_proj_usual_file_txt);
                    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(stab_proj_cut_file_txt);
                    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_grad(stab_proj_ill_dofs_file_txt);
                }
            }
            if (RUN_HHO) {
                std::pair<VecTuple, VecTuple> Pairs = make_pair_KO_pair_OK(msh);
                // Loop on POK subcells 
                for (auto& pair : Pairs.first) { 
                    auto cl = msh.cells[std::get<0>(pair)];
                    auto contrib = method.make_contrib_POK(msh, pair, test_case, hdi);
                    auto lc = contrib.first;
                    auto f = contrib.second;
                    assembler.assemble_ext(msh, pair, lc, f);  
                } 
                // Loop on PKO subcells 
                for (auto& pair : Pairs.second) {  
                    auto cl = msh.cells[std::get<0>(pair)];
                    auto contrib = method.make_contrib_PKO(msh, pair, test_case, hdi);
                    auto lc = contrib.first;
                    auto f = contrib.second;
                    assembler.assemble_ext(msh, pair, lc, f);  
                } 
                assembler.finalize();
                Kg = assembler.LHS;
                
                // ##################################################
                // ################################################## Solver  
                // ##################################################
                
                linear_solver<RealType> analysis;
                analysis.set_Kg(Kg);
                if (direct_solver_Q) 
                    analysis.set_direct_solver(true);
                else
                    analysis.set_iterative_solver();
                analysis.factorize();

                Matrix<RealType, Dynamic, 1> x_dof = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows(),1);
                x_dof = analysis.solve(assembler.RHS);

                // ##################################################
                // ################################################## Postprocess  
                // ##################################################
                
                auto errors = postprocessor<cuthho_poly_mesh<RealType>>::compute_error_elliptic_second_order_poly_ext(msh, Pairs.first, hdi, assembler, x_dof, test_case.sol_fun, test_case.sol_grad, previous_h, previous_L2, previous_H1, error_file);
                previous_h  = errors[0]; 
                previous_H1 = errors[1];
                previous_L2 = errors[2];
                
                if (dump_debug) {
                    bool SILO = false;
                    if (SILO) {
                        std::string silo_file_name_sol = "sol_cut_steady_scalar_k_" + std::to_string(k)   + "_l" + std::to_string(l);
                        postprocessor<cuthho_poly_mesh<RealType>>::write_silo_poly_ext(silo_file_name_sol, l, msh, hdi, x_dof, test_case, assembler);  
                    }
                }

                tcl.toc();
                std::cout << bold << yellow << "         Run l = " << l << " completed: " << tcl << " seconds" << reset << std::endl;

            }
        }

        error_file << std::endl << std::endl;
        grad_proj_error_file << std::endl << std::endl;
        stab_proj_error_file << std::endl << std::endl;
        stab_proj_usual_error_file << std::endl << std::endl;
        stab_proj_cut_error_file << std::endl << std::endl;
        stab_proj_ill_dofs_error_file << std::endl << std::endl;
        tck.toc();
        std::cout << bold << cyan << "      Run k = " << k << " completed: " << tck << " seconds" << reset << std::endl;

    }
    
    error_file.close();
    grad_proj_error_file.close();
    stab_proj_error_file.close();
    stab_proj_usual_error_file.close();
    stab_proj_cut_error_file.close();
    stab_proj_ill_dofs_error_file.close();
    tc.toc();
    std::cout << std::endl << bold << red << "   Run completed: " << tc << " seconds" << reset << std::endl << std::endl;

}

