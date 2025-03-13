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
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
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

void writeMatrixToCSV(const std::string& filename, const Eigen::MatrixXd& matrix) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                file << matrix(i, j);
                if (j < matrix.cols() - 1) 
                file << ",";
            }
            file << "\n"; 
        }
        file.close();
    } 
    else 
    std::cerr << "Impossible d'ouvrir le fichier pour écriture." << std::endl;
}

void CutHHOSecondOrderConvTest(int argc, char **argv);
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
    bool sc_Q = false;

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
    sim_infos << std::endl << "   CONVERGENCE TEST ON SECOND ORDER ELLIPTIC CASE - POLYNOMIAL EXTENSION";
    std::cout << std::endl << bold << red << "   CONVERGENCE TEST ON SECOND ORDER ELLIPTIC CASE - POLYNOMIAL EXTENSION";
    #ifndef subcell_centering
    sim_infos << std::endl << std:: endl << "   POLYNOMIAL BASES CENTERED ON THE WHOLE CELLS";
    std::cout << std::endl << std:: endl << bold << red << "   POLYNOMIAL BASES CENTERED ON THE WHOLE CELLS";
    #else 
    sim_infos << std::endl << std::endl << "   POLYNOMIAL BASES CENTERED ON THE SUBCELLS";
    std::cout << std::endl << std::endl << bold << red << "   POLYNOMIAL BASES CENTERED ON THE SUBCELLS";
    #endif
    sim_infos << std::endl << std::endl << "   SIMULATION PARAMETERS : " << std::endl;
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

    // MATERIAL PROPERTIES
    auto parms = params<T>();
    parms.kappa_1 = 1.0; 
    parms.kappa_2 = 1.0;
    sim_infos << "   Kappa_2                       : " << parms.kappa_2 << std::endl;

    SparseMatrix<RealType> Kg, Mg; 

    // ##################################################
    // ################################################## Loop over polynomial degree
    // ##################################################

    for(size_t k = 3; k <= degree; k++){

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

            // ########## Level set function
            RealType h = 0.1/std::pow(2,l);
            RealType line_y = 0.5015625; 
            RealType p = 0.0;
            RealType radius = 1.0/3.0 + p/32.0;  
            RealType a = 0.5*1e-6;
            RealType b = h/2.0;
            RealType square_min = std::round(0.25/h)*h-a;
            RealType square_max = std::round(0.75/h)*h+a;
            // auto level_set_function = line_level_set<RealType>(line_y);
            // auto level_set_function = square_level_set<RealType>(square_max+2, square_min+2, square_min-b+2, square_max-b+2);
            // auto level_set_function = square_level_set<RealType>(0.70+a, 0.3-a, 0.25, 0.75);
            auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);          
            // auto level_set_function = flower_level_set<RealType>(radius, 0.5, 0.5, 8, 0.03);  
            // auto level_set_function = flower_level_set<RealType>(radius, 0.5, 0.5, 6, 0.045);  
            
            mesh_type msh = MeshGeneration(level_set_function, l, int_refsteps);
            if (dump_debug) {
                output_mesh_info(msh, level_set_function);
                std::string mesh_info = "cuthho_meshinfo_l" + std::to_string(l) + ".silo";
                std::string command = "mv cuthho_meshinfo.silo " + mesh_info;
                std::string interface = "interface_l" + std::to_string(l) + ".3D";
                std::string command2 = "mv interface.3D " + interface;
                std::system(command.c_str());
                std::system(command2.c_str());
            }

            // ##################################################
            // ################################################## Test case & Computation of local Stiff matrices  
            // ##################################################
    
            // HOMOGENEOUS WITHOUT JUMPS - SAME SOLUTION ACROSS THE INTERFACE
            auto test_case = make_test_case_laplacian_sin_sin(msh, level_set_function);

            // (NON) HOMOGENEOUS WITHOUT JUMPS 
            // auto test_case = make_test_case_laplacian_contrast_6(msh, level_set_function, parms);
            
            // (NON) HOMOGENEOUS WITH NEUMANN JUMP WITHOUT DIRICHLET JUMP
            // auto test_case = make_test_case_laplacian_contrast_jump_gN(msh, level_set_function, parms);

            // NON HOMOGENEOUS WITH DIRICHLET JUMPS 
            // auto test_case = make_test_case_laplacian_contrast_jump_gD(msh, level_set_function, parms);

            // HOMOGENEOUS WITH JUMPS 
            // auto test_case = make_test_case_laplacian_jumps_2(msh, level_set_function); 
            // auto test_case = make_test_case_laplacian_jumps_3(msh, level_set_function); 
  
            // ##################################################
            // ################################################## Assembly  
            // ##################################################
           
            auto method = make_gradrec_interface_method(msh, 1.0, test_case);
            auto bcs_fun = test_case.bcs_fun;
            hho_degree_info hdi(k+1, k);
            auto assembler = make_interface_assembler(msh, bcs_fun, hdi);
            
            // SPASITY PROFILES 
            bool sparsity = false;

            std::pair<VecTuple, VecTuple> Pairs = make_pair_KO_pair_OK(msh);
            // Loop on POK subcells 
            for (auto& pair : Pairs.first) { 
                auto cl = msh.cells[std::get<0>(pair)];
                auto contrib = method.make_contrib_POK(msh, pair, test_case, hdi);
                auto lc = contrib.first;
                auto f = contrib.second;
                assembler.assemble_ext(msh, pair, lc, f);  
                if (dump_debug && sparsity)
                    assembler.assemble_sparsity(msh, pair, lc);
            } 
            // Loop on PKO subcells 
            for (auto& pair : Pairs.second) {  
                auto cl = msh.cells[std::get<0>(pair)];
                auto contrib = method.make_contrib_PKO(msh, pair, test_case, hdi);
                auto lc = contrib.first;
                auto f = contrib.second;
                assembler.assemble_ext(msh, pair, lc, f);  
                if (dump_debug && sparsity) 
                    assembler.assemble_sparsity(msh, pair, lc);
            } 
            assembler.finalize();
            Kg = assembler.LHS;

            if (dump_debug && sparsity) {
                auto sparse = assembler.condensed_Kg(msh, assembler.SPARSITY);
                writeMatrixToCSV("LHS_zip.csv", sparse); 
            }

            bool CONDITIONING = false;
            if (dump_debug && CONDITIONING) {
                RealType sigma_max, sigma_min;
                Spectra::SparseSymMatProd<RealType> op(Kg);
                // BIGEST EIGENVALUE
                Spectra::SymEigsSolver< RealType, Spectra::LARGEST_MAGN,Spectra::SparseSymMatProd<RealType> > max_eigs(&op, 1, 100);
                max_eigs.init();
                max_eigs.compute();
                if(max_eigs.info() == Spectra::SUCCESSFUL)
                    sigma_max = max_eigs.eigenvalues()(0);
                // SMALLEST EIGENVALUE
                Spectra::SymEigsSolver< RealType, Spectra::SMALLEST_MAGN, Spectra::SparseSymMatProd<RealType> > min_eigs(&op, 1, 100);
                min_eigs.init();
                min_eigs.compute();
                if(min_eigs.info() == Spectra::SUCCESSFUL)
                    sigma_min = min_eigs.eigenvalues()(0);
                // COMPUTE CONDITION NUMBER
                RealType cond = sigma_max / sigma_min;
                std::cout << bold << yellow << "         Largest eigenvalue: " << sigma_max << reset << std::endl;
                std::cout << bold << yellow << "         Smallest eigenvalue: " << sigma_min << reset << std::endl;
                std::cout << bold << yellow << "         Condition number: " << cond << reset << std::endl;
                error_file << "condition number: " << cond << std::endl;
            }

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
            
            auto errors = postprocessor<cuthho_poly_mesh<RealType>>::compute_error_elliptic_second_order_poly_ext(msh, Pairs.first, hdi, assembler, x_dof, test_case,  previous_h, previous_L2, previous_H1, error_file);
            previous_h  = errors[0]; 
            previous_H1 = errors[1];
            previous_L2 = errors[2];
            
            bool SILO = false;
            bool DEBUG_OPERATORS = false;
            bool GRAD = true;
            bool STAB = true;
            if (dump_debug && (SILO || DEBUG_OPERATORS)) {
                if (SILO) {
                    std::string silo_file_name_sol = "sol_cut_steady_scalar_k_" + std::to_string(k)   + "_l" + std::to_string(l);
                    postprocessor<cuthho_poly_mesh<RealType>>::write_silo_poly_ext(silo_file_name_sol, l, msh, hdi, x_dof, test_case, assembler);  
                }
                if (DEBUG_OPERATORS) {
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
            }
            
            tcl.toc();
            std::cout << bold << yellow << "         Run l = " << l << " completed: " << tcl << " seconds" << reset << std::endl;
            
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

int main(int argc, char **argv) {
    CutHHOSecondOrderConvTest(argc, argv);
    return 0;
}