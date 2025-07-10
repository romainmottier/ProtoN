/* 
///////////////////// COMPILATION
export CC=clang-15
export CXX=clang++-15
cmake ../Proton 
(verification that clang is used)
///////////////////// RUN
source /opt/intel/oneapi/setvars.sh intel64
// ../Elliptic_Interface_Agglo -k 2 -l 6 -r 10 -c 1 -s 1 -f 1
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

// Agglomeration routines
#include "operators.hpp"
#include "cutmesh.hpp"

#define scaled_stab_Q 0

// ----- common data types ------------------------------
using RealType = double;
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
void CutHHOSecondOrderConvTest (int argc, char **argv) {
    
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

    std::cout << std::endl << bold << red << "   CONVERGENCE TEST ON SECOND ORDER ELLIPTIC CASE - AGGLOMERATION";
    std::cout << std::endl << std::endl << "   SIMULATION PARAMETERS : " << reset << bold << cyan << std::endl;
    std::cout << "   " << "Polynomial degree          -k : " << degree << "     (Face unknowns)"  << std::endl;
    std::cout << "   " << "Space refinement level     -l : " << l_divs << std::endl;
    std::cout << "   " << "Time refinement level      -n : " << nt_divs << std::endl;
    std::cout << "   " << "Interface refinement level -r : " << int_refsteps << std::endl;
    std::cout << "   " << "Static condensation        -c : " << sc_Q << std::endl;
    std::cout << "   " << "Direct solver              -s : " << direct_solver_Q << std::endl;
    std::cout << "   " << "Debug & Silo files         -f : " << dump_debug << std::endl;

    // ##################################################
    // ################################################## Space discretization
    // ##################################################
    
    SparseMatrix<RealType> Kg, Mg;

    std::string error_file_txt = "solution_error_file.txt";
    std::ofstream error_file(error_file_txt);
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_sol(error_file_txt);

    // ##################################################
    // ################################################## Loop over polynomial degree
    // ##################################################

    for(size_t k = 0; k <= degree; k++) {

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
        for(size_t l = 0; l <= l_divs; l++) {

            tcl.tic();
            std::cout << bold << cyan << "      Space refinment level -l : " << l << reset << std::endl;
            error_file << "Space refinment level -l : " << l << std::endl;
        
            // ##################################################
            // ################################################## Mesh generation 
            // ##################################################

            // ########## Level set function
            RealType h = 0.1/std::pow(2,l);
            RealType line_y = 0.5015625; 
            RealType p = 4.0;
            RealType radius = 1.0/3.0 + p/32.0;  
            RealType a = 5e-3;
            RealType b = h/2.0;
            RealType square_min = std::round(0.25/h)*h-a;
            RealType square_max = std::round(0.75/h)*h+a;
            // auto level_set_function = line_level_set<RealType>(line_y);
            // auto level_set_function = square_level_set<RealType>(square_max, square_min, square_min-b, square_max-b);
            auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);          
            // auto level_set_function = flower_level_set<RealType>(radius, 0.5, 0.5, 8, 0.03);  
            // auto level_set_function = flower_level_set<RealType>(radius, 0.5, 0.5, 6, 0.045);  

            mesh_type msh = MeshGeneration(level_set_function, l, int_refsteps);
            if (dump_debug) 
                output_mesh_info(msh, level_set_function); 

            // ##################################################
            // ################################################## Computation of local Stiff matrices  
            // ##################################################
            
            // MATERIAL PROPERTIES
            auto parms = params<T>();
            parms.kappa_1 = 1.0; 
            parms.kappa_2 = 1.0;
            
            // TEST CASES
            auto test_case = make_test_case_laplacian_sin_sin(msh, level_set_function);
            // auto test_case = make_test_case_laplacian_contrast_6(msh, level_set_function, parms);
            // auto test_case = make_test_case_laplacian_contrast_jump_gN(msh, level_set_function, parms);
            
            auto method = make_gradrec_interface_method(msh, 1.0, test_case);

            // ##################################################
            // ################################################## Assembly  
            // ##################################################

            // SPASITY PROFILES 
            bool sparsity = false;

            auto bcs_fun = test_case.bcs_fun;
            hho_degree_info hdi(k+1, k);
            auto assembler = make_interface_assembler(msh, bcs_fun, hdi);
            for (auto& cl : msh.cells) {
                auto contrib = method.make_contrib(msh, cl, test_case, hdi);
                auto lc = contrib.first;
                auto f = contrib.second;
                assembler.assemble(msh, cl, lc, f);
                if (dump_debug && sparsity)
                    assembler.assemble_sparsity(msh, cl, lc);
            }
            assembler.finalize();
            Kg = assembler.LHS;

            if (dump_debug && sparsity) {
                auto sparse = assembler.condensed_Kg(msh, assembler.SPARSITY);
                writeMatrixToCSV("LHS_zip.csv", sparse); 
            }

            bool CONDITIONING = false;
            if (dump_debug && CONDITIONING) {
                // LARGEST EIGENVALUE
                RealType sigma_max = 0.0; 
                Spectra::SparseSymMatProd<RealType> op(Kg);
                // Spectra::SymEigsSolver< RealType, Spectra::LARGEST_MAGN, Spectra::SparseSymMatProd<RealType> > max_eigs(&op, 1, 200);
                Spectra::SymEigsSolver< RealType, Spectra::LARGEST_ALGE, Spectra::SparseSymMatProd<RealType> > max_eigs(&op, 1, 200);
                max_eigs.init();
                // max_eigs.compute(Spectra::LARGEST_MAGN, 5000, 1e-5);
                max_eigs.compute(Spectra::LARGEST_ALGE, 2000, 1e-10);
                if (max_eigs.info() == Spectra::SUCCESSFUL) {
                    sigma_max = max_eigs.eigenvalues()(0);
                }
                else {
                    std::cout << "SEARCHING FOR THE MAXIMAL EIGENVALUE FAILED" << std::endl;
                }
                // SMALLEST EIGENVALUE
                RealType sigma_min = 0.0;
                Spectra::SparseSymMatProd<RealType> op_min(Kg);
                // Spectra::SymEigsSolver< RealType, Spectra::SMALLEST_MAGN, Spectra::SparseSymMatProd<RealType> > min_eigs(&op_min, 1, 200);
                Spectra::SymEigsSolver< RealType, Spectra::SMALLEST_ALGE, Spectra::SparseSymMatProd<RealType> > min_eigs(&op_min, 1, 200);
                min_eigs.init();
                // min_eigs.compute(Spectra::SMALLEST_MAGN, 2000, 1e-10);
                min_eigs.compute(Spectra::SMALLEST_ALGE, 2000, 1e-10);
                if (min_eigs.info() == Spectra::SUCCESSFUL) {
                    sigma_min = min_eigs.eigenvalues()(0);
                }
                else {
                    std::cout << "SEARCHING FOR THE MINIMAL EIGENVALUE FAILED" << std::endl;
                }
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

            auto errors = postprocessor<cuthho_poly_mesh<RealType>>::compute_error_elliptic_second_order_agglo(msh, hdi, assembler, x_dof, test_case.sol_fun, test_case.sol_grad, previous_h, previous_L2, previous_H1, error_file);
            previous_h  = errors[0]; 
            previous_H1 = errors[1];
            previous_L2 = errors[2]; 

            tcl.toc();
            std::cout << bold << yellow << "         Run l = " << l << " completed: " << tcl << " seconds" << reset << std::endl;

        }
    
        error_file << std::endl << std::endl;
        tck.toc();
        std::cout << bold << cyan << "      Run k = " << k << " completed: " << tck << " seconds" << reset << std::endl;

    }
    
    error_file.close();
    tc.toc();
    std::cout << std::endl << bold << red << "   Run completed: " << tc << " seconds" << reset << std::endl << std::endl;

}


int main(int argc, char **argv) {
    CutHHOSecondOrderConvTest(argc, argv);
    return 0;
}