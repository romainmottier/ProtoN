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

void CutHHOSecondOrderConvTest(int argc, char **argv);

int main(int argc, char **argv) {
    CutHHOSecondOrderConvTest(argc, argv);
    return 0;
}

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

            mesh_type msh = MeshGeneration(level_set_function, l, int_refsteps);
            if (dump_debug) {
                dump_mesh(msh);
                output_mesh_info(msh, level_set_function); 
            }

            // ##################################################
            // ################################################## Computation of local Stiff matrices  
            // ##################################################
            
            // MATERIAL PROPERTIES
            auto parms = params<T>();
            parms.kappa_1 = 1.0; 
            parms.kappa_2 = 10000.0;
            
            // TEST CASES
            auto test_case = make_test_case_laplacian_sin_sin(msh, level_set_function);
            // auto test_case = make_test_case_laplacian_contrast_6(msh, level_set_function, parms);
            // auto test_case = make_test_case_laplacian_contrast_jump_gN(msh, level_set_function, parms);
            
            auto method = make_gradrec_interface_method(msh, 1.0, test_case);

            // ##################################################
            // ################################################## Assembly  
            // ##################################################

            auto bcs_fun = test_case.bcs_fun;
            hho_degree_info hdi(k+1, k);
            auto assembler = make_interface_assembler(msh, bcs_fun, hdi);
            for (auto& cl : msh.cells) {
                auto contrib = method.make_contrib(msh, cl, test_case, hdi);
                auto lc = contrib.first;
                auto f = contrib.second;
                assembler.assemble(msh, cl, lc, f);
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

