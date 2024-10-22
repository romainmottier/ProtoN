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
#include "Agglomeration/cutmesh.hpp"
#include "Agglomeration/methods.hpp"
#include "Agglomeration/postpro.hpp"
#include "Agglomeration/test_cases.hpp"

#define scaled_stab_Q 0

// ----- common data types ------------------------------
using RealType = double;
typedef cuthho_poly_mesh<RealType>  mesh_type;

void CutHHOSecondOrderConvTest(int argc, char **argv);

int main(int argc, char **argv) {
    CutHHOSecondOrderConvTest(argc, argv);
    return 0;
}

void CutHHOSecondOrderConvTest(int argc, char **argv){
    
    bool direct_solver_Q = true;
    bool sc_Q = true;
    size_t degree           = 3;
    size_t l_divs           = 6;
    size_t nt_divs          = 0;
    size_t int_refsteps     = 2;
    bool dump_debug         = false;

     int ch;
     while ( (ch = getopt(argc, argv, "k:l:r:n:d")) != -1 ) {
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

    std::ofstream error_file("steady_state_one_field_error.txt");
    
    RealType radius = 1.0/3.0;
    auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);
    
    timecounter tc;
    SparseMatrix<RealType> Kg, Mg;

    for(size_t k = 0; k <= degree; k++){
        std::cout << bold << cyan << "Running an approximation with k : " << k << reset << std::endl;
        error_file << "Approximation with k : " << k << std::endl;
        
        hho_degree_info hdi(k+1, k);
        for(size_t l = 0; l <= l_divs; l++){
        std::cout << bold << cyan << "Running an approximation with l : " << l << reset << std::endl;
        error_file << "Approximation with l : " << l << std::endl;
            
            mesh_type msh = SquareCutMesh(level_set_function,l,int_refsteps);
            if (dump_debug)
            {
                dump_mesh(msh);
                output_mesh_info(msh, level_set_function);
            }
            auto test_case = make_test_case_laplacian_conv(msh, level_set_function);
            auto method = make_call_methods(msh, 1.0, test_case);
            
            std::vector<std::pair<size_t,size_t>> cell_basis_data = create_kg_and_mg_cuthho_interface(msh, hdi, method, test_case, Kg, Mg);
            
            linear_solver<RealType> analysis;
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
            
            if (direct_solver_Q) {
                analysis.set_direct_solver(true);
            }else{
                analysis.set_iterative_solver();
            }
            analysis.factorize();
            
            auto assembler = make_one_field_interface_assembler(msh, test_case.bcs_fun, hdi);
            assembler.RHS.setZero(); // assuming null dirichlet data on boundary.
            for (auto& cl : msh.cells)
            {
                auto f = method.make_contrib_rhs(msh, cl, test_case, hdi);
                assembler.assemble_rhs(msh, cl, f);
            }
            Matrix<RealType, Dynamic, 1> x_dof = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows(),1);
            x_dof = analysis.solve(assembler.RHS);
            error_file << "Number of equations : " << analysis.n_equations() << std::endl;
            if (dump_debug)
            {
                std::string silo_file_name = "cut_steady_scalar_k_" + std::to_string(k) + "_";
                postprocessor<cuthho_poly_mesh<RealType>>::write_silo_one_field(silo_file_name, l, msh, hdi, assembler, x_dof, test_case.sol_fun, false);
            }
            postprocessor<cuthho_poly_mesh<RealType>>::compute_errors_one_field(msh, hdi, assembler, x_dof, test_case.sol_fun, test_case.sol_grad,error_file);
        }
        error_file << std::endl << std::endl;
    }
    error_file.close();
}

