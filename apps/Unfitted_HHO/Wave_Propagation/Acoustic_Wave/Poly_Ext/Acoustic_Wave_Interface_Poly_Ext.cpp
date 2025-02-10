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
#include "../../../common/postprocessor.hpp"
#include "../../../common/newmark_hho_scheme.hpp"
#include "../../../common/dirk_hho_scheme.hpp"
#include "../../../common/dirk_butcher_tableau.hpp"
#include "../../../common/erk_hho_scheme.hpp"
#include "../../../common/erk_butcher_tableau.hpp"
#include "../../../common/analytical_functions.hpp"

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

template<typename T, typename Function, typename Mesh>
class test_case_laplacian_waves: public test_case_laplacian<T, Function, Mesh>
{
   public:
//    test_case_laplacian_waves(T t,Function level_set__)
//        : test_case_laplacian<T, Function, Mesh>
//        (level_set__, params<T>(),
//         [level_set__,t](const typename Mesh::point_type& pt) -> T { /* sol */
//            if(level_set__(pt) > 0)
//                return t*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
//            else return t*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());},
//         [level_set__,t](const typename Mesh::point_type& pt) -> T { /* rhs */
//             if(level_set__(pt) > 0)
//                 return 2.0*(1.0 + M_PI*M_PI*t*t)*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
//            else return 2.0*(1.0 + M_PI*M_PI*t*t)*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());},
//         [level_set__,t](const typename Mesh::point_type& pt) -> T { // bcs
//             if(level_set__(pt) > 0)
//                return t*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
//            else return t*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());},
//         [level_set__,t](const typename Mesh::point_type& pt) -> auto { // grad
//             Matrix<T, 1, 2> ret;
//             if(level_set__(pt) > 0)
//             {
//                 ret(0) = M_PI*t*t*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
//                 ret(1) = M_PI*t*t*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
//                 return ret;
//             }
//             else {
//                 ret(0) = M_PI*t*t*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
//                 ret(1) = M_PI*t*t*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
//                 return ret;}},
//         [](const typename Mesh::point_type& pt) -> T {/* Null Dir */
//             return 0;},
//         [level_set__](const typename Mesh::point_type& pt) -> T {/* Null Neu */
//             return 0;})
//        {}
    
    test_case_laplacian_waves(T t,Function level_set__)
        : test_case_laplacian<T, Function, Mesh>
        (level_set__, params<T>(),
         [level_set__,t](const typename Mesh::point_type& pt) -> T { /* sol */
            if(level_set__(pt) > 0)
                return (1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::sin(std::sqrt(2.0)*M_PI*t);
            else return (1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::sin(std::sqrt(2.0)*M_PI*t);},
         [level_set__,t](const typename Mesh::point_type& pt) -> T { /* rhs */
            T x,y;
            x = pt.x();
            y = pt.y();
             if(level_set__(pt) > 0)
                 return 2*(x - x*x + y - M_PI*M_PI*(-1 + x)*x*(-1 + y)*y - y*y)*std::sin(std::sqrt(2.0)*M_PI*t);
            else return 2*(x - x*x + y - M_PI*M_PI*(-1 + x)*x*(-1 + y)*y - y*y)*std::sin(std::sqrt(2.0)*M_PI*t);},
         [level_set__,t](const typename Mesh::point_type& pt) -> T { // bcs
            T x,y;
            x = pt.x();
            y = pt.y();
             if(level_set__(pt) > 0)
                return (1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::sin(std::sqrt(2.0)*M_PI*t);
            else return (1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::sin(std::sqrt(2.0)*M_PI*t);},
         [level_set__,t](const typename Mesh::point_type& pt) -> auto { // grad
             Matrix<T, 1, 2> ret;
            T x,y;
            x = pt.x();
            y = pt.y();
             if(level_set__(pt) > 0)
             {
                 ret(0) = (1 - x)*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t) - x*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t);
                 ret(1) = (1 - x)*x*(1 - y)*std::sin(std::sqrt(2.0)*M_PI*t) - (1 - x)*x*y*std::sin(std::sqrt(2.0)*M_PI*t);
                 return ret;
             }
             else {
                 ret(0) = (1 - x)*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t) - x*(1 - y)*y*std::sin(std::sqrt(2.0)*M_PI*t);
                 ret(1) = (1 - x)*x*(1 - y)*std::sin(std::sqrt(2.0)*M_PI*t) - (1 - x)*x*y*std::sin(std::sqrt(2.0)*M_PI*t);
                 return ret;}},
         [](const typename Mesh::point_type& pt) -> T {/* Null Dir */
             return 0;},
         [level_set__](const typename Mesh::point_type& pt) -> T {/* Null Neu */
             return 0;})
        {}
    
//    test_case_laplacian_waves(T t,Function level_set__)
//    : test_case_laplacian<T, Function, Mesh>
//    (level_set__, params<T>(),
//     [level_set__,t](const typename Mesh::point_type& pt) -> T { /* sol */
//        if(level_set__(pt) > 0)
//            return (1.0/(std::sqrt(2.0)*M_PI))*std::sin(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
//        else return (1.0/(std::sqrt(2.0)*M_PI))*std::sin(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());},
//     [level_set__,t](const typename Mesh::point_type& pt) -> T { /* rhs */
//         if(level_set__(pt) > 0)
//             return 0;
//        else return 0;},
//     [level_set__,t](const typename Mesh::point_type& pt) -> T { // bcs
//         if(level_set__(pt) > 0)
//            return (1.0/(std::sqrt(2.0)*M_PI))*std::sin(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
//        else return (1.0/(std::sqrt(2.0)*M_PI))*std::sin(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());},
//     [level_set__,t](const typename Mesh::point_type& pt) -> auto { // grad
//         Matrix<T, 1, 2> ret;
//        T x,y;
//        x = pt.x();
//        y = pt.y();
//         if(level_set__(pt) > 0)
//         {
//             ret(0) = (std::sin(std::sqrt(2)*M_PI*t)*std::cos(M_PI*x)*std::sin(M_PI*y))/std::sqrt(2.0);
//             ret(1) = (std::sin(std::sqrt(2)*M_PI*t)*std::sin(M_PI*x)*std::cos(M_PI*y))/std::sqrt(2.0);
//             return ret;
//         }
//         else {
//             ret(0) = (std::sin(std::sqrt(2)*M_PI*t)*std::cos(M_PI*x)*std::sin(M_PI*y))/std::sqrt(2.0);
//             ret(1) = (std::sin(std::sqrt(2)*M_PI*t)*std::sin(M_PI*x)*std::cos(M_PI*y))/std::sqrt(2.0);
//             return ret;}},
//     [](const typename Mesh::point_type& pt) -> T {/* Null Dir */
//         return 0;},
//     [level_set__](const typename Mesh::point_type& pt) -> T {/* Null Neu */
//         return 0;})
//    {}

};

template<typename Mesh, typename Function>
auto make_test_case_laplacian_waves(double t, const Mesh& msh, Function level_set_function) {
    return test_case_laplacian_waves<typename Mesh::coordinate_type, Function, Mesh>(t,level_set_function);
}

template<typename Mesh, typename testType, typename meth>
void
newmark_step_cuthho_interface(size_t it, double  t, typename Mesh::coordinate_type dt, typename Mesh::coordinate_type beta, typename Mesh::coordinate_type gamma, Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<double, Dynamic, 1> & u_dof_n, Matrix<double, Dynamic, 1> & v_dof_n, Matrix<double, Dynamic, 1> & a_dof_n, SparseMatrix<typename Mesh::coordinate_type> & Kg, linear_solver<typename Mesh::coordinate_type> & analysis) {
    using RealType = typename Mesh::coordinate_type;
    bool write_silo_Q = false;
    auto level_set_function = test_case.level_set_;

    auto rhs_fun = test_case.rhs_fun;
    auto sol_fun = test_case.sol_fun;
    auto sol_grad = test_case.sol_grad;
    auto bcs_fun = test_case.bcs_fun;
    auto dirichlet_jump = test_case.dirichlet_jump;
    auto neumann_jump = test_case.neumann_jump;
    struct params<RealType> parms = test_case.parms;
    
    timecounter tc;
    
    tc.tic();
    auto assembler = make_interface_assembler(msh, bcs_fun, hdi);
    
    if (u_dof_n.rows() == 0) {
        size_t n_dof = assembler.LHS.rows();
        u_dof_n = Matrix<RealType, Dynamic, 1>::Zero(n_dof,1);
        v_dof_n = Matrix<RealType, Dynamic, 1>::Zero(n_dof,1);
        a_dof_n = Matrix<RealType, Dynamic, 1>::Zero(n_dof,1);
        
        RealType t = 0;
        auto u_fun = [&t](const typename Mesh::point_type& pt) -> typename Mesh::coordinate_type {
            return (1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::sin(std::sqrt(2.0)*M_PI*t);
        };
        assembler.project_over_cells_and_faces(msh, hdi, u_dof_n, u_fun);

        auto v_fun = [&t](const typename Mesh::point_type& pt) -> typename Mesh::coordinate_type {
            return std::sqrt(2.0)*M_PI*(1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::cos(std::sqrt(2.0)*M_PI*t);
        };
        assembler.project_over_cells_and_faces(msh, hdi, v_dof_n, v_fun);

        auto a_fun = [&t](const typename Mesh::point_type& pt) -> typename Mesh::coordinate_type {
            return -2*M_PI*M_PI*(1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::sin(std::sqrt(2)*M_PI*t);
        };
        assembler.project_over_cells_and_faces(msh, hdi, a_dof_n, a_fun);
      
        size_t it = 0;
        if(write_silo_Q){
            std::string silo_file_name = "cut_hho_one_field_";
            postprocessor<Mesh>::write_silo_one_field(silo_file_name, it, msh, hdi, assembler, v_dof_n, v_fun, false);
        }
    }
    
    assembler.RHS.setZero(); // assuming null dirichlet data on boundary.
    #ifdef HAVE_INTEL_TBB
            size_t n_cells = msh.cells.size();
            tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
                [&msh,&method,&test_case,&hdi,&assembler] (size_t & cell_ind){
                    auto& cell = msh.cells.at(cell_ind);
                    auto f = method.make_contrib_rhs(msh, cell, test_case, hdi);
                    assembler.assemble_rhs(msh, cell, f);
            }
        );
    #else
        for (auto& cell : msh.cells) {
            auto f = method.make_contrib_rhs(msh, cell, test_case, hdi);
            assembler.assemble_rhs(msh, cell, f);
        }
    #endif

    tc.toc();
    // std::cout << bold << yellow << "RHS assembly: " << tc << " seconds" << reset << std::endl;
    
    // Compute intermediate state for scalar and rate
    u_dof_n = u_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
    v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
    Matrix<RealType, Dynamic, 1> res = Kg*u_dof_n;
    assembler.RHS -= res;
    
    tc.tic();
    a_dof_n = analysis.solve(assembler.RHS); // new acceleration
    tc.toc();
    // std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    // update scalar and rate
    u_dof_n += beta*dt*dt*a_dof_n;
    v_dof_n += gamma*dt*a_dof_n;
    
    if(write_silo_Q){
        std::string silo_file_name = "cut_hho_one_field_";
        postprocessor<Mesh>::write_silo_one_field(silo_file_name, it, msh, hdi, assembler, u_dof_n, sol_fun, false);
    }
    
}

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
                // std::cout << "wrong arguments" << std::endl;
            exit(1);
        }
    }

    argc -= optind;
    argv += optind;

    std::cout << std::endl << bold << red << "   CONVERGENCE TEST ON SECOND ORDER ELLIPTIC CASE - DEBUG POLYNOMIAL EXTENSION";
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
    
    SparseMatrix<RealType> Kg, Kg_c, Mg;

    std::string error_file_txt = "solution_error_file_centered.txt";
    std::ofstream error_file(error_file_txt);
    postprocessor<cuthho_poly_mesh<RealType>>::write_conv_sol(error_file_txt);

    // ##################################################
    // ################################################## Loop over polynomial degree
    // ##################################################

    for(size_t k = degree; k <= degree; k++){

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
            // ################################################## Time discretization 
            // ##################################################
            size_t nt = 10;
            for (unsigned int i = 0; i < nt_divs; i++) 
                nt *= 2;
            RealType ti = 0.0;
            RealType tf = 1.0;
            RealType dt = (tf-ti)/nt;
            RealType t = ti;

            RealType beta = 0.25;
            RealType gamma = 0.5;

            // ##################################################
            // ################################################## Computation of local Stiff matrices  
            // ##################################################

            auto test_case = make_test_case_laplacian_waves(t,msh, level_set_function);
            auto method = make_gradrec_interface_method(msh, 1.0, test_case);

            // ##################################################
            // ################################################## Assembly  
            // ##################################################

            auto bcs_fun = test_case.bcs_fun;
            hho_degree_info hdi(k+1, k);
            auto assembler = make_interface_assembler(msh, bcs_fun, hdi);
            std::pair<VecTuple,VecTuple> Pairs = make_pair_KO_pair_OK(msh);
            // Loop on POK subcells 
            for (auto& pair : Pairs.first) { 
                auto cl = msh.cells[std::get<0>(pair)];
                auto contrib = method.make_contrib_POK(msh, pair, test_case, hdi);
                auto lc = contrib.first;
                auto f = contrib.second*0.0;
                assembler.assemble_ext(msh, pair, lc, f);  
            } 
            // Loop on PKO subcells 
            for (auto& pair : Pairs.second) {  
                auto cl = msh.cells[std::get<0>(pair)];
                auto contrib = method.make_contrib_PKO(msh, pair, test_case, hdi);
                auto lc = contrib.first;
                auto f = contrib.second*0.0;
                assembler.assemble_ext(msh, pair, lc, f);  
            } 
            assembler.finalize();
            Kg = assembler.LHS;
            Kg_c = Kg;
            for (auto& cell : msh.cells) {
                auto cell_mass = method.make_contrib_mass(msh, cell, test_case, hdi);
                size_t n_dof = assembler.n_dof(msh,cell);
                Matrix<RealType, Dynamic, Dynamic> mass = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof,n_dof);
                mass.block(0,0,cell_mass.rows(),cell_mass.cols()) = cell_mass;
                assembler.assemble_mass(msh, cell, mass);
            }
            Mg = assembler.MASS;

            // ##################################################
            // ################################################## Solver  
            // ##################################################

            linear_solver<RealType> analysis;
            Kg *= beta*(dt*dt);
            Kg += Mg;
            analysis.set_Kg(Kg);
            if (direct_solver_Q) 
                analysis.set_direct_solver(true);
            else
                analysis.set_iterative_solver();
            analysis.factorize();

            std::ofstream sensor_1_log("s1_cut_acoustic_one_field.csv");
            std::ofstream sensor_2_log("s2_cut_acoustic_one_field.csv");
            std::ofstream sensor_3_log("s3_cut_acoustic_one_field.csv");
            typename mesh_type::point_type s1_pt(1.0/3.0, 1.0/3.0);
            typename mesh_type::point_type s2_pt(1.0/3.0, 2.0/3.0);
            typename mesh_type::point_type s3_pt(1.2, 1.0);
            std::pair<typename mesh_type::point_type,size_t> s1_pt_cell = std::make_pair(s1_pt, -1);
            std::pair<typename mesh_type::point_type,size_t> s2_pt_cell = std::make_pair(s2_pt, -1);
            std::pair<typename mesh_type::point_type,size_t> s3_pt_cell = std::make_pair(s3_pt, -1);
    
            // Projecting initial scalar, velocity and acceleration
            Matrix<RealType, Dynamic, 1> u_dof_n, v_dof_n, a_dof_n;
            for(size_t it = 1; it <= nt; it++) { 
                RealType t = dt*it+ti;
                int mod = static_cast<int>(std::round(nt / 15.0)); // Number of silo files 
                if ((it == 1) || (it == std::round(nt/3)) || (it == std::round(nt/2)) || (it == std::round(3*nt/2)) || (it == nt)) 
                    std::cout << "Time step number" << it << " : " << t << "seconds" << std::endl; 
                auto test_case = make_test_case_laplacian_waves(t,msh, level_set_function);
                auto method = make_gradrec_interface_method(msh, 1.0, test_case);
                newmark_step_cuthho_interface(it, t, dt, beta, gamma, msh, hdi, method, test_case, u_dof_n,  v_dof_n, a_dof_n, Kg_c, analysis);
                if (it == nt) {     
                    // auto errors = postprocessor<cuthho_poly_mesh<RealType>>::compute_error_elliptic_second_order_poly_ext(msh, Pairs.first, hdi, assembler, u_dof_n, test_case.sol_fun, test_case.sol_grad, previous_h, previous_L2, previous_H1, error_file);
                    // previous_h  = errors[0]; 
                    // previous_H1 = errors[1];
                    // previous_L2 = errors[2];  
                    // postprocessor<cuthho_poly_mesh<RealType>>::compute_errors_one_field_bis(msh, hdi, assembler, u_dof_n, test_case.sol_fun, test_case.sol_grad, error_file);
                    std::cout << "Number of equations : " << analysis.n_equations() << std::endl;
                    std::cout << "Number of steps : " <<  nt << std::endl;
                    std::cout << "Time step size : " <<  dt << std::endl;
                }
            }
        }
    }  
}

