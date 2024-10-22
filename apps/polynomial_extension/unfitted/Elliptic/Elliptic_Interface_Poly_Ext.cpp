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


#include "../../common/preprocessor.hpp"
#include "../../common/postprocessor.hpp"
#include "../../common/newmark_hho_scheme.hpp"
#include "../../common/dirk_hho_scheme.hpp"
#include "../../common/dirk_butcher_tableau.hpp"
#include "../../common/erk_hho_scheme.hpp"
#include "../../common/erk_butcher_tableau.hpp"
#include "../../common/analytical_functions.hpp"

#define scaled_stab_Q 0

// ----- common data types ------------------------------
using RealType = double;
typedef cuthho_poly_mesh<RealType>  mesh_type;

template<typename T, size_t ET, typename testType>
class interface_method
{
    using Mat  = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

protected:
    interface_method(){}

    virtual std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType &test_case, const hho_degree_info hdi) = 0;
    
    virtual Vect
    make_contrib_rhs_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType &test_case, const hho_degree_info hdi) = 0;

public:
    std::pair<Mat, Vect>
    make_contrib_uncut(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType &test_case)
    {
        T kappa;
        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
            kappa = test_case.parms.kappa_1;
        else
            kappa = test_case.parms.kappa_2;

        auto gr = make_hho_gradrec_vector(msh, cl, hdi);
        Mat stab = make_hho_naive_stabilization(msh, cl, hdi);
        Mat lc = kappa * (gr.second + stab);
        Mat f = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
        
//        std::cout << "r = " << gr.second << std::endl;
//        std::cout << "s = " << stab << std::endl;
//        std::cout << "f = " << f << std::endl;
        
        return std::make_pair(lc, f);
    }
    
    Vect
    make_contrib_rhs_uncut(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType &test_case)
    {
        Mat f = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
        return f;
    }

    std::pair<Mat, Vect>
    make_contrib(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const testType &test_case, const hho_degree_info hdi)
    {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return make_contrib_uncut(msh, cl, hdi, test_case);
        else // on interface
            return make_contrib_cut(msh, cl, test_case, hdi);
    }
    
    Vect
    make_contrib_rhs(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const testType &test_case, const hho_degree_info hdi)
    {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return make_contrib_rhs_uncut(msh, cl, hdi, test_case);
        else // on interface
            return make_contrib_rhs_cut(msh, cl, test_case, hdi);
    }
    
    Mat
    make_contrib_mass(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const testType &test_case, const hho_degree_info hdi)
    {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return make_contrib_uncut_mass(msh, cl, hdi, test_case);
        else // on interface
            return make_contrib_cut_mass(msh, cl, hdi, test_case);
    }

    Mat
    make_contrib_uncut_mass(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType &test_case)
    {
        T c;
        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
            c = test_case.parms.c_1;
        else
            c = test_case.parms.c_2;
        Mat mass = make_mass_matrix(msh, cl, hdi.cell_degree());
        mass *= (1.0/(c*c*test_case.parms.kappa_1));
        return mass;
    }
    
    Mat
    make_contrib_cut_mass(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType &test_case)
    {

        Mat mass_neg = make_mass_matrix(msh, cl,
                                        hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
        Mat mass_pos = make_mass_matrix(msh, cl,
                                        hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
        mass_neg *= (1.0/(test_case.parms.c_1*test_case.parms.c_1*test_case.parms.kappa_1));
        mass_pos *= (1.0/(test_case.parms.c_2*test_case.parms.c_2*test_case.parms.kappa_2));
        
        size_t n_data_neg = mass_neg.rows();
        size_t n_data_pos = mass_pos.rows();
        size_t n_data = n_data_neg + n_data_pos;
        
        Mat mass = Mat::Zero(n_data,n_data);
        mass.block(0,0,n_data_neg,n_data_neg) = mass_neg;
        mass.block(n_data_neg,n_data_neg,n_data_pos,n_data_pos) = mass_pos;

        return mass;
    }
    
};

template<typename T, size_t ET, typename testType>
class gradrec_interface_method : public interface_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    gradrec_interface_method(T eta_)
        : interface_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType &test_case, const hho_degree_info hdi)
    {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        ///////////////    LHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        // GR
        T factor = 0.0;
        if (1.0/(parms.kappa_1) < 1.0/(parms.kappa_2)) {
            factor = 1.0;
        }
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_NEGATIVE_SIDE, 1.0-factor);
        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_POSITIVE_SIDE, factor);

        // stab
        auto stab_parms = test_case.parms;
        stab_parms.kappa_1 = 1.0/(parms.kappa_1);// rho_1 = kappa_1
        stab_parms.kappa_2 = 1.0/(parms.kappa_2);// rho_2 = kappa_2
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, stab_parms);
        
        T penalty_scale = std::min(1.0/(parms.kappa_1), 1.0/(parms.kappa_2));
        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) += penalty_scale* penalty;
        stab.block(0, cbs, cbs, cbs) -= penalty_scale * penalty;
        stab.block(cbs, 0, cbs, cbs) -= penalty_scale * penalty;
        stab.block(cbs, cbs, cbs, cbs) += penalty_scale * penalty;
        
//        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, parms);
//
//        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
//        stab.block(0, 0, cbs, cbs) += parms.kappa_1 * penalty;
//        stab.block(0, cbs, cbs, cbs) -= parms.kappa_1 * penalty;
//        stab.block(cbs, 0, cbs, cbs) -= parms.kappa_1 * penalty;
//        stab.block(cbs, cbs, cbs, cbs) += parms.kappa_1 * penalty;


        Mat lc = stab + stab_parms.kappa_1 * gr_n.second + stab_parms.kappa_2 * gr_p.second;
        
        ///////////////    RHS
        Vect f = Vect::Zero(lc.rows());
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);
        // we use element_location::IN_POSITIVE_SIDE to get rid of the Nitsche term
        // (see definition of make_Dirichlet_jump)
        f.head(cbs) -= parms.kappa_1 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);
        f.block(cbs, 0, cbs, 1) += parms.kappa_1 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);
        f.block(cbs, 0, cbs, 1)
            += make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                    test_case.neumann_jump);


        // rhs term with GR
        auto gbs = vector_cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.grad_degree());
        vector_cell_basis<cuthho_poly_mesh<T>, T> gb( msh, cl, hdi.grad_degree() );
        Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero( gbs );
        auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(),
                                        element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : iqps)
        {
            const auto g_phi    = gb.eval_basis(qp.first);
            const Matrix<T,2,1> n      = level_set_function.normal(qp.first);

            F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
        }
        f -= F_bis.transpose() * (parms.kappa_1 * gr_n.first );

        return std::make_pair(lc, f);
    }

    Vect
    make_contrib_rhs_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType &test_case, const hho_degree_info hdi)
    {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        ///////////////    RHS
        Vect f = Vect::Zero(cbs*2);
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);
//        // we use element_location::IN_POSITIVE_SIDE to get rid of the Nitsche term
//        // (see definition of make_Dirichlet_jump)
//        f.head(cbs) -= parms.kappa_1 *
//            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
//                                level_set_function, dir_jump, eta);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);
//        f.block(cbs, 0, cbs, 1) += parms.kappa_1 *
//            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
//                                level_set_function, dir_jump, eta);
//        f.block(cbs, 0, cbs, 1)
//            += make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
//                                    test_case.neumann_jump);

        return f;
    }
};

template<typename T, size_t ET, typename testType>
auto make_gradrec_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                   testType test_case)
{
    return gradrec_interface_method<T, ET, testType>(eta_);
}

template<typename Mesh, typename testType, typename meth>
std::vector<std::pair<size_t,size_t>>
create_kg_and_mg_cuthho_interface(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg);

///// test_case_laplacian_conv
template<typename T, typename Function, typename Mesh>
class test_case_laplacian_conv: public test_case_laplacian<T, Function, Mesh>
{
   public:
    
    test_case_laplacian_conv(Function level_set__)
    : test_case_laplacian<T, Function, Mesh>
    (level_set__, params<T>(),
     [level_set__](const typename Mesh::point_type& pt) -> T { /* sol */
        if(level_set__(pt) > 0)
            return std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
        else return std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());},
     [level_set__](const typename Mesh::point_type& pt) -> T { /* rhs */
         if(level_set__(pt) > 0)
             return 2.0*(M_PI*M_PI)*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
        else return 2.0*(M_PI*M_PI)*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());},
     [level_set__](const typename Mesh::point_type& pt) -> T { // bcs
         if(level_set__(pt) > 0)
            return std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
        else return std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());},
     [level_set__](const typename Mesh::point_type& pt) -> auto { // grad
         Matrix<T, 1, 2> ret;
         if(level_set__(pt) > 0)
         {
             ret(0) = M_PI*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
             ret(1) = M_PI*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
             return ret;
         }
         else {
             ret(0) = M_PI*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
             ret(1) = M_PI*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
             return ret;}},
     [](const typename Mesh::point_type& pt) -> T {/* Null Dir */
         return 0;},
     [level_set__](const typename Mesh::point_type& pt) -> T {/* Null Neu */
         return 0;})
    {}
    
//    test_case_laplacian_conv(Function level_set__)
//    : test_case_laplacian<T, Function, Mesh>
//    (level_set__, params<T>(),
//     [level_set__](const typename Mesh::point_type& pt) -> T { /* sol */
//        if(level_set__(pt) > 0)
//            return (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();
//        else return (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();},
//     [level_set__](const typename Mesh::point_type& pt) -> T { /* rhs */
//        T x,y;
//        x = pt.x();
//        y = pt.y();
//         if(level_set__(pt) > 0)
//             return -2.0*((x - 1)*x + (y - 1)*y);
//        else return -2.0*((x - 1)*x + (y - 1)*y);},
//     [level_set__](const typename Mesh::point_type& pt) -> T { // bcs
//         if(level_set__(pt) > 0)
//            return (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();
//        else return (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();},
//     [level_set__](const typename Mesh::point_type& pt) -> auto { // grad
//         Matrix<T, 1, 2> ret;
//        T x,y;
//        x = pt.x();
//        y = pt.y();
//         if(level_set__(pt) > 0)
//         {
//             ret(0) = (1 - x)*(1 - y)*y - x*(1 - y)*y;
//             ret(1) = (1 - x)*x*(1 - y) - (1 - x)*x*y;
//             return ret;
//         }
//         else {
//             ret(0) = (1 - x)*(1 - y)*y - x*(1 - y)*y;
//             ret(1) = (1 - x)*x*(1 - y) - (1 - x)*x*y;
//             return ret;}},
//     [](const typename Mesh::point_type& pt) -> T {/* Null Dir */
//         return 0;},
//     [level_set__](const typename Mesh::point_type& pt) -> T {/* Null Neu */
//         return 0;})
//    {}
    
};

template<typename Mesh, typename Function>
auto make_test_case_laplacian_conv(const Mesh& msh, Function level_set_function)
{
    return test_case_laplacian_conv<typename Mesh::coordinate_type, Function, Mesh>(level_set_function);
}

template<typename Mesh>
void PrintIntegrationRule(const Mesh& msh, hho_degree_info & hdi);

template<typename Mesh>
void PrintAgglomeratedCells(const Mesh& msh);

mesh_type SquareCutMesh(level_set<RealType> & level_set_function, size_t l_divs, size_t int_refsteps = 4);
mesh_type SquareGar6moreCutMesh(level_set<RealType> & level_set_function, size_t l_divs, size_t int_refsteps);

void CutMesh(mesh_type & msh, level_set<RealType> & level_set_function, size_t int_refsteps, bool agglomerate_Q = true);

void CutHHOSecondOrderConvTest(int argc, char **argv);

int main(int argc, char **argv) {

     CutHHOSecondOrderConvTest(argc, argv);
    return 0;
}

mesh_type SquareCutMesh(level_set<RealType> & level_set_function, size_t l_divs, size_t int_refsteps){
    
    mesh_init_params<RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;
    l_divs += 1;
    for (unsigned int i = 0; i < l_divs; i++) {
      mip.Nx *= 2;
      mip.Ny *= 2;
    }

    timecounter tc;

    tc.tic();
    mesh_type msh(mip);
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;

    CutMesh(msh,level_set_function,int_refsteps, true);
    return msh;
}

void CutMesh(mesh_type & msh, level_set<RealType> & level_set_function, size_t int_refsteps, bool agglomerate_Q){
    
    timecounter tc;
    tc.tic();
    detect_node_position(msh, level_set_function); // ok
    detect_cut_faces(msh, level_set_function); // it could be improved
    detect_cut_cells(msh, level_set_function);
    
    if (agglomerate_Q) {
        detect_cell_agglo_set(msh, level_set_function);
        make_neighbors_info_cartesian(msh);
        refine_interface(msh, level_set_function, int_refsteps);
        make_agglomeration(msh, level_set_function);
    }else{
        refine_interface(msh, level_set_function, int_refsteps);
    }
    
    tc.toc();
    std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;
}

void CutHHOSecondOrderConvTest(int argc, char **argv){
    
    bool direct_solver_Q = true;
    bool sc_Q = true;
    size_t degree           = 0;
    size_t l_divs          = 0;
    size_t nt_divs       = 0;
    size_t int_refsteps     = 4;
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
            auto method = make_gradrec_interface_method(msh, 1.0, test_case);
            
            std::vector<std::pair<size_t,size_t>> cell_basis_data = create_kg_and_mg_cuthho_interface(msh, hdi, method, test_case, Kg, Mg);
            
            std::cout << "JUST BEFORE STATIC CONDENSATION " << std::endl;

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

            std::cout << "JUST BETWEEN STATIC CONDENSATION AND FACORISZATION" << std::endl;
            
            if (direct_solver_Q) {
                analysis.set_direct_solver(true);
            }else{
                analysis.set_iterative_solver();
            }
            analysis.factorize();
            
            std::cout << "JUST AFTER FACORISZATION" << std::endl;

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

template<typename Mesh, typename testType, typename meth>
std::vector<std::pair<size_t,size_t>>
create_kg_and_mg_cuthho_interface(const Mesh& msh, hho_degree_info & hdi, meth &method, testType & test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg){
    
    using RealType = typename Mesh::coordinate_type;

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
    auto assembler = make_one_field_interface_assembler(msh, bcs_fun, hdi);
    std::vector<std::pair<size_t,size_t>> cell_basis_data = assembler.compute_cell_basis_data(msh);
    size_t cell_ind = 0;
    for (auto& cell : msh.cells) {
        auto contrib = method.make_contrib(msh, cell, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;

        auto cell_mass = method.make_contrib_mass(msh, cell, test_case, hdi);
        size_t n_dof = assembler.n_dof(msh,cell);
        Matrix<RealType, Dynamic, Dynamic> mass = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof,n_dof);
        mass.block(0,0,cell_mass.rows(),cell_mass.cols()) = cell_mass;
        assembler.assemble(msh, cell, lc, f);
        assembler.assemble_mass(msh, cell, mass);
        cell_ind++;
    }
    assembler.finalize();
    
    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

    Kg = assembler.LHS;
    Mg = assembler.MASS;


    return cell_basis_data;

}

template<typename Mesh>
void PrintIntegrationRule(const Mesh& msh, hho_degree_info & hdi){
    
    std::ofstream int_rule_file("cut_integration_rule.txt");
    for (auto& cl : msh.cells)
    {
        cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        auto cbs = cb.size();
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto fbs = face_basis<cuthho_poly_mesh<RealType>,RealType>::size(hdi.face_degree());
        
        Matrix<RealType, Dynamic, 1> locdata_n, locdata_p, locdata;
        Matrix<RealType, Dynamic, 1> cell_dofs_n, cell_dofs_p, cell_dofs;

        if (location(msh, cl) == element_location::ON_INTERFACE)
        {
            
            auto qps_n = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                int_rule_file << qp.first.x() << " " << qp.first.y() << std::endl;
            }
            
            
            auto qps_p = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                int_rule_file << qp.first.x() << " " << qp.first.y() << std::endl;
            }
        }
        else
        {

            auto qps = integrate(msh, cl, 2*hdi.cell_degree());
            for (auto& qp : qps)
            {
                int_rule_file << qp.first.x() << " " << qp.first.y() << std::endl;
            }
        }
    }
    int_rule_file.flush();
}

template<typename Mesh>
void PrintAgglomeratedCells(const Mesh& msh){
    
    std::ofstream agglo_cells_file("agglomerated_cells.txt");
    for (auto& cl : msh.cells)
    {

        if (location(msh, cl) == element_location::ON_INTERFACE)
        {
            auto pts = points(msh, cl);
            if (pts.size() == 4) {
                continue;
            }
            for (auto point : pts) {
                agglo_cells_file << " ";
                agglo_cells_file << point.x() << " " << point.y();
            }
            agglo_cells_file << std::endl;
        }
    }
    agglo_cells_file.flush();
}
