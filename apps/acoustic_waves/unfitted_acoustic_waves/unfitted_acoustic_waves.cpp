/*
 *       /\        Omar Duran 2019
 *      /__\       omar.duran@enpc.fr
 *     /_\/_\      École Nationale des Ponts et Chaussées - CERMICS
 *    /\    /\
 *   /__\  /__\    This is ProtoN, a library for fast Prototyping of
 *  /_\/_\/_\/_\   Numerical methods.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * If you use this code or parts of it for scientific publications, you
 * are required to cite it as following:
 *
 * Implementation of Discontinuous Skeletal methods on arbitrary-dimensional,
 * polytopal meshes using generic programming.
 * M. Cicuttin, D. A. Di Pietro, A. Ern.
 * Journal of Computational and Applied Mathematics.
 * DOI: 10.1016/j.cam.2017.09.017
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
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

using namespace Eigen;

#include "core/core"
#include "core/solvers"
#include "dataio/silo_io.hpp"
#include "methods/hho"
#include "methods/cuthho"


#include "../common/preprocessor.hpp"
#include "../common/postprocessor.hpp"
#include "../common/newmark_hho_scheme.hpp"
#include "../common/analytical_functions.hpp"


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
                     const testType &test_case, const hho_degree_info hdi)
    {
    }
    
    virtual Vect
    make_contrib_rhs_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType &test_case, const hho_degree_info hdi)
    {
    }

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
        mass *= (1.0/(c*c)); // assuming rho = 1
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
        mass_neg *= (1.0/(test_case.parms.c_1*test_case.parms.c_1));
        mass_pos *= (1.0/(test_case.parms.c_2*test_case.parms.c_2));
        
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
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_NEGATIVE_SIDE, 1.0);
        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_POSITIVE_SIDE, 0.0);

        // stab
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, parms);

        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) += parms.kappa_1 * penalty;
        stab.block(0, cbs, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, 0, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, cbs, cbs, cbs) += parms.kappa_1 * penalty;

        Mat lc = stab + parms.kappa_1 * gr_n.second + parms.kappa_2 * gr_p.second;

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

        return f;
    }
};

template<typename T, size_t ET, typename testType>
auto make_gradrec_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                   testType test_case)
{
    return gradrec_interface_method<T, ET, testType>(eta_);
}

template<typename T, size_t ET, typename testType>
class mixed_interface_method
{
    using Mat  = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

protected:
    mixed_interface_method(){}

    virtual std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType &test_case, const hho_degree_info hdi)
    {
    }
    
    virtual Vect
    make_contrib_rhs_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType &test_case, const hho_degree_info hdi)
    {
    }

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

        auto gr = make_hho_gradrec_mixed_vector(msh, cl, hdi);
        Mat stab = make_hho_naive_stabilization(msh, cl, hdi);
        
        Matrix<T, Dynamic, Dynamic> R_operator = gr.second;
        auto n_rows = R_operator.rows();
        auto n_cols = R_operator.cols();
        
        Matrix<T, Dynamic, Dynamic> S_operator = Matrix<T, Dynamic, Dynamic>::Zero(n_rows, n_cols);
        auto stabilization_operator    = make_hho_naive_stabilization(msh, cl, hdi);
        auto n_s_rows = stabilization_operator.rows();
        auto n_s_cols = stabilization_operator.cols();
        S_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, n_s_rows, n_s_cols) = stabilization_operator;

        T rho = kappa;
        T vp = kappa;
        Mat lc = R_operator + S_operator;
        Mat f = make_mixed_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
        return std::make_pair(lc, f);
    }
    
    Vect
    make_contrib_rhs_uncut(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType &test_case)
    {
        Mat f = make_mixed_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
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
                 const testType &test_case, const hho_degree_info hdi, bool add_scalar_mass_Q = true)
    {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return make_contrib_uncut_mass(msh, cl, hdi, test_case, add_scalar_mass_Q);
        else // on interface
            return make_contrib_cut_mass(msh, cl, hdi, test_case, add_scalar_mass_Q);
    }
    Mat uncut_vec_mass_matrix(const Mesh& msh, const typename Mesh::cell_type& cl,
    const hho_degree_info hdi)
    {
         typedef Matrix<T, Dynamic, Dynamic> matrix_type;
         typedef Matrix<T, Dynamic, 1>       vector_type;

         const auto celdeg  = hdi.cell_degree();
         const auto facdeg  = hdi.face_degree();
         const auto graddeg = hdi.grad_degree();

         cell_basis<Mesh,T>            cb(msh, cl, celdeg);
         vector_cell_basis<Mesh,T>     gb(msh, cl, graddeg);

         auto cbs = cell_basis<Mesh,T>::size(celdeg);
         auto fbs = face_basis<Mesh,T>::size(facdeg);
         auto gbs = vector_cell_basis<Mesh,T>::size(graddeg);

         matrix_type         mass = matrix_type::Zero(gbs, gbs);
         const auto qps = integrate(msh, cl, celdeg - 1 + facdeg);
         for (auto& qp : qps)
         {
             const auto g_phi  = gb.eval_basis(qp.first);

             mass += qp.second * g_phi * g_phi.transpose();
         }
        return mass;
    }
    
    Mat
    make_contrib_uncut_mass(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType &test_case, bool add_scalar_mass_Q = true)
    {
        
        T c;
        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
            c = test_case.parms.c_1;
        else
            c = test_case.parms.c_2;
        
        const auto celdeg  = hdi.cell_degree();
        const auto graddeg = hdi.grad_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto gbs = vector_cell_basis<Mesh,T>::size(graddeg);
        
        Mat mass_sigma = uncut_vec_mass_matrix(msh, cl, hdi);
        Mat mass_v = make_mass_matrix(msh, cl, hdi.cell_degree());
        
        size_t n_data = mass_sigma.rows() + mass_v.rows();
        Mat mass = Mat::Zero(cbs+gbs,cbs+gbs);
        mass.block(0,0,gbs,gbs) = (1.0/(c*c))*mass_sigma;
        if (add_scalar_mass_Q) {
            mass.block(gbs,gbs,cbs,cbs) = (1.0/(c*c))*mass_v;
        }
        return mass;
    }
    
    Mat
    make_contrib_cut_mass(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType &test_case, bool add_scalar_mass_Q = true)
    {

        Mat mass_sigma_neg = make_vec_mass_matrix(msh, cl,
                                        hdi.grad_degree(), element_location::IN_NEGATIVE_SIDE);
        Mat mass_sigma_pos = make_vec_mass_matrix(msh, cl,
                                        hdi.grad_degree(), element_location::IN_POSITIVE_SIDE);
        
        Mat mass_v_neg = make_mass_matrix(msh, cl,
                                        hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
        Mat mass_v_pos = make_mass_matrix(msh, cl,
                                        hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
        mass_v_neg *= (1.0/(test_case.parms.c_1*test_case.parms.c_1));
        mass_v_pos *= (1.0/(test_case.parms.c_2*test_case.parms.c_2));
        
        size_t n_s_data_neg = mass_sigma_neg.rows();
        size_t n_s_data_pos = mass_sigma_pos.rows();
        size_t n_s_data = n_s_data_neg + n_s_data_pos;
        
        size_t n_data_neg = mass_v_neg.rows();
        size_t n_data_pos = mass_v_pos.rows();
        size_t n_v_data = n_data_neg + n_data_pos;
        
        size_t n_data = n_s_data + n_v_data;
        Mat mass = Mat::Zero(n_data,n_data);
        
        mass.block(0,0,n_s_data_neg,n_s_data_neg) = mass_sigma_neg;
        mass.block(n_s_data_neg,n_s_data_neg,n_s_data_pos,n_s_data_pos) = mass_sigma_pos;
        if (add_scalar_mass_Q) {
            mass.block(n_s_data,n_s_data,n_data_neg,n_data_neg) = mass_v_neg;
            mass.block(n_s_data+n_data_neg,n_s_data+n_data_neg,n_data_pos,n_data_pos) = mass_v_pos;
        }
        return mass;
    }
    
};

template<typename T, size_t ET, typename testType>
class gradrec_mixed_interface_method : public mixed_interface_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    gradrec_mixed_interface_method(T eta_)
        : mixed_interface_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType & test_case, const hho_degree_info hdi)
    {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        ///////////////    LHS
        const auto celdeg  = hdi.cell_degree();
        const auto facdeg  = hdi.face_degree();
        const auto graddeg = hdi.grad_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto rbs = vector_cell_basis<Mesh,T>::size(graddeg);

        // GR
        auto gr_n = make_hho_gradrec_mixed_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_NEGATIVE_SIDE, 1.0);
        auto gr_p = make_hho_gradrec_mixed_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_POSITIVE_SIDE, 0.0);

        // stab
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, parms);
        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) += parms.kappa_1 * penalty;
        stab.block(0, cbs, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, 0, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, cbs, cbs, cbs) += parms.kappa_1 * penalty;

        Matrix<T, Dynamic, Dynamic> R_operator = parms.kappa_1 * gr_n.second + parms.kappa_2 * gr_p.second;
        auto n_rows = R_operator.rows();
        auto n_cols = R_operator.cols();
        
        Matrix<T, Dynamic, Dynamic> S_operator = Matrix<T, Dynamic, Dynamic>::Zero(n_rows, n_cols);
        auto stabilization_operator    = stab;
        auto n_s_rows = stabilization_operator.rows();
        auto n_s_cols = stabilization_operator.cols();
        S_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, n_s_rows, n_s_cols) = stabilization_operator;
        Mat lc = R_operator + S_operator;

        ///////////////    RHS
        Vect f = Vect::Zero((cbs+rbs)*2);
        // neg part
        f.block(2*rbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);
        // pos part
        f.block(2*rbs+cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);
        return std::make_pair(lc, f);
    }

    Vect
    make_contrib_rhs_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType &test_case, const hho_degree_info hdi)
    {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        const auto celdeg  = hdi.cell_degree();
        const auto facdeg  = hdi.face_degree();
        const auto graddeg = hdi.grad_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto rbs = vector_cell_basis<Mesh,T>::size(graddeg);

        ///////////////    RHS
        Vect f = Vect::Zero((cbs+rbs)*2);
        // neg part
        f.block(2*rbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);
        // pos part
        f.block(2*rbs+cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);

        return f;
    }
};

template<typename T, size_t ET, typename testType>
auto make_gradrec_mixed_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                   testType & test_case)
{
    return gradrec_mixed_interface_method<T, ET, testType>(eta_);
}

template<typename Mesh, typename testType, typename meth>
test_info<typename Mesh::coordinate_type>
create_kg_and_mg_cuthho_interface(const Mesh& msh, hho_degree_info & hdi, meth method, testType test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg);

template<typename Mesh, typename testType, typename meth>
test_info<typename Mesh::coordinate_type>
create_mixed_kg_and_mg_cuthho_interface(const Mesh& msh, hho_degree_info & hdi, meth method, testType test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg, bool add_scalar_mass_Q = true);

template<typename Mesh, typename testType, typename meth>
void
newmark_step_cuthho_interface(size_t it, typename Mesh::coordinate_type dt, typename Mesh::coordinate_type beta, typename Mesh::coordinate_type gamma, Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<double, Dynamic, 1> & u_dof_n, Matrix<double, Dynamic, 1> & v_dof_n, Matrix<double, Dynamic, 1> & a_dof_n, SparseMatrix<typename Mesh::coordinate_type> & Kg, linear_solver<typename Mesh::coordinate_type> & analysis, bool write_error_Q = false);

template<typename Mesh, typename testType, typename meth>
void newmark_step_cuthho_interface_scatter(size_t it, typename Mesh::coordinate_type dt, typename Mesh::coordinate_type beta, typename Mesh::coordinate_type gamma, Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<double, Dynamic, 1> & u_dof_n, Matrix<double, Dynamic, 1> & v_dof_n, Matrix<double, Dynamic, 1> & a_dof_n, SparseMatrix<typename Mesh::coordinate_type> & Kg, linear_solver<typename Mesh::coordinate_type> & analysis, std::ofstream &sensor_1_log, std::ofstream &sensor_2_log, std::ofstream &sensor_3_log, std::pair<typename Mesh::point_type,size_t> &s1_pt_cell, std::pair<typename Mesh::point_type,size_t> &s2_pt_cell, std::pair<typename Mesh::point_type,size_t> &s3_pt_cell);

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
};

///// test_case_laplacian_waves
// exact solution : t*t*sin(\pi x) sin(\pi y)               in \Omega_1
//                  t*t*sin(\pi x) sin(\pi y)               in \Omega_2
// (\kappa_1,\rho_1) = (\kappa_2,\rho_2) = (1,1)
template<typename T, typename Function, typename Mesh>
class test_case_laplacian_waves: public test_case_laplacian<T, Function, Mesh>
{
   public:
    test_case_laplacian_waves(T t,Function level_set__)
        : test_case_laplacian<T, Function, Mesh>
        (level_set__, params<T>(),
         [level_set__,t](const typename Mesh::point_type& pt) -> T { /* sol */
            if(level_set__(pt) > 0)
                return t*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
            else return t*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());},
         [level_set__,t](const typename Mesh::point_type& pt) -> T { /* rhs */
             if(level_set__(pt) > 0)
                 return 2.0*(1.0 + M_PI*M_PI*t*t)*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
            else return 2.0*(1.0 + M_PI*M_PI*t*t)*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());},
         [level_set__,t](const typename Mesh::point_type& pt) -> T { // bcs
             if(level_set__(pt) > 0)
                return t*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
            else return t*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());},
         [level_set__,t](const typename Mesh::point_type& pt) -> auto { // grad
             Matrix<T, 1, 2> ret;
             if(level_set__(pt) > 0)
             {
                 ret(0) = M_PI*t*t*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
                 ret(1) = M_PI*t*t*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
                 return ret;
             }
             else {
                 ret(0) = M_PI*t*t*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
                 ret(1) = M_PI*t*t*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
                 return ret;}},
         [](const typename Mesh::point_type& pt) -> T {/* Null Dir */
             return 0;},
         [level_set__](const typename Mesh::point_type& pt) -> T {/* Null Neu */
             return 0;})
        {}

};



///// test_case_laplacian_waves_scatter
// (\kappa_1,\rho_1) = (2,1)
// (\kappa_2,\rho_2) = (1,1)
template<typename T, typename Function, typename Mesh>
class test_case_laplacian_waves_scatter: public test_case_laplacian<T, Function, Mesh>
{
   public:

test_case_laplacian_waves_scatter(T t,Function level_set__)
    : test_case_laplacian<T, Function, Mesh>
    (level_set__, params<T>(),
     [level_set__,t](const typename Mesh::point_type& pt) -> T { /* sol */
        if(level_set__(pt) > 0)
        {
            T x,y,xc,yc,r,wave,vx,vy,v,c,lp,factor;
            x = pt.x();
            y = pt.y();
            xc = 0.0;
            yc = 2.0/3.0;
            c = 10.0;
            lp = std::sqrt(9.0)/10.0;
            r = std::sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc));
            wave = (c)/(std::exp((1.0/(lp*lp))*r*r*M_PI*M_PI));
            factor = (lp*lp/(2.0*M_PI*M_PI));
            return factor*wave;
        }
        else {
//            T u,r,r0,dx,dy;
//            r0 = 0.1;
//            dx = pt.x() -0.5;
//            dy = pt.y() -2.0/3.0;
//            r = std::sqrt(dx*dx+dy*dy);
//            if(r < r0){
//                u = 1.0 + std::cos(M_PI*r/r0);
//            }else{
//                u = 0.0;
//            }
//            return u;
            T x,y,xc,yc,r,wave,vx,vy,v,c,lp,factor;
            x = pt.x();
            y = pt.y();
            xc = 0.0;
            yc = 2.0/3.0;
            c = 10.0;
            lp = std::sqrt(9.0)/10.0;
            r = std::sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc));
            wave = (c)/(std::exp((1.0/(lp*lp))*r*r*M_PI*M_PI));
            factor = (lp*lp/(2.0*M_PI*M_PI));
            return factor*wave;
        }},
     [level_set__,t](const typename Mesh::point_type& pt) -> T { /* rhs */
         if(level_set__(pt) > 0)
             return 0.0;
        else return 0.0;},
     [level_set__,t](const typename Mesh::point_type& pt) -> T { // bcs
         if(level_set__(pt) > 0)
            return 0.0;
        else return 0.0;},
     [level_set__,t](const typename Mesh::point_type& pt) -> auto { // grad
         Matrix<T, 1, 2> ret;
         if(level_set__(pt) > 0)
         {
             ret(0) = M_PI*t*t*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
             ret(1) = M_PI*t*t*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
             return ret;
         }
         else {
             ret(0) = M_PI*t*t*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
             ret(1) = M_PI*t*t*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
             return ret;}},
     [](const typename Mesh::point_type& pt) -> T {/* Null Dir */
         return 0;},
     [level_set__](const typename Mesh::point_type& pt) -> T {/* Null Neu */
         return 0;})
    {}
    
};

template<typename Mesh, typename Function>
auto make_test_case_laplacian_conv(const Mesh& msh, Function level_set_function)
{
    return test_case_laplacian_conv<typename Mesh::coordinate_type, Function, Mesh>(level_set_function);
}


template<typename Mesh, typename Function>
auto make_test_case_laplacian_waves(double t, const Mesh& msh, Function level_set_function)
{
    return test_case_laplacian_waves<typename Mesh::coordinate_type, Function, Mesh>(t,level_set_function);
}

template<typename Mesh, typename Function>
auto make_test_case_laplacian_waves_scatter(double t, const Mesh& msh, Function level_set_function)
{
    return test_case_laplacian_waves_scatter<typename Mesh::coordinate_type, Function, Mesh>(t,level_set_function);
}

void ICutHHOSecondOrder(int argc, char **argv);
void HeterogeneousGar6moreICutHHOSecondOrder(int argc, char **argv);

// Convergence with homogeneous material props
mesh_type SquareCutMesh(level_set<RealType> & level_set_function, size_t l_divs, size_t int_refsteps = 4);
void CutHHOSecondOrderConvTest(int argc, char **argv);
void CutHHOFirstOrderConvTest(int argc, char **argv);


int main(int argc, char **argv)
{
    
//    ICutHHOSecondOrder(argc, argv);
//    HeterogeneousGar6moreICutHHOSecondOrder(argc, argv);
    
//    CutHHOSecondOrderConvTest(argc, argv);
//    CutHHOFirstOrderConvTest(argc, argv);
    return 0;
}

mesh_type SquareCutMesh(level_set<RealType> & level_set_function, size_t l_divs, size_t int_refsteps){
    
    mesh_init_params<RealType> mip;
    mip.Nx = 2;
    mip.Ny = 2;
    l_divs += 2;
    for (unsigned int i = 0; i < l_divs; i++) {
      mip.Nx *= 2;
      mip.Ny *= 2;
    }

    timecounter tc;

    tc.tic();
    mesh_type msh(mip);
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;

    tc.tic();
    detect_node_position(msh, level_set_function); // ok
    detect_cut_faces(msh, level_set_function); // it could be improved
    detect_cut_cells(msh, level_set_function);

    // Agglomerate.
    bool agglomerate_Q = true;
    if (agglomerate_Q) {
      detect_cell_agglo_set(msh, level_set_function);
      make_neighbors_info_cartesian(msh);
      refine_interface(msh, level_set_function, int_refsteps);
      make_agglomeration(msh, level_set_function);
    }

    tc.toc();
    std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;
    return msh;
}

void CutHHOSecondOrderConvTest(int argc, char **argv){
    
    size_t degree           = 0;
    size_t l_divs          = 0;
    size_t nt_divs       = 0;
    size_t int_refsteps     = 4;
    bool dump_debug         = false;

    /* k <deg>:     method degree
     * l <num>:     number of cells in x and y direction
     * r <num>:     number of interface refinement steps
     * d:           dump debug data
     */

    // Simplified input
     int ch;
     while ( (ch = getopt(argc, argv, "k:l:r:n:d")) != -1 )
     {
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
        for(size_t l = 0; l < l_divs; l++){
            
            mesh_type msh = SquareCutMesh(level_set_function,l,int_refsteps);
            if (dump_debug)
            {
                dump_mesh(msh);
                output_mesh_info(msh, level_set_function);
            }
            auto test_case = make_test_case_laplacian_conv(msh, level_set_function);
            auto method = make_gradrec_interface_method(msh, 1.0, test_case);
            
            create_kg_and_mg_cuthho_interface(msh, hdi, method, test_case, Kg, Mg);
            
            bool direct_solver_Q = true;
            linear_solver<RealType> analysis;
            analysis.set_Kg(Kg);

            if (!direct_solver_Q) {
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

void CutHHOFirstOrderConvTest(int argc, char **argv){
    
    size_t degree           = 0;
    size_t l_divs          = 0;
    size_t nt_divs       = 0;
    size_t int_refsteps     = 4;
    bool dump_debug         = false;

    /* k <deg>:     method degree
     * l <num>:     number of cells in x and y direction
     * r <num>:     number of interface refinement steps
     * d:           dump debug data
     */

    // Simplified input
     int ch;
     while ( (ch = getopt(argc, argv, "k:l:r:n:d")) != -1 )
     {
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

    std::ofstream error_file("steady_state_two_fields_error.txt");
    
    RealType radius = 1.0/3.0;
    auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);

    
    timecounter tc;
    SparseMatrix<RealType> Kg, Mg;

    for(size_t k = 0; k <= degree; k++){
        std::cout << bold << cyan << "Running an approximation with k : " << k << reset << std::endl;
        error_file << "Approximation with k : " << k << std::endl;
        
        hho_degree_info hdi(k+1, k);
        for(size_t l = 0; l < l_divs; l++){
            
            mesh_type msh = SquareCutMesh(level_set_function,l,int_refsteps);
            if (dump_debug)
            {
                dump_mesh(msh);
                output_mesh_info(msh, level_set_function);
            }
            auto test_case = make_test_case_laplacian_conv(msh, level_set_function);
            auto method = make_gradrec_mixed_interface_method(msh, 1.0, test_case);
            
            create_mixed_kg_and_mg_cuthho_interface(msh, hdi, method, test_case, Kg, Mg, false);
            bool direct_solver_Q = true;
            linear_solver<RealType> analysis;
            Kg += Mg;
            analysis.set_Kg(Kg);
            tc.tic();
            if (!direct_solver_Q) {
                analysis.set_iterative_solver();
            }
            analysis.factorize();
            
            auto assembler = make_two_fields_interface_assembler(msh, test_case.bcs_fun, hdi);
            assembler.RHS.setZero(); // assuming null dirichlet data on boundary.
            for (auto& cl : msh.cells)
            {
                auto f = method.make_contrib_rhs(msh, cl, test_case, hdi);
                assembler.assemble_rhs(msh, cl, f);
            }
            Matrix<RealType, Dynamic, 1> x_dof = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows(),1);
            x_dof = analysis.solve(assembler.RHS);

            if (dump_debug)
            {
                std::string silo_file_name = "cut_steady_mixed_k_" + std::to_string(k) + "_";
                postprocessor<cuthho_poly_mesh<RealType>>::write_silo_two_fields(silo_file_name, l, msh, hdi, assembler, x_dof, test_case.sol_fun, false);
            }
            postprocessor<cuthho_poly_mesh<RealType>>::compute_errors_two_fields(msh, hdi, assembler, x_dof, test_case.sol_fun, test_case.sol_grad, error_file);
        }
        error_file << std::endl << std::endl;
    }
    error_file.close();
}

void ICutHHOSecondOrder(int argc, char **argv){
    
    
    size_t degree           = 0;
    size_t l_divs          = 0;
    size_t nt_divs       = 0;
    size_t int_refsteps     = 4;
    bool dump_debug         = false;

    mesh_init_params<RealType> mip;
    mip.Nx = 8;
    mip.Ny = 8;

    /* k <deg>:     method degree
     * l <num>:     number of cells in x and y direction
     * r <num>:     number of interface refinement steps
     * d:           dump debug data
     */

    // Simplified input
     int ch;
     while ( (ch = getopt(argc, argv, "k:l:r:n:d")) != -1 )
     {
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

    mip.Nx = 2;
    mip.Ny = 2;
    for (unsigned int i = 0; i < l_divs; i++) {
        mip.Nx *= 2;
        mip.Ny *= 2;
    }

  
    timecounter tc;

    /************** BUILD MESH **************/
    tc.tic();
    cuthho_poly_mesh<RealType> msh(mip);
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;
    /************** LEVEL SET FUNCTION **************/
    RealType radius = 1.0/3.0;
    auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);
    

    tc.tic();
    detect_node_position(msh, level_set_function); // ok
    detect_cut_faces(msh, level_set_function); // it could be improved
    detect_cut_cells(msh, level_set_function);
    
    
    // Agglomerate.
    bool agglomerate_Q = true;
    if (agglomerate_Q) {
        detect_cell_agglo_set(msh, level_set_function);
        make_neighbors_info_cartesian(msh);
        // make_neighbors_info(msh);
        refine_interface(msh, level_set_function, int_refsteps);
        make_agglomeration(msh, level_set_function);
    }


    tc.toc();
    std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

    if (dump_debug)
    {
        dump_mesh(msh);
        output_mesh_info(msh, level_set_function);
//        test_projection(msh, level_set_function, degree);
    }

    // Time controls : Final time value 1.0
    size_t nt = 10;
    for (unsigned int i = 0; i < nt_divs; i++) {
        nt *= 2;
    }
    RealType ti = 0.0;
    RealType tf = 1.0;
    RealType dt = (tf-ti)/nt;
    RealType t = ti;
    
    RealType beta = 0.25;
    RealType gamma = 0.5;
    
    // Create static data
    SparseMatrix<RealType> Kg, Kg_c, Mg;
    hho_degree_info hdi(degree+1, degree);
    auto test_case = make_test_case_laplacian_waves(t,msh, level_set_function);
    auto method = make_gradrec_interface_method(msh, 1.0, test_case);
    create_kg_and_mg_cuthho_interface(msh, hdi, method, test_case, Kg, Mg);
    
    bool direct_solver_Q = true;
    linear_solver<RealType> analysis;
    Kg_c = Kg;
    Kg *= beta*(dt*dt);
    Kg += Mg;
    analysis.set_Kg(Kg);
    tc.tic();
    if (!direct_solver_Q) {
        analysis.set_iterative_solver();
    }
    analysis.factorize();
    
    // Projecting initial scalar, velocity and acceleration
    Matrix<RealType, Dynamic, 1> u_dof_n, v_dof_n, a_dof_n;
    
    bool write_error_Q  = false;
    for(size_t it = 1; it <= nt; it++){ // for each time step
        // Manufactured solution
        std::cout << std::endl;
        std::cout << "Time step number: " <<  it << std::endl;
        RealType t = dt*it+ti;
        auto test_case = make_test_case_laplacian_waves(t,msh, level_set_function);
        auto method = make_gradrec_interface_method(msh, 1.0, test_case);
        if (it == nt) {
            write_error_Q = true;
        }
        newmark_step_cuthho_interface(it, dt, beta, gamma, msh, hdi, method, test_case, u_dof_n,  v_dof_n, a_dof_n, Kg_c, analysis, write_error_Q);
    }
    RealType hx = 1.0/mip.Nx;
    RealType hy = 1.0/mip.Ny;
    RealType h = std::sqrt(hx*hx+hy*hy);
    std::cout << "h = " << h << std::endl;
}

void HeterogeneousGar6moreICutHHOSecondOrder(int argc, char **argv){
        
    size_t degree           = 0;
    size_t l_divs          = 0;
    size_t nt_divs       = 0;
    size_t int_refsteps     = 4;
    bool dump_debug         = false;

    mesh_init_params<RealType> mip;
    mip.Nx = 8;
    mip.Ny = 8;
    mip.min_x = -1.5;
    mip.max_x = 1.5;
    mip.min_y = -1.5;
    mip.max_y = 1.5;

    /* k <deg>:     method degree
     * l <num>:     number of cells in x and y direction
     * r <num>:     number of interface refinement steps
     * d:           dump debug data
     */

    // Simplified input
    int ch;
    while ( (ch = getopt(argc, argv, "k:l:r:n:d")) != -1 )
    {
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

    mip.Nx = 3;
    mip.Ny = 3;
    for (unsigned int i = 0; i < l_divs; i++) {
        mip.Nx *= 2;
        mip.Ny *= 2;
    }

  
    timecounter tc;

    /************** BUILD MESH **************/
    tc.tic();
    mesh_type msh(mip);
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;
    /************** LEVEL SET FUNCTION **************/
    RealType cy = 1.0e-15;
    auto level_set_function = line_level_set<RealType>(cy);
    

    tc.tic();
    detect_node_position(msh, level_set_function); // ok
    detect_cut_faces(msh, level_set_function); // it could be improved
    detect_cut_cells(msh, level_set_function);
    
    
    // Agglomerate.
    bool agglomerate_Q = true;
    if (agglomerate_Q) {
        detect_cell_agglo_set(msh, level_set_function);
        make_neighbors_info_cartesian(msh);
        // make_neighbors_info(msh);
        refine_interface(msh, level_set_function, int_refsteps);
        make_agglomeration(msh, level_set_function);
    }


    tc.toc();
    std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

    if (dump_debug)
    {
        dump_mesh(msh);
        output_mesh_info(msh, level_set_function);
//        test_projection(msh, level_set_function, degree);
    }

    // Time controls : Final time value 1.0
    size_t nt = 10;
    for (unsigned int i = 0; i < nt_divs; i++) {
        nt *= 2;
    }
    RealType ti = 0.0;
    RealType tf = 1.0;
    RealType dt = (tf-ti)/nt;
    RealType t = ti;
    
    
    RealType beta = 0.25;
    RealType gamma = 0.5;
    
    // Create static data
    SparseMatrix<RealType> Kg, Kg_c, Mg;
    hho_degree_info hdi(degree+1, degree);
    auto test_case = make_test_case_laplacian_waves_scatter(t,msh, level_set_function);
    test_case.parms.kappa_1 = 1.0;
    test_case.parms.kappa_2 = 1.0;
    test_case.parms.c_1 = std::sqrt(3.0);
    test_case.parms.c_2 = std::sqrt(9.0);
    auto method = make_gradrec_interface_method(msh, 1.0, test_case);
    create_kg_and_mg_cuthho_interface(msh, hdi, method, test_case, Kg, Mg);
    
    bool direct_solver_Q = true;
    linear_solver<RealType> analysis;
    Kg_c = Kg;
    Kg *= beta*(dt*dt);
    Kg += Mg;
    analysis.set_Kg(Kg);
    tc.tic();
    if (!direct_solver_Q) {
        analysis.set_iterative_solver();
    }
    analysis.factorize();
    
    std::ofstream sensor_1_log("s1_cut_acoustic_one_field.csv");
    std::ofstream sensor_2_log("s2_cut_acoustic_one_field.csv");
    std::ofstream sensor_3_log("s3_cut_acoustic_one_field.csv");
    
    typename mesh_type::point_type s1_pt(-1.0/3.0, -1.0/3.0);
    typename mesh_type::point_type s2_pt( 0.0, -1.0/3.0);
    typename mesh_type::point_type s3_pt(+1.0/3.0, -1.0/3.0);
    std::pair<typename mesh_type::point_type,size_t> s1_pt_cell = std::make_pair(s1_pt, -1);
    std::pair<typename mesh_type::point_type,size_t> s2_pt_cell = std::make_pair(s2_pt, -1);
    std::pair<typename mesh_type::point_type,size_t> s3_pt_cell = std::make_pair(s3_pt, -1);
    
    // Projecting initial scalar, velocity and acceleration
    Matrix<RealType, Dynamic, 1> u_dof_n, v_dof_n, a_dof_n;
    for(size_t it = 1; it <= nt; it++){ // for each time step
        // Manufactured solution
        std::cout << std::endl;
        std::cout << "Time step number: " <<  it << std::endl;
        RealType t = dt*it+ti;
        auto test_case = make_test_case_laplacian_waves_scatter(t,msh, level_set_function);
        auto method = make_gradrec_interface_method(msh, 1.0, test_case);
        newmark_step_cuthho_interface_scatter(it, dt, beta, gamma, msh, hdi, method, test_case, u_dof_n,  v_dof_n, a_dof_n, Kg_c, analysis, sensor_1_log, sensor_2_log, sensor_3_log, s1_pt_cell, s2_pt_cell, s3_pt_cell);
        
    }
    RealType hx = 1.0/mip.Nx;
    RealType hy = 1.0/mip.Ny;
    RealType h = std::sqrt(hx*hx+hy*hy);
    std::cout << "h = " << h << std::endl;
    
}

template<typename Mesh, typename testType, typename meth>
test_info<typename Mesh::coordinate_type>
create_kg_and_mg_cuthho_interface(const Mesh& msh, hho_degree_info & hdi, meth method, testType test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg){
    
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

    bool sc = false; // static condensation
    assert(sc==false); // This case will implemented properly
    
    tc.tic();
    auto assembler = make_one_field_interface_assembler(msh, bcs_fun, hdi);
    auto assembler_sc = make_interface_condensed_assembler(msh, bcs_fun, hdi);
        
    size_t cell_ind = 0;
    for (auto& cl : msh.cells)
    {
        auto contrib = method.make_contrib(msh, cl, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;

        auto cell_mass = method.make_contrib_mass(msh, cl, test_case, hdi);
        size_t n_dof = assembler.n_dof(msh,cl);
        Matrix<RealType, Dynamic, Dynamic> mass = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof,n_dof);
        mass.block(0,0,cell_mass.rows(),cell_mass.cols()) = cell_mass;
        
        if( sc )
            assembler_sc.assemble(msh, cl, lc, f);
        else
            assembler.assemble(msh, cl, lc, f);
            assembler.assemble_mass(msh, cl, mass);
        
        cell_ind++;
    }

    if( sc )
        assembler_sc.finalize();
    else
        assembler.finalize();

    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;
    
    Kg = assembler.LHS;
    Mg = assembler.MASS;
}

template<typename Mesh, typename testType, typename meth>
test_info<typename Mesh::coordinate_type>
create_mixed_kg_and_mg_cuthho_interface(const Mesh& msh, hho_degree_info & hdi, meth method, testType test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg, bool add_scalar_mass_Q){
    
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

    bool sc = false; // static condensation
    assert(sc==false); // This case will implemented properly
    
    tc.tic();
    auto assembler = make_two_fields_interface_assembler(msh, bcs_fun, hdi);
    auto assembler_sc = make_interface_condensed_assembler(msh, bcs_fun, hdi);
        
    size_t cell_ind = 0;
    for (auto& cl : msh.cells)
    {
        auto contrib = method.make_contrib(msh, cl, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;

        auto cell_mass = method.make_contrib_mass(msh, cl, test_case, hdi, add_scalar_mass_Q);
        size_t n_dof = assembler.n_dof(msh,cl);
        Matrix<RealType, Dynamic, Dynamic> mass = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof,n_dof);
        mass.block(0,0,cell_mass.rows(),cell_mass.cols()) = cell_mass;
        
        if( sc )
            assembler_sc.assemble(msh, cl, lc, f);
        else
            assembler.assemble(msh, cl, lc, f);
            assembler.assemble_mass(msh, cl, mass);
        
        cell_ind++;
    }

    if( sc )
        assembler_sc.finalize();
    else
        assembler.finalize();

    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;
    
    Kg = assembler.LHS;
    Mg = assembler.MASS;
}

template<typename Mesh, typename testType, typename meth>
void
newmark_step_cuthho_interface(size_t it, typename Mesh::coordinate_type dt, typename Mesh::coordinate_type beta, typename Mesh::coordinate_type gamma, Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<double, Dynamic, 1> & u_dof_n, Matrix<double, Dynamic, 1> & v_dof_n, Matrix<double, Dynamic, 1> & a_dof_n, SparseMatrix<typename Mesh::coordinate_type> & Kg, linear_solver<typename Mesh::coordinate_type> & analysis, bool write_error_Q)
{
    using RealType = typename Mesh::coordinate_type;
    bool write_silo_Q = true;
    auto level_set_function = test_case.level_set_;

    auto rhs_fun = test_case.rhs_fun;
    auto sol_fun = test_case.sol_fun;
    auto sol_grad = test_case.sol_grad;
    auto bcs_fun = test_case.bcs_fun;
    auto dirichlet_jump = test_case.dirichlet_jump;
    auto neumann_jump = test_case.neumann_jump;
    struct params<RealType> parms = test_case.parms;
    
    timecounter tc;

    bool sc = false; // static condensation
    assert(sc==false); // This case will implemented properly
    
    tc.tic();
    auto assembler = make_one_field_interface_assembler(msh, bcs_fun, hdi);
    auto assembler_sc = make_interface_condensed_assembler(msh, bcs_fun, hdi);
    
    if (u_dof_n.rows() == 0) {
        size_t n_dof = assembler.LHS.rows();
        u_dof_n = Matrix<RealType, Dynamic, 1>::Zero(n_dof,1);
        v_dof_n = Matrix<RealType, Dynamic, 1>::Zero(n_dof,1);
        a_dof_n = Matrix<RealType, Dynamic, 1>::Zero(n_dof,1);
        
        auto a_fun = [](const typename Mesh::point_type& pt) -> typename Mesh::coordinate_type {
            return 2.0*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
        };

        assembler.project_over_cells(msh, hdi, a_dof_n, a_fun);
        
        size_t it = 0;
        if(write_silo_Q){
            std::string silo_file_name = "cut_hho_one_field_";
            postprocessor<Mesh>::write_silo_one_field(silo_file_name, it, msh, hdi, assembler, u_dof_n, sol_fun, false);
        }
    }
    
    assembler.RHS.setZero(); // assuming null dirichlet data on boundary.
    for (auto& cl : msh.cells)
    {
        auto f = method.make_contrib_rhs(msh, cl, test_case, hdi);
        assembler.assemble_rhs(msh, cl, f);
    }

    tc.toc();
    std::cout << bold << yellow << "RHS assembly: " << tc << " seconds" << reset << std::endl;
    
    // Compute intermediate state for scalar and rate
    u_dof_n = u_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
    v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
    Matrix<RealType, Dynamic, 1> res = Kg*u_dof_n;
    assembler.RHS -= res;
    
    if( sc )
        std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
    else
        std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;

    tc.tic();
    a_dof_n = analysis.solve(assembler.RHS); // new acceleration
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    // update scalar and rate
    u_dof_n += beta*dt*dt*a_dof_n;
    v_dof_n += gamma*dt*a_dof_n;
    
    if(write_silo_Q){
        std::string silo_file_name = "cut_hho_one_field_";
        postprocessor<Mesh>::write_silo_one_field(silo_file_name, it, msh, hdi, assembler, u_dof_n, sol_fun, false);
    }
    
    if(write_error_Q){
        postprocessor<Mesh>::compute_errors_one_field(msh, hdi, assembler, u_dof_n, sol_fun, sol_grad);
    }
}

template<typename Mesh, typename testType, typename meth>
void newmark_step_cuthho_interface_scatter(size_t it, typename Mesh::coordinate_type dt, typename Mesh::coordinate_type beta, typename Mesh::coordinate_type gamma, Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<double, Dynamic, 1> & u_dof_n, Matrix<double, Dynamic, 1> & v_dof_n, Matrix<double, Dynamic, 1> & a_dof_n, SparseMatrix<typename Mesh::coordinate_type> & Kg, linear_solver<typename Mesh::coordinate_type> & analysis, std::ofstream &sensor_1_log, std::ofstream &sensor_2_log, std::ofstream &sensor_3_log, std::pair<typename Mesh::point_type,size_t> &s1_pt_cell, std::pair<typename Mesh::point_type,size_t> &s2_pt_cell, std::pair<typename Mesh::point_type,size_t> &s3_pt_cell)
{
    using RealType = typename Mesh::coordinate_type;
    bool write_silo_Q = true;
    auto level_set_function = test_case.level_set_;

    auto rhs_fun = test_case.rhs_fun;
    auto sol_fun = test_case.sol_fun;
    auto sol_grad = test_case.sol_grad;
    auto bcs_fun = test_case.bcs_fun;
    auto dirichlet_jump = test_case.dirichlet_jump;
    auto neumann_jump = test_case.neumann_jump;
    struct params<RealType> parms = test_case.parms;
    
    timecounter tc;

    bool sc = false; // static condensation
    assert(sc==false); // This case will implemented properly
    
    tc.tic();
    auto assembler = make_one_field_interface_assembler(msh, bcs_fun, hdi);
    auto assembler_sc = make_interface_condensed_assembler(msh, bcs_fun, hdi);
        
    if (u_dof_n.rows() == 0) {
        size_t n_dof = assembler.LHS.rows();
        u_dof_n = Matrix<RealType, Dynamic, 1>::Zero(n_dof,1);
        v_dof_n = Matrix<RealType, Dynamic, 1>::Zero(n_dof,1);
        a_dof_n = Matrix<RealType, Dynamic, 1>::Zero(n_dof,1);
        
        auto u_fun = [](const typename Mesh::point_type& pt) -> typename Mesh::coordinate_type {
//            RealType u,r,r0,dx,dy;
//            r0 = 0.1;
//            dx = pt.x() -0.5;
//            dy = pt.y() -2.0/3.0;
//            r = std::sqrt(dx*dx+dy*dy);
//            if(r < r0){
//                u = 1.0 + std::cos(M_PI*r/r0);
//            }else{
//                u = 0.0;
//            }
//            return u;
            RealType x,y,xc,yc,r,wave,vx,vy,v,c,lp,factor;
            x = pt.x();
            y = pt.y();
            xc = 0.0;
            yc = 2.0/3.0;
            c = 10.0;
            lp = std::sqrt(9.0)/10.0;
            r = std::sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc));
            wave = (c)/(std::exp((1.0/(lp*lp))*r*r*M_PI*M_PI));
            factor = (lp*lp/(2.0*M_PI*M_PI));
            return factor*wave;
        };

        assembler.project_over_cells(msh, hdi, u_dof_n, u_fun);
        
        size_t it = 0;
        if(write_silo_Q){
            std::string silo_file_name = "cut_hho_one_field_";
            postprocessor<Mesh>::write_silo_one_field(silo_file_name, it, msh, hdi, assembler, u_dof_n, sol_fun, false);
        }
        
        postprocessor<mesh_type>::record_data_acoustic_two_fields(it, s1_pt_cell, msh, hdi, assembler, u_dof_n, sensor_1_log);
        postprocessor<mesh_type>::record_data_acoustic_two_fields(it, s2_pt_cell, msh, hdi, assembler, u_dof_n, sensor_2_log);
        postprocessor<mesh_type>::record_data_acoustic_two_fields(it, s3_pt_cell, msh, hdi, assembler, u_dof_n, sensor_3_log);
        
    }
    
    assembler.RHS.setZero(); // assuming null dirichlet data on boundary.
    for (auto& cl : msh.cells)
    {
        auto f = method.make_contrib_rhs(msh, cl, test_case, hdi);
        assembler.assemble_rhs(msh, cl, f);
    }

    tc.toc();
    std::cout << bold << yellow << "RHS assembly: " << tc << " seconds" << reset << std::endl;
    
    // Compute intermediate state for scalar and rate
    u_dof_n = u_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
    v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
    Matrix<RealType, Dynamic, 1> res = Kg*u_dof_n;
    assembler.RHS -= res;
    
    if( sc )
        std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
    else
        std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;

    tc.tic();
    a_dof_n = analysis.solve(assembler.RHS); // new acceleration
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    // update scalar and rate
    u_dof_n += beta*dt*dt*a_dof_n;
    v_dof_n += gamma*dt*a_dof_n;
    
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
    
    if(write_silo_Q){
        std::string silo_file_name = "cut_hho_one_field_";
        postprocessor<Mesh>::write_silo_one_field(silo_file_name, it, msh, hdi, assembler, u_dof_n, sol_fun, false);
    }
    
    postprocessor<mesh_type>::record_data_acoustic_two_fields(it, s1_pt_cell, msh, hdi, assembler, u_dof_n, sensor_1_log);
    postprocessor<mesh_type>::record_data_acoustic_two_fields(it, s2_pt_cell, msh, hdi, assembler, u_dof_n, sensor_2_log);
    postprocessor<mesh_type>::record_data_acoustic_two_fields(it, s3_pt_cell, msh, hdi, assembler, u_dof_n, sensor_3_log);
    
}

