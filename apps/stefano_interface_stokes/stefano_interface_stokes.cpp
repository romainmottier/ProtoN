/*
 *       /\        Guillaume Delay 2018,2019
 *      /__\       guillaume.delay@enpc.fr
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

#include <unsupported/Eigen/MatrixFunctions> // ADD BY STEFANO

using namespace Eigen;

#include "core/core"
#include "core/solvers"
#include "dataio/silo_io.hpp"

#include "methods/hho"
#include "methods/cuthho"
#include "methods/transport"

#include "tbb/tbb.h"
#define HAVE_INTEL_TBB
//#include "/usr/local/Cellar/tbb/2020_U2/include/tbb/tbb.h"
//#include "/opt/intel/compilers_and_libraries_2020.1.216⁩/mac/tbb/include/tbb/tbb.h"

//using namespace tbb;





/*****************************************************************************
*   PREVIOUS CODE STOKES HHO
*****************************************************************************/



///////////////////////   FICTITIOUS DOMAIN METHODS  ///////////////////////////

template<typename T, size_t ET, typename testType>
class stokes_fictdom_method
{
    using Mat  = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

protected:
    bool sym_grad;

    stokes_fictdom_method(bool sym)
        : sym_grad(sym)
        {}

    virtual std::pair< std::pair<Mat,Mat>, std::pair<Vect,Vect> >
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi,
                     const element_location where = element_location::IN_NEGATIVE_SIDE,
                     const params<T>& parms = params<T>())
    {}

public:
    std::pair< std::pair<Mat,Mat>, std::pair<Vect,Vect> >
    make_contrib_uncut(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType test_case)
    {
        Mat gr2;
        if(sym_grad)
            gr2 = make_hho_gradrec_sym_matrix(msh, cl, hdi).second;
        else
            gr2 = make_hho_gradrec_matrix(msh, cl, hdi).second;
        Mat stab = make_hho_vector_naive_stabilization(msh, cl, hdi);
        Mat lc = gr2 + stab;
        auto dr = make_hho_divergence_reconstruction(msh, cl, hdi);
        Vect f = make_vector_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
        Vect p_rhs = Vect::Zero( dr.first.rows() );
        return std::make_pair( std::make_pair(lc, dr.second) , std::make_pair(f,p_rhs) );
    }


    std::pair< std::pair<Mat,Mat>, std::pair<Vect,Vect> >
    make_contrib(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const testType test_case, const hho_degree_info hdi,
                 const element_location where = element_location::IN_NEGATIVE_SIDE,
                 const params<T>& parms = params<T>())
    {
        if( location(msh, cl) == where )
            return make_contrib_uncut(msh, cl, hdi, test_case);
        else if( location(msh, cl) != element_location::ON_INTERFACE )
        {
            Mat lc;
            Vect f;
            return std::make_pair(std::make_pair(lc, lc) , std::make_pair(f,f) );
        }
        else // on interface
            return make_contrib_cut(msh, cl, test_case, hdi, where, parms);
    }
};

/////////////////////////  GRADREC_FICTITIOUS_METHOD

template<typename T, size_t ET, typename testType>
class gradrec_stokes_fictdom_method : public stokes_fictdom_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    gradrec_stokes_fictdom_method(T eta_, bool sym)
        : stokes_fictdom_method<T,ET,testType>(sym), eta(eta_) {}

    std::pair< std::pair<Mat,Mat>, std::pair<Vect,Vect> >
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi,
                     const element_location where = element_location::IN_NEGATIVE_SIDE,
                     const params<T>& parms = params<T>())
    {
        // LHS
        Mat gr1, gr2;
        if( this->sym_grad )
        {
            auto gr = make_hho_gradrec_sym_matrix(msh, cl, test_case.level_set_, hdi, where, 1.0);
            gr1 = gr.first;
            gr2 = gr.second;
        }
        else
        {
            auto gr = make_hho_gradrec_matrix(msh, cl, test_case.level_set_, hdi, where, 1.0);
            gr1 = gr.first;
            gr2 = gr.second;
        }
        Mat stab = make_hho_vector_cut_stabilization(msh, cl, hdi, where)
            + make_hho_cut_interface_vector_penalty(msh, cl, hdi, eta);
        Mat lc = gr2 + stab;
        auto dr = make_hho_divergence_reconstruction(msh, cl, test_case.level_set_,
                                                     hdi, where, 1.0);

        // RHS
        auto celdeg = hdi.cell_degree();
        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);

        Vect f = Vect::Zero(lc.rows());
        f.block(0, 0, cbs, 1) += make_vector_rhs(msh, cl, celdeg, test_case.rhs_fun, where);
        f.block(0, 0, cbs, 1) += make_vector_rhs_penalty(msh, cl, celdeg, test_case.bcs_vel, eta);
        f += make_vector_GR_rhs(msh, cl, celdeg, test_case.bcs_vel, test_case.level_set_,
                                gr1, this->sym_grad);
        auto p_rhs = make_pressure_rhs(msh, cl, hdi.face_degree(), where,
                                       test_case.level_set_, test_case.bcs_vel);

        return std::make_pair(std::make_pair(lc, dr.second), std::make_pair(f,p_rhs) );
    }
};



template<typename T, size_t ET, typename testType>
auto make_gradrec_stokes_fictdom_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                        const testType test_case, bool sym)
{
    return gradrec_stokes_fictdom_method<T, ET, testType>(eta_, sym);
}

///////////////////////////

template<typename Mesh, typename testType>
stokes_test_info<typename Mesh::coordinate_type>
run_cuthho_fictdom(const Mesh& msh, size_t degree, testType test_case)
{
    using RealType = typename Mesh::coordinate_type;

    auto level_set_function = test_case.level_set_;

    auto sol_vel = test_case.sol_vel;
    auto vel_grad = test_case.vel_grad;
    auto bcs_fun = test_case.bcs_vel;


    /************** OPEN SILO DATABASE **************/
    silo_database silo;
    silo.create("cuthho_fictdom.silo");
    silo.add_mesh(msh, "mesh");

    /************** MAKE A SILO VARIABLE FOR CELL POSITIONING **************/
    std::vector<RealType> cut_cell_markers;
    for (auto& cl : msh.cells)
    {
        if ( location(msh, cl) == element_location::IN_POSITIVE_SIDE )
            cut_cell_markers.push_back(1.0);
        else if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
            cut_cell_markers.push_back(-1.0);
        else if ( location(msh, cl) == element_location::ON_INTERFACE )
            cut_cell_markers.push_back(0.0);
        else
            throw std::logic_error("shouldn't have arrived here...");
    }
    silo.add_variable("mesh", "cut_cells", cut_cell_markers.data(), cut_cell_markers.size(), zonal_variable_t);

    /************** MAKE A SILO VARIABLE FOR LEVEL SET FUNCTION **************/
    std::vector<RealType> level_set_vals;
    for (auto& pt : msh.points)
        level_set_vals.push_back( level_set_function(pt) );
    silo.add_variable("mesh", "level_set", level_set_vals.data(), level_set_vals.size(), nodal_variable_t);

    /************** MAKE A SILO VARIABLE FOR NODE POSITIONING **************/
    std::vector<RealType> node_pos;
    for (auto& n : msh.nodes)
        node_pos.push_back( location(msh, n) == element_location::IN_POSITIVE_SIDE ? +1.0 : -1.0 );
    silo.add_variable("mesh", "node_pos", node_pos.data(), node_pos.size(), nodal_variable_t);


    timecounter tc;

    bool sc = true; // static condensation

    /************** ASSEMBLE PROBLEM **************/
    hho_degree_info hdi(degree+1, degree);

    element_location where = element_location::IN_NEGATIVE_SIDE;

    tc.tic();
    auto assembler = make_stokes_fict_assembler(msh, bcs_fun, hdi, where);
    auto assembler_sc = make_stokes_fict_condensed_assembler(msh, bcs_fun, hdi, where);

    // method with gradient reconstruction (penalty-free)
    auto class_meth = make_gradrec_stokes_fictdom_method(msh, 1.0, test_case, true);

    for (auto& cl : msh.cells)
    {
        if( !(location(msh, cl) == element_location::ON_INTERFACE || location(msh, cl) == where) )
            continue;
        auto contrib = class_meth.make_contrib(msh, cl, test_case, hdi,
                                               element_location::IN_NEGATIVE_SIDE);
        auto lc_A = contrib.first.first;
        auto lc_B = -contrib.first.second;
        auto rhs_A = contrib.second.first;
        auto rhs_B = -contrib.second.second;

        if( sc )
            assembler_sc.assemble(msh, cl, lc_A, lc_B, rhs_A, rhs_B);
        else
            assembler.assemble(msh, cl, lc_A, lc_B, rhs_A, rhs_B);
    }

    if( sc )
        assembler_sc.finalize();
    else
        assembler.finalize();

    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

    if( sc )
        std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
    else
        std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;


    /************** SOLVE **************/
    tc.tic();
#if 1
    SparseLU<SparseMatrix<RealType>>  solver;
    Matrix<RealType, Dynamic, 1> sol;

    if( sc )
    {
        solver.analyzePattern(assembler_sc.LHS);
        solver.factorize(assembler_sc.LHS);
        sol = solver.solve(assembler_sc.RHS);
    }
    else
    {
        solver.analyzePattern(assembler.LHS);
        solver.factorize(assembler.LHS);
        sol = solver.solve(assembler.RHS);
    }
#endif
#if 0
    Matrix<RealType, Dynamic, 1> sol;
    cg_params<RealType> cgp;
    cgp.histfile = "cuthho_cg_hist.dat";
    cgp.verbose = true;
    cgp.apply_preconditioner = true;
    if( sc )
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
        cgp.max_iter = assembler_sc.LHS.rows();
        conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
    }
    else
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
        cgp.max_iter = assembler.LHS.rows();
        conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
    }
#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    /************** POSTPROCESS **************/



    postprocess_output<RealType>  postoutput;

    auto uT_l2_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_uT_norm.dat");
    auto uT1_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_uT1.dat");
    auto uT2_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_uT2.dat");
    auto p_gp    = std::make_shared< gnuplot_output_object<RealType> >("fictdom_p.dat");

    tc.tic();
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
    RealType    L2_pressure_error = 0.0;
    for (auto& cl : msh.cells)
    {
        bool hide_fict_dom = true; // hide the fictitious domain in the gnuplot outputs
        if (hide_fict_dom && location(msh,cl) == element_location::IN_POSITIVE_SIDE)
            continue;

        vector_cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        cell_basis<cuthho_poly_mesh<RealType>, RealType> s_cb(msh, cl, hdi.face_degree());

        auto cbs = cb.size();

        Matrix<RealType, Dynamic, 1> locdata_vel, locdata_p;
        if( sc )
        {
            locdata_vel = assembler_sc.take_velocity(msh, cl, sol);
            locdata_p   = assembler_sc.take_pressure(msh, cl, sol);
        }
        else
        {
            locdata_vel = assembler.take_velocity(msh, cl, sol);
            locdata_p   = assembler.take_pressure(msh, cl, sol);
        }

        Matrix<RealType, Dynamic, 1> cell_v_dofs = locdata_vel.head(cbs);

        //auto bar = barycenter(msh, cl, element_location::IN_NEGATIVE_SIDE);

        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE ||
             location(msh, cl) == element_location::ON_INTERFACE )
        {
            //Matrix<RealType, 1, 2> real_grad_int = Matrix<RealType, 1, 2>::Zero();
            //Matrix<RealType, 1, 2> comp_grad_int = Matrix<RealType, 1, 2>::Zero();
            auto qps = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 0; i < cbs; i++ )
                    grad += cell_v_dofs(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;

                H1_error += qp.second * inner_product(grad_diff , grad_diff);


                /* L2 - error */
                auto t_phi = cb.eval_basis( qp.first );
                auto v = t_phi.transpose() * cell_v_dofs;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                uT1_gp->add_data( qp.first, v(0) );
                uT2_gp->add_data( qp.first, v(1) );
                uT_l2_gp->add_data( qp.first, std::sqrt( v(0)*v(0) + v(1)*v(1) ) );

                /* L2 - pressure - error */
                auto s_cphi = s_cb.eval_basis( qp.first );
                RealType p_num = s_cphi.dot(locdata_p);
                RealType p_diff = test_case.sol_p( qp.first ) - p_num;
                L2_pressure_error += qp.second * p_diff * p_diff;

                p_gp->add_data( qp.first, p_num );
            }
        }
    }

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
    std::cout << bold << green << "L2 - pressure - error:                " << std::sqrt(L2_pressure_error) << std::endl;

    postoutput.add_object(uT_l2_gp);
    postoutput.add_object(uT1_gp);
    postoutput.add_object(uT2_gp);
    postoutput.add_object(p_gp);
    postoutput.write();

    stokes_test_info<RealType> TI;
    TI.H1_vel = std::sqrt(H1_error);
    TI.L2_vel = std::sqrt(L2_error);
    TI.L2_p   = std::sqrt(L2_pressure_error);

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;

    return TI;
}



//////////////////////////////  INTERFACE METHODS  ///////////////////////////

template<typename T, size_t ET, typename testType>
class stokes_interface_method
{
    using Mat  = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

protected:
    bool sym_grad;

    stokes_interface_method(bool sym)
        : sym_grad(sym) {}

    virtual std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi)
    {
    }

public:
    std::pair<Mat, Vect>
    make_contrib_uncut(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType test_case)
    {
        T kappa;
        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
            kappa = test_case.parms.kappa_1;
        else
            kappa = test_case.parms.kappa_2;

        Mat gr2;
        if(sym_grad)
            gr2 = make_hho_gradrec_sym_matrix(msh, cl, hdi).second;
        else
            gr2 = make_hho_gradrec_matrix(msh, cl, hdi).second;
        Mat stab = make_hho_vector_naive_stabilization(msh, cl, hdi);
        Mat lc = kappa * (gr2 + stab);
        auto dr = make_hho_divergence_reconstruction(msh, cl, hdi);
        Mat f = make_vector_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);

        size_t v_size = gr2.rows();
        size_t p_size = dr.first.rows();
        size_t loc_size = v_size + p_size;
        Mat lhs = Mat::Zero( loc_size, loc_size );
        Vect rhs = Vect::Zero( loc_size );

        lhs.block(0, 0, v_size, v_size) = lc;
        lhs.block(0, v_size, v_size, p_size) = -dr.second.transpose();
        lhs.block(v_size, 0, p_size, v_size) = -dr.second;

        rhs.head(f.rows()) = f;
        return std::make_pair(lhs, rhs);
    }


    std::pair<Mat, Vect>
    make_contrib(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const testType test_case, const hho_degree_info hdi)
    {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return make_contrib_uncut(msh, cl, hdi, test_case);
        else // on interface
            return make_contrib_cut(msh, cl, test_case, hdi);
    }
};


////////////////////////  SYMMETRIC GRADREC INTERFACE METHOD


template<typename T, size_t ET, typename testType>
class Sym_gradrec_stokes_interface_method : public stokes_interface_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta, gamma_0;

    Sym_gradrec_stokes_interface_method(T eta_, T gamma_, bool sym)
        : stokes_interface_method<T,ET,testType>(sym), eta(eta_), gamma_0(gamma_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi)
    {
        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        level_set_function.cell_assignment( cl );
        ///////////////   LHS
        auto celdeg = hdi.cell_degree();
        auto pdeg = hdi.face_degree();
        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        // GR
        Mat gr2_n, gr2_p;
        if(this->sym_grad)
        {
            // ---- THIS CASE IS ROBUST FOR K_1 SIMILAR TO K_2 -----
            
            // 0.5 is the weight coefficient that scale the interface term between inner and outer interface (Omega_1 and Omega_2)
            /// Paper: Un Unfitted HHO method with cell agglomeration for elliptic interface pb.
            /// -->  This is the variant 2.5 (pag.7)
            gr2_n = make_hho_gradrec_sym_matrix_interface
                (msh, cl, level_set_function, hdi,element_location::IN_NEGATIVE_SIDE, 0.5).second;
            gr2_p = make_hho_gradrec_sym_matrix_interface
                (msh, cl, level_set_function, hdi,element_location::IN_POSITIVE_SIDE, 0.5).second;
        }
        else
        {
            gr2_n = make_hho_gradrec_matrix_interface
                (msh, cl, level_set_function, hdi,element_location::IN_NEGATIVE_SIDE, 0.5).second;
            gr2_p = make_hho_gradrec_matrix_interface
                (msh, cl, level_set_function, hdi,element_location::IN_POSITIVE_SIDE, 0.5).second;
        }

        // stab
        Mat stab = make_hho_vector_stabilization_interface(msh, cl, level_set_function, hdi,parms);
        // Penalty conforming to variant 2.5, paper "Un Unfitted HHO method.."
        Mat penalty = make_hho_cut_interface_vector_penalty(msh, cl, hdi, eta).block(0,0,cbs,cbs);
        stab.block(0, 0, cbs, cbs) += parms.kappa_2 * penalty;
        stab.block(0, cbs, cbs, cbs) -= parms.kappa_2 * penalty;
        stab.block(cbs, 0, cbs, cbs) -= parms.kappa_2 * penalty;
        stab.block(cbs, cbs, cbs, cbs) += parms.kappa_2 * penalty;
        // This is term \tilde{a_T} (eq.15), paper "Un Unfitted HHO method.."
        Mat lc = stab + parms.kappa_1 * gr2_n + parms.kappa_2 * gr2_p;

        // DR : Penalty divided:
        // (1.0 - coeff) * interface_term in NEGATIVE SIDE + coeff contribute into positive side
        // (coeff- 1.0 ) * interface_term in POSITIVE SIDE - coeff contribute into negative side
        auto dr_n = make_hho_divergence_reconstruction_interface
            (msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE, 0.5);
        auto dr_p = make_hho_divergence_reconstruction_interface
            (msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE, 0.5);


        Mat lhs = Mat::Zero(lc.rows() + 2*pbs, lc.rows() + 2*pbs);
        lhs.block(0, 0, lc.rows(), lc.rows()) = lc;
        lhs.block(0, lc.rows(), lc.rows(), pbs) -= dr_n.second.transpose();
        lhs.block(0, lc.rows() + pbs, lc.rows(), pbs) -= dr_p.second.transpose();
        lhs.block(lc.rows(), 0, pbs, lc.rows()) -= dr_n.second;
        lhs.block(lc.rows() + pbs, 0, pbs, lc.rows()) -= dr_p.second;


        // stokes stabilization terms
        auto stokes_stab = make_stokes_interface_stabilization(msh, cl, hdi, level_set_function);
        lhs.block(0, 0, 2*cbs, 2*cbs) -= gamma_0 * stokes_stab.block(0, 0, 2*cbs, 2*cbs);
        lhs.block(0, lc.rows(), 2*cbs, 2*pbs) -= gamma_0 * stokes_stab.block(0,2*cbs,2*cbs,2*pbs);
        lhs.block(lc.rows(), 0, 2*pbs, 2*cbs) -= gamma_0 * stokes_stab.block(2*cbs,0,2*pbs,2*cbs);
        lhs.block(lc.rows(), lc.rows(), 2*pbs, 2*pbs)
            -= gamma_0 * stokes_stab.block(2*cbs, 2*cbs, 2*pbs, 2*pbs);



        ////////////////    RHS

        Vect f = Vect::Zero(lc.rows());
        // neg part
        f.block(0, 0, cbs, 1) += make_vector_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                                 element_location::IN_NEGATIVE_SIDE);
        f.head(cbs) += 0.5*make_vector_flux_jump(msh,cl,celdeg, element_location::IN_NEGATIVE_SIDE,
                                                 test_case.neumann_jump);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_vector_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                                   element_location::IN_POSITIVE_SIDE);
        f.block(cbs, 0, cbs, 1)
            += 0.5 * make_vector_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                           test_case.neumann_jump);


        Vect rhs = Vect::Zero(lc.rows() + 2*pbs);
        rhs.head(lc.rows()) = f;

        // stokes stabilization rhs
        auto stab_rhs = make_stokes_interface_stabilization_RHS
            (msh, cl, hdi, level_set_function, test_case.neumann_jump);

        rhs.head(2*cbs) -= gamma_0 * stab_rhs.head(2*cbs);
        rhs.tail(2*pbs) -= gamma_0 * stab_rhs.tail(2*pbs);

        return std::make_pair(lhs, rhs);
    }
};

template<typename T, size_t ET, typename testType>
auto make_sym_gradrec_stokes_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                              const T gamma_, testType test_case, bool sym)
{
    return Sym_gradrec_stokes_interface_method<T, ET, testType>(eta_, gamma_, sym);
}


template<typename Mesh, typename testType, typename meth>
stokes_test_info<typename Mesh::coordinate_type>
run_cuthho_interface(const Mesh& msh, size_t degree, meth method, testType test_case)
{
    using RealType = typename Mesh::coordinate_type;

    auto level_set_function = test_case.level_set_;

    auto rhs_fun = test_case.rhs_fun;
    auto sol_vel = test_case.sol_vel;
    auto sol_p = test_case.sol_p;
    auto vel_grad = test_case.vel_grad;
    auto bcs_vel = test_case.bcs_vel;
    auto neumann_jump = test_case.neumann_jump;
    struct params<RealType> parms = test_case.parms;

    timecounter tc;

    bool sc = true; // static condensation


    /************** ASSEMBLE PROBLEM **************/
    hho_degree_info hdi(degree+1, degree);

    tc.tic();
    auto assembler = make_stokes_interface_assembler(msh, bcs_vel, hdi);
    auto assembler_sc = make_stokes_interface_condensed_assembler(msh, bcs_vel, hdi);
    for (auto& cl : msh.cells)
    {
        auto contrib = method.make_contrib(msh, cl, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;

        if( sc )
            assembler_sc.assemble(msh, cl, lc, f);
        else
            assembler.assemble(msh, cl, lc, f);
    }

    if( sc )
        assembler_sc.finalize();
    else
        assembler.finalize();

    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

    if( sc )
        std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
    else
        std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;

    /************** SOLVE **************/
    tc.tic();
#if 1
    SparseLU<SparseMatrix<RealType>>  solver;
    Matrix<RealType, Dynamic, 1> sol;

    if( sc )
    {
        solver.analyzePattern(assembler_sc.LHS);
        solver.factorize(assembler_sc.LHS);
        sol = solver.solve(assembler_sc.RHS);
    }
    else
    {
        solver.analyzePattern(assembler.LHS);
        solver.factorize(assembler.LHS);
        sol = solver.solve(assembler.RHS);
    }
#endif
#if 0
    Matrix<RealType, Dynamic, 1> sol;
    cg_params<RealType> cgp;
    cgp.histfile = "cuthho_cg_hist.dat";
    cgp.verbose = true;
    cgp.apply_preconditioner = true;
    if( sc )
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
        cgp.max_iter = assembler_sc.LHS.rows();
        conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
    }
    else
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
        cgp.max_iter = assembler.LHS.rows();
        conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
    }
#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    /************** POSTPROCESS **************/


    postprocess_output<RealType>  postoutput;

    auto uT1_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT1.dat");
    auto uT2_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT2.dat");
    auto p_gp    = std::make_shared< gnuplot_output_object<RealType> >("interface_p.dat");

    tc.tic();
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
    RealType    L2_pressure_error = 0.0;
    for (auto& cl : msh.cells)
    {
        vector_cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        cell_basis<cuthho_poly_mesh<RealType>, RealType> pb(msh, cl, hdi.face_degree());
        auto cbs = cb.size();
        auto pbs = pb.size();

        Matrix<RealType, Dynamic, 1> vel_locdata_n, vel_locdata_p, vel_locdata;
        Matrix<RealType, Dynamic, 1> P_locdata_n, P_locdata_p, P_locdata;
        Matrix<RealType, Dynamic, 1> vel_cell_dofs_n, vel_cell_dofs_p, vel_cell_dofs;

        if (location(msh, cl) == element_location::ON_INTERFACE)
        {
            if( sc )
            {
                vel_locdata_n = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                vel_locdata_p = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata_n = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                P_locdata_p = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
            {
                vel_locdata_n = assembler.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                vel_locdata_p = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata_n = assembler.take_pressure(msh,cl, sol, element_location::IN_NEGATIVE_SIDE);
                P_locdata_p = assembler.take_pressure(msh,cl, sol, element_location::IN_POSITIVE_SIDE);
            }

            vel_cell_dofs_n = vel_locdata_n.head(cbs);
            vel_cell_dofs_p = vel_locdata_p.head(cbs);


            auto qps_n = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += vel_cell_dofs_n(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                H1_error += qp.second * inner_product(grad_diff , grad_diff);


                /* Compute L2-error */
                auto t_phi = cb.eval_basis( qp.first );
                auto v = t_phi.transpose() * vel_cell_dofs_n;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                uT1_gp->add_data( qp.first, v(0) );
                uT2_gp->add_data( qp.first, v(1) );

                /* L2 - pressure - error */
                auto p_phi = pb.eval_basis( qp.first );
                RealType p_num = p_phi.dot(P_locdata_n);
                RealType p_diff = test_case.sol_p( qp.first ) - p_num;
                L2_pressure_error += qp.second * p_diff * p_diff;

                p_gp->add_data( qp.first, p_num );
            }

            auto qps_p = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += vel_cell_dofs_p(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                H1_error += qp.second * inner_product(grad_diff , grad_diff);

                /* Compute L2-error */
                auto t_phi = cb.eval_basis( qp.first );
                auto v = t_phi.transpose() * vel_cell_dofs_p;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                uT1_gp->add_data( qp.first, v(0) );
                uT2_gp->add_data( qp.first, v(1) );

                /* L2 - pressure - error */
                auto p_phi = pb.eval_basis( qp.first );
                RealType p_num = p_phi.dot(P_locdata_p);
                RealType p_diff = test_case.sol_p( qp.first ) - p_num;
                L2_pressure_error += qp.second * p_diff * p_diff;

                p_gp->add_data( qp.first, p_num );
            }
        }
        else
        {
            if( sc )
            {
                vel_locdata = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
            {
                vel_locdata = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata = assembler.take_pressure(msh,cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            vel_cell_dofs = vel_locdata.head(cbs);

            auto qps = integrate(msh, cl, 2*hdi.cell_degree());
            for (auto& qp : qps)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += vel_cell_dofs(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                H1_error += qp.second * inner_product(grad_diff , grad_diff);

                /* Compute L2-error */
                auto t_phi = cb.eval_basis( qp.first );
                auto v = t_phi.transpose() * vel_cell_dofs;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                uT1_gp->add_data( qp.first, v(0) );
                uT2_gp->add_data( qp.first, v(1) );

                /* L2 - pressure - error */
                auto p_phi = pb.eval_basis( qp.first );
                RealType p_num = p_phi.dot(P_locdata);
                RealType p_diff = test_case.sol_p( qp.first ) - p_num;
                L2_pressure_error += qp.second * p_diff * p_diff;

                p_gp->add_data( qp.first, p_num );
            }
        }

    }

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:               " << std::sqrt(L2_error) << std::endl;
    std::cout << bold << green << "Pressure L2-norm absolute error:      " << std::sqrt(L2_pressure_error) << std::endl;

    postoutput.add_object(uT1_gp);
    postoutput.add_object(uT2_gp);
    postoutput.add_object(p_gp);
    postoutput.write();



    stokes_test_info<RealType> TI;
    TI.H1_vel = std::sqrt(H1_error);
    TI.L2_vel = std::sqrt(L2_error);
    TI.L2_p = std::sqrt(L2_pressure_error);

    if (false)
    {
        /////////////// compute condition number
        SparseMatrix<RealType> Mat;
        // Matrix<RealType, Dynamic, Dynamic> Mat;
        if (sc)
            Mat = assembler_sc.LHS;
        else
            Mat = assembler.LHS;


        RealType sigma_max, sigma_min;

        // Construct matrix operation object using the wrapper class SparseSymMatProd
        Spectra::SparseSymMatProd<RealType> op(Mat);
        // Construct eigen solver object, requesting the largest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::LARGEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > max_eigs(&op, 1, 10);
        max_eigs.init();
        max_eigs.compute();
        if(max_eigs.info() == Spectra::SUCCESSFUL)
            sigma_max = max_eigs.eigenvalues()(0);


        // Construct eigen solver object, requesting the smallest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::SMALLEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > min_eigs(&op, 1, 10);

        min_eigs.init();
        min_eigs.compute();
        if(min_eigs.info() == Spectra::SUCCESSFUL)
            sigma_min = min_eigs.eigenvalues()(0);

        // compute condition number
        RealType cond = sigma_max / sigma_min;
        TI.cond = cond;
        std::cout << "sigma_max = " << sigma_max << "   sigma_min = "
                  << sigma_min << "  cond = " << cond
                  << std::endl;
    }
    else
        TI.cond = 0.0;

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


    return TI;
}



/////////////////////////   AUTOMATIC TESTS  //////////////////////////


void convergence_test(void)
{
    using T = double;

    std::vector<size_t> mesh_sizes, pol_orders;

    // meshes
    mesh_sizes.push_back(8);
    mesh_sizes.push_back(16);
    mesh_sizes.push_back(32);
    mesh_sizes.push_back(64);
    // mesh_sizes.push_back(128);
    // mesh_sizes.push_back(256);

    // polynomial orders
    pol_orders.push_back(0);
    pol_orders.push_back(1);
    pol_orders.push_back(2);
    pol_orders.push_back(3);


    // export to files ...
    std::vector<std::string> files;
    files.push_back("./output/test_k0.txt");
    files.push_back("./output/test_k1.txt");
    files.push_back("./output/test_k2.txt");
    files.push_back("./output/test_k3.txt");

    for (std::vector<size_t>::iterator it = pol_orders.begin(); it != pol_orders.end(); it++)
    {
        size_t k = *it;

        std::cout << "start tests for k = " << k << std::endl;

        // init the file
        std::ofstream output;
        output.open (files.at(*it), std::ios::in | std::ios::trunc);
        if (!output.is_open())
            throw std::logic_error("file not open");

        // output << "N\th\tH1\tordre1\tL2\tordre2" << std::endl;
        output << "N\th\tH1\tordre1\tL2\tordre2\tLp\tordre3\tcond" << std::endl;

        // convergence tests
        T previous_H1 = 0.0;
        T previous_L2 = 0.0;
        T previous_p = 0.0;
        T previous_h = 0.0;
        for (std::vector<size_t>::iterator it_msh = mesh_sizes.begin();
             it_msh != mesh_sizes.end(); it_msh++)
        {
            size_t N = *it_msh;

            // init mesh (with agglomeration)
            mesh_init_params<T> mip;
            mip.Nx = N;
            mip.Ny = N;
            cuthho_poly_mesh<T> msh(mip);
            size_t int_refsteps = 1;
            //T radius = 1.0/3.0;
            //auto circle_level_set_function = circle_level_set<T>(radius, 0.5, 0.5);

            // auto level_set_function = flower_level_set<T>(0.31, 0.5, 0.5, 4, 0.04);
            // auto level_set_function = circle_level_set<T>(radius, 0.5, 0.5);
            // auto level_set_function = square_level_set<T>(1.05, -0.05, -0.05, 1.05);
            // auto level_set_function = square_level_set<T>(1.0, -0.0, -0.0, 1.0);
            auto level_set_function = square_level_set<T>(0.76, 0.24, 0.24, 0.76);
            detect_node_position(msh, level_set_function);
            detect_cut_faces(msh, level_set_function);
            if(1)  // AGGLOMERATION
            {
                detect_cut_cells(msh, level_set_function);
                detect_cell_agglo_set(msh, level_set_function);
                make_neighbors_info_cartesian(msh);
                refine_interface(msh, level_set_function, int_refsteps);
                make_agglomeration(msh, level_set_function);
            }
            else  // NODE DISPLACEMENT
            {
                move_nodes(msh, level_set_function);
                detect_cut_faces(msh, level_set_function); //do it again to update intersection points
                detect_cut_cells(msh, level_set_function);
                refine_interface(msh, level_set_function, int_refsteps);
            }

            // compute solution/errors
            stokes_test_info<T> TI;

            if(1)
            {
                // auto test_case = make_test_case_stokes_1(msh, level_set_function);
                auto test_case = make_test_case_stokes_2(msh, level_set_function);
                TI = run_cuthho_fictdom(msh, k, test_case);
                // auto method = make_sym_gradrec_stokes_interface_method(msh, 1.0, 0.0, test_case, true);
                // TI = run_cuthho_interface(msh, k, method, test_case);
            }

            // report info in the file
            T h = 1.0/N;
            if (it_msh == mesh_sizes.begin())
            {
                output << N << "\t" << h << "\t" << TI.H1_vel << "\t" << "."
                       << "\t" << TI.L2_vel << "\t" << "." << "\t" << TI.L2_p
                       << "\t" << "." << "\t" << TI.cond
                       << std::endl;
            }
            else
            {
                T orderH = log(previous_H1 / TI.H1_vel) / log(previous_h / h);
                T orderL = log(previous_L2 / TI.L2_vel) / log(previous_h / h);
                T orderp = log(previous_p / TI.L2_p) / log(previous_h / h);
                output << N << "\t" << h << "\t" << TI.H1_vel << "\t" << orderH
                       << "\t" << TI.L2_vel << "\t" << orderL << "\t" << TI.L2_p
                       << "\t" << orderp << "\t" << TI.cond
                       << std::endl;
            }
            previous_H1 = TI.H1_vel;
            previous_L2 = TI.L2_vel;
            previous_p = TI.L2_p;
            previous_h = h;
        }
        // close the file
        output.close();
    }

    // update the gnuplot curves
    system("gnuplot './output/gnuplot_script_stokes.txt'");

    // update the .pdf file
    system("pdflatex ./output/autom_tests_stokes.tex");

    // open the .pdf file
    system("xdg-open ./autom_tests_stokes.pdf");
}


//////////////////////////     MAIN        ////////////////////////////
#if 0
int main(int argc, char **argv)
{
    convergence_test();
    // tests_stabilization();
    // interface_residus();
    return 1;
}
#endif

#if 0
int main(int argc, char **argv)
{
    using RealType = double;

    size_t degree           = 0;
    size_t int_refsteps     = 4;

    bool dump_debug         = false;
    bool solve_interface    = false;
    bool solve_fictdom      = false;
    bool agglomeration      = false;

    mesh_init_params<RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;

    /* k <deg>:     method degree
     * M <num>:     number of cells in x direction
     * N <num>:     number of cells in y direction
     * r <num>:     number of interface refinement steps
     *
     * i:           solve interface problem
     * f:           solve fictitious domain problem
     *
     * D:           use node displacement to solve bad cuts (default)
     * A:           use agglomeration to solve bad cuts
     *
     * d:           dump debug data
     */

    int ch;
    while ( (ch = getopt(argc, argv, "k:M:N:r:ifDAd")) != -1 )
    {
        switch(ch)
        {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
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


    timecounter tc;

    /************** BUILD MESH **************/
    tc.tic();
    cuthho_poly_mesh<RealType> msh(mip);
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;
    /************** LEVEL SET FUNCTION **************/
    RealType radius = 1.0/3.0;
    auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);
    // auto level_set_function = line_level_set<RealType>(0.5);
    // auto level_set_function = flower_level_set<RealType>(0.31, 0.5, 0.5, 4, 0.04);
    /************** DO cutHHO MESH PROCESSING **************/

    tc.tic();
    detect_node_position(msh, level_set_function);
    detect_cut_faces(msh, level_set_function);

    if (agglomeration)
    {
        detect_cut_cells(msh, level_set_function);
        detect_cell_agglo_set(msh, level_set_function);
        make_neighbors_info_cartesian(msh);
        // make_neighbors_info(msh);
        refine_interface(msh, level_set_function, int_refsteps);
        make_agglomeration(msh, level_set_function);
    }
    else
    {
        move_nodes(msh, level_set_function);
        detect_cut_faces(msh, level_set_function); //do it again to update intersection points
        detect_cut_cells(msh, level_set_function);
        refine_interface(msh, level_set_function, int_refsteps);
    }


    tc.toc();
    std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

    if (dump_debug)
    {
        dump_mesh(msh);
        output_mesh_info(msh, level_set_function);
    }

    output_mesh_info(msh, level_set_function);

    // auto test_case = make_test_case_stokes_1(msh, level_set_function);
    auto test_case = make_test_case_stokes_2(msh, level_set_function);

    auto method = make_sym_gradrec_stokes_interface_method(msh, 1.0, 0.0, test_case, true);

    if (solve_interface)
        run_cuthho_interface(msh, degree, method, test_case);

    if (solve_fictdom)
        run_cuthho_fictdom(msh, degree, test_case);


    return 0;
}
#endif



// Interface Stokes Problem: DIRICHLET BDRY COND SU TUTTO IL DOMINIO
#if 0
int main(int argc, char **argv)
{
    using RealType = double;

    size_t degree           = 0;
    size_t int_refsteps     = 4;
    size_t degree_FEM       = 0;

    bool dump_debug         = false;
    bool solve_interface    = false;
    bool solve_fictdom      = false;
    bool agglomeration      = false;
    
    bool high_order = false ; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
    bool cut_off_active = false ; // IF FALSE IS SMOOTH, IF TRUE  CUT_OFF

    mesh_init_params<RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;
    RealType d = 0.5;
    size_t T_N = 0;
    /* k <deg>:     method degree
     * g<deg>:  method FEM degree
     * M <num>:     number of cells in x direction
     * N <num>:     number of cells in y direction
     * r <num>:     number of interface refinement steps
     *
     * i:           solve interface problem
     * f:           solve fictitious domain problem
     *
     * D:           use node displacement to solve bad cuts (default)
     * A:           use agglomeration to solve bad cuts
     *
     * d:           dump debug data
     */

    int ch;
    while ( (ch = getopt(argc, argv, "k:q:M:N:r:T:ifDAdhc")) != -1 )
    {
        switch(ch)
        {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'q':
                degree_FEM = atoi(optarg);
                break;
        
            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;
            
            case 'T':
                T_N = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;
            
            case 'h':
                high_order = true;
            break;
            
            case 'c':
                cut_off_active = true;
            break;
            

            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;

    
    
    timecounter tc;

    /************** BUILD MESH **************/
    tc.tic();
    cuthho_poly_mesh<RealType> msh(mip);
    typedef cuthho_poly_mesh<RealType> Mesh;
    offset_definition(msh);
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;
       
    /*
#ifdef HAVE_INTEL_TBB
    size_t n_cells = msh.cells_size();
    std::cout<<" I m in parallel zone"<<std::endl;
    tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
    [&msh] (size_t & cell_ind){
            auto& cell = msh.cells[cell_ind];
             }
    );
#else
    std::cout<<" I m in sequential zone"<<std::endl;
    for (size_t cell_ind = 0; cell_ind < msh.cells.size(); cell_ind++)
    {
        auto& cell = msh.cells[cell_ind];
    }
#endif
    */
    
    /************** FINITE ELEMENT INITIALIZATION **************/
    auto fe_data = Finite_Element<RealType,Mesh>( msh , degree_FEM , mip ) ;
    typedef Finite_Element<RealType,Mesh> FiniteSpace;
    
    /************** ANALYTIC LEVEL SET FUNCTION  **************/
    typedef RealType T;
    
    bool circle_new = false , circle_old = false , ellipse_new = false , ellipse_old = true ;
    
    RealType radius_a , radius_b , radius ;
    RealType x_centre = 0.5;
    RealType y_centre = 0.5;
    RealType C ;
    if(circle_old)
    {
        radius = 1.0/9.0;
    }
        
    if(circle_new)
    {
        radius = 1.0/9.0;
    }
    if(ellipse_old)
    {
        radius_a = 1.0/6.0;
        radius_b = 1.0/12.0;
        //radius_a = 1.0/9.0;
        //radius_b = 1.0/9.0;
        std::cout << bold << yellow << "Initial Analytic Area of the ELLIPSE: "<< M_PI*radius_a*radius_b << std::endl;
        radius = sqrt( radius_a * radius_b ) ;
        std::cout << bold << yellow << "Final radius expected of the circle : " << radius <<reset<<std::endl;
    }
    if(ellipse_new)
    {
        radius_a = 1.0/6.0;
        radius_b = 1.0/12.0;
        //radius_a = 1.0/9.0;
        //radius_b = 1.0/9.0;
        std::cout << bold << yellow << "Initial Analytic Area of the ELLIPSE: "<< M_PI*radius_a*radius_b << std::endl;
        radius = sqrt( radius_a * radius_b ) ;
        std::cout << bold << yellow << "Final radius expected of the circle : " << radius <<reset<<std::endl;
    }
        
    
    /// THIS DATA BELOW HAS TO BE UPLOAD DEPENDING ON THE PROBLEM.
    
    auto level_set_function_anal = elliptic_level_set<RealType>( radius_a, radius_b, x_centre, y_centre);

    //auto level_set_function_anal = circle_level_set<RealType>(radius, x_centre, y_centre );
    
    //auto level_set_function_anal = elliptic_level_set_new<RealType>( radius_a, radius_b, x_centre, y_centre , 1.0/16.0 );
    
    T h = std::max( fe_data.hx , fe_data.hy) ;
    //auto level_set_function_anal = circle_level_set_new<RealType>(radius, x_centre, y_centre , 4.0*h );
    
    
    
    typedef  elliptic_level_set<T> Fonction;
    //typedef  circle_level_set<T> Fonction;
    //typedef  circle_level_set_new <T> Fonction;
    //typedef  elliptic_level_set_new <T> Fonction;
    
    
   
    /**************  VELOCITY FIELD   **************/ // JUST TO CHECK IMPLEMENTATION
  
    auto u_projected = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
    
    /************** LEVEL SET FUNCTION DISCRETISATION **************/
    std::cout<<"degree FEM "<<degree_FEM<<std::endl;
   
    
    auto level_set_function = L2projected_level_set_high_order< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);
    
    //level_set_function.smooth_cut_off( C , x_centre , y_centre , radius );
    //level_set_function.cut_off( C );
    
    
    //testing_level_set(msh,level_set_function,level_set_function_anal);
   
    T r0 , rad_max ;
    
    if( circle_old || ellipse_old )
    {
        
        if( circle_old ) // FOR CIRCLE
        {
            r0 = radius + (x_centre - radius + 2.0/16.0)/2.0;
            //r0 = radius + h*sqrt(2.0);
            C = r0*r0 - radius*radius ;
        }
        if( ellipse_old )  // FOR ELLIPSE
        {
            rad_max = std::max(radius_a,radius_b) ;
            T r0 = rad_max + (x_centre - rad_max + 2.0/16.0)/2.0;
            //r0 = rad_max + h*sqrt(2.0);
            C = r0*r0*radius_b*radius_b - radius_a*radius_a*radius_b*radius_b;
        }
             
        T C_bis = std::abs(level_set_function.phi_min) ;
        if( C_bis > C)
            C = C_bis;
             
        if(cut_off_active)
        {
            level_set_function.cut_off( C );
            std::cout<<bold<<yellow<<"----> USING CUT_OFF!!!!! "<<reset<<std::endl;
            std::cout<<"C = "<<C<<std::endl;
        }
        else
        {
            //C = 0.2;  // Imposed max value at the boundary
            T delta = r0/8.0;
            std::cout<<"r_max = "<<rad_max<<" , r0 = "<<r0<<" , delta = "<<delta<<" , max (hx,hy) = "<< h <<std::endl;
            if( circle_old ) // FOR CIRCLE
            {
                std::cout<<"Value in alfa in position radius R  = "<<( radius - r0)/delta << " -> its approximation error = "<< 1.0 - (1.0 - tanh((radius - r0 )/delta) )/2.0<<std::endl;
                T pos_bdry = std::min(x_centre , y_centre) ;
                std::cout<<"value in alfa in the boundary = "<<(pos_bdry - r0)/delta<< " -> its approximation error = "<< (1.0 - tanh((pos_bdry - r0 )/delta) )/2.0<<std::endl;
              
            }
            if( ellipse_old ) // FOR CIRCLE
            {
                std::cout<<"Value in alfa in position radius r_a  = "<<( radius_a - r0)/delta << " -> its approximation error = "<< 1.0 - (1.0 - tanh((radius_a - r0 )/delta) )/2.0<<std::endl;
                std::cout<<"Value in alfa in position radius r_b  = "<<( radius_b - r0)/delta << " -> its approximation error = "<< 1.0 - (1.0 - tanh((radius_b - r0 )/delta) )/2.0<<std::endl;
                T pos_bdry = std::min(x_centre , y_centre) ;
                std::cout<<"value in alfa in the boundary = "<<(pos_bdry - r0)/delta<< " -> its approximation error = "<< (1.0 - tanh((pos_bdry - r0 )/delta) )/2.0<<std::endl;
              
            }
            
            level_set_function.smooth_cut_off( C , r0 , delta, x_centre , y_centre , radius ,radius_a ,radius_b );
            std::cout<<bold<<yellow<<"----> USING SMOOTH!!!!! "<<reset<<std::endl;
            std::cout<<"C = "<<C<<std::endl;
        }
             
    }
    
   
    if(high_order)
        std::cout<<bold<<yellow<<"----> USING phi_HP HIGH order!!!!! "<<reset<<std::endl;
    else
        std::cout<<bold<<yellow<<"----> USING phi_L LOW order!!!!! "<<reset<<std::endl;
    
    
    testing_level_set(msh,level_set_function,level_set_function_anal);
     
    // Initiliatisation data for time routine
    auto crr_mesh =  Current_Mesh<Mesh>(msh);
    
    /// Initialisation area , mass -> I calculate them just the first and last time step.
    T initial_area = 0. , initial_mass = 0.;
    T area_previous_time = 0. , mass_previous_time = 0. , dt = 0. ;
    
    /// DATA CHECK INITIALISATION
    T d_a = 0. ; // It is related to area-> calculated just first and last time
    T error_normal_global = 0. ;
    T error_normal_local = 0. ;
    T perimeter_initial = 0. ;
    T centre_mass_x_inital = 0. , centre_mass_y_inital = 0. ;
    
    
    /// ERROR FEM (transport problem) wrt ANALYTICAL SOLUTION
    //T l1_L1_error = 0 , l2_L1_error = 0 , linf_L1_error = 0;
    //T l1_L2_error = 0 ,  l2_L2_error = 0 , linf_L2_error = 0;
    //T l1_Linf_error = 0 ,  l2_Linf_error = 0 , linf_Linf_error = 0;
    
    //T l1_W11_error = 0 , l2_W11_error = 0 , linf_W11_error = 0;
    //T l1_W12_error = 0 ,  l2_W12_error = 0 , linf_W12_error = 0;
    //T l1_W1inf_error = 0 ,  l2_W1inf_error = 0 , linf_W1inf_error = 0;
    //size_t time_step = 0;
    T check = 10.0;
    T time_pos = 0.;
    
    T tot_time = 0.;
    //while(check > 0.0112265)
    //{
    for (size_t time_step = 0; time_step<=T_N; time_step++)
    {
       
        // ************** Re-Initialization mesh **************
        crr_mesh.current_mesh = msh;
        Mesh msh_i =  crr_mesh.current_mesh;
        offset_definition(msh_i);
        

        //************ DO cutHHO MESH PROCESSING **************
        tc.tic();
        detect_node_position2(msh_i, level_set_function); // In cuthho_geom
        detect_cut_faces2(msh_i, level_set_function); // In cuthho_geom
        
        if (agglomeration)
        {
            detect_cut_cells2(msh_i, level_set_function); // In cuthho_geom
            detect_cell_agglo_set(msh_i, level_set_function); // Non serve modificarla
            make_neighbors_info_cartesian(msh_i); // Non serve modificarla
            //refine_interface_angle(msh_i, level_set_function, int_refsteps);
            refine_interface_pro(msh_i, level_set_function, int_refsteps);
            //refine_interface2(msh_i, level_set_function, int_refsteps); // IN cuthho_geom
            make_agglomeration(msh_i, level_set_function); // Non serve modificarla
        }
        else
        {
            //move_nodes(msh_i, level_set_function);
            //detect_cut_faces2(msh_i, level_set_function); //do it again to update intersection points
            detect_cut_cells2(msh_i, level_set_function);
            //refine_interface2(msh_i, level_set_function, int_refsteps);
            //refine_interface_angle(msh_i, level_set_function, int_refsteps);
            refine_interface_pro(msh_i, level_set_function, int_refsteps);
        }
       
        tc.toc();
        std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

        if (dump_debug)
        {
            dump_mesh(msh_i);
            output_mesh_info(msh_i, level_set_function);
        }
   
        // IN cuthho_export..Points/Nodes don't change-> it's fast
        if(time_step == 0){
            output_mesh_info2_pre_FEM(msh_i, level_set_function); // IN cuthho_export
        }
            //output_mesh_info2(msh_i, level_set_function);
        
        typedef L2projected_level_set_high_order< Mesh , Fonction , FiniteSpace , T > Level_Set;
        auto ls_cell = LS_cell_L2proj_high_order< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);
        
        
        u_projected.set_agglo_mesh( msh_i );
        // CALCULATION OF AREA AND MASS AT TIME STEP t^n
        // CALCULATION ALSO OF CENTRE OF MASS
        
        /// DATA CHECK INITIALISATION
        T area0 = 0. , mass0 = 0. , global_mass0 = 0. ;
        T diff_area = 0. , diff_mass = 0. ;
        T error_normal_global0 = 0. ;
        T centre_mass0_x = 0. , centre_mass0_y = 0. ;
        T perimeter0 = 0.;
        T normal_interface_status = 0. ;
        size_t counter_interface_pts = 0;
        T divergence_error = 0.;
        // PLOTTING OF NORMAL
                 
        
        postprocess_output<double> postoutput_vec;
        auto vec_normal = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_Stokes_initial.dat");
        
        postprocess_output<double> postoutput_div2;
        auto test_interface_divergence2  = std::make_shared< gnuplot_output_object<double> >("divergence_interface_Stokes_initial.dat");
        
       
        std::vector<T> val_u_nx , val_u_ny , val_u_n ;
        std::vector< point<T, 2> > interface_points_plot ;
        std::vector< std::pair<T,T> > interface_normals ;
        
        for(auto& cl : msh_i.cells)
        {
            ls_cell.cell_assignment(cl);
            u_projected.cell_assignment(cl);
            
            if( location(msh_i, cl) == element_location::IN_NEGATIVE_SIDE || location(msh_i, cl) == element_location::ON_INTERFACE )
            {
                T partial_area = measure( msh_i, cl, element_location::IN_NEGATIVE_SIDE);
                area0 += partial_area;
                auto qps = integrate( msh_i , cl , 2*degree_FEM+1 , element_location::IN_NEGATIVE_SIDE);
                for(auto& qp:qps){
                    mass0 += qp.second * ls_cell(qp.first);
                    centre_mass0_x += qp.second * qp.first.x() ;
                    centre_mass0_y += qp.second * qp.first.y() ;
                }
            }
            if(cl.user_data.location == element_location::ON_INTERFACE)
            {
                /*
                std::cout<<"CELL = "<<offset(msh,cl)<<std::endl;
                if(offset(msh,cl) == 101 || offset(msh,cl) == 104 )
                {
                    for(size_t kk = 0 ; kk < u_projected.sol_HHO.first.rows() ; kk++ )
                    std::cout<< u_projected.sol_HHO.first(kk,offset(msh,cl)) << " " << u_projected.sol_HHO.second(kk,offset(msh,cl)) << '\n';
                    std::cout<<'\n'<<std::endl;
                }
                */
                for(auto interface_point = cl.user_data.interface.begin() ; interface_point < cl.user_data.interface.end() -1 ; interface_point++ )
                {
                    
                    
                    T val = ls_cell.divergence( *interface_point );
                    divergence_error += pow((std::abs(val) - 1.0/radius),2) ;
                    
                    Eigen::Matrix<T,2,1> normal = ls_cell.normal(*interface_point);
                    std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                    
                    interface_normals.push_back( normal_vec ) ;
                    
                    if( time_step == 0 )
                    {
                        vec_normal->add_data(*interface_point,normal_vec);
                        test_interface_divergence2->add_data(*interface_point , val);
                    }
                    
                    perimeter0 += ( *(interface_point+1) - *interface_point ).to_vector().norm();
                    
                    normal_interface_status += pow( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) , 2) + pow( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) , 2 );
                    
                    interface_points_plot.push_back(*(interface_point)) ;
                    val_u_nx.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) );
                    val_u_ny.push_back( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                    val_u_n.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) + u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                    //std::cout<<"*(interface_point) = "<<*(interface_point)<<std::endl;
                    //std::cout<<" u_projected primo = "<<u_projected(*(interface_point)).first  << " u_projected second = "<< u_projected(*(interface_point)).second  << " somma tot con olds = "<<normal_interface_status<<std::endl;
                    //std::cout<<" ls_cell.normal(*(interface_point))(0)  = "<<ls_cell.normal(*(interface_point))(0)  << " ls_cell.normal(*(interface_point))(1)  = "<< ls_cell.normal(*(interface_point))(1)  << std::endl;
                    
                    counter_interface_pts++;
                       
                }
                
                T val = ls_cell.divergence(*(cl.user_data.interface.end()-1));
                divergence_error += pow((std::abs(val) - 1.0/radius),2) ;
                Eigen::Matrix<T,2,1> normal = ls_cell.normal(*(cl.user_data.interface.end()-1));
                std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                
                interface_normals.push_back( normal_vec ) ;
                
                if( time_step == 0 )
                {
                    
                    vec_normal->add_data(*(cl.user_data.interface.end()-1) ,normal_vec);
                    test_interface_divergence2->add_data( *(cl.user_data.interface.end()-1) ,val );
                }
                                   
               
                normal_interface_status += pow( u_projected (*(cl.user_data.interface.end()-1) ).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0),2) + pow( u_projected(*( cl.user_data.interface.end()-1) ).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1), 2);
                
                
            interface_points_plot.push_back(*(cl.user_data.interface.end()-1)) ;
                val_u_nx.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) );
                val_u_ny.push_back( u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                val_u_n.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) + u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                
                
                counter_interface_pts++;
                   
               
            
            }
            
        }
        
        if( time_step == 0 )
        {
            postoutput_div2.add_object(test_interface_divergence2);
            postoutput_div2.write();
            
            postoutput_vec.add_object(vec_normal);
            postoutput_vec.write();
            
        }
        
        if( time_step == 0 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
            
        if( time_step == 2 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 4 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 6 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
            
        if( time_step == 8 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
       
        
        divergence_error /= counter_interface_pts;
        divergence_error = sqrt(divergence_error);
        std::cout<<yellow<<bold<<"The l2 error of the DIVERGENCE at the INTERFACE, at time "<<reset<< dt*time_step <<" is " << divergence_error <<std::endl;
                   
        
        std::cout<<"number of interface points is " << counter_interface_pts << std::endl;
        normal_interface_status /= counter_interface_pts;
        normal_interface_status = sqrt(normal_interface_status);
        std::cout<<yellow<<bold<<"The l2 error of u*n over the INTERFACE, at time "<<reset<< tot_time <<" is " << normal_interface_status << std::endl;
        if(time_step == 0)
            check = 10.0;
        else
            check = normal_interface_status ;
        
        std::cout << "Area at time step: "<<tot_time<<" is "<< area0  << reset << std::endl;
        std::cout  << "Internal mass at time step: "<<tot_time<<" is "<<reset<< mass0   << std::endl;
        //std::cout<<bold<<yellow << "GLOBAL Mass at time step: "<<time_step<<" is "<<reset<<global_mass0<< reset << std::endl;
         std::cout << "CENTRE OF MASS at time step: " <<tot_time<<" is "<<" ( "<< centre_mass0_x/area0  << " , "<< centre_mass0_y/area0 <<" ). " << reset << std::endl;
        
         d_a = sqrt(4.0*area0/M_PI) ;
        
        std::cout<<"The PERIMETER, at time "<< tot_time <<" is " << perimeter0 <<std::endl;
        
        std::cout<<yellow<<bold<<"The CIRCULARITY, at time "<< tot_time <<" is "<<reset << M_PI*d_a/perimeter0 <<std::endl;
        
        
        
        if(time_step == 0 ){
            initial_area  = area0 ;
            initial_mass = mass0 ;
            centre_mass_x_inital = centre_mass0_x ;
            centre_mass_y_inital = centre_mass0_y ;
            perimeter_initial = perimeter0 ;
        }
        
        if(time_step > 0 )
        {
            diff_area = (area0 - initial_area)/initial_area ;
            diff_mass = (std::abs((mass0 - initial_mass)))/(std::abs(initial_mass)) ;
            std::cout << bold << yellow << "Normalised difference in Area (new - old)/old at time step: "<<tot_time<<" is "<<reset<< diff_area  << reset << std::endl;
            std::cout << bold << yellow << "Difference in internal MASS |new - old|/|old| at time step: "<<tot_time<<" is "<<reset<< diff_mass  << reset << std::endl;
            
            std::cout << "TRANSLATION OF THE CENTRE OF MASS at time step: "  <<tot_time<<" is "<<" ( " << centre_mass0_x/area0 - centre_mass_x_inital/initial_area <<" , " << centre_mass0_y/area0 - centre_mass_y_inital/initial_area<<" ). " << std::endl;
            std::cout<<"NORMALISED DIFFERENCE PERIMETER, at time " << tot_time <<" is " << (perimeter0 - perimeter_initial)/perimeter_initial <<std::endl;
                   
        }
        
        // auto test_case = make_test_case_stokes_1(msh, level_set_function);
        // auto test_case = make_test_case_stokes_2(msh, ls_cell); //level_set_function);
        
/// DA AGGIUNGERE UNA VOLTA SISTEMATO IL CODICE

        if(check < 1e-15 )
        {
            std::cout<<" check = "<<check<<" , STOP!"<<std::endl;
            return 0;
        }
        
        bool sym_grad = TRUE;
        
        auto prm = params<T>();
        prm.kappa_1 = 1.0;
        prm.kappa_2 = 1.0;
        
        //auto test_case = make_test_case_eshelby(msh_i, ls_cell,  prm , sym_grad);
       
        auto test_case = make_test_case_eshelby_2(msh_i, ls_cell,  prm , sym_grad);
        
        auto method = make_sym_gradrec_stokes_interface_method(msh_i, 1.0, 0.0, test_case, sym_grad);
       
        
        //auto u_projected = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
       
    
        if(solve_interface){
            run_cuthho_interface_velocity(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad );
            
            //run_cuthho_interface(msh, degree, method, test_case);
        }
        if (solve_fictdom)
            run_cuthho_fictdom(msh_i, degree, test_case);
        

        
        
        /*********************** FEM -  PROCESSING **************************/
        /// ORA HO SMOOTH OPERATOR! USE L2 PROJECTION
        if( 1 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: SMOOTH OPERATOR FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.smooth_converting_into_FE_formulation( u_projected.sol_HHO );
        }
        if( 0 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: OLD OPERATOR FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.converting_into_FE_formulation( u_projected.sol_HHO );
        }
        if( 0 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: L^2 PROJECTION FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.L2_proj_into_FE_formulation(level_set_function , msh);
        }
        
        /*
        auto u_prova = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
        u_prova.sol_HHO = u_projected.sol_HHO ;
        u_prova.L2_proj_into_FE_formulation( level_set_function , msh );
        
        
        testing_velocity_field_L2projected(msh , u_prova) ;
        
        //std::cout<<"CHECK SMOOTH CONVERTING  FEM ----> FIRST"<<'\n'<<(u_prova.sol_FEM.first - u_projected.sol_FEM.first)<<'\n' <<std::endl;
        //std::cout<<"CHECK SMOOTH CONVERTING FEM ----> SECOND"<<'\n'<<(u_prova.sol_FEM.second - u_projected.sol_FEM.second)<<'\n' <<std::endl;
        
        testing_velocity_field(msh , u_projected) ;
        */
        
        T eps = 0.48 ; // factor to be inside CFL stability zone
        //T dt1 = time_step_CFL( u , mip , eps ); // OLD IMPLEMENTATION
        T dt1 = time_step_CFL_new( u_projected , mip , eps );
        //std::cout<<"dt1 is "<<dt1<<std::endl;
        T dt2 = 0.1;
        dt = std::min(dt1 , dt2);
        std::cout<<"MAX dt = "<<dt<<" AND HEURISTIC CFL IS "<<dt1<<std::endl;
        /*
        if(high_order)
            run_FEM_BERNSTEIN_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
            //run_FEM_BERNSTEIN_CORRECT( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
        else
            run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
            //run_FEM_BERNSTEIN_LOW_ORDER_CORRECT( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
        */
        
        // I can create a sub-time. I solve several time the FEM problem, given a Stokes field. The amount of time is s.t. at maximum there is a displacement of a cell of the interface and no more than a maximum T
        T sub_time = 0.;
        T sub_dt = std::min(1e-3 , dt ) ;
        std::cout<<"Implemented dt = "<<dt<<std::endl;
        while( (sub_time < sub_dt*10) && (sub_time < dt1) )
        {
            if(high_order)
                run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
                //run_FEM_BERNSTEIN_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
                
            else
                run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
               
            sub_time += sub_dt ;
      
        }
        std::cout<<yellow<<bold<<"SUB TIME REPETITION OF TRANSPORT PB DATA:"<<reset<<" sub_dt = "<<sub_dt<< " , number time steps = "<<sub_time/sub_dt<<std::endl;
        
        tot_time += sub_time ;
        /// OLD IMPLEMENTATION
        //for(size_t j=1; j<4 ; j++)
        //run_FEM_levelset( level_set_function.msh,degree_FEM,level_set_function,u,dt,mip);
       
        /************** FEM -  POST-PROCESSING **************/
        if( (T_N - time_step)==0 )
        {
            // Uploading mesh data to check out differences in mass and areas
            crr_mesh.current_mesh = msh;
            Mesh msh_i2 =  crr_mesh.current_mesh;
            offset_definition(msh_i2);
            tc.tic();
            detect_node_position2(msh_i2, level_set_function); // In cuthho_geom
            detect_cut_faces2(msh_i2, level_set_function); // In cuthho_geom
        
       
            if (agglomeration)
            {
                // std::cout<<"i m here 1"<<std::endl;
                detect_cut_cells2(msh_i2, level_set_function); // In cuthho_geom
           
                detect_cell_agglo_set(msh_i2, level_set_function); // Non serve modificarla
           
                make_neighbors_info_cartesian(msh_i2); // Non serve modificarla
                //refine_interface_angle(msh_i2, level_set_function, int_refsteps); // IN cuthho_geom
                refine_interface_pro(msh_i2, level_set_function, int_refsteps); // IN cuthho_geom
                make_agglomeration(msh_i2, level_set_function); // Non serve modificarla
            }
            else
            {
                //move_nodes(msh_i2, level_set_function);
                //detect_cut_faces2(msh_i2, level_set_function); //do it again to update intersection     points
                detect_cut_cells2(msh_i2, level_set_function);
                //refine_interface2(msh_i2, level_set_function, int_refsteps);
                //refine_interface_angle(msh_i2, level_set_function, int_refsteps);
                refine_interface_pro(msh_i2, level_set_function, int_refsteps);
            }
        
            tc.toc();
            std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

            if (dump_debug)
            {
                dump_mesh(msh_i2);
                output_mesh_info(msh_i2, level_set_function);
            }
        
            output_mesh_info2(msh_i2, level_set_function);
            // IN cuthho_export..Points/Nodes don't change
           
            // Uploading level set
            ls_cell.level_set = level_set_function;
            ls_cell.agglo_msh = msh_i2;
            u_projected.set_agglo_mesh( msh_i2 );
            
           
            T mass_fin = 0. , area_fin = 0. ;
            T centre_mass_x = 0. , centre_mass_y = 0. ;
        
            T perimeter = 0. ;
            normal_interface_status = 0. ;
            counter_interface_pts = 0;
            T divergence_error_fin = 0.;
            
            postprocess_output<double> postoutput_vec;
            auto vec_normal_fin = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_Stokes_final.dat");
            
            postprocess_output<double> postoutput_div2;
            auto test_interface_divergence_fin  = std::make_shared< gnuplot_output_object<double> >("divergence_interface_Stokes_final.dat");
            
            std::vector<T> val_u_nx_fin , val_u_ny_fin , val_u_n_fin ;
            std::vector< point<T, 2> > interface_points_plot_fin ;
            std::vector< std::pair<T,T> > interface_normals_fin ;
            
            for(auto& cl : msh_i2.cells)
            {
                ls_cell.cell_assignment(cl);
                u_projected.cell_assignment(cl);
                
                if( (location(msh_i2, cl) == element_location::IN_NEGATIVE_SIDE) || (location(msh_i2, cl) == element_location::ON_INTERFACE) )
                {
                    
                    T partial_area = measure( msh_i2, cl, element_location::IN_NEGATIVE_SIDE);
                 
                    area_fin += partial_area;
                    
                   
                    auto qps_fin = integrate( msh_i2 , cl , 2*degree_FEM+1 , element_location::IN_NEGATIVE_SIDE);
                   
                    for(auto& qp:qps_fin){
                        mass_fin += qp.second * ls_cell(qp.first);
                        centre_mass_x += qp.second * qp.first.x() ;
                        centre_mass_y += qp.second * qp.first.y() ;
                    }
                   
                }
                if(cl.user_data.location == element_location::ON_INTERFACE)
                {
                    
                    for(auto interface_point = cl.user_data.interface.begin() ; interface_point < cl.user_data.interface.end() -1 ; interface_point++ )
                    {
                        perimeter += ( *(interface_point+1) - *interface_point ).to_vector().norm();
                        
                        normal_interface_status += pow( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) , 2) + pow( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) , 2 );
                        
                        
                        T val = ls_cell.divergence( *interface_point );
                        divergence_error_fin += pow((std::abs(val) - 1.0/radius),2) ;
                        Eigen::Matrix<T,2,1> normal = ls_cell.normal(*interface_point);
                        std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                        
                        interface_normals_fin.push_back( normal_vec ) ;
                        
                        vec_normal_fin->add_data(*interface_point,normal_vec);
                        test_interface_divergence_fin->add_data( *interface_point , val);
                        
                        interface_points_plot_fin.push_back( *(interface_point) ) ;
                        val_u_nx_fin.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) );
                        val_u_ny_fin.push_back( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                        val_u_n_fin.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) + u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                        
                        
                        counter_interface_pts++;
                    }
                    
                    T val = ls_cell.divergence( *(cl.user_data.interface.end()-1) );
                    divergence_error_fin += pow((std::abs(val) - 1.0/radius),2) ;
                    
                    Eigen::Matrix<T,2,1> normal = ls_cell.normal(*(cl.user_data.interface.end()-1));
                                       
                    std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                    
                    interface_normals_fin.push_back( normal_vec ) ;
                    
                    vec_normal_fin->add_data( *(cl.user_data.interface.end()-1) ,normal_vec);
                    test_interface_divergence_fin->add_data( *(cl.user_data.interface.end()-1) ,val );
                    
                    normal_interface_status += pow( u_projected (*(cl.user_data.interface.end()-1) ).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0),2) + pow( u_projected(*( cl.user_data.interface.end()-1) ).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1), 2);
                    
                    interface_points_plot_fin.push_back( *(cl.user_data.interface.end()-1) ) ;
                    val_u_nx_fin.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) );
                    val_u_ny_fin.push_back( u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                    val_u_n_fin.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) + u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                                
                    counter_interface_pts++;

                }
                
            }
            
            postoutput_div2.add_object(test_interface_divergence_fin);
            postoutput_div2.write();
                       
            postoutput_vec.add_object(vec_normal_fin);
            postoutput_vec.write();
            
            if( time_step == 9 ){
                goal_quantities_time(msh , tot_time, interface_points_plot_fin , val_u_nx_fin , val_u_ny_fin , val_u_n_fin , interface_normals_fin ) ;
                testing_level_set_time(msh,level_set_function, tot_time);
            }
            
            divergence_error_fin /= counter_interface_pts;
            divergence_error_fin = sqrt(divergence_error_fin);
            std::cout<<yellow<<bold<<"The l2 error of the DIVERGENCE at the INTERFACE, at time "<<reset<< tot_time <<" is " << divergence_error_fin <<std::endl;
            
            std::cout<<"number of interface points is " << counter_interface_pts << std::endl;
            
            normal_interface_status /= counter_interface_pts;
            normal_interface_status = sqrt(normal_interface_status);
            
            std::cout<<yellow<<bold<<"The l2 error of u*n over the INTERFACE, at time "<<reset<< tot_time <<" is " << normal_interface_status << std::endl;
            
            
            
            std::cout<<"The PERIMETER, at time "<< tot_time <<" is " << perimeter <<std::endl;
            
            std::cout<<"perimeter = "<< perimeter << " AND  perimeter0 =  "<<perimeter_initial<<std::endl;
            std::cout<< bold << yellow<<"NORMALISED DIFFERENCE PERIMETER, at time "<<reset<< tot_time <<" is " << (perimeter - perimeter_initial)/perimeter_initial <<std::endl;
            
            d_a = sqrt(4.0*area_fin/M_PI) ;
            
            std::cout<< bold << yellow<<"The CIRCULARITY, at time "<< tot_time<<reset <<" is " << M_PI*d_a/perimeter <<std::endl;
            
            std::cout  << "Area at time step: " <<tot_time<<" is "<< area_fin << std::endl;
            std::cout << "Internal mass at time step: "<<tot_time<<" is "<< mass_fin << reset << std::endl;
            
            std::cout << bold << yellow << "NORMALISED Difference in AREA AT TIME "<<tot_time<<" IS "<< reset<< (area_fin - initial_area)/initial_area << std::endl;
            std::cout << bold << yellow << "NORMALISED Difference in INTERNAL MASS AT TIME "<<tot_time<<" IS "<< reset<< (std::abs(mass_fin - initial_mass))/(std::abs( initial_mass )) << std::endl;
            std::cout << "CENTRE OF MASS at time step: "<<tot_time<<" is "<<" ( " << centre_mass_x/area_fin <<" , " << centre_mass_y/area_fin<<" ). " << std::endl;
            std::cout << "TRANSLATION OF THE CENTRE OF MASS at time step: "  <<tot_time<<" is "<<" ( " << centre_mass_x/area_fin - centre_mass_x_inital/initial_area <<" , " << centre_mass_y/area_fin - centre_mass_y_inital/initial_area<<" ). " << std::endl;
            
            
            
           
        } // END OF T = FINAL TIME
       
        time_pos +=dt ;
        //time_step++;
    } // End of the temporal loop
    
    std::cout<< bold << yellow <<"FINAL TIME IS t = "<< reset<<tot_time<<std::endl;
    return 0;
}
#endif





// Interface Stokes Problem: INLET DIRICHLET BDRY CONDITIONS
#if 0
int main(int argc, char **argv)
{
    using RealType = double;

    size_t degree           = 0;
    size_t int_refsteps     = 4;
    size_t degree_FEM       = 0;

    bool dump_debug         = false;
    bool solve_interface    = false;
    bool solve_fictdom      = false;
    bool agglomeration      = false;
    
    bool high_order = false ; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
    bool cut_off_active = false ; // IF FALSE IS SMOOTH, IF TRUE  CUT_OFF

    mesh_init_params<RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;
    RealType d = 0.5;
    size_t T_N = 0;
    /* k <deg>:     method degree
     * g<deg>:  method FEM degree
     * M <num>:     number of cells in x direction
     * N <num>:     number of cells in y direction
     * r <num>:     number of interface refinement steps
     *
     * i:           solve interface problem
     * f:           solve fictitious domain problem
     *
     * D:           use node displacement to solve bad cuts (default)
     * A:           use agglomeration to solve bad cuts
     *
     * d:           dump debug data
     */

    int ch;
    while ( (ch = getopt(argc, argv, "k:q:M:N:r:T:ifDAdhc")) != -1 )
    {
        switch(ch)
        {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'q':
                degree_FEM = atoi(optarg);
                break;
        
            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;
            
            case 'T':
                T_N = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;
            
            case 'h':
                high_order = true;
            break;
            
            case 'c':
                cut_off_active = true;
            break;
            

            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;

    
    
    timecounter tc;

    /************** BUILD MESH **************/
    tc.tic();
    cuthho_poly_mesh<RealType> msh(mip);
    typedef cuthho_poly_mesh<RealType> Mesh;
    offset_definition(msh);
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;
    
    /*
    size_t n_cells = msh.cells.size();
    std::cout<<" I m in parallel zone"<<std::endl;
    tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
    [&msh] (size_t & cell_ind){
            auto& cell = msh.cells[cell_ind];
             }
    );
    */
    /*
#ifdef HAVE_INTEL_TBB
    size_t n_cells = msh.cells_size();
    std::cout<<" I m in parallel zone"<<std::endl;
    tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
    [&msh] (size_t & cell_ind){
            auto& cell = msh.cells[cell_ind];
             }
    );
#else
    std::cout<<" I m in sequential zone"<<std::endl;
    for (size_t cell_ind = 0; cell_ind < msh.cells.size(); cell_ind++)
    {
        auto& cell = msh.cells[cell_ind];
    }
#endif
    */
    
    /************** FINITE ELEMENT INITIALIZATION **************/
    auto fe_data = Finite_Element<RealType,Mesh>( msh , degree_FEM , mip ) ;
    typedef Finite_Element<RealType,Mesh> FiniteSpace;
    
    /************** ANALYTIC LEVEL SET FUNCTION  **************/
    typedef RealType T;
    
    bool circle = false , ellipse = true ;
    
    RealType radius_a , radius_b , radius ;
    RealType x_centre = 0.5;
    RealType y_centre = 0.5;
    RealType C ;
    
    if(circle)
    {
        radius = 1.0/9.0;
    }
        
    if(ellipse)
    {
        radius_a = 1.0/6.0;
        radius_b = 1.0/12.0;
        //radius_a = 1.0/9.0;
        //radius_b = 1.0/9.0;
        std::cout << bold << yellow << "Initial Analytic Area of the ELLIPSE: "<< M_PI*radius_a*radius_b << std::endl;
        radius = sqrt( radius_a * radius_b ) ;
        std::cout << bold << yellow << "Final radius expected of the circle : " << radius <<reset<<std::endl;
    }
  
        
    
    /// THIS DATA BELOW HAS TO BE UPLOAD DEPENDING ON THE PROBLEM.
    
    //auto level_set_function_anal = circle_level_set<RealType>(radius, x_centre, y_centre );
    //typedef  circle_level_set<T> Fonction;
    
    auto level_set_function_anal = elliptic_level_set<RealType>( radius_a, radius_b, x_centre, y_centre);
    typedef  elliptic_level_set<T> Fonction;
    
    T h = std::max( fe_data.hx , fe_data.hy) ;
    
    
   
    /**************  VELOCITY FIELD  INITIALISATION  **************/
  
    auto u_projected = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
    
    /************** LEVEL SET FUNCTION DISCRETISATION **************/
    std::cout<<"degree FEM "<<degree_FEM<<std::endl;
   
    
    auto level_set_function = L2projected_level_set_high_order< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);
    
    //level_set_function.coefficients_mapping_quadratic( );

   
    
   
    if(high_order)
        std::cout<<bold<<yellow<<"----> USING phi_HP HIGH order!!!!! "<<reset<<std::endl;
    else
        std::cout<<bold<<yellow<<"----> USING phi_L LOW order!!!!! "<<reset<<std::endl;
    
    
    testing_level_set(msh,level_set_function);
     
    // Initiliatisation data for time routine
    auto crr_mesh =  Current_Mesh<Mesh>(msh);
    
    // Initialisation area , mass
    T initial_area = 0. , initial_mass = 0.;
    T area_previous_time = 0. , mass_previous_time = 0. , dt = 0. ;
    
    /// DATA CHECK INITIALISATION
    T d_a = 0. ;
    T error_normal_global = 0. ;
    T error_normal_local = 0. ;
    T perimeter_initial = 0. ;
    T centre_mass_x_inital = 0. , centre_mass_y_inital = 0. ;
    
    
    
    T check = 10.0;
    T time_pos = 0.;
    
    T tot_time = 0.;
    
    bool bdry_bottom = false , bdry_up = false ;
    bool bdry_left = false , bdry_right = false ;
    
    check_inlet( msh , fe_data , bdry_bottom , bdry_right , bdry_up , bdry_left, 1e-14 );
    
    //while(check > 0.0112265)
    //{
    for (size_t time_step = 0; time_step<=T_N; time_step++)
    {
       
        

        // ************** Re-Initialization mesh **************
        crr_mesh.current_mesh = msh;
        Mesh msh_i =  crr_mesh.current_mesh;
        offset_definition(msh_i);
        

        //************ DO cutHHO MESH PROCESSING **************
        tc.tic();
        detect_node_position2(msh_i, level_set_function); // In cuthho_geom
        detect_cut_faces2(msh_i, level_set_function); // In cuthho_geom
        
        if (agglomeration)
        {
            detect_cut_cells2(msh_i, level_set_function); // In cuthho_geom
            detect_cell_agglo_set(msh_i, level_set_function); // Non serve modificarla
            make_neighbors_info_cartesian(msh_i); // Non serve modificarla
            //refine_interface_angle(msh_i, level_set_function, int_refsteps);
            refine_interface_pro(msh_i, level_set_function, int_refsteps);
            //refine_interface2(msh_i, level_set_function, int_refsteps); // IN cuthho_geom
            make_agglomeration(msh_i, level_set_function); // Non serve modificarla
        }
        else
        {
            //move_nodes(msh_i, level_set_function);
            //detect_cut_faces2(msh_i, level_set_function); //do it again to update intersection points
            detect_cut_cells2(msh_i, level_set_function);
            //refine_interface2(msh_i, level_set_function, int_refsteps);
            //refine_interface_angle(msh_i, level_set_function, int_refsteps);
            refine_interface_pro(msh_i, level_set_function, int_refsteps);
        }
       
        tc.toc();
        std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

        if (dump_debug)
        {
            dump_mesh(msh_i);
            output_mesh_info(msh_i, level_set_function);
        }
   
        // IN cuthho_export..Points/Nodes don't change-> it's fast
        if(time_step == 0){
            output_mesh_info2_pre_FEM(msh_i, level_set_function); // IN cuthho_export
        }
        else
            output_mesh_info2(msh_i, level_set_function);
            //output_mesh_info2(msh_i, level_set_function);
        
        typedef L2projected_level_set_high_order< Mesh , Fonction , FiniteSpace , T > Level_Set;
        auto ls_cell = LS_cell_L2proj_high_order< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);
        
        
        u_projected.set_agglo_mesh( msh_i );
        // CALCULATION OF AREA AND MASS AT TIME STEP t^n
        // CALCULATION ALSO OF CENTRE OF MASS
        
        
        /// DATA CHECK INITIALISATION
        T area0 = 0. , mass0 = 0. , global_mass0 = 0. ;
        T diff_area = 0. , diff_mass = 0. ;
        T error_normal_global0 = 0. ;
        T centre_mass0_x = 0. , centre_mass0_y = 0. ;
        T perimeter0 = 0.;
        T normal_interface_status = 0. ;
        size_t counter_interface_pts = 0;
        T divergence_error = 0.;
        // PLOTTING OF NORMAL
                 
        
        postprocess_output<double> postoutput_vec;
        auto vec_normal = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_Stokes_initial.dat");
        
        postprocess_output<double> postoutput_div2;
        auto test_interface_divergence2  = std::make_shared< gnuplot_output_object<double> >("divergence_interface_Stokes_initial.dat");
        
       
        std::vector<T> val_u_nx , val_u_ny , val_u_n ;
        std::vector< point<T, 2> > interface_points_plot ;
        std::vector< std::pair<T,T> > interface_normals ;
        
        for(auto& cl : msh_i.cells)
        {
            ls_cell.cell_assignment(cl);
            u_projected.cell_assignment(cl);
            
            if( location(msh_i, cl) == element_location::IN_NEGATIVE_SIDE || location(msh_i, cl) == element_location::ON_INTERFACE )
            {
                T partial_area = measure( msh_i, cl, element_location::IN_NEGATIVE_SIDE);
                area0 += partial_area;
                auto qps = integrate( msh_i , cl , 2*degree_FEM+1 , element_location::IN_NEGATIVE_SIDE);
                for(auto& qp:qps){
                    mass0 += qp.second * ls_cell(qp.first);
                    centre_mass0_x += qp.second * qp.first.x() ;
                    centre_mass0_y += qp.second * qp.first.y() ;
                }
            }
            if(cl.user_data.location == element_location::ON_INTERFACE)
            {
                /*
                std::cout<<"CELL = "<<offset(msh,cl)<<std::endl;
                if(offset(msh,cl) == 101 || offset(msh,cl) == 104 )
                {
                    for(size_t kk = 0 ; kk < u_projected.sol_HHO.first.rows() ; kk++ )
                    std::cout<< u_projected.sol_HHO.first(kk,offset(msh,cl)) << " " << u_projected.sol_HHO.second(kk,offset(msh,cl)) << '\n';
                    std::cout<<'\n'<<std::endl;
                }
                */
                for(auto interface_point = cl.user_data.interface.begin() ; interface_point < cl.user_data.interface.end() -1 ; interface_point++ )
                {
                    
                    
                    T val = ls_cell.divergence( *interface_point );
                    divergence_error += pow((std::abs(val) - 1.0/radius),2) ;
                    Eigen::Matrix<T,2,1> normal = ls_cell.normal(*interface_point);
                    std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                    interface_normals.push_back( normal_vec ) ;
                    
                    if( time_step == 0 )
                    {
                        
                        vec_normal->add_data(*interface_point,normal_vec);
                        test_interface_divergence2->add_data(*interface_point , val);
                    }
                    
                    perimeter0 += ( *(interface_point+1) - *interface_point ).to_vector().norm();
                    
                    normal_interface_status += pow( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) , 2) + pow( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) , 2 );
                    
                    interface_points_plot.push_back(*(interface_point)) ;
                    val_u_nx.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) );
                    val_u_ny.push_back( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                    val_u_n.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) + u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                    //std::cout<<"*(interface_point) = "<<*(interface_point)<<std::endl;
                    //std::cout<<" u_projected primo = "<<u_projected(*(interface_point)).first  << " u_projected second = "<< u_projected(*(interface_point)).second  << " somma tot con olds = "<<normal_interface_status<<std::endl;
                    //std::cout<<" ls_cell.normal(*(interface_point))(0)  = "<<ls_cell.normal(*(interface_point))(0)  << " ls_cell.normal(*(interface_point))(1)  = "<< ls_cell.normal(*(interface_point))(1)  << std::endl;
                    
                    counter_interface_pts++;
                       
                }
                
                T val = ls_cell.divergence(*(cl.user_data.interface.end()-1));
                divergence_error += pow((std::abs(val) - 1.0/radius),2) ;
                Eigen::Matrix<T,2,1> normal = ls_cell.normal(*(cl.user_data.interface.end()-1));
                std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                interface_normals.push_back( normal_vec ) ;
                
                if( time_step == 0 )
                {
                    
                    vec_normal->add_data(*(cl.user_data.interface.end()-1) ,normal_vec);
                    test_interface_divergence2->add_data( *(cl.user_data.interface.end()-1) ,val );
                }
                                   
               
                normal_interface_status += pow( u_projected (*(cl.user_data.interface.end()-1) ).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0),2) + pow( u_projected(*( cl.user_data.interface.end()-1) ).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1), 2);
                
                
            interface_points_plot.push_back(*(cl.user_data.interface.end()-1)) ;
                val_u_nx.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) );
                val_u_ny.push_back( u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                val_u_n.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) + u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                
                
                counter_interface_pts++;
                   
               
            
            }
            
        }
        
        if( time_step == 0 )
        {
            postoutput_div2.add_object(test_interface_divergence2);
            postoutput_div2.write();
            
            postoutput_vec.add_object(vec_normal);
            postoutput_vec.write();
            
        }
        goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
                   
        testing_level_set_time(msh,level_set_function,tot_time);
        /*
        if( time_step == 0 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
            
        if( time_step == 5 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 10 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 15 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
            
        if( time_step == 20 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        if( time_step == 30 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 40 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        */
        
        divergence_error /= counter_interface_pts;
        divergence_error = sqrt(divergence_error);
        std::cout<<yellow<<bold<<"The l2 error of the DIVERGENCE at the INTERFACE, at time "<<reset<< dt*time_step <<" is " << divergence_error <<std::endl;
                   
        
        std::cout<<"number of interface points is " << counter_interface_pts << std::endl;
        normal_interface_status /= counter_interface_pts;
        normal_interface_status = sqrt(normal_interface_status);
        std::cout<<yellow<<bold<<"The l2 error of u*n over the INTERFACE, at time "<<reset<< tot_time <<" is " << normal_interface_status << std::endl;
        if(time_step == 0)
            check = 10.0;
        else
            check = normal_interface_status ;
        
        std::cout << "Area at time step: "<<tot_time<<" is "<< area0  << reset << std::endl;
        std::cout  << "Internal mass at time step: "<<tot_time<<" is "<<reset<< mass0   << std::endl;
        //std::cout<<bold<<yellow << "GLOBAL Mass at time step: "<<time_step<<" is "<<reset<<global_mass0<< reset << std::endl;
         std::cout << "CENTRE OF MASS at time step: " <<tot_time<<" is "<<" ( "<< centre_mass0_x/area0  << " , "<< centre_mass0_y/area0 <<" ). " << reset << std::endl;
        
         d_a = sqrt(4.0*area0/M_PI) ;
        
        std::cout<<"The PERIMETER, at time "<< tot_time <<" is " << perimeter0 <<std::endl;
        
        std::cout<<yellow<<bold<<"The CIRCULARITY, at time "<< tot_time <<" is "<<reset << M_PI*d_a/perimeter0 <<std::endl;
        
        
        
        if(time_step == 0 ){
            initial_area  = area0 ;
            initial_mass = mass0 ;
            centre_mass_x_inital = centre_mass0_x ;
            centre_mass_y_inital = centre_mass0_y ;
            perimeter_initial = perimeter0 ;
        }
        
        if(time_step > 0 )
        {
            diff_area = (area0 - initial_area)/initial_area ;
            diff_mass = (std::abs((mass0 - initial_mass)))/(std::abs(initial_mass)) ;
            std::cout << bold << yellow << "Normalised difference in Area (new - old)/old at time step: "<<tot_time<<" is "<<reset<< diff_area  << reset << std::endl;
            std::cout << bold << yellow << "Difference in internal MASS |new - old|/|old| at time step: "<<tot_time<<" is "<<reset<< diff_mass  << reset << std::endl;
            
            std::cout << "TRANSLATION OF THE CENTRE OF MASS at time step: "  <<tot_time<<" is "<<" ( " << centre_mass0_x/area0 - centre_mass_x_inital/initial_area <<" , " << centre_mass0_y/area0 - centre_mass_y_inital/initial_area<<" ). " << std::endl;
            std::cout<<"NORMALISED DIFFERENCE PERIMETER, at time " << tot_time <<" is " << (perimeter0 - perimeter_initial)/perimeter_initial <<std::endl;
                   
        }
        
        // auto test_case = make_test_case_stokes_1(msh, level_set_function);
        // auto test_case = make_test_case_stokes_2(msh, ls_cell); //level_set_function);
        
/// DA AGGIUNGERE UNA VOLTA SISTEMATO IL CODICE

        if(check < 1e-5 )
        {
            std::cout<<" check = "<<check<<" , STOP!"<<std::endl;
            return 0;
        }
        
        bool sym_grad = TRUE;
        
        auto prm = params<T>();
        prm.kappa_1 = 1.0;
        prm.kappa_2 = 1.0;
        
        //auto test_case = make_test_case_eshelby(msh_i, ls_cell,  prm , sym_grad);
       
        auto test_case = make_test_case_eshelby_2(msh_i, ls_cell,  prm , sym_grad);
        
        auto method = make_sym_gradrec_stokes_interface_method(msh_i, 1.0, 0.0, test_case, sym_grad);
       
        
        //auto u_projected = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
       
    
        if(solve_interface){
            run_cuthho_interface_velocity(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad );
            
            //run_cuthho_interface(msh, degree, method, test_case);
        }
        if (solve_fictdom)
            run_cuthho_fictdom(msh_i, degree, test_case);
        

        
        
        /*********************** FEM -  PROCESSING **************************/
        /// ORA HO SMOOTH OPERATOR! USE L2 PROJECTION
        if( 1 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: SMOOTH OPERATOR FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.smooth_converting_into_FE_formulation( u_projected.sol_HHO );
        }
        if( 0 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: OLD OPERATOR FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.converting_into_FE_formulation( u_projected.sol_HHO );
        }
        if( 0 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: L^2 PROJECTION FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.L2_proj_into_FE_formulation(level_set_function , msh);
        }
        
        /*
        auto u_prova = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
        u_prova.sol_HHO = u_projected.sol_HHO ;
        u_prova.L2_proj_into_FE_formulation( level_set_function , msh );
        
        
        testing_velocity_field_L2projected(msh , u_prova) ;
        
        //std::cout<<"CHECK SMOOTH CONVERTING  FEM ----> FIRST"<<'\n'<<(u_prova.sol_FEM.first - u_projected.sol_FEM.first)<<'\n' <<std::endl;
        //std::cout<<"CHECK SMOOTH CONVERTING FEM ----> SECOND"<<'\n'<<(u_prova.sol_FEM.second - u_projected.sol_FEM.second)<<'\n' <<std::endl;
        
        testing_velocity_field(msh , u_projected) ;
        */
        
        //check_inlet( msh , fe_data ,  u_projected , 1e-14 );
        
        T eps = 0.48 ; // factor to be inside CFL stability zone
        //T dt1 = time_step_CFL( u , mip , eps ); // OLD IMPLEMENTATION
        T dt1 = time_step_CFL_new( u_projected , mip , eps );
        //std::cout<<"dt1 is "<<dt1<<std::endl;
        T dt2 = 0.1;
        dt = std::min(dt1 , dt2);
        std::cout<<"MAX dt = "<<dt<<" AND HEURISTIC CFL IS "<<dt1<<std::endl;
        /*
        if(high_order)
            run_FEM_BERNSTEIN_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
            //run_FEM_BERNSTEIN_CORRECT( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
        else
            run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
            //run_FEM_BERNSTEIN_LOW_ORDER_CORRECT( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
        */
        
        // I can create a sub-time. I solve several time the FEM problem, given a Stokes field. The amount of time is s.t. at maximum there is a displacement of a cell of the interface and no more than a maximum T
        T sub_time = 0.;
        T sub_dt = std::min(4e-4 , dt ) ;
        std::cout<<"Implemented dt = "<<dt<<std::endl;
        while( (sub_time < sub_dt*10) && (sub_time < dt1) )
        {
            if(high_order){
                run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D_NEW_DIRICHLET_COND( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt );

                //run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
                //run_FEM_BERNSTEIN_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
            }
            else
                run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
               
            sub_time += sub_dt ;
      
        }
        std::cout<<yellow<<bold<<"SUB TIME REPETITION OF TRANSPORT PB DATA:"<<reset<<" sub_dt = "<<sub_dt<< " , number time steps = "<<sub_time/sub_dt<<std::endl;
        
        tot_time += sub_time ;
        
         testing_level_set2(msh,level_set_function);
        /// OLD IMPLEMENTATION
        //for(size_t j=1; j<4 ; j++)
        //run_FEM_levelset( level_set_function.msh,degree_FEM,level_set_function,u,dt,mip);
       
        /************** FEM -  POST-PROCESSING **************/
        if( (T_N - time_step)==0 )
        {
            // Uploading mesh data to check out differences in mass and areas
            crr_mesh.current_mesh = msh;
            Mesh msh_i2 =  crr_mesh.current_mesh;
            offset_definition(msh_i2);
            tc.tic();
            detect_node_position2(msh_i2, level_set_function); // In cuthho_geom
            detect_cut_faces2(msh_i2, level_set_function); // In cuthho_geom
        
       
            if (agglomeration)
            {
                // std::cout<<"i m here 1"<<std::endl;
                detect_cut_cells2(msh_i2, level_set_function); // In cuthho_geom
           
                detect_cell_agglo_set(msh_i2, level_set_function); // Non serve modificarla
           
                make_neighbors_info_cartesian(msh_i2); // Non serve modificarla
                //refine_interface_angle(msh_i2, level_set_function, int_refsteps); // IN cuthho_geom
                refine_interface_pro(msh_i2, level_set_function, int_refsteps); // IN cuthho_geom
                make_agglomeration(msh_i2, level_set_function); // Non serve modificarla
            }
            else
            {
                //move_nodes(msh_i2, level_set_function);
                //detect_cut_faces2(msh_i2, level_set_function); //do it again to update intersection     points
                detect_cut_cells2(msh_i2, level_set_function);
                //refine_interface2(msh_i2, level_set_function, int_refsteps);
                //refine_interface_angle(msh_i2, level_set_function, int_refsteps);
                refine_interface_pro(msh_i2, level_set_function, int_refsteps);
            }
        
            tc.toc();
            std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

            if (dump_debug)
            {
                dump_mesh(msh_i2);
                output_mesh_info(msh_i2, level_set_function);
            }
        
            output_mesh_info2(msh_i2, level_set_function);
            // IN cuthho_export..Points/Nodes don't change
           
            // Uploading level set
            ls_cell.level_set = level_set_function;
            ls_cell.agglo_msh = msh_i2;
            u_projected.set_agglo_mesh( msh_i2 );
            
           
            T mass_fin = 0. , area_fin = 0. ;
            T centre_mass_x = 0. , centre_mass_y = 0. ;
        
            T perimeter = 0. ;
            normal_interface_status = 0. ;
            counter_interface_pts = 0;
            T divergence_error_fin = 0.;
            
            postprocess_output<double> postoutput_vec;
            auto vec_normal_fin = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_Stokes_final.dat");
            
            postprocess_output<double> postoutput_div2;
            auto test_interface_divergence_fin  = std::make_shared< gnuplot_output_object<double> >("divergence_interface_Stokes_final.dat");
            
            std::vector<T> val_u_nx_fin , val_u_ny_fin , val_u_n_fin ;
            std::vector< point<T, 2> > interface_points_plot_fin ;
            std::vector< std::pair<T,T> > interface_normals_fin ;
            
            for(auto& cl : msh_i2.cells)
            {
                ls_cell.cell_assignment(cl);
                u_projected.cell_assignment(cl);
                
                if( (location(msh_i2, cl) == element_location::IN_NEGATIVE_SIDE) || (location(msh_i2, cl) == element_location::ON_INTERFACE) )
                {
                    
                    T partial_area = measure( msh_i2, cl, element_location::IN_NEGATIVE_SIDE);
                 
                    area_fin += partial_area;
                    
                   
                    auto qps_fin = integrate( msh_i2 , cl , 2*degree_FEM+1 , element_location::IN_NEGATIVE_SIDE);
                   
                    for(auto& qp:qps_fin){
                        mass_fin += qp.second * ls_cell(qp.first);
                        centre_mass_x += qp.second * qp.first.x() ;
                        centre_mass_y += qp.second * qp.first.y() ;
                    }
                   
                }
                if(cl.user_data.location == element_location::ON_INTERFACE)
                {
                    
                    for(auto interface_point = cl.user_data.interface.begin() ; interface_point < cl.user_data.interface.end() -1 ; interface_point++ )
                    {
                        perimeter += ( *(interface_point+1) - *interface_point ).to_vector().norm();
                        
                        normal_interface_status += pow( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) , 2) + pow( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) , 2 );
                        
                        
                        T val = ls_cell.divergence( *interface_point );
                        divergence_error_fin += pow((std::abs(val) - 1.0/radius),2) ;
                        Eigen::Matrix<T,2,1> normal = ls_cell.normal(*interface_point);
                        std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                        interface_normals_fin.push_back( normal_vec ) ;
                        
                        vec_normal_fin->add_data(*interface_point,normal_vec);
                        test_interface_divergence_fin->add_data( *interface_point , val);
                        
                        interface_points_plot_fin.push_back( *(interface_point) ) ;
                        val_u_nx_fin.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) );
                        val_u_ny_fin.push_back( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                        val_u_n_fin.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) + u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                        
                        
                        counter_interface_pts++;
                    }
                    
                    T val = ls_cell.divergence( *(cl.user_data.interface.end()-1) );
                    divergence_error_fin += pow((std::abs(val) - 1.0/radius),2) ;
                    
                    Eigen::Matrix<T,2,1> normal = ls_cell.normal(*(cl.user_data.interface.end()-1));
                                       
                    std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                    interface_normals_fin.push_back( normal_vec ) ;
                    
                    vec_normal_fin->add_data( *(cl.user_data.interface.end()-1) ,normal_vec);
                    test_interface_divergence_fin->add_data( *(cl.user_data.interface.end()-1) ,val );
                    
                    normal_interface_status += pow( u_projected (*(cl.user_data.interface.end()-1) ).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0),2) + pow( u_projected(*( cl.user_data.interface.end()-1) ).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1), 2);
                    
                    interface_points_plot_fin.push_back( *(cl.user_data.interface.end()-1) ) ;
                    val_u_nx_fin.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) );
                    val_u_ny_fin.push_back( u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                    val_u_n_fin.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) + u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                                
                    counter_interface_pts++;

                }
                
            }
            
            postoutput_div2.add_object(test_interface_divergence_fin);
            postoutput_div2.write();
                       
            postoutput_vec.add_object(vec_normal_fin);
            postoutput_vec.write();
            goal_quantities_time(msh , tot_time, interface_points_plot_fin , val_u_nx_fin , val_u_ny_fin , val_u_n_fin , interface_normals_fin ) ;
            testing_level_set_time(msh,level_set_function, tot_time);
            /*
            if( time_step == 9 ){
                goal_quantities_time(msh , tot_time, interface_points_plot_fin , val_u_nx_fin , val_u_ny_fin , val_u_n_fin , interface_normals_fin ) ;
                testing_level_set_time(msh,level_set_function, tot_time);
            }
            */
            divergence_error_fin /= counter_interface_pts;
            divergence_error_fin = sqrt(divergence_error_fin);
            std::cout<<yellow<<bold<<"The l2 error of the DIVERGENCE at the INTERFACE, at time "<<reset<< tot_time <<" is " << divergence_error_fin <<std::endl;
            
            std::cout<<"number of interface points is " << counter_interface_pts << std::endl;
            
            normal_interface_status /= counter_interface_pts;
            normal_interface_status = sqrt(normal_interface_status);
            
            std::cout<<yellow<<bold<<"The l2 error of u*n over the INTERFACE, at time "<<reset<< tot_time <<" is " << normal_interface_status << std::endl;
            
            
            
            std::cout<<"The PERIMETER, at time "<< tot_time <<" is " << perimeter <<std::endl;
            
            std::cout<<"perimeter = "<< perimeter << " AND  perimeter0 =  "<<perimeter_initial<<std::endl;
            std::cout<< bold << yellow<<"NORMALISED DIFFERENCE PERIMETER, at time "<<reset<< tot_time <<" is " << (perimeter - perimeter_initial)/perimeter_initial <<std::endl;
            
            d_a = sqrt(4.0*area_fin/M_PI) ;
            
            std::cout<< bold << yellow<<"The CIRCULARITY, at time "<< tot_time<<reset <<" is " << M_PI*d_a/perimeter <<std::endl;
            
            std::cout  << "Area at time step: " <<tot_time<<" is "<< area_fin << std::endl;
            std::cout << "Internal mass at time step: "<<tot_time<<" is "<< mass_fin << reset << std::endl;
            
            std::cout << bold << yellow << "NORMALISED Difference in AREA AT TIME "<<tot_time<<" IS "<< reset<< (area_fin - initial_area)/initial_area << std::endl;
            std::cout << bold << yellow << "NORMALISED Difference in INTERNAL MASS AT TIME "<<tot_time<<" IS "<< reset<< (std::abs(mass_fin - initial_mass))/(std::abs( initial_mass )) << std::endl;
            std::cout << "CENTRE OF MASS at time step: "<<tot_time<<" is "<<" ( " << centre_mass_x/area_fin <<" , " << centre_mass_y/area_fin<<" ). " << std::endl;
            std::cout << "TRANSLATION OF THE CENTRE OF MASS at time step: "  <<tot_time<<" is "<<" ( " << centre_mass_x/area_fin - centre_mass_x_inital/initial_area <<" , " << centre_mass_y/area_fin - centre_mass_y_inital/initial_area<<" ). " << std::endl;
            
            
            
           
        } // END OF T = FINAL TIME
       
        time_pos +=dt ;
        //time_step++;
    } // End of the temporal loop
    
    std::cout<< bold << yellow <<"FINAL TIME IS t = "<< reset<<tot_time<<std::endl;
    return 0;
}
#endif




// Interface Stokes Problem: INLET DIRICHLET BDRY CONDITIONS -> NEW INTERFACE = 1/2 !!!!
#if 0
int main(int argc, char **argv)
{
    using RealType = double;

    size_t degree           = 0;
    size_t int_refsteps     = 4;
    size_t degree_FEM       = 0;

    bool dump_debug         = false;
    bool solve_interface    = false;
    bool solve_fictdom      = false;
    bool agglomeration      = false;
    
    bool high_order = false ; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
    bool cut_off_active = false ; // IF FALSE IS SMOOTH, IF TRUE  CUT_OFF

    mesh_init_params<RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;
    //RealType d = 0.5;
    size_t T_N = 0;
    /* k <deg>:     method degree
     * g<deg>:  method FEM degree
     * M <num>:     number of cells in x direction
     * N <num>:     number of cells in y direction
     * r <num>:     number of interface refinement steps
     *
     * i:           solve interface problem
     * f:           solve fictitious domain problem
     *
     * D:           use node displacement to solve bad cuts (default)
     * A:           use agglomeration to solve bad cuts
     *
     * d:           dump debug data
     */

    int ch;
    while ( (ch = getopt(argc, argv, "k:q:M:N:r:T:ifDAdhc")) != -1 )
    {
        switch(ch)
        {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'q':
                degree_FEM = atoi(optarg);
                break;
        
            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;
            
            case 'T':
                T_N = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;
            
            case 'h':
                high_order = true;
            break;
            
            case 'c':
                cut_off_active = true;
            break;
            

            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;

    
    
    timecounter tc;

    /************** BUILD MESH **************/
    tc.tic();
    cuthho_poly_mesh<RealType> msh(mip);
    typedef cuthho_poly_mesh<RealType> Mesh;
    offset_definition(msh);
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;
     
    /*
    std::cout<<" I m in parallel zone"<<std::endl;
    tbb::task_scheduler_init init(1);
    //tbb::task_scheduler_init init(tbb::task_scheduler_init::default_num_threads());
    tc.tic();
    size_t n_cells = msh.cells.size();
    tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
    [&msh] (size_t & cell_ind){
            auto& cl = msh.cells[cell_ind];
            auto qps = equidistriduted_nodes_ordered_bis<RealType>(msh, cl , 2) ;
            //std::cout<<"CELL = "<<offset(msh,cell)<<std::endl;
             }
    );
    tc.toc();
    std::cout << bold << yellow << "PARALLEL LOOP: " << tc << " seconds" << reset <<'\n'<< std::endl;
    
    tc.tic();
    for(auto& cl : msh.cells){
        auto qps = equidistriduted_nodes_ordered_bis<RealType>(msh, cl , 2) ;
    }
                
    
    tc.toc();
    std::cout << bold << yellow << "SEQUENTIAL LOOP: " << tc << " seconds" << reset <<'\n'<< std::endl;
    */
  
    
    /************** FINITE ELEMENT INITIALIZATION **************/
    std::cout<<"--> Finite_Element è ancora lineare!"<<std::endl;
    auto fe_data = Finite_Element<RealType,Mesh>( msh , degree_FEM , mip ) ;
    typedef Finite_Element<RealType,Mesh> FiniteSpace;
    
    /************** ANALYTIC LEVEL SET FUNCTION  **************/
    typedef RealType T;
    
    bool circle = true , ellipse = false ;
    
    RealType radius_a , radius_b , radius ;
    RealType x_centre = 0.5;
    RealType y_centre = 0.5;
    //RealType C ;
    //T h = std::max( fe_data.hx , fe_data.hy) ;
    if(circle)
    {
        radius = 1.0/9.0;
    }
        
    if(ellipse)
    {
        radius_a = 1.0/6.0;
        radius_b = 1.0/12.0;
        //radius_a = 1.0/9.0;
        //radius_b = 1.0/9.0;
        std::cout << bold << yellow << "Initial Analytic Area of the ELLIPSE: "<< M_PI*radius_a*radius_b << std::endl;
        radius = sqrt( radius_a * radius_b ) ;
        std::cout << bold << yellow << "Final radius expected of the circle : " << radius <<reset<<std::endl;
    }
  
        
    
    /// THIS DATA BELOW HAS TO BE UPLOAD DEPENDING ON THE PROBLEM.
    timecounter tc_agglo;
    auto level_set_function_anal = circle_level_set<RealType>(radius, x_centre, y_centre );
    typedef  circle_level_set<T> Fonction;
    
    //auto level_set_function_anal = elliptic_level_set<RealType>( radius_a, radius_b, x_centre, y_centre);
    //typedef  elliptic_level_set<T> Fonction;
    
    //auto level_set_function_anal = elliptic_distance_ls<RealType>( radius_a, radius_b, x_centre, y_centre , h);
    //typedef  elliptic_distance_ls<T> Fonction;
    
    //auto level_set_function_anal = circle_distance_ls<RealType>(radius, x_centre, y_centre , 2*h );
    //typedef  circle_distance_ls<T> Fonction;
    
    
   
    /**************  VELOCITY FIELD  INITIALISATION  **************/
      
    tc_agglo.tic();
    auto u_projected = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
    tc_agglo.toc();
        
    std::cout << bold << yellow << "VELOCITY FIELD: " << tc_agglo << " seconds" << reset << std::endl;
    /************** LEVEL SET FUNCTION DISCRETISATION **************/
    std::cout<<"degree FEM "<<degree_FEM<<std::endl;
    
    tc_agglo.tic();
    auto level_set_function_parallel = L2projected_level_set_high_order_parallelize< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);
      
    tc_agglo.toc();
             
    std::cout << bold << yellow << "L2projected_level_set_high_order_parallelize: " << tc_agglo << " seconds" << reset << std::endl;
   
    tc_agglo.tic();
    auto level_set_function = L2projected_level_set_high_order< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);
    tc_agglo.toc();
           
    std::cout << bold << yellow << "L2projected_level_set_high_order: " << tc_agglo << " seconds" << reset << std::endl;
    
   
    
    //level_set_function.iso_val_interface = 0.5 ;
    //level_set_function.coefficients_mapping_quadratic( );
    //level_set_function.coefficients_mapping_MAX_MAX( );
    //level_set_function.coefficients_sfasamento( );
    
   
    if(high_order)
        std::cout<<bold<<yellow<<"----> USING phi_HP HIGH order!!!!! "<<reset<<std::endl;
    else
        std::cout<<bold<<yellow<<"----> USING phi_L LOW order!!!!! "<<reset<<std::endl;
    
    tc_agglo.tic();
    testing_level_set(msh,level_set_function);
    tc_agglo.toc();
         
    std::cout << bold << yellow << "testing_level_set: " << tc_agglo << " seconds" << reset << std::endl;
    // Initiliatisation data for time routine
    auto crr_mesh =  Current_Mesh<Mesh>(msh);
    
    // Initialisation area , mass
    T initial_area = 0. , initial_mass = 0.;
    T  dt = 0. ; // area_previous_time = 0. , mass_previous_time = 0. ,
    
    /// DATA CHECK INITIALISATION
    T d_a = 0. ;
    //T error_normal_global = 0. ;
    //T error_normal_local = 0. ;
    T perimeter_initial = 0. ;
    T centre_mass_x_inital = 0. , centre_mass_y_inital = 0. ;
    
    
    
    T check = 10.0;
    T time_pos = 0.;
    
    T tot_time = 0.;
    
    bool bdry_bottom = false , bdry_up = false ;
    bool bdry_left = false , bdry_right = false ;
    tc_agglo.tic();
    check_inlet( msh , fe_data , bdry_bottom , bdry_right , bdry_up , bdry_left, 1e-14 );
    tc_agglo.toc();
          
    std::cout << bold << yellow << "check_inlet: " << tc_agglo << " seconds" << reset << std::endl;
   
    for (size_t time_step = 0; time_step<=T_N; time_step++)
    {
        
        tc_agglo.tic();
        level_set_function_parallel.normal_continuous_setting();
        tc_agglo.toc();
        std::cout << bold << yellow << "normal_continuous_setting PARALLEL: " << tc_agglo << " seconds" << reset << std::endl;
        
        tc_agglo.tic();
        // Uploading continuous normal function
        level_set_function.normal_continuous_setting();
        
        tc_agglo.toc();
        std::cout << bold << yellow << "normal_continuous_setting: " << tc_agglo << " seconds" << reset << std::endl;
        
        // ************** Re-Initialization mesh **************
        crr_mesh.current_mesh = msh;
        Mesh msh_i =  crr_mesh.current_mesh;
        offset_definition(msh_i);
        
       
        

        //************ DO cutHHO MESH PROCESSING **************
        tc.tic();
        // The sequential detect_node_position3 is fastern than the parallel one: DIMOSTRATO.
        //detect_node_position3_parallel(msh_i, level_set_function); // In cuthho_geom
        detect_node_position3(msh_i, level_set_function); // In cuthho_geom
        
        detect_cut_faces3(msh_i, level_set_function); // In cuthho_geom
        //std::cout<<"FINE DETECT NODES AND FACES."<<std::endl;
        if (agglomeration)
        {
            detect_cut_cells3(msh_i, level_set_function); // In cuthho_geom
            //detect_cut_cells3_parallelized(msh_i, level_set_function); // In cuthho_geom
            std::cout<<"FINE DETECT CELLS."<<std::endl;
           
            tc_agglo.tic();
            detect_cell_agglo_set(msh_i, level_set_function); // Non serve modificarla
            tc_agglo.toc();
            std::cout << bold << yellow << "detect_cell_agglo_set: " << tc_agglo << " seconds" << reset << std::endl;
            tc_agglo.tic();
            make_neighbors_info_cartesian(msh_i); // Non serve modificarla
            //refine_interface_angle(msh_i, level_set_function, int_refsteps);
            tc_agglo.toc();
            std::cout << bold << yellow << "make_neighbors_info_cartesian: " << tc_agglo << " seconds" << reset << std::endl;
            tc_agglo.tic();
            refine_interface_pro3(msh_i, level_set_function, int_refsteps);
            tc_agglo.toc();
            std::cout << bold << yellow << "refine_interface_pro3: " << tc_agglo << " seconds" << reset << std::endl;
            //refine_interface2(msh_i, level_set_function, int_refsteps); // IN cuthho_geom
            tc_agglo.tic();
            make_agglomeration(msh_i, level_set_function); // Non serve modificarla
            tc_agglo.toc();
                      
            std::cout << bold << yellow << "make_agglomeration: " << tc_agglo << " seconds" << reset << std::endl;
        }
        else
        {
            //move_nodes(msh_i, level_set_function);
            //detect_cut_faces2(msh_i, level_set_function); //do it again to update intersection points
            detect_cut_cells3(msh_i, level_set_function);
            //refine_interface2(msh_i, level_set_function, int_refsteps);
            //refine_interface_angle(msh_i, level_set_function, int_refsteps);
            refine_interface_pro3(msh_i, level_set_function, int_refsteps);
        }
       
        tc.toc();
        std::cout << bold << yellow << "-----> TIME -----> cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

        if (dump_debug)
        {
            dump_mesh(msh_i);
            output_mesh_info(msh_i, level_set_function);
        }
   
        
        //refine_interface2(msh_i, level_set_function, int_refsteps); // IN cuthho_geom
       
        // IN cuthho_export..Points/Nodes don't change-> it's fast
        if(time_step == 0){
            output_mesh_info2_pre_FEM(msh_i, level_set_function); // IN cuthho_export
        }
        else
            output_mesh_info2(msh_i, level_set_function);
            //output_mesh_info2(msh_i, level_set_function);
       
  
       
        typedef L2projected_level_set_high_order< Mesh , Fonction , FiniteSpace , T > Level_Set;
        // typedef L2projected_level_set_high_order_parallelize< Mesh , Fonction , FiniteSpace , T > Level_Set;
        auto ls_cell = LS_cell_L2proj_high_order< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);
        
        u_projected.set_agglo_mesh( msh_i );
        // CALCULATION OF AREA AND MASS AT TIME STEP t^n
        // CALCULATION ALSO OF CENTRE OF MASS
        
        
        /// DATA CHECK INITIALISATION
        T area0 = 0. , mass0 = 0.; // , global_mass0 = 0. ;
        T diff_area = 0. , diff_mass = 0. ;
        //T error_normal_global0 = 0. ;
        T centre_mass0_x = 0. , centre_mass0_y = 0. ;
        T perimeter0 = 0.;
        T normal_interface_status = 0. ;
        size_t counter_interface_pts = 0;
        T divergence_error = 0.;
        // PLOTTING OF NORMAL
                 
        
        postprocess_output<double> postoutput_vec;
        auto vec_normal = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_Stokes_initial.dat");
        
        auto vec_normal_cont = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_Continuos_Stokes.dat");
        
        postprocess_output<double> postoutput_div2;
        auto test_interface_divergence2  = std::make_shared< gnuplot_output_object<double> >("divergence_interface_Stokes_initial.dat");
        
       
        std::vector<T> val_u_nx , val_u_ny , val_u_n ;
        std::vector< point<T, 2> > interface_points_plot ;
        std::vector< std::pair<T,T> > interface_normals , interface_normals_cont ;
        
       
        tc_agglo.tic();
        for(auto& cl : msh_i.cells)
        {
            ls_cell.cell_assignment(cl);
            u_projected.cell_assignment(cl);
            
            if( location(msh_i, cl) == element_location::IN_NEGATIVE_SIDE || location(msh_i, cl) == element_location::ON_INTERFACE )
            {
                T partial_area = measure( msh_i, cl, element_location::IN_NEGATIVE_SIDE);
                area0 += partial_area;
                auto qps = integrate( msh_i , cl , 2*degree_FEM+1 , element_location::IN_NEGATIVE_SIDE);
                for(auto& qp:qps){
                    mass0 += qp.second * ls_cell(qp.first);
                    centre_mass0_x += qp.second * qp.first.x() ;
                    centre_mass0_y += qp.second * qp.first.y() ;
                }
            }
            if(cl.user_data.location == element_location::ON_INTERFACE)
            {
                /*
                std::cout<<"CELL = "<<offset(msh,cl)<<std::endl;
                if(offset(msh,cl) == 101 || offset(msh,cl) == 104 )
                {
                    for(size_t kk = 0 ; kk < u_projected.sol_HHO.first.rows() ; kk++ )
                    std::cout<< u_projected.sol_HHO.first(kk,offset(msh,cl)) << " " << u_projected.sol_HHO.second(kk,offset(msh,cl)) << '\n';
                    std::cout<<'\n'<<std::endl;
                }
                */
                for(auto interface_point = cl.user_data.interface.begin() ; interface_point < cl.user_data.interface.end() -1 ; interface_point++ )
                {
                    
                    
                    T val = ls_cell.divergence( *interface_point );
                    divergence_error += pow((std::abs(val) - 1.0/radius),2) ;
                    
                    Eigen::Matrix<T,2,1> normal = ls_cell.normal(*interface_point);
                    std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                    interface_normals.push_back( normal_vec ) ;
                    
                    /// COSE  PER NORMALE CONTINUA
                    /*
                    Eigen::Matrix<T,2,1> normal_cont = ls_cell.normal_continuous(*interface_point);
                    //std::cout<<"normal = "<<'\n'<<normal<<" , CONTIUOUS_normal = "<<'\n'<<normal_cont<<std::endl;
                    std::pair<T,T> normal_vec_cont = std::make_pair(normal_cont(0),normal_cont(1));
                    interface_normals_cont.push_back( normal_vec_cont ) ;
                    vec_normal_cont->add_data(*interface_point,normal_vec_cont);
                    */
                    if( time_step == 0 )
                    {
                        
                        vec_normal->add_data(*interface_point,normal_vec);
                        test_interface_divergence2->add_data(*interface_point , val);
                    }
                    
                    perimeter0 += ( *(interface_point+1) - *interface_point ).to_vector().norm();
                    
                    normal_interface_status += pow( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) , 2) + pow( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) , 2 );
                    
                    interface_points_plot.push_back(*(interface_point)) ;
                    val_u_nx.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) );
                    val_u_ny.push_back( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                    val_u_n.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) + u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                    //std::cout<<"*(interface_point) = "<<*(interface_point)<<std::endl;
                    //std::cout<<" u_projected primo = "<<u_projected(*(interface_point)).first  << " u_projected second = "<< u_projected(*(interface_point)).second  << " somma tot con olds = "<<normal_interface_status<<std::endl;
                    //std::cout<<" ls_cell.normal(*(interface_point))(0)  = "<<ls_cell.normal(*(interface_point))(0)  << " ls_cell.normal(*(interface_point))(1)  = "<< ls_cell.normal(*(interface_point))(1)  << std::endl;
                    
                    counter_interface_pts++;
                       
                }
                
                T val = ls_cell.divergence(*(cl.user_data.interface.end()-1));
                divergence_error += pow((std::abs(val) - 1.0/radius),2) ;
                
                Eigen::Matrix<T,2,1> normal = ls_cell.normal(*(cl.user_data.interface.end()-1));
                std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                interface_normals.push_back( normal_vec ) ;
                
                /// COSE  PER NORMALE CONTINUA
                /*
                Eigen::Matrix<T,2,1> normal_cont = ls_cell.normal_continuous(*(cl.user_data.interface.end()-1));
                std::pair<T,T> normal_vec_cont = std::make_pair(normal_cont(0),normal_cont(1));
                interface_normals_cont.push_back( normal_vec_cont ) ;
               
                 vec_normal_cont->add_data(*(cl.user_data.interface.end()-1) ,normal_vec_cont);
                */
                
                if( time_step == 0 )
                {
                    
                    vec_normal->add_data(*(cl.user_data.interface.end()-1) ,normal_vec);
                    test_interface_divergence2->add_data( *(cl.user_data.interface.end()-1) ,val );
                }
                                   
               
                normal_interface_status += pow( u_projected (*(cl.user_data.interface.end()-1) ).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0),2) + pow( u_projected(*( cl.user_data.interface.end()-1) ).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1), 2);
                
                
            interface_points_plot.push_back(*(cl.user_data.interface.end()-1)) ;
                val_u_nx.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) );
                val_u_ny.push_back( u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                val_u_n.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) + u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                
                
                counter_interface_pts++;
                   
               
            
            }
            
        }
        
        if( time_step == 0 )
        {
            postoutput_div2.add_object(test_interface_divergence2);
            postoutput_div2.write();
            
            postoutput_vec.add_object(vec_normal);
            postoutput_vec.add_object(vec_normal_cont);
            postoutput_vec.write();
            
        }
        goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
                   
        testing_level_set_time(msh,level_set_function,tot_time);
        /*
        if( time_step == 0 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
            
        if( time_step == 5 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 10 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 15 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
            
        if( time_step == 20 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        if( time_step == 30 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 40 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        */
        
        divergence_error /= counter_interface_pts;
        divergence_error = sqrt(divergence_error);
        std::cout<<yellow<<bold<<"The l2 error of the DIVERGENCE at the INTERFACE, at time "<<reset<< dt*time_step <<" is " << divergence_error <<std::endl;
                   
        
        std::cout<<"number of interface points is " << counter_interface_pts << std::endl;
        normal_interface_status /= counter_interface_pts;
        normal_interface_status = sqrt(normal_interface_status);
        std::cout<<yellow<<bold<<"The l2 error of u*n over the INTERFACE, at time "<<reset<< tot_time <<" is " << normal_interface_status << std::endl;
        if(time_step == 0)
            check = 10.0;
        else
            check = normal_interface_status ;
        
        std::cout << "Area at time step: "<<tot_time<<" is "<< area0  << reset << std::endl;
        std::cout  << "Internal mass at time step: "<<tot_time<<" is "<<reset<< mass0   << std::endl;
        //std::cout<<bold<<yellow << "GLOBAL Mass at time step: "<<time_step<<" is "<<reset<<global_mass0<< reset << std::endl;
         std::cout << "CENTRE OF MASS at time step: " <<tot_time<<" is "<<" ( "<< centre_mass0_x/area0  << " , "<< centre_mass0_y/area0 <<" ). " << reset << std::endl;
        
         d_a = sqrt(4.0*area0/M_PI) ;
        
        std::cout<<"The PERIMETER, at time "<< tot_time <<" is " << perimeter0 <<std::endl;
        
        std::cout<<yellow<<bold<<"The CIRCULARITY, at time "<< tot_time <<" is "<<reset << M_PI*d_a/perimeter0 <<std::endl;
        
        
        
        if(time_step == 0 ){
            initial_area  = area0 ;
            initial_mass = mass0 ;
            centre_mass_x_inital = centre_mass0_x ;
            centre_mass_y_inital = centre_mass0_y ;
            perimeter_initial = perimeter0 ;
        }
        
        if(time_step > 0 )
        {
            diff_area = (area0 - initial_area)/initial_area ;
            diff_mass = (std::abs((mass0 - initial_mass)))/(std::abs(initial_mass)) ;
            std::cout << bold << yellow << "Normalised difference in Area (new - old)/old at time step: "<<tot_time<<" is "<<reset<< diff_area  << reset << std::endl;
            std::cout << bold << yellow << "Difference in internal MASS |new - old|/|old| at time step: "<<tot_time<<" is "<<reset<< diff_mass  << reset << std::endl;
            
            std::cout << "TRANSLATION OF THE CENTRE OF MASS at time step: "  <<tot_time<<" is "<<" ( " << centre_mass0_x/area0 - centre_mass_x_inital/initial_area <<" , " << centre_mass0_y/area0 - centre_mass_y_inital/initial_area<<" ). " << std::endl;
            std::cout<<"NORMALISED DIFFERENCE PERIMETER, at time " << tot_time <<" is " << (perimeter0 - perimeter_initial)/perimeter_initial <<std::endl;
                   
        }
        
        
        tc_agglo.toc();
        std::cout << bold << yellow << "------> TIME ---->CHECK GOAL QUANTITIES: " << tc_agglo << " seconds" << reset << std::endl;
        // auto test_case = make_test_case_stokes_1(msh, level_set_function);
        // auto test_case = make_test_case_stokes_2(msh, ls_cell); //level_set_function);
        
/// DA AGGIUNGERE UNA VOLTA SISTEMATO IL CODICE
        /*
        if(check < 1e-8 )
        {
            std::cout<<" check = "<<check<<" , STOP!"<<std::endl;
            return 0;
        }
        */
        bool sym_grad = TRUE;
        
        auto prm = params<T>();
        prm.kappa_1 = 1.0;
        prm.kappa_2 = 1.0;
        
        //auto test_case = make_test_case_eshelby(msh_i, ls_cell,  prm , sym_grad);
     
        
        // Non serve modificare Gamma = 1/2
        tc_agglo.tic();
        auto test_case = make_test_case_eshelby_2(msh_i, ls_cell,  prm , sym_grad );
        //auto test_case = make_test_case_eshelby_analytic(msh_i, ls_cell,  prm , sym_grad , radius);
        tc_agglo.toc();
        std::cout << bold << yellow << "make_test_case_eshelby_2: " << tc_agglo << " seconds" << reset << std::endl;
        tc_agglo.tic();
        
        auto test_case_prova = make_test_case_eshelby_2_prova(msh_i, ls_cell,  prm , sym_grad );
        tc_agglo.toc();
        
        
       std::cout << bold << yellow << "make_test_case_eshelby_2 PROVA: " << tc_agglo << " seconds" << reset << std::endl;
        //auto method = make_sym_gradrec_stokes_interface_method(msh_i, 1.0, 0.0, test_case, sym_grad);
        
        auto method = make_sym_gradrec_stokes_interface_method(msh_i, 1.0, 0.0, test_case_prova, sym_grad);
      
        //auto u_projected = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
       tc_agglo.tic();
    
        if(solve_interface){
            //run_cuthho_interface_velocity_parallel(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad );
            run_cuthho_interface_velocity_prova(msh_i, degree, method,test_case_prova, ls_cell , u_projected ,sym_grad );
            //run_cuthho_interface_velocity(msh_i, degree, method, test_case, ls_cell , u_projected ,sym_grad );
            
            // OLD
            //run_cuthho_interface_velocity_analytic(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad ,radius );
            
            //run_cuthho_interface(msh, degree, method, test_case);
        }
        if (solve_fictdom)
            run_cuthho_fictdom(msh_i, degree, test_case);
        
        tc_agglo.toc();
        std::cout << bold << yellow << "run_cuthho_interface_velocity: " << tc_agglo << " seconds" << reset << std::endl;
        
        
        /*********************** FEM -  PROCESSING **************************/
        /// ORA HO SMOOTH OPERATOR! USE L2 PROJECTION
        
        tc_agglo.tic();
        if( 1 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: SMOOTH OPERATOR FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.smooth_converting_into_FE_formulation( u_projected.sol_HHO );
        }
        if( 0 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: OLD OPERATOR FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.converting_into_FE_formulation( u_projected.sol_HHO );
        }
        if( 0 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: L^2 PROJECTION FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.L2_proj_into_FE_formulation(level_set_function , msh);
        }
        
        tc_agglo.toc();
        std::cout << bold << yellow << "smooth_converting_into_FE_formulation: " << tc_agglo << " seconds" << reset << std::endl;
        
        /*
        auto u_prova = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
        u_prova.sol_HHO = u_projected.sol_HHO ;
        u_prova.L2_proj_into_FE_formulation( level_set_function , msh );
        
        
        testing_velocity_field_L2projected(msh , u_prova) ;
        
        //std::cout<<"CHECK SMOOTH CONVERTING  FEM ----> FIRST"<<'\n'<<(u_prova.sol_FEM.first - u_projected.sol_FEM.first)<<'\n' <<std::endl;
        //std::cout<<"CHECK SMOOTH CONVERTING FEM ----> SECOND"<<'\n'<<(u_prova.sol_FEM.second - u_projected.sol_FEM.second)<<'\n' <<std::endl;
        
        testing_velocity_field(msh , u_projected) ;
        */
        
         tc_agglo.tic();
        testing_velocity_field(msh , u_projected) ;
        
        tc_agglo.toc();
        std::cout << bold << yellow << "testing_velocity_field: " << tc_agglo << " seconds" << reset << std::endl;
        //check_inlet( msh , fe_data ,  u_projected , 1e-14 );
        //std::cout<<"u_projected.sol_FEM.first = "<<'\n'<<u_projected.sol_FEM.first <<std::endl;
        //std::cout<<"u_projected.sol_FEM.second = "<<'\n'<<u_projected.sol_FEM.second <<std::endl;
        T eps = 0.48 ; // factor to be inside CFL stability zone
        //T dt1 = time_step_CFL( u , mip , eps ); // OLD IMPLEMENTATION
        T dt1 = time_step_CFL_new( u_projected , mip , eps );
        //std::cout<<"dt1 is "<<dt1<<std::endl;
        T dt2 = 0.1;
        dt = std::min(dt1 , dt2);
        std::cout<<"MAX dt = "<<dt<<" AND HEURISTIC CFL IS "<<dt1<<std::endl;
        
        /*
        if(high_order)
            run_FEM_BERNSTEIN_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
            //run_FEM_BERNSTEIN_CORRECT( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
        else
            run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
            //run_FEM_BERNSTEIN_LOW_ORDER_CORRECT( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
        */
        
        // I can create a sub-time. I solve several time the FEM problem, given a Stokes field. The amount of time is s.t. at maximum there is a displacement of a cell of the interface and no more than a maximum T
        T sub_time = 0.;
        T sub_dt = std::min(4e-4 , dt ) ;
        std::cout<<"Implemented dt = "<<dt<<std::endl;
        while( (sub_time < sub_dt*10) && (sub_time < dt1) )
        {
            if(high_order){
                run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D_NEW_DIRICHLET_COND( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt );

                //run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
                //run_FEM_BERNSTEIN_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
            }
            else
                run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
               
            sub_time += sub_dt ;
      
        }
        std::cout<<yellow<<bold<<"SUB TIME REPETITION OF TRANSPORT PB DATA:"<<reset<<" sub_dt = "<<sub_dt<< " , number time steps = "<<sub_time/sub_dt<<std::endl;
        
        tot_time += sub_time ;
        
        testing_level_set2(msh,level_set_function);
        
        /// OLD IMPLEMENTATION
        //for(size_t j=1; j<4 ; j++)
        //run_FEM_levelset( level_set_function.msh,degree_FEM,level_set_function,u,dt,mip);
       
        /************** FEM -  POST-PROCESSING **************/
        if( (T_N - time_step)==0 )
        {
            // Uploading continuous normal function
            level_set_function.normal_continuous_setting();
            // Uploading mesh data to check out differences in mass and areas
            crr_mesh.current_mesh = msh;
            Mesh msh_i2 =  crr_mesh.current_mesh;
            offset_definition(msh_i2);
            tc.tic();
            detect_node_position3(msh_i2, level_set_function); // In cuthho_geom
            detect_cut_faces3(msh_i2, level_set_function); // In cuthho_geom
        
       
            if (agglomeration)
            {
                // std::cout<<"i m here 1"<<std::endl;
                detect_cut_cells3(msh_i2, level_set_function); // In cuthho_geom
           
                detect_cell_agglo_set(msh_i2, level_set_function); // Non serve modificarla
           
                make_neighbors_info_cartesian(msh_i2); // Non serve modificarla
                //refine_interface_angle(msh_i2, level_set_function, int_refsteps); // IN cuthho_geom
                refine_interface_pro3(msh_i2, level_set_function, int_refsteps); // IN cuthho_geom
                make_agglomeration(msh_i2, level_set_function); // Non serve modificarla
            }
            else
            {
                //move_nodes(msh_i2, level_set_function);
                //detect_cut_faces2(msh_i2, level_set_function); //do it again to update intersection     points
                detect_cut_cells3(msh_i2, level_set_function);
                //refine_interface2(msh_i2, level_set_function, int_refsteps);
                //refine_interface_angle(msh_i2, level_set_function, int_refsteps);
                refine_interface_pro3(msh_i2, level_set_function, int_refsteps);
            }
        
            tc.toc();
            std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

            if (dump_debug)
            {
                dump_mesh(msh_i2);
                output_mesh_info(msh_i2, level_set_function);
            }
        
            output_mesh_info2(msh_i2, level_set_function);
            // IN cuthho_export..Points/Nodes don't change
           
            // Uploading level set
            ls_cell.level_set = level_set_function;
            ls_cell.agglo_msh = msh_i2;
            u_projected.set_agglo_mesh( msh_i2 );
            
           
            T mass_fin = 0. , area_fin = 0. ;
            T centre_mass_x = 0. , centre_mass_y = 0. ;
        
            T perimeter = 0. ;
            normal_interface_status = 0. ;
            counter_interface_pts = 0;
            T divergence_error_fin = 0.;
            
            postprocess_output<double> postoutput_vec;
            auto vec_normal_fin = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_Stokes_final.dat");
            
            postprocess_output<double> postoutput_div2;
            auto test_interface_divergence_fin  = std::make_shared< gnuplot_output_object<double> >("divergence_interface_Stokes_final.dat");
            
            std::vector<T> val_u_nx_fin , val_u_ny_fin , val_u_n_fin ;
            std::vector< point<T, 2> > interface_points_plot_fin ;
            std::vector< std::pair<T,T> > interface_normals_fin ;
            
            for(auto& cl : msh_i2.cells)
            {
                ls_cell.cell_assignment(cl);
                u_projected.cell_assignment(cl);
                
                if( (location(msh_i2, cl) == element_location::IN_NEGATIVE_SIDE) || (location(msh_i2, cl) == element_location::ON_INTERFACE) )
                {
                    
                    T partial_area = measure( msh_i2, cl, element_location::IN_NEGATIVE_SIDE);
                 
                    area_fin += partial_area;
                    
                   
                    auto qps_fin = integrate( msh_i2 , cl , 2*degree_FEM+1 , element_location::IN_NEGATIVE_SIDE);
                   
                    for(auto& qp:qps_fin){
                        mass_fin += qp.second * ls_cell(qp.first);
                        centre_mass_x += qp.second * qp.first.x() ;
                        centre_mass_y += qp.second * qp.first.y() ;
                    }
                   
                }
                if(cl.user_data.location == element_location::ON_INTERFACE)
                {
                    
                    for(auto interface_point = cl.user_data.interface.begin() ; interface_point < cl.user_data.interface.end() -1 ; interface_point++ )
                    {
                        perimeter += ( *(interface_point+1) - *interface_point ).to_vector().norm();
                        
                        normal_interface_status += pow( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) , 2) + pow( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) , 2 );
                        
                        
                        T val = ls_cell.divergence( *interface_point );
                        divergence_error_fin += pow((std::abs(val) - 1.0/radius),2) ;
                        Eigen::Matrix<T,2,1> normal = ls_cell.normal(*interface_point);
                        std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                        interface_normals_fin.push_back( normal_vec ) ;
                        
                        vec_normal_fin->add_data(*interface_point,normal_vec);
                        test_interface_divergence_fin->add_data( *interface_point , val);
                        
                        interface_points_plot_fin.push_back( *(interface_point) ) ;
                        val_u_nx_fin.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) );
                        val_u_ny_fin.push_back( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                        val_u_n_fin.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) + u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                        
                        
                        counter_interface_pts++;
                    }
                    
                    T val = ls_cell.divergence( *(cl.user_data.interface.end()-1) );
                    divergence_error_fin += pow((std::abs(val) - 1.0/radius),2) ;
                    
                    Eigen::Matrix<T,2,1> normal = ls_cell.normal(*(cl.user_data.interface.end()-1));
                                       
                    std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                    interface_normals_fin.push_back( normal_vec ) ;
                    
                    vec_normal_fin->add_data( *(cl.user_data.interface.end()-1) ,normal_vec);
                    test_interface_divergence_fin->add_data( *(cl.user_data.interface.end()-1) ,val );
                    
                    normal_interface_status += pow( u_projected (*(cl.user_data.interface.end()-1) ).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0),2) + pow( u_projected(*( cl.user_data.interface.end()-1) ).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1), 2);
                    
                    interface_points_plot_fin.push_back( *(cl.user_data.interface.end()-1) ) ;
                    val_u_nx_fin.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) );
                    val_u_ny_fin.push_back( u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                    val_u_n_fin.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) + u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                                
                    counter_interface_pts++;

                }
                
            }
            
            postoutput_div2.add_object(test_interface_divergence_fin);
            postoutput_div2.write();
                       
            postoutput_vec.add_object(vec_normal_fin);
            postoutput_vec.write();
            goal_quantities_time(msh , tot_time, interface_points_plot_fin , val_u_nx_fin , val_u_ny_fin , val_u_n_fin , interface_normals_fin ) ;
            testing_level_set_time(msh,level_set_function, tot_time);
            /*
            if( time_step == 9 ){
                goal_quantities_time(msh , tot_time, interface_points_plot_fin , val_u_nx_fin , val_u_ny_fin , val_u_n_fin , interface_normals_fin ) ;
                testing_level_set_time(msh,level_set_function, tot_time);
            }
            */
            divergence_error_fin /= counter_interface_pts;
            divergence_error_fin = sqrt(divergence_error_fin);
            std::cout<<yellow<<bold<<"The l2 error of the DIVERGENCE at the INTERFACE, at time "<<reset<< tot_time <<" is " << divergence_error_fin <<std::endl;
            
            std::cout<<"number of interface points is " << counter_interface_pts << std::endl;
            
            normal_interface_status /= counter_interface_pts;
            normal_interface_status = sqrt(normal_interface_status);
            
            std::cout<<yellow<<bold<<"The l2 error of u*n over the INTERFACE, at time "<<reset<< tot_time <<" is " << normal_interface_status << std::endl;
            
            
            
            std::cout<<"The PERIMETER, at time "<< tot_time <<" is " << perimeter <<std::endl;
            
            std::cout<<"perimeter = "<< perimeter << " AND  perimeter0 =  "<<perimeter_initial<<std::endl;
            std::cout<< bold << yellow<<"NORMALISED DIFFERENCE PERIMETER, at time "<<reset<< tot_time <<" is " << (perimeter - perimeter_initial)/perimeter_initial <<std::endl;
            
            d_a = sqrt(4.0*area_fin/M_PI) ;
            
            std::cout<< bold << yellow<<"The CIRCULARITY, at time "<< tot_time<<reset <<" is " << M_PI*d_a/perimeter <<std::endl;
            
            std::cout  << "Area at time step: " <<tot_time<<" is "<< area_fin << std::endl;
            std::cout << "Internal mass at time step: "<<tot_time<<" is "<< mass_fin << reset << std::endl;
            
            std::cout << bold << yellow << "NORMALISED Difference in AREA AT TIME "<<tot_time<<" IS "<< reset<< (area_fin - initial_area)/initial_area << std::endl;
            std::cout << bold << yellow << "NORMALISED Difference in INTERNAL MASS AT TIME "<<tot_time<<" IS "<< reset<< (std::abs(mass_fin - initial_mass))/(std::abs( initial_mass )) << std::endl;
            std::cout << "CENTRE OF MASS at time step: "<<tot_time<<" is "<<" ( " << centre_mass_x/area_fin <<" , " << centre_mass_y/area_fin<<" ). " << std::endl;
            std::cout << "TRANSLATION OF THE CENTRE OF MASS at time step: "  <<tot_time<<" is "<<" ( " << centre_mass_x/area_fin - centre_mass_x_inital/initial_area <<" , " << centre_mass_y/area_fin - centre_mass_y_inital/initial_area<<" ). " << std::endl;
            
            
            
           
        } // END OF T = FINAL TIME
       
        time_pos +=dt ;
        //time_step++;
    } // End of the temporal loop
    
    std::cout<< bold << yellow <<"FINAL TIME IS t = "<< reset<<tot_time<<std::endl;
    return 0;
}
#endif


// CHECK gamma = 1/2 . POI CANCELLARE
#if 0
int main(int argc, char **argv)
{
    using RealType = double;

    size_t degree           = 0;
    size_t int_refsteps     = 4;
    size_t degree_FEM       = 0;

    bool dump_debug         = false;
    bool solve_interface    = false;
    bool solve_fictdom      = false;
    bool agglomeration      = false;
    
    bool high_order = false ; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
    bool cut_off_active = false ; // IF FALSE IS SMOOTH, IF TRUE  CUT_OFF

    mesh_init_params<RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;
    RealType d = 0.5;
    size_t T_N = 0;
    /* k <deg>:     method degree
     * g<deg>:  method FEM degree
     * M <num>:     number of cells in x direction
     * N <num>:     number of cells in y direction
     * r <num>:     number of interface refinement steps
     *
     * i:           solve interface problem
     * f:           solve fictitious domain problem
     *
     * D:           use node displacement to solve bad cuts (default)
     * A:           use agglomeration to solve bad cuts
     *
     * d:           dump debug data
     */

    int ch;
    while ( (ch = getopt(argc, argv, "k:q:M:N:r:T:ifDAdhc")) != -1 )
    {
        switch(ch)
        {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'q':
                degree_FEM = atoi(optarg);
                break;
        
            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;
            
            case 'T':
                T_N = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;
            
            case 'h':
                high_order = true;
            break;
            
            case 'c':
                cut_off_active = true;
            break;
            

            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;

    
    
    timecounter tc;

    /************** BUILD MESH **************/
    Matrix<RealType, Dynamic, 1> vel0 , vel1 ;
    Matrix<RealType, Dynamic, 1> phi0 , phi1 , phi2 , phi3;
    Matrix<RealType, Dynamic, 1> vecprova0 , vecprova1 , vecprova2 ;
    Matrix<RealType, Dynamic, 1> vecprovabis0 , vecprovabis1  ,vecprovabis7 ,vecprovabis6 ;
    Matrix<RealType, Dynamic, Dynamic> vecprovabis2 ,vecprovabis3 ,vecprovabis4,vecprovabis5 ;
    tc.tic();
    cuthho_poly_mesh<RealType> msh(mip);
    typedef cuthho_poly_mesh<RealType> Mesh;
    offset_definition(msh);
    /************** FINITE ELEMENT INITIALIZATION **************/
       auto fe_data = Finite_Element<RealType,Mesh>( msh , degree_FEM , mip ) ;
       typedef Finite_Element<RealType,Mesh> FiniteSpace;
       
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;
    vecprovabis2 = Matrix<RealType, Dynamic, Dynamic>::Zero(fe_data.local_ndof , msh.cells.size() ) ;
        
    vecprovabis3 = Matrix<RealType, Dynamic, Dynamic>::Zero(fe_data.local_ndof , msh.cells.size() ) ;
    
    vecprovabis4 = Matrix<RealType, Dynamic, Dynamic>::Zero(fe_data.local_ndof , msh.cells.size() ) ;
    
    vecprovabis5 = Matrix<RealType, Dynamic, Dynamic>::Zero(fe_data.local_ndof , msh.cells.size() ) ;
           
    std::vector<point<RealType, 2>> node_interface0 , node_interface1   ;
    
    for( int i = 0 ; i < 2 ; i++)
    {
    
    
     /*
    size_t n_cells = msh.cells.size();
    std::cout<<" I m in parallel zone"<<std::endl;
    tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
    [&msh] (size_t & cell_ind){
            auto& cell = msh.cells[cell_ind];
            std::cout<<"CELL = "<<offset(msh,cell)<<std::endl;
             }
    );
    
   
#ifdef HAVE_INTEL_TBB
    size_t n_cells = msh.cells_size();
    std::cout<<" I m in parallel zone"<<std::endl;
    tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
    [&msh] (size_t & cell_ind){
            auto& cell = msh.cells[cell_ind];
             }
    );
#else
    std::cout<<" I m in sequential zone"<<std::endl;
    for (size_t cell_ind = 0; cell_ind < msh.cells.size(); cell_ind++)
    {
        auto& cell = msh.cells[cell_ind];
    }
#endif
    */
    
   
    /************** ANALYTIC LEVEL SET FUNCTION  **************/
    typedef RealType T;
    
    bool circle = false , ellipse = true ;
    
    RealType radius_a , radius_b , radius ;
    RealType x_centre = 0.5;
    RealType y_centre = 0.5;
    RealType C ;
    T h = std::max( fe_data.hx , fe_data.hy) ;
    if(circle)
    {
        radius = 1.0/9.0;
    }
        
    if(ellipse)
    {
        radius_a = 1.0/6.0;
        radius_b = 1.0/12.0;
        //radius_a = 1.0/9.0;
        //radius_b = 1.0/9.0;
        std::cout << bold << yellow << "Initial Analytic Area of the ELLIPSE: "<< M_PI*radius_a*radius_b << std::endl;
        radius = sqrt( radius_a * radius_b ) ;
        std::cout << bold << yellow << "Final radius expected of the circle : " << radius <<reset<<std::endl;
    }
  
        
    
    /// THIS DATA BELOW HAS TO BE UPLOAD DEPENDING ON THE PROBLEM.
    
    //auto level_set_function_anal = circle_level_set<RealType>(radius, x_centre, y_centre );
    //typedef  circle_level_set<T> Fonction;
    
    auto level_set_function_anal = elliptic_level_set<RealType>( radius_a, radius_b, x_centre, y_centre);
    typedef  elliptic_level_set<T> Fonction;
    
    //auto level_set_function_anal = elliptic_distance_ls<RealType>( radius_a, radius_b, x_centre, y_centre , h);
    //typedef  elliptic_distance_ls<T> Fonction;
    
    //auto level_set_function_anal = circle_distance_ls<RealType>(radius, x_centre, y_centre , 2*h );
    //typedef  circle_distance_ls<T> Fonction;
    
    
   
    /**************  VELOCITY FIELD  INITIALISATION  **************/
  
    auto u_projected = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
    
    /************** LEVEL SET FUNCTION DISCRETISATION **************/
    std::cout<<"degree FEM "<<degree_FEM<<std::endl;
   
    
    auto level_set_function = L2projected_level_set_high_order< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);
    
   
    //level_set_function.iso_val_interface = 0.5 ;
    //level_set_function.coefficients_mapping_quadratic( );
    //level_set_function.coefficients_mapping_MAX_MAX( );
    if(i == 0)
        level_set_function.coefficients_sfasamento( );
    
   
    if(high_order)
        std::cout<<bold<<yellow<<"----> USING phi_HP HIGH order!!!!! "<<reset<<std::endl;
    else
        std::cout<<bold<<yellow<<"----> USING phi_L LOW order!!!!! "<<reset<<std::endl;
    
    
    testing_level_set(msh,level_set_function);
     
    // Initiliatisation data for time routine
    auto crr_mesh =  Current_Mesh<Mesh>(msh);
    
    // Initialisation area , mass
    T initial_area = 0. , initial_mass = 0.;
    T area_previous_time = 0. , mass_previous_time = 0. , dt = 0. ;
    
    /// DATA CHECK INITIALISATION
    T d_a = 0. ;
    T error_normal_global = 0. ;
    T error_normal_local = 0. ;
    T perimeter_initial = 0. ;
    T centre_mass_x_inital = 0. , centre_mass_y_inital = 0. ;
    
    
    
    T check = 10.0;
    T time_pos = 0.;
    
    T tot_time = 0.;
    
    bool bdry_bottom = false , bdry_up = false ;
    bool bdry_left = false , bdry_right = false ;
    
    check_inlet( msh , fe_data , bdry_bottom , bdry_right , bdry_up , bdry_left, 1e-14 );
    
    size_t time_step = 0; // add to prove velocity error
   // for (size_t time_step = 0; time_step<=T_N; time_step++)
   // {
       
        

        // ************** Re-Initialization mesh **************
        crr_mesh.current_mesh = msh;
        Mesh msh_i =  crr_mesh.current_mesh;
        offset_definition(msh_i);
        

        //************ DO cutHHO MESH PROCESSING **************
        tc.tic();
        detect_node_position3(msh_i, level_set_function); // In cuthho_geom
        detect_cut_faces3(msh_i, level_set_function); // In cuthho_geom
        std::cout<<"FINE DETECT NODES AND FACES."<<std::endl;
        if (agglomeration)
        {
            detect_cut_cells3(msh_i, level_set_function); // In cuthho_geom
            std::cout<<"FINE DETECT CELLS."<<std::endl;
            detect_cell_agglo_set(msh_i, level_set_function); // Non serve modificarla
            make_neighbors_info_cartesian(msh_i); // Non serve modificarla
            //refine_interface_angle(msh_i, level_set_function, int_refsteps);
            refine_interface_pro3(msh_i, level_set_function, int_refsteps);
            //refine_interface2(msh_i, level_set_function, int_refsteps); // IN cuthho_geom
            make_agglomeration(msh_i, level_set_function); // Non serve modificarla
        }
        else
        {
            //move_nodes(msh_i, level_set_function);
            //detect_cut_faces2(msh_i, level_set_function); //do it again to update intersection points
            detect_cut_cells3(msh_i, level_set_function);
            //refine_interface2(msh_i, level_set_function, int_refsteps);
            //refine_interface_angle(msh_i, level_set_function, int_refsteps);
            refine_interface_pro3(msh_i, level_set_function, int_refsteps);
        }
       
        tc.toc();
        std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

        
        for(auto& cl : msh_i.cells)
        {
            for(auto& pt : cl.user_data.interface)
            {
                if(i == 0)
                    node_interface0.push_back( pt );
                else
                    node_interface1.push_back( pt );
            }
                
        }
        std::cout<<"node_interface0 - node_interface1"<<'\n';
        for(size_t ciccia = 0 ; ciccia <  node_interface1.size() ; ciccia ++)
            std::cout<<node_interface0[ciccia] - node_interface1[ciccia]<<std::endl;
        
        
        
        if (dump_debug)
        {
            dump_mesh(msh_i);
            output_mesh_info(msh_i, level_set_function);
        }
   
        // IN cuthho_export..Points/Nodes don't change-> it's fast
        if(time_step == 0){
            output_mesh_info2_pre_FEM(msh_i, level_set_function); // IN cuthho_export
        }
        else
            output_mesh_info2(msh_i, level_set_function);
            //output_mesh_info2(msh_i, level_set_function);
        
        typedef L2projected_level_set_high_order< Mesh , Fonction , FiniteSpace , T > Level_Set;
        auto ls_cell = LS_cell_L2proj_high_order< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);
        
        
        u_projected.set_agglo_mesh( msh_i );
        // CALCULATION OF AREA AND MASS AT TIME STEP t^n
        // CALCULATION ALSO OF CENTRE OF MASS
        
        
        /// DATA CHECK INITIALISATION
        T area0 = 0. , mass0 = 0. , global_mass0 = 0. ;
        T diff_area = 0. , diff_mass = 0. ;
        T error_normal_global0 = 0. ;
        T centre_mass0_x = 0. , centre_mass0_y = 0. ;
        T perimeter0 = 0.;
        T normal_interface_status = 0. ;
        size_t counter_interface_pts = 0;
        T divergence_error = 0.;
        // PLOTTING OF NORMAL
                 
        
        postprocess_output<double> postoutput_vec;
        auto vec_normal = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_Stokes_initial.dat");
        
        postprocess_output<double> postoutput_div2;
        auto test_interface_divergence2  = std::make_shared< gnuplot_output_object<double> >("divergence_interface_Stokes_initial.dat");
        
       
        std::vector<T> val_u_nx , val_u_ny , val_u_n ;
        std::vector< point<T, 2> > interface_points_plot ;
        std::vector< std::pair<T,T> > interface_normals ;
        
        for(auto& cl : msh_i.cells)
        {
            ls_cell.cell_assignment(cl);
            u_projected.cell_assignment(cl);
            
            if( location(msh_i, cl) == element_location::IN_NEGATIVE_SIDE || location(msh_i, cl) == element_location::ON_INTERFACE )
            {
                T partial_area = measure( msh_i, cl, element_location::IN_NEGATIVE_SIDE);
                area0 += partial_area;
                auto qps = integrate( msh_i , cl , 2*degree_FEM+1 , element_location::IN_NEGATIVE_SIDE);
                for(auto& qp:qps){
                    mass0 += qp.second * ls_cell(qp.first);
                    centre_mass0_x += qp.second * qp.first.x() ;
                    centre_mass0_y += qp.second * qp.first.y() ;
                }
            }
            if(cl.user_data.location == element_location::ON_INTERFACE)
            {
                /*
                std::cout<<"CELL = "<<offset(msh,cl)<<std::endl;
                if(offset(msh,cl) == 101 || offset(msh,cl) == 104 )
                {
                    for(size_t kk = 0 ; kk < u_projected.sol_HHO.first.rows() ; kk++ )
                    std::cout<< u_projected.sol_HHO.first(kk,offset(msh,cl)) << " " << u_projected.sol_HHO.second(kk,offset(msh,cl)) << '\n';
                    std::cout<<'\n'<<std::endl;
                }
                */
                for(auto interface_point = cl.user_data.interface.begin() ; interface_point < cl.user_data.interface.end() -1 ; interface_point++ )
                {
                    
                    
                    T val = ls_cell.divergence( *interface_point );
                    divergence_error += pow((std::abs(val) - 1.0/radius),2) ;
                    Eigen::Matrix<T,2,1> normal = ls_cell.normal(*interface_point);
                    std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                    interface_normals.push_back( normal_vec ) ;
                    
                    if( time_step == 0 )
                    {
                        
                        vec_normal->add_data(*interface_point,normal_vec);
                        test_interface_divergence2->add_data(*interface_point , val);
                    }
                    
                    perimeter0 += ( *(interface_point+1) - *interface_point ).to_vector().norm();
                    
                    normal_interface_status += pow( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) , 2) + pow( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) , 2 );
                    
                    interface_points_plot.push_back(*(interface_point)) ;
                    val_u_nx.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) );
                    val_u_ny.push_back( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                    val_u_n.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) + u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                    //std::cout<<"*(interface_point) = "<<*(interface_point)<<std::endl;
                    //std::cout<<" u_projected primo = "<<u_projected(*(interface_point)).first  << " u_projected second = "<< u_projected(*(interface_point)).second  << " somma tot con olds = "<<normal_interface_status<<std::endl;
                    //std::cout<<" ls_cell.normal(*(interface_point))(0)  = "<<ls_cell.normal(*(interface_point))(0)  << " ls_cell.normal(*(interface_point))(1)  = "<< ls_cell.normal(*(interface_point))(1)  << std::endl;
                    
                    counter_interface_pts++;
                       
                }
                
                T val = ls_cell.divergence(*(cl.user_data.interface.end()-1));
                divergence_error += pow((std::abs(val) - 1.0/radius),2) ;
                Eigen::Matrix<T,2,1> normal = ls_cell.normal(*(cl.user_data.interface.end()-1));
                std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                interface_normals.push_back( normal_vec ) ;
                
                if( time_step == 0 )
                {
                    
                    vec_normal->add_data(*(cl.user_data.interface.end()-1) ,normal_vec);
                    test_interface_divergence2->add_data( *(cl.user_data.interface.end()-1) ,val );
                }
                                   
               
                normal_interface_status += pow( u_projected (*(cl.user_data.interface.end()-1) ).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0),2) + pow( u_projected(*( cl.user_data.interface.end()-1) ).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1), 2);
                
                
            interface_points_plot.push_back(*(cl.user_data.interface.end()-1)) ;
                val_u_nx.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) );
                val_u_ny.push_back( u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                val_u_n.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) + u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                
                
                counter_interface_pts++;
                   
               
            
            }
            
        }
        
        if( time_step == 0 )
        {
            postoutput_div2.add_object(test_interface_divergence2);
            postoutput_div2.write();
            
            postoutput_vec.add_object(vec_normal);
            postoutput_vec.write();
            
        }
        goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
                   
        testing_level_set_time(msh,level_set_function,tot_time);
        /*
        if( time_step == 0 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
            
        if( time_step == 5 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 10 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 15 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
            
        if( time_step == 20 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        if( time_step == 30 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 40 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        */
        
        divergence_error /= counter_interface_pts;
        divergence_error = sqrt(divergence_error);
        std::cout<<yellow<<bold<<"The l2 error of the DIVERGENCE at the INTERFACE, at time "<<reset<< dt*time_step <<" is " << divergence_error <<std::endl;
                   
        
        std::cout<<"number of interface points is " << counter_interface_pts << std::endl;
        normal_interface_status /= counter_interface_pts;
        normal_interface_status = sqrt(normal_interface_status);
        std::cout<<yellow<<bold<<"The l2 error of u*n over the INTERFACE, at time "<<reset<< tot_time <<" is " << normal_interface_status << std::endl;
        if(time_step == 0)
            check = 10.0;
        else
            check = normal_interface_status ;
        
        std::cout << "Area at time step: "<<tot_time<<" is "<< area0  << reset << std::endl;
        std::cout  << "Internal mass at time step: "<<tot_time<<" is "<<reset<< mass0   << std::endl;
        //std::cout<<bold<<yellow << "GLOBAL Mass at time step: "<<time_step<<" is "<<reset<<global_mass0<< reset << std::endl;
         std::cout << "CENTRE OF MASS at time step: " <<tot_time<<" is "<<" ( "<< centre_mass0_x/area0  << " , "<< centre_mass0_y/area0 <<" ). " << reset << std::endl;
        
         d_a = sqrt(4.0*area0/M_PI) ;
        
        std::cout<<"The PERIMETER, at time "<< tot_time <<" is " << perimeter0 <<std::endl;
        
        std::cout<<yellow<<bold<<"The CIRCULARITY, at time "<< tot_time <<" is "<<reset << M_PI*d_a/perimeter0 <<std::endl;
        
        
        
        if(time_step == 0 ){
            initial_area  = area0 ;
            initial_mass = mass0 ;
            centre_mass_x_inital = centre_mass0_x ;
            centre_mass_y_inital = centre_mass0_y ;
            perimeter_initial = perimeter0 ;
        }
        
        if(time_step > 0 )
        {
            diff_area = (area0 - initial_area)/initial_area ;
            diff_mass = (std::abs((mass0 - initial_mass)))/(std::abs(initial_mass)) ;
            std::cout << bold << yellow << "Normalised difference in Area (new - old)/old at time step: "<<tot_time<<" is "<<reset<< diff_area  << reset << std::endl;
            std::cout << bold << yellow << "Difference in internal MASS |new - old|/|old| at time step: "<<tot_time<<" is "<<reset<< diff_mass  << reset << std::endl;
            
            std::cout << "TRANSLATION OF THE CENTRE OF MASS at time step: "  <<tot_time<<" is "<<" ( " << centre_mass0_x/area0 - centre_mass_x_inital/initial_area <<" , " << centre_mass0_y/area0 - centre_mass_y_inital/initial_area<<" ). " << std::endl;
            std::cout<<"NORMALISED DIFFERENCE PERIMETER, at time " << tot_time <<" is " << (perimeter0 - perimeter_initial)/perimeter_initial <<std::endl;
                   
        }
        
        // auto test_case = make_test_case_stokes_1(msh, level_set_function);
        // auto test_case = make_test_case_stokes_2(msh, ls_cell); //level_set_function);
        
/// DA AGGIUNGERE UNA VOLTA SISTEMATO IL CODICE

        if(check < 1e-8 )
        {
            std::cout<<" check = "<<check<<" , STOP!"<<std::endl;
            return 0;
        }
        
        bool sym_grad = TRUE;
        
        auto prm = params<T>();
        prm.kappa_1 = 1.0;
        prm.kappa_2 = 1.0;
        
        //auto test_case = make_test_case_eshelby(msh_i, ls_cell,  prm , sym_grad);
       
        
        // Non serve modificare Gamma = 1/2
        auto test_case = make_test_case_eshelby_2(msh_i, ls_cell,  prm , sym_grad);
        
        auto method = make_sym_gradrec_stokes_interface_method(msh_i, 1.0, 0.0, test_case, sym_grad);
       
        
        //auto u_projected = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
       
    
        if(solve_interface){
            run_cuthho_interface_velocity(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad );
            
            //run_cuthho_interface(msh, degree, method, test_case);
        }
        if (solve_fictdom)
            run_cuthho_fictdom(msh_i, degree, test_case);
        

        
        
        /*********************** FEM -  PROCESSING **************************/
        /// ORA HO SMOOTH OPERATOR! USE L2 PROJECTION
        if( 1 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: SMOOTH OPERATOR FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.smooth_converting_into_FE_formulation( u_projected.sol_HHO );
        }
        if( 0 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: OLD OPERATOR FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.converting_into_FE_formulation( u_projected.sol_HHO );
        }
        if( 0 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: L^2 PROJECTION FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.L2_proj_into_FE_formulation(level_set_function , msh);
        }
        
        /*
        auto u_prova = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
        u_prova.sol_HHO = u_projected.sol_HHO ;
        u_prova.L2_proj_into_FE_formulation( level_set_function , msh );
        
        
        testing_velocity_field_L2projected(msh , u_prova) ;
        
        //std::cout<<"CHECK SMOOTH CONVERTING  FEM ----> FIRST"<<'\n'<<(u_prova.sol_FEM.first - u_projected.sol_FEM.first)<<'\n' <<std::endl;
        //std::cout<<"CHECK SMOOTH CONVERTING FEM ----> SECOND"<<'\n'<<(u_prova.sol_FEM.second - u_projected.sol_FEM.second)<<'\n' <<std::endl;
        
        testing_velocity_field(msh , u_projected) ;
        */
        testing_velocity_field(msh , u_projected) ;
        //check_inlet( msh , fe_data ,  u_projected , 1e-14 );
        //std::cout<<"u_projected.sol_FEM.first = "<<'\n'<<u_projected.sol_FEM.first <<std::endl;
        //std::cout<<"u_projected.sol_FEM.second = "<<'\n'<<u_projected.sol_FEM.second <<std::endl;
        T eps = 0.48 ; // factor to be inside CFL stability zone
        //T dt1 = time_step_CFL( u , mip , eps ); // OLD IMPLEMENTATION
        T dt1 = time_step_CFL_new( u_projected , mip , eps );
        //std::cout<<"dt1 is "<<dt1<<std::endl;
        T dt2 = 0.1;
        dt = std::min(dt1 , dt2);
        std::cout<<"MAX dt = "<<dt<<" AND HEURISTIC CFL IS "<<dt1<<std::endl;
        if ( i == 1){
            std::cout<<"----> CHECK vel u_projected.sol_FEM.first = "<<'\n'<<vel0 - u_projected.sol_FEM.first <<std::endl;
            std::cout<<"----> CHECK vel u_projected.sol_FEM.second = "<<'\n'<<vel1 - u_projected.sol_FEM.second <<std::endl;
        }
        vel0 = u_projected.sol_FEM.first ;
        vel1 = u_projected.sol_FEM.second ;
        
        //if ( i == 1){
        //    std::cout<<"----> CHECK level_set_function.sol_FEM PRE TRANSPORT = "<<'\n'<<phi1 - level_set_function.sol_FEM <<std::endl;
        //}
        //phi1 = level_set_function.sol_FEM ;
        
       
        T sub_time = 0.;
        T sub_dt = std::min(4e-4 , dt ) ;
        std::cout<<"Implemented dt = "<<dt<<std::endl;
        
       
        while( (sub_time < sub_dt*1) && (sub_time < dt1) )
        {
            if(high_order){
                run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D_NEW_DIRICHLET_COND( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt );

                //run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
                //run_FEM_BERNSTEIN_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
            }
            else
                run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST_to_check( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt , phi2 , phi3 , i , vecprova0 , vecprova1 , vecprova2 , vecprovabis0 , vecprovabis1 , vecprovabis2 ,vecprovabis3 ,vecprovabis4, vecprovabis5, vecprovabis6 ,vecprovabis7 );
               
            sub_time += sub_dt ;
      
        }
        std::cout<<yellow<<bold<<"SUB TIME REPETITION OF TRANSPORT PB DATA:"<<reset<<" sub_dt = "<<sub_dt<< " , number time steps = "<<sub_time/sub_dt<<std::endl;
        
        tot_time += sub_time ;
        
        if ( i == 1){
            Matrix<T, Dynamic, 1> one1 = Matrix<T, Dynamic, 1>::Ones(level_set_function.sol_FEM.size());
            std::cout<<"----> CHECK level_set_function.sol_FEM = "<<'\n'<<0.5*one1 - (phi0 - level_set_function.sol_FEM) <<std::endl;
        }
        phi0 = level_set_function.sol_FEM ;
       
        if ( i == 0){
            testing_level_set2(msh,level_set_function);
        }
        else
            testing_level_set2_bis(msh,level_set_function);
    
    }
         exit(1);
   
   
    return 0;
}
#endif




// CHECK gamma = 1/2 . POI CANCELLARE con analytica velocity
#if 0
int main(int argc, char **argv)
{
    using RealType = double;

    size_t degree           = 0;
    size_t int_refsteps     = 4;
    size_t degree_FEM       = 0;

    bool dump_debug         = false;
    bool solve_interface    = false;
    bool solve_fictdom      = false;
    bool agglomeration      = false;
    
    bool high_order = false ; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
    bool cut_off_active = false ; // IF FALSE IS SMOOTH, IF TRUE  CUT_OFF

    mesh_init_params<RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;
    RealType d = 0.5;
    size_t T_N = 0;
    /* k <deg>:     method degree
     * g<deg>:  method FEM degree
     * M <num>:     number of cells in x direction
     * N <num>:     number of cells in y direction
     * r <num>:     number of interface refinement steps
     *
     * i:           solve interface problem
     * f:           solve fictitious domain problem
     *
     * D:           use node displacement to solve bad cuts (default)
     * A:           use agglomeration to solve bad cuts
     *
     * d:           dump debug data
     */

    int ch;
    while ( (ch = getopt(argc, argv, "k:q:M:N:r:T:ifDAdhc")) != -1 )
    {
        switch(ch)
        {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'q':
                degree_FEM = atoi(optarg);
                break;
        
            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;
            
            case 'T':
                T_N = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;
            
            case 'h':
                high_order = true;
            break;
            
            case 'c':
                cut_off_active = true;
            break;
            

            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;

    
    
    timecounter tc;

    /************** BUILD MESH **************/
    Matrix<RealType, Dynamic, Dynamic> phi0 ;
    Matrix<RealType, Dynamic, 1> vel0 , vel1 ;
    Matrix<RealType, Dynamic, 1>  phi1 , phi2 , phi3;
    Matrix<RealType, Dynamic, 1> vecprova0 , vecprova1 , vecprova2 ;
    Matrix<RealType, Dynamic, 1> vecprovabis0 , vecprovabis1   , vecprovabis7  ,vecprovabis6 ;
    Matrix<RealType, Dynamic, Dynamic> vecprovabis2 ,vecprovabis3 ,vecprovabis4,vecprovabis5;
    tc.tic();
    cuthho_poly_mesh<RealType> msh(mip);
    typedef cuthho_poly_mesh<RealType> Mesh;
    offset_definition(msh);
    tc.toc();
    auto fe_data = Finite_Element<RealType,Mesh>( msh , degree_FEM , mip ) ;
    typedef Finite_Element<RealType,Mesh> FiniteSpace;
    
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;
    std::vector<point<RealType, 2>> node_interface0 , node_interface1   ;
    vecprovabis2 = Matrix<RealType, Dynamic, Dynamic>::Zero(fe_data.local_ndof , msh.cells.size() ) ;
    vecprovabis3 = Matrix<RealType, Dynamic, Dynamic>::Zero(fe_data.local_ndof , msh.cells.size() ) ;
    vecprovabis4 = Matrix<RealType, Dynamic, Dynamic>::Zero(fe_data.local_ndof , msh.cells.size() ) ;
    vecprovabis5 = Matrix<RealType, Dynamic, Dynamic>::Zero(fe_data.local_ndof , msh.cells.size() ) ;
    phi0 = Matrix<RealType, Dynamic, Dynamic>::Zero(fe_data.local_ndof , msh.cells.size() ) ;
    for( int i = 0 ; i < 2 ; i++)
    {
    
    
     /*
    size_t n_cells = msh.cells.size();
    std::cout<<" I m in parallel zone"<<std::endl;
    tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
    [&msh] (size_t & cell_ind){
            auto& cell = msh.cells[cell_ind];
            std::cout<<"CELL = "<<offset(msh,cell)<<std::endl;
             }
    );
    
   
#ifdef HAVE_INTEL_TBB
    size_t n_cells = msh.cells_size();
    std::cout<<" I m in parallel zone"<<std::endl;
    tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
    [&msh] (size_t & cell_ind){
            auto& cell = msh.cells[cell_ind];
             }
    );
#else
    std::cout<<" I m in sequential zone"<<std::endl;
    for (size_t cell_ind = 0; cell_ind < msh.cells.size(); cell_ind++)
    {
        auto& cell = msh.cells[cell_ind];
    }
#endif
    */
    
    /************** FINITE ELEMENT INITIALIZATION **************/
    
    /************** ANALYTIC LEVEL SET FUNCTION  **************/
    typedef RealType T;
    
    bool circle = false , ellipse = true ;
    
    RealType radius_a , radius_b , radius ;
    RealType x_centre = 0.5;
    RealType y_centre = 0.5;
    RealType C ;
    T h = std::max( fe_data.hx , fe_data.hy) ;
    if(circle)
    {
        radius = 1.0/9.0;
    }
        
    if(ellipse)
    {
        radius_a = 1.0/6.0;
        radius_b = 1.0/12.0;
        //radius_a = 1.0/9.0;
        //radius_b = 1.0/9.0;
        std::cout << bold << yellow << "Initial Analytic Area of the ELLIPSE: "<< M_PI*radius_a*radius_b << std::endl;
        radius = sqrt( radius_a * radius_b ) ;
        std::cout << bold << yellow << "Final radius expected of the circle : " << radius <<reset<<std::endl;
    }
  
        
    
    /// THIS DATA BELOW HAS TO BE UPLOAD DEPENDING ON THE PROBLEM.
    
    //auto level_set_function_anal = circle_level_set<RealType>(radius, x_centre, y_centre );
    //typedef  circle_level_set<T> Fonction;
    
    auto level_set_function_anal = elliptic_level_set<RealType>( radius_a, radius_b, x_centre, y_centre);
    typedef  elliptic_level_set<T> Fonction;
    
    //auto level_set_function_anal = elliptic_distance_ls<RealType>( radius_a, radius_b, x_centre, y_centre , h);
    //typedef  elliptic_distance_ls<T> Fonction;
    
    //auto level_set_function_anal = circle_distance_ls<RealType>(radius, x_centre, y_centre , 2*h );
    //typedef  circle_distance_ls<T> Fonction;
    
    
   
    /**************  VELOCITY FIELD  INITIALISATION  **************/
  
    //auto u_projected = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
    T u_0 = 1.00 ;
    T u_1 = 0.00 ;
    //auto u = linear_velocity_field<RealType>(0,u_0,0,u_1); // analytic velocity (0,0,0,0)
    //typedef linear_velocity_field<RealType> Velocity;
    //auto u = rotational_velocity_field<RealType>( x_centre , y_centre , 1.0);
    //typedef rotational_velocity_field<RealType> Velocity;
    auto u = taylor_green_vortex<RealType>( 1.0, false );
    typedef taylor_green_vortex<RealType> Velocity;
    auto u_projected = projection_velocity_high_order< Mesh,Velocity,FiniteSpace,T >(fe_data , u , msh);
    /************** LEVEL SET FUNCTION DISCRETISATION **************/
    std::cout<<"degree FEM "<<degree_FEM<<std::endl;
   
    
    auto level_set_function = L2projected_level_set_high_order< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);
    
   
    //level_set_function.iso_val_interface = 0.5 ;
    //level_set_function.coefficients_mapping_quadratic( );
    //level_set_function.coefficients_mapping_MAX_MAX( );
    if(i == 0)
        level_set_function.coefficients_sfasamento( );
    
   
    if(high_order)
        std::cout<<bold<<yellow<<"----> USING phi_HP HIGH order!!!!! "<<reset<<std::endl;
    else
        std::cout<<bold<<yellow<<"----> USING phi_L LOW order!!!!! "<<reset<<std::endl;
    
    
    testing_level_set(msh,level_set_function);
     
    // Initiliatisation data for time routine
    auto crr_mesh =  Current_Mesh<Mesh>(msh);
    
    // Initialisation area , mass
    T initial_area = 0. , initial_mass = 0.;
    T area_previous_time = 0. , mass_previous_time = 0. , dt = 0. ;
    
    /// DATA CHECK INITIALISATION
    T d_a = 0. ;
    T error_normal_global = 0. ;
    T error_normal_local = 0. ;
    T perimeter_initial = 0. ;
    T centre_mass_x_inital = 0. , centre_mass_y_inital = 0. ;
    
    
    
    T check = 10.0;
    T time_pos = 0.;
    
    T tot_time = 0.;
    
    bool bdry_bottom = false , bdry_up = false ;
    bool bdry_left = false , bdry_right = false ;
    
    check_inlet( msh , fe_data , bdry_bottom , bdry_right , bdry_up , bdry_left, 1e-14 );
    
    size_t time_step = 0; // add to prove velocity error
   // for (size_t time_step = 0; time_step<=T_N; time_step++)
   // {
       
        

        // ************** Re-Initialization mesh **************
        crr_mesh.current_mesh = msh;
        Mesh msh_i =  crr_mesh.current_mesh;
        offset_definition(msh_i);
        

        //************ DO cutHHO MESH PROCESSING **************
        tc.tic();
        detect_node_position3(msh_i, level_set_function); // In cuthho_geom
        detect_cut_faces3(msh_i, level_set_function); // In cuthho_geom
        std::cout<<"FINE DETECT NODES AND FACES."<<std::endl;
        if (agglomeration)
        {
            detect_cut_cells3(msh_i, level_set_function); // In cuthho_geom
            std::cout<<"FINE DETECT CELLS."<<std::endl;
            detect_cell_agglo_set(msh_i, level_set_function); // Non serve modificarla
            make_neighbors_info_cartesian(msh_i); // Non serve modificarla
            //refine_interface_angle(msh_i, level_set_function, int_refsteps);
            refine_interface_pro3(msh_i, level_set_function, int_refsteps);
            //refine_interface2(msh_i, level_set_function, int_refsteps); // IN cuthho_geom
            make_agglomeration(msh_i, level_set_function); // Non serve modificarla
        }
        else
        {
            //move_nodes(msh_i, level_set_function);
            //detect_cut_faces2(msh_i, level_set_function); //do it again to update intersection points
            detect_cut_cells3(msh_i, level_set_function);
            //refine_interface2(msh_i, level_set_function, int_refsteps);
            //refine_interface_angle(msh_i, level_set_function, int_refsteps);
            refine_interface_pro3(msh_i, level_set_function, int_refsteps);
        }
       
        tc.toc();
        std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

        
        for(auto& cl : msh_i.cells)
        {
            for(auto& pt : cl.user_data.interface)
            {
                if(i == 0)
                    node_interface0.push_back( pt );
                else
                    node_interface1.push_back( pt );
            }
                
        }
        std::cout<<"node_interface0 - node_interface1"<<'\n';
        for(size_t ciccia = 0 ; ciccia <  node_interface1.size() ; ciccia ++)
            std::cout<<node_interface0[ciccia] - node_interface1[ciccia]<<std::endl;
        
        
        
        if (dump_debug)
        {
            dump_mesh(msh_i);
            output_mesh_info(msh_i, level_set_function);
        }
   
        // IN cuthho_export..Points/Nodes don't change-> it's fast
        if(time_step == 0){
            output_mesh_info2_pre_FEM(msh_i, level_set_function); // IN cuthho_export
        }
        else
            output_mesh_info2(msh_i, level_set_function);
            //output_mesh_info2(msh_i, level_set_function);
        
        typedef L2projected_level_set_high_order< Mesh , Fonction , FiniteSpace , T > Level_Set;
        auto ls_cell = LS_cell_L2proj_high_order< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);
        
        
        //u_projected.set_agglo_mesh( msh_i );
        // CALCULATION OF AREA AND MASS AT TIME STEP t^n
        // CALCULATION ALSO OF CENTRE OF MASS
        
        
        /// DATA CHECK INITIALISATION
        T area0 = 0. , mass0 = 0. , global_mass0 = 0. ;
        T diff_area = 0. , diff_mass = 0. ;
        T error_normal_global0 = 0. ;
        T centre_mass0_x = 0. , centre_mass0_y = 0. ;
        T perimeter0 = 0.;
        T normal_interface_status = 0. ;
        size_t counter_interface_pts = 0;
        T divergence_error = 0.;
        // PLOTTING OF NORMAL
                 
        
        postprocess_output<double> postoutput_vec;
        auto vec_normal = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_Stokes_initial.dat");
        
        postprocess_output<double> postoutput_div2;
        auto test_interface_divergence2  = std::make_shared< gnuplot_output_object<double> >("divergence_interface_Stokes_initial.dat");
        
       
        std::vector<T> val_u_nx , val_u_ny , val_u_n ;
        std::vector< point<T, 2> > interface_points_plot ;
        std::vector< std::pair<T,T> > interface_normals ;
        
        for(auto& cl : msh_i.cells)
        {
            ls_cell.cell_assignment(cl);
            //u_projected.cell_assignment(cl);
            
            if( location(msh_i, cl) == element_location::IN_NEGATIVE_SIDE || location(msh_i, cl) == element_location::ON_INTERFACE )
            {
                T partial_area = measure( msh_i, cl, element_location::IN_NEGATIVE_SIDE);
                area0 += partial_area;
                auto qps = integrate( msh_i , cl , 2*degree_FEM+1 , element_location::IN_NEGATIVE_SIDE);
                for(auto& qp:qps){
                    mass0 += qp.second * ls_cell(qp.first);
                    centre_mass0_x += qp.second * qp.first.x() ;
                    centre_mass0_y += qp.second * qp.first.y() ;
                }
            }
            if(cl.user_data.location == element_location::ON_INTERFACE)
            {
                /*
                std::cout<<"CELL = "<<offset(msh,cl)<<std::endl;
                if(offset(msh,cl) == 101 || offset(msh,cl) == 104 )
                {
                    for(size_t kk = 0 ; kk < u_projected.sol_HHO.first.rows() ; kk++ )
                    std::cout<< u_projected.sol_HHO.first(kk,offset(msh,cl)) << " " << u_projected.sol_HHO.second(kk,offset(msh,cl)) << '\n';
                    std::cout<<'\n'<<std::endl;
                }
                */
                for(auto interface_point = cl.user_data.interface.begin() ; interface_point < cl.user_data.interface.end() -1 ; interface_point++ )
                {
                    
                    
                    T val = ls_cell.divergence( *interface_point );
                    divergence_error += pow((std::abs(val) - 1.0/radius),2) ;
                    Eigen::Matrix<T,2,1> normal = ls_cell.normal(*interface_point);
                    std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                    interface_normals.push_back( normal_vec ) ;
                    
                    if( time_step == 0 )
                    {
                        
                        vec_normal->add_data(*interface_point,normal_vec);
                        test_interface_divergence2->add_data(*interface_point , val);
                    }
                    
                    perimeter0 += ( *(interface_point+1) - *interface_point ).to_vector().norm();
                    
                    
                    
                    counter_interface_pts++;
                       
                }
                
                T val = ls_cell.divergence(*(cl.user_data.interface.end()-1));
                divergence_error += pow((std::abs(val) - 1.0/radius),2) ;
                Eigen::Matrix<T,2,1> normal = ls_cell.normal(*(cl.user_data.interface.end()-1));
                std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                interface_normals.push_back( normal_vec ) ;
                
                if( time_step == 0 )
                {
                    
                    vec_normal->add_data(*(cl.user_data.interface.end()-1) ,normal_vec);
                    test_interface_divergence2->add_data( *(cl.user_data.interface.end()-1) ,val );
                }
                                   
               
               
                
                counter_interface_pts++;
                   
               
            
            }
            
        }
        
        if( time_step == 0 )
        {
            postoutput_div2.add_object(test_interface_divergence2);
            postoutput_div2.write();
            
            postoutput_vec.add_object(vec_normal);
            postoutput_vec.write();
            
        }
        goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
                   
        testing_level_set_time(msh,level_set_function,tot_time);
        /*
        if( time_step == 0 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
            
        if( time_step == 5 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 10 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 15 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
            
        if( time_step == 20 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        if( time_step == 30 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 40 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        */
        
        divergence_error /= counter_interface_pts;
        divergence_error = sqrt(divergence_error);
        std::cout<<yellow<<bold<<"The l2 error of the DIVERGENCE at the INTERFACE, at time "<<reset<< dt*time_step <<" is " << divergence_error <<std::endl;
                   
        
        std::cout<<"number of interface points is " << counter_interface_pts << std::endl;
        normal_interface_status /= counter_interface_pts;
        normal_interface_status = sqrt(normal_interface_status);
        std::cout<<yellow<<bold<<"The l2 error of u*n over the INTERFACE, at time "<<reset<< tot_time <<" is " << normal_interface_status << std::endl;
        if(time_step == 0)
            check = 10.0;
        else
            check = normal_interface_status ;
        
        std::cout << "Area at time step: "<<tot_time<<" is "<< area0  << reset << std::endl;
        std::cout  << "Internal mass at time step: "<<tot_time<<" is "<<reset<< mass0   << std::endl;
        //std::cout<<bold<<yellow << "GLOBAL Mass at time step: "<<time_step<<" is "<<reset<<global_mass0<< reset << std::endl;
         std::cout << "CENTRE OF MASS at time step: " <<tot_time<<" is "<<" ( "<< centre_mass0_x/area0  << " , "<< centre_mass0_y/area0 <<" ). " << reset << std::endl;
        
         d_a = sqrt(4.0*area0/M_PI) ;
        
        std::cout<<"The PERIMETER, at time "<< tot_time <<" is " << perimeter0 <<std::endl;
        
        std::cout<<yellow<<bold<<"The CIRCULARITY, at time "<< tot_time <<" is "<<reset << M_PI*d_a/perimeter0 <<std::endl;
        
        
        
        if(time_step == 0 ){
            initial_area  = area0 ;
            initial_mass = mass0 ;
            centre_mass_x_inital = centre_mass0_x ;
            centre_mass_y_inital = centre_mass0_y ;
            perimeter_initial = perimeter0 ;
        }
        
        if(time_step > 0 )
        {
            diff_area = (area0 - initial_area)/initial_area ;
            diff_mass = (std::abs((mass0 - initial_mass)))/(std::abs(initial_mass)) ;
            std::cout << bold << yellow << "Normalised difference in Area (new - old)/old at time step: "<<tot_time<<" is "<<reset<< diff_area  << reset << std::endl;
            std::cout << bold << yellow << "Difference in internal MASS |new - old|/|old| at time step: "<<tot_time<<" is "<<reset<< diff_mass  << reset << std::endl;
            
            std::cout << "TRANSLATION OF THE CENTRE OF MASS at time step: "  <<tot_time<<" is "<<" ( " << centre_mass0_x/area0 - centre_mass_x_inital/initial_area <<" , " << centre_mass0_y/area0 - centre_mass_y_inital/initial_area<<" ). " << std::endl;
            std::cout<<"NORMALISED DIFFERENCE PERIMETER, at time " << tot_time <<" is " << (perimeter0 - perimeter_initial)/perimeter_initial <<std::endl;
                   
        }
        
        // auto test_case = make_test_case_stokes_1(msh, level_set_function);
        // auto test_case = make_test_case_stokes_2(msh, ls_cell); //level_set_function);
        
/// DA AGGIUNGERE UNA VOLTA SISTEMATO IL CODICE

        if(check < 1e-8 )
        {
            std::cout<<" check = "<<check<<" , STOP!"<<std::endl;
            return 0;
        }
 /*
        bool sym_grad = TRUE;
        
        auto prm = params<T>();
        prm.kappa_1 = 1.0;
        prm.kappa_2 = 1.0;
        
        //auto test_case = make_test_case_eshelby(msh_i, ls_cell,  prm , sym_grad);
       
        
        // Non serve modificare Gamma = 1/2
        auto test_case = make_test_case_eshelby_2(msh_i, ls_cell,  prm , sym_grad);
        
        auto method = make_sym_gradrec_stokes_interface_method(msh_i, 1.0, 0.0, test_case, sym_grad);
       
        
        //auto u_projected = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
       
    
        if(solve_interface){
            run_cuthho_interface_velocity(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad );
            
            //run_cuthho_interface(msh, degree, method, test_case);
        }
        if (solve_fictdom)
            run_cuthho_fictdom(msh_i, degree, test_case);
        

        
        
        // *********************** FEM -  PROCESSING ************************** //
        /// ORA HO SMOOTH OPERATOR! USE L2 PROJECTION
        if( 1 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: SMOOTH OPERATOR FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.smooth_converting_into_FE_formulation( u_projected.sol_HHO );
        }
        if( 0 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: OLD OPERATOR FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.converting_into_FE_formulation( u_projected.sol_HHO );
        }
        if( 0 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: L^2 PROJECTION FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.L2_proj_into_FE_formulation(level_set_function , msh);
        }
  */
        /*
        auto u_prova = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
        u_prova.sol_HHO = u_projected.sol_HHO ;
        u_prova.L2_proj_into_FE_formulation( level_set_function , msh );
        
        
        testing_velocity_field_L2projected(msh , u_prova) ;
        
        //std::cout<<"CHECK SMOOTH CONVERTING  FEM ----> FIRST"<<'\n'<<(u_prova.sol_FEM.first - u_projected.sol_FEM.first)<<'\n' <<std::endl;
        //std::cout<<"CHECK SMOOTH CONVERTING FEM ----> SECOND"<<'\n'<<(u_prova.sol_FEM.second - u_projected.sol_FEM.second)<<'\n' <<std::endl;
        
        testing_velocity_field(msh , u_projected) ;
        */
        //testing_velocity_field(msh , u_projected) ;
        //check_inlet( msh , fe_data ,  u_projected , 1e-14 );
        //std::cout<<"u_projected.sol_FEM.first = "<<'\n'<<u_projected.sol_FEM.first <<std::endl;
        //std::cout<<"u_projected.sol_FEM.second = "<<'\n'<<u_projected.sol_FEM.second <<std::endl;
        T eps = 0.48 ; // factor to be inside CFL stability zone
        //T dt1 = time_step_CFL( u , mip , eps ); // OLD IMPLEMENTATION
        dt = 0.1;
        //std::cout<<"MAX dt = "<<dt<<" AND HEURISTIC CFL IS "<<dt1<<std::endl;
        
        
        //if ( i == 1){
        //    std::cout<<"----> CHECK level_set_function.sol_FEM PRE TRANSPORT = "<<'\n'<<phi1 - level_set_function.sol_FEM <<std::endl;
        //}
        phi1 = level_set_function.sol_FEM ;
        
       
        T sub_time = 0.;
        T sub_dt = std::min(5e-3 , dt ) ;
        std::cout<<"Implemented dt = "<<dt<<std::endl;
        
        while( (sub_time < sub_dt*1) )
        {
            if(high_order){
                run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D_NEW_DIRICHLET_COND( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt );

                //run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
                //run_FEM_BERNSTEIN_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
            }
            else
                run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST_to_check_C_NEW( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt , phi2 , phi3 , i , vecprova0 , vecprova1 , vecprova2 , vecprovabis0 , vecprovabis1 , vecprovabis2 ,vecprovabis3 ,vecprovabis4, vecprovabis5, vecprovabis6 , vecprovabis7 );
                //run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST_to_check( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt , phi2 , phi3 , i , vecprova0 , vecprova1 , vecprova2 , vecprovabis0 , vecprovabis1 , vecprovabis2 ,vecprovabis3 ,vecprovabis4, vecprovabis5, vecprovabis6 , vecprovabis7 );
              
            sub_time += sub_dt ;
      
        }
        std::cout<<yellow<<bold<<"SUB TIME REPETITION OF TRANSPORT PB DATA:"<<reset<<" sub_dt = "<<sub_dt<< " , number time steps = "<<sub_time/sub_dt<<std::endl;
        
        tot_time += sub_time ;
        
        if ( i == 1){
            //Matrix<T, Dynamic, 1> one1 = Matrix<T, Dynamic, 1>::Ones(level_set_function.sol_FEM.size());
            //std::cout<<"----> CHECK level_set_function.sol_FEM = "<<'\n'<<0.5*one1 - (phi0 - level_set_function.sol_FEM) <<std::endl;
            for(auto& cl:msh.cells)
            {
                size_t cell_offset = offset(msh, cl) ;
                auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, fe_data.order);
                for (size_t i = 0; i < fe_data.local_ndof; i++)
                {
                    auto pt = pts[i];
                    if( 0.5 -std::abs(level_set_function(pt,msh,cl) - phi0(i,cell_offset)) > 1e-6)
                    std::cout<<"In pt = "<<pt<<" --> phi(pt) = "<<level_set_function(pt,msh,cl)<<" and ERROR = "<< 0.5 -std::abs(level_set_function(pt,msh,cl) - phi0(i,cell_offset))<<std::endl;
                }
                
            }
            
        }
        if( i == 0)
        {
        
            for(auto& cl:msh.cells)
            {
                size_t cell_offset = offset(msh, cl) ;
                auto pts = equidistriduted_nodes_ordered_bis<T,Mesh>( msh, cl, fe_data.order);
                for (size_t i = 0; i < fe_data.local_ndof; i++)
                {
                    auto pt = pts[i];
                    phi0(i,cell_offset) = level_set_function(pt,msh,cl) ;
                }
                
            }
       
       
        }
        
        if ( i == 0){
            testing_level_set2(msh,level_set_function);
        }
        else
            testing_level_set2_bis(msh,level_set_function);
    
    }
         exit(1);
   
   
    return 0;
}
#endif




// CHECK CURVATURE -> CONVERGENCE
#if 0
int main(int argc, char **argv)
{

        using RealType = double;

        size_t degree           = 0;
        size_t int_refsteps     = 4;
        size_t degree_FEM       = 0;

        bool dump_debug         = false;
        bool solve_interface    = false;
        bool solve_fictdom      = false;
        bool agglomeration      = false;
        
        bool high_order = false ; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
        bool cut_off_active = false ; // IF FALSE IS SMOOTH, IF TRUE  CUT_OFF

        mesh_init_params<RealType> mip;
        mip.Nx = 5;
        mip.Ny = 5;
        RealType d = 0.5;
        size_t T_N = 0;
        /* k <deg>:     method degree
         * g<deg>:  method FEM degree
         * M <num>:     number of cells in x direction
         * N <num>:     number of cells in y direction
         * r <num>:     number of interface refinement steps
         *
         * i:           solve interface problem
         * f:           solve fictitious domain problem
         *
         * D:           use node displacement to solve bad cuts (default)
         * A:           use agglomeration to solve bad cuts
         *
         * d:           dump debug data
         */

        int ch;
        while ( (ch = getopt(argc, argv, "k:q:M:N:r:T:ifDAdhc")) != -1 )
        {
            switch(ch)
            {
                case 'k':
                    degree = atoi(optarg);
                    break;

                case 'q':
                    degree_FEM = atoi(optarg);
                    break;
            
                case 'M':
                    mip.Nx = atoi(optarg);
                    break;

                case 'N':
                    mip.Ny = atoi(optarg);
                    break;

                case 'r':
                    int_refsteps = atoi(optarg);
                    break;
                
                case 'T':
                    T_N = atoi(optarg);
                    break;

                case 'i':
                    solve_interface = true;
                    break;

                case 'f':
                    solve_fictdom = true;
                    break;

                case 'D':
                    agglomeration = false;
                    break;

                case 'A':
                    agglomeration = true;
                    break;

                case 'd':
                    dump_debug = true;
                    break;
                
                case 'h':
                    high_order = true;
                break;
                
                case 'c':
                    cut_off_active = true;
                break;
                

                case '?':
                default:
                    std::cout << "wrong arguments" << std::endl;
                    exit(1);
            }
        }

        argc -= optind;
        argv += optind;

        
        
        timecounter tc;

        /************** BUILD MESH **************/
        tc.tic();
        cuthho_poly_mesh<RealType> msh(mip);
        typedef cuthho_poly_mesh<RealType> Mesh;
        offset_definition(msh);
        tc.toc();
        std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;
        
    
        
        /************** FINITE ELEMENT INITIALIZATION **************/
        auto fe_data = Finite_Element<RealType,Mesh>( msh , degree_FEM , mip ) ;
        typedef Finite_Element<RealType,Mesh> FiniteSpace;
        
        /************** ANALYTIC LEVEL SET FUNCTION  **************/
        typedef RealType T;
        
        bool circle = true , ellipse = false ;
        
        RealType radius_a , radius_b , radius ;
        RealType x_centre = 0.5;
        RealType y_centre = 0.5;
        RealType C ;
        T h = std::max( fe_data.hx , fe_data.hy) ;
        if(circle)
        {
            radius = 1.0/9.0;
        }
            
        if(ellipse)
        {
            radius_a = 1.0/6.0;
            radius_b = 1.0/12.0;
            //radius_a = 1.0/9.0;
            //radius_b = 1.0/9.0;
            std::cout << bold << yellow << "Initial Analytic Area of the ELLIPSE: "<< M_PI*radius_a*radius_b << std::endl;
            radius = sqrt( radius_a * radius_b ) ;
            std::cout << bold << yellow << "Final radius expected of the circle : " << radius <<reset<<std::endl;
        }
      
            
        
        /// THIS DATA BELOW HAS TO BE UPLOAD DEPENDING ON THE PROBLEM.
        
        auto level_set_function_anal = circle_level_set<RealType>(radius, x_centre, y_centre );
        typedef  circle_level_set<T> Fonction;
        
        //auto level_set_function_anal = elliptic_level_set<RealType>( radius_a, radius_b, x_centre, y_centre);
        //typedef  elliptic_level_set<T> Fonction;
        
        //auto level_set_function_anal = elliptic_distance_ls<RealType>( radius_a, radius_b, x_centre, y_centre , h);
        //typedef  elliptic_distance_ls<T> Fonction;
        
        //auto level_set_function_anal = circle_distance_ls<RealType>(radius, x_centre, y_centre , 2*h );
        //typedef  circle_distance_ls<T> Fonction;
        
        
       
      
        /************** LEVEL SET FUNCTION DISCRETISATION **************/
        std::cout<<"degree FEM "<<degree_FEM<<std::endl;
       
        
        auto level_set_function = L2projected_level_set_high_order< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);
        
        
        //level_set_function.iso_val_interface = 0.5 ;
        //level_set_function.coefficients_mapping_quadratic( );
        //level_set_function.coefficients_mapping_MAX_MAX( );
        //level_set_function.coefficients_sfasamento( );
        
       
        if(high_order)
            std::cout<<bold<<yellow<<"----> USING phi_HP HIGH order!!!!! "<<reset<<std::endl;
        else
            std::cout<<bold<<yellow<<"----> USING phi_L LOW order!!!!! "<<reset<<std::endl;
        
        
        testing_level_set(msh,level_set_function);
         
        // Initiliatisation data for time routine
        auto crr_mesh =  Current_Mesh<Mesh>(msh);
        
        // Initialisation area , mass
        T initial_area = 0. , initial_mass = 0.;
        T area_previous_time = 0. , mass_previous_time = 0. , dt = 0. ;
        
        /// DATA CHECK INITIALISATION
        T d_a = 0. ;
        T error_normal_global = 0. ;
        T error_normal_local = 0. ;
        T perimeter_initial = 0. ;
        T centre_mass_x_inital = 0. , centre_mass_y_inital = 0. ;
        
        
        
        T check = 10.0;
        T time_pos = 0.;
        
        T tot_time = 0.;
        
        bool bdry_bottom = false , bdry_up = false ;
        bool bdry_left = false , bdry_right = false ;
        
        check_inlet( msh , fe_data , bdry_bottom , bdry_right , bdry_up , bdry_left, 1e-14 );
        
        size_t time_step = 0 ;
        
           
            // Uploading continuous normal function
            level_set_function.normal_continuous_setting();
            level_set_function.gradient_continuous_setting();

            // ************** Re-Initialization mesh **************
            crr_mesh.current_mesh = msh;
            Mesh msh_i =  crr_mesh.current_mesh;
            offset_definition(msh_i);
            

            //************ DO cutHHO MESH PROCESSING **************
            tc.tic();
            detect_node_position3(msh_i, level_set_function); // In cuthho_geom
            detect_cut_faces3(msh_i, level_set_function); // In cuthho_geom
            std::cout<<"FINE DETECT NODES AND FACES."<<std::endl;
            if (agglomeration)
            {
                detect_cut_cells3(msh_i, level_set_function); // In cuthho_geom
                std::cout<<"FINE DETECT CELLS."<<std::endl;
                detect_cell_agglo_set(msh_i, level_set_function); // Non serve modificarla
                make_neighbors_info_cartesian(msh_i); // Non serve modificarla
                //refine_interface_angle(msh_i, level_set_function, int_refsteps);
                refine_interface_pro3(msh_i, level_set_function, int_refsteps);
                //refine_interface2(msh_i, level_set_function, int_refsteps); // IN cuthho_geom
                make_agglomeration(msh_i, level_set_function); // Non serve modificarla
            }
            else
            {
                //move_nodes(msh_i, level_set_function);
                //detect_cut_faces2(msh_i, level_set_function); //do it again to update intersection points
                detect_cut_cells3(msh_i, level_set_function);
                //refine_interface2(msh_i, level_set_function, int_refsteps);
                //refine_interface_angle(msh_i, level_set_function, int_refsteps);
                refine_interface_pro3(msh_i, level_set_function, int_refsteps);
            }
           
            tc.toc();
            std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

            if (dump_debug)
            {
                dump_mesh(msh_i);
                output_mesh_info(msh_i, level_set_function);
            }
       
            // IN cuthho_export..Points/Nodes don't change-> it's fast
            if(time_step == 0){
                output_mesh_info2_pre_FEM(msh_i, level_set_function); // IN cuthho_export
            }
            else
                output_mesh_info2(msh_i, level_set_function);
                //output_mesh_info2(msh_i, level_set_function);
            
            
           
            typedef L2projected_level_set_high_order< Mesh , Fonction , FiniteSpace , T > Level_Set;
            auto ls_cell = LS_cell_L2proj_high_order< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);
            
            
            //u_projected.set_agglo_mesh( msh_i );
            // CALCULATION OF AREA AND MASS AT TIME STEP t^n
            // CALCULATION ALSO OF CENTRE OF MASS
            
            
            /// DATA CHECK INITIALISATION
            T area0 = 0. , mass0 = 0. , global_mass0 = 0. ;
            T diff_area = 0. , diff_mass = 0. ;
            T error_normal_global0 = 0. ;
            T centre_mass0_x = 0. , centre_mass0_y = 0. ;
            T perimeter0 = 0.;
            T normal_interface_status = 0. ;
            size_t counter_interface_pts = 0;
            T divergence_error_disc_old = 0. , divergence_error_disc_new = 0. , divergence_error_cont = 0. , divergence_error_cont_grad = 0.;
            // PLOTTING OF NORMAL
                     
            
            postprocess_output<double> postoutput_vec;
            auto vec_normal = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_Stokes_initial.dat");
            
            auto vec_normal_cont = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_Continuos_Stokes.dat");
            
            postprocess_output<double> postoutput_div2;
            auto test_interface_divergence2  = std::make_shared<
            gnuplot_output_object<double> >("divergence_interface_Stokes_initial.dat");
            
           
            std::vector<T> val_u_nx , val_u_ny , val_u_n ;
            std::vector< point<T, 2> > interface_points_plot ;
            std::vector< std::pair<T,T> > interface_normals , interface_normals_cont ;
            
            
            
            for(auto& cl : msh_i.cells)
            {
                ls_cell.cell_assignment(cl);
                //u_projected.cell_assignment(cl);
                
                
                if(cl.user_data.location == element_location::ON_INTERFACE)
                {
                    /*
                    std::cout<<"CELL = "<<offset(msh,cl)<<std::endl;
                    if(offset(msh,cl) == 101 || offset(msh,cl) == 104 )
                    {
                        for(size_t kk = 0 ; kk < u_projected.sol_HHO.first.rows() ; kk++ )
                        std::cout<< u_projected.sol_HHO.first(kk,offset(msh,cl)) << " " << u_projected.sol_HHO.second(kk,offset(msh,cl)) << '\n';
                        std::cout<<'\n'<<std::endl;
                    }
                    */
                    for(auto interface_point = cl.user_data.interface.begin() ; interface_point < cl.user_data.interface.end() -1 ; interface_point++ )
                    {
                        
                        
                        T val_disc_old = ls_cell.divergence( *interface_point );
                        //T val_disc_new = ls_cell.divergence_disc_new( *interface_point );
                        T val_cont = ls_cell.divergence_cont( *interface_point );
                        T val_cont_grad = ls_cell.divergence_cont_grad( *interface_point );
                        
                        divergence_error_disc_old += pow((std::abs(val_disc_old) - 1.0/radius),2) ;
                        //divergence_error_disc_new += pow((std::abs(val_disc_new) - 1.0/radius),2) ;
                        divergence_error_cont += pow((std::abs(val_cont) - 1.0/radius),2) ;
                        divergence_error_cont_grad+= pow((std::abs(val_cont_grad) - 1.0/radius),2) ;
                        //Eigen::Matrix<T,2,1> normal = ls_cell.normal(*interface_point);
                        //std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                        //interface_normals.push_back( normal_vec ) ;
                        
                        /// COSE  PER NORMALE CONTINUA
                        /*
                        Eigen::Matrix<T,2,1> normal_cont = ls_cell.normal_continuous(*interface_point);
                        //std::cout<<"normal = "<<'\n'<<normal<<" , CONTIUOUS_normal = "<<'\n'<<normal_cont<<std::endl;
                        std::pair<T,T> normal_vec_cont = std::make_pair(normal_cont(0),normal_cont(1));
                        interface_normals_cont.push_back( normal_vec_cont ) ;
                        vec_normal_cont->add_data(*interface_point,normal_vec_cont);
                        */
                        if( time_step == 0 )
                        {
                            
                            //vec_normal->add_data(*interface_point,normal_vec);
                            test_interface_divergence2->add_data(*interface_point , val_disc_old);
                        }
                        
                        //perimeter0 += ( *(interface_point+1) - *interface_point ).to_vector().norm();
                        
                        //normal_interface_status += pow( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) , 2) + pow( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) , 2 );
                        
                        //interface_points_plot.push_back(*(interface_point)) ;
                        //val_u_nx.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) );
                        //val_u_ny.push_back( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                        //val_u_n.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) + u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                        //std::cout<<"*(interface_point) = "<<*(interface_point)<<std::endl;
                        //std::cout<<" u_projected primo = "<<u_projected(*(interface_point)).first  << " u_projected second = "<< u_projected(*(interface_point)).second  << " somma tot con olds = "<<normal_interface_status<<std::endl;
                        //std::cout<<" ls_cell.normal(*(interface_point))(0)  = "<<ls_cell.normal(*(interface_point))(0)  << " ls_cell.normal(*(interface_point))(1)  = "<< ls_cell.normal(*(interface_point))(1)  << std::endl;
                        std::cout<<"val_disc_old = "<<val_disc_old<<std::endl;
                        //std::cout<<"val_disc_new = "<<val_disc_new<<std::endl;
                        std::cout<<"val_cont = "<<val_cont<<std::endl;
                        std::cout<<"val_cont_grad = "<<val_cont_grad<<std::endl;
                        counter_interface_pts++;
                           
                    }
                    
                    T val_disc_old = ls_cell.divergence( *(cl.user_data.interface.end()-1) );
                    //T val_disc_new = ls_cell.divergence_disc_new( *(cl.user_data.interface.end()-1) );
                    T val_cont = ls_cell.divergence_cont( *(cl.user_data.interface.end()-1) );
                    T val_cont_grad = ls_cell.divergence_cont_grad( *(cl.user_data.interface.end()-1) );
                    divergence_error_disc_old += pow((std::abs(val_disc_old) - 1.0/radius),2) ;
                    //divergence_error_disc_new += pow((std::abs(val_disc_new) - 1.0/radius),2) ;
                    divergence_error_cont += pow((std::abs(val_cont) - 1.0/radius),2) ;
                    divergence_error_cont_grad+= pow((std::abs(val_cont_grad) - 1.0/radius),2) ;
                    std::cout<<"val_disc_old = "<<val_disc_old<<std::endl;
                    //std::cout<<"val_disc_new = "<<val_disc_new<<std::endl;
                    std::cout<<"val_cont = "<<val_cont<<std::endl;
                    std::cout<<"val_cont_grad = "<<val_cont_grad<<std::endl;
                    //Eigen::Matrix<T,2,1> normal = ls_cell.normal(*(cl.user_data.interface.end()-1));
                    //std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                    //interface_normals.push_back( normal_vec ) ;
                    
                    /// COSE  PER NORMALE CONTINUA
                    /*
                    Eigen::Matrix<T,2,1> normal_cont = ls_cell.normal_continuous(*(cl.user_data.interface.end()-1));
                    std::pair<T,T> normal_vec_cont = std::make_pair(normal_cont(0),normal_cont(1));
                    interface_normals_cont.push_back( normal_vec_cont ) ;
                   
                     vec_normal_cont->add_data(*(cl.user_data.interface.end()-1) ,normal_vec_cont);
                    */
                    /*
                    if( time_step == 0 )
                    {
                        
                        vec_normal->add_data(*(cl.user_data.interface.end()-1) ,normal_vec);
                        test_interface_divergence2->add_data( *(cl.user_data.interface.end()-1) ,val );
                    }
                                       
                   
                    normal_interface_status += pow( u_projected (*(cl.user_data.interface.end()-1) ).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0),2) + pow( u_projected(*( cl.user_data.interface.end()-1) ).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1), 2);
                    
                    
                interface_points_plot.push_back(*(cl.user_data.interface.end()-1)) ;
                    val_u_nx.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) );
                    val_u_ny.push_back( u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                    val_u_n.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) + u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                    
                    */
                    counter_interface_pts++;
                       
                   
                
                }
                
            }
            
            if( time_step == 0 )
            {
                postoutput_div2.add_object(test_interface_divergence2);
                postoutput_div2.write();
                
                postoutput_vec.add_object(vec_normal);
                postoutput_vec.add_object(vec_normal_cont);
                postoutput_vec.write();
                
            }
            //goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
                       
            //testing_level_set_time(msh,level_set_function,tot_time);
           
            divergence_error_disc_old /= counter_interface_pts;
            divergence_error_disc_old = sqrt(divergence_error_disc_old);
    
   // divergence_error_disc_new /= counter_interface_pts;
   // divergence_error_disc_new = sqrt(divergence_error_disc_new);
    
    divergence_error_cont /= counter_interface_pts;
    divergence_error_cont = sqrt(divergence_error_cont);
    
    divergence_error_cont_grad/= counter_interface_pts;
    divergence_error_cont_grad = sqrt(divergence_error_cont_grad);
    
            std::cout<<yellow<<bold<<"The l2 error of the DIVERGENCE DISC OLD at the INTERFACE, at time "<<reset<< dt*time_step <<" is " << divergence_error_disc_old <<std::endl;
    
    // std::cout<<yellow<<bold<<"The l2 error of the DIVERGENCE DISC NEW at the INTERFACE, at time "<<reset<< dt*time_step <<" is " << divergence_error_disc_new <<std::endl;
    
     std::cout<<yellow<<bold<<"The l2 error of the DIVERGENCE CONTINUOUS at the INTERFACE, at time "<<reset<< dt*time_step <<" is " << divergence_error_cont <<std::endl;
            
    std::cout<<yellow<<bold<<"The l2 error of the DIVERGENCE CONTINUOUS (VIA GRADIENT) at the INTERFACE, at time "<<reset<< dt*time_step <<" is " << divergence_error_cont_grad <<std::endl;
    
            
        
            
            
            
            
      
        
     
        return 0;
    
}
#endif






// Interface Stokes Problem: INLET DIRICHLET BDRY CONDITIONS
// Generic Interface: Gamma = 0 or 1/2 or generic ( SEE MAX-MAX mapping)
// LAST UPDATE 28/07/2020
#if 0
int main(int argc, char **argv)
{
    using RealType = double;

    size_t degree           = 0;
    size_t int_refsteps     = 4;
    size_t degree_FEM       = 0;

    bool dump_debug         = false;
    bool solve_interface    = false;
    bool solve_fictdom      = false;
    bool agglomeration      = false;
    
    bool high_order = false ; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
    bool cut_off_active = false ; // IF FALSE IS SMOOTH, IF TRUE  CUT_OFF

    mesh_init_params<RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;
    //RealType d = 0.5;
    size_t T_N = 0;
    /* k <deg>:     method degree
     * g<deg>:  method FEM degree
     * M <num>:     number of cells in x direction
     * N <num>:     number of cells in y direction
     * r <num>:     number of interface refinement steps
     *
     * i:           solve interface problem
     * f:           solve fictitious domain problem
     *
     * D:           use node displacement to solve bad cuts (default)
     * A:           use agglomeration to solve bad cuts
     *
     * d:           dump debug data
     */

    int ch;
    while ( (ch = getopt(argc, argv, "k:q:M:N:r:T:ifDAdhc")) != -1 )
    {
        switch(ch)
        {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'q':
                degree_FEM = atoi(optarg);
                break;
        
            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;
            
            case 'T':
                T_N = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;
            
            case 'h':
                high_order = true;
            break;
            
            case 'c':
                cut_off_active = true;
            break;
            

            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;

    
    
    timecounter tc;

    /************** BUILD MESH **************/
    
    cuthho_poly_mesh<RealType> msh(mip);
    typedef cuthho_poly_mesh<RealType> Mesh;
    offset_definition(msh);
    
  
    
    /************** FINITE ELEMENT INITIALIZATION **************/
    auto fe_data = Finite_Element<RealType,Mesh>( msh , degree_FEM , mip ) ;
    typedef Finite_Element<RealType,Mesh> FiniteSpace;
    
    /************** ANALYTIC LEVEL SET FUNCTION  **************/
    typedef RealType T;
    
    bool circle = false , ellipse = true ;
    
    RealType radius_a , radius_b , radius ;
    RealType x_centre = 0.5;
    RealType y_centre = 0.5;
    //RealType C ;
    //T h = std::max( fe_data.hx , fe_data.hy) ;
    if(circle)
    {
        radius = 1.0/9.0;
    }
        
    if(ellipse)
    {
        radius_a = 1.0/6.0;
        radius_b = 1.0/12.0;
        //radius_a = 1.0/9.0;
        //radius_b = 1.0/9.0;
        std::cout << bold << yellow << "Initial Analytic Area of the ELLIPSE: "<< M_PI*radius_a*radius_b << std::endl;
        radius = sqrt( radius_a * radius_b ) ;
        std::cout << bold << yellow << "Final radius expected of the circle : " << radius <<reset<<std::endl;
    }
  
        
    
    ///---->  THIS DATA BELOW HAS TO BE UPLOAD DEPENDING ON THE PROBLEM.
    //auto level_set_function_anal = circle_level_set<RealType>(radius, x_centre, y_centre );
    //typedef  circle_level_set<T> Fonction;
    
    auto level_set_function_anal = elliptic_level_set<RealType>( radius_a, radius_b, x_centre, y_centre);
    typedef  elliptic_level_set<T> Fonction;
    
    //auto level_set_function_anal = elliptic_distance_ls<RealType>( radius_a, radius_b, x_centre, y_centre , h);
    //typedef  elliptic_distance_ls<T> Fonction;
    
    //auto level_set_function_anal = circle_distance_ls<RealType>(radius, x_centre, y_centre , 2*h );
    //typedef  circle_distance_ls<T> Fonction;
    
    
    timecounter tc_agglo;
    
    /**************  VELOCITY FIELD  INITIALISATION  **************/
    
    auto u_projected = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
  
    /************** LEVEL SET FUNCTION DISCRETISATION **************/
    std::cout<<"degree FEM "<<degree_FEM<<std::endl;
    
    auto level_set_function = L2projected_level_set_high_order< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);
    
    /************** LEVEL SET  MAPPING **************/
    bool mapping = false ;
    if(mapping)
        level_set_function.coefficients_mapping_MAX_MAX( );
    
    //level_set_function.iso_val_interface = 0.5 ;
    //level_set_function.coefficients_mapping_quadratic( );
    //level_set_function.coefficients_mapping_MAX_MAX( );
    //level_set_function.coefficients_sfasamento( );
    
   
    if(high_order)
        std::cout<<bold<<yellow<<"----> USING phi_HP HIGH order!!!!! "<<reset<<std::endl;
    else
        std::cout<<bold<<yellow<<"----> USING phi_L LOW order!!!!! "<<reset<<std::endl;
    
    tc_agglo.tic();
    testing_level_set(msh,level_set_function);
    tc_agglo.toc();
    std::cout << bold << yellow << "testing_level_set: time = " << tc_agglo << " seconds" << reset << std::endl;
    
    // Initiliatisation data for time routine
    auto crr_mesh =  Current_Mesh<Mesh>(msh);
    
    // Initialisation area , mass
    T initial_area = 0. , initial_mass = 0.;
    T  dt = 0. ; // area_previous_time = 0. , mass_previous_time = 0. ,
    
    /// DATA CHECK INITIALISATION
    T d_a = 0. ;
    //T error_normal_global = 0. ;
    //T error_normal_local = 0. ;
    T perimeter_initial = 0. ;
    T centre_mass_x_inital = 0. , centre_mass_y_inital = 0. ;
    
    
    
    T check = 10.0;
    T time_pos = 0.;
    T tot_time = 0.;
    
    bool bdry_bottom = false , bdry_up = false ;
    bool bdry_left = false , bdry_right = false ;
  
    check_inlet( msh , fe_data , bdry_bottom , bdry_right , bdry_up , bdry_left, 1e-14 );
   
    for (size_t time_step = 0; time_step<=T_N; time_step++)
    {
        tc_agglo.tic();
        // Uploading continuous normal function
        level_set_function.normal_continuous_setting() ;
        level_set_function.gradient_continuous_setting() ;
        tc_agglo.toc();
        std::cout << bold << yellow << "normal_continuous_setting: " << tc_agglo << " seconds" << reset << std::endl;
        
        // ************** Re-Initialization mesh **************
        crr_mesh.current_mesh = msh;
        Mesh msh_i =  crr_mesh.current_mesh;
        offset_definition(msh_i);
        
       
        

        //************ DO cutHHO MESH PROCESSING **************
        tc.tic();
        // The sequential detect_node_position3 is fastern than the parallel one: DIMOSTRATO.
        //detect_node_position3_parallel(msh_i, level_set_function); // In cuthho_geom
        detect_node_position3(msh_i, level_set_function); // In cuthho_geom
        
        detect_cut_faces3(msh_i, level_set_function); // In cuthho_geom
        //std::cout<<"FINE DETECT NODES AND FACES."<<std::endl;
        if (agglomeration)
        {
            detect_cut_cells3(msh_i, level_set_function); // In cuthho_geom
            //detect_cut_cells3_parallelized(msh_i, level_set_function); // In cuthho_geom
            std::cout<<"----> Fine of detect_cut_cells3."<<std::endl;

            detect_cell_agglo_set(msh_i, level_set_function); // Non serve modificarla
            make_neighbors_info_cartesian(msh_i); // Non serve modificarla
            //refine_interface_angle(msh_i, level_set_function, int_refsteps);
            refine_interface_pro3(msh_i, level_set_function, int_refsteps);
            make_agglomeration(msh_i, level_set_function); // Non serve modificarla
          
        }
        else
        {
            //move_nodes(msh_i, level_set_function);
            //detect_cut_faces2(msh_i, level_set_function); //do it again to update intersection points
            detect_cut_cells3(msh_i, level_set_function);
            //refine_interface2(msh_i, level_set_function, int_refsteps);
            //refine_interface_angle(msh_i, level_set_function, int_refsteps);
            refine_interface_pro3(msh_i, level_set_function, int_refsteps);
        }
       
        tc.toc();
        std::cout << bold << yellow <<'\n' <<"-----> TIME -----> cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset <<'\n' << std::endl;

        if (dump_debug)
        {
            dump_mesh(msh_i);
            output_mesh_info(msh_i, level_set_function);
        }
   
    
        // IN cuthho_export..Points/Nodes don't change-> it's fast
        if(time_step == 0){
            output_mesh_info2_pre_FEM(msh_i, level_set_function); // IN cuthho_export
        }
        else
            output_mesh_info2(msh_i, level_set_function);
          
  
       
        typedef L2projected_level_set_high_order< Mesh , Fonction , FiniteSpace , T > Level_Set;
        
       
        auto ls_cell = LS_cell_L2proj_high_order< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);
        
        u_projected.set_agglo_mesh( msh_i );
        // CALCULATION OF AREA AND MASS AT TIME STEP t^n
        // CALCULATION ALSO OF CENTRE OF MASS
        
        
        /// DATA CHECK INITIALISATION
        T area0 = 0. , mass0 = 0.; // , global_mass0 = 0. ;
        T diff_area = 0. , diff_mass = 0. ;
        //T error_normal_global0 = 0. ;
        T centre_mass0_x = 0. , centre_mass0_y = 0. ;
        T perimeter0 = 0.;
        T normal_interface_status = 0. ;
        size_t counter_interface_pts = 0;
       
        
        // PLOTTING OF NORMAL
        postprocess_output<double> postoutput_vec;
        auto vec_normal = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_Stokes_initial.dat");
        
        auto vec_normal_n_cont = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_continuos_n_Stokes.dat");
        
        auto vec_normal_grad_cont = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_continuos_grad_Stokes.dat");
        
        postprocess_output<double> postoutput_div2;
        
        
        auto test_interface_divergence0  = std::make_shared< gnuplot_output_object<double> >("k0_divergence_interface_Stokes_initial.dat");
        auto test_interface_divergence1  = std::make_shared< gnuplot_output_object<double> >("k1_divergence_interface_Stokes_initial.dat");
        auto test_interface_divergence2  = std::make_shared< gnuplot_output_object<double> >("k2_divergence_interface_Stokes_initial.dat");
       
        std::vector<T> val_u_nx , val_u_ny , val_u_n ;
        std::vector< point<T, 2> > interface_points_plot ;
        std::vector< std::pair<T,T> > interface_normals , interface_normals_n_cont , interface_normals_grad_cont ;
         T divergence_error0 = 0. , divergence_error1 = 0. , divergence_error2 = 0.;
        
        for(auto& cl : msh_i.cells)
        {
            ls_cell.cell_assignment(cl);
            u_projected.cell_assignment(cl);
            
            if( location(msh_i, cl) == element_location::IN_NEGATIVE_SIDE || location(msh_i, cl) == element_location::ON_INTERFACE )
            {
                T partial_area = measure( msh_i, cl, element_location::IN_NEGATIVE_SIDE);
                area0 += partial_area;
                auto qps = integrate( msh_i , cl , 2*degree_FEM+1 , element_location::IN_NEGATIVE_SIDE);
                for(auto& qp:qps){
                    mass0 += qp.second * ls_cell(qp.first);
                    centre_mass0_x += qp.second * qp.first.x() ;
                    centre_mass0_y += qp.second * qp.first.y() ;
                }
            }
            if(cl.user_data.location == element_location::ON_INTERFACE)
            {
                /*
                std::cout<<"CELL = "<<offset(msh,cl)<<std::endl;
                if(offset(msh,cl) == 101 || offset(msh,cl) == 104 )
                {
                    for(size_t kk = 0 ; kk < u_projected.sol_HHO.first.rows() ; kk++ )
                    std::cout<< u_projected.sol_HHO.first(kk,offset(msh,cl)) << " " << u_projected.sol_HHO.second(kk,offset(msh,cl)) << '\n';
                    std::cout<<'\n'<<std::endl;
                }
                */
                for(auto interface_point = cl.user_data.interface.begin() ; interface_point < cl.user_data.interface.end() -1 ; interface_point++ )
                {
                    
                    
                    T val0 = ls_cell.divergence( *interface_point );
                    divergence_error0 += pow((std::abs(val0) - 1.0/radius),2) ;
                    
                    T val1 = ls_cell.divergence_cont( *interface_point );
                    divergence_error1 += pow((std::abs(val1) - 1.0/radius),2) ;
                    
                    T val2 = ls_cell.divergence_cont_grad( *interface_point );
                    divergence_error2 += pow((std::abs(val2) - 1.0/radius),2) ;
                    
                    Eigen::Matrix<T,2,1> normal = ls_cell.normal(*interface_point);
                    std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                    interface_normals.push_back( normal_vec ) ;
                  
                    Eigen::Matrix<T,2,1> normal_cont = ls_cell.normal_cont(*interface_point);
                    //std::cout<<"normal = "<<'\n'<<normal<<" , CONTIUOUS_normal = "<<'\n'<<normal_cont<<std::endl;
                    std::pair<T,T> normal_vec_cont = std::make_pair(normal_cont(0),normal_cont(1));
                    interface_normals_n_cont.push_back( normal_vec_cont ) ;
                    
                    Eigen::Matrix<T,2,1> normal_cont_grad = ls_cell.normal_cont_normalised(*interface_point);
                    //std::cout<<"normal = "<<'\n'<<normal<<" , CONTIUOUS_normal = "<<'\n'<<normal_cont<<std::endl;
                    std::pair<T,T> normal_vec_cont_grad = std::make_pair(normal_cont_grad(0),normal_cont_grad(1));
                    interface_normals_grad_cont.push_back( normal_vec_cont_grad ) ;
                    
                    
                    
                    if( time_step == 0 )
                    {
                        
                        vec_normal->add_data(*interface_point,normal_vec);
                        test_interface_divergence0->add_data(*interface_point , val0);
                        test_interface_divergence1->add_data(*interface_point , val1);
                        test_interface_divergence2->add_data(*interface_point , val2);
                        vec_normal_n_cont->add_data(*interface_point,normal_vec_cont);
                        vec_normal_grad_cont->add_data(*interface_point,normal_vec_cont_grad);
                    }
                    
                    perimeter0 += ( *(interface_point+1) - *interface_point ).to_vector().norm();
                    
                    normal_interface_status += pow( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) , 2) + pow( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) , 2 );
                    
                    interface_points_plot.push_back(*(interface_point)) ;
                    val_u_nx.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) );
                    val_u_ny.push_back( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                    val_u_n.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) + u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                    //std::cout<<"*(interface_point) = "<<*(interface_point)<<std::endl;
                    //std::cout<<" u_projected primo = "<<u_projected(*(interface_point)).first  << " u_projected second = "<< u_projected(*(interface_point)).second  << " somma tot con olds = "<<normal_interface_status<<std::endl;
                    //std::cout<<" ls_cell.normal(*(interface_point))(0)  = "<<ls_cell.normal(*(interface_point))(0)  << " ls_cell.normal(*(interface_point))(1)  = "<< ls_cell.normal(*(interface_point))(1)  << std::endl;
                    
                    counter_interface_pts++;
                       
                }
                
                T val0 = ls_cell.divergence(*(cl.user_data.interface.end()-1));
                T val1 = ls_cell.divergence_cont(*(cl.user_data.interface.end()-1));
                T val2 = ls_cell.divergence_cont_grad(*(cl.user_data.interface.end()-1));
                divergence_error0 += pow((std::abs(val0) - 1.0/radius),2) ;
                divergence_error1 += pow((std::abs(val1) - 1.0/radius),2) ;
                divergence_error2 += pow((std::abs(val2) - 1.0/radius),2) ;
                
                Eigen::Matrix<T,2,1> normal = ls_cell.normal(*(cl.user_data.interface.end()-1));
                std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                interface_normals.push_back( normal_vec ) ;
                
                Eigen::Matrix<T,2,1> normal_cont_n = ls_cell.normal_cont(*(cl.user_data.interface.end()-1));
                std::pair<T,T> normal_vec_n_cont = std::make_pair(normal_cont_n(0),normal_cont_n(1));
                interface_normals_n_cont.push_back( normal_vec_n_cont ) ;
                
                Eigen::Matrix<T,2,1> normal_cont_grad = ls_cell.normal_cont_normalised(*(cl.user_data.interface.end()-1));
                std::pair<T,T> normal_vec_grad_cont = std::make_pair(normal_cont_grad(0),normal_cont_grad(1));
                interface_normals_grad_cont.push_back( normal_vec_grad_cont ) ;
                
                /// COSE  PER NORMALE CONTINUA
                /*
                Eigen::Matrix<T,2,1> normal_cont = ls_cell.normal_continuous(*(cl.user_data.interface.end()-1));
                std::pair<T,T> normal_vec_cont = std::make_pair(normal_cont(0),normal_cont(1));
                interface_normals_cont.push_back( normal_vec_cont ) ;
               
                 vec_normal_cont->add_data(*(cl.user_data.interface.end()-1) ,normal_vec_cont);
                */
                
                if( time_step == 0 )
                {
                    
                    vec_normal->add_data(*(cl.user_data.interface.end()-1) ,normal_vec);
                    test_interface_divergence0->add_data( *(cl.user_data.interface.end()-1) ,val0 );
                    test_interface_divergence1->add_data( *(cl.user_data.interface.end()-1) ,val1 );
                    test_interface_divergence2->add_data( *(cl.user_data.interface.end()-1) ,val2 );
                    vec_normal_n_cont->add_data(*(cl.user_data.interface.end()-1), normal_vec_n_cont);
                    vec_normal_grad_cont->add_data(*(cl.user_data.interface.end()-1), normal_vec_grad_cont);
                }
                                   
               
                normal_interface_status += pow( u_projected (*(cl.user_data.interface.end()-1) ).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0),2) + pow( u_projected(*( cl.user_data.interface.end()-1) ).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1), 2);
                
                
           
                interface_points_plot.push_back(*(cl.user_data.interface.end()-1)) ;
                val_u_nx.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) );
                val_u_ny.push_back( u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                val_u_n.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) + u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                
                
                counter_interface_pts++;
                   
               
            
            }
            
        }
        
        if( time_step == 0 )
        {
            postoutput_div2.add_object(test_interface_divergence0);
            postoutput_div2.add_object(test_interface_divergence1);
            postoutput_div2.add_object(test_interface_divergence2);
            postoutput_div2.write();
            
            postoutput_vec.add_object(vec_normal);
            postoutput_vec.add_object(vec_normal_n_cont);
            postoutput_vec.add_object(vec_normal_grad_cont);
            postoutput_vec.write();
            
        }
        
        std::cout<<"NOTICE:--> The terms u \cdot n are made with n = normal (NO CONTINUOUS)."<<std::endl;
        
        goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
        testing_level_set_time(msh,level_set_function,tot_time);
        /*
        if( time_step == 0 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
            
        if( time_step == 5 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 10 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 15 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
            
        if( time_step == 20 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        if( time_step == 30 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 40 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        */
        
        divergence_error0 /= counter_interface_pts;
        divergence_error0 = sqrt(divergence_error0);
        
        divergence_error1 /= counter_interface_pts;
        divergence_error1 = sqrt(divergence_error1);
        
        divergence_error2 /= counter_interface_pts;
        divergence_error2 = sqrt(divergence_error2);
        std::cout<<yellow<<bold<<"The l2 error of the CURVATURE (OLD) at the INTERFACE, at time "<<reset<< dt*time_step <<" is " << divergence_error0 <<std::endl;
        std::cout<<yellow<<bold<<"The l2 error of the CURVATURE (n^c continuous) at the INTERFACE, at time "<<reset<< dt*time_step <<" is " << divergence_error1 <<std::endl;
        std::cout<<yellow<<bold<<"The l2 error of the CURVATURE ((\nabla(phi))^c continuous) at the INTERFACE, at time "<<reset<< dt*time_step <<" is " << divergence_error2 <<std::endl;
                   
        
        std::cout<<"number of interface points is " << counter_interface_pts << std::endl;
        normal_interface_status /= counter_interface_pts;
        normal_interface_status = sqrt(normal_interface_status);
        std::cout<<yellow<<bold<<"The l2 error of u*n over the INTERFACE, at time "<<reset<< tot_time <<" is " << normal_interface_status << std::endl;
        if(time_step == 0)
            check = 10.0;
        else
            check = normal_interface_status ;
        
        std::cout << "Area at time step: "<<tot_time<<" is "<< area0  << reset << std::endl;
        std::cout  << "Internal mass at time step: "<<tot_time<<" is "<<reset<< mass0   << std::endl;
        //std::cout<<bold<<yellow << "GLOBAL Mass at time step: "<<time_step<<" is "<<reset<<global_mass0<< reset << std::endl;
         std::cout << "CENTRE OF MASS at time step: " <<tot_time<<" is "<<" ( "<< centre_mass0_x/area0  << " , "<< centre_mass0_y/area0 <<" ). " << reset << std::endl;
        
         d_a = sqrt(4.0*area0/M_PI) ;
        
        std::cout<<"The PERIMETER, at time "<< tot_time <<" is " << perimeter0 <<std::endl;
        
        std::cout<<yellow<<bold<<"The CIRCULARITY, at time "<< tot_time <<" is "<<reset << M_PI*d_a/perimeter0 <<std::endl;
        
        
        
        if(time_step == 0 ){
            initial_area  = area0 ;
            initial_mass = mass0 ;
            centre_mass_x_inital = centre_mass0_x ;
            centre_mass_y_inital = centre_mass0_y ;
            perimeter_initial = perimeter0 ;
        }
        
        if(time_step > 0 )
        {
            diff_area = (area0 - initial_area)/initial_area ;
            diff_mass = (std::abs((mass0 - initial_mass)))/(std::abs(initial_mass)) ;
            std::cout << bold << yellow << "Normalised difference in Area (new - old)/old at time step: "<<tot_time<<" is "<<reset<< diff_area  << reset << std::endl;
            std::cout << bold << yellow << "Difference in internal MASS |new - old|/|old| at time step: "<<tot_time<<" is "<<reset<< diff_mass  << reset << std::endl;
            
            std::cout << "TRANSLATION OF THE CENTRE OF MASS at time step: "  <<tot_time<<" is "<<" ( " << centre_mass0_x/area0 - centre_mass_x_inital/initial_area <<" , " << centre_mass0_y/area0 - centre_mass_y_inital/initial_area<<" ). " << std::endl;
            std::cout<<"NORMALISED DIFFERENCE PERIMETER, at time " << tot_time <<" is " << (perimeter0 - perimeter_initial)/perimeter_initial <<std::endl;
                   
        }
        
        
        tc_agglo.toc();
        std::cout << bold << yellow << "------> TIME FOR CHECKING GOAL QUANTITIES: " << tc_agglo << " seconds" << reset << std::endl;
        // auto test_case = make_test_case_stokes_1(msh, level_set_function);
        // auto test_case = make_test_case_stokes_2(msh, ls_cell); //level_set_function);
        
/// DA AGGIUNGERE UNA VOLTA SISTEMATO IL CODICE
        /*
        if(check < 1e-8 )
        {
            std::cout<<" check = "<<check<<" , STOP!"<<std::endl;
            return 0;
        }
        */
        bool sym_grad = TRUE;
        
        auto prm = params<T>();
        prm.kappa_1 = 1.0;
        prm.kappa_2 = 1.0;
        
        //auto test_case = make_test_case_eshelby(msh_i, ls_cell,  prm , sym_grad);
     
        
        // Non serve modificare Gamma = 1/2
        //auto test_case = make_test_case_eshelby_2(msh_i, ls_cell,  prm , sym_grad );
        //auto test_case = make_test_case_eshelby_analytic(msh_i, ls_cell,  prm , sym_grad , radius);
     
        auto test_case_prova = make_test_case_eshelby_2_prova(msh_i, ls_cell,  prm , sym_grad );
       
        //auto method = make_sym_gradrec_stokes_interface_method(msh_i, 1.0, 0.0, test_case, sym_grad);
        
        auto method = make_sym_gradrec_stokes_interface_method(msh_i, 1.0, 0.0, test_case_prova, sym_grad);
      
        //auto u_projected = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
       
        tc_agglo.tic();
    
        if(solve_interface){
            //run_cuthho_interface_velocity_parallel(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad );
            run_cuthho_interface_velocity_prova(msh_i, degree, method,test_case_prova, ls_cell , u_projected ,sym_grad );
            //run_cuthho_interface_velocity(msh_i, degree, method, test_case, ls_cell , u_projected ,sym_grad );
            
            // OLD
            //run_cuthho_interface_velocity_analytic(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad ,radius );
            
            //run_cuthho_interface(msh, degree, method, test_case);
        }
        if (solve_fictdom)
            run_cuthho_fictdom(msh_i, degree, test_case_prova);
        
        tc_agglo.toc();
        std::cout << bold << yellow << "TIME-----> run_cuthho_interface_velocity: " << tc_agglo << " seconds" << reset << std::endl;
        
        
        /*********************** FEM -  PROCESSING **************************/
        /// ORA HO SMOOTH OPERATOR! USE L2 PROJECTION
        
        if( 1 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: SMOOTH OPERATOR FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.smooth_converting_into_FE_formulation( u_projected.sol_HHO );
        }
        if( 0 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: OLD OPERATOR FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.converting_into_FE_formulation( u_projected.sol_HHO );
        }
        if( 0 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: L^2 PROJECTION FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.L2_proj_into_FE_formulation(level_set_function , msh);
        }
        
        
        /*
        auto u_prova = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
        u_prova.sol_HHO = u_projected.sol_HHO ;
        u_prova.L2_proj_into_FE_formulation( level_set_function , msh );
        
        
        testing_velocity_field_L2projected(msh , u_prova) ;
        
        //std::cout<<"CHECK SMOOTH CONVERTING  FEM ----> FIRST"<<'\n'<<(u_prova.sol_FEM.first - u_projected.sol_FEM.first)<<'\n' <<std::endl;
        //std::cout<<"CHECK SMOOTH CONVERTING FEM ----> SECOND"<<'\n'<<(u_prova.sol_FEM.second - u_projected.sol_FEM.second)<<'\n' <<std::endl;
        
        testing_velocity_field(msh , u_projected) ;
        */
        
        testing_velocity_field(msh , u_projected) ;
        
        //check_inlet( msh , fe_data ,  u_projected , 1e-14 );
        //std::cout<<"u_projected.sol_FEM.first = "<<'\n'<<u_projected.sol_FEM.first <<std::endl;
        //std::cout<<"u_projected.sol_FEM.second = "<<'\n'<<u_projected.sol_FEM.second <<std::endl;
        T eps = 0.48 ; // factor to be inside CFL stability zone
        //T dt1 = time_step_CFL( u , mip , eps ); // OLD IMPLEMENTATION
        T dt1 = time_step_CFL_new( u_projected , mip , eps );
        //std::cout<<"dt1 is "<<dt1<<std::endl;
        T dt2 = 0.1;
        dt = std::min(dt1 , dt2);
        std::cout<<"MAX dt = "<<dt<<" AND HEURISTIC CFL IS "<<dt1<<std::endl;
        
        /*
        if(high_order)
            run_FEM_BERNSTEIN_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
            //run_FEM_BERNSTEIN_CORRECT( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
        else
            run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
            //run_FEM_BERNSTEIN_LOW_ORDER_CORRECT( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
        */
        
        // I can create a sub-time. I solve several time the FEM problem, given a Stokes field. The amount of time is s.t. at maximum there is a displacement of a cell of the interface and no more than a maximum T
        T sub_time = 0.;
        T sub_dt = std::min(4e-4 , dt ) ;
        std::cout<<"Implemented dt = "<<dt<<std::endl;
        while( (sub_time < sub_dt*10) && (sub_time < dt1) )
        {
            if(high_order){
                run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D_NEW_DIRICHLET_COND( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt , mapping );

                //run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
                //run_FEM_BERNSTEIN_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
            }
            else
                run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
               
            sub_time += sub_dt ;
      
        }
        std::cout<<yellow<<bold<<"SUB TIME REPETITION OF TRANSPORT PB DATA:"<<reset<<" sub_dt = "<<sub_dt<< " , number time steps = "<<sub_time/sub_dt<<std::endl;
        
        tot_time += sub_time ;
        
        testing_level_set2(msh,level_set_function);
        
        /// OLD IMPLEMENTATION
        //for(size_t j=1; j<4 ; j++)
        //run_FEM_levelset( level_set_function.msh,degree_FEM,level_set_function,u,dt,mip);
       
        /************** FEM -  POST-PROCESSING **************/
        if( (T_N - time_step)==0 )
        {
            // Uploading continuous normal function
            level_set_function.normal_continuous_setting();
            level_set_function.gradient_continuous_setting() ;
            // Uploading mesh data to check out differences in mass and areas
            crr_mesh.current_mesh = msh;
            Mesh msh_i2 =  crr_mesh.current_mesh;
            offset_definition(msh_i2);
            tc.tic();
            detect_node_position3(msh_i2, level_set_function); // In cuthho_geom
            detect_cut_faces3(msh_i2, level_set_function); // In cuthho_geom
        
       
            if (agglomeration)
            {
                // std::cout<<"i m here 1"<<std::endl;
                detect_cut_cells3(msh_i2, level_set_function); // In cuthho_geom
           
                detect_cell_agglo_set(msh_i2, level_set_function); // Non serve modificarla
           
                make_neighbors_info_cartesian(msh_i2); // Non serve modificarla
                //refine_interface_angle(msh_i2, level_set_function, int_refsteps); // IN cuthho_geom
                refine_interface_pro3(msh_i2, level_set_function, int_refsteps); // IN cuthho_geom
                make_agglomeration(msh_i2, level_set_function); // Non serve modificarla
            }
            else
            {
                //move_nodes(msh_i2, level_set_function);
                //detect_cut_faces2(msh_i2, level_set_function); //do it again to update intersection     points
                detect_cut_cells3(msh_i2, level_set_function);
                //refine_interface2(msh_i2, level_set_function, int_refsteps);
                //refine_interface_angle(msh_i2, level_set_function, int_refsteps);
                refine_interface_pro3(msh_i2, level_set_function, int_refsteps);
            }
        
            tc.toc();
            std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

            if (dump_debug)
            {
                dump_mesh(msh_i2);
                output_mesh_info(msh_i2, level_set_function);
            }
        
            output_mesh_info2(msh_i2, level_set_function);
            // IN cuthho_export..Points/Nodes don't change
           
            // Uploading level set
            ls_cell.level_set = level_set_function;
            ls_cell.agglo_msh = msh_i2;
            u_projected.set_agglo_mesh( msh_i2 );
            
           
            T mass_fin = 0. , area_fin = 0. ;
            T centre_mass_x = 0. , centre_mass_y = 0. ;
        
            T perimeter = 0. ;
            normal_interface_status = 0. ;
            counter_interface_pts = 0;
   
            
            postprocess_output<double> postoutput_vec;
            auto vec_normal_fin = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_Stokes_final.dat");
            auto vec_normal_n_cont_fin = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_continuos_n_Stokes_final.dat");
             
            auto vec_normal_grad_cont_fin = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_continuos_grad_Stokes_final.dat");
            
            postprocess_output<double> postoutput_div2;
            auto test_interface_divergence_fin0  = std::make_shared< gnuplot_output_object<double> >("k0_fin_divergence_interface_Stokes_final.dat");
            auto test_interface_divergence_fin1  = std::make_shared< gnuplot_output_object<double> >("k1_fin_divergence_interface_Stokes_final.dat");
            auto test_interface_divergence_fin2  = std::make_shared< gnuplot_output_object<double> >("k2_fin_divergence_interface_Stokes_final.dat");
            
            std::vector<T> val_u_nx_fin , val_u_ny_fin , val_u_n_fin ;
            std::vector< point<T, 2> > interface_points_plot_fin ;
            std::vector< std::pair<T,T> > interface_normals_fin ,interface_normals_n_cont_fin ,interface_normals_grad_cont_fin ;
            
            T divergence_error_fin0 = 0. , divergence_error_fin1 = 0. , divergence_error_fin2 = 0.;
            
            
            
            for(auto& cl : msh_i2.cells)
            {
                ls_cell.cell_assignment(cl);
                u_projected.cell_assignment(cl);
                
                if( (location(msh_i2, cl) == element_location::IN_NEGATIVE_SIDE) || (location(msh_i2, cl) == element_location::ON_INTERFACE) )
                {
                    
                    T partial_area = measure( msh_i2, cl, element_location::IN_NEGATIVE_SIDE);
                 
                    area_fin += partial_area;
                    
                   
                    auto qps_fin = integrate( msh_i2 , cl , 2*degree_FEM+1 , element_location::IN_NEGATIVE_SIDE);
                   
                    for(auto& qp:qps_fin){
                        mass_fin += qp.second * ls_cell(qp.first);
                        centre_mass_x += qp.second * qp.first.x() ;
                        centre_mass_y += qp.second * qp.first.y() ;
                    }
                   
                }
                if(cl.user_data.location == element_location::ON_INTERFACE)
                {
                    
                    for(auto interface_point = cl.user_data.interface.begin() ; interface_point < cl.user_data.interface.end() -1 ; interface_point++ )
                    {
                        perimeter += ( *(interface_point+1) - *interface_point ).to_vector().norm();
                        
                        normal_interface_status += pow( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) , 2) + pow( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) , 2 );
                        
                        
                        T val0 = ls_cell.divergence( *interface_point );
                        divergence_error_fin0 += pow((std::abs(val0) - 1.0/radius),2) ;
                        T val1 = ls_cell.divergence_cont( *interface_point );
                        divergence_error_fin1 += pow((std::abs(val1) - 1.0/radius),2) ;
                        T val2 = ls_cell.divergence_cont_grad( *interface_point );
                        divergence_error_fin2 += pow((std::abs(val2) - 1.0/radius),2) ;
                        
                        Eigen::Matrix<T,2,1> normal = ls_cell.normal(*interface_point);
                        std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                        interface_normals_fin.push_back( normal_vec ) ;
                        
                        Eigen::Matrix<T,2,1> normal_cont_n = ls_cell.normal_cont(*interface_point);
                        std::pair<T,T> normal_vec_n_cont = std::make_pair(normal_cont_n(0),normal_cont_n(1));
                        interface_normals_n_cont_fin.push_back( normal_vec_n_cont ) ;
                        
                        Eigen::Matrix<T,2,1> normal_cont_grad = ls_cell.normal_cont_normalised(*interface_point);
                        std::pair<T,T> normal_vec_grad_cont = std::make_pair(normal_cont_grad(0),normal_cont_grad(1));
                        interface_normals_grad_cont_fin.push_back( normal_vec_grad_cont ) ;
                        
                        vec_normal_fin->add_data(*interface_point,normal_vec);
                        vec_normal_n_cont_fin->add_data(*interface_point,normal_vec_n_cont);
                        vec_normal_grad_cont_fin->add_data(*interface_point,normal_vec_grad_cont);
                        test_interface_divergence_fin0->add_data( *interface_point , val0);
                        test_interface_divergence_fin1->add_data( *interface_point , val1);
                        test_interface_divergence_fin2->add_data( *interface_point , val2);
                        
                        interface_points_plot_fin.push_back( *(interface_point) ) ;
                        val_u_nx_fin.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) );
                        val_u_ny_fin.push_back( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                        val_u_n_fin.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) + u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                        
                        counter_interface_pts++;
                    }
                    
                    T val0 = ls_cell.divergence( *(cl.user_data.interface.end()-1) );
                    divergence_error_fin0 += pow((std::abs(val0) - 1.0/radius),2) ;
                    
                    T val1 = ls_cell.divergence_cont( *(cl.user_data.interface.end()-1) );
                    divergence_error_fin1 += pow((std::abs(val1) - 1.0/radius),2) ;
                    
                    T val2 = ls_cell.divergence_cont_grad( *(cl.user_data.interface.end()-1) );
                    divergence_error_fin2 += pow((std::abs(val2) - 1.0/radius),2) ;
                    
                    Eigen::Matrix<T,2,1> normal = ls_cell.normal(*(cl.user_data.interface.end()-1));
                    std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                    interface_normals_fin.push_back( normal_vec ) ;
                    
                    Eigen::Matrix<T,2,1> normal_n_cont = ls_cell.normal_cont(*(cl.user_data.interface.end()-1));
                    std::pair<T,T> normal_vec_n_cont = std::make_pair(normal_n_cont(0),normal_n_cont(1));
                    interface_normals_n_cont_fin.push_back( normal_vec_n_cont ) ;
                    
                    Eigen::Matrix<T,2,1> normal_grad_cont = ls_cell.normal_cont_normalised(*(cl.user_data.interface.end()-1));
                    std::pair<T,T> normal_vec_grad_norm = std::make_pair(normal_grad_cont(0),normal_grad_cont(1));
                    interface_normals_grad_cont_fin.push_back( normal_vec_grad_norm ) ;
                    
                    vec_normal_fin->add_data( *(cl.user_data.interface.end()-1) ,normal_vec);
                    vec_normal_n_cont_fin->add_data( *(cl.user_data.interface.end()-1) ,normal_vec_n_cont);
                    vec_normal_grad_cont_fin->add_data( *(cl.user_data.interface.end()-1) ,normal_vec_grad_norm);
                    test_interface_divergence_fin0->add_data( *(cl.user_data.interface.end()-1) ,val0 );
                    test_interface_divergence_fin1->add_data( *(cl.user_data.interface.end()-1) ,val1 );
                    test_interface_divergence_fin2->add_data( *(cl.user_data.interface.end()-1) ,val2 );
                    
                    normal_interface_status += pow( u_projected (*(cl.user_data.interface.end()-1) ).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0),2) + pow( u_projected(*( cl.user_data.interface.end()-1) ).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1), 2);
                    
                    interface_points_plot_fin.push_back( *(cl.user_data.interface.end()-1) ) ;
                    val_u_nx_fin.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) );
                    val_u_ny_fin.push_back( u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                    val_u_n_fin.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) + u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                                
                    counter_interface_pts++;

                }
                
            }
            
            postoutput_div2.add_object(test_interface_divergence_fin0);
            postoutput_div2.add_object(test_interface_divergence_fin1);
            postoutput_div2.add_object(test_interface_divergence_fin2);
            postoutput_div2.write();
                       
            postoutput_vec.add_object(vec_normal_fin);
            postoutput_vec.add_object(vec_normal_n_cont_fin);
            postoutput_vec.add_object(vec_normal_grad_cont_fin);
            postoutput_vec.write();
            
            goal_quantities_time(msh , tot_time, interface_points_plot_fin , val_u_nx_fin , val_u_ny_fin , val_u_n_fin , interface_normals_fin ) ;
            testing_level_set_time(msh,level_set_function, tot_time);
            /*
            if( time_step == 9 ){
                goal_quantities_time(msh , tot_time, interface_points_plot_fin , val_u_nx_fin , val_u_ny_fin , val_u_n_fin , interface_normals_fin ) ;
                testing_level_set_time(msh,level_set_function, tot_time);
            }
            */
            divergence_error_fin0 /= counter_interface_pts;
            divergence_error_fin0 = sqrt(divergence_error_fin0);
            
            divergence_error_fin1 /= counter_interface_pts;
            divergence_error_fin1 = sqrt(divergence_error_fin1);
            
            divergence_error_fin2 /= counter_interface_pts;
            divergence_error_fin2 = sqrt(divergence_error_fin2);
            
            std::cout<<yellow<<bold<<"The l2 error of the CURVATURE at the INTERFACE, at time "<<reset<< tot_time <<" is " << divergence_error_fin0 <<std::endl;
            
            std::cout<<yellow<<bold<<"The l2 error of the CURVATURE at the INTERFACE, at time "<<reset<< tot_time <<" is " << divergence_error_fin1 <<std::endl;
            
            std::cout<<yellow<<bold<<"The l2 error of the CURVATURE at the INTERFACE, at time "<<reset<< tot_time <<" is " << divergence_error_fin2 <<std::endl;
            
            std::cout<<"number of interface points is " << counter_interface_pts << std::endl;
            
            normal_interface_status /= counter_interface_pts;
            normal_interface_status = sqrt(normal_interface_status);
            
            std::cout<<yellow<<bold<<"The l2 error of u*n over the INTERFACE, at time "<<reset<< tot_time <<" is " << normal_interface_status << std::endl;
            
            
            
            std::cout<<"The PERIMETER, at time "<< tot_time <<" is " << perimeter <<std::endl;
            
            std::cout<<"perimeter = "<< perimeter << " AND  perimeter0 =  "<<perimeter_initial<<std::endl;
            std::cout<< bold << yellow<<"NORMALISED DIFFERENCE PERIMETER, at time "<<reset<< tot_time <<" is " << (perimeter - perimeter_initial)/perimeter_initial <<std::endl;
            
            d_a = sqrt(4.0*area_fin/M_PI) ;
            
            std::cout<< bold << yellow<<"The CIRCULARITY, at time "<< tot_time<<reset <<" is " << M_PI*d_a/perimeter <<std::endl;
            
            std::cout  << "Area at time step: " <<tot_time<<" is "<< area_fin << std::endl;
            std::cout << "Internal mass at time step: "<<tot_time<<" is "<< mass_fin << reset << std::endl;
            
            std::cout << bold << yellow << "NORMALISED Difference in AREA AT TIME "<<tot_time<<" IS "<< reset<< (area_fin - initial_area)/initial_area << std::endl;
            std::cout << bold << yellow << "NORMALISED Difference in INTERNAL MASS AT TIME "<<tot_time<<" IS "<< reset<< (std::abs(mass_fin - initial_mass))/(std::abs( initial_mass )) << std::endl;
            std::cout << "CENTRE OF MASS at time step: "<<tot_time<<" is "<<" ( " << centre_mass_x/area_fin <<" , " << centre_mass_y/area_fin<<" ). " << std::endl;
            std::cout << "TRANSLATION OF THE CENTRE OF MASS at time step: "  <<tot_time<<" is "<<" ( " << centre_mass_x/area_fin - centre_mass_x_inital/initial_area <<" , " << centre_mass_y/area_fin - centre_mass_y_inital/initial_area<<" ). " << std::endl;
            std::cout  << "Abs error over expected radius = "<< std::abs( sqrt(area_fin/M_PI) - radius ) << std::endl;
            
            
           
        } // END OF T = FINAL TIME
       
        time_pos +=dt ;
        //time_step++;
    } // End of the temporal loop
    
    std::cout<< bold << yellow <<"FINAL TIME IS t = "<< reset<<tot_time<<std::endl;
    return 0;
}
#endif



// Interface Stokes Problem: INLET DIRICHLET BDRY CONDITIONS
// Generic Interface: Gamma = 0 or 1/2 or generic ( SEE MAX-MAX mapping)
// LAST UPDATE 28/07/2020 -> GRAD CONTINUOS IMPLEMENTATION
#if 1
int main(int argc, char **argv)
{
    using RealType = double;
    std::cout<<"STEFANO FILE IS NOT UPDATED, SMOOTH FOR ANALYTICS LS (ALSO CONSTRUCTOR WITH phi_fem)!"
    exit(9);
    size_t degree           = 0;
    size_t int_refsteps     = 4;
    size_t degree_FEM       = 0;

    bool dump_debug         = false;
    bool solve_interface    = false;
    bool solve_fictdom      = false;
    bool agglomeration      = false;
    
    bool high_order = false ; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
    bool cut_off_active = false ; // IF FALSE IS SMOOTH, IF TRUE  CUT_OFF

    mesh_init_params<RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;
    //RealType d = 0.5;
    size_t T_N = 0;
    /* k <deg>:     method degree
     * g<deg>:  method FEM degree
     * M <num>:     number of cells in x direction
     * N <num>:     number of cells in y direction
     * r <num>:     number of interface refinement steps
     *
     * i:           solve interface problem
     * f:           solve fictitious domain problem
     *
     * D:           use node displacement to solve bad cuts (default)
     * A:           use agglomeration to solve bad cuts
     *
     * d:           dump debug data
     */

    int ch;
    while ( (ch = getopt(argc, argv, "k:q:M:N:r:T:ifDAdhc")) != -1 )
    {
        switch(ch)
        {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'q':
                degree_FEM = atoi(optarg);
                break;
        
            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;
            
            case 'T':
                T_N = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;
            
            case 'h':
                high_order = true;
            break;
            
            case 'c':
                cut_off_active = true;
            break;
            

            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;

    
    
    timecounter tc;

    /************** BUILD MESH **************/
    
    cuthho_poly_mesh<RealType> msh(mip);
    typedef cuthho_poly_mesh<RealType> Mesh;
    offset_definition(msh);
    
  
    std::cout << ansi::foreground_red << "PROVA in red" << ansi::reset << std::endl;
    
    /************** FINITE ELEMENT INITIALIZATION **************/
    auto fe_data = Finite_Element<RealType,Mesh>( msh , degree_FEM , mip ) ;
    typedef Finite_Element<RealType,Mesh> FiniteSpace;
    
    /************** ANALYTIC LEVEL SET FUNCTION  **************/
    typedef RealType T;
    
    bool circle = false , ellipse = true ;
    
    RealType radius_a , radius_b , radius ;
    RealType x_centre = 0.5;
    RealType y_centre = 0.5;
    //RealType C ;
    //T h = std::max( fe_data.hx , fe_data.hy) ;
    if(circle)
    {
        radius = 1.0/9.0;
    }
        
    if(ellipse)
    {
        radius_a = 1.0/6.0;
        radius_b = 1.0/12.0;
        //radius_a = 1.0/9.0;
        //radius_b = 1.0/9.0;
        std::cout << bold << yellow << "Initial Analytic Area of the ELLIPSE: "<< M_PI*radius_a*radius_b << std::endl;
        radius = sqrt( radius_a * radius_b ) ;
        std::cout << bold << yellow << "Final radius expected of the circle : " << radius <<reset<<std::endl;
    }
  
        
    
    ///---->  THIS DATA BELOW HAS TO BE UPLOAD DEPENDING ON THE PROBLEM.
    //auto level_set_function_anal = circle_level_set<RealType>(radius, x_centre, y_centre );
    //typedef  circle_level_set<T> Fonction;
    
    auto level_set_function_anal = elliptic_level_set<RealType>( radius_a, radius_b, x_centre, y_centre);
    typedef  elliptic_level_set<T> Fonction;
    
    //auto level_set_function_anal = elliptic_distance_ls<RealType>( radius_a, radius_b, x_centre, y_centre , h);
    //typedef  elliptic_distance_ls<T> Fonction;
    
    //auto level_set_function_anal = circle_distance_ls<RealType>(radius, x_centre, y_centre , 2*h );
    //typedef  circle_distance_ls<T> Fonction;
    
    
    timecounter tc_agglo;
    
    /**************  VELOCITY FIELD  INITIALISATION  **************/
    
    auto u_projected = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
  
    /************** LEVEL SET FUNCTION DISCRETISATION **************/
    std::cout<<"degree FEM "<<degree_FEM<<std::endl;
    
    //auto level_set_function = L2projected_level_set_high_order< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);
    auto level_set_function = L2projected_level_set_high_order_grad_cont< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);
    
    /************** LEVEL SET  MAPPING **************/
    bool mapping = false ;
    if(mapping)
        level_set_function.coefficients_mapping_MAX_MAX( );
    
    //level_set_function.iso_val_interface = 0.5 ;
    //level_set_function.coefficients_mapping_quadratic( );
    //level_set_function.coefficients_mapping_MAX_MAX( );
    //level_set_function.coefficients_sfasamento( );
    
   
    if(high_order)
        std::cout<<bold<<yellow<<"----> USING phi_HP HIGH order!!!!! "<<reset<<std::endl;
    else
        std::cout<<bold<<yellow<<"----> USING phi_L LOW order!!!!! "<<reset<<std::endl;
    
    tc_agglo.tic();
    testing_level_set(msh,level_set_function);
    tc_agglo.toc();
    std::cout << bold << yellow << "testing_level_set: time = " << tc_agglo << " seconds" << reset << std::endl;
    
    // Initiliatisation data for time routine
    auto crr_mesh =  Current_Mesh<Mesh>(msh);
    
    // Initialisation area , mass
    T initial_area = 0. , initial_mass = 0.;
    T  dt = 0. ; // area_previous_time = 0. , mass_previous_time = 0. ,
    
    /// DATA CHECK INITIALISATION
    T d_a = 0. ;
    //T error_normal_global = 0. ;
    //T error_normal_local = 0. ;
    T perimeter_initial = 0. ;
    T centre_mass_x_inital = 0. , centre_mass_y_inital = 0. ;
    
    
    
    T check = 10.0;
    T time_pos = 0.;
    T tot_time = 0.;
    
    bool bdry_bottom = false , bdry_up = false ;
    bool bdry_left = false , bdry_right = false ;
  
    check_inlet( msh , fe_data , bdry_bottom , bdry_right , bdry_up , bdry_left, 1e-14 );
   
    for (size_t time_step = 0; time_step<=T_N; time_step++)
    {
        tc_agglo.tic();
        // Uploading continuous normal function
        level_set_function.normal_continuous_setting() ;
        level_set_function.gradient_continuous_setting() ;
        tc_agglo.toc();
        std::cout << bold << yellow << "normal_continuous_setting: " << tc_agglo << " seconds" << reset << std::endl;
        
        // ************** Re-Initialization mesh **************
        crr_mesh.current_mesh = msh;
        Mesh msh_i =  crr_mesh.current_mesh;
        offset_definition(msh_i);
        
       
        

        //************ DO cutHHO MESH PROCESSING **************
        tc.tic();
        // The sequential detect_node_position3 is fastern than the parallel one: DIMOSTRATO.
        //detect_node_position3_parallel(msh_i, level_set_function); // In cuthho_geom
        detect_node_position3(msh_i, level_set_function); // In cuthho_geom
        
        detect_cut_faces3(msh_i, level_set_function); // In cuthho_geom
        //std::cout<<"FINE DETECT NODES AND FACES."<<std::endl;
        if (agglomeration)
        {
            detect_cut_cells3(msh_i, level_set_function); // In cuthho_geom
            //detect_cut_cells3_parallelized(msh_i, level_set_function); // In cuthho_geom
            std::cout<<"----> Fine of detect_cut_cells3."<<std::endl;

            detect_cell_agglo_set(msh_i, level_set_function); // Non serve modificarla
            make_neighbors_info_cartesian(msh_i); // Non serve modificarla
            //refine_interface_angle(msh_i, level_set_function, int_refsteps);
            refine_interface_pro3(msh_i, level_set_function, int_refsteps);
            make_agglomeration(msh_i, level_set_function); // Non serve modificarla
          
        }
        else
        {
            //move_nodes(msh_i, level_set_function);
            //detect_cut_faces2(msh_i, level_set_function); //do it again to update intersection points
            detect_cut_cells3(msh_i, level_set_function);
            //refine_interface2(msh_i, level_set_function, int_refsteps);
            //refine_interface_angle(msh_i, level_set_function, int_refsteps);
            refine_interface_pro3(msh_i, level_set_function, int_refsteps);
        }
       
        tc.toc();
        std::cout << bold << yellow <<'\n' <<"-----> TIME -----> cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset <<'\n' << std::endl;

        if (dump_debug)
        {
            dump_mesh(msh_i);
            output_mesh_info(msh_i, level_set_function);
        }
   
    
        // IN cuthho_export..Points/Nodes don't change-> it's fast
        if(time_step == 0){
            output_mesh_info2_pre_FEM(msh_i, level_set_function); // IN cuthho_export
        }
        else
            output_mesh_info2(msh_i, level_set_function);
          
  
       
        //typedef L2projected_level_set_high_order< Mesh , Fonction , FiniteSpace , T > Level_Set;
        typedef L2projected_level_set_high_order_grad_cont< Mesh , Fonction , FiniteSpace , T > Level_Set;
       
        auto ls_cell = LS_cell_L2proj_high_order_grad_cont< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);
        
        u_projected.set_agglo_mesh( msh_i );
        // CALCULATION OF AREA AND MASS AT TIME STEP t^n
        // CALCULATION ALSO OF CENTRE OF MASS
        
        
        /// DATA CHECK INITIALISATION
        T area0 = 0. , mass0 = 0.; // , global_mass0 = 0. ;
        T diff_area = 0. , diff_mass = 0. ;
        //T error_normal_global0 = 0. ;
        T centre_mass0_x = 0. , centre_mass0_y = 0. ;
        T perimeter0 = 0.;
        T normal_interface_status = 0. ;
        size_t counter_interface_pts = 0;
       
        
        // PLOTTING OF NORMAL
        postprocess_output<double> postoutput_vec;
        auto vec_normal = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_Stokes_initial.dat");
        
        auto vec_normal_n_cont = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_continuos_n_Stokes.dat");
        
        auto vec_normal_grad_cont = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_continuos_grad_Stokes.dat");
        
        postprocess_output<double> postoutput_div2;
        
        
        auto test_interface_divergence0  = std::make_shared< gnuplot_output_object<double> >("k0_divergence_interface_Stokes_initial.dat");
        auto test_interface_divergence1  = std::make_shared< gnuplot_output_object<double> >("k1_divergence_interface_Stokes_initial.dat");
        auto test_interface_divergence2  = std::make_shared< gnuplot_output_object<double> >("k2_divergence_interface_Stokes_initial.dat");
       
        std::vector<T> val_u_nx , val_u_ny , val_u_n ;
        std::vector< point<T, 2> > interface_points_plot ;
        std::vector< std::pair<T,T> > interface_normals , interface_normals_n_cont , interface_normals_grad_cont ;
         T divergence_error0 = 0. , divergence_error1 = 0. , divergence_error2 = 0.;
        
        for(auto& cl : msh_i.cells)
        {
            ls_cell.cell_assignment(cl);
            u_projected.cell_assignment(cl);
            
            if( location(msh_i, cl) == element_location::IN_NEGATIVE_SIDE || location(msh_i, cl) == element_location::ON_INTERFACE )
            {
                T partial_area = measure( msh_i, cl, element_location::IN_NEGATIVE_SIDE);
                area0 += partial_area;
                auto qps = integrate( msh_i , cl , 2*degree_FEM+1 , element_location::IN_NEGATIVE_SIDE);
                for(auto& qp:qps){
                    mass0 += qp.second * ls_cell(qp.first);
                    centre_mass0_x += qp.second * qp.first.x() ;
                    centre_mass0_y += qp.second * qp.first.y() ;
                }
            }
            if(cl.user_data.location == element_location::ON_INTERFACE)
            {
                /*
                std::cout<<"CELL = "<<offset(msh,cl)<<std::endl;
                if(offset(msh,cl) == 101 || offset(msh,cl) == 104 )
                {
                    for(size_t kk = 0 ; kk < u_projected.sol_HHO.first.rows() ; kk++ )
                    std::cout<< u_projected.sol_HHO.first(kk,offset(msh,cl)) << " " << u_projected.sol_HHO.second(kk,offset(msh,cl)) << '\n';
                    std::cout<<'\n'<<std::endl;
                }
                */
                for(auto interface_point = cl.user_data.interface.begin() ; interface_point < cl.user_data.interface.end() -1 ; interface_point++ )
                {
                    
                    
                    T val0 = ls_cell.divergence( *interface_point );
                    divergence_error0 += pow((std::abs(val0) - 1.0/radius),2) ;
                    
                    T val1 = ls_cell.divergence_cont( *interface_point );
                    divergence_error1 += pow((std::abs(val1) - 1.0/radius),2) ;
                    
                    T val2 = ls_cell.divergence_disc( *interface_point );
                    divergence_error2 += pow((std::abs(val2) - 1.0/radius),2) ;
                    
                    Eigen::Matrix<T,2,1> normal = ls_cell.normal(*interface_point);
                    std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                    interface_normals.push_back( normal_vec ) ;
                  
                    Eigen::Matrix<T,2,1> normal_cont = ls_cell.normal_cont(*interface_point);
                    //std::cout<<"normal = "<<'\n'<<normal<<" , CONTIUOUS_normal = "<<'\n'<<normal_cont<<std::endl;
                    std::pair<T,T> normal_vec_cont = std::make_pair(normal_cont(0),normal_cont(1));
                    interface_normals_n_cont.push_back( normal_vec_cont ) ;
                    
                    Eigen::Matrix<T,2,1> normal_cont_grad = ls_cell.normal_disc(*interface_point);
                    //std::cout<<"normal = "<<'\n'<<normal<<" , CONTIUOUS_normal = "<<'\n'<<normal_cont<<std::endl;
                    std::pair<T,T> normal_vec_cont_grad = std::make_pair(normal_cont_grad(0),normal_cont_grad(1));
                    interface_normals_grad_cont.push_back( normal_vec_cont_grad ) ;
                    
                    
                    
                    if( time_step == 0 )
                    {
                        
                        vec_normal->add_data(*interface_point,normal_vec);
                        test_interface_divergence0->add_data(*interface_point , val0);
                        test_interface_divergence1->add_data(*interface_point , val1);
                        test_interface_divergence2->add_data(*interface_point , val2);
                        vec_normal_n_cont->add_data(*interface_point,normal_vec_cont);
                        vec_normal_grad_cont->add_data(*interface_point,normal_vec_cont_grad);
                    }
                    
                    perimeter0 += ( *(interface_point+1) - *interface_point ).to_vector().norm();
                    
                    normal_interface_status += pow( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) , 2) + pow( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) , 2 );
                    
                    interface_points_plot.push_back(*(interface_point)) ;
                    val_u_nx.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) );
                    val_u_ny.push_back( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                    val_u_n.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) + u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                    //std::cout<<"*(interface_point) = "<<*(interface_point)<<std::endl;
                    //std::cout<<" u_projected primo = "<<u_projected(*(interface_point)).first  << " u_projected second = "<< u_projected(*(interface_point)).second  << " somma tot con olds = "<<normal_interface_status<<std::endl;
                    //std::cout<<" ls_cell.normal(*(interface_point))(0)  = "<<ls_cell.normal(*(interface_point))(0)  << " ls_cell.normal(*(interface_point))(1)  = "<< ls_cell.normal(*(interface_point))(1)  << std::endl;
                    
                    counter_interface_pts++;
                       
                }
                
                T val0 = ls_cell.divergence(*(cl.user_data.interface.end()-1));
                T val1 = ls_cell.divergence_cont(*(cl.user_data.interface.end()-1));
                T val2 = ls_cell.divergence_disc(*(cl.user_data.interface.end()-1));
                divergence_error0 += pow((std::abs(val0) - 1.0/radius),2) ;
                divergence_error1 += pow((std::abs(val1) - 1.0/radius),2) ;
                divergence_error2 += pow((std::abs(val2) - 1.0/radius),2) ;
                
                Eigen::Matrix<T,2,1> normal = ls_cell.normal(*(cl.user_data.interface.end()-1));
                std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                interface_normals.push_back( normal_vec ) ;
                
                Eigen::Matrix<T,2,1> normal_cont_n = ls_cell.normal_cont(*(cl.user_data.interface.end()-1));
                std::pair<T,T> normal_vec_n_cont = std::make_pair(normal_cont_n(0),normal_cont_n(1));
                interface_normals_n_cont.push_back( normal_vec_n_cont ) ;
                
                Eigen::Matrix<T,2,1> normal_cont_grad = ls_cell.normal_disc(*(cl.user_data.interface.end()-1));
                std::pair<T,T> normal_vec_grad_cont = std::make_pair(normal_cont_grad(0),normal_cont_grad(1));
                interface_normals_grad_cont.push_back( normal_vec_grad_cont ) ;
                
                /// COSE  PER NORMALE CONTINUA
                /*
                Eigen::Matrix<T,2,1> normal_cont = ls_cell.normal_continuous(*(cl.user_data.interface.end()-1));
                std::pair<T,T> normal_vec_cont = std::make_pair(normal_cont(0),normal_cont(1));
                interface_normals_cont.push_back( normal_vec_cont ) ;
               
                 vec_normal_cont->add_data(*(cl.user_data.interface.end()-1) ,normal_vec_cont);
                */
                
                if( time_step == 0 )
                {
                    
                    vec_normal->add_data(*(cl.user_data.interface.end()-1) ,normal_vec);
                    test_interface_divergence0->add_data( *(cl.user_data.interface.end()-1) ,val0 );
                    test_interface_divergence1->add_data( *(cl.user_data.interface.end()-1) ,val1 );
                    test_interface_divergence2->add_data( *(cl.user_data.interface.end()-1) ,val2 );
                    vec_normal_n_cont->add_data(*(cl.user_data.interface.end()-1), normal_vec_n_cont);
                    vec_normal_grad_cont->add_data(*(cl.user_data.interface.end()-1), normal_vec_grad_cont);
                }
                                   
               
                normal_interface_status += pow( u_projected (*(cl.user_data.interface.end()-1) ).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0),2) + pow( u_projected(*( cl.user_data.interface.end()-1) ).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1), 2);
                
                
           
                interface_points_plot.push_back(*(cl.user_data.interface.end()-1)) ;
                val_u_nx.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) );
                val_u_ny.push_back( u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                val_u_n.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) + u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                
                
                counter_interface_pts++;
                   
               
            
            }
            
        }
        
        if( time_step == 0 )
        {
            postoutput_div2.add_object(test_interface_divergence0);
            postoutput_div2.add_object(test_interface_divergence1);
            postoutput_div2.add_object(test_interface_divergence2);
            postoutput_div2.write();
            
            postoutput_vec.add_object(vec_normal);
            postoutput_vec.add_object(vec_normal_n_cont);
            postoutput_vec.add_object(vec_normal_grad_cont);
            postoutput_vec.write();
            
        }
        
        std::cout<<"NOTICE:--> The terms u \cdot n are made with n = normal (NO CONTINUOUS)."<<std::endl;
        
        goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
        testing_level_set_time(msh,level_set_function,tot_time);
        /*
        if( time_step == 0 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
            
        if( time_step == 5 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 10 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 15 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
            
        if( time_step == 20 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        if( time_step == 30 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        
        if( time_step == 40 ){
            goal_quantities_time(msh , tot_time, interface_points_plot , val_u_nx , val_u_ny , val_u_n , interface_normals ) ;
            testing_level_set_time(msh,level_set_function,tot_time);
        }
        */
        
        divergence_error0 /= counter_interface_pts;
        divergence_error0 = sqrt(divergence_error0);
        
        divergence_error1 /= counter_interface_pts;
        divergence_error1 = sqrt(divergence_error1);
        
        divergence_error2 /= counter_interface_pts;
        divergence_error2 = sqrt(divergence_error2);
        std::cout<<yellow<<bold<<"The l2 error of the CURVATURE at the INTERFACE, at time "<<reset<< dt*time_step <<" is " << divergence_error0 <<std::endl;
        std::cout<<yellow<<bold<<"The l2 error of the CURVATURE at the INTERFACE, at time "<<reset<< dt*time_step <<" is " << divergence_error1 <<std::endl;
        std::cout<<yellow<<bold<<"The l2 error of the CURVATURE at the INTERFACE, at time "<<reset<< dt*time_step <<" is " << divergence_error2 <<std::endl;
                   
        
        std::cout<<"number of interface points is " << counter_interface_pts << std::endl;
        normal_interface_status /= counter_interface_pts;
        normal_interface_status = sqrt(normal_interface_status);
        std::cout<<yellow<<bold<<"The l2 error of u*n over the INTERFACE, at time "<<reset<< tot_time <<" is " << normal_interface_status << std::endl;
        if(time_step == 0)
            check = 10.0;
        else
            check = normal_interface_status ;
        
        std::cout << "Area at time step: "<<tot_time<<" is "<< area0  << reset << std::endl;
        std::cout  << "Internal mass at time step: "<<tot_time<<" is "<<reset<< mass0   << std::endl;
        //std::cout<<bold<<yellow << "GLOBAL Mass at time step: "<<time_step<<" is "<<reset<<global_mass0<< reset << std::endl;
         std::cout << "CENTRE OF MASS at time step: " <<tot_time<<" is "<<" ( "<< centre_mass0_x/area0  << " , "<< centre_mass0_y/area0 <<" ). " << reset << std::endl;
        
         d_a = sqrt(4.0*area0/M_PI) ;
        
        std::cout<<"The PERIMETER, at time "<< tot_time <<" is " << perimeter0 <<std::endl;
        
        std::cout<<yellow<<bold<<"The CIRCULARITY, at time "<< tot_time <<" is "<<reset << M_PI*d_a/perimeter0 <<std::endl;
        
        
        
        if(time_step == 0 ){
            initial_area  = area0 ;
            initial_mass = mass0 ;
            centre_mass_x_inital = centre_mass0_x ;
            centre_mass_y_inital = centre_mass0_y ;
            perimeter_initial = perimeter0 ;
        }
        
        if(time_step > 0 )
        {
            diff_area = (area0 - initial_area)/initial_area ;
            diff_mass = (std::abs((mass0 - initial_mass)))/(std::abs(initial_mass)) ;
            std::cout << bold << yellow << "Normalised difference in Area (new - old)/old at time step: "<<tot_time<<" is "<<reset<< diff_area  << reset << std::endl;
            std::cout << bold << yellow << "Difference in internal MASS |new - old|/|old| at time step: "<<tot_time<<" is "<<reset<< diff_mass  << reset << std::endl;
            
            std::cout << "TRANSLATION OF THE CENTRE OF MASS at time step: "  <<tot_time<<" is "<<" ( " << centre_mass0_x/area0 - centre_mass_x_inital/initial_area <<" , " << centre_mass0_y/area0 - centre_mass_y_inital/initial_area<<" ). " << std::endl;
            std::cout<<"NORMALISED DIFFERENCE PERIMETER, at time " << tot_time <<" is " << (perimeter0 - perimeter_initial)/perimeter_initial <<std::endl;
                   
        }
        
        
        tc_agglo.toc();
        std::cout << bold << yellow << "------> TIME FOR CHECKING GOAL QUANTITIES: " << tc_agglo << " seconds" << reset << std::endl;
        // auto test_case = make_test_case_stokes_1(msh, level_set_function);
        // auto test_case = make_test_case_stokes_2(msh, ls_cell); //level_set_function);
        
/// DA AGGIUNGERE UNA VOLTA SISTEMATO IL CODICE
        /*
        if(check < 1e-8 )
        {
            std::cout<<" check = "<<check<<" , STOP!"<<std::endl;
            return 0;
        }
        */
        bool sym_grad = TRUE;
        
        auto prm = params<T>();
        prm.kappa_1 = 1.0;
        prm.kappa_2 = 1.0;
        
        //auto test_case = make_test_case_eshelby(msh_i, ls_cell,  prm , sym_grad);
     
        
        // Non serve modificare Gamma = 1/2
        //auto test_case = make_test_case_eshelby_2(msh_i, ls_cell,  prm , sym_grad );
        //auto test_case = make_test_case_eshelby_analytic(msh_i, ls_cell,  prm , sym_grad , radius);
     
        auto test_case_prova = make_test_case_eshelby_2_prova(msh_i, ls_cell,  prm , sym_grad );
       
        //auto method = make_sym_gradrec_stokes_interface_method(msh_i, 1.0, 0.0, test_case, sym_grad);
        
        auto method = make_sym_gradrec_stokes_interface_method(msh_i, 1.0, 0.0, test_case_prova, sym_grad);
      
        //auto u_projected = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
       
        tc_agglo.tic();
    
        if(solve_interface){
            //run_cuthho_interface_velocity_parallel(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad );
            run_cuthho_interface_velocity_prova(msh_i, degree, method,test_case_prova, ls_cell , u_projected ,sym_grad );
            //run_cuthho_interface_velocity(msh_i, degree, method, test_case, ls_cell , u_projected ,sym_grad );
            
            // OLD
            //run_cuthho_interface_velocity_analytic(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad ,radius );
            
            //run_cuthho_interface(msh, degree, method, test_case);
        }
        if (solve_fictdom)
            run_cuthho_fictdom(msh_i, degree, test_case_prova);
        
        tc_agglo.toc();
        std::cout << bold << yellow << "TIME-----> run_cuthho_interface_velocity: " << tc_agglo << " seconds" << reset << std::endl;
        
        
        /*********************** FEM -  PROCESSING **************************/
        /// ORA HO SMOOTH OPERATOR! USE L2 PROJECTION
        
        if( 1 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: SMOOTH OPERATOR FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.smooth_converting_into_FE_formulation( u_projected.sol_HHO );
        }
        if( 0 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: OLD OPERATOR FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.converting_into_FE_formulation( u_projected.sol_HHO );
        }
        if( 0 )
        {
            std::cout<<yellow<<bold<<"------------------>>>>NOTICE: L^2 PROJECTION FROM HHO TO FEM."<<reset<<std::endl;
            u_projected.L2_proj_into_FE_formulation(level_set_function , msh);
        }
        
        
        /*
        auto u_prova = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
        u_prova.sol_HHO = u_projected.sol_HHO ;
        u_prova.L2_proj_into_FE_formulation( level_set_function , msh );
        
        
        testing_velocity_field_L2projected(msh , u_prova) ;
        
        //std::cout<<"CHECK SMOOTH CONVERTING  FEM ----> FIRST"<<'\n'<<(u_prova.sol_FEM.first - u_projected.sol_FEM.first)<<'\n' <<std::endl;
        //std::cout<<"CHECK SMOOTH CONVERTING FEM ----> SECOND"<<'\n'<<(u_prova.sol_FEM.second - u_projected.sol_FEM.second)<<'\n' <<std::endl;
        
        testing_velocity_field(msh , u_projected) ;
        */
        
        testing_velocity_field(msh , u_projected) ;
        
        //check_inlet( msh , fe_data ,  u_projected , 1e-14 );
        //std::cout<<"u_projected.sol_FEM.first = "<<'\n'<<u_projected.sol_FEM.first <<std::endl;
        //std::cout<<"u_projected.sol_FEM.second = "<<'\n'<<u_projected.sol_FEM.second <<std::endl;
        T eps = 0.48 ; // factor to be inside CFL stability zone
        //T dt1 = time_step_CFL( u , mip , eps ); // OLD IMPLEMENTATION
        T dt1 = time_step_CFL_new( u_projected , mip , eps );
        //std::cout<<"dt1 is "<<dt1<<std::endl;
        T dt2 = 0.1;
        dt = std::min(dt1 , dt2);
        std::cout<<"MAX dt = "<<dt<<" AND HEURISTIC CFL IS "<<dt1<<std::endl;
        
        /*
        if(high_order)
            run_FEM_BERNSTEIN_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
            //run_FEM_BERNSTEIN_CORRECT( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
        else
            run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
            //run_FEM_BERNSTEIN_LOW_ORDER_CORRECT( level_set_function.msh , fe_data , level_set_function , u_projected , dt);
        */
        
        // I can create a sub-time. I solve several time the FEM problem, given a Stokes field. The amount of time is s.t. at maximum there is a displacement of a cell of the interface and no more than a maximum T
        T sub_time = 0.;
        T sub_dt = std::min(4e-4 , dt ) ;
        std::cout<<"Implemented dt = "<<dt<<std::endl;
        while( (sub_time < sub_dt*10) && (sub_time < dt1) )
        {
            if(high_order){
                run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D_NEW_DIRICHLET_COND( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt , mapping );

                //run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
                //run_FEM_BERNSTEIN_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
            }
            else
                run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST( level_set_function.msh , fe_data , level_set_function , u_projected , sub_dt);
               
            sub_time += sub_dt ;
      
        }
        std::cout<<yellow<<bold<<"SUB TIME REPETITION OF TRANSPORT PB DATA:"<<reset<<" sub_dt = "<<sub_dt<< " , number time steps = "<<sub_time/sub_dt<<std::endl;
        
        tot_time += sub_time ;
        
        testing_level_set2(msh,level_set_function);
        
        /// OLD IMPLEMENTATION
        //for(size_t j=1; j<4 ; j++)
        //run_FEM_levelset( level_set_function.msh,degree_FEM,level_set_function,u,dt,mip);
       
        /************** FEM -  POST-PROCESSING **************/
        if( (T_N - time_step)==0 )
        {
            // Uploading continuous normal function
            level_set_function.normal_continuous_setting();
            level_set_function.gradient_continuous_setting() ;
            // Uploading mesh data to check out differences in mass and areas
            crr_mesh.current_mesh = msh;
            Mesh msh_i2 =  crr_mesh.current_mesh;
            offset_definition(msh_i2);
            tc.tic();
            detect_node_position3(msh_i2, level_set_function); // In cuthho_geom
            detect_cut_faces3(msh_i2, level_set_function); // In cuthho_geom
        
       
            if (agglomeration)
            {
                // std::cout<<"i m here 1"<<std::endl;
                detect_cut_cells3(msh_i2, level_set_function); // In cuthho_geom
           
                detect_cell_agglo_set(msh_i2, level_set_function); // Non serve modificarla
           
                make_neighbors_info_cartesian(msh_i2); // Non serve modificarla
                //refine_interface_angle(msh_i2, level_set_function, int_refsteps); // IN cuthho_geom
                refine_interface_pro3(msh_i2, level_set_function, int_refsteps); // IN cuthho_geom
                make_agglomeration(msh_i2, level_set_function); // Non serve modificarla
            }
            else
            {
                //move_nodes(msh_i2, level_set_function);
                //detect_cut_faces2(msh_i2, level_set_function); //do it again to update intersection     points
                detect_cut_cells3(msh_i2, level_set_function);
                //refine_interface2(msh_i2, level_set_function, int_refsteps);
                //refine_interface_angle(msh_i2, level_set_function, int_refsteps);
                refine_interface_pro3(msh_i2, level_set_function, int_refsteps);
            }
        
            tc.toc();
            std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

            if (dump_debug)
            {
                dump_mesh(msh_i2);
                output_mesh_info(msh_i2, level_set_function);
            }
        
            output_mesh_info2(msh_i2, level_set_function);
            // IN cuthho_export..Points/Nodes don't change
           
            // Uploading level set
            ls_cell.level_set = level_set_function;
            ls_cell.agglo_msh = msh_i2;
            u_projected.set_agglo_mesh( msh_i2 );
            
           
            T mass_fin = 0. , area_fin = 0. ;
            T centre_mass_x = 0. , centre_mass_y = 0. ;
        
            T perimeter = 0. ;
            normal_interface_status = 0. ;
            counter_interface_pts = 0;
   
            
            postprocess_output<double> postoutput_vec;
            auto vec_normal_fin = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_Stokes_final.dat");
            auto vec_normal_n_cont_fin = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_continuos_n_Stokes_final.dat");
             
            auto vec_normal_grad_cont_fin = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_continuos_grad_Stokes_final.dat");
            
            postprocess_output<double> postoutput_div2;
            auto test_interface_divergence_fin0  = std::make_shared< gnuplot_output_object<double> >("k0_fin_divergence_interface_Stokes_final.dat");
            auto test_interface_divergence_fin1  = std::make_shared< gnuplot_output_object<double> >("k1_fin_divergence_interface_Stokes_final.dat");
            auto test_interface_divergence_fin2  = std::make_shared< gnuplot_output_object<double> >("k2_fin_divergence_interface_Stokes_final.dat");
            
            std::vector<T> val_u_nx_fin , val_u_ny_fin , val_u_n_fin ;
            std::vector< point<T, 2> > interface_points_plot_fin ;
            std::vector< std::pair<T,T> > interface_normals_fin ,interface_normals_n_cont_fin ,interface_normals_grad_cont_fin ;
            
            T divergence_error_fin0 = 0. , divergence_error_fin1 = 0. , divergence_error_fin2 = 0.;
            
            
            
            for(auto& cl : msh_i2.cells)
            {
                ls_cell.cell_assignment(cl);
                u_projected.cell_assignment(cl);
                
                if( (location(msh_i2, cl) == element_location::IN_NEGATIVE_SIDE) || (location(msh_i2, cl) == element_location::ON_INTERFACE) )
                {
                    
                    T partial_area = measure( msh_i2, cl, element_location::IN_NEGATIVE_SIDE);
                 
                    area_fin += partial_area;
                    
                   
                    auto qps_fin = integrate( msh_i2 , cl , 2*degree_FEM+1 , element_location::IN_NEGATIVE_SIDE);
                   
                    for(auto& qp:qps_fin){
                        mass_fin += qp.second * ls_cell(qp.first);
                        centre_mass_x += qp.second * qp.first.x() ;
                        centre_mass_y += qp.second * qp.first.y() ;
                    }
                   
                }
                if(cl.user_data.location == element_location::ON_INTERFACE)
                {
                    
                    for(auto interface_point = cl.user_data.interface.begin() ; interface_point < cl.user_data.interface.end() -1 ; interface_point++ )
                    {
                        perimeter += ( *(interface_point+1) - *interface_point ).to_vector().norm();
                        
                        normal_interface_status += pow( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) , 2) + pow( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) , 2 );
                        
                        
                        T val0 = ls_cell.divergence( *interface_point );
                        divergence_error_fin0 += pow((std::abs(val0) - 1.0/radius),2) ;
                        T val1 = ls_cell.divergence_cont( *interface_point );
                        divergence_error_fin1 += pow((std::abs(val1) - 1.0/radius),2) ;
                        T val2 = ls_cell.divergence_disc( *interface_point );
                        divergence_error_fin2 += pow((std::abs(val2) - 1.0/radius),2) ;
                        
                        Eigen::Matrix<T,2,1> normal = ls_cell.normal(*interface_point);
                        std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                        interface_normals_fin.push_back( normal_vec ) ;
                        
                        Eigen::Matrix<T,2,1> normal_cont_n = ls_cell.normal_cont(*interface_point);
                        std::pair<T,T> normal_vec_n_cont = std::make_pair(normal_cont_n(0),normal_cont_n(1));
                        interface_normals_n_cont_fin.push_back( normal_vec_n_cont ) ;
                        
                        Eigen::Matrix<T,2,1> normal_cont_grad = ls_cell.normal_disc(*interface_point);
                        std::pair<T,T> normal_vec_grad_cont = std::make_pair(normal_cont_grad(0),normal_cont_grad(1));
                        interface_normals_grad_cont_fin.push_back( normal_vec_grad_cont ) ;
                        
                        vec_normal_fin->add_data(*interface_point,normal_vec);
                        vec_normal_n_cont_fin->add_data(*interface_point,normal_vec_n_cont);
                        vec_normal_grad_cont_fin->add_data(*interface_point,normal_vec_grad_cont);
                        test_interface_divergence_fin0->add_data( *interface_point , val0);
                        test_interface_divergence_fin1->add_data( *interface_point , val1);
                        test_interface_divergence_fin2->add_data( *interface_point , val2);
                        
                        interface_points_plot_fin.push_back( *(interface_point) ) ;
                        val_u_nx_fin.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) );
                        val_u_ny_fin.push_back( u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                        val_u_n_fin.push_back( u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0) + u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1) );
                        
                        counter_interface_pts++;
                    }
                    
                    T val0 = ls_cell.divergence( *(cl.user_data.interface.end()-1) );
                    divergence_error_fin0 += pow((std::abs(val0) - 1.0/radius),2) ;
                    
                    T val1 = ls_cell.divergence_cont( *(cl.user_data.interface.end()-1) );
                    divergence_error_fin1 += pow((std::abs(val1) - 1.0/radius),2) ;
                    
                    T val2 = ls_cell.divergence_disc( *(cl.user_data.interface.end()-1) );
                    divergence_error_fin2 += pow((std::abs(val2) - 1.0/radius),2) ;
                    
                    Eigen::Matrix<T,2,1> normal = ls_cell.normal(*(cl.user_data.interface.end()-1));
                    std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                    interface_normals_fin.push_back( normal_vec ) ;
                    
                    Eigen::Matrix<T,2,1> normal_n_cont = ls_cell.normal_cont(*(cl.user_data.interface.end()-1));
                    std::pair<T,T> normal_vec_n_cont = std::make_pair(normal_n_cont(0),normal_n_cont(1));
                    interface_normals_n_cont_fin.push_back( normal_vec_n_cont ) ;
                    
                    Eigen::Matrix<T,2,1> normal_grad_cont = ls_cell.normal_disc(*(cl.user_data.interface.end()-1));
                    std::pair<T,T> normal_vec_grad_norm = std::make_pair(normal_grad_cont(0),normal_grad_cont(1));
                    interface_normals_grad_cont_fin.push_back( normal_vec_grad_norm ) ;
                    
                    vec_normal_fin->add_data( *(cl.user_data.interface.end()-1) ,normal_vec);
                    vec_normal_n_cont_fin->add_data( *(cl.user_data.interface.end()-1) ,normal_vec_n_cont);
                    vec_normal_grad_cont_fin->add_data( *(cl.user_data.interface.end()-1) ,normal_vec_grad_norm);
                    test_interface_divergence_fin0->add_data( *(cl.user_data.interface.end()-1) ,val0 );
                    test_interface_divergence_fin1->add_data( *(cl.user_data.interface.end()-1) ,val1 );
                    test_interface_divergence_fin2->add_data( *(cl.user_data.interface.end()-1) ,val2 );
                    
                    normal_interface_status += pow( u_projected (*(cl.user_data.interface.end()-1) ).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0),2) + pow( u_projected(*( cl.user_data.interface.end()-1) ).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1), 2);
                    
                    interface_points_plot_fin.push_back( *(cl.user_data.interface.end()-1) ) ;
                    val_u_nx_fin.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) );
                    val_u_ny_fin.push_back( u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                    val_u_n_fin.push_back( u_projected(*(cl.user_data.interface.end()-1)).first * ls_cell.normal(*(cl.user_data.interface.end()-1))(0) + u_projected(*(cl.user_data.interface.end()-1)).second * ls_cell.normal(*(cl.user_data.interface.end()-1))(1) );
                                
                    counter_interface_pts++;

                }
                
            }
            
            postoutput_div2.add_object(test_interface_divergence_fin0);
            postoutput_div2.add_object(test_interface_divergence_fin1);
            postoutput_div2.add_object(test_interface_divergence_fin2);
            postoutput_div2.write();
                       
            postoutput_vec.add_object(vec_normal_fin);
            postoutput_vec.add_object(vec_normal_n_cont_fin);
            postoutput_vec.add_object(vec_normal_grad_cont_fin);
            postoutput_vec.write();
            
            goal_quantities_time(msh , tot_time, interface_points_plot_fin , val_u_nx_fin , val_u_ny_fin , val_u_n_fin , interface_normals_fin ) ;
            testing_level_set_time(msh,level_set_function, tot_time);
            /*
            if( time_step == 9 ){
                goal_quantities_time(msh , tot_time, interface_points_plot_fin , val_u_nx_fin , val_u_ny_fin , val_u_n_fin , interface_normals_fin ) ;
                testing_level_set_time(msh,level_set_function, tot_time);
            }
            */
            divergence_error_fin0 /= counter_interface_pts;
            divergence_error_fin0 = sqrt(divergence_error_fin0);
            
            divergence_error_fin1 /= counter_interface_pts;
            divergence_error_fin1 = sqrt(divergence_error_fin1);
            
            divergence_error_fin2 /= counter_interface_pts;
            divergence_error_fin2 = sqrt(divergence_error_fin2);
            
            std::cout<<yellow<<bold<<"The l2 error of the CURVATURE at the INTERFACE, at time "<<reset<< tot_time <<" is " << divergence_error_fin0 <<std::endl;
            
            std::cout<<yellow<<bold<<"The l2 error of the CURVATURE at the INTERFACE, at time "<<reset<< tot_time <<" is " << divergence_error_fin1 <<std::endl;
            
            std::cout<<yellow<<bold<<"The l2 error of the CURVATURE at the INTERFACE, at time "<<reset<< tot_time <<" is " << divergence_error_fin2 <<std::endl;
            
            std::cout<<"number of interface points is " << counter_interface_pts << std::endl;
            
            normal_interface_status /= counter_interface_pts;
            normal_interface_status = sqrt(normal_interface_status);
            
            std::cout<<yellow<<bold<<"The l2 error of u*n over the INTERFACE, at time "<<reset<< tot_time <<" is " << normal_interface_status << std::endl;
            
            
            
            std::cout<<"The PERIMETER, at time "<< tot_time <<" is " << perimeter <<std::endl;
            
            std::cout<<"perimeter = "<< perimeter << " AND  perimeter0 =  "<<perimeter_initial<<std::endl;
            std::cout<< bold << yellow<<"NORMALISED DIFFERENCE PERIMETER, at time "<<reset<< tot_time <<" is " << (perimeter - perimeter_initial)/perimeter_initial <<std::endl;
            
            d_a = sqrt(4.0*area_fin/M_PI) ;
            
            std::cout<< bold << yellow<<"The CIRCULARITY, at time "<< tot_time<<reset <<" is " << M_PI*d_a/perimeter <<std::endl;
            
            std::cout  << "Area at time step: " <<tot_time<<" is "<< area_fin << std::endl;
            std::cout << "Internal mass at time step: "<<tot_time<<" is "<< mass_fin << reset << std::endl;
            
            std::cout << bold << yellow << "NORMALISED Difference in AREA AT TIME "<<tot_time<<" IS "<< reset<< (area_fin - initial_area)/initial_area << std::endl;
            std::cout << bold << yellow << "NORMALISED Difference in INTERNAL MASS AT TIME "<<tot_time<<" IS "<< reset<< (std::abs(mass_fin - initial_mass))/(std::abs( initial_mass )) << std::endl;
            std::cout << "CENTRE OF MASS at time step: "<<tot_time<<" is "<<" ( " << centre_mass_x/area_fin <<" , " << centre_mass_y/area_fin<<" ). " << std::endl;
            std::cout << "TRANSLATION OF THE CENTRE OF MASS at time step: "  <<tot_time<<" is "<<" ( " << centre_mass_x/area_fin - centre_mass_x_inital/initial_area <<" , " << centre_mass_y/area_fin - centre_mass_y_inital/initial_area<<" ). " << std::endl;
            std::cout  << "Abs error over expected radius = "<< std::abs( sqrt(area_fin/M_PI) - radius ) << std::endl;
            
            
           
        } // END OF T = FINAL TIME
       
        time_pos +=dt ;
        //time_step++;
    } // End of the temporal loop
    
    std::cout<< bold << yellow <<"FINAL TIME IS t = "<< reset<<tot_time<<std::endl;
    return 0;
}
#endif

