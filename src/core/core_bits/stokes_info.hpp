/*
 *       /\        Matteo Cicuttin (C) 2017,2018
 *      /__\       matteo.cicuttin@enpc.fr
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

#pragma once


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
#include <map>
#include <iomanip>

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

#include "level_set_transport_problem.hpp"
using namespace level_set_transport;

namespace stokes_info{

    template<typename Mesh, typename testType, typename meth>
    stokes_test_info<typename Mesh::coordinate_type>
    run_cuthho_interface(const Mesh &msh, size_t degree, meth &method, testType &test_case, bool normal_analysis = FALSE) {
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
        hho_degree_info hdi(degree + 1, degree);

        tc.tic();
        auto assembler = make_stokes_interface_assembler(msh, bcs_vel, hdi);
        auto assembler_sc = make_stokes_interface_condensed_assembler(msh, bcs_vel, hdi);
        for (auto &cl: msh.cells) {
            auto contrib = method.make_contrib(msh, cl, test_case, hdi);
            auto lc = contrib.first;
            auto f = contrib.second;

            if (sc)
                assembler_sc.assemble(msh, cl, lc, f);
            else
                assembler.assemble(msh, cl, lc, f);
        }

        if (sc)
            assembler_sc.finalize();
        else
            assembler.finalize();

        tc.toc();
        std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

        if (sc)
            std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
        else
            std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

        std::cout << "Cells: " << msh.cells.size() << std::endl;
        std::cout << "Faces: " << msh.faces.size() << std::endl;

/************** SOLVE **************/
        tc.tic();
#if 1
        SparseLU <SparseMatrix<RealType>> solver;
        Matrix<RealType, Dynamic, 1> sol;

        if (sc) {
            solver.analyzePattern(assembler_sc.LHS);
            solver.factorize(assembler_sc.LHS);
            sol = solver.solve(assembler_sc.RHS);
        } else {
            solver.analyzePattern(assembler.LHS);
            solver.factorize(assembler.LHS);
            sol = solver.solve(assembler.RHS);
        }
#endif
#if 0
        Matrix<RealType, Dynamic, 1> sol;
        cg_params <RealType> cgp;
        cgp.histfile = "cuthho_cg_hist.dat";
        cgp.verbose = true;
        cgp.apply_preconditioner = true;
        if (sc) {
            sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
            cgp.max_iter = assembler_sc.LHS.rows();
            conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
        } else {
            sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
            cgp.max_iter = assembler.LHS.rows();
            conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
        }
#endif
        tc.toc();
        std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

/************** POSTPROCESS **************/


        postprocess_output <RealType> postoutput;

        auto uT1_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_uT1.dat");
        auto uT2_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_uT2.dat");
        auto uT_l2_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_uT_norm.dat");
        auto p_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_p.dat");

        tc.tic();
        RealType H1_error = 0.0;
        RealType L2_error = 0.0;
        RealType L2_pressure_error = 0.0;
        RealType l2_u_n_error = 0.0;
        RealType l1_u_n_error = 0.0;
        RealType linf_u_n_error = 0.0;
        size_t counter_interface_pts = 0;

        for (auto &cl: msh.cells) {
            vector_cell_basis <cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
            cell_basis <cuthho_poly_mesh<RealType>, RealType> pb(msh, cl, hdi.face_degree());
            auto cbs = cb.size();
            auto pbs = pb.size();

            Matrix<RealType, Dynamic, 1> vel_locdata_n, vel_locdata_p, vel_locdata;
            Matrix<RealType, Dynamic, 1> P_locdata_n, P_locdata_p, P_locdata;
            Matrix<RealType, Dynamic, 1> vel_cell_dofs_n, vel_cell_dofs_p, vel_cell_dofs;

            if (location(msh, cl) == element_location::ON_INTERFACE) {
                if (sc) {
                    vel_locdata_n = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                    vel_locdata_p = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                    P_locdata_n = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                    P_locdata_p = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                } else {
                    vel_locdata_n = assembler.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                    vel_locdata_p = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                    P_locdata_n = assembler.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                    P_locdata_p = assembler.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                }

                vel_cell_dofs_n = vel_locdata_n.head(cbs);
                vel_cell_dofs_p = vel_locdata_p.head(cbs);


                auto qps_n = integrate(msh, cl, 2 * hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
                for (auto &qp: qps_n) {
/* Compute H1-error */
                    auto t_dphi = cb.eval_gradients(qp.first);
                    Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                    for (size_t i = 1; i < cbs; i++)
                        grad += vel_cell_dofs_n(i) * t_dphi[i].block(0, 0, 2, 2);

                    Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                    Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());

                    H1_error += qp.second * test_case.parms.kappa_1 * inner_product(grad_sym_diff, grad_sym_diff);



/* Compute L2-error */
                    auto t_phi = cb.eval_basis(qp.first);
                    auto v = t_phi.transpose() * vel_cell_dofs_n;
                    Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                    L2_error += qp.second * test_case.parms.kappa_1 * sol_diff.dot(sol_diff);

                    uT1_gp->add_data(qp.first, v(0));
                    uT2_gp->add_data(qp.first, v(1));
                    uT_l2_gp->add_data(qp.first, std::sqrt(v(0) * v(0) + v(1) * v(1)));

/* L2 - pressure - error */
                    auto p_phi = pb.eval_basis(qp.first);
                    RealType p_num = p_phi.dot(P_locdata_n);
                    RealType p_diff = test_case.sol_p(qp.first) - p_num;
                    auto p_prova = test_case.sol_p(qp.first);
//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                    L2_pressure_error += qp.second * p_diff * p_diff / test_case.parms.kappa_1;
                    p_gp->add_data(qp.first, p_num);


                }

                auto qps_p = integrate(msh, cl, 2 * hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
                for (auto &qp: qps_p) {
/* Compute H1-error */
                    auto t_dphi = cb.eval_gradients(qp.first);
                    Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                    for (size_t i = 1; i < cbs; i++)
                        grad += vel_cell_dofs_p(i) * t_dphi[i].block(0, 0, 2, 2);

                    Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                    Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                    H1_error += qp.second * test_case.parms.kappa_2 * inner_product(grad_sym_diff, grad_sym_diff);

/* Compute L2-error */
                    auto t_phi = cb.eval_basis(qp.first);
                    auto v = t_phi.transpose() * vel_cell_dofs_p;
                    Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                    L2_error += qp.second * test_case.parms.kappa_2 * sol_diff.dot(sol_diff);

                    uT1_gp->add_data(qp.first, v(0));
                    uT2_gp->add_data(qp.first, v(1));
                    uT_l2_gp->add_data(qp.first, std::sqrt(v(0) * v(0) + v(1) * v(1)));

/* L2 - pressure - error */
                    auto p_phi = pb.eval_basis(qp.first);
                    RealType p_num = p_phi.dot(P_locdata_p);
                    RealType p_diff = test_case.sol_p(qp.first) - p_num;
                    auto p_prova = test_case.sol_p(qp.first);
//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                    L2_pressure_error += qp.second * p_diff * p_diff / test_case.parms.kappa_2;
                    p_gp->add_data(qp.first, p_num);
                }

                if (normal_analysis) {
                    for (auto &interface_point: cl.user_data.interface) {
                        auto t_phi = cb.eval_basis(interface_point);
                        auto v = t_phi.transpose() * vel_cell_dofs_p;
                        auto n = level_set_function.normal(interface_point);
                        auto v_n = v.dot(n);
                        l2_u_n_error += pow(v_n, 2.0);
                        l1_u_n_error += std::abs(v_n);
                        linf_u_n_error = std::max(linf_u_n_error, std::abs(v_n));
                        counter_interface_pts++;
                    }
                }

            } else {
                if (sc) {
                    vel_locdata = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                    P_locdata = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                } else {
                    vel_locdata = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                    P_locdata = assembler.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                }
                vel_cell_dofs = vel_locdata.head(cbs);

                RealType kappa = test_case.parms.kappa_1;
                if (location(msh, cl) == element_location::IN_POSITIVE_SIDE)
                    kappa = test_case.parms.kappa_2;

                auto qps = integrate(msh, cl, 2 * hdi.cell_degree());
                for (auto &qp: qps) {
/* Compute H1-error */
                    auto t_dphi = cb.eval_gradients(qp.first);
                    Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                    for (size_t i = 1; i < cbs; i++)
                        grad += vel_cell_dofs(i) * t_dphi[i].block(0, 0, 2, 2);

                    Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                    Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                    H1_error += qp.second * kappa * inner_product(grad_sym_diff, grad_sym_diff);

/* Compute L2-error */
                    auto t_phi = cb.eval_basis(qp.first);
                    auto v = t_phi.transpose() * vel_cell_dofs;
                    Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                    L2_error += qp.second * kappa * sol_diff.dot(sol_diff);

                    uT1_gp->add_data(qp.first, v(0));
                    uT2_gp->add_data(qp.first, v(1));
                    uT_l2_gp->add_data(qp.first, std::sqrt(v(0) * v(0) + v(1) * v(1)));

/* L2 - pressure - error */
                    auto p_phi = pb.eval_basis(qp.first);
                    RealType p_num = p_phi.dot(P_locdata);
                    RealType p_ex = test_case.sol_p(qp.first);
                    RealType p_diff = test_case.sol_p(qp.first) - p_num;

//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                    L2_pressure_error += qp.second * p_diff * p_diff / kappa;

                    p_gp->add_data(qp.first, p_num);
                }
            }

        }

        std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
        std::cout << bold << green << "L2-norm absolute error:               " << std::sqrt(L2_error) << std::endl;
        std::cout << bold << green << "Pressure L2-norm absolute error:      " << std::sqrt(L2_pressure_error) << std::endl;

// Stefano: I dont want plots (in the code uTi_gp and p_gp still present). Just commented these.
        postoutput.add_object(uT1_gp);
        postoutput.add_object(uT2_gp);
        postoutput.add_object(uT_l2_gp);

        postoutput.add_object(p_gp);
        postoutput.write();


        stokes_test_info <RealType> TI;
        TI.H1_vel = std::sqrt(H1_error);
        TI.L2_vel = std::sqrt(L2_error);
        TI.L2_p = std::sqrt(L2_pressure_error);

        if (normal_analysis) {
            TI.l2_normal_vel = std::sqrt(l2_u_n_error / counter_interface_pts);
            TI.l1_normal_vel = l1_u_n_error / counter_interface_pts;
            TI.linf_normal_vel = linf_u_n_error;
        }

        if (0) {
/////////////// compute condition number
            SparseMatrix <RealType> Mat;
// Matrix<RealType, Dynamic, Dynamic> Mat;
            if (sc)
                Mat = assembler_sc.LHS;
            else
                Mat = assembler.LHS;


// Add by Stefano
            Eigen::BDCSVD <Eigen::MatrixXd> SVD(Mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
            double cond = SVD.singularValues()(0) / SVD.singularValues()(SVD.singularValues().size() - 1);
            std::cout << "cond_numb = " << cond << std::endl;



// Erased by Stefano
/*
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

        */
// compute condition number
//RealType cond = sigma_max / sigma_min;
            TI.cond = cond;
//std::cout << "sigma_max = " << sigma_max << "   sigma_min = " << sigma_min << "  cond = " << cond << std::endl;

        } else
            TI.cond = 0.0;

        tc.toc();
        std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


        return TI;
    }

    template<typename Mesh, typename testType, typename meth, typename Fonction>
    stokes_test_info<typename Mesh::coordinate_type>
    run_cuthho_interface_numerical_ls(const Mesh &msh, size_t degree, meth &method, testType &test_case,
                                      Fonction &level_set_function, bool normal_analysis = FALSE) {
        using RealType = typename Mesh::coordinate_type;

//    bool sym_grad = true ;
//
//    struct params<RealType> parms = test_case.parms;


        auto bcs_vel = test_case.bcs_vel;
        test_case.test_case_mesh_assignment(msh);

        timecounter tc;

        bool sc = true; // static condensation

        std::cout
                << "WARNING: interface integration points made by linear approximation. Integration error h^2 order. To be developped higher order integration."
                << std::endl;
//std::cout<<"WARNING: check integration points: it seems there is a repetition in agglomerated cells."<<std::endl; --> NOT TRUE, THE WEIGHTS ARE ZEROS!

/************** ASSEMBLE PROBLEM **************/
        hho_degree_info hdi(degree + 1, degree);

        tc.tic();
        auto assembler = make_stokes_interface_assembler(msh, bcs_vel, hdi);
        auto assembler_sc = make_stokes_interface_condensed_assembler(msh, bcs_vel, hdi);
// Dir condition out the for loop -> CHECK IT

        assembler_sc.set_dir_func(bcs_vel);
        for (auto &cl: msh.cells) {

            test_case.test_case_cell_assignment(cl);
//auto level_set_function = test_case.level_set_;
//test_case.refresh_lambdas(level_set_function, parms , sym_grad );

//auto rhs_fun = test_case.rhs_fun;
//auto sol_vel = test_case.sol_vel;
//auto sol_p = test_case.sol_p;
//auto vel_grad = test_case.vel_grad;
//auto bcs_vel = test_case.bcs_vel;
//auto neumann_jump = test_case.neumann_jump;




//std::cout<<"CHECK cl_refresh:"<<'\n'<<"---> CEll loop = "<<offset(msh,cl)<<" , refreshed cell = "<<offset(msh,test_case.upload_cl())<<std::endl;

            auto contrib = method.make_contrib(msh, cl, test_case, hdi);
            auto lc = contrib.first;
            auto f = contrib.second;

            if (sc)
                assembler_sc.assemble(msh, cl, lc, f);
            else
                assembler.assemble(msh, cl, lc, f);
        }

        if (sc)
            assembler_sc.finalize();
        else
            assembler.finalize();

        tc.toc();
        std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

        if (sc)
            std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
        else
            std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

        std::cout << "Cells: " << msh.cells.size() << std::endl;
        std::cout << "Faces: " << msh.faces.size() << std::endl;

/************** SOLVE **************/
        tc.tic();
#if 1
        SparseLU <SparseMatrix<RealType>> solver;
        Matrix<RealType, Dynamic, 1> sol;

        if (sc) {
            solver.analyzePattern(assembler_sc.LHS);
            solver.factorize(assembler_sc.LHS);
            sol = solver.solve(assembler_sc.RHS);
        } else {
            solver.analyzePattern(assembler.LHS);
            solver.factorize(assembler.LHS);
            sol = solver.solve(assembler.RHS);
        }
#endif
#if 0
        Matrix<RealType, Dynamic, 1> sol;
        cg_params <RealType> cgp;
        cgp.histfile = "cuthho_cg_hist.dat";
        cgp.verbose = true;
        cgp.apply_preconditioner = true;
        if (sc) {
            sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
            cgp.max_iter = assembler_sc.LHS.rows();
            conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
        } else {
            sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
            cgp.max_iter = assembler.LHS.rows();
            conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
        }
#endif
        tc.toc();
        std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

/************** POSTPROCESS **************/


        postprocess_output <RealType> postoutput;

        auto uT1_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_uT1.dat");
        auto uT2_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_uT2.dat");
        auto p_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_p.dat");
        auto uT_l2_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_uT_norm.dat");

        tc.tic();
        RealType H1_error = 0.0;
        RealType L2_error = 0.0;
        RealType L2_pressure_error = 0.0;
        RealType l1_u_n_error = 0.0;
//    RealType    l2_u_n_error = 0.0;
        RealType linf_u_n_error = 0.0;
        size_t counter_interface_pts = 0;
        RealType flux_interface = 0.0;
        RealType rise_vel0 = 0.0, rise_vel1 = 0.0, area_fin = 0.0;


        for (auto &cl: msh.cells) {
            vector_cell_basis <cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
            cell_basis <cuthho_poly_mesh<RealType>, RealType> pb(msh, cl, hdi.face_degree());
            auto cbs = cb.size();
//        auto pbs = pb.size();

            Matrix<RealType, Dynamic, 1> vel_locdata_n, vel_locdata_p, vel_locdata;
            Matrix<RealType, Dynamic, 1> P_locdata_n, P_locdata_p, P_locdata;
            Matrix<RealType, Dynamic, 1> vel_cell_dofs_n, vel_cell_dofs_p, vel_cell_dofs;

//auto level_set_function = test_case.level_set_;
            level_set_function.cell_assignment(cl);
            test_case.test_case_cell_assignment(cl);
//test_case.refresh_lambdas(level_set_function, parms , sym_grad );

            auto rhs_fun = test_case.rhs_fun;
            auto sol_vel = test_case.sol_vel;
            auto sol_p = test_case.sol_p;
            auto vel_grad = test_case.vel_grad;
            auto bcs_vel = test_case.bcs_vel;
            auto neumann_jump = test_case.neumann_jump;

//assembler_sc.set_dir_func( bcs_vel);

//std::cout<<"CHECK cl_refresh:"<<'\n'<<"---> CEll loop = "<<offset(msh,cl)<<" , refreshed cell = "<<offset(msh,test_case.upload_cl())<<std::endl;

            if (location(msh, cl) == element_location::ON_INTERFACE) {
                if (sc) {
                    vel_locdata_n = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                    vel_locdata_p = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                    P_locdata_n = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                    P_locdata_p = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                } else {
                    vel_locdata_n = assembler.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                    vel_locdata_p = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                    P_locdata_n = assembler.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                    P_locdata_p = assembler.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                }

                vel_cell_dofs_n = vel_locdata_n.head(cbs);
                vel_cell_dofs_p = vel_locdata_p.head(cbs);

//             if( cl.user_data.offset_subcells[0] == 83 || cl.user_data.offset_subcells[0] == 163 )
//             {
//                 std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) <<"cell = "<<offset(msh,cl)<<" , vertices:"<<std::endl;
//                 for(auto& pt:points(msh,cl))
//                     std::cout<<" pt = "<<pt;
//                 std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) <<'\n'<<"Interface points:"<<std::endl;
//                 for(auto& pt: cl.user_data.integration_msh.points)
//                     std::cout<<" pt = "<<pt;
//                 std::cout<<std::endl;
//             }

                auto qps_n = integrate(msh, cl, 2 * hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
                RealType local_H1_err_cut_n = 0.;
                for (auto &qp: qps_n) {
/* Compute H1-error */
                    auto t_dphi = cb.eval_gradients(qp.first);
                    Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                    for (size_t i = 1; i < cbs; i++)
                        grad += vel_cell_dofs_n(i) * t_dphi[i].block(0, 0, 2, 2);

                    Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                    Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                    H1_error += qp.second * test_case.parms.kappa_1 * inner_product(grad_sym_diff, grad_sym_diff);
                    local_H1_err_cut_n += qp.second * inner_product(grad_sym_diff, grad_sym_diff);


/* Compute L2-error */
                    auto t_phi = cb.eval_basis(qp.first);
                    auto v = t_phi.transpose() * vel_cell_dofs_n;
                    Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                    L2_error += qp.second * test_case.parms.kappa_1 * sol_diff.dot(sol_diff);

                    uT1_gp->add_data(qp.first, v(0));
                    uT2_gp->add_data(qp.first, v(1));
                    uT_l2_gp->add_data(qp.first, std::sqrt(v(0) * v(0) + v(1) * v(1)));

/* L2 - pressure - error */
                    auto p_phi = pb.eval_basis(qp.first);
                    RealType p_num = p_phi.dot(P_locdata_n);
                    RealType p_diff = test_case.sol_p(qp.first) - p_num;
//                auto p_prova = test_case.sol_p( qp.first ) ;
//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                    L2_pressure_error += qp.second * p_diff * p_diff / test_case.parms.kappa_1;
//                if( std::abs(qp.second * p_diff * p_diff) > 1e-15 )
//                    std::cout<<"L2 local pressure error (NEGATIVE CUT) = "<<qp.second * p_diff * p_diff<<std::endl;

                    p_gp->add_data(qp.first, p_num);


                }

//            std::cout<<"H1 local error (negative cut cell) = "<<local_H1_err_cut_n<<std::endl;
                RealType local_H1_err_cut_p = 0.;
                auto qps_p = integrate(msh, cl, 2 * hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);


                for (auto &qp: qps_p) {
/* Compute H1-error */
                    auto t_dphi = cb.eval_gradients(qp.first);
                    Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                    for (size_t i = 1; i < cbs; i++)
                        grad += vel_cell_dofs_p(i) * t_dphi[i].block(0, 0, 2, 2);

                    Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                    Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                    H1_error += qp.second * test_case.parms.kappa_2 * inner_product(grad_sym_diff, grad_sym_diff);
                    local_H1_err_cut_p += qp.second * inner_product(grad_sym_diff, grad_sym_diff);

/* Compute L2-error */
                    auto t_phi = cb.eval_basis(qp.first);
                    auto v = t_phi.transpose() * vel_cell_dofs_p;
                    Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                    L2_error += qp.second * test_case.parms.kappa_2 * sol_diff.dot(sol_diff);

                    uT1_gp->add_data(qp.first, v(0));
                    uT2_gp->add_data(qp.first, v(1));
                    uT_l2_gp->add_data(qp.first, std::sqrt(v(0) * v(0) + v(1) * v(1)));

/* L2 - pressure - error */
                    auto p_phi = pb.eval_basis(qp.first);
                    RealType p_num = p_phi.dot(P_locdata_p);
                    RealType p_diff = test_case.sol_p(qp.first) - p_num;
//                auto p_prova = test_case.sol_p( qp.first ) ;
//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                    L2_pressure_error += qp.second * p_diff * p_diff / test_case.parms.kappa_2;

//                if( std::abs(qp.second * p_diff * p_diff) > 1e-15 )
//                    std::cout<<"L2 local pressure error (POSITIVE CUT) = "<<qp.second * p_diff * p_diff<<std::endl;
                    p_gp->add_data(qp.first, p_num);
                }
//            std::cout<<"H1 local error (positive cut cell) = "<<local_H1_err_cut_p<<std::endl;

                if (normal_analysis) {
                    auto qps = integrate_interface(msh, cl, level_set_function.level_set.degree_FEM + hdi.cell_degree(),
                                                   element_location::ON_INTERFACE);
                    for (auto &qp: qps) {
                        auto t_phi = cb.eval_basis(qp.first);
                        auto v = t_phi.transpose() * vel_cell_dofs_p;
                        auto n = level_set_function.normal(qp.first);
                        auto v_n = v.dot(n);
                        l1_u_n_error += std::abs(v_n);
                        flux_interface += qp.second * std::abs(v_n);
                        linf_u_n_error = std::max(linf_u_n_error, std::abs(v_n));
                        counter_interface_pts++;

                    }


                    auto qps_n = integrate(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
                    RealType partial_area = measure(msh, cl, element_location::IN_NEGATIVE_SIDE);
                    area_fin += partial_area;

                    for (auto &qp: qps_n) {
                        auto t_phi = cb.eval_basis(qp.first);
                        auto v = t_phi.transpose() * vel_cell_dofs_n;
                        rise_vel0 += qp.second * v[0];
                        rise_vel1 += qp.second * v[1];
                    }


                }

            } else {
                if (sc) {
                    vel_locdata = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                    P_locdata = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                } else {
                    vel_locdata = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                    P_locdata = assembler.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                }
                vel_cell_dofs = vel_locdata.head(cbs);

                RealType kappa = test_case.parms.kappa_1;
                if (location(msh, cl) == element_location::IN_POSITIVE_SIDE)
                    kappa = test_case.parms.kappa_2;

                RealType local_H1_err_uncut = 0.;
                auto qps = integrate(msh, cl, 2 * hdi.cell_degree());
                for (auto &qp: qps) {
/* Compute H1-error */
                    auto t_dphi = cb.eval_gradients(qp.first);
                    Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                    for (size_t i = 1; i < cbs; i++)
                        grad += vel_cell_dofs(i) * t_dphi[i].block(0, 0, 2, 2);

                    Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                    Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                    H1_error += qp.second * kappa * inner_product(grad_sym_diff, grad_sym_diff);

                    local_H1_err_uncut += qp.second * inner_product(grad_sym_diff, grad_sym_diff);
/* Compute L2-error */
                    auto t_phi = cb.eval_basis(qp.first);
                    auto v = t_phi.transpose() * vel_cell_dofs;
                    Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                    L2_error += qp.second * kappa * sol_diff.dot(sol_diff);

                    uT1_gp->add_data(qp.first, v(0));
                    uT2_gp->add_data(qp.first, v(1));
                    uT_l2_gp->add_data(qp.first, std::sqrt(v(0) * v(0) + v(1) * v(1)));

/* L2 - pressure - error */
                    auto p_phi = pb.eval_basis(qp.first);
                    RealType p_num = p_phi.dot(P_locdata);
                    RealType p_diff = test_case.sol_p(qp.first) - p_num;
//                auto p_prova = test_case.sol_p( qp.first ) ;
//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                    L2_pressure_error += qp.second * p_diff * p_diff / kappa;

//                if( std::abs(qp.second * p_diff * p_diff) > 1e-15 )
//                    std::cout<<"L2 local pressure error (UNCUT) = "<<qp.second * p_diff * p_diff<<std::endl;
                    p_gp->add_data(qp.first, p_num);
                }
//            std::cout<<"H1 local error (uncut cell) = "<<local_H1_err_uncut<<std::endl;
                if (normal_analysis) {
                    auto qps = integrate(msh, cl, degree + 1);

                    RealType partial_area = measure(msh, cl);
                    area_fin += partial_area;

                    for (auto &qp: qps) {
                        auto t_phi = cb.eval_basis(qp.first);
                        auto v = t_phi.transpose() * vel_cell_dofs;
                        rise_vel0 += qp.second * v[0];
                        rise_vel1 += qp.second * v[1];
                    }
                }
            }

        }


        postoutput.add_object(uT1_gp);
        postoutput.add_object(uT2_gp);
        postoutput.add_object(p_gp);
        postoutput.add_object(uT_l2_gp);
        postoutput.write();


        stokes_test_info <RealType> TI;
        TI.H1_vel = std::sqrt(H1_error);
        TI.L2_vel = std::sqrt(L2_error);
        TI.L2_p = std::sqrt(L2_pressure_error);

        if (normal_analysis) {
            TI.l1_normal_vel = l1_u_n_error / counter_interface_pts;
            TI.flux_interface = flux_interface;
            TI.linf_normal_vel = linf_u_n_error;
        }


        std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
        std::cout << bold << green << "L2-norm absolute error:               " << std::sqrt(L2_error) << std::endl;
        std::cout << bold << green << "Pressure L2-norm absolute error:      " << std::sqrt(L2_pressure_error) << std::endl;
        std::cout << bold << green << "l1-norm u*n error:               " << TI.l1_normal_vel << std::endl;
        std::cout << bold << green << "linf-norm u*n error:               " << TI.linf_normal_vel << std::endl;
        std::cout << bold << green << "Flux interface:               " << flux_interface << std::endl;
        std::cout << bold << green << "Rise velocity x :               " << rise_vel0 / area_fin << std::endl;
        std::cout << bold << green << "Rise velocity y :               " << rise_vel1 / area_fin << std::endl;
        std::cout << bold << green << "|Rise velocity| :               "
                  << std::abs(rise_vel0 / area_fin) + std::abs(rise_vel1 / area_fin) << std::endl;
        std::cout << bold << green << "Rise velocity :               " << rise_vel0 / area_fin + rise_vel1 / area_fin
                  << std::endl;

        if (0) {
/////////////// compute condition number
            SparseMatrix <RealType> Mat;
// Matrix<RealType, Dynamic, Dynamic> Mat;
            if (sc)
                Mat = assembler_sc.LHS;
            else
                Mat = assembler.LHS;


// Add by Stefano
            Eigen::BDCSVD <Eigen::MatrixXd> SVD(Mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
            double cond = SVD.singularValues()(0) / SVD.singularValues()(SVD.singularValues().size() - 1);
            std::cout << "cond_numb = " << cond << std::endl;

/*
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
        */
            TI.cond = cond;
//std::cout << "sigma_max = " << sigma_max << "   sigma_min = "
//         << sigma_min << "  cond = " << cond
//         << std::endl;
        } else
            TI.cond = 0.0;

        tc.toc();
        std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


        return TI;
    }

    template<typename Mesh, typename testType, typename meth, typename Fonction, typename Velocity>
    stokes_test_info<typename Mesh::coordinate_type>
    run_cuthho_interface_numerical_ls_velocity(const Mesh &msh, size_t degree, meth &method, testType &test_case,
                                               Fonction &level_set_function, Velocity &velocity,
                                               bool normal_analysis = FALSE) {
        using RealType = typename Mesh::coordinate_type;

        bool sym_grad = true;

        struct params<RealType> parms = test_case.parms;


        auto bcs_vel = test_case.bcs_vel;
        test_case.test_case_mesh_assignment(msh);

        timecounter tc;

        bool sc = true; // static condensation

        std::cout
                << "WARNING: interface integration points made by linear approximation. Integration error h^2 order. To be developped higher order integration."
                << std::endl;


/************** ASSEMBLE PROBLEM **************/
        hho_degree_info hdi(degree + 1, degree);

        tc.tic();
        auto assembler = make_stokes_interface_assembler(msh, bcs_vel, hdi);
        auto assembler_sc = make_stokes_interface_condensed_assembler(msh, bcs_vel, hdi);
// Dir condition out the for loop -> CHECK IT

        assembler_sc.set_dir_func(bcs_vel);
        for (auto &cl: msh.cells) {

            test_case.test_case_cell_assignment(cl);

            auto contrib = method.make_contrib(msh, cl, test_case, hdi);
            auto lc = contrib.first;
            auto f = contrib.second;

            if (sc)
                assembler_sc.assemble(msh, cl, lc, f);
            else
                assembler.assemble(msh, cl, lc, f);
        }

        if (sc)
            assembler_sc.finalize();
        else
            assembler.finalize();

        tc.toc();
        std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

        if (sc)
            std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
        else
            std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

        std::cout << "Cells: " << msh.cells.size() << std::endl;
        std::cout << "Faces: " << msh.faces.size() << std::endl;

/************** SOLVE **************/
        tc.tic();
#if 1
        SparseLU <SparseMatrix<RealType>> solver;
        Matrix<RealType, Dynamic, 1> sol;

        if (sc) {
            solver.analyzePattern(assembler_sc.LHS);
            solver.factorize(assembler_sc.LHS);
            sol = solver.solve(assembler_sc.RHS);
        } else {
            solver.analyzePattern(assembler.LHS);
            solver.factorize(assembler.LHS);
            sol = solver.solve(assembler.RHS);
        }
#endif
#if 0
        Matrix<RealType, Dynamic, 1> sol;
        cg_params <RealType> cgp;
        cgp.histfile = "cuthho_cg_hist.dat";
        cgp.verbose = true;
        cgp.apply_preconditioner = true;
        if (sc) {
            sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
            cgp.max_iter = assembler_sc.LHS.rows();
            conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
        } else {
            sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
            cgp.max_iter = assembler.LHS.rows();
            conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
        }
#endif
        tc.toc();
        std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

/************** POSTPROCESS **************/


        postprocess_output <RealType> postoutput;

        auto uT1_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_uT1.dat");
        auto uT2_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_uT2.dat");
        auto p_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_p.dat");

        tc.tic();
        RealType H1_error = 0.0;
        RealType L2_error = 0.0;
        RealType L2_pressure_error = 0.0;
        RealType l1_u_n_error = 0.0;
        RealType l2_u_n_error = 0.0;
        RealType linf_u_n_error = 0.0;
        size_t counter_interface_pts = 0;
        RealType flux_interface = 0.0;
        RealType rise_vel0 = 0.0, rise_vel1 = 0.0, area_fin = 0.0;


        for (auto &cl: msh.cells) {
            vector_cell_basis <cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
            cell_basis <cuthho_poly_mesh<RealType>, RealType> pb(msh, cl, hdi.face_degree());
            auto cbs = cb.size();
            auto pbs = pb.size();

            Matrix<RealType, Dynamic, 1> vel_locdata_n, vel_locdata_p, vel_locdata;
            Matrix<RealType, Dynamic, 1> P_locdata_n, P_locdata_p, P_locdata;
            Matrix<RealType, Dynamic, 1> vel_cell_dofs_n, vel_cell_dofs_p, vel_cell_dofs;


            level_set_function.cell_assignment(cl);
            test_case.test_case_cell_assignment(cl);

            auto rhs_fun = test_case.rhs_fun;
            auto sol_vel = test_case.sol_vel;
            auto sol_p = test_case.sol_p;
            auto vel_grad = test_case.vel_grad;
            auto bcs_vel = test_case.bcs_vel;
            auto neumann_jump = test_case.neumann_jump;

//assembler_sc.set_dir_func( bcs_vel);


            if (location(msh, cl) == element_location::ON_INTERFACE) {
                if (sc) {
                    vel_locdata_n = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                    vel_locdata_p = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                    P_locdata_n = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                    P_locdata_p = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                } else {
                    vel_locdata_n = assembler.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                    vel_locdata_p = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                    P_locdata_n = assembler.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                    P_locdata_p = assembler.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                }

                vel_cell_dofs_n = vel_locdata_n.head(cbs);
                vel_cell_dofs_p = vel_locdata_p.head(cbs);

//---------------------- Updating velocity field by STE ----------------------

                if (level_set_function.subcells.size() < 1)  // NOT AGGLO CELL
                {
                    assert(level_set_function.agglo_LS_cl.user_data.offset_subcells.size() == 2);
                    assert(level_set_function.agglo_LS_cl.user_data.offset_subcells[0] ==
                           level_set_function.agglo_LS_cl.user_data.offset_subcells[1]);
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType, Mesh>(velocity.msh, cl_old,
                                                                                               velocity.degree_FEM);
                    size_t i_local = 0;
                    for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                        if (level_set_function(ln_Qk) > 0.0) {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                            velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local, offset_old) = vel(1);
                            i_local++;
                        } else {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                            velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local, offset_old) = vel(1);
                            i_local++;
                        }
                    }

                } else // AGGLO CELL
                {
                    for (size_t i_subcell = 0;
                         i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size(); i_subcell++) {
                        auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];

                        auto cl_old = velocity.msh.cells[offset_old];
                        auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType, Mesh>(velocity.msh, cl_old,
                                                                                                   velocity.degree_FEM);
                        size_t i_local = 0;
                        for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                            if (level_set_function(ln_Qk) > 0.0) {
                                auto phi_HHO = cb.eval_basis(ln_Qk);
                                auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                                velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                                velocity.sol_HHO.second(i_local, offset_old) = vel(1);

                                i_local++;
                            } else {
                                auto phi_HHO = cb.eval_basis(ln_Qk);
                                auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                                velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                                velocity.sol_HHO.second(i_local, offset_old) = vel(1);

                                i_local++;
                            }
                        }

                    } // FINE FOR
                } // FINE ELSE


                auto qps_n = integrate(msh, cl, 2 * hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
                for (auto &qp: qps_n) {
/* Compute H1-error */
                    auto t_dphi = cb.eval_gradients(qp.first);
                    Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                    for (size_t i = 1; i < cbs; i++)
                        grad += vel_cell_dofs_n(i) * t_dphi[i].block(0, 0, 2, 2);

                    Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                    H1_error += qp.second * inner_product(grad_diff, grad_diff);


/* Compute L2-error */
                    auto t_phi = cb.eval_basis(qp.first);
                    auto v = t_phi.transpose() * vel_cell_dofs_n;
                    Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                    L2_error += qp.second * sol_diff.dot(sol_diff);

                    uT1_gp->add_data(qp.first, v(0));
                    uT2_gp->add_data(qp.first, v(1));

/* L2 - pressure - error */
                    auto p_phi = pb.eval_basis(qp.first);
                    RealType p_num = p_phi.dot(P_locdata_n);
                    RealType p_diff = test_case.sol_p(qp.first) - p_num;
                    auto p_prova = test_case.sol_p(qp.first);
//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
                    L2_pressure_error += qp.second * p_diff * p_diff;

                    p_gp->add_data(qp.first, p_num);


                }

                auto qps_p = integrate(msh, cl, 2 * hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
                for (auto &qp: qps_p) {
/* Compute H1-error */
                    auto t_dphi = cb.eval_gradients(qp.first);
                    Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                    for (size_t i = 1; i < cbs; i++)
                        grad += vel_cell_dofs_p(i) * t_dphi[i].block(0, 0, 2, 2);

                    Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                    H1_error += qp.second * inner_product(grad_diff, grad_diff);

/* Compute L2-error */
                    auto t_phi = cb.eval_basis(qp.first);
                    auto v = t_phi.transpose() * vel_cell_dofs_p;
                    Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                    L2_error += qp.second * sol_diff.dot(sol_diff);

                    uT1_gp->add_data(qp.first, v(0));
                    uT2_gp->add_data(qp.first, v(1));

/* L2 - pressure - error */
                    auto p_phi = pb.eval_basis(qp.first);
                    RealType p_num = p_phi.dot(P_locdata_p);
                    RealType p_diff = test_case.sol_p(qp.first) - p_num;
                    auto p_prova = test_case.sol_p(qp.first);
//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
                    L2_pressure_error += qp.second * p_diff * p_diff;

                    p_gp->add_data(qp.first, p_num);
                }

                if (normal_analysis) {
                    for (auto &interface_point: cl.user_data.interface) {
                        auto t_phi = cb.eval_basis(interface_point);
                        auto v = t_phi.transpose() * vel_cell_dofs_p;
                        auto n = level_set_function.normal(interface_point);
                        auto v_n = v.dot(n);
                        l1_u_n_error += std::abs(v_n);
                        l2_u_n_error += pow(v_n, 2.0);
                        linf_u_n_error = std::max(linf_u_n_error, std::abs(v_n));
                        counter_interface_pts++;
                    }

                    for (auto interface_point = cl.user_data.interface.begin();
                         interface_point < cl.user_data.interface.end() - 1; interface_point++) {
                        RealType segment = (*(interface_point + 1) - *interface_point).to_vector().norm();
                        auto t_phi0 = cb.eval_basis(*interface_point);
                        auto v0 = t_phi0.transpose() * vel_cell_dofs_p;
                        auto n0 = level_set_function.normal(*interface_point);
                        auto v_n0 = v0.dot(n0);

                        auto t_phi1 = cb.eval_basis(*(interface_point + 1));
                        auto v1 = t_phi1.transpose() * vel_cell_dofs_p;
                        auto n1 = level_set_function.normal(*(interface_point + 1));
                        auto v_n1 = v1.dot(n1);

                        flux_interface += segment * 0.5 * (v_n0 + v_n1);

                    }

                    auto qps = integrate(msh, cl, degree + 1, element_location::IN_NEGATIVE_SIDE);
                    RealType partial_area = measure(msh, cl, element_location::IN_NEGATIVE_SIDE);
                    area_fin += partial_area;

                    for (auto &qp: qps) {
                        auto t_phi = cb.eval_basis(qp.first);
                        auto v = t_phi.transpose() * vel_cell_dofs_n;
                        rise_vel0 += qp.second * v[0];
                        rise_vel1 += qp.second * v[1];
                    }


                }

            } else {
                if (sc) {
                    vel_locdata = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                    P_locdata = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                } else {
                    vel_locdata = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                    P_locdata = assembler.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                }
                vel_cell_dofs = vel_locdata.head(cbs);


//---------------------- Updating velocity field by STE ----------------------

                if (level_set_function.subcells.size() < 1) // NOT AGGLO CELL
                {
                    assert(level_set_function.agglo_LS_cl.user_data.offset_subcells.size() == 2);
                    assert(level_set_function.agglo_LS_cl.user_data.offset_subcells[0] ==
                           level_set_function.agglo_LS_cl.user_data.offset_subcells[1]);
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType, Mesh>(velocity.msh, cl_old,
                                                                                               velocity.degree_FEM);
                    size_t i_local = 0;
                    for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                        auto phi_HHO = cb.eval_basis(ln_Qk);
                        auto vel = phi_HHO.transpose() * vel_cell_dofs;
                        velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local, offset_old) = vel(1);
                        i_local++;

                    }

                } else // AGGLO CELL
                {
                    for (size_t i_subcell = 0;
                         i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size(); i_subcell++) {
                        auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];

                        auto cl_old = velocity.msh.cells[offset_old];
                        auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType, Mesh>(velocity.msh, cl_old,
                                                                                                   velocity.degree_FEM);
                        size_t i_local = 0;
                        for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs;
                            velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local, offset_old) = vel(1);
                            i_local++;

                        }

                    }
                }


                auto qps = integrate(msh, cl, 2 * hdi.cell_degree());
                for (auto &qp: qps) {
/* Compute H1-error */
                    auto t_dphi = cb.eval_gradients(qp.first);
                    Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                    for (size_t i = 1; i < cbs; i++)
                        grad += vel_cell_dofs(i) * t_dphi[i].block(0, 0, 2, 2);

                    Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                    H1_error += qp.second * inner_product(grad_diff, grad_diff);

/* Compute L2-error */
                    auto t_phi = cb.eval_basis(qp.first);
                    auto v = t_phi.transpose() * vel_cell_dofs;
                    Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                    L2_error += qp.second * sol_diff.dot(sol_diff);

                    uT1_gp->add_data(qp.first, v(0));
                    uT2_gp->add_data(qp.first, v(1));

/* L2 - pressure - error */
                    auto p_phi = pb.eval_basis(qp.first);
                    RealType p_num = p_phi.dot(P_locdata);
                    RealType p_diff = test_case.sol_p(qp.first) - p_num;
                    auto p_prova = test_case.sol_p(qp.first);
//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
                    L2_pressure_error += qp.second * p_diff * p_diff;

                    p_gp->add_data(qp.first, p_num);
                }

                if (normal_analysis) {
                    auto qps = integrate(msh, cl, degree + 1, element_location::IN_NEGATIVE_SIDE);

                    RealType partial_area = measure(msh, cl, element_location::IN_NEGATIVE_SIDE);
                    area_fin += partial_area;

                    for (auto &qp: qps) {
                        auto t_phi = cb.eval_basis(qp.first);
                        auto v = t_phi.transpose() * vel_cell_dofs;
                        rise_vel0 += qp.second * v[0];
                        rise_vel1 += qp.second * v[1];
                    }
                }
            }

        }



//postoutput.add_object(uT1_gp);
//postoutput.add_object(uT2_gp);
//postoutput.add_object(p_gp);
//postoutput.write();



        stokes_test_info <RealType> TI;
        TI.H1_vel = std::sqrt(H1_error);
        TI.L2_vel = std::sqrt(L2_error);
        TI.L2_p = std::sqrt(L2_pressure_error);

        if (normal_analysis) {
            TI.l1_normal_vel = l1_u_n_error / counter_interface_pts;
            TI.l2_normal_vel = std::sqrt(l2_u_n_error / counter_interface_pts);
            TI.linf_normal_vel = linf_u_n_error;
        }
        std::cout << "Error H1(u) = " << TI.H1_vel << " , error L2(u) = " << TI.L2_vel << " , error L2(p) = " << TI.L2_p
                  << "." << '\n' << "Error l2(u*n) = " << TI.l2_normal_vel << " , error linf(u*n) = " << TI.linf_normal_vel
                  << "Flux interface = " << flux_interface << std::endl;

        std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
        std::cout << bold << green << "L2-norm absolute error:               " << std::sqrt(L2_error) << std::endl;
        std::cout << bold << green << "Pressure L2-norm absolute error:      " << std::sqrt(L2_pressure_error) << std::endl;
        std::cout << bold << green << "l1-norm u*n error:               " << TI.l1_normal_vel << std::endl;
        std::cout << bold << green << "l2-norm u*n error:               " << TI.l2_normal_vel << std::endl;
        std::cout << bold << green << "linf-norm u*n error:               " << TI.linf_normal_vel << std::endl;
        std::cout << bold << green << "Flux interface:               " << flux_interface << std::endl;
        std::cout << bold << green << "Rise velocity x :               " << rise_vel0 / area_fin << std::endl;
        std::cout << bold << green << "Rise velocity y :               " << rise_vel1 / area_fin << std::endl;
        std::cout << bold << green << "|Rise velocity| :               "
                  << std::abs(rise_vel0 / area_fin) + std::abs(rise_vel1 / area_fin) << std::endl;
        std::cout << bold << green << "Rise velocity :               " << rise_vel0 / area_fin + rise_vel1 / area_fin
                  << std::endl;

        if (0) {
/////////////// compute condition number
            SparseMatrix <RealType> Mat;
// Matrix<RealType, Dynamic, Dynamic> Mat;
            if (sc)
                Mat = assembler_sc.LHS;
            else
                Mat = assembler.LHS;


// Add by Stefano
            Eigen::BDCSVD <Eigen::MatrixXd> SVD(Mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
            double cond = SVD.singularValues()(0) / SVD.singularValues()(SVD.singularValues().size() - 1);
            std::cout << "cond_numb = " << cond << std::endl;

/*
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
        */
            TI.cond = cond;
//std::cout << "sigma_max = " << sigma_max << "   sigma_min = "
//         << sigma_min << "  cond = " << cond
//         << std::endl;
        } else
            TI.cond = 0.0;

        tc.toc();
        std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


        return TI;
    }


    template<typename Mesh, typename testType, typename meth, typename Fonction, typename Velocity>
    stokes_test_info<typename Mesh::coordinate_type>
    run_cuthho_interface_velocity_new_post_processingLS(const Mesh &msh, size_t degree, meth &method, testType &test_case,
                                                        Fonction &level_set_function, Velocity &velocity, bool sym_grad,
                                                        size_t time, int time_gap) {
        using RealType = typename Mesh::coordinate_type;

        auto iso_val_interface = level_set_function.iso_val_interface;


        auto bcs_vel = test_case.bcs_vel;


        timecounter tc;

        bool sc = true;  // static condensation


// ************** ASSEMBLE PROBLEM **************
        hho_degree_info hdi(degree + 1, degree);

        tc.tic();


        auto assembler = make_stokes_interface_assembler(msh, bcs_vel, hdi);

        auto assembler_sc = make_stokes_interface_condensed_assembler(msh, bcs_vel, hdi);

// IT MAY GO INTO THE LOOP ( IF YES ADD ALSO IN THE POST-PROCESSING LOOP )
        assembler_sc.set_dir_func(bcs_vel); // DOVE VA? INTO LOOP cl? SE CAMBIASSE bcs_vel in spazio forse si!

        test_case.test_case_mesh_assignment(msh);

        for (auto &cl: msh.cells) {

//std::cout<<yellow<<bold<<"CELL = "<<offset(msh,cl) <<reset<<std::endl;
            test_case.test_case_cell_assignment(cl);
            auto contrib = method.make_contrib(msh, cl, test_case, hdi);
            auto lc = contrib.first;
            auto f = contrib.second;

            if (sc)
                assembler_sc.assemble(msh, cl, lc, f);
            else
                assembler.assemble(msh, cl, lc, f);

        }


        if (sc)
            assembler_sc.finalize();
        else
            assembler.finalize();


        tc.toc();
        std::cout << "Matrix assembly: " << tc << " seconds" << std::endl;

        if (sc)
            std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
        else
            std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

        std::cout << "Cells: " << msh.cells.size() << std::endl;
        std::cout << "Faces: " << msh.faces.size() << std::endl;

// ************** SOLVE **************
        tc.tic();
#if 1
        SparseLU <SparseMatrix<RealType>> solver;
        Matrix<RealType, Dynamic, 1> sol;

        if (sc) {
            std::cout << "First step: analyze pattern... " << std::endl;
            solver.analyzePattern(assembler_sc.LHS);
            std::cout << "Pattern ok. Second step: assembling... " << std::endl;
            solver.factorize(assembler_sc.LHS);
            std::cout << "Assembling ok. Third step: solving... " << std::endl;
            sol = solver.solve(assembler_sc.RHS);
            std::cout << "..problem solved. " << std::endl;
        } else {
            solver.analyzePattern(assembler.LHS);
            solver.factorize(assembler.LHS);
            sol = solver.solve(assembler.RHS);
        }
#endif
#if 0
        Matrix<RealType, Dynamic, 1> sol;
        cg_params <RealType> cgp;
        cgp.histfile = "cuthho_cg_hist.dat";
        cgp.verbose = true;
        cgp.apply_preconditioner = true;
        if (sc) {
            sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
            cgp.max_iter = assembler_sc.LHS.rows();
//        conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);


            ConjugateGradient < SparseMatrix < RealType > , Lower | Upper > cg;
            cg.compute(assembler_sc.LHS);
            sol = cg.solve(assembler_sc.RHS);
            std::cout << "#iterations:     " << cg.iterations() << std::endl;
            std::cout << "estimated error: " << cg.error() << std::endl;
// conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);

        } else {
            sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
            cgp.max_iter = assembler.LHS.rows();
            conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
        }
#endif
        tc.toc();
        std::cout << "Linear solver: " << tc << " seconds" << std::endl;

// ************** POSTPROCESS **************

        std::string path = "simulation_1x1_/";
        postprocess_output <RealType> postoutput;
        std::string filename_interface_uT = path + "interface_uT_" + std::to_string(time) + ".3D";
        std::ofstream interface_file(filename_interface_uT, std::ios::out | std::ios::trunc);

        if (interface_file) {
// instructions
            interface_file << "X   Y   val0   val1" << std::endl;
        } else
            std::cerr << "Interface_file has not been opened" << std::endl;


        auto uT1_gp = std::make_shared < gnuplot_output_object < RealType > >
                      (path + "interface_uT1_" + std::to_string(time) + ".dat");
        auto uT2_gp = std::make_shared < gnuplot_output_object < RealType > >
                      (path + "interface_uT2_" + std::to_string(time) + ".dat");
//    auto uT_gp  = std::make_shared< gnuplot_output_object_vec<RealType> >("interface_uT.dat");

//    std::string filename_pressure = "interface_p_" + std::to_string(time) + ".dat";
        auto p_gp = std::make_shared < gnuplot_output_object < RealType > >
                    (path + "interface_p_" + std::to_string(time) + ".dat");
        auto p1_gp = std::make_shared < gnuplot_output_object < RealType > >
                     (path + "interface_pIN_" + std::to_string(time) + ".dat");
        auto p2_gp = std::make_shared < gnuplot_output_object < RealType > >
                     (path + "interface_pOUT_" + std::to_string(time) + ".dat");
//auto p_gp    = std::make_shared< gnuplot_output_object<RealType> >(filename_pressure);

        std::string filename_gammaH = path + "gamma_H_" + std::to_string(time) + ".dat";
        auto test_gammaH = std::make_shared < gnuplot_output_object < double > > (filename_gammaH);

        std::string filename_press_jump = path + "pressure_jump_" + std::to_string(time) + ".dat";
        auto test_press_jump = std::make_shared < gnuplot_output_object < double > > (filename_press_jump);

        std::string filename_grad_vel_jump = path + "grad_vel_jump_" + std::to_string(time) + ".dat";
        auto test_grad_vel_jump = std::make_shared < gnuplot_output_object < double > > (filename_grad_vel_jump);


        std::string filename_grad_vel_t_n = path + "grad_vel_t_n_" + std::to_string(time) + ".dat";
        auto test_grad_vel_t_n = std::make_shared < gnuplot_output_object < double > > (filename_grad_vel_t_n);

        std::string filename_vel_n = path + "vel_u_n_" + std::to_string(time) + ".dat";
        auto test_vel_n = std::make_shared < gnuplot_output_object < double > > (filename_vel_n);
//    auto force_pressure    = std::make_shared< gnuplot_output_object<RealType> >("pressure_force.dat");

        RealType pos = 0.0;
        RealType force_pressure_avg = 0.0, force_pressure_max = 0.0;
        RealType force_gradVel_avg = 0.0, force_gradVel_max = 0.0;

        size_t counter_pt_Gamma = 0;


        tc.tic();
        RealType H1_error = 0.0;
        RealType L2_error = 0.0;
        RealType L2_pressure_error = 0.0;
        RealType l1_u_n_error = 0.0;
        RealType l2_u_n_error = 0.0;
        RealType linf_u_n_error = 0.0;
        size_t counter_interface_pts = 0;
        RealType distance_pts = 0.0;

//    timecounter tc1;
//    tc1.tic();
//    size_t i_global = 0 ;
        auto hdi_cell = hdi.cell_degree();
        auto hdi_face = hdi.face_degree();
        auto msh_vel = velocity.msh;
        auto degree_vel = velocity.degree_FEM;

        timecounter tc_p1;

        tc_p1.tic();

        std::vector <size_t> cut_cell_cointainer, uncut_cell_cointainer;

        for (auto &cl: msh.cells) {
            if (location(msh, cl) == element_location::ON_INTERFACE)
                cut_cell_cointainer.push_back(offset(msh, cl));

            else
                uncut_cell_cointainer.push_back(offset(msh, cl));


        }


        point<RealType, 2> first_point;
        point<RealType, 2> cell_end_point;
        bool first_cut_cell_found = FALSE;

//    size_t deg_size = 2; // ESEMPIOOO
        auto sol_vel = test_case.sol_vel;
        auto sol_p = test_case.sol_p;
        auto vel_grad = test_case.vel_grad;


        size_t cl_i = 0;
        while (cut_cell_cointainer.size() > 0) {
            if (cl_i > cut_cell_cointainer.size() - 1)
                std::cout << "stop: first_pt = " << first_point << " pt_to_find = " << cell_end_point << std::endl;
            size_t k_offset = cut_cell_cointainer[cl_i];
            auto cl = msh.cells[k_offset];
//        auto msh_int = cl.user_data.integration_msh ;

            if (!first_cut_cell_found) {
                cut_cell_cointainer.erase(cut_cell_cointainer.begin()); //pop_front();

                post_processing_functionLS(msh, cl, hdi_cell, hdi_face, level_set_function, test_case, assembler_sc,
                                           bcs_vel, sol, velocity, H1_error, L2_error, uT1_gp, uT2_gp, p_gp, interface_file,
                                           L2_pressure_error, l1_u_n_error, l2_u_n_error, linf_u_n_error,
                                           counter_interface_pts, degree, force_pressure_avg, force_pressure_max,
                                           counter_pt_Gamma, test_gammaH, test_press_jump, test_grad_vel_jump, distance_pts,
                                           force_gradVel_max, force_gradVel_avg, test_vel_n, p1_gp, p2_gp,
                                           test_grad_vel_t_n, sol_vel, sol_p, vel_grad);

                first_cut_cell_found = TRUE;
                first_point = *cl.user_data.interface.begin();
                cell_end_point = *(cl.user_data.interface.end() - 1);
                cl_i = 0;
            } else if (first_cut_cell_found && cell_end_point == *cl.user_data.interface.begin() &&
                       !(first_point == cell_end_point)) {
                cut_cell_cointainer.erase(cut_cell_cointainer.begin() + cl_i);
//                cut_cell_cointainer.pop_front();

                post_processing_functionLS(msh, cl, hdi_cell, hdi_face, level_set_function, test_case, assembler_sc,
                                           bcs_vel, sol, velocity, H1_error, L2_error, uT1_gp, uT2_gp, p_gp, interface_file,
                                           L2_pressure_error, l1_u_n_error, l2_u_n_error, linf_u_n_error,
                                           counter_interface_pts, degree, force_pressure_avg, force_pressure_max,
                                           counter_pt_Gamma, test_gammaH, test_press_jump, test_grad_vel_jump, distance_pts,
                                           force_gradVel_max, force_gradVel_avg, test_vel_n, p1_gp, p2_gp,
                                           test_grad_vel_t_n, sol_vel, sol_p, vel_grad);

                cell_end_point = *(cl.user_data.interface.end() - 1);
                cl_i = 0;

            } else if (first_point == cell_end_point)
                break;
            else
                cl_i++;

        }

        std::cout << "First Point curvilinear variable: " << first_point << std::endl;
        tc_p1.toc();
        std::cout << "Interface_analysis time: " << tc_p1 << std::endl;
        tc_p1.tic();


        for (auto &i_cl: uncut_cell_cointainer) {
            auto cl = msh.cells[i_cl];
            post_processing_functionLS_fast(msh, cl, hdi_cell, hdi_face, level_set_function, test_case, assembler_sc,
                                            bcs_vel, sol, velocity, H1_error, L2_error, uT1_gp, uT2_gp, p_gp,
                                            interface_file, L2_pressure_error, l1_u_n_error, l2_u_n_error, linf_u_n_error,
                                            counter_interface_pts, degree, force_pressure_avg, force_pressure_max,
                                            counter_pt_Gamma, test_gammaH, test_press_jump, test_grad_vel_jump,
                                            distance_pts, force_gradVel_max, force_gradVel_avg, test_vel_n, p1_gp, p2_gp,
                                            test_grad_vel_t_n, sol_vel, sol_p, vel_grad);

        }

        tc_p1.toc();
        std::cout << "Not cut cell analysis time: " << tc_p1 << std::endl;


        std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
        std::cout << bold << green << "L2-norm absolute error:               " << std::sqrt(L2_error) << std::endl;
        std::cout << bold << green << "Pressure L2-norm absolute error:      " << std::sqrt(L2_pressure_error) << std::endl;
        std::cout << bold << green << "l1-norm u*n error:               " << l1_u_n_error / counter_interface_pts
                  << std::endl;
        std::cout << bold << green << "l2-norm u*n error:               " << std::sqrt(l2_u_n_error / counter_interface_pts)
                  << std::endl;
        std::cout << bold << green << "linf-norm u*n error:               " << linf_u_n_error << std::endl;


        std::cout << bold << green << "AVG force pressure = " << force_pressure_avg / counter_pt_Gamma << std::endl;
        std::cout << bold << green << "linf-norm force pressure = " << force_pressure_max << std::endl;

        std::cout << bold << green << "AVG force grad_s velocity = " << force_gradVel_avg / counter_pt_Gamma << std::endl;
        std::cout << bold << green << "linf-norm force grad_s velocity = " << force_gradVel_max << std::endl;

        if (time % time_gap == 0) {
            postoutput.add_object(uT1_gp);
            postoutput.add_object(uT2_gp);
            postoutput.add_object(p_gp);
            postoutput.add_object(p1_gp);
            postoutput.add_object(p2_gp);

            postoutput.add_object(test_gammaH);
            postoutput.add_object(test_press_jump);
            postoutput.add_object(test_grad_vel_jump);
            postoutput.add_object(test_vel_n);
            postoutput.add_object(test_grad_vel_t_n);


            postoutput.write();
            if (interface_file) {

                interface_file.close();

            }

        }

        stokes_test_info <RealType> TI;
        TI.H1_vel = std::sqrt(H1_error);
        TI.L2_vel = std::sqrt(L2_error);
        TI.L2_p = std::sqrt(L2_pressure_error);
        if (1) {
            TI.l1_normal_vel = l1_u_n_error / counter_interface_pts;
            TI.l2_normal_vel = std::sqrt(l2_u_n_error / counter_interface_pts);
            TI.linf_normal_vel = linf_u_n_error;
        }


        if (0) {
/////////////// compute condition number
            SparseMatrix <RealType> Mat;
// Matrix<RealType, Dynamic, Dynamic> Mat;
            if (sc)
                Mat = assembler_sc.LHS;
            else
                Mat = assembler.LHS;

            {
                JacobiSVD <MatrixXd> svd(Mat);
                RealType cond = svd.singularValues()(0)
                                / svd.singularValues()(svd.singularValues().size() - 1);
                std::cout << "cond numb = " << cond << std::endl;
            }

            RealType sigma_max, sigma_min;

// Construct matrix operation object using the wrapper class SparseSymMatProd
            Spectra::SparseSymMatProd <RealType> op(Mat);
// Construct eigen solver object, requesting the largest eigenvalue
            Spectra::SymEigsSolver <RealType, Spectra::LARGEST_MAGN,
            Spectra::SparseSymMatProd<RealType>> max_eigs(&op, 1, 10);
            max_eigs.init();
            max_eigs.compute();
            if (max_eigs.info() == Spectra::SUCCESSFUL)
                sigma_max = max_eigs.eigenvalues()(0);


// Construct eigen solver object, requesting the smallest eigenvalue
            Spectra::SymEigsSolver <RealType, Spectra::SMALLEST_MAGN,
            Spectra::SparseSymMatProd<RealType>> min_eigs(&op, 1, 10);

            min_eigs.init();
            min_eigs.compute();
            if (min_eigs.info() == Spectra::SUCCESSFUL)
                sigma_min = min_eigs.eigenvalues()(0);

// compute condition number
            RealType cond = sigma_max / sigma_min;
            TI.cond = cond;
            std::cout << "sigma_max = " << sigma_max << "   sigma_min = "
                      << sigma_min << "  cond = " << cond
                      << std::endl;
        } else
            TI.cond = 0.0;

        tc.toc();
        std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


        return TI;
    }





    template<typename Mesh, typename Cell, typename LS, typename TC, typename RealType, typename ASS, typename BDRY, typename SOL, typename VEL, typename PP1, typename PP2, typename Function1, typename Function2, typename Function3>
    void
    post_processing_functionLS_fast_2(const Mesh &msh, Cell &cl, size_t hdi_cell, size_t hdi_face,
                                      LS &level_set_function, TC &test_case, ASS &assembler_sc, BDRY &bcs_vel,
                                      const SOL &sol, VEL &velocity, RealType &H1_error, RealType &L2_error, PP1 &uT1_gp,
                                      PP1 &uT2_gp,
                                      PP1 &p_gp, PP2 &interface_file, RealType &L2_pressure_error, RealType &l1_u_n_error,
                                      RealType &l2_u_n_error, RealType &linf_u_n_error, size_t &counter_interface_pts,
                                      size_t &degree,
                                      RealType &force_pressure_avg, RealType &force_pressure_max, size_t &counter_pt_Gamma,
                                      PP1 &test_gammaH,
                                      PP1 &test_press_jump, PP1 &test_grad_vel_jump, RealType &distance_pts,
                                      RealType &force_gradVel_max,
                                      RealType &force_gradVel_avg, PP1 &p1_gp, PP1 &p2_gp,
                                      Function1 &sol_vel, Function2 &sol_p, Function3 &vel_grad) {


        vector_cell_basis <cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi_cell);
        RealType kappa_1 = test_case.parms.kappa_1;
        RealType kappa_2 = test_case.parms.kappa_2;

        cell_basis <cuthho_poly_mesh<RealType>, RealType> pb(msh, cl, hdi_face);
        auto cbs = cb.size();
        auto pbs = pb.size();


        level_set_function.cell_assignment(cl); // ----------------------------> TOGLIERE????
        test_case.test_case_cell_assignment(cl); // ----------------------------> TOGLIERE????

//    auto sol_vel = test_case.sol_vel;
//    auto sol_p = test_case.sol_p;
//    auto vel_grad = test_case.vel_grad;

        assembler_sc.set_dir_func(bcs_vel); // CAMBIA QUALCOSA?? // ----------------------------> TOGLIERE????


        Matrix<RealType, Dynamic, 1> vel_locdata_n, vel_locdata_p, vel_locdata;
        Matrix<RealType, Dynamic, 1> P_locdata_n, P_locdata_p, P_locdata;
        Matrix<RealType, Dynamic, 1> vel_cell_dofs_n, vel_cell_dofs_p, vel_cell_dofs;


        vel_locdata = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
        P_locdata = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);

        vel_cell_dofs = vel_locdata.head(cbs);



// NOT AGGLO CELL
        if (level_set_function.subcells.size() < 1) {
//                assert(level_set_function.agglo_LS_cl.user_data.offset_subcells.size()==2);
//                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
            auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = velocity.msh.cells[offset_old];
            auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
            size_t i_local = 0;
            for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                auto phi_HHO = cb.eval_basis(ln_Qk);
                auto vel = phi_HHO.transpose() * vel_cell_dofs;
// velocity.sol_HHO.first(i_local,offset_old) = vel(0);
// velocity.sol_HHO.second(i_local,offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                i_local++;

            }

        } else // AGGLO CELL
        {
            for (size_t i_subcell = 0;
                 i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size(); i_subcell++) {
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
//std::cout<<"offset_old = "<<offset_old<<std::endl;
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                    auto phi_HHO = cb.eval_basis(ln_Qk);
                    auto vel = phi_HHO.transpose() * vel_cell_dofs;
// velocity.sol_HHO.first(i_local,offset_old) = vel(0);
// velocity.sol_HHO.second(i_local,offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                    i_local++;

                }

            }
        }

        RealType kappa = test_case.parms.kappa_1;
        if (location(msh, cl) == element_location::IN_POSITIVE_SIDE)
            kappa = test_case.parms.kappa_2;

        auto qps = integrate(msh, cl, 2 * hdi_cell);
        for (auto &qp: qps) {
// Compute H1-error //
            auto t_dphi = cb.eval_gradients(qp.first);
            Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

            for (size_t i = 1; i < cbs; i++)
                grad += vel_cell_dofs(i) * t_dphi[i].block(0, 0, 2, 2);

            Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
            Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
            H1_error += qp.second * kappa * inner_product(grad_sym_diff, grad_sym_diff);

// Compute L2-error //
            auto t_phi = cb.eval_basis(qp.first);
            auto v = t_phi.transpose() * vel_cell_dofs;
            Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;

            L2_error += qp.second * kappa * sol_diff.dot(sol_diff);

            uT1_gp->add_data(qp.first, v(0));
            uT2_gp->add_data(qp.first, v(1));
            if (interface_file) {

                interface_file << qp.first.x() << "   " << qp.first.y() << "   " << v(0) << "   " << v(1) << std::endl;

            }

// L2 - pressure - error //
            auto p_phi = pb.eval_basis(qp.first);
            RealType p_num = p_phi.dot(P_locdata);
            RealType p_diff = test_case.sol_p(qp.first) - p_num; // era test_case STE

            L2_pressure_error += qp.second * p_diff * p_diff / kappa;

            p_gp->add_data(qp.first, p_num);
            if (level_set_function(qp.first, cl) > 0.0)
                p2_gp->add_data(qp.first, p_num);
            else
                p1_gp->add_data(qp.first, p_num);
        }


    }


    template<typename Mesh, typename Cell, typename LS, typename TC, typename RealType, typename ASS, typename BDRY, typename SOL, typename VEL, typename PP1, typename PP2, typename Function1, typename Function2, typename Function3>
    void
    post_processing_functionLS_2(const Mesh &msh, Cell &cl, size_t hdi_cell, size_t hdi_face,
                                 LS &level_set_function, TC &test_case, ASS &assembler_sc, BDRY &bcs_vel,
                                 const SOL &sol, VEL &velocity, RealType &H1_error, RealType &L2_error, PP1 &uT1_gp,
                                 PP1 &uT2_gp, PP1 &p_gp, PP2 &interface_file, RealType &L2_pressure_error,
                                 RealType &l1_u_n_error,
                                 RealType &l2_u_n_error, RealType &linf_u_n_error, size_t &counter_interface_pts,
                                 size_t &degree,
                                 RealType &force_pressure_avg, RealType &force_pressure_max, size_t &counter_pt_Gamma,
                                 PP1 &test_gammaH,
                                 PP1 &test_press_jump, PP1 &test_grad_vel_jump, RealType &distance_pts,
                                 RealType &force_gradVel_max,
                                 RealType &force_gradVel_avg, PP1 &test_veln_n, PP1 &p1_gp, PP1 &p2_gp,
                                 PP1 &test_grad_vel_t_n,
                                 Function1 &sol_vel, Function2 &sol_p, Function3 &vel_grad, PP1 &test_veln_t,
                                 PP1 &test_velp_n, PP1 &test_velp_t) {


        vector_cell_basis <cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi_cell);
        RealType kappa_1 = test_case.parms.kappa_1;
        RealType kappa_2 = test_case.parms.kappa_2;

        cell_basis <cuthho_poly_mesh<RealType>, RealType> pb(msh, cl, hdi_face);
        auto cbs = cb.size();
//auto pbs = pb.size();
//    auto sol_vel = test_case.sol_vel;
//    auto sol_p = test_case.sol_p;
//    auto vel_grad = test_case.vel_grad;

        level_set_function.cell_assignment(cl); // ----------------------------> TOGLIERE????
        test_case.test_case_cell_assignment(cl); // ----------------------------> TOGLIERE????


//auto bcs_vel = test_case.bcs_vel;
//auto neumann_jump = test_case.neumann_jump;
        assembler_sc.set_dir_func(bcs_vel); // CAMBIA QUALCOSA?? // ----------------------------> TOGLIERE????


        Matrix<RealType, Dynamic, 1> vel_locdata_n, vel_locdata_p, vel_locdata;
        Matrix<RealType, Dynamic, 1> P_locdata_n, P_locdata_p, P_locdata;
        Matrix<RealType, Dynamic, 1> vel_cell_dofs_n, vel_cell_dofs_p, vel_cell_dofs;

        if (location(msh, cl) == element_location::ON_INTERFACE) {
            vel_locdata_n = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
            vel_locdata_p = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            P_locdata_n = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
            P_locdata_p = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);


            vel_cell_dofs_n = vel_locdata_n.head(cbs);
            vel_cell_dofs_p = vel_locdata_p.head(cbs);


// NOT AGGLO CELL
            if (level_set_function.subcells.size() < 1) {

                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
                size_t i_local = 0;
                for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                    if (level_set_function(ln_Qk, cl_old) > 0.0) {
                        auto phi_HHO = cb.eval_basis(ln_Qk);
                        auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
// velocity.sol_HHO.first(i_local,offset_old) = vel(0);
// velocity.sol_HHO.second(i_local,offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
                    } else {
                        auto phi_HHO = cb.eval_basis(ln_Qk);
                        auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
// velocity.sol_HHO.first(i_local,offset_old) = vel(0);
// velocity.sol_HHO.second(i_local,offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
//  velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
//  velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                    }
                }

            } else // AGGLO CELL
            {

                for (size_t i_subcell = 0;
                     i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size(); i_subcell++) {
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
//std::cout<<"offset_old = "<<offset_old<<std::endl;
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                    size_t i_local = 0;
                    for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                        if (level_set_function(ln_Qk, cl_old) > 0.0) {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
// velocity.sol_HHO.first(i_local,offset_old) = vel(0);
// velocity.sol_HHO.second(i_local,offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
                        } else {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
// velocity.sol_HHO.first(i_local,offset_old) = vel(0);
// velocity.sol_HHO.second(i_local,offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
//  velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
//  velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                        }
                    }

                }
            }

//            tc2.toc();
//            std::cout<<"time tc2 3 = "<<tc2<<std::endl;
//            tc2.tic();
            auto qps_n = integrate(msh, cl, 2 * hdi_cell, element_location::IN_NEGATIVE_SIDE);
            for (auto &qp: qps_n) {
// Compute H1-error //
                auto t_dphi = cb.eval_gradients(qp.first);
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++)
                    grad += vel_cell_dofs_n(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                H1_error += qp.second * kappa_1 * inner_product(grad_sym_diff, grad_sym_diff);

// Compute L2-error //
                auto t_phi = cb.eval_basis(qp.first);
                auto v = t_phi.transpose() * vel_cell_dofs_n;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                L2_error += qp.second * kappa_1 * sol_diff.dot(sol_diff);

                uT1_gp->add_data(qp.first, v(0));
                uT2_gp->add_data(qp.first, v(1));

                interface_file << qp.first.x() << "   " << qp.first.y() << "   " << v(0) << "   " << v(1) << std::endl;



// L2 - pressure - error //
                auto p_phi = pb.eval_basis(qp.first);
                RealType p_num = p_phi.dot(P_locdata_n);
                RealType p_diff = test_case.sol_p(qp.first) - p_num; // era test_case STE
//                auto p_prova = test_case.sol_p( qp.first ) ;
//                std::cout<<"In pt = "<<qp.first<<" --> pressure ANAL  = "<<p_prova<<" , pressure NUM = "<< p_num<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                L2_pressure_error += qp.second * p_diff * p_diff / kappa_1;
                p_gp->add_data(qp.first, p_num);
                p1_gp->add_data(qp.first, p_num);
            }

            auto qps_p = integrate(msh, cl, 2 * hdi_cell, element_location::IN_POSITIVE_SIDE);
            for (auto &qp: qps_p) {
// Compute H1-error //
                auto t_dphi = cb.eval_gradients(qp.first);
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++)
                    grad += vel_cell_dofs_p(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                H1_error += qp.second * kappa_2 * inner_product(grad_sym_diff, grad_sym_diff);

// Compute L2-error //
                auto t_phi = cb.eval_basis(qp.first);
                auto v = t_phi.transpose() * vel_cell_dofs_p;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                L2_error += qp.second * kappa_2 * sol_diff.dot(sol_diff);

                uT1_gp->add_data(qp.first, v(0));
                uT2_gp->add_data(qp.first, v(1));
//                uT_gp->add_data( qp.first, std::make_pair(v(0),v(1)) );
                if (interface_file) {

                    interface_file << qp.first.x() << "   " << qp.first.y() << "   " << v(0) << "   " << v(1) << std::endl;

                }
// L2 - pressure - error //
                auto p_phi = pb.eval_basis(qp.first);
                RealType p_num = p_phi.dot(P_locdata_p);
                RealType p_diff = test_case.sol_p(qp.first) - p_num; // era test_case STE
//auto p_prova = test_case.sol_p( qp.first ) ;
//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                L2_pressure_error += qp.second * p_diff * p_diff / kappa_2;

                p_gp->add_data(qp.first, p_num);
                p2_gp->add_data(qp.first, p_num);
            }
            if (1) {
                for (auto &interface_point: cl.user_data.interface) {
                    auto t_phi = cb.eval_basis(interface_point);
                    auto v = t_phi.transpose() * vel_cell_dofs_p;
                    auto n = level_set_function.normal(interface_point);
                    auto v_n = v.dot(n);
                    l1_u_n_error += std::abs(v_n);
                    l2_u_n_error += pow(v_n, 2.0);
                    linf_u_n_error = std::max(linf_u_n_error, std::abs(v_n));
                    counter_interface_pts++;
                }
            }
//            tc2.toc();
//            std::cout<<"time tc2 2 = "<<tc2<<std::endl;

            if (1) // analysis power of pressure
            {
                auto parametric_interface = test_case.parametric_interface;
                auto gamma = test_case.gamma;
                auto msh_int = cl.user_data.integration_msh;
                auto global_cells_i = parametric_interface.get_global_cells_interface(msh, cl);
                Matrix<RealType, 2, 1> phi_t;
                size_t degree_curve = msh_int.degree_curve;
                RealType tot = 10.0;
//                Interface_parametrisation_mesh1d curve(degree_curve) ;
//            degree += 3*degree_curve -4 ; // 2*degree_curve ; // TO BE CHECKED
//            auto qps = edge_quadrature<RealType>(degree);
                auto neumann = test_case.neumann_jump;
                for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++) {
                    auto pts = points(msh_int, msh_int.cells[i_cell]);
                    size_t global_cl_i = global_cells_i[i_cell];
//                auto qp_old = 0.5 *(*(qps.begin())).first.x() + 0.5;
//                auto p = parametric_interface(t , pts , degree_curve ) ;
//                point<RealType,2> pt_old ; //= typename Mesh::point_type( p(0) , p(1) ) ;

                    for (RealType i = 0.0; i <= tot; i++) {
                        auto t = 0.0 + i / tot;
                        auto p = parametric_interface(t, pts, degree_curve);
                        point<RealType, 2> pt = typename Mesh::point_type(p(0), p(1));
//                    if( t == 0.0 )
//                        pt_old = pt;
                        auto p_phi = pb.eval_basis(pt);
                        RealType p_pos = p_phi.dot(P_locdata_p);
                        RealType p_neg = p_phi.dot(P_locdata_n);


                        auto phi_HHO = cb.eval_basis(pt);
                        auto t_dphi = cb.eval_gradients(pt);
                        Matrix<RealType, 2, 2> grad_p = Matrix<RealType, 2, 2>::Zero();
                        Matrix<RealType, 2, 2> grad_n = Matrix<RealType, 2, 2>::Zero();

                        for (size_t i = 1; i < cbs; i++) {
                            grad_p += vel_cell_dofs_p(i) * t_dphi[i].block(0, 0, 2, 2);
                            grad_n += vel_cell_dofs_n(i) * t_dphi[i].block(0, 0, 2, 2);

                        }
                        Matrix<RealType, 2, 2> grad_sym_p = 0.5 * (grad_p + grad_p.transpose());
                        Matrix<RealType, 2, 2> grad_sym_n = 0.5 * (grad_n + grad_n.transpose());
                        auto vel_n = phi_HHO.transpose() * vel_cell_dofs_n;

                        auto vel_p = phi_HHO.transpose() * vel_cell_dofs_p; // added 26/07/2023

                        Matrix<RealType, 2, 1> phi_n = level_set_function.normal(pt);

                        RealType val_un_n = (vel_n).transpose() * (phi_n);
                        RealType val_up_n = (vel_p).transpose() * (phi_n);


                        auto val_p = (p_pos - p_neg);
                        Matrix<RealType, 2, 1> val_grad_u_n =
                                (2.0 * kappa_1 * grad_sym_n - 2.0 * kappa_2 * grad_sym_p) * (phi_n);
                        RealType val_u = (phi_n.transpose()) * val_grad_u_n;
                        phi_t(0) = -phi_n(1);
                        phi_t(1) = phi_n(0);

                        RealType val_un_t = (vel_n).transpose() * (phi_t);
                        RealType val_up_t = (vel_p).transpose() * (phi_t);

                        RealType t_val_u_n = (phi_t.transpose()) * val_grad_u_n;
//                    RealType val_u ;
//                    if( signbit(phi_n(0)) == signbit(grads_u_n(0)) && signbit(phi_n(1)) == signbit(grads_u_n(1)) )
//                        val_u = grads_u_n.norm();
//                    else
//                        val_u = -grads_u_n.norm();
                        point<RealType, 2> curv_var = typename Mesh::point_type(distance_pts, 0.0);
                        auto val_H = gamma * level_set_function.divergence(pt);
                        test_press_jump->add_data(curv_var, val_p);
                        test_gammaH->add_data(curv_var, val_H);
                        test_grad_vel_jump->add_data(curv_var, val_u);
                        test_grad_vel_t_n->add_data(curv_var, t_val_u_n);

                        test_veln_n->add_data(curv_var, val_un_n);
                        test_veln_t->add_data(curv_var, val_un_t);

                        test_velp_n->add_data(curv_var, val_up_n);
                        test_velp_t->add_data(curv_var, val_up_t);

                        force_pressure_avg += val_p / val_H;
                        force_pressure_max = std::max(force_pressure_max, std::abs(val_p / val_H));

                        force_gradVel_avg += val_u / val_H;
                        force_gradVel_max = std::max(force_gradVel_max, std::abs(val_u / val_H));


                        counter_pt_Gamma++;

                        RealType dist;

                        if (t == 1)
                            dist = 0.0;
                        else
                            dist = (parametric_interface(t + 1.0 / tot, pts, degree_curve) - p).norm();

                        distance_pts += dist;

                    }
                }


            }

        } else {
//            tc2.tic();
            vel_locdata = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            P_locdata = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);

            vel_cell_dofs = vel_locdata.head(cbs);



// NOT AGGLO CELL
            if (level_set_function.subcells.size() < 1) {
//                assert(level_set_function.agglo_LS_cl.user_data.offset_subcells.size()==2);
//                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                    auto phi_HHO = cb.eval_basis(ln_Qk);
                    auto vel = phi_HHO.transpose() * vel_cell_dofs;
// velocity.sol_HHO.first(i_local,offset_old) = vel(0);
// velocity.sol_HHO.second(i_local,offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                    i_local++;

                }

            } else // AGGLO CELL
            {
                for (size_t i_subcell = 0;
                     i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size(); i_subcell++) {
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
//std::cout<<"offset_old = "<<offset_old<<std::endl;
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                    size_t i_local = 0;
                    for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                        auto phi_HHO = cb.eval_basis(ln_Qk);
                        auto vel = phi_HHO.transpose() * vel_cell_dofs;
// velocity.sol_HHO.first(i_local,offset_old) = vel(0);
// velocity.sol_HHO.second(i_local,offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;

                    }

                }
            }
//            tc2.toc();
//            std::cout<<"time tc2 1 = "<<tc2<<std::endl;
//            tc2.tic();
            RealType kappa = test_case.parms.kappa_1;
            if (location(msh, cl) == element_location::IN_POSITIVE_SIDE)
                kappa = test_case.parms.kappa_2;

            auto qps = integrate(msh, cl, 2 * hdi_cell);
            for (auto &qp: qps) {
// Compute H1-error //
                auto t_dphi = cb.eval_gradients(qp.first);
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++)
                    grad += vel_cell_dofs(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                H1_error += qp.second * kappa * inner_product(grad_sym_diff, grad_sym_diff);

// Compute L2-error //
                auto t_phi = cb.eval_basis(qp.first);
                auto v = t_phi.transpose() * vel_cell_dofs;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                L2_error += qp.second * kappa * sol_diff.dot(sol_diff);

                uT1_gp->add_data(qp.first, v(0));
                uT2_gp->add_data(qp.first, v(1));
//                uT_gp->add_data( qp.first, std::make_pair(v(0),v(1)) );
                if (interface_file) {

                    interface_file << qp.first.x() << "   " << qp.first.y() << "   " << v(0) << "   " << v(1) << std::endl;

                }

// L2 - pressure - error //
                auto p_phi = pb.eval_basis(qp.first);
                RealType p_num = p_phi.dot(P_locdata);
                RealType p_diff = test_case.sol_p(qp.first) - p_num; // era test_case STE
//auto p_prova = test_case.sol_p( qp.first ) ;
//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                L2_pressure_error += qp.second * p_diff * p_diff / kappa;

                p_gp->add_data(qp.first, p_num);
                if (level_set_function(qp.first, cl) > 0.0)
                    p2_gp->add_data(qp.first, p_num);
                else
                    p1_gp->add_data(qp.first, p_num);
            }
//            tc2.toc();
//            std::cout<<"time tc2 0 = "<<tc2<<std::endl;
        }


    }


    template<typename Mesh, typename Cell, typename LS, typename TC, typename RealType, typename ASS, typename BDRY, typename SOL, typename VEL, typename PP1, typename PP2, typename Function1, typename Function2, typename Function3>
    void
    post_processing_functionLS(const Mesh &msh, Cell &cl, size_t hdi_cell, size_t hdi_face, LS &level_set_function,
                               TC &test_case, ASS &assembler_sc, BDRY &bcs_vel, const SOL &sol, VEL &velocity,
                               RealType &H1_error, RealType &L2_error, PP1 &uT1_gp, PP1 &uT2_gp, PP1 &p_gp,
                               PP2 &interface_file, RealType &L2_pressure_error, RealType &l1_u_n_error,
                               RealType &l2_u_n_error, RealType &linf_u_n_error, size_t &counter_interface_pts,
                               size_t &degree, RealType &force_pressure_avg, RealType &force_pressure_max,
                               size_t &counter_pt_Gamma, PP1 &test_gammaH, PP1 &test_press_jump, PP1 &test_grad_vel_jump,
                               RealType &distance_pts, RealType &force_gradVel_max, RealType &force_gradVel_avg,
                               PP1 &test_vel_n, PP1 &p1_gp, PP1 &p2_gp, PP1 &test_grad_vel_t_n, Function1 &sol_vel,
                               Function2 &sol_p, Function3 &vel_grad) {


        vector_cell_basis <cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi_cell);
        RealType kappa_1 = test_case.parms.kappa_1;
        RealType kappa_2 = test_case.parms.kappa_2;

        cell_basis <cuthho_poly_mesh<RealType>, RealType> pb(msh, cl, hdi_face);
        auto cbs = cb.size();
//auto pbs = pb.size();
//    auto sol_vel = test_case.sol_vel;
//    auto sol_p = test_case.sol_p;
//    auto vel_grad = test_case.vel_grad;

        level_set_function.cell_assignment(cl); // ----------------------------> TOGLIERE????
        test_case.test_case_cell_assignment(cl); // ----------------------------> TOGLIERE????


//auto bcs_vel = test_case.bcs_vel;
//auto neumann_jump = test_case.neumann_jump;
        assembler_sc.set_dir_func(bcs_vel); // CAMBIA QUALCOSA?? // ----------------------------> TOGLIERE????


        Matrix<RealType, Dynamic, 1> vel_locdata_n, vel_locdata_p, vel_locdata;
        Matrix<RealType, Dynamic, 1> P_locdata_n, P_locdata_p, P_locdata;
        Matrix<RealType, Dynamic, 1> vel_cell_dofs_n, vel_cell_dofs_p, vel_cell_dofs;

        if (location(msh, cl) == element_location::ON_INTERFACE) {
            vel_locdata_n = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
            vel_locdata_p = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            P_locdata_n = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
            P_locdata_p = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);


            vel_cell_dofs_n = vel_locdata_n.head(cbs);
            vel_cell_dofs_p = vel_locdata_p.head(cbs);


// NOT AGGLO CELL
            if (level_set_function.subcells.size() < 1) {

                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
                size_t i_local = 0;
                for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                    if (level_set_function(ln_Qk, cl_old) > 0.0) {
                        auto phi_HHO = cb.eval_basis(ln_Qk);
                        auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                        velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
                    } else {
                        auto phi_HHO = cb.eval_basis(ln_Qk);
                        auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                        velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
//  velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
//  velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                    }
                }

            } else // AGGLO CELL
            {

                for (size_t i_subcell = 0;
                     i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size(); i_subcell++) {
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
//std::cout<<"offset_old = "<<offset_old<<std::endl;
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                    size_t i_local = 0;
                    for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                        if (level_set_function(ln_Qk, cl_old) > 0.0) {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                            velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
                        } else {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                            velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
//  velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
//  velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                        }
                    }

                }
            }

//            tc2.toc();
//            std::cout<<"time tc2 3 = "<<tc2<<std::endl;
//            tc2.tic();
            auto qps_n = integrate(msh, cl, 2 * hdi_cell, element_location::IN_NEGATIVE_SIDE);
            for (auto &qp: qps_n) {
// Compute H1-error //
                auto t_dphi = cb.eval_gradients(qp.first);
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++)
                    grad += vel_cell_dofs_n(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                H1_error += qp.second * kappa_1 * inner_product(grad_sym_diff, grad_sym_diff);

// Compute L2-error //
                auto t_phi = cb.eval_basis(qp.first);
                auto v = t_phi.transpose() * vel_cell_dofs_n;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                L2_error += qp.second * kappa_1 * sol_diff.dot(sol_diff);

                uT1_gp->add_data(qp.first, v(0));
                uT2_gp->add_data(qp.first, v(1));

                interface_file << qp.first.x() << "   " << qp.first.y() << "   " << v(0) << "   " << v(1) << std::endl;



// L2 - pressure - error //
                auto p_phi = pb.eval_basis(qp.first);
                RealType p_num = p_phi.dot(P_locdata_n);
                RealType p_diff = test_case.sol_p(qp.first) - p_num; // era test_case STE
//                auto p_prova = test_case.sol_p( qp.first ) ;
//                std::cout<<"In pt = "<<qp.first<<" --> pressure ANAL  = "<<p_prova<<" , pressure NUM = "<< p_num<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                L2_pressure_error += qp.second * p_diff * p_diff / kappa_1;
                p_gp->add_data(qp.first, p_num);
                p1_gp->add_data(qp.first, p_num);
            }

            auto qps_p = integrate(msh, cl, 2 * hdi_cell, element_location::IN_POSITIVE_SIDE);
            for (auto &qp: qps_p) {
// Compute H1-error //
                auto t_dphi = cb.eval_gradients(qp.first);
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++)
                    grad += vel_cell_dofs_p(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                H1_error += qp.second * kappa_2 * inner_product(grad_sym_diff, grad_sym_diff);

// Compute L2-error //
                auto t_phi = cb.eval_basis(qp.first);
                auto v = t_phi.transpose() * vel_cell_dofs_p;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                L2_error += qp.second * kappa_2 * sol_diff.dot(sol_diff);

                uT1_gp->add_data(qp.first, v(0));
                uT2_gp->add_data(qp.first, v(1));
//                uT_gp->add_data( qp.first, std::make_pair(v(0),v(1)) );
                if (interface_file) {

                    interface_file << qp.first.x() << "   " << qp.first.y() << "   " << v(0) << "   " << v(1) << std::endl;

                }
// L2 - pressure - error //
                auto p_phi = pb.eval_basis(qp.first);
                RealType p_num = p_phi.dot(P_locdata_p);
                RealType p_diff = test_case.sol_p(qp.first) - p_num; // era test_case STE
//auto p_prova = test_case.sol_p( qp.first ) ;
//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                L2_pressure_error += qp.second * p_diff * p_diff / kappa_2;

                p_gp->add_data(qp.first, p_num);
                p2_gp->add_data(qp.first, p_num);
            }
            if (1) {
                for (auto &interface_point: cl.user_data.interface) {
                    auto t_phi = cb.eval_basis(interface_point);
                    auto v = t_phi.transpose() * vel_cell_dofs_p;
                    auto n = level_set_function.normal(interface_point);
                    auto v_n = v.dot(n);
                    l1_u_n_error += std::abs(v_n);
                    l2_u_n_error += pow(v_n, 2.0);
                    linf_u_n_error = std::max(linf_u_n_error, std::abs(v_n));
                    counter_interface_pts++;
                }
            }
//            tc2.toc();
//            std::cout<<"time tc2 2 = "<<tc2<<std::endl;

            if (1) // analysis power of pressure
            {
                auto parametric_interface = test_case.parametric_interface;
                auto gamma = test_case.gamma;
                auto msh_int = cl.user_data.integration_msh;
                auto global_cells_i = parametric_interface.get_global_cells_interface(msh, cl);
                Matrix<RealType, 2, 1> phi_t;
                size_t degree_curve = msh_int.degree_curve;
                RealType tot = 10.0;
//                Interface_parametrisation_mesh1d curve(degree_curve) ;
//            degree += 3*degree_curve -4 ; // 2*degree_curve ; // TO BE CHECKED
//            auto qps = edge_quadrature<RealType>(degree);
                auto neumann = test_case.neumann_jump;
                for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++) {
                    auto pts = points(msh_int, msh_int.cells[i_cell]);
                    size_t global_cl_i = global_cells_i[i_cell];
//                auto qp_old = 0.5 *(*(qps.begin())).first.x() + 0.5;
//                auto p = parametric_interface(t , pts , degree_curve ) ;
//                point<RealType,2> pt_old ; //= typename Mesh::point_type( p(0) , p(1) ) ;

                    for (RealType i = 0.0; i <= tot; i++) {
                        auto t = 0.0 + i / tot;
                        auto p = parametric_interface(t, pts, degree_curve);
                        point<RealType, 2> pt = typename Mesh::point_type(p(0), p(1));
//                    if( t == 0.0 )
//                        pt_old = pt;
                        auto p_phi = pb.eval_basis(pt);
                        RealType p_pos = p_phi.dot(P_locdata_p);
                        RealType p_neg = p_phi.dot(P_locdata_n);


                        auto phi_HHO = cb.eval_basis(pt);
                        auto t_dphi = cb.eval_gradients(pt);
                        Matrix<RealType, 2, 2> grad_p = Matrix<RealType, 2, 2>::Zero();
                        Matrix<RealType, 2, 2> grad_n = Matrix<RealType, 2, 2>::Zero();

                        for (size_t i = 1; i < cbs; i++) {
                            grad_p += vel_cell_dofs_p(i) * t_dphi[i].block(0, 0, 2, 2);
                            grad_n += vel_cell_dofs_n(i) * t_dphi[i].block(0, 0, 2, 2);

                        }
                        Matrix<RealType, 2, 2> grad_sym_p = 0.5 * (grad_p + grad_p.transpose());
                        Matrix<RealType, 2, 2> grad_sym_n = 0.5 * (grad_n + grad_n.transpose());
                        auto vel_n = phi_HHO.transpose() * vel_cell_dofs_n;

                        Matrix<RealType, 2, 1> phi_n = level_set_function.normal(pt);
                        RealType val_u_n = (vel_n).transpose() * (phi_n);

                        auto val_p = (p_pos - p_neg);
                        Matrix<RealType, 2, 1> val_grad_u_n =
                                (2.0 * kappa_1 * grad_sym_n - 2.0 * kappa_2 * grad_sym_p) * (phi_n);
                        RealType val_u = (phi_n.transpose()) * val_grad_u_n;
                        phi_t(0) = -phi_n(1);
                        phi_t(1) = phi_n(0);
                        RealType t_val_u_n = (phi_t.transpose()) * val_grad_u_n;
//                    RealType val_u ;
//                    if( signbit(phi_n(0)) == signbit(grads_u_n(0)) && signbit(phi_n(1)) == signbit(grads_u_n(1)) )
//                        val_u = grads_u_n.norm();
//                    else
//                        val_u = -grads_u_n.norm();
                        point<RealType, 2> curv_var = typename Mesh::point_type(distance_pts, 0.0);
                        auto val_H = gamma * level_set_function.divergence(pt);
                        test_press_jump->add_data(curv_var, val_p);
                        test_gammaH->add_data(curv_var, val_H);
                        test_grad_vel_jump->add_data(curv_var, val_u);
                        test_grad_vel_t_n->add_data(curv_var, t_val_u_n);
                        test_vel_n->add_data(curv_var, val_u_n);

                        force_pressure_avg += val_p / val_H;
                        force_pressure_max = std::max(force_pressure_max, std::abs(val_p / val_H));

                        force_gradVel_avg += val_u / val_H;
                        force_gradVel_max = std::max(force_gradVel_max, std::abs(val_u / val_H));


                        counter_pt_Gamma++;

                        RealType dist;

                        if (t == 1)
                            dist = 0.0;
                        else
                            dist = (parametric_interface(t + 1.0 / tot, pts, degree_curve) - p).norm();

                        distance_pts += dist;

                    }
                }


            }

        } else {
//            tc2.tic();
            vel_locdata = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            P_locdata = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);

            vel_cell_dofs = vel_locdata.head(cbs);



// NOT AGGLO CELL
            if (level_set_function.subcells.size() < 1) {
//                assert(level_set_function.agglo_LS_cl.user_data.offset_subcells.size()==2);
//                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                    auto phi_HHO = cb.eval_basis(ln_Qk);
                    auto vel = phi_HHO.transpose() * vel_cell_dofs;
                    velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                    velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                    i_local++;

                }

            } else // AGGLO CELL
            {
                for (size_t i_subcell = 0;
                     i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size(); i_subcell++) {
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
//std::cout<<"offset_old = "<<offset_old<<std::endl;
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                    size_t i_local = 0;
                    for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                        auto phi_HHO = cb.eval_basis(ln_Qk);
                        auto vel = phi_HHO.transpose() * vel_cell_dofs;
                        velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;

                    }

                }
            }
//            tc2.toc();
//            std::cout<<"time tc2 1 = "<<tc2<<std::endl;
//            tc2.tic();
            RealType kappa = test_case.parms.kappa_1;
            if (location(msh, cl) == element_location::IN_POSITIVE_SIDE)
                kappa = test_case.parms.kappa_2;

            auto qps = integrate(msh, cl, 2 * hdi_cell);
            for (auto &qp: qps) {
// Compute H1-error //
                auto t_dphi = cb.eval_gradients(qp.first);
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++)
                    grad += vel_cell_dofs(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                H1_error += qp.second * kappa * inner_product(grad_sym_diff, grad_sym_diff);

// Compute L2-error //
                auto t_phi = cb.eval_basis(qp.first);
                auto v = t_phi.transpose() * vel_cell_dofs;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                L2_error += qp.second * kappa * sol_diff.dot(sol_diff);

                uT1_gp->add_data(qp.first, v(0));
                uT2_gp->add_data(qp.first, v(1));
//                uT_gp->add_data( qp.first, std::make_pair(v(0),v(1)) );
                if (interface_file) {

                    interface_file << qp.first.x() << "   " << qp.first.y() << "   " << v(0) << "   " << v(1) << std::endl;

                }

// L2 - pressure - error //
                auto p_phi = pb.eval_basis(qp.first);
                RealType p_num = p_phi.dot(P_locdata);
                RealType p_diff = test_case.sol_p(qp.first) - p_num; // era test_case STE
//auto p_prova = test_case.sol_p( qp.first ) ;
//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                L2_pressure_error += qp.second * p_diff * p_diff / kappa;

                p_gp->add_data(qp.first, p_num);
                if (level_set_function(qp.first, cl) > 0.0)
                    p2_gp->add_data(qp.first, p_num);
                else
                    p1_gp->add_data(qp.first, p_num);
            }
//            tc2.toc();
//            std::cout<<"time tc2 0 = "<<tc2<<std::endl;
        }


    }


    template<typename Mesh, typename Cell, typename LS, typename TC, typename RealType, typename ASS, typename BDRY, typename SOL, typename VEL, typename PP1, typename PP2, typename Function1, typename Function2, typename Function3>
    void
    post_processing_functionLS_fast(const Mesh &msh, Cell &cl, size_t hdi_cell, size_t hdi_face, LS &level_set_function,
                                    TC &test_case, ASS &assembler_sc, BDRY &bcs_vel, const SOL &sol, VEL &velocity,
                                    RealType &H1_error, RealType &L2_error, PP1 &uT1_gp, PP1 &uT2_gp, PP1 &p_gp,
                                    PP2 &interface_file, RealType &L2_pressure_error, RealType &l1_u_n_error,
                                    RealType &l2_u_n_error, RealType &linf_u_n_error, size_t &counter_interface_pts,
                                    size_t &degree, RealType &force_pressure_avg, RealType &force_pressure_max,
                                    size_t &counter_pt_Gamma, PP1 &test_gammaH, PP1 &test_press_jump,
                                    PP1 &test_grad_vel_jump, RealType &distance_pts, RealType &force_gradVel_max,
                                    RealType &force_gradVel_avg, PP1 &test_vel_n, PP1 &p1_gp, PP1 &p2_gp,
                                    PP1 &test_grad_vel_t_n, Function1 &sol_vel, Function2 &sol_p, Function3 &vel_grad) {


        vector_cell_basis <cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi_cell);
        RealType kappa_1 = test_case.parms.kappa_1;
        RealType kappa_2 = test_case.parms.kappa_2;

        cell_basis <cuthho_poly_mesh<RealType>, RealType> pb(msh, cl, hdi_face);
        auto cbs = cb.size();
        auto pbs = pb.size();


        level_set_function.cell_assignment(cl); // ----------------------------> TOGLIERE????
        test_case.test_case_cell_assignment(cl); // ----------------------------> TOGLIERE????

//    auto sol_vel = test_case.sol_vel;
//    auto sol_p = test_case.sol_p;
//    auto vel_grad = test_case.vel_grad;

        assembler_sc.set_dir_func(bcs_vel); // CAMBIA QUALCOSA?? // ----------------------------> TOGLIERE????


        Matrix<RealType, Dynamic, 1> vel_locdata_n, vel_locdata_p, vel_locdata;
        Matrix<RealType, Dynamic, 1> P_locdata_n, P_locdata_p, P_locdata;
        Matrix<RealType, Dynamic, 1> vel_cell_dofs_n, vel_cell_dofs_p, vel_cell_dofs;


        vel_locdata = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
        P_locdata = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);

        vel_cell_dofs = vel_locdata.head(cbs);



// NOT AGGLO CELL
        if (level_set_function.subcells.size() < 1) {
//                assert(level_set_function.agglo_LS_cl.user_data.offset_subcells.size()==2);
//                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
            auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
            auto cl_old = velocity.msh.cells[offset_old];
            auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
            size_t i_local = 0;
            for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                auto phi_HHO = cb.eval_basis(ln_Qk);
                auto vel = phi_HHO.transpose() * vel_cell_dofs;
                velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                i_local++;

            }

        } else // AGGLO CELL
        {
            for (size_t i_subcell = 0;
                 i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size(); i_subcell++) {
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
//std::cout<<"offset_old = "<<offset_old<<std::endl;
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                    auto phi_HHO = cb.eval_basis(ln_Qk);
                    auto vel = phi_HHO.transpose() * vel_cell_dofs;
                    velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                    velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                    i_local++;

                }

            }
        }

        RealType kappa = test_case.parms.kappa_1;
        if (location(msh, cl) == element_location::IN_POSITIVE_SIDE)
            kappa = test_case.parms.kappa_2;

        auto qps = integrate(msh, cl, 2 * hdi_cell);
        for (auto &qp: qps) {
// Compute H1-error //
            auto t_dphi = cb.eval_gradients(qp.first);
            Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

            for (size_t i = 1; i < cbs; i++)
                grad += vel_cell_dofs(i) * t_dphi[i].block(0, 0, 2, 2);

            Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
            Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
            H1_error += qp.second * kappa * inner_product(grad_sym_diff, grad_sym_diff);

// Compute L2-error //
            auto t_phi = cb.eval_basis(qp.first);
            auto v = t_phi.transpose() * vel_cell_dofs;
            Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;

            L2_error += qp.second * kappa * sol_diff.dot(sol_diff);

            uT1_gp->add_data(qp.first, v(0));
            uT2_gp->add_data(qp.first, v(1));
            if (interface_file) {

                interface_file << qp.first.x() << "   " << qp.first.y() << "   " << v(0) << "   " << v(1) << std::endl;

            }

// L2 - pressure - error //
            auto p_phi = pb.eval_basis(qp.first);
            RealType p_num = p_phi.dot(P_locdata);
            RealType p_diff = test_case.sol_p(qp.first) - p_num; // era test_case STE

            L2_pressure_error += qp.second * p_diff * p_diff / kappa;

            p_gp->add_data(qp.first, p_num);
            if (level_set_function(qp.first, cl) > 0.0)
                p2_gp->add_data(qp.first, p_num);
            else
                p1_gp->add_data(qp.first, p_num);
        }


    }




    template<typename Mesh, typename testType, typename meth, typename Fonction, typename Velocity>
    stokes_test_info<typename Mesh::coordinate_type>
    run_cuthho_interface_velocity_complete(const Mesh &msh, size_t degree, meth &method, testType &test_case,
                                           Fonction &level_set_function, Velocity &velocity, bool sym_grad, size_t time,
                                           int time_gap, std::string &path) {
        using RealType = typename Mesh::coordinate_type;

        auto iso_val_interface = level_set_function.iso_val_interface;


        auto bcs_vel = test_case.bcs_vel;


        timecounter tc;

        bool sc = true;  // static condensation


// ************** ASSEMBLE PROBLEM **************
        hho_degree_info hdi(degree + 1, degree);

        tc.tic();


        auto assembler = make_stokes_interface_assembler(msh, bcs_vel, hdi);

        auto assembler_sc = make_stokes_interface_condensed_assembler(msh, bcs_vel, hdi);

// IT MAY GO INTO THE LOOP ( IF YES ADD ALSO IN THE POST-PROCESSING LOOP )
        assembler_sc.set_dir_func(bcs_vel); // DOVE VA? INTO LOOP cl? SE CAMBIASSE bcs_vel in spazio forse si!

        test_case.test_case_mesh_assignment(msh);

        for (auto &cl: msh.cells) {

//std::cout<<yellow<<bold<<"CELL = "<<offset(msh,cl) <<reset<<std::endl;
            test_case.test_case_cell_assignment(cl);
            auto contrib = method.make_contrib(msh, cl, test_case, hdi);
            auto lc = contrib.first;
            auto f = contrib.second;

            if (sc)
                assembler_sc.assemble(msh, cl, lc, f);
            else
                assembler.assemble(msh, cl, lc, f);

        }


        if (sc)
            assembler_sc.finalize();
        else
            assembler.finalize();


        tc.toc();
        std::cout << "Matrix assembly: " << tc << " seconds" << std::endl;

        if (sc)
            std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
        else
            std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

        std::cout << "Cells: " << msh.cells.size() << std::endl;
        std::cout << "Faces: " << msh.faces.size() << std::endl;

// ************** SOLVE **************
        tc.tic();
#if 1
        SparseLU <SparseMatrix<RealType>> solver;
        Matrix<RealType, Dynamic, 1> sol;

        if (sc) {
            std::cout << "First step: analyze pattern... " << std::endl;
            solver.analyzePattern(assembler_sc.LHS);
            std::cout << "Pattern ok. Second step: assembling... " << std::endl;
            solver.factorize(assembler_sc.LHS);
            std::cout << "Assembling ok. Third step: solving... " << std::endl;
            sol = solver.solve(assembler_sc.RHS);
            std::cout << "..problem solved. " << std::endl;
        } else {
            solver.analyzePattern(assembler.LHS);
            solver.factorize(assembler.LHS);
            sol = solver.solve(assembler.RHS);
        }
#endif
#if 0
        Matrix<RealType, Dynamic, 1> sol;
        cg_params <RealType> cgp;
        cgp.histfile = "cuthho_cg_hist.dat";
        cgp.verbose = true;
        cgp.apply_preconditioner = true;
        if (sc) {
            sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
            cgp.max_iter = assembler_sc.LHS.rows();
//        conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);


            ConjugateGradient < SparseMatrix < RealType > , Lower | Upper > cg;
            cg.compute(assembler_sc.LHS);
            sol = cg.solve(assembler_sc.RHS);
            std::cout << "#iterations:     " << cg.iterations() << std::endl;
            std::cout << "estimated error: " << cg.error() << std::endl;
// conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);

        } else {
            sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
            cgp.max_iter = assembler.LHS.rows();
            conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
        }
#endif
        tc.toc();
        std::cout << "Linear solver: " << tc << " seconds" << std::endl;

// ************** POSTPROCESS **************


        postprocess_output <RealType> postoutput;
        std::string filename_interface_uT = path + "interface_uT_" + std::to_string(time) + ".3D";
        std::ofstream interface_file(filename_interface_uT, std::ios::out | std::ios::trunc);

        if (interface_file) {
// instructions
            interface_file << "X   Y   val0   val1" << std::endl;
        } else
            std::cerr << "Interface_file has not been opened" << std::endl;


        auto uT1_gp = std::make_shared < gnuplot_output_object < RealType > >
                      (path + "interface_uT1_" + std::to_string(time) + ".dat");
        auto uT2_gp = std::make_shared < gnuplot_output_object < RealType > >
                      (path + "interface_uT2_" + std::to_string(time) + ".dat");
//    auto uT_gp  = std::make_shared< gnuplot_output_object_vec<RealType> >("interface_uT.dat");

//    std::string filename_pressure = "interface_p_" + std::to_string(time) + ".dat";
        auto p_gp = std::make_shared < gnuplot_output_object < RealType > >
                    (path + "interface_p_" + std::to_string(time) + ".dat");
        auto p1_gp = std::make_shared < gnuplot_output_object < RealType > >
                     (path + "interface_pIN_" + std::to_string(time) + ".dat");
        auto p2_gp = std::make_shared < gnuplot_output_object < RealType > >
                     (path + "interface_pOUT_" + std::to_string(time) + ".dat");
//auto p_gp    = std::make_shared< gnuplot_output_object<RealType> >(filename_pressure);

        std::string filename_gammaH = path + "gamma_H_" + std::to_string(time) + ".dat";
        auto test_gammaH = std::make_shared < gnuplot_output_object < double > > (filename_gammaH);

        std::string filename_press_jump = path + "pressure_jump_" + std::to_string(time) + ".dat";
        auto test_press_jump = std::make_shared < gnuplot_output_object < double > > (filename_press_jump);

        std::string filename_grad_vel_jump = path + "grad_vel_jump_" + std::to_string(time) + ".dat";
        auto test_grad_vel_jump = std::make_shared < gnuplot_output_object < double > > (filename_grad_vel_jump);


        std::string filename_grad_vel_t_n = path + "grad_vel_t_n_" + std::to_string(time) + ".dat";
        auto test_grad_vel_t_n = std::make_shared < gnuplot_output_object < double > > (filename_grad_vel_t_n);

        std::string filename_veln_n = path + "vel_un_n_" + std::to_string(time) + ".dat";
        auto test_veln_n = std::make_shared < gnuplot_output_object < double > > (filename_veln_n);

        std::string filename_veln_t = path + "vel_un_t_" + std::to_string(time) + ".dat";
        auto test_veln_t = std::make_shared < gnuplot_output_object < double > > (filename_veln_t);

        std::string filename_velp_n = path + "vel_up_n_" + std::to_string(time) + ".dat";
        auto test_velp_n = std::make_shared < gnuplot_output_object < double > > (filename_velp_n);

        std::string filename_velp_t = path + "vel_up_t_" + std::to_string(time) + ".dat";
        auto test_velp_t = std::make_shared < gnuplot_output_object < double > > (filename_velp_t);

        RealType pos = 0.0;
        RealType force_pressure_avg = 0.0, force_pressure_max = 0.0;
        RealType force_gradVel_avg = 0.0, force_gradVel_max = 0.0;

        size_t counter_pt_Gamma = 0;


        tc.tic();
        RealType H1_error = 0.0;
        RealType L2_error = 0.0;
        RealType L2_pressure_error = 0.0;
        RealType l1_u_n_error = 0.0;
        RealType l2_u_n_error = 0.0;
        RealType linf_u_n_error = 0.0;
        size_t counter_interface_pts = 0;
        RealType distance_pts = 0.0;

        timecounter tc1;
        tc1.tic();
//    size_t i_global = 0 ;
        auto hdi_cell = hdi.cell_degree();
        auto hdi_face = hdi.face_degree();
        auto msh_vel = velocity.msh;
        auto degree_vel = velocity.degree_FEM;


        for (auto &cl: msh.cells) {


            vector_cell_basis <cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi_cell);
            auto cbs = cb.size();


            level_set_function.cell_assignment(cl); // Useful to pick if a cell is agglo or not
//        test_case.test_case_cell_assignment(cl) ; // ----------------------------> TOGLIERE????


            assembler_sc.set_dir_func(bcs_vel); // CAMBIA QUALCOSA?? // ----------------------------> TOGLIERE????


            Matrix<RealType, Dynamic, 1> vel_locdata_n, vel_locdata_p, vel_locdata;
//        Matrix<RealType, Dynamic, 1> P_locdata_n, P_locdata_p, P_locdata;
            Matrix<RealType, Dynamic, 1> vel_cell_dofs_n, vel_cell_dofs_p, vel_cell_dofs;

            if (location(msh, cl) == element_location::ON_INTERFACE) {
                vel_locdata_n = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                vel_locdata_p = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
//                P_locdata_n = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
//                P_locdata_p = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);


                vel_cell_dofs_n = vel_locdata_n.head(cbs);
                vel_cell_dofs_p = vel_locdata_p.head(cbs);



// Updating velocity field by STE
//std::cout<<"------------>>> CUT CELL"<<std::endl;
//std::cout<<"subcells.size() = "<<level_set_function.subcells.size()<<std::endl;


// NOT AGGLO CELL
                if (level_set_function.subcells.size() < 1) {
//                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells.size() == 2);
//                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                    auto cl_old = msh_vel.cells[offset_old];
                    auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;


                    velocity.set_weight_area(offset_old, 1.0);

                    size_t i_local = 0;
                    for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                        if (level_set_function(ln_Qk, cl_old) > iso_val_interface) {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                            velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
                        } else {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                            velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
                        }
                    }

                } else // AGGLO CELL
                {

                    RealType nOfSubCellsAgglo = level_set_function.agglo_LS_cl.user_data.offset_subcells.size();
                    for (size_t i_subcell = 0;
                         i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size(); i_subcell++) {
                        auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];

                        velocity.set_weight_area(offset_old, nOfSubCellsAgglo);
//std::cout<<"offset_old = "<<offset_old<<std::endl;
                        auto cl_old = msh_vel.cells[offset_old];
                        auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
                        size_t i_local = 0;
                        for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                            if (level_set_function(ln_Qk, cl_old) > iso_val_interface) {
                                auto phi_HHO = cb.eval_basis(ln_Qk);
                                auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                                velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                                velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                                i_local++;
                            } else {
                                auto phi_HHO = cb.eval_basis(ln_Qk);
                                auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                                velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                                velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                                i_local++;
                            }
                        }

                    }
                }


            } else {

                vel_locdata = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
//                P_locdata = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);

                vel_cell_dofs = vel_locdata.head(cbs);


// NOT AGGLO CELL
                if (level_set_function.subcells.size() < 1) {
//                assert(level_set_function.agglo_LS_cl.user_data.offset_subcells.size()==2);
//                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                    velocity.set_weight_area(offset_old, 1.0);
                    auto cl_old = msh_vel.cells[offset_old];
                    auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
                    size_t i_local = 0;
                    for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                        auto phi_HHO = cb.eval_basis(ln_Qk);
                        auto vel = phi_HHO.transpose() * vel_cell_dofs;
                        velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;

                    }

                } else // AGGLO CELL
                {
                    RealType nOfSubCellsAgglo = level_set_function.agglo_LS_cl.user_data.offset_subcells.size();
                    for (size_t i_subcell = 0;
                         i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size(); i_subcell++) {
                        auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
                        velocity.set_weight_area(offset_old, nOfSubCellsAgglo);
//std::cout<<"offset_old = "<<offset_old<<std::endl;
                        auto cl_old = msh_vel.cells[offset_old];
                        auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
                        size_t i_local = 0;
                        for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs;
                            velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;

                        }

                    }
                }


            }


        }

        tc.toc();
        std::cout << bold << yellow << "HHO velocity postprocessing: " << tc << " seconds" << reset << std::endl;


        timecounter tc_p1;

        tc_p1.tic();

        std::vector <size_t> cut_cell_cointainer, uncut_cell_cointainer;

        for (auto &cl: msh.cells) {
            if (location(msh, cl) == element_location::ON_INTERFACE)
                cut_cell_cointainer.push_back(offset(msh, cl));

            else
                uncut_cell_cointainer.push_back(offset(msh, cl));


        }


        point<RealType, 2> first_point;
        point<RealType, 2> cell_end_point;
        bool first_cut_cell_found = FALSE;

//    size_t deg_size = 2; // ESEMPIOOO
        auto sol_vel = test_case.sol_vel;
        auto sol_p = test_case.sol_p;
        auto vel_grad = test_case.vel_grad;


        size_t cl_i = 0;
        while (cut_cell_cointainer.size() > 0) {
            if (cl_i > cut_cell_cointainer.size() - 1)
                std::cout << "stop: first_pt = " << first_point << " pt_to_find = " << cell_end_point << std::endl;
            size_t k_offset = cut_cell_cointainer[cl_i];
            auto cl = msh.cells[k_offset];
//        auto msh_int = cl.user_data.integration_msh ;

            if (!first_cut_cell_found) {
                cut_cell_cointainer.erase(cut_cell_cointainer.begin()); //pop_front();

                post_processing_functionLS_2(msh, cl, hdi_cell, hdi_face, level_set_function, test_case,
                                             assembler_sc, bcs_vel, sol, velocity, H1_error, L2_error, uT1_gp, uT2_gp,
                                             p_gp, interface_file, L2_pressure_error, l1_u_n_error, l2_u_n_error,
                                             linf_u_n_error, counter_interface_pts, degree, force_pressure_avg,
                                             force_pressure_max, counter_pt_Gamma, test_gammaH, test_press_jump,
                                             test_grad_vel_jump, distance_pts, force_gradVel_max, force_gradVel_avg,
                                             test_veln_n,
                                             p1_gp, p2_gp, test_grad_vel_t_n, sol_vel, sol_p, vel_grad, test_veln_t,
                                             test_velp_n, test_velp_t);

                first_cut_cell_found = TRUE;
                first_point = *cl.user_data.interface.begin();
                cell_end_point = *(cl.user_data.interface.end() - 1);
                cl_i = 0;
            } else if (first_cut_cell_found && cell_end_point == *cl.user_data.interface.begin() &&
                       !(first_point == cell_end_point)) {
                cut_cell_cointainer.erase(cut_cell_cointainer.begin() + cl_i);
//                cut_cell_cointainer.pop_front();

                post_processing_functionLS_2(msh, cl, hdi_cell, hdi_face, level_set_function, test_case,
                                             assembler_sc, bcs_vel, sol, velocity, H1_error, L2_error, uT1_gp, uT2_gp, p_gp,
                                             interface_file, L2_pressure_error, l1_u_n_error, l2_u_n_error, linf_u_n_error,
                                             counter_interface_pts, degree, force_pressure_avg, force_pressure_max,
                                             counter_pt_Gamma, test_gammaH, test_press_jump, test_grad_vel_jump,
                                             distance_pts,
                                             force_gradVel_max, force_gradVel_avg, test_veln_n,
                                             p1_gp, p2_gp, test_grad_vel_t_n, sol_vel, sol_p, vel_grad, test_veln_t,
                                             test_velp_n, test_velp_t);

                cell_end_point = *(cl.user_data.interface.end() - 1);
                cl_i = 0;

            } else if (first_point == cell_end_point)
                break;
            else
                cl_i++;

        }

        std::cout << "First Point curvilinear variable: " << first_point << std::endl;
        tc_p1.toc();
        std::cout << "Interface_analysis time: " << tc_p1 << std::endl;
        tc_p1.tic();


        for (auto &i_cl: uncut_cell_cointainer) {
            auto cl = msh.cells[i_cl];
            post_processing_functionLS_fast_2(msh, cl, hdi_cell, hdi_face, level_set_function,
                                              test_case, assembler_sc, bcs_vel, sol, velocity, H1_error, L2_error,
                                              uT1_gp, uT2_gp, p_gp, interface_file, L2_pressure_error, l1_u_n_error,
                                              l2_u_n_error, linf_u_n_error, counter_interface_pts, degree,
                                              force_pressure_avg,
                                              force_pressure_max, counter_pt_Gamma, test_gammaH, test_press_jump,
                                              test_grad_vel_jump,
                                              distance_pts, force_gradVel_max, force_gradVel_avg,
                                              p1_gp, p2_gp, sol_vel, sol_p, vel_grad);

        }

        tc_p1.toc();
        std::cout << "Not cut cell analysis time: " << tc_p1 << std::endl;


        std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
        std::cout << bold << green << "L2-norm absolute error:               " << std::sqrt(L2_error) << std::endl;
        std::cout << bold << green << "Pressure L2-norm absolute error:      " << std::sqrt(L2_pressure_error) << std::endl;
        std::cout << bold << green << "l1-norm u*n error:               " << l1_u_n_error / counter_interface_pts
                  << std::endl;
        std::cout << bold << green << "l2-norm u*n error:               " << std::sqrt(l2_u_n_error / counter_interface_pts)
                  << std::endl;
        std::cout << bold << green << "linf-norm u*n error:               " << linf_u_n_error << std::endl;


        std::cout << bold << green << "AVG force pressure = " << force_pressure_avg / counter_pt_Gamma << std::endl;
        std::cout << bold << green << "linf-norm force pressure = " << force_pressure_max << std::endl;

        std::cout << bold << green << "AVG force grad_s velocity = " << force_gradVel_avg / counter_pt_Gamma << std::endl;
        std::cout << bold << green << "linf-norm force grad_s velocity = " << force_gradVel_max << std::endl;

        if (time % time_gap == 0) {
            postoutput.add_object(uT1_gp);
            postoutput.add_object(uT2_gp);
            postoutput.add_object(p_gp);
            postoutput.add_object(p1_gp);
            postoutput.add_object(p2_gp);

            postoutput.add_object(test_gammaH);
            postoutput.add_object(test_press_jump);
            postoutput.add_object(test_grad_vel_jump);
            postoutput.add_object(test_grad_vel_t_n);
            postoutput.add_object(test_veln_n);
            postoutput.add_object(test_veln_t);
            postoutput.add_object(test_velp_n);
            postoutput.add_object(test_velp_t);


            postoutput.write();
            if (interface_file) {

                interface_file.close();

            }

        }

        stokes_test_info <RealType> TI;
        TI.H1_vel = std::sqrt(H1_error);
        TI.L2_vel = std::sqrt(L2_error);
        TI.L2_p = std::sqrt(L2_pressure_error);
        if (1) {
            TI.l1_normal_vel = l1_u_n_error / counter_interface_pts;
            TI.l2_normal_vel = std::sqrt(l2_u_n_error / counter_interface_pts);
            TI.linf_normal_vel = linf_u_n_error;
        }


        if (0) {
/////////////// compute condition number
            SparseMatrix <RealType> Mat;
// Matrix<RealType, Dynamic, Dynamic> Mat;
            if (sc)
                Mat = assembler_sc.LHS;
            else
                Mat = assembler.LHS;

            {
                JacobiSVD <MatrixXd> svd(Mat);
                RealType cond = svd.singularValues()(0)
                                / svd.singularValues()(svd.singularValues().size() - 1);
                std::cout << "cond numb = " << cond << std::endl;
            }

            RealType sigma_max, sigma_min;

// Construct matrix operation object using the wrapper class SparseSymMatProd
            Spectra::SparseSymMatProd <RealType> op(Mat);
// Construct eigen solver object, requesting the largest eigenvalue
            Spectra::SymEigsSolver <RealType, Spectra::LARGEST_MAGN,
            Spectra::SparseSymMatProd<RealType>> max_eigs(&op, 1, 10);
            max_eigs.init();
            max_eigs.compute();
            if (max_eigs.info() == Spectra::SUCCESSFUL)
                sigma_max = max_eigs.eigenvalues()(0);


// Construct eigen solver object, requesting the smallest eigenvalue
            Spectra::SymEigsSolver <RealType, Spectra::SMALLEST_MAGN,
            Spectra::SparseSymMatProd<RealType>> min_eigs(&op, 1, 10);

            min_eigs.init();
            min_eigs.compute();
            if (min_eigs.info() == Spectra::SUCCESSFUL)
                sigma_min = min_eigs.eigenvalues()(0);

// compute condition number
            RealType cond = sigma_max / sigma_min;
            TI.cond = cond;
            std::cout << "sigma_max = " << sigma_max << "   sigma_min = "
                      << sigma_min << "  cond = " << cond
                      << std::endl;
        } else
            TI.cond = 0.0;

        tc.toc();
        std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


        return TI;
    }


    template<typename Mesh, typename testType, typename meth, typename Fonction, typename Velocity>
    void
    run_cuthho_interface_velocity_fast(const Mesh &msh, size_t degree, meth &method, testType &test_case,
                                       Fonction &level_set_function, Velocity &velocity, bool sym_grad, size_t time) {
        using RealType = typename Mesh::coordinate_type;

        auto iso_val_interface = level_set_function.iso_val_interface;


        auto bcs_vel = test_case.bcs_vel;


        timecounter tc;

        bool sc = true;  // static condensation


// ************** ASSEMBLE PROBLEM **************
        hho_degree_info hdi(degree + 1, degree);

        tc.tic();


//    auto assembler = make_stokes_interface_assembler(msh, bcs_vel, hdi);

        auto assembler_sc = make_stokes_interface_condensed_assembler(msh, bcs_vel, hdi);

// IT MAY GO INTO THE LOOP ( IF YES ADD ALSO IN THE POST-PROCESSING LOOP )
        assembler_sc.set_dir_func(bcs_vel); // DOVE VA? INTO LOOP cl? SE CAMBIASSE bcs_vel in spazio forse si!

        test_case.test_case_mesh_assignment(msh);

        for (auto &cl: msh.cells) {

//std::cout<<yellow<<bold<<"CELL = "<<offset(msh,cl) <<reset<<std::endl;
//        test_case.test_case_cell_assignment(cl); // DIREI CHE NON SERVE PIU
            auto contrib = method.make_contrib(msh, cl, test_case, hdi);
            auto lc = contrib.first;
            auto f = contrib.second;


            assembler_sc.assemble(msh, cl, lc, f);


        }


        assembler_sc.finalize();


        tc.toc();
        std::cout << "Matrix assembly: " << tc << " seconds" << std::endl;


        std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;


        std::cout << "Cells: " << msh.cells.size() << std::endl;
        std::cout << "Faces: " << msh.faces.size() << std::endl;

// ************** SOLVE **************
        tc.tic();
#if 1
        SparseLU <SparseMatrix<RealType>> solver;
        Matrix<RealType, Dynamic, 1> sol;

        solver.analyzePattern(assembler_sc.LHS);
        solver.factorize(assembler_sc.LHS);
        sol = solver.solve(assembler_sc.RHS);

#endif
#if 0
        Matrix<RealType, Dynamic, 1> sol;
        cg_params <RealType> cgp;
        cgp.histfile = "cuthho_cg_hist.dat";
        cgp.verbose = true;
        cgp.apply_preconditioner = true;
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
        cgp.max_iter = assembler_sc.LHS.rows();
        conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);

#endif
        tc.toc();
        std::cout << "Linear solver: " << tc << " seconds" << std::endl;

// ************** POSTPROCESS **************


        postprocess_output <RealType> postoutput;


//
//    auto uT1_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT1.dat");
//    auto uT2_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT2.dat");
//    auto uT_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT.dat");
//
////    std::string filename_pressure = "interface_p_" + std::to_string(time) + ".dat";
//    auto p_gp    = std::make_shared< gnuplot_output_object<RealType> >("interface_p.dat");
//    //auto p_gp    = std::make_shared< gnuplot_output_object<RealType> >(filename_pressure);

        tc.tic();


        auto hdi_cell = hdi.cell_degree();
        auto hdi_face = hdi.face_degree();
        auto msh_vel = velocity.msh;
        auto degree_vel = velocity.degree_FEM;
        for (auto &cl: msh.cells) {


            vector_cell_basis <cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi_cell);
            auto cbs = cb.size();


            level_set_function.cell_assignment(cl); // Useful to pick if a cell is agglo or not
//        test_case.test_case_cell_assignment(cl) ; // ----------------------------> TOGLIERE????


            assembler_sc.set_dir_func(bcs_vel); // CAMBIA QUALCOSA?? // ----------------------------> TOGLIERE????


            Matrix<RealType, Dynamic, 1> vel_locdata_n, vel_locdata_p, vel_locdata;
//        Matrix<RealType, Dynamic, 1> P_locdata_n, P_locdata_p, P_locdata;
            Matrix<RealType, Dynamic, 1> vel_cell_dofs_n, vel_cell_dofs_p, vel_cell_dofs;

            if (location(msh, cl) == element_location::ON_INTERFACE) {
                vel_locdata_n = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                vel_locdata_p = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
//                P_locdata_n = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
//                P_locdata_p = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);


                vel_cell_dofs_n = vel_locdata_n.head(cbs);
                vel_cell_dofs_p = vel_locdata_p.head(cbs);



// Updating velocity field by STE
//std::cout<<"------------>>> CUT CELL"<<std::endl;
//std::cout<<"subcells.size() = "<<level_set_function.subcells.size()<<std::endl;


// NOT AGGLO CELL
                if (level_set_function.subcells.size() < 1) {
//                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells.size() == 2);
//                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                    auto cl_old = msh_vel.cells[offset_old];
                    auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;


                    velocity.set_weight_area(offset_old, 1.0);

                    size_t i_local = 0;
                    for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                        if (level_set_function(ln_Qk, cl_old) > iso_val_interface) {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                            velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
                        } else {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                            velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
                        }
                    }

                } else // AGGLO CELL
                {

                    RealType nOfSubCellsAgglo = level_set_function.agglo_LS_cl.user_data.offset_subcells.size();
                    for (size_t i_subcell = 0;
                         i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size(); i_subcell++) {
                        auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];

                        velocity.set_weight_area(offset_old, nOfSubCellsAgglo);
//std::cout<<"offset_old = "<<offset_old<<std::endl;
                        auto cl_old = msh_vel.cells[offset_old];
                        auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
                        size_t i_local = 0;
                        for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                            if (level_set_function(ln_Qk, cl_old) > iso_val_interface) {
                                auto phi_HHO = cb.eval_basis(ln_Qk);
                                auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                                velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                                velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                                i_local++;
                            } else {
                                auto phi_HHO = cb.eval_basis(ln_Qk);
                                auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                                velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                                velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                                i_local++;
                            }
                        }

                    }
                }


            } else {

                vel_locdata = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
//                P_locdata = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);

                vel_cell_dofs = vel_locdata.head(cbs);


// NOT AGGLO CELL
                if (level_set_function.subcells.size() < 1) {
//                assert(level_set_function.agglo_LS_cl.user_data.offset_subcells.size()==2);
//                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                    velocity.set_weight_area(offset_old, 1.0);
                    auto cl_old = msh_vel.cells[offset_old];
                    auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
                    size_t i_local = 0;
                    for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                        auto phi_HHO = cb.eval_basis(ln_Qk);
                        auto vel = phi_HHO.transpose() * vel_cell_dofs;
                        velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;

                    }

                } else // AGGLO CELL
                {
                    RealType nOfSubCellsAgglo = level_set_function.agglo_LS_cl.user_data.offset_subcells.size();
                    for (size_t i_subcell = 0;
                         i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size(); i_subcell++) {
                        auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
                        velocity.set_weight_area(offset_old, nOfSubCellsAgglo);
//std::cout<<"offset_old = "<<offset_old<<std::endl;
                        auto cl_old = msh_vel.cells[offset_old];
                        auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
                        size_t i_local = 0;
                        for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs;
                            velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;

                        }

                    }
                }


            }


        }


        tc.toc();
        std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


    }


    template<typename Mesh, typename testType, typename meth, typename Fonction, typename Velocity>
    stokes_test_info<typename Mesh::coordinate_type>
    run_cuthho_interface_velocity_new(const Mesh &msh, size_t degree, meth &method, testType &test_case,
                                      Fonction &level_set_function, Velocity &velocity, bool sym_grad, size_t time) {
        using RealType = typename Mesh::coordinate_type;

        auto iso_val_interface = level_set_function.iso_val_interface;


        auto bcs_vel = test_case.bcs_vel;


        timecounter tc;

        bool sc = true;  // static condensation


// ************** ASSEMBLE PROBLEM **************
        hho_degree_info hdi(degree + 1, degree);

        tc.tic();


        auto assembler = make_stokes_interface_assembler(msh, bcs_vel, hdi);

        auto assembler_sc = make_stokes_interface_condensed_assembler(msh, bcs_vel, hdi);

// IT MAY GO INTO THE LOOP ( IF YES ADD ALSO IN THE POST-PROCESSING LOOP )
        assembler_sc.set_dir_func(bcs_vel); // DOVE VA? INTO LOOP cl? SE CAMBIASSE bcs_vel in spazio forse si!

        test_case.test_case_mesh_assignment(msh);

        for (auto &cl: msh.cells) {

//std::cout<<yellow<<bold<<"CELL = "<<offset(msh,cl) <<reset<<std::endl;
//        test_case.test_case_cell_assignment(cl); // DIREI CHE NON SERVE PIU
            auto contrib = method.make_contrib(msh, cl, test_case, hdi);
            auto lc = contrib.first;
            auto f = contrib.second;

            if (sc)
                assembler_sc.assemble(msh, cl, lc, f);
            else
                assembler.assemble(msh, cl, lc, f);

        }


        if (sc)
            assembler_sc.finalize();
        else
            assembler.finalize();


        tc.toc();
        std::cout << "Matrix assembly: " << tc << " seconds" << std::endl;

        if (sc)
            std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
        else
            std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

        std::cout << "Cells: " << msh.cells.size() << std::endl;
        std::cout << "Faces: " << msh.faces.size() << std::endl;

// ************** SOLVE **************
        tc.tic();
#if 1
        SparseLU <SparseMatrix<RealType>> solver;
        Matrix<RealType, Dynamic, 1> sol;

        if (sc) {
            solver.analyzePattern(assembler_sc.LHS);
            solver.factorize(assembler_sc.LHS);
            sol = solver.solve(assembler_sc.RHS);
        } else {
            solver.analyzePattern(assembler.LHS);
            solver.factorize(assembler.LHS);
            sol = solver.solve(assembler.RHS);
        }
#endif
#if 0
        Matrix<RealType, Dynamic, 1> sol;
        cg_params <RealType> cgp;
        cgp.histfile = "cuthho_cg_hist.dat";
        cgp.verbose = true;
        cgp.apply_preconditioner = true;
        if (sc) {
            sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
            cgp.max_iter = assembler_sc.LHS.rows();
            conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
        } else {
            sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
            cgp.max_iter = assembler.LHS.rows();
            conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
        }
#endif
        tc.toc();
        std::cout << "Linear solver: " << tc << " seconds" << std::endl;

// ************** POSTPROCESS **************


        postprocess_output <RealType> postoutput;
        std::string filename_interface_uT = "interface_uT.3D";
        std::ofstream interface_file(filename_interface_uT, std::ios::out | std::ios::trunc);

        if (interface_file) {
// instructions
            interface_file << "X   Y   val0   val1" << std::endl;
        } else
            std::cerr << "Interface_file has not been opened" << std::endl;


        auto uT1_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_uT1.dat");
        auto uT2_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_uT2.dat");
//    auto uT_gp  = std::make_shared< gnuplot_output_object_vec<RealType> >("interface_uT.dat");

//    std::string filename_pressure = "interface_p_" + std::to_string(time) + ".dat";
        auto p_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_p.dat");
//auto p_gp    = std::make_shared< gnuplot_output_object<RealType> >(filename_pressure);

//    auto force_pressure    = std::make_shared< gnuplot_output_object<RealType> >("pressure_force.dat");

        RealType pos = 0.0;
        RealType force_pressure_avg = 0.0, force_pressure_max = 0.0;
        size_t counter_pt_Gamma = 0;


        tc.tic();
        RealType H1_error = 0.0;
        RealType L2_error = 0.0;
        RealType L2_pressure_error = 0.0;
        RealType l1_u_n_error = 0.0;
        RealType l2_u_n_error = 0.0;
        RealType linf_u_n_error = 0.0;
        size_t counter_interface_pts = 0;


//    timecounter tc1;
//    tc1.tic();
//    size_t i_global = 0 ;
        auto hdi_cell = hdi.cell_degree();
        auto hdi_face = hdi.face_degree();
        auto msh_vel = velocity.msh;
        auto degree_vel = velocity.degree_FEM;


        for (auto &cl: msh.cells) {

//        timecounter tc2;
//        tc2.tic();
            vector_cell_basis <cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi_cell);


            cell_basis <cuthho_poly_mesh<RealType>, RealType> pb(msh, cl, hdi_face);
            auto cbs = cb.size();
//auto pbs = pb.size();


            level_set_function.cell_assignment(cl); // ----------------------------> TOGLIERE????
            test_case.test_case_cell_assignment(cl); // ----------------------------> TOGLIERE????

            auto sol_vel = test_case.sol_vel;
            auto sol_p = test_case.sol_p;
            auto vel_grad = test_case.vel_grad;
//auto bcs_vel = test_case.bcs_vel;
//auto neumann_jump = test_case.neumann_jump;
            assembler_sc.set_dir_func(bcs_vel); // CAMBIA QUALCOSA?? // ----------------------------> TOGLIERE????


            Matrix<RealType, Dynamic, 1> vel_locdata_n, vel_locdata_p, vel_locdata;
            Matrix<RealType, Dynamic, 1> P_locdata_n, P_locdata_p, P_locdata;
            Matrix<RealType, Dynamic, 1> vel_cell_dofs_n, vel_cell_dofs_p, vel_cell_dofs;

            if (location(msh, cl) == element_location::ON_INTERFACE) {
                if (sc) {
                    vel_locdata_n = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                    vel_locdata_p = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                    P_locdata_n = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                    P_locdata_p = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                } else {
                    vel_locdata_n = assembler.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                    vel_locdata_p = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                    P_locdata_n = assembler.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                    P_locdata_p = assembler.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                }

                vel_cell_dofs_n = vel_locdata_n.head(cbs);
                vel_cell_dofs_p = vel_locdata_p.head(cbs);

//            tc2.toc();
//            std::cout<<"time tc2 4 = "<<tc2<<std::endl;

// Updating velocity field by STE
//std::cout<<"------------>>> CUT CELL"<<std::endl;
//std::cout<<"subcells.size() = "<<level_set_function.subcells.size()<<std::endl;
//            tc2.tic();
// NOT AGGLO CELL
                if (level_set_function.subcells.size() < 1) {
//                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells.size() == 2);
//                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                    size_t i_local = 0;
                    for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                        if (level_set_function(ln_Qk, cl_old) > iso_val_interface) {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                            velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
                        } else {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                            velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
//  velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
//  velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                        }
                    }

                } else // AGGLO CELL
                {
//for(auto pt:points(msh,cl))
//   std::cout<<"pt = "<<pt<<std::endl;

                    for (size_t i_subcell = 0;
                         i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size(); i_subcell++) {
                        auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
//std::cout<<"offset_old = "<<offset_old<<std::endl;
                        auto cl_old = velocity.msh.cells[offset_old];
                        auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                        size_t i_local = 0;
                        for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                            if (level_set_function(ln_Qk, cl_old) > iso_val_interface) {
                                auto phi_HHO = cb.eval_basis(ln_Qk);
                                auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                                velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                                velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                                i_local++;
                            } else {
                                auto phi_HHO = cb.eval_basis(ln_Qk);
                                auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                                velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                                velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                                i_local++;
//  velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
//  velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                            }
                        }

                    }
                }

//            tc2.toc();
//            std::cout<<"time tc2 3 = "<<tc2<<std::endl;
//            tc2.tic();
                auto qps_n = integrate(msh, cl, 2 * hdi_cell, element_location::IN_NEGATIVE_SIDE);
                for (auto &qp: qps_n) {
// Compute H1-error //
                    auto t_dphi = cb.eval_gradients(qp.first);
                    Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                    for (size_t i = 1; i < cbs; i++)
                        grad += vel_cell_dofs_n(i) * t_dphi[i].block(0, 0, 2, 2);

                    Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                    Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                    H1_error += qp.second * test_case.parms.kappa_1 * inner_product(grad_sym_diff, grad_sym_diff);

// Compute L2-error //
                    auto t_phi = cb.eval_basis(qp.first);
                    auto v = t_phi.transpose() * vel_cell_dofs_n;
                    Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                    L2_error += qp.second * test_case.parms.kappa_1 * sol_diff.dot(sol_diff);

                    uT1_gp->add_data(qp.first, v(0));
                    uT2_gp->add_data(qp.first, v(1));
//                uT_gp->add_data( qp.first, std::make_pair(v(0),v(1)) );
                    if (interface_file) {

                        interface_file << qp.first.x() << "   " << qp.first.y() << "   " << v(0) << "   " << v(1)
                                       << std::endl;

                    }

// L2 - pressure - error //
                    auto p_phi = pb.eval_basis(qp.first);
                    RealType p_num = p_phi.dot(P_locdata_n);
                    RealType p_diff = test_case.sol_p(qp.first) - p_num; // era test_case STE
//                auto p_prova = test_case.sol_p( qp.first ) ;
//                std::cout<<"In pt = "<<qp.first<<" --> pressure ANAL  = "<<p_prova<<" , pressure NUM = "<< p_num<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                    L2_pressure_error += qp.second * p_diff * p_diff / test_case.parms.kappa_1;
                    p_gp->add_data(qp.first, p_num);
                }

                auto qps_p = integrate(msh, cl, 2 * hdi_cell, element_location::IN_POSITIVE_SIDE);
                for (auto &qp: qps_p) {
// Compute H1-error //
                    auto t_dphi = cb.eval_gradients(qp.first);
                    Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                    for (size_t i = 1; i < cbs; i++)
                        grad += vel_cell_dofs_p(i) * t_dphi[i].block(0, 0, 2, 2);

                    Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                    Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                    H1_error += qp.second * test_case.parms.kappa_2 * inner_product(grad_sym_diff, grad_sym_diff);

// Compute L2-error //
                    auto t_phi = cb.eval_basis(qp.first);
                    auto v = t_phi.transpose() * vel_cell_dofs_p;
                    Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                    L2_error += qp.second * test_case.parms.kappa_2 * sol_diff.dot(sol_diff);

                    uT1_gp->add_data(qp.first, v(0));
                    uT2_gp->add_data(qp.first, v(1));
//                uT_gp->add_data( qp.first, std::make_pair(v(0),v(1)) );
                    if (interface_file) {

                        interface_file << qp.first.x() << "   " << qp.first.y() << "   " << v(0) << "   " << v(1)
                                       << std::endl;

                    }
// L2 - pressure - error //
                    auto p_phi = pb.eval_basis(qp.first);
                    RealType p_num = p_phi.dot(P_locdata_p);
                    RealType p_diff = test_case.sol_p(qp.first) - p_num; // era test_case STE
//auto p_prova = test_case.sol_p( qp.first ) ;
//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                    L2_pressure_error += qp.second * p_diff * p_diff / test_case.parms.kappa_2;

                    p_gp->add_data(qp.first, p_num);
                }
                if (1) {
                    for (auto &interface_point: cl.user_data.interface) {
                        auto t_phi = cb.eval_basis(interface_point);
                        auto v = t_phi.transpose() * vel_cell_dofs_p;
                        auto n = level_set_function.normal(interface_point);
                        auto v_n = v.dot(n);
                        l1_u_n_error += std::abs(v_n);
                        l2_u_n_error += pow(v_n, 2.0);
                        linf_u_n_error = std::max(linf_u_n_error, std::abs(v_n));
                        counter_interface_pts++;
                    }
                }
//            tc2.toc();
//            std::cout<<"time tc2 2 = "<<tc2<<std::endl;

                if (1) // analysis power of pressure
                {
                    auto parametric_interface = test_case.parametric_interface;
                    auto gamma = test_case.gamma;
                    auto msh_int = cl.user_data.integration_msh;
                    auto global_cells_i = parametric_interface.get_global_cells_interface(msh, cl);
                    size_t degree_curve = msh_int.degree_curve;
//                Interface_parametrisation_mesh1d curve(degree_curve) ;
                    degree += 3 * degree_curve - 4; // 2*degree_curve ; // TO BE CHECKED
                    auto qps = edge_quadrature<RealType>(degree);
                    auto neumann = test_case.neumann_jump;
                    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++) {
                        auto pts = points(msh_int, msh_int.cells[i_cell]);
                        size_t global_cl_i = global_cells_i[i_cell];
                        for (auto &qp: qps) {
                            auto t = 0.5 * qp.first.x() + 0.5;
                            auto p = parametric_interface(t, pts, degree_curve);
                            point<RealType, 2> pt = typename Mesh::point_type(p(0), p(1));

                            auto p_phi = pb.eval_basis(pt);
                            RealType p_pos = p_phi.dot(P_locdata_p);
                            RealType p_neg = p_phi.dot(P_locdata_n);

                            auto val = std::abs(p_pos - p_neg) /
                                       (std::abs(gamma * parametric_interface.curvature_cont(t, global_cl_i)));
//                        pos +=
//                        force_pressure->add_data( pos, val );
                            force_pressure_avg += val;
                            force_pressure_max = std::max(force_pressure_max, val);
                            counter_pt_Gamma++;

                        }
                    }


                }

            } else {
//            tc2.tic();
                if (sc) {
                    vel_locdata = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                    P_locdata = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                } else {
                    vel_locdata = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                    P_locdata = assembler.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                }
                vel_cell_dofs = vel_locdata.head(cbs);

//std::cout<<"------------>>> NOT CUT CELL!!!!!"<<std::endl;
//std::cout<<"subcells.size() = "<<level_set_function.subcells.size()<<std::endl;
/*
            for(size_t i_subcell = 0 ; i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size() ; i_subcell++ )
            {
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
                 std::cout<<"offset_old = "<<offset_old<<std::endl;
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                {
                    auto phi_HHO = cb.eval_basis( ln_Qk );
                    auto vel = phi_HHO.transpose() * vel_cell_dofs;
                    velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                    velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                    //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                    i_local++;

                }

            }
            */

// NOT AGGLO CELL
                if (level_set_function.subcells.size() < 1) {
//                assert(level_set_function.agglo_LS_cl.user_data.offset_subcells.size()==2);
//                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                    size_t i_local = 0;
                    for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                        auto phi_HHO = cb.eval_basis(ln_Qk);
                        auto vel = phi_HHO.transpose() * vel_cell_dofs;
                        velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;

                    }

                } else // AGGLO CELL
                {
                    for (size_t i_subcell = 0;
                         i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size(); i_subcell++) {
                        auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
//std::cout<<"offset_old = "<<offset_old<<std::endl;
                        auto cl_old = velocity.msh.cells[offset_old];
                        auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                        size_t i_local = 0;
                        for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs;
                            velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;

                        }

                    }
                }
//            tc2.toc();
//            std::cout<<"time tc2 1 = "<<tc2<<std::endl;
//            tc2.tic();
                RealType kappa = test_case.parms.kappa_1;
                if (location(msh, cl) == element_location::IN_POSITIVE_SIDE)
                    kappa = test_case.parms.kappa_2;

                auto qps = integrate(msh, cl, 2 * hdi.cell_degree());
                for (auto &qp: qps) {
// Compute H1-error //
                    auto t_dphi = cb.eval_gradients(qp.first);
                    Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                    for (size_t i = 1; i < cbs; i++)
                        grad += vel_cell_dofs(i) * t_dphi[i].block(0, 0, 2, 2);

                    Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                    Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                    H1_error += qp.second * kappa * inner_product(grad_sym_diff, grad_sym_diff);

// Compute L2-error //
                    auto t_phi = cb.eval_basis(qp.first);
                    auto v = t_phi.transpose() * vel_cell_dofs;
                    Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                    L2_error += qp.second * kappa * sol_diff.dot(sol_diff);

                    uT1_gp->add_data(qp.first, v(0));
                    uT2_gp->add_data(qp.first, v(1));
//                uT_gp->add_data( qp.first, std::make_pair(v(0),v(1)) );
                    if (interface_file) {

                        interface_file << qp.first.x() << "   " << qp.first.y() << "   " << v(0) << "   " << v(1)
                                       << std::endl;

                    }

// L2 - pressure - error //
                    auto p_phi = pb.eval_basis(qp.first);
                    RealType p_num = p_phi.dot(P_locdata);
                    RealType p_diff = test_case.sol_p(qp.first) - p_num; // era test_case STE
//auto p_prova = test_case.sol_p( qp.first ) ;
//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                    L2_pressure_error += qp.second * p_diff * p_diff / kappa;

                    p_gp->add_data(qp.first, p_num);
                }
//            tc2.toc();
//            std::cout<<"time tc2 0 = "<<tc2<<std::endl;
            }

//        i_global++;
        }
//std::cout<<"velocity.sol_HHO.first"<<'\n'<<velocity.sol_HHO.first<<std::endl;
//std::cout<<"velocity.sol_HHO.second"<<'\n'<<velocity.sol_HHO.second<<std::endl;
//    tc1.toc();
//    std::cout<<"time tc intero = "<<tc1<<std::endl;
        std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
        std::cout << bold << green << "L2-norm absolute error:               " << std::sqrt(L2_error) << std::endl;
        std::cout << bold << green << "Pressure L2-norm absolute error:      " << std::sqrt(L2_pressure_error) << std::endl;
        std::cout << bold << green << "l1-norm u*n error:               " << l1_u_n_error / counter_interface_pts
                  << std::endl;
        std::cout << bold << green << "l2-norm u*n error:               " << std::sqrt(l2_u_n_error / counter_interface_pts)
                  << std::endl;
        std::cout << bold << green << "linf-norm u*n error:               " << linf_u_n_error << std::endl;


        std::cout << bold << green << "AVG force pressure = " << force_pressure_avg / counter_pt_Gamma << std::endl;
        std::cout << bold << green << "linf-norm force pressure = " << force_pressure_max << std::endl;

        postoutput.add_object(uT1_gp);
        postoutput.add_object(uT2_gp);
        postoutput.add_object(p_gp);
        postoutput.write();
        if (interface_file) {

            interface_file.close();

        }


        stokes_test_info <RealType> TI;
        TI.H1_vel = std::sqrt(H1_error);
        TI.L2_vel = std::sqrt(L2_error);
        TI.L2_p = std::sqrt(L2_pressure_error);
        if (1) {
            TI.l1_normal_vel = l1_u_n_error / counter_interface_pts;
            TI.l2_normal_vel = std::sqrt(l2_u_n_error / counter_interface_pts);
            TI.linf_normal_vel = linf_u_n_error;
        }


        if (0) {
/////////////// compute condition number
            SparseMatrix <RealType> Mat;
// Matrix<RealType, Dynamic, Dynamic> Mat;
            if (sc)
                Mat = assembler_sc.LHS;
            else
                Mat = assembler.LHS;

            {
                JacobiSVD <MatrixXd> svd(Mat);
                RealType cond = svd.singularValues()(0)
                                / svd.singularValues()(svd.singularValues().size() - 1);
                std::cout << "cond numb = " << cond << std::endl;
            }

            RealType sigma_max, sigma_min;

// Construct matrix operation object using the wrapper class SparseSymMatProd
            Spectra::SparseSymMatProd <RealType> op(Mat);
// Construct eigen solver object, requesting the largest eigenvalue
            Spectra::SymEigsSolver <RealType, Spectra::LARGEST_MAGN,
            Spectra::SparseSymMatProd<RealType>> max_eigs(&op, 1, 10);
            max_eigs.init();
            max_eigs.compute();
            if (max_eigs.info() == Spectra::SUCCESSFUL)
                sigma_max = max_eigs.eigenvalues()(0);


// Construct eigen solver object, requesting the smallest eigenvalue
            Spectra::SymEigsSolver <RealType, Spectra::SMALLEST_MAGN,
            Spectra::SparseSymMatProd<RealType>> min_eigs(&op, 1, 10);

            min_eigs.init();
            min_eigs.compute();
            if (min_eigs.info() == Spectra::SUCCESSFUL)
                sigma_min = min_eigs.eigenvalues()(0);

// compute condition number
            RealType cond = sigma_max / sigma_min;
            TI.cond = cond;
            std::cout << "sigma_max = " << sigma_max << "   sigma_min = "
                      << sigma_min << "  cond = " << cond
                      << std::endl;
        } else
            TI.cond = 0.0;

        tc.toc();
        std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


        return TI;
    }


    template<typename Mesh, typename testType, typename meth, typename Fonction, typename Velocity>
    stokes_test_info<typename Mesh::coordinate_type>
    run_cuthho_interface_velocity_new_post_processing(const Mesh &msh, size_t degree, meth &method, testType &test_case,
                                                      Fonction &level_set_function, Velocity &velocity, bool sym_grad,
                                                      size_t time) {
        using RealType = typename Mesh::coordinate_type;

        auto iso_val_interface = level_set_function.iso_val_interface;


        auto bcs_vel = test_case.bcs_vel;


        timecounter tc;

        bool sc = true;  // static condensation


// ************** ASSEMBLE PROBLEM **************
        hho_degree_info hdi(degree + 1, degree);

        tc.tic();


        auto assembler = make_stokes_interface_assembler(msh, bcs_vel, hdi);

        auto assembler_sc = make_stokes_interface_condensed_assembler(msh, bcs_vel, hdi);

// IT MAY GO INTO THE LOOP ( IF YES ADD ALSO IN THE POST-PROCESSING LOOP )
        assembler_sc.set_dir_func(bcs_vel); // DOVE VA? INTO LOOP cl? SE CAMBIASSE bcs_vel in spazio forse si!

        test_case.test_case_mesh_assignment(msh);

        for (auto &cl: msh.cells) {

//std::cout<<yellow<<bold<<"CELL = "<<offset(msh,cl) <<reset<<std::endl;
//        test_case.test_case_cell_assignment(cl); // DIREI CHE NON SERVE PIU
            auto contrib = method.make_contrib(msh, cl, test_case, hdi);
            auto lc = contrib.first;
            auto f = contrib.second;

            if (sc)
                assembler_sc.assemble(msh, cl, lc, f);
            else
                assembler.assemble(msh, cl, lc, f);

        }


        if (sc)
            assembler_sc.finalize();
        else
            assembler.finalize();


        tc.toc();
        std::cout << "Matrix assembly: " << tc << " seconds" << std::endl;

        if (sc)
            std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
        else
            std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

        std::cout << "Cells: " << msh.cells.size() << std::endl;
        std::cout << "Faces: " << msh.faces.size() << std::endl;

// ************** SOLVE **************
        tc.tic();
#if 1
        SparseLU <SparseMatrix<RealType>> solver;
        Matrix<RealType, Dynamic, 1> sol;

        if (sc) {
            solver.analyzePattern(assembler_sc.LHS);
            solver.factorize(assembler_sc.LHS);
            sol = solver.solve(assembler_sc.RHS);
        } else {
            solver.analyzePattern(assembler.LHS);
            solver.factorize(assembler.LHS);
            sol = solver.solve(assembler.RHS);
        }
#endif
#if 0
        Matrix<RealType, Dynamic, 1> sol;
        cg_params <RealType> cgp;
        cgp.histfile = "cuthho_cg_hist.dat";
        cgp.verbose = true;
        cgp.apply_preconditioner = true;
        if (sc) {
            sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
            cgp.max_iter = assembler_sc.LHS.rows();
            conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
        } else {
            sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
            cgp.max_iter = assembler.LHS.rows();
            conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
        }
#endif
        tc.toc();
        std::cout << "Linear solver: " << tc << " seconds" << std::endl;

// ************** POSTPROCESS **************


        postprocess_output <RealType> postoutput;
        std::string filename_interface_uT = "interface_uT.3D";
        std::ofstream interface_file(filename_interface_uT, std::ios::out | std::ios::trunc);

        if (interface_file) {
// instructions
            interface_file << "X   Y   val0   val1" << std::endl;
        } else
            std::cerr << "Interface_file has not been opened" << std::endl;


        auto uT1_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_uT1.dat");
        auto uT2_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_uT2.dat");
//    auto uT_gp  = std::make_shared< gnuplot_output_object_vec<RealType> >("interface_uT.dat");

//    std::string filename_pressure = "interface_p_" + std::to_string(time) + ".dat";
        auto p_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_p.dat");
        auto p1_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_pIN.dat");
        auto p2_gp = std::make_shared < gnuplot_output_object < RealType > > ("interface_pOUT.dat");
//auto p_gp    = std::make_shared< gnuplot_output_object<RealType> >(filename_pressure);

        std::string filename_gammaH = "gamma_H_" + std::to_string(time) + ".dat";
        auto test_gammaH = std::make_shared < gnuplot_output_object < double > > (filename_gammaH);

        std::string filename_press_jump = "pressure_jump_" + std::to_string(time) + ".dat";
        auto test_press_jump = std::make_shared < gnuplot_output_object < double > > (filename_press_jump);

        std::string filename_grad_vel_jump = "grad_vel_jump_" + std::to_string(time) + ".dat";
        auto test_grad_vel_jump = std::make_shared < gnuplot_output_object < double > > (filename_grad_vel_jump);

        std::string filename_vel_n = "vel_u_n_" + std::to_string(time) + ".dat";
        auto test_vel_n = std::make_shared < gnuplot_output_object < double > > (filename_vel_n);
//    auto force_pressure    = std::make_shared< gnuplot_output_object<RealType> >("pressure_force.dat");

        RealType pos = 0.0;
        RealType force_pressure_avg = 0.0, force_pressure_max = 0.0;
        RealType force_gradVel_avg = 0.0, force_gradVel_max = 0.0;

        size_t counter_pt_Gamma = 0;


        tc.tic();
        RealType H1_error = 0.0;
        RealType L2_error = 0.0;
        RealType L2_pressure_error = 0.0;
        RealType l1_u_n_error = 0.0;
        RealType l2_u_n_error = 0.0;
        RealType linf_u_n_error = 0.0;
        size_t counter_interface_pts = 0;
        RealType distance_pts = 0.0;

//    timecounter tc1;
//    tc1.tic();
//    size_t i_global = 0 ;
        auto hdi_cell = hdi.cell_degree();
        auto hdi_face = hdi.face_degree();
        auto msh_vel = velocity.msh;
        auto degree_vel = velocity.degree_FEM;

        timecounter tc_p1;

        tc_p1.tic();

        std::vector <size_t> cut_cell_cointainer, uncut_cell_cointainer;
        size_t counter_cut_cls = 0;
        size_t counter = 0;
        size_t counter_subcls = 0;
        for (auto &cl: msh.cells) {
            if (location(msh, cl) == element_location::ON_INTERFACE) {
                cut_cell_cointainer.push_back(offset(msh, cl));
                counter_cut_cls++;
                for (size_t i_cell = 0; i_cell < cl.user_data.integration_msh.cells.size(); i_cell++) {
//                   connectivity_cells[counter].push_back(counter_subcls);
                    counter_subcls++;
                }

            } else {
                uncut_cell_cointainer.push_back(offset(msh, cl));
            }
            counter++;
        }


        point<RealType, 2> first_point;
        point<RealType, 2> cell_end_point;
        bool first_cut_cell_found = FALSE;

        size_t deg_size = 2; // ESEMPIOOO


        size_t cl_i = 0;
        while (cut_cell_cointainer.size() > 0) {
            if (cl_i > cut_cell_cointainer.size() - 1)
                std::cout << "stop: first_pt = " << first_point << " pt_to_find = " << cell_end_point << std::endl;
            size_t k_offset = cut_cell_cointainer[cl_i];
            auto cl = msh.cells[k_offset];
            auto msh_int = cl.user_data.integration_msh;

            if (!first_cut_cell_found) {
                cut_cell_cointainer.erase(cut_cell_cointainer.begin()); //pop_front();

                post_processing_function(msh, cl, hdi_cell, hdi_face, level_set_function, test_case, assembler_sc, bcs_vel,
                                         sol, velocity, H1_error, L2_error, uT1_gp, uT2_gp, p_gp, interface_file,
                                         L2_pressure_error, l1_u_n_error, l2_u_n_error, linf_u_n_error,
                                         counter_interface_pts, degree, force_pressure_avg, force_pressure_max,
                                         counter_pt_Gamma, test_gammaH, test_press_jump, test_grad_vel_jump, distance_pts,
                                         force_gradVel_max, force_gradVel_avg, test_vel_n, p1_gp, p2_gp);

                first_cut_cell_found = TRUE;
                first_point = *cl.user_data.interface.begin();
                cell_end_point = *(cl.user_data.interface.end() - 1);
                cl_i = 0;
            } else if (first_cut_cell_found && cell_end_point == *cl.user_data.interface.begin() &&
                       !(first_point == cell_end_point)) {
                cut_cell_cointainer.erase(cut_cell_cointainer.begin() + cl_i);
//                cut_cell_cointainer.pop_front();

                post_processing_function(msh, cl, hdi_cell, hdi_face, level_set_function, test_case, assembler_sc, bcs_vel,
                                         sol, velocity, H1_error, L2_error, uT1_gp, uT2_gp, p_gp, interface_file,
                                         L2_pressure_error, l1_u_n_error, l2_u_n_error, linf_u_n_error,
                                         counter_interface_pts, degree, force_pressure_avg, force_pressure_max,
                                         counter_pt_Gamma, test_gammaH, test_press_jump, test_grad_vel_jump, distance_pts,
                                         force_gradVel_max, force_gradVel_avg, test_vel_n, p1_gp, p2_gp);

                cell_end_point = *(cl.user_data.interface.end() - 1);
                cl_i = 0;

            } else if (first_point == cell_end_point)
                break;
            else
                cl_i++;

        }

        std::cout << "First Point curvilinear variable: " << first_point << std::endl;
        tc_p1.toc();
        std::cout << "Interface_analysis time: " << tc_p1 << std::endl;
        tc_p1.tic();
        for (auto &i_cl: uncut_cell_cointainer) {
            auto cl = msh.cells[i_cl];
            post_processing_function(msh, cl, hdi_cell, hdi_face, level_set_function, test_case, assembler_sc, bcs_vel, sol,
                                     velocity, H1_error, L2_error, uT1_gp, uT2_gp, p_gp, interface_file, L2_pressure_error,
                                     l1_u_n_error, l2_u_n_error, linf_u_n_error, counter_interface_pts, degree,
                                     force_pressure_avg, force_pressure_max, counter_pt_Gamma, test_gammaH, test_press_jump,
                                     test_grad_vel_jump, distance_pts, force_gradVel_max, force_gradVel_avg, test_vel_n,
                                     p1_gp, p2_gp);

        }

        tc_p1.toc();
        std::cout << "Not cut cell analysis time: " << tc_p1 << std::endl;


        std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
        std::cout << bold << green << "L2-norm absolute error:               " << std::sqrt(L2_error) << std::endl;
        std::cout << bold << green << "Pressure L2-norm absolute error:      " << std::sqrt(L2_pressure_error) << std::endl;
        std::cout << bold << green << "l1-norm u*n error:               " << l1_u_n_error / counter_interface_pts
                  << std::endl;
        std::cout << bold << green << "l2-norm u*n error:               " << std::sqrt(l2_u_n_error / counter_interface_pts)
                  << std::endl;
        std::cout << bold << green << "linf-norm u*n error:               " << linf_u_n_error << std::endl;


        std::cout << bold << green << "AVG force pressure = " << force_pressure_avg / counter_pt_Gamma << std::endl;
        std::cout << bold << green << "linf-norm force pressure = " << force_pressure_max << std::endl;

        std::cout << bold << green << "AVG force grad_s velocity = " << force_gradVel_avg / counter_pt_Gamma << std::endl;
        std::cout << bold << green << "linf-norm force grad_s velocity = " << force_gradVel_max << std::endl;

        postoutput.add_object(uT1_gp);
        postoutput.add_object(uT2_gp);
        postoutput.add_object(p_gp);
        postoutput.add_object(p1_gp);
        postoutput.add_object(p2_gp);

        postoutput.add_object(test_gammaH);
        postoutput.add_object(test_press_jump);
        postoutput.add_object(test_grad_vel_jump);
        postoutput.add_object(test_vel_n);


        postoutput.write();
        if (interface_file) {

            interface_file.close();

        }


        stokes_test_info <RealType> TI;
        TI.H1_vel = std::sqrt(H1_error);
        TI.L2_vel = std::sqrt(L2_error);
        TI.L2_p = std::sqrt(L2_pressure_error);
        if (1) {
            TI.l1_normal_vel = l1_u_n_error / counter_interface_pts;
            TI.l2_normal_vel = std::sqrt(l2_u_n_error / counter_interface_pts);
            TI.linf_normal_vel = linf_u_n_error;
        }


        if (0) {
/////////////// compute condition number
            SparseMatrix <RealType> Mat;
// Matrix<RealType, Dynamic, Dynamic> Mat;
            if (sc)
                Mat = assembler_sc.LHS;
            else
                Mat = assembler.LHS;

            {
                JacobiSVD <MatrixXd> svd(Mat);
                RealType cond = svd.singularValues()(0)
                                / svd.singularValues()(svd.singularValues().size() - 1);
                std::cout << "cond numb = " << cond << std::endl;
            }

            RealType sigma_max, sigma_min;

// Construct matrix operation object using the wrapper class SparseSymMatProd
            Spectra::SparseSymMatProd <RealType> op(Mat);
// Construct eigen solver object, requesting the largest eigenvalue
            Spectra::SymEigsSolver <RealType, Spectra::LARGEST_MAGN,
            Spectra::SparseSymMatProd<RealType>> max_eigs(&op, 1, 10);
            max_eigs.init();
            max_eigs.compute();
            if (max_eigs.info() == Spectra::SUCCESSFUL)
                sigma_max = max_eigs.eigenvalues()(0);


// Construct eigen solver object, requesting the smallest eigenvalue
            Spectra::SymEigsSolver <RealType, Spectra::SMALLEST_MAGN,
            Spectra::SparseSymMatProd<RealType>> min_eigs(&op, 1, 10);

            min_eigs.init();
            min_eigs.compute();
            if (min_eigs.info() == Spectra::SUCCESSFUL)
                sigma_min = min_eigs.eigenvalues()(0);

// compute condition number
            RealType cond = sigma_max / sigma_min;
            TI.cond = cond;
            std::cout << "sigma_max = " << sigma_max << "   sigma_min = "
                      << sigma_min << "  cond = " << cond
                      << std::endl;
        } else
            TI.cond = 0.0;

        tc.toc();
        std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


        return TI;
    }


    template<typename Mesh, typename Cell, typename LS, typename TC, typename RealType, typename ASS, typename BDRY, typename SOL, typename VEL, typename PP1, typename PP2>
    void
    post_processing_function(const Mesh &msh, Cell &cl, size_t hdi_cell, size_t hdi_face, LS &level_set_function,
                             TC &test_case, ASS &assembler_sc, BDRY &bcs_vel, const SOL &sol, VEL &velocity,
                             RealType &H1_error, RealType &L2_error, PP1 &uT1_gp, PP1 &uT2_gp, PP1 &p_gp,
                             PP2 &interface_file, RealType &L2_pressure_error, RealType &l1_u_n_error,
                             RealType &l2_u_n_error, RealType &linf_u_n_error, size_t &counter_interface_pts,
                             size_t &degree, RealType &force_pressure_avg, RealType &force_pressure_max,
                             size_t &counter_pt_Gamma, PP1 &test_gammaH, PP1 &test_press_jump, PP1 &test_grad_vel_jump,
                             RealType &distance_pts, RealType &force_gradVel_max, RealType &force_gradVel_avg,
                             PP1 &test_vel_n, PP1 &p1_gp, PP1 &p2_gp) {


        vector_cell_basis <cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi_cell);
        RealType kappa_1 = test_case.parms.kappa_1;
        RealType kappa_2 = test_case.parms.kappa_2;

        cell_basis <cuthho_poly_mesh<RealType>, RealType> pb(msh, cl, hdi_face);
        auto cbs = cb.size();
//auto pbs = pb.size();


        level_set_function.cell_assignment(cl); // ----------------------------> TOGLIERE????
        test_case.test_case_cell_assignment(cl); // ----------------------------> TOGLIERE????

        auto sol_vel = test_case.sol_vel;
        auto sol_p = test_case.sol_p;
        auto vel_grad = test_case.vel_grad;
//auto bcs_vel = test_case.bcs_vel;
//auto neumann_jump = test_case.neumann_jump;
        assembler_sc.set_dir_func(bcs_vel); // CAMBIA QUALCOSA?? // ----------------------------> TOGLIERE????


        Matrix<RealType, Dynamic, 1> vel_locdata_n, vel_locdata_p, vel_locdata;
        Matrix<RealType, Dynamic, 1> P_locdata_n, P_locdata_p, P_locdata;
        Matrix<RealType, Dynamic, 1> vel_cell_dofs_n, vel_cell_dofs_p, vel_cell_dofs;

        if (location(msh, cl) == element_location::ON_INTERFACE) {
            vel_locdata_n = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
            vel_locdata_p = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            P_locdata_n = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
            P_locdata_p = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);


            vel_cell_dofs_n = vel_locdata_n.head(cbs);
            vel_cell_dofs_p = vel_locdata_p.head(cbs);


// NOT AGGLO CELL
            if (level_set_function.subcells.size() < 1) {

                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
                size_t i_local = 0;
                for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                    if (level_set_function(ln_Qk, cl_old) > 0.0) {
                        auto phi_HHO = cb.eval_basis(ln_Qk);
                        auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                        velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
                    } else {
                        auto phi_HHO = cb.eval_basis(ln_Qk);
                        auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                        velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
//  velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
//  velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                    }
                }

            } else // AGGLO CELL
            {
//for(auto pt:points(msh,cl))
//   std::cout<<"pt = "<<pt<<std::endl;

                for (size_t i_subcell = 0;
                     i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size(); i_subcell++) {
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
//std::cout<<"offset_old = "<<offset_old<<std::endl;
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                    size_t i_local = 0;
                    for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                        if (level_set_function(ln_Qk, cl_old) > 0.0) {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                            velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
                        } else {
                            auto phi_HHO = cb.eval_basis(ln_Qk);
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                            velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
//  velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
//  velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                        }
                    }

                }
            }

//            tc2.toc();
//            std::cout<<"time tc2 3 = "<<tc2<<std::endl;
//            tc2.tic();
            auto qps_n = integrate(msh, cl, 2 * hdi_cell, element_location::IN_NEGATIVE_SIDE);
            for (auto &qp: qps_n) {
// Compute H1-error //
                auto t_dphi = cb.eval_gradients(qp.first);
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++)
                    grad += vel_cell_dofs_n(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                H1_error += qp.second * kappa_1 * inner_product(grad_sym_diff, grad_sym_diff);

// Compute L2-error //
                auto t_phi = cb.eval_basis(qp.first);
                auto v = t_phi.transpose() * vel_cell_dofs_n;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                L2_error += qp.second * kappa_1 * sol_diff.dot(sol_diff);

                uT1_gp->add_data(qp.first, v(0));
                uT2_gp->add_data(qp.first, v(1));

                interface_file << qp.first.x() << "   " << qp.first.y() << "   " << v(0) << "   " << v(1) << std::endl;



// L2 - pressure - error //
                auto p_phi = pb.eval_basis(qp.first);
                RealType p_num = p_phi.dot(P_locdata_n);
                RealType p_diff = test_case.sol_p(qp.first) - p_num; // era test_case STE
//                auto p_prova = test_case.sol_p( qp.first ) ;
//                std::cout<<"In pt = "<<qp.first<<" --> pressure ANAL  = "<<p_prova<<" , pressure NUM = "<< p_num<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                L2_pressure_error += qp.second * p_diff * p_diff / kappa_1;
                p_gp->add_data(qp.first, p_num);
                p1_gp->add_data(qp.first, p_num);
            }

            auto qps_p = integrate(msh, cl, 2 * hdi_cell, element_location::IN_POSITIVE_SIDE);
            for (auto &qp: qps_p) {
// Compute H1-error //
                auto t_dphi = cb.eval_gradients(qp.first);
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++)
                    grad += vel_cell_dofs_p(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                H1_error += qp.second * kappa_2 * inner_product(grad_sym_diff, grad_sym_diff);

// Compute L2-error //
                auto t_phi = cb.eval_basis(qp.first);
                auto v = t_phi.transpose() * vel_cell_dofs_p;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                L2_error += qp.second * kappa_2 * sol_diff.dot(sol_diff);

                uT1_gp->add_data(qp.first, v(0));
                uT2_gp->add_data(qp.first, v(1));
//                uT_gp->add_data( qp.first, std::make_pair(v(0),v(1)) );
                if (interface_file) {

                    interface_file << qp.first.x() << "   " << qp.first.y() << "   " << v(0) << "   " << v(1) << std::endl;

                }
// L2 - pressure - error //
                auto p_phi = pb.eval_basis(qp.first);
                RealType p_num = p_phi.dot(P_locdata_p);
                RealType p_diff = test_case.sol_p(qp.first) - p_num; // era test_case STE
//auto p_prova = test_case.sol_p( qp.first ) ;
//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                L2_pressure_error += qp.second * p_diff * p_diff / kappa_2;

                p_gp->add_data(qp.first, p_num);
                p2_gp->add_data(qp.first, p_num);
            }
            if (1) {
                for (auto &interface_point: cl.user_data.interface) {
                    auto t_phi = cb.eval_basis(interface_point);
                    auto v = t_phi.transpose() * vel_cell_dofs_p;
                    auto n = level_set_function.normal(interface_point);
                    auto v_n = v.dot(n);
                    l1_u_n_error += std::abs(v_n);
                    l2_u_n_error += pow(v_n, 2.0);
                    linf_u_n_error = std::max(linf_u_n_error, std::abs(v_n));
                    counter_interface_pts++;
                }
            }
//            tc2.toc();
//            std::cout<<"time tc2 2 = "<<tc2<<std::endl;

            if (1) // analysis power of pressure
            {
                auto parametric_interface = test_case.parametric_interface;
                auto gamma = test_case.gamma;
                auto msh_int = cl.user_data.integration_msh;
                auto global_cells_i = parametric_interface.get_global_cells_interface(msh, cl);
                size_t degree_curve = msh_int.degree_curve;
                RealType tot = 10.0;
//                Interface_parametrisation_mesh1d curve(degree_curve) ;
//            degree += 3*degree_curve -4 ; // 2*degree_curve ; // TO BE CHECKED
//            auto qps = edge_quadrature<RealType>(degree);
                auto neumann = test_case.neumann_jump;
                for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++) {
                    auto pts = points(msh_int, msh_int.cells[i_cell]);
                    size_t global_cl_i = global_cells_i[i_cell];
//                auto qp_old = 0.5 *(*(qps.begin())).first.x() + 0.5;
//                auto p = parametric_interface(t , pts , degree_curve ) ;
//                point<RealType,2> pt_old ; //= typename Mesh::point_type( p(0) , p(1) ) ;

                    for (RealType i = 0.0; i <= tot; i++) {
                        auto t = 0.0 + i / tot;
                        auto p = parametric_interface(t, pts, degree_curve);
                        point<RealType, 2> pt = typename Mesh::point_type(p(0), p(1));
//                    if( t == 0.0 )
//                        pt_old = pt;
                        auto p_phi = pb.eval_basis(pt);
                        RealType p_pos = p_phi.dot(P_locdata_p);
                        RealType p_neg = p_phi.dot(P_locdata_n);


                        auto phi_HHO = cb.eval_basis(pt);
                        auto t_dphi = cb.eval_gradients(pt);
                        Matrix<RealType, 2, 2> grad_p = Matrix<RealType, 2, 2>::Zero();
                        Matrix<RealType, 2, 2> grad_n = Matrix<RealType, 2, 2>::Zero();

                        for (size_t i = 1; i < cbs; i++) {
                            grad_p += vel_cell_dofs_p(i) * t_dphi[i].block(0, 0, 2, 2);
                            grad_n += vel_cell_dofs_n(i) * t_dphi[i].block(0, 0, 2, 2);

                        }
                        Matrix<RealType, 2, 2> grad_sym_p = 0.5 * (grad_p + grad_p.transpose());
                        Matrix<RealType, 2, 2> grad_sym_n = 0.5 * (grad_n + grad_n.transpose());
                        auto vel_n = phi_HHO.transpose() * vel_cell_dofs_n;
                        Matrix<RealType, 2, 1> phi_n = parametric_interface.normal_cont(t, global_cl_i);
                        RealType val_u_n = (vel_n).transpose() * (phi_n);

                        auto val_p = (p_pos - p_neg);
                        RealType val_u =
                                (phi_n.transpose()) * (2.0 * kappa_1 * grad_sym_n - 2.0 * kappa_2 * grad_sym_p) * (phi_n);

//                    RealType val_u ;
//                    if( signbit(phi_n(0)) == signbit(grads_u_n(0)) && signbit(phi_n(1)) == signbit(grads_u_n(1)) )
//                        val_u = grads_u_n.norm();
//                    else
//                        val_u = -grads_u_n.norm();
                        point<RealType, 2> curv_var = typename Mesh::point_type(distance_pts, 0.0);
                        auto val_H = gamma * parametric_interface.curvature_cont(t, global_cl_i);
                        test_press_jump->add_data(curv_var, val_p);
                        test_gammaH->add_data(curv_var, val_H);
                        test_grad_vel_jump->add_data(curv_var, val_u);

                        test_vel_n->add_data(curv_var, val_u_n);

                        force_pressure_avg += val_p / val_H;
                        force_pressure_max = std::max(force_pressure_max, std::abs(val_p / val_H));

                        force_gradVel_avg += val_u / val_H;
                        force_gradVel_max = std::max(force_gradVel_max, std::abs(val_u / val_H));


                        counter_pt_Gamma++;

                        RealType dist;

                        if (t == 1)
                            dist = 0.0;
                        else
                            dist = (parametric_interface(t + 1.0 / tot, pts, degree_curve) - p).norm();

                        distance_pts += dist;

                    }
                }


            }

        } else {
//            tc2.tic();
            vel_locdata = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            P_locdata = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);

            vel_cell_dofs = vel_locdata.head(cbs);



// NOT AGGLO CELL
            if (level_set_function.subcells.size() < 1) {
//                assert(level_set_function.agglo_LS_cl.user_data.offset_subcells.size()==2);
//                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                    auto phi_HHO = cb.eval_basis(ln_Qk);
                    auto vel = phi_HHO.transpose() * vel_cell_dofs;
                    velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                    velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                    i_local++;

                }

            } else // AGGLO CELL
            {
                for (size_t i_subcell = 0;
                     i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size(); i_subcell++) {
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
//std::cout<<"offset_old = "<<offset_old<<std::endl;
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = cl_old.user_data.Lagrange_nodes_Qk;
//                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                    size_t i_local = 0;
                    for (const auto &ln_Qk: Lagrange_nodes_Qk) {
                        auto phi_HHO = cb.eval_basis(ln_Qk);
                        auto vel = phi_HHO.transpose() * vel_cell_dofs;
                        velocity.sol_HHO.first(i_local, offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local, offset_old) = vel(1);
//std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;

                    }

                }
            }
//            tc2.toc();
//            std::cout<<"time tc2 1 = "<<tc2<<std::endl;
//            tc2.tic();
            RealType kappa = test_case.parms.kappa_1;
            if (location(msh, cl) == element_location::IN_POSITIVE_SIDE)
                kappa = test_case.parms.kappa_2;

            auto qps = integrate(msh, cl, 2 * hdi_cell);
            for (auto &qp: qps) {
// Compute H1-error //
                auto t_dphi = cb.eval_gradients(qp.first);
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++)
                    grad += vel_cell_dofs(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
//                H1_error += qp.second * inner_product(grad_diff , grad_diff);
                Matrix<RealType, 2, 2> grad_sym_diff = 0.5 * (grad_diff + grad_diff.transpose());
                H1_error += qp.second * kappa * inner_product(grad_sym_diff, grad_sym_diff);

// Compute L2-error //
                auto t_phi = cb.eval_basis(qp.first);
                auto v = t_phi.transpose() * vel_cell_dofs;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
//                L2_error += qp.second * sol_diff.dot(sol_diff);
                L2_error += qp.second * kappa * sol_diff.dot(sol_diff);

                uT1_gp->add_data(qp.first, v(0));
                uT2_gp->add_data(qp.first, v(1));
//                uT_gp->add_data( qp.first, std::make_pair(v(0),v(1)) );
                if (interface_file) {

                    interface_file << qp.first.x() << "   " << qp.first.y() << "   " << v(0) << "   " << v(1) << std::endl;

                }

// L2 - pressure - error //
                auto p_phi = pb.eval_basis(qp.first);
                RealType p_num = p_phi.dot(P_locdata);
                RealType p_diff = test_case.sol_p(qp.first) - p_num; // era test_case STE
//auto p_prova = test_case.sol_p( qp.first ) ;
//std::cout<<"pressure ANAL  = "<<p_prova<<std::endl;
//                L2_pressure_error += qp.second * p_diff * p_diff;
                L2_pressure_error += qp.second * p_diff * p_diff / kappa;

                p_gp->add_data(qp.first, p_num);
                if (level_set_function(qp.first, cl) > 0.0)
                    p2_gp->add_data(qp.first, p_num);
                else
                    p1_gp->add_data(qp.first, p_num);
            }
//            tc2.toc();
//            std::cout<<"time tc2 0 = "<<tc2<<std::endl;
        }


    }



}

// unsteady Stokes interface problem
namespace unsteady_stokes_terms {

    template<typename Mesh>
    Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>
    make_mass_matrix
            (const Mesh &msh, const typename Mesh::cell_type &cl, const hho_degree_info &di) {
        using T = typename Mesh::coordinate_type;
        typedef Matrix <T, Dynamic, Dynamic> matrix_type;
//typedef Matrix<T, Dynamic, 1>       vector_type;

        const auto celdeg = di.cell_degree();
//const auto facdeg  = di.face_degree();
//const auto graddeg = di.grad_degree();

        vector_cell_basis <Mesh, T> cb(msh, cl, celdeg);
//sym_matrix_cell_basis<Mesh,T>     gb(msh, cl, graddeg);

        auto cbs = vector_cell_basis<Mesh, T>::size(celdeg);
//auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
//auto gbs = sym_matrix_cell_basis<Mesh,T>::size(graddeg);

//const auto num_faces = faces(msh, cl).size();
        matrix_type mass_mat = matrix_type::Zero(cbs, cbs);
//matrix_type         gr_lhs = matrix_type::Zero(gbs, gbs);
//matrix_type         gr_rhs = matrix_type::Zero(gbs, cbs + num_faces * fbs);

        if (celdeg > 0) {
            const auto qps = integrate(msh, cl, 2 * celdeg); // - 1 + facdeg); //HO ANCHE ADD 2*
            for (auto &qp: qps) {
//const auto c_dphi = cb.eval_gradients(qp.first);
//const auto g_phi  = gb.eval_basis(qp.first);
                const auto c_phi = cb.eval_basis(qp.first);
                mass_mat.block(0, 0, cbs, cbs) += qp.second * c_phi * c_phi.transpose();
//gr_lhs.block(0, 0, gbs, gbs) += qp.second * inner_product(g_phi, g_phi);
// we use here the symmetry of the basis gb
//gr_rhs.block(0, 0, gbs, cbs) += qp.second * inner_product(g_phi, c_dphi);
            }
        }


        return mass_mat;
    }

    template<typename T, size_t ET>
    Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
    make_mass_matrix
            (const cuthho_mesh <T, ET> &msh, const typename cuthho_mesh<T, ET>::cell_type &cl,
             const hho_degree_info &di, element_location where) {
        if (!is_cut(msh, cl))
            return make_mass_matrix(msh, cl, di);

        typedef Matrix <T, Dynamic, Dynamic> matrix_type;
//typedef Matrix<T, Dynamic, 1>       vector_type;

        const auto celdeg = di.cell_degree();

        vector_cell_basis <cuthho_mesh<T, ET>, T> cb(msh, cl, celdeg);


        auto cbs = vector_cell_basis < cuthho_mesh < T, ET>, T > ::size(celdeg);

        matrix_type mass_mat = matrix_type::Zero(cbs, cbs);


        const auto qps = integrate(msh, cl, 2 * celdeg, where);
        for (auto &qp: qps) {
            const auto c_phi = cb.eval_basis(qp.first);


            mass_mat.block(0, 0, cbs, cbs) += qp.second * c_phi * c_phi.transpose();
        }


        return mass_mat;
    }


    template<typename Mesh, typename Finite_Element, typename Velocity_HHO>
    Matrix<typename Mesh::coordinate_type, Dynamic, 1>
    make_term_rhs_time
            (const Mesh &msh, const typename Mesh::cell_type &cl, const hho_degree_info &di, const Finite_Element &fe_data,
             Velocity_HHO &vel_old) {
// msh = msh t^{N+1}
// msh_n = msh t^N
// msh_orig = msh t^0

        using T = typename Mesh::coordinate_type;
        typedef Matrix <T, Dynamic, Dynamic> matrix_type;

        typedef Matrix<T, Dynamic, 1> vector_type;

        const auto celdeg = di.cell_degree();
        auto cbs = vector_cell_basis<Mesh, T>::size(celdeg);


        vector_type ret = matrix_type::Zero(cbs, 1);


        Mesh msh_orig = fe_data.msh;
        Mesh msh_n = fe_data.msh_last;
        auto cl_n_offsets = fe_data.mapping_S(cl);


//    std::cout<<"k^{n+1} = "<<offset(msh,cl)<<" -> S(k^{n+1}): "<<std::endl;
//    for(auto& i : cl_n_offsets)
//        std::cout<<" ( K^n = "<<i.first <<" , K^0 = "<<i.second<<" ) --";
//    std::cout<<std::endl;


//auto cl_n_prova = fe_data.mapping_S_prova( cl ) ;
//std::cout<<"cl_n_prova = "<<std::endl;
//for(auto& i : cl_n_prova)
//    std::cout<<" ("<<i.first <<" , "<<i.second<<" ) --";
//std::cout<<std::endl;


        for (auto &i: cl_n_offsets) {
            matrix_type mass_mat = matrix_type::Zero(cbs, cbs);

            auto i_cl_n = i.first;
            auto vel_cell_old = vel_old.vel_global.col(i_cl_n);
            auto cl_n = msh_n.cells[i_cl_n];
            vector_cell_basis <Mesh, T> cb_n(msh, cl_n, celdeg);
            vector_cell_basis <Mesh, T> cb(msh, cl, celdeg);

//for(auto& i_integration: cl_n.user_data.offset_subcells )
//{
            auto cl_integration = msh_orig.cells[i.second]; // [i_integration]


            if (celdeg > 0) {
//std::cout<<"Potrebbe mancare un ciclo di integrazione qua sulle sottocelle! capire se ci va"<<std::endl;
                const auto qps = integrate(msh_orig, cl_integration, 2 * celdeg);
//It was celdeg-1 + facdeg,  I ADDED 2*
                for (auto &qp: qps) {
//const auto c_dphi = cb.eval_gradients(qp.first);
//const auto g_phi  = gb.eval_basis(qp.first);
                    const auto c_phi = cb.eval_basis(qp.first);
                    const auto c_phiR = cb_n.eval_basis(qp.first);
                    mass_mat.block(0, 0, cbs, cbs) += qp.second * c_phi * c_phiR.transpose();
//gr_lhs.block(0, 0, gbs, gbs) += qp.second * inner_product(g_phi, g_phi);
// we use here the symmetry of the basis gb
//gr_rhs.block(0, 0, gbs, cbs) += qp.second * inner_product(g_phi, c_dphi);
                }
            }

//}

            ret += mass_mat * vel_cell_old;


        }


        return ret;
    }

    template<typename Mesh, typename Finite_Element, typename Velocity_HHO>
    Matrix<typename Mesh::coordinate_type, Dynamic, 1>
    make_term_rhs_time
            (const Mesh &msh, const typename Mesh::cell_type &cl, const hho_degree_info &di, const Finite_Element &fe_data,
             Velocity_HHO &vel_old, element_location where) {
// msh = msh t^{N+1}
// msh_n = msh t^N
// msh_orig = msh t^0
        if (!is_cut(msh, cl))
            return make_term_rhs_time(msh, cl, di, fe_data, vel_old);


        using T = typename Mesh::coordinate_type;
        typedef Matrix <T, Dynamic, Dynamic> matrix_type;

        typedef Matrix<T, Dynamic, 1> vector_type;

        const auto celdeg = di.cell_degree();
        auto cbs = vector_cell_basis<Mesh, T>::size(celdeg);

        matrix_type mass_mat = matrix_type::Zero(cbs, cbs);
        vector_type ret = matrix_type::Zero(cbs, 1);
        vector_type vel_cell_old = matrix_type::Zero(cbs, 1);

        Mesh msh_orig = fe_data.msh;
        Mesh msh_n = fe_data.msh_last;
        auto cl_n_offsets = fe_data.mapping_S(cl);

//    std::cout<<"k^{n+1} = "<<offset(msh,cl)<<" -> S(k^{n+1}): "<<std::endl;
//    for(auto& i : cl_n_offsets)
//        std::cout<<" ( K^n = "<<i.first <<" , K^0 = "<<i.second<<" ) --";
//    std::cout<<std::endl;

//auto cl_n_prova = fe_data.mapping_S_prova( cl ) ;
//std::cout<<"cl_n_prova = "<<std::endl;
//for(auto& i : cl_n_prova)
//    std::cout<<" ("<<i.first <<" , "<<i.second<<" ) --";
//std::cout<<std::endl;


        for (auto &i: cl_n_offsets) {
            auto i_cl_n = i.first;

            if (where == element_location::IN_NEGATIVE_SIDE)
                vel_cell_old = vel_old.vel_global_n.col(i_cl_n);
            else
                vel_cell_old = vel_old.vel_global_p.col(i_cl_n);

            auto cl_n = msh_n.cells[i_cl_n];
            vector_cell_basis <Mesh, T> cb_n(msh, cl_n, celdeg);

            auto cl_integration = msh_orig.cells[i.second];

            vector_cell_basis <Mesh, T> cb(msh, cl, celdeg);

            if (celdeg > 0) {
//const auto qps = integrate( msh_orig , cl_integration , 2*celdeg ) ;
                const auto qps = integrate(msh_orig, cl_integration, 2 * celdeg, where);
//It was celdeg-1 + facdeg,  I ADDED 2*
                for (auto &qp: qps) {
//const auto c_dphi = cb.eval_gradients(qp.first);
//const auto g_phi  = gb.eval_basis(qp.first);
                    const auto c_phi = cb.eval_basis(qp.first);
                    const auto c_phiR = cb_n.eval_basis(qp.first);
                    mass_mat.block(0, 0, cbs, cbs) += qp.second * c_phi * c_phiR.transpose();
//gr_lhs.block(0, 0, gbs, gbs) += qp.second * inner_product(g_phi, g_phi);
// we use here the symmetry of the basis gb
//gr_rhs.block(0, 0, gbs, cbs) += qp.second * inner_product(g_phi, c_dphi);
                }


            }
            ret += mass_mat * vel_cell_old;


        }


        return ret;
    }





/*****************************************************************************
*   PREVIOUS CODE STOKES HHO
*****************************************************************************/



///////////////////////   FICTITIOUS DOMAIN METHODS  ///////////////////////////

    template<typename T, size_t ET, typename testType>
    class stokes_fictdom_method {
        using Mat = Matrix<T, Dynamic, Dynamic>;
        using Vect = Matrix<T, Dynamic, 1>;
        using Mesh = cuthho_mesh<T, ET>;

    protected:
        bool sym_grad;

        stokes_fictdom_method(bool sym)
                : sym_grad(sym) {
        }

        virtual std::pair <std::pair<Mat, Mat>, std::pair<Vect, Vect>>
        make_contrib_cut(const Mesh &msh, const typename Mesh::cell_type &cl,
                         const testType test_case, const hho_degree_info hdi,
                         const element_location where = element_location::IN_NEGATIVE_SIDE,
                         const params <T> &parms = params<T>()) {
        }

    public:
        std::pair <std::pair<Mat, Mat>, std::pair<Vect, Vect>>
        make_contrib_uncut(const Mesh &msh, const typename Mesh::cell_type &cl,
                           const hho_degree_info hdi, const testType test_case) {
            Mat gr2;
            if (sym_grad)
                gr2 = make_hho_gradrec_sym_matrix(msh, cl, hdi).second;
            else
                gr2 = make_hho_gradrec_matrix(msh, cl, hdi).second;
            Mat stab = make_hho_vector_naive_stabilization(msh, cl, hdi);
            Mat lc = gr2 + stab;
            auto dr = make_hho_divergence_reconstruction(msh, cl, hdi);
            Vect f = make_vector_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
            Vect p_rhs = Vect::Zero(dr.first.rows());
            return std::make_pair(std::make_pair(lc, dr.second), std::make_pair(f, p_rhs));
        }


        std::pair <std::pair<Mat, Mat>, std::pair<Vect, Vect>>
        make_contrib(const Mesh &msh, const typename Mesh::cell_type &cl,
                     const testType test_case, const hho_degree_info hdi,
                     const element_location where = element_location::IN_NEGATIVE_SIDE,
                     const params <T> &parms = params<T>()) {
            if (location(msh, cl) == where)
                return make_contrib_uncut(msh, cl, hdi, test_case);
            else if (location(msh, cl) != element_location::ON_INTERFACE) {
                Mat lc;
                Vect f;
                return std::make_pair(std::make_pair(lc, lc), std::make_pair(f, f));
            } else // on interface
                return make_contrib_cut(msh, cl, test_case, hdi, where, parms);
        }
    };

/////////////////////////  GRADREC_FICTITIOUS_METHOD

    template<typename T, size_t ET, typename testType>
    class gradrec_stokes_fictdom_method :
            public stokes_fictdom_method<T, ET, testType> {
        using Mat = Matrix<T, Dynamic, Dynamic>;
        using Vect = Matrix<T, Dynamic, 1>;
        using Mesh = cuthho_mesh<T, ET>;

    public:
        T eta;

        gradrec_stokes_fictdom_method(T eta_, bool sym)
                : stokes_fictdom_method<T, ET, testType>(sym), eta(eta_) {
        }

        std::pair <std::pair<Mat, Mat>, std::pair<Vect, Vect>>
        make_contrib_cut(const Mesh &msh, const typename Mesh::cell_type &cl,
                         const testType test_case, const hho_degree_info hdi,
                         const element_location where = element_location::IN_NEGATIVE_SIDE,
                         const params <T> &parms = params<T>()) {
// LHS
            Mat gr1, gr2;
            if (this->sym_grad) {
                auto gr = make_hho_gradrec_sym_matrix(msh, cl, test_case.level_set_, hdi, where, 1.0);
                gr1 = gr.first;
                gr2 = gr.second;
            } else {
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
            auto cbs = vector_cell_basis<Mesh, T>::size(celdeg);

            Vect f = Vect::Zero(lc.rows());
            f.block(0, 0, cbs, 1) += make_vector_rhs(msh, cl, celdeg, test_case.rhs_fun, where);
            f.block(0, 0, cbs, 1) += make_vector_rhs_penalty(msh, cl, celdeg, test_case.bcs_vel, eta);
            f += make_vector_GR_rhs(msh, cl, celdeg, test_case.bcs_vel, test_case.level_set_,
                                    gr1, this->sym_grad);
            auto p_rhs = make_pressure_rhs(msh, cl, hdi.face_degree(), where,
                                           test_case.level_set_, test_case.bcs_vel);

            return std::make_pair(std::make_pair(lc, dr.second), std::make_pair(f, p_rhs));
        }
    };


    template<typename T, size_t ET, typename testType>
    auto make_gradrec_stokes_fictdom_method(const cuthho_mesh <T, ET> &msh, const T eta_,
                                            const testType test_case, bool sym) {
        return gradrec_stokes_fictdom_method<T, ET, testType>(eta_, sym);
    }

///////////////////////////

    template<typename Mesh, typename testType>
    stokes_test_info<typename Mesh::coordinate_type>
    run_cuthho_fictdom(const Mesh &msh, size_t degree, testType test_case) {
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
        std::vector <RealType> cut_cell_markers;
        for (auto &cl: msh.cells) {
            if (location(msh, cl) == element_location::IN_POSITIVE_SIDE)
                cut_cell_markers.push_back(1.0);
            else if (location(msh, cl) == element_location::IN_NEGATIVE_SIDE)
                cut_cell_markers.push_back(-1.0);
            else if (location(msh, cl) == element_location::ON_INTERFACE)
                cut_cell_markers.push_back(0.0);
            else
                throw std::logic_error("shouldn't have arrived here...");
        }
        silo.add_variable("mesh", "cut_cells", cut_cell_markers.data(), cut_cell_markers.size(), zonal_variable_t);

/************** MAKE A SILO VARIABLE FOR LEVEL SET FUNCTION **************/
        std::vector <RealType> level_set_vals;
        for (auto &pt: msh.points)
            level_set_vals.push_back(level_set_function(pt));
        silo.add_variable("mesh", "level_set", level_set_vals.data(), level_set_vals.size(), nodal_variable_t);

/************** MAKE A SILO VARIABLE FOR NODE POSITIONING **************/
        std::vector <RealType> node_pos;
        for (auto &n: msh.nodes)
            node_pos.push_back(location(msh, n) == element_location::IN_POSITIVE_SIDE ? +1.0 : -1.0);
        silo.add_variable("mesh", "node_pos", node_pos.data(), node_pos.size(), nodal_variable_t);


        timecounter tc;

        bool sc = true; // static condensation

/************** ASSEMBLE PROBLEM **************/
        hho_degree_info hdi(degree + 1, degree);
//    std::cout<<"HHO degree k both cells and faces."<<std::endl;
//    hho_degree_info hdi(degree, degree);

        element_location where = element_location::IN_NEGATIVE_SIDE;

        tc.tic();
        auto assembler = make_stokes_fict_assembler(msh, bcs_fun, hdi, where);
        auto assembler_sc = make_stokes_fict_condensed_assembler(msh, bcs_fun, hdi, where);

// method with gradient reconstruction (penalty-free)
        bool sym = true; // true
        auto class_meth = make_gradrec_stokes_fictdom_method(msh, 1.0, test_case, sym);

        for (auto &cl: msh.cells) {
            if (!(location(msh, cl) == element_location::ON_INTERFACE || location(msh, cl) == where))
                continue;
            auto contrib = class_meth.make_contrib(msh, cl, test_case, hdi,
                                                   element_location::IN_NEGATIVE_SIDE);
            auto lc_A = contrib.first.first;
            auto lc_B = -contrib.first.second;
            auto rhs_A = contrib.second.first;
            auto rhs_B = -contrib.second.second;

            if (sc)
                assembler_sc.assemble(msh, cl, lc_A, lc_B, rhs_A, rhs_B);
            else
                assembler.assemble(msh, cl, lc_A, lc_B, rhs_A, rhs_B);


        }

        if (sc)
            assembler_sc.finalize();
        else
            assembler.finalize();

        tc.toc();
        std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

        if (sc)
            std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
        else
            std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

        std::cout << "Cells: " << msh.cells.size() << std::endl;
        std::cout << "Faces: " << msh.faces.size() << std::endl;


/************** SOLVE **************/
        tc.tic();
#if 1
        SparseLU <SparseMatrix<RealType>> solver;
        Matrix<RealType, Dynamic, 1> sol;

        if (sc) {
            solver.analyzePattern(assembler_sc.LHS);
            solver.factorize(assembler_sc.LHS);
            sol = solver.solve(assembler_sc.RHS);
        } else {
            solver.analyzePattern(assembler.LHS);
            solver.factorize(assembler.LHS);
            sol = solver.solve(assembler.RHS);
        }
#endif
#if 0
        Matrix<RealType, Dynamic, 1> sol;
        cg_params <RealType> cgp;
        cgp.histfile = "cuthho_cg_hist.dat";
        cgp.verbose = true;
        cgp.apply_preconditioner = true;
        if (sc) {
            sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
            cgp.max_iter = assembler_sc.LHS.rows();
            conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
        } else {
            sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
            cgp.max_iter = assembler.LHS.rows();
            conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
        }
#endif
        tc.toc();
        std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

/************** POSTPROCESS **************/



        postprocess_output <RealType> postoutput;

        auto uT_l2_gp = std::make_shared < gnuplot_output_object < RealType > > ("fictdom_uT_norm.dat");
        auto uT1_gp = std::make_shared < gnuplot_output_object < RealType > > ("fictdom_uT1.dat");
        auto uT2_gp = std::make_shared < gnuplot_output_object < RealType > > ("fictdom_uT2.dat");
        auto p_gp = std::make_shared < gnuplot_output_object < RealType > > ("fictdom_p.dat");

        tc.tic();
        RealType H1_error = 0.0;
        RealType L2_error = 0.0;
        RealType L2_pressure_error = 0.0;
        for (auto &cl: msh.cells) {
            RealType L2_pressure_error_cell = 0.0;
            bool hide_fict_dom = true; // hide the fictitious domain in the gnuplot outputs


            if (hide_fict_dom && location(msh, cl) == element_location::IN_POSITIVE_SIDE)
                continue;

            vector_cell_basis <cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
            cell_basis <cuthho_poly_mesh<RealType>, RealType> s_cb(msh, cl, hdi.face_degree());

            auto cbs = cb.size();

            Matrix<RealType, Dynamic, 1> locdata_vel, locdata_p;
            if (sc) {
                locdata_vel = assembler_sc.take_velocity(msh, cl, sol);
                locdata_p = assembler_sc.take_pressure(msh, cl, sol);
            } else {
                locdata_vel = assembler.take_velocity(msh, cl, sol);
                locdata_p = assembler.take_pressure(msh, cl, sol);
            }

            Matrix<RealType, Dynamic, 1> cell_v_dofs = locdata_vel.head(cbs);

//auto bar = barycenter(msh, cl, element_location::IN_NEGATIVE_SIDE);

            if (location(msh, cl) == element_location::IN_NEGATIVE_SIDE ||
                location(msh, cl) == element_location::ON_INTERFACE) {
//Matrix<RealType, 1, 2> real_grad_int = Matrix<RealType, 1, 2>::Zero();
//Matrix<RealType, 1, 2> comp_grad_int = Matrix<RealType, 1, 2>::Zero();
                auto qps = integrate(msh, cl, 2 * hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
                for (auto &qp: qps) {
/* Compute H1-error */
                    auto t_dphi = cb.eval_gradients(qp.first);
                    Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                    for (size_t i = 0; i < cbs; i++)
                        grad += cell_v_dofs(i) * t_dphi[i].block(0, 0, 2, 2);

                    Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;

                    H1_error += qp.second * inner_product(grad_diff, grad_diff);


/* L2 - error */
                    auto t_phi = cb.eval_basis(qp.first);
                    auto v = t_phi.transpose() * cell_v_dofs;
                    Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                    L2_error += qp.second * sol_diff.dot(sol_diff);

                    uT1_gp->add_data(qp.first, v(0));
                    uT2_gp->add_data(qp.first, v(1));
                    uT_l2_gp->add_data(qp.first, std::sqrt(v(0) * v(0) + v(1) * v(1)));

/* L2 - pressure - error */
                    auto s_cphi = s_cb.eval_basis(qp.first);
                    RealType p_num = s_cphi.dot(locdata_p);
                    RealType p_diff = test_case.sol_p(qp.first) - p_num;
                    L2_pressure_error += qp.second * p_diff * p_diff;
                    L2_pressure_error_cell += qp.second * p_diff * p_diff;
                    p_gp->add_data(qp.first, p_num);

                }
                std::cout << "L2_pressure_error_cell" << L2_pressure_error_cell << std::endl;
            }

        }

        std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
        std::cout << bold << green << "L2 - pressure - error:                " << std::sqrt(L2_pressure_error) << std::endl;

        postoutput.add_object(uT_l2_gp);
        postoutput.add_object(uT1_gp);
        postoutput.add_object(uT2_gp);
        postoutput.add_object(p_gp);
        postoutput.write();

        stokes_test_info <RealType> TI;
        TI.H1_vel = std::sqrt(H1_error);
        TI.L2_vel = std::sqrt(L2_error);
        TI.L2_p = std::sqrt(L2_pressure_error);

        tc.toc();
        std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


        SparseMatrix <RealType> Mat;
// Matrix<RealType, Dynamic, Dynamic> Mat;
        if (sc)
            Mat = assembler_sc.LHS;
        else
            Mat = assembler.LHS;


// Add by Stefano
        Eigen::BDCSVD <Eigen::MatrixXd> SVD(Mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
        double cond = SVD.singularValues()(0) / SVD.singularValues()(SVD.singularValues().size() - 1);
        std::cout << "cond_numb = " << cond << std::endl;

        return TI;
    }



//////////////////////////////  INTERFACE METHODS  ///////////////////////////

    template<typename T, size_t ET, typename testType>
    class stokes_interface_method {
        using Mat = Matrix<T, Dynamic, Dynamic>;
        using Vect = Matrix<T, Dynamic, 1>;
        using Mesh = cuthho_mesh<T, ET>;

    protected:
        bool sym_grad;

        stokes_interface_method(bool sym)
                : sym_grad(sym) {
        }

        stokes_interface_method(stokes_interface_method &other) {
            sym_grad = other.sym_grad;

        }

        stokes_interface_method(const stokes_interface_method &other) {
            sym_grad = other.sym_grad;

        }

        virtual std::pair <Mat, Vect>
        make_contrib_cut(const Mesh &msh, const typename Mesh::cell_type &cl,
                         const testType &test_case, const hho_degree_info &hdi) {
        }

    public:
        std::pair <Mat, Vect>
        make_contrib_uncut(const Mesh &msh, const typename Mesh::cell_type &cl,
                           const hho_degree_info &hdi, const testType &test_case) {
            T kappa;
            if (location(msh, cl) == element_location::IN_NEGATIVE_SIDE)
                kappa = test_case.parms.kappa_1;
            else
                kappa = test_case.parms.kappa_2;

            Mat gr2;
            if (sym_grad)
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
            Mat lhs = Mat::Zero(loc_size, loc_size);
            Vect rhs = Vect::Zero(loc_size);

            lhs.block(0, 0, v_size, v_size) = lc;
            lhs.block(0, v_size, v_size, p_size) = -dr.second.transpose();
            lhs.block(v_size, 0, p_size, v_size) = -dr.second;

            rhs.head(f.rows()) = f;
            return std::make_pair(lhs, rhs);
        }


        std::pair <Mat, Vect>
        make_contrib(const Mesh &msh, const typename Mesh::cell_type &cl,
                     const testType &test_case, const hho_degree_info &hdi) {
            if (location(msh, cl) != element_location::ON_INTERFACE)
                return make_contrib_uncut(msh, cl, hdi, test_case);
            else // on interface
                return make_contrib_cut(msh, cl, test_case, hdi);
        }
    };


    template<typename T, size_t ET, typename testType>
    class stokes_interface_method_ref_pts {
        using Mat = Matrix<T, Dynamic, Dynamic>;
        using Vect = Matrix<T, Dynamic, 1>;
        using Mesh = cuthho_mesh<T, ET>;

    protected:
        bool sym_grad;

        stokes_interface_method_ref_pts(bool sym)
                : sym_grad(sym) {
        }

        virtual std::pair <Mat, Vect>
        make_contrib_cut(const Mesh &msh, const typename Mesh::cell_type &cl,
                         const testType &test_case, const hho_degree_info hdi) {
        }

    public:
        std::pair <Mat, Vect>
        make_contrib_uncut(const Mesh &msh, const typename Mesh::cell_type &cl,
                           const hho_degree_info hdi, const testType &test_case) {
            T kappa;
            if (location(msh, cl) == element_location::IN_NEGATIVE_SIDE)
                kappa = test_case.parms.kappa_1;
            else
                kappa = test_case.parms.kappa_2;

            Mat gr2;
            if (sym_grad)
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
            Mat lhs = Mat::Zero(loc_size, loc_size);
            Vect rhs = Vect::Zero(loc_size);

            lhs.block(0, 0, v_size, v_size) = lc;
            lhs.block(0, v_size, v_size, p_size) = -dr.second.transpose();
            lhs.block(v_size, 0, p_size, v_size) = -dr.second;

            rhs.head(f.rows()) = f;
            return std::make_pair(lhs, rhs);
        }


        std::pair <Mat, Vect>
        make_contrib(const Mesh &msh, const typename Mesh::cell_type &cl,
                     const testType &test_case, const hho_degree_info hdi) {
            if (location(msh, cl) != element_location::ON_INTERFACE)
                return make_contrib_uncut(msh, cl, hdi, test_case);
            else // on interface
                return make_contrib_cut(msh, cl, test_case, hdi);
        }
    };


    template<typename T, size_t ET, typename testType>
    class stokes_interface_method_ref_pts_cont {
        using Mat = Matrix<T, Dynamic, Dynamic>;
        using Vect = Matrix<T, Dynamic, 1>;
        using Mesh = cuthho_mesh<T, ET>;

    protected:
        bool sym_grad;

        stokes_interface_method_ref_pts_cont(bool sym)
                : sym_grad(sym) {
        }

        virtual std::pair <Mat, Vect>
        make_contrib_cut(const Mesh &msh, const typename Mesh::cell_type &cl,
                         const testType &test_case, const hho_degree_info hdi) {
        }

    public:
        std::pair <Mat, Vect>
        make_contrib_uncut(const Mesh &msh, const typename Mesh::cell_type &cl,
                           const hho_degree_info hdi, const testType &test_case) {
            T kappa;
            if (location(msh, cl) == element_location::IN_NEGATIVE_SIDE)
                kappa = test_case.parms.kappa_1;
            else
                kappa = test_case.parms.kappa_2;

            Mat gr2;
            if (sym_grad)
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
            Mat lhs = Mat::Zero(loc_size, loc_size);
            Vect rhs = Vect::Zero(loc_size);

            lhs.block(0, 0, v_size, v_size) = lc;
            lhs.block(0, v_size, v_size, p_size) = -dr.second.transpose();
            lhs.block(v_size, 0, p_size, v_size) = -dr.second;

            rhs.head(f.rows()) = f;
            return std::make_pair(lhs, rhs);
        }


        std::pair <Mat, Vect>
        make_contrib(const Mesh &msh, const typename Mesh::cell_type &cl,
                     const testType &test_case, const hho_degree_info hdi) {
            if (location(msh, cl) != element_location::ON_INTERFACE)
                return make_contrib_uncut(msh, cl, hdi, test_case);
            else // on interface
                return make_contrib_cut(msh, cl, test_case, hdi);
        }
    };



////////////////////////  SYMMETRIC GRADREC INTERFACE METHOD


    template<typename T, size_t ET, typename testType>
    class Sym_gradrec_stokes_interface_method :
            public stokes_interface_method<T, ET, testType> {
        using Mat = Matrix<T, Dynamic, Dynamic>;
        using Vect = Matrix<T, Dynamic, 1>;
        using Mesh = cuthho_mesh<T, ET>;

    public:
        T eta, gamma_0;

        Sym_gradrec_stokes_interface_method(T eta_, T gamma_, bool sym)
                : stokes_interface_method<T, ET, testType>(sym), eta(eta_), gamma_0(gamma_) {
        }

        std::pair <Mat, Vect>
        make_contrib_cut(const Mesh &msh, const typename Mesh::cell_type &cl,
                         const testType &test_case, const hho_degree_info &hdi) {
            auto parms = test_case.parms;
            auto level_set_function = test_case.level_set_;
            level_set_function.cell_assignment(cl);
///////////////   LHS
            auto celdeg = hdi.cell_degree();
            auto pdeg = hdi.face_degree();
            auto cbs = vector_cell_basis<Mesh, T>::size(celdeg);
            auto pbs = cell_basis<Mesh, T>::size(pdeg);

// GR
            Mat gr2_n, gr2_p;
            if (this->sym_grad) {
// ---- THIS CASE IS ROBUST FOR K_1 SIMILAR TO K_2 -----

// 0.5 is the weight coefficient that scale the interface term between inner and outer interface (Omega_1 and Omega_2)
/// Paper: Un Unfitted HHO method with cell agglomeration for elliptic interface pb.
/// -->  This is the variant 2.5 (pag.7)
                gr2_n = make_hho_gradrec_sym_matrix_interface
                        (msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE, 0.5).second;
                gr2_p = make_hho_gradrec_sym_matrix_interface
                        (msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE, 0.5).second;
            } else {
                gr2_n = make_hho_gradrec_matrix_interface
                        (msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE, 0.5).second;
                gr2_p = make_hho_gradrec_matrix_interface
                        (msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE, 0.5).second;
            }

// stab
            Mat stab = make_hho_vector_stabilization_interface(msh, cl, level_set_function, hdi, parms);
// Penalty conforming to variant 2.5, paper "Un Unfitted HHO method.."
            Mat penalty = make_hho_cut_interface_vector_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
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


            Mat lhs = Mat::Zero(lc.rows() + 2 * pbs, lc.rows() + 2 * pbs);
            lhs.block(0, 0, lc.rows(), lc.rows()) = lc;
            lhs.block(0, lc.rows(), lc.rows(), pbs) -= dr_n.second.transpose();
            lhs.block(0, lc.rows() + pbs, lc.rows(), pbs) -= dr_p.second.transpose();
            lhs.block(lc.rows(), 0, pbs, lc.rows()) -= dr_n.second;
            lhs.block(lc.rows() + pbs, 0, pbs, lc.rows()) -= dr_p.second;


// stokes stabilization terms
            auto stokes_stab = make_stokes_interface_stabilization(msh, cl, hdi, level_set_function);
            lhs.block(0, 0, 2 * cbs, 2 * cbs) -= gamma_0 * stokes_stab.block(0, 0, 2 * cbs, 2 * cbs);
            lhs.block(0, lc.rows(), 2 * cbs, 2 * pbs) -= gamma_0 * stokes_stab.block(0, 2 * cbs, 2 * cbs, 2 * pbs);
            lhs.block(lc.rows(), 0, 2 * pbs, 2 * cbs) -= gamma_0 * stokes_stab.block(2 * cbs, 0, 2 * pbs, 2 * cbs);
            lhs.block(lc.rows(), lc.rows(), 2 * pbs, 2 * pbs)
                    -= gamma_0 * stokes_stab.block(2 * cbs, 2 * cbs, 2 * pbs, 2 * pbs);



////////////////    RHS

            Vect f = Vect::Zero(lc.rows());
// neg part
            f.block(0, 0, cbs, 1) += make_vector_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                                     element_location::IN_NEGATIVE_SIDE);
//        f.head(cbs) += 0.5*make_vector_flux_jump(msh,cl,celdeg, element_location::IN_NEGATIVE_SIDE, test_case.neumann_jump);
            f.head(cbs) += 0.5 * make_vector_flux_jump(msh, cl, celdeg, element_location::IN_NEGATIVE_SIDE,
                                                       test_case.neumann_jump); // 0.5*

// pos part
            f.block(cbs, 0, cbs, 1) += make_vector_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                                       element_location::IN_POSITIVE_SIDE);
//        f.block(cbs, 0, cbs, 1)
//            += 0.5 * make_vector_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,  test_case.neumann_jump);
            f.block(cbs, 0, cbs, 1) += 0.5 * make_vector_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                                                   test_case.neumann_jump); // 0.5 *


            Vect rhs = Vect::Zero(lc.rows() + 2 * pbs);
            rhs.head(lc.rows()) = f;

// stokes stabilization rhs
            auto stab_rhs = make_stokes_interface_stabilization_RHS
                    (msh, cl, hdi, level_set_function, test_case.neumann_jump);

            rhs.head(2 * cbs) -= gamma_0 * stab_rhs.head(2 * cbs);
            rhs.tail(2 * pbs) -= gamma_0 * stab_rhs.tail(2 * pbs);

            return std::make_pair(lhs, rhs);
        }
    };

    template<typename T, size_t ET, typename testType>
    auto make_sym_gradrec_stokes_interface_method(const cuthho_mesh <T, ET> &msh, const T eta_,
                                                  const T gamma_, testType &test_case, bool sym) {
        return Sym_gradrec_stokes_interface_method<T, ET, testType>(eta_, gamma_, sym);
    }


    template<typename T, size_t ET, typename testType>
    class Sym_gradrec_stokes_interface_method_alfai :
            public stokes_interface_method<T, ET, testType> {
        using Mat = Matrix<T, Dynamic, Dynamic>;
        using Vect = Matrix<T, Dynamic, 1>;
        using Mesh = cuthho_mesh<T, ET>;


    public:
        T eta, gamma_0;
        T alfa1, alfa2;


        Sym_gradrec_stokes_interface_method_alfai(Sym_gradrec_stokes_interface_method_alfai &other)
                : stokes_interface_method<T, ET, testType>(other) {
            eta = other.eta;
            gamma_0 = other.gamma_0;
            alfa1 = other.alfa1;
            alfa2 = other.alfa2;
        }

        Sym_gradrec_stokes_interface_method_alfai(const Sym_gradrec_stokes_interface_method_alfai &other)
                : stokes_interface_method<T, ET, testType>(other) {
            eta = other.eta;
            gamma_0 = other.gamma_0;
            alfa1 = other.alfa1;
            alfa2 = other.alfa2;
        }


        Sym_gradrec_stokes_interface_method_alfai(T eta_, T gamma_, bool sym, T alfa1_, T alfa2_)
                : stokes_interface_method<T, ET, testType>(sym), eta(eta_), gamma_0(gamma_), alfa1(alfa1_), alfa2(alfa2_) {
        }

        std::pair <Mat, Vect>
        make_contrib_cut(const Mesh &msh, const typename Mesh::cell_type &cl,
                         const testType &test_case, const hho_degree_info &hdi) {
            auto parms = test_case.parms;
            auto level_set_function = test_case.level_set_;
            level_set_function.cell_assignment(cl);
///////////////   LHS
            auto celdeg = hdi.cell_degree();
            auto pdeg = hdi.face_degree();
            auto cbs = vector_cell_basis<Mesh, T>::size(celdeg);
            auto pbs = cell_basis<Mesh, T>::size(pdeg);

// GR
            Mat gr2_n, gr2_p;
            if (this->sym_grad) {
// ---- THIS CASE IS ROBUST FOR K_1 SIMILAR TO K_2 -----

// 0.5 is the weight coefficient that scale the interface term between inner and outer interface (Omega_1 and Omega_2)
/// Paper: Un Unfitted HHO method with cell agglomeration for elliptic interface pb.
/// -->  This is the variant 2.5 (pag.7)
                gr2_n = make_hho_gradrec_sym_matrix_interface
                        (msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE, alfa1).second;
                gr2_p = make_hho_gradrec_sym_matrix_interface
                        (msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE, alfa2).second;
            } else {
                gr2_n = make_hho_gradrec_matrix_interface
                        (msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE, alfa1).second;
                gr2_p = make_hho_gradrec_matrix_interface
                        (msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE, alfa2).second;
            }

// stab
            Mat stab = make_hho_vector_stabilization_interface(msh, cl, level_set_function, hdi, parms);
// Penalty conforming to variant 2.5, paper "Un Unfitted HHO method.."
            Mat penalty = make_hho_cut_interface_vector_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);

// It should be k_i s.t. k_i <= k_j
            auto k_min = std::min(parms.kappa_1, parms.kappa_2);
            auto k_arm = 2.0 / (1.0 / parms.kappa_1 + 1.0 / parms.kappa_2);
            auto k_penalty = k_arm;
            stab.block(0, 0, cbs, cbs) += k_penalty * penalty;
            stab.block(0, cbs, cbs, cbs) -= k_penalty * penalty;
            stab.block(cbs, 0, cbs, cbs) -= k_penalty * penalty;
            stab.block(cbs, cbs, cbs, cbs) += k_penalty * penalty;

//        stab.block(0, 0, cbs, cbs) += parms.kappa_2 * penalty;
//        stab.block(0, cbs, cbs, cbs) -= parms.kappa_2 * penalty;
//        stab.block(cbs, 0, cbs, cbs) -= parms.kappa_2 * penalty;
//        stab.block(cbs, cbs, cbs, cbs) += parms.kappa_2 * penalty;


// This is term \tilde{a_T} (eq.15), paper "Un Unfitted HHO method.."
            Mat lc = stab + parms.kappa_1 * gr2_n + parms.kappa_2 * gr2_p;

// DR : Penalty divided:
// (1.0 - coeff) * interface_term in NEGATIVE SIDE + coeff contribute into positive side
// (coeff- 1.0 ) * interface_term in POSITIVE SIDE - coeff contribute into negative side
            auto dr_n = make_hho_divergence_reconstruction_interface
                    (msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE,
                     alfa1);//0.5); // Modified by Stefano
            auto dr_p = make_hho_divergence_reconstruction_interface
                    (msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE,
                     alfa2);// 0.5);  // Modified by Stefano


            Mat lhs = Mat::Zero(lc.rows() + 2 * pbs, lc.rows() + 2 * pbs);
            lhs.block(0, 0, lc.rows(), lc.rows()) = lc;
            lhs.block(0, lc.rows(), lc.rows(), pbs) -= dr_n.second.transpose();
            lhs.block(0, lc.rows() + pbs, lc.rows(), pbs) -= dr_p.second.transpose();
            lhs.block(lc.rows(), 0, pbs, lc.rows()) -= dr_n.second;
            lhs.block(lc.rows() + pbs, 0, pbs, lc.rows()) -= dr_p.second;


// stokes stabilization terms
            auto stokes_stab = make_stokes_interface_stabilization(msh, cl, hdi, level_set_function);
            lhs.block(0, 0, 2 * cbs, 2 * cbs) -= gamma_0 * stokes_stab.block(0, 0, 2 * cbs, 2 * cbs);
            lhs.block(0, lc.rows(), 2 * cbs, 2 * pbs) -= gamma_0 * stokes_stab.block(0, 2 * cbs, 2 * cbs, 2 * pbs);
            lhs.block(lc.rows(), 0, 2 * pbs, 2 * cbs) -= gamma_0 * stokes_stab.block(2 * cbs, 0, 2 * pbs, 2 * cbs);
            lhs.block(lc.rows(), lc.rows(), 2 * pbs, 2 * pbs)
                    -= gamma_0 * stokes_stab.block(2 * cbs, 2 * cbs, 2 * pbs, 2 * pbs);



////////////////    RHS

            Vect f = Vect::Zero(lc.rows());
// neg part
            f.block(0, 0, cbs, 1) += make_vector_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                                     element_location::IN_NEGATIVE_SIDE);
//        f.head(cbs) += 0.5*make_vector_flux_jump(msh,cl,celdeg, element_location::IN_NEGATIVE_SIDE, test_case.neumann_jump);
            f.head(cbs) += alfa2 * make_vector_flux_jump(msh, cl, celdeg, element_location::IN_NEGATIVE_SIDE,
                                                         test_case.neumann_jump); // 0.5*

// pos part
            f.block(cbs, 0, cbs, 1) += make_vector_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                                       element_location::IN_POSITIVE_SIDE);
//        f.block(cbs, 0, cbs, 1)
//            += 0.5 * make_vector_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,  test_case.neumann_jump);
            f.block(cbs, 0, cbs, 1) += alfa1 * make_vector_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                                                     test_case.neumann_jump); // 0.5 *


            Vect rhs = Vect::Zero(lc.rows() + 2 * pbs);
            rhs.head(lc.rows()) = f;

// stokes stabilization rhs
            auto stab_rhs = make_stokes_interface_stabilization_RHS
                    (msh, cl, hdi, level_set_function, test_case.neumann_jump);

            rhs.head(2 * cbs) -= gamma_0 * stab_rhs.head(2 * cbs);
            rhs.tail(2 * pbs) -= gamma_0 * stab_rhs.tail(2 * pbs);

            return std::make_pair(lhs, rhs);
        }
    };


    template<typename T, size_t ET, typename testType>
    auto make_sym_gradrec_stokes_interface_method_alfai(const cuthho_mesh <T, ET> &msh, const T eta_,
                                                        const T gamma_, testType &test_case, bool sym, T alfa1, T alfa2) {
        return Sym_gradrec_stokes_interface_method_alfai<T, ET, testType>(eta_, gamma_, sym, alfa1, alfa2);
    }


    template<typename T, size_t ET, typename testType>
    class Sym_gradrec_stokes_interface_method_ref_pts :
            public stokes_interface_method_ref_pts<T, ET, testType> {
        using Mat = Matrix<T, Dynamic, Dynamic>;
        using Vect = Matrix<T, Dynamic, 1>;
        using Mesh = cuthho_mesh<T, ET>;

    public:
        T eta, gamma_0;

        Sym_gradrec_stokes_interface_method_ref_pts(T eta_, T gamma_, bool sym)
                : stokes_interface_method_ref_pts<T, ET, testType>(sym), eta(eta_), gamma_0(gamma_) {
        }

        std::pair <Mat, Vect>
        make_contrib_cut(const Mesh &msh, const typename Mesh::cell_type &cl,
                         const testType &test_case, const hho_degree_info hdi) {
            auto parms = test_case.parms;
            auto level_set_function = test_case.level_set_;
            level_set_function.cell_assignment(cl);
///////////////   LHS
            auto celdeg = hdi.cell_degree();
            auto pdeg = hdi.face_degree();
            auto cbs = vector_cell_basis<Mesh, T>::size(celdeg);
            auto pbs = cell_basis<Mesh, T>::size(pdeg);

// GR
            Mat gr2_n, gr2_p;
            if (this->sym_grad) {
// ---- THIS CASE IS ROBUST FOR K_1 SIMILAR TO K_2 -----

// 0.5 is the weight coefficient that scale the interface term between inner and outer interface (Omega_1 and Omega_2)
/// Paper: Un Unfitted HHO method with cell agglomeration for elliptic interface pb.
/// -->  This is the variant 2.5 (pag.7)
                gr2_n = make_hho_gradrec_sym_matrix_interface_ref_pts
                        (msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE, 0.5).second;
                gr2_p = make_hho_gradrec_sym_matrix_interface_ref_pts
                        (msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE, 0.5).second;
            } else {
// ANCORA DA MODIFICAREEEEEE
                std::cout << "STILL TO BE MODIFIED WITH parametric curve" << std::endl;
                gr2_n = make_hho_gradrec_matrix_interface
                        (msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE, 0.5).second;
                gr2_p = make_hho_gradrec_matrix_interface
                        (msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE, 0.5).second;
            }

// stab
            Mat stab = make_hho_vector_stabilization_interface(msh, cl, level_set_function, hdi, parms);
// Penalty conforming to variant 2.5, paper "Un Unfitted HHO method.."
            Mat penalty = make_hho_cut_interface_vector_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
            stab.block(0, 0, cbs, cbs) += parms.kappa_2 * penalty;
            stab.block(0, cbs, cbs, cbs) -= parms.kappa_2 * penalty;
            stab.block(cbs, 0, cbs, cbs) -= parms.kappa_2 * penalty;
            stab.block(cbs, cbs, cbs, cbs) += parms.kappa_2 * penalty;
// This is term \tilde{a_T} (eq.15), paper "Un Unfitted HHO method.."
            Mat lc = stab + parms.kappa_1 * gr2_n + parms.kappa_2 * gr2_p;

// DR : Penalty divided:
// (1.0 - coeff) * interface_term in NEGATIVE SIDE + coeff contribute into positive side
// (coeff- 1.0 ) * interface_term in POSITIVE SIDE - coeff contribute into negative side
            auto dr_n = make_hho_divergence_reconstruction_interface_ref_pts
                    (msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE, 0.5);
            auto dr_p = make_hho_divergence_reconstruction_interface_ref_pts
                    (msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE, 0.5);


            Mat lhs = Mat::Zero(lc.rows() + 2 * pbs, lc.rows() + 2 * pbs);
            lhs.block(0, 0, lc.rows(), lc.rows()) = lc;
            lhs.block(0, lc.rows(), lc.rows(), pbs) -= dr_n.second.transpose();
            lhs.block(0, lc.rows() + pbs, lc.rows(), pbs) -= dr_p.second.transpose();
            lhs.block(lc.rows(), 0, pbs, lc.rows()) -= dr_n.second;
            lhs.block(lc.rows() + pbs, 0, pbs, lc.rows()) -= dr_p.second;


// stokes stabilization terms
            auto stokes_stab = make_stokes_interface_stabilization_ref_pts(msh, cl, hdi, level_set_function);
            lhs.block(0, 0, 2 * cbs, 2 * cbs) -= gamma_0 * stokes_stab.block(0, 0, 2 * cbs, 2 * cbs);
            lhs.block(0, lc.rows(), 2 * cbs, 2 * pbs) -= gamma_0 * stokes_stab.block(0, 2 * cbs, 2 * cbs, 2 * pbs);
            lhs.block(lc.rows(), 0, 2 * pbs, 2 * cbs) -= gamma_0 * stokes_stab.block(2 * cbs, 0, 2 * pbs, 2 * cbs);
            lhs.block(lc.rows(), lc.rows(), 2 * pbs, 2 * pbs)
                    -= gamma_0 * stokes_stab.block(2 * cbs, 2 * cbs, 2 * pbs, 2 * pbs);



////////////////    RHS

            Vect f = Vect::Zero(lc.rows());
// neg part
            f.block(0, 0, cbs, 1) += make_vector_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                                     element_location::IN_NEGATIVE_SIDE);
//        f.head(cbs) += 0.5*make_vector_flux_jump(msh,cl,celdeg, element_location::IN_NEGATIVE_SIDE, test_case.neumann_jump);
            f.head(cbs) += 0.5 * make_vector_flux_jump_reference_pts(msh, cl, celdeg, element_location::IN_NEGATIVE_SIDE,
                                                                     test_case.neumann_jump);

// pos part
            f.block(cbs, 0, cbs, 1) += make_vector_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                                       element_location::IN_POSITIVE_SIDE);
//        f.block(cbs, 0, cbs, 1)
//            += 0.5 * make_vector_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,  test_case.neumann_jump);
            f.block(cbs, 0, cbs, 1) += 0.5 * make_vector_flux_jump_reference_pts(msh, cl, celdeg,
                                                                                 element_location::IN_POSITIVE_SIDE,
                                                                                 test_case.neumann_jump);


            Vect rhs = Vect::Zero(lc.rows() + 2 * pbs);
            rhs.head(lc.rows()) = f;

// stokes stabilization rhs
            auto stab_rhs = make_stokes_interface_stabilization_RHS_ref_pts
                    (msh, cl, hdi, level_set_function, test_case.neumann_jump);

            rhs.head(2 * cbs) -= gamma_0 * stab_rhs.head(2 * cbs);
            rhs.tail(2 * pbs) -= gamma_0 * stab_rhs.tail(2 * pbs);

            return std::make_pair(lhs, rhs);
        }
    };

    template<typename T, size_t ET, typename testType>
    auto make_sym_gradrec_stokes_interface_method_ref_pts(const cuthho_mesh <T, ET> &msh, const T eta_,
                                                          const T gamma_, testType &test_case, bool sym) {
        return Sym_gradrec_stokes_interface_method_ref_pts<T, ET, testType>(eta_, gamma_, sym);
    }


    template<typename T, size_t ET, typename testType>
    class Sym_gradrec_stokes_interface_method_ref_pts_cont :
            public stokes_interface_method_ref_pts_cont<T, ET, testType> {
        using Mat = Matrix<T, Dynamic, Dynamic>;
        using Vect = Matrix<T, Dynamic, 1>;
        using Mesh = cuthho_mesh<T, ET>;

    public:
        T eta, gamma_0;

        Sym_gradrec_stokes_interface_method_ref_pts_cont(T eta_, T gamma_, bool sym)
                : stokes_interface_method_ref_pts_cont<T, ET, testType>(sym), eta(eta_), gamma_0(gamma_) {
        }

        std::pair <Mat, Vect>
        make_contrib_cut(const Mesh &msh, const typename Mesh::cell_type &cl,
                         const testType &test_case, const hho_degree_info hdi) {
            auto parms = test_case.parms;
            auto parametric_interface = test_case.parametric_interface;
            auto level_set_function = test_case.level_set_;
            level_set_function.cell_assignment(cl);
///////////////   LHS
            auto celdeg = hdi.cell_degree();
            auto pdeg = hdi.face_degree();
            auto cbs = vector_cell_basis<Mesh, T>::size(celdeg);
            auto pbs = cell_basis<Mesh, T>::size(pdeg);

// GR
            Mat gr2_n, gr2_p;
            if (this->sym_grad) {
// ---- THIS CASE IS ROBUST FOR K_1 SIMILAR TO K_2 -----

// 0.5 is the weight coefficient that scale the interface term between inner and outer interface (Omega_1 and Omega_2)
/// Paper: Un Unfitted HHO method with cell agglomeration for elliptic interface pb.
/// -->  This is the variant 2.5 (pag.7)
                gr2_n = make_hho_gradrec_sym_matrix_interface_ref_pts_cont
                        (msh, cl, parametric_interface, hdi, element_location::IN_NEGATIVE_SIDE, 0.5).second;
                gr2_p = make_hho_gradrec_sym_matrix_interface_ref_pts_cont
                        (msh, cl, parametric_interface, hdi, element_location::IN_POSITIVE_SIDE, 0.5).second;
            } else {
// ANCORA DA MODIFICAREEEEEE
                std::cout << "STILL TO BE MODIFIED WITH parametric curve" << std::endl;
                gr2_n = make_hho_gradrec_matrix_interface
                        (msh, cl, level_set_function, hdi, element_location::IN_NEGATIVE_SIDE, 0.5).second;
                gr2_p = make_hho_gradrec_matrix_interface
                        (msh, cl, level_set_function, hdi, element_location::IN_POSITIVE_SIDE, 0.5).second;
            }

// stab
            Mat stab = make_hho_vector_stabilization_interface(msh, cl, level_set_function, hdi, parms);
// Penalty conforming to variant 2.5, paper "Un Unfitted HHO method.."
            Mat penalty = make_hho_cut_interface_vector_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
            stab.block(0, 0, cbs, cbs) += parms.kappa_2 * penalty;
            stab.block(0, cbs, cbs, cbs) -= parms.kappa_2 * penalty;
            stab.block(cbs, 0, cbs, cbs) -= parms.kappa_2 * penalty;
            stab.block(cbs, cbs, cbs, cbs) += parms.kappa_2 * penalty;
// This is term \tilde{a_T} (eq.15), paper "Un Unfitted HHO method.."
            Mat lc = stab + parms.kappa_1 * gr2_n + parms.kappa_2 * gr2_p;

// DR : Penalty divided:
// (1.0 - coeff) * interface_term in NEGATIVE SIDE + coeff contribute into positive side
// (coeff- 1.0 ) * interface_term in POSITIVE SIDE - coeff contribute into negative side
            auto dr_n = make_hho_divergence_reconstruction_interface_ref_pts_cont
                    (msh, cl, parametric_interface, hdi, element_location::IN_NEGATIVE_SIDE, 0.5);
            auto dr_p = make_hho_divergence_reconstruction_interface_ref_pts_cont
                    (msh, cl, parametric_interface, hdi, element_location::IN_POSITIVE_SIDE, 0.5);


            Mat lhs = Mat::Zero(lc.rows() + 2 * pbs, lc.rows() + 2 * pbs);
            lhs.block(0, 0, lc.rows(), lc.rows()) = lc;
            lhs.block(0, lc.rows(), lc.rows(), pbs) -= dr_n.second.transpose();
            lhs.block(0, lc.rows() + pbs, lc.rows(), pbs) -= dr_p.second.transpose();
            lhs.block(lc.rows(), 0, pbs, lc.rows()) -= dr_n.second;
            lhs.block(lc.rows() + pbs, 0, pbs, lc.rows()) -= dr_p.second;


// stokes stabilization terms
            auto stokes_stab = make_stokes_interface_stabilization_ref_pts_cont(msh, cl, hdi, parametric_interface);
            lhs.block(0, 0, 2 * cbs, 2 * cbs) -= gamma_0 * stokes_stab.block(0, 0, 2 * cbs, 2 * cbs);
            lhs.block(0, lc.rows(), 2 * cbs, 2 * pbs) -= gamma_0 * stokes_stab.block(0, 2 * cbs, 2 * cbs, 2 * pbs);
            lhs.block(lc.rows(), 0, 2 * pbs, 2 * cbs) -= gamma_0 * stokes_stab.block(2 * cbs, 0, 2 * pbs, 2 * cbs);
            lhs.block(lc.rows(), lc.rows(), 2 * pbs, 2 * pbs)
                    -= gamma_0 * stokes_stab.block(2 * cbs, 2 * cbs, 2 * pbs, 2 * pbs);



////////////////    RHS

            Vect f = Vect::Zero(lc.rows());
// neg part
            f.block(0, 0, cbs, 1) += make_vector_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                                     element_location::IN_NEGATIVE_SIDE);
//        f.head(cbs) += 0.5*make_vector_flux_jump(msh,cl,celdeg, element_location::IN_NEGATIVE_SIDE, test_case.neumann_jump);
            f.head(cbs) += 0.5 *
                           make_vector_flux_jump_reference_pts_cont(msh, cl, celdeg, element_location::IN_NEGATIVE_SIDE,
                                                                    test_case.neumann_jump, parametric_interface);

// pos part
            f.block(cbs, 0, cbs, 1) += make_vector_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                                       element_location::IN_POSITIVE_SIDE);
//        f.block(cbs, 0, cbs, 1)
//            += 0.5 * make_vector_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,  test_case.neumann_jump);
            f.block(cbs, 0, cbs, 1) += 0.5 * make_vector_flux_jump_reference_pts_cont(msh, cl, celdeg,
                                                                                      element_location::IN_POSITIVE_SIDE,
                                                                                      test_case.neumann_jump,
                                                                                      parametric_interface);


            Vect rhs = Vect::Zero(lc.rows() + 2 * pbs);
            rhs.head(lc.rows()) = f;

// stokes stabilization rhs
            auto stab_rhs = make_stokes_interface_stabilization_RHS_ref_pts_cont
                    (msh, cl, hdi, parametric_interface, test_case.neumann_jump);

            rhs.head(2 * cbs) -= gamma_0 * stab_rhs.head(2 * cbs);
            rhs.tail(2 * pbs) -= gamma_0 * stab_rhs.tail(2 * pbs);

            return std::make_pair(lhs, rhs);
        }
    };

    template<typename T, size_t ET, typename testType>
    auto make_sym_gradrec_stokes_interface_method_ref_pts_cont(const cuthho_mesh <T, ET> &msh, const T eta_, const T gamma_,
                                                               testType &test_case, bool sym) {
        return Sym_gradrec_stokes_interface_method_ref_pts_cont<T, ET, testType>(eta_, gamma_, sym);
    }



}


namespace pre_processing {


    template<typename FiniteSpace, typename Mesh, typename T>
    void
    check_inlet(const Mesh &msh, FiniteSpace &fe_data, bool bdry_bottom, bool bdry_right, bool bdry_up, bool bdry_left,
                T eps) {
        std::cout << "Checking inlet boundary condition for transport problem." << std::endl;
        std::vector < std::vector < std::pair < size_t, bool>>> connectivity_matrix = fe_data.connectivity_matrix;


        for (const auto &cl: msh.cells) {
            size_t cell_offset = offset(msh, cl);
            auto pts = equidistriduted_nodes_ordered_bis<T, Mesh>(msh, cl, fe_data.order);
            for (size_t i = 0; i < fe_data.local_ndof; i++) {
                auto pt = pts[i];
                size_t asm_map = connectivity_matrix[cell_offset][i].first;
                if (connectivity_matrix[cell_offset][i].second) {
                    if (bdry_bottom && (std::abs(pt.y()) < eps))
                        fe_data.Dirichlet_boundary_inlet[asm_map] = TRUE;
                    else if (bdry_right && (std::abs(pt.x() - 1.0) < eps))
                        fe_data.Dirichlet_boundary_inlet[asm_map] = TRUE;
                    else if (bdry_up && (std::abs(pt.y() - 1.0) < eps))
                        fe_data.Dirichlet_boundary_inlet[asm_map] = TRUE;
                    else if (bdry_left && (std::abs(pt.x()) < eps))
                        fe_data.Dirichlet_boundary_inlet[asm_map] = TRUE;
                    else
                        fe_data.Dirichlet_boundary_inlet[asm_map] = FALSE;

                } else
                    fe_data.Dirichlet_boundary_inlet[asm_map] = FALSE;

//            std::cout<<"fe_data.Dirichlet_boundary_inlet[asm_map] = "<<fe_data.Dirichlet_boundary_inlet[asm_map]<<std::endl;

            }
        }

    }

    template<typename FiniteSpace, typename Mesh, typename Vel_Field, typename T>
    void check_inlet(const Mesh &msh, FiniteSpace &fe_data, const Vel_Field &u, T eps) {
        std::cout << "Checking inlet boundary condition for numerical flows." << std::endl;
        std::vector < std::vector < std::pair < size_t, bool>>> connectivity_matrix = fe_data.connectivity_matrix;


        for (const auto &cl: msh.cells) {
            size_t cell_offset = offset(msh, cl);
            auto pts = equidistriduted_nodes_ordered_bis<T, Mesh>(msh, cl, fe_data.order);
            for (size_t i = 0; i < fe_data.local_ndof; i++) {
                auto pt = pts[i];
                size_t asm_map = connectivity_matrix[cell_offset][i].first;
                if (connectivity_matrix[cell_offset][i].second) {
                    if ((u(pt, msh, cl).second > eps) && (pt.y() == 0.0))
                        fe_data.Dirichlet_boundary_inlet[asm_map] = TRUE;
                    if ((u(pt, msh, cl).first < -eps) && (pt.x() == 1.0))
                        fe_data.Dirichlet_boundary_inlet[asm_map] = TRUE;
                    if ((u(pt, msh, cl).second < -eps) && (pt.y() == 1.0))
                        fe_data.Dirichlet_boundary_inlet[asm_map] = TRUE;
                    if ((u(pt, msh, cl).first > eps) && (pt.x() == 0.0))
                        fe_data.Dirichlet_boundary_inlet[asm_map] = TRUE;
                    else
                        fe_data.Dirichlet_boundary_inlet[asm_map] = FALSE;

                } else
                    fe_data.Dirichlet_boundary_inlet[asm_map] = FALSE;


            }
        }

    }

    template<typename T>
    T time_step_STK(T eta, T gamma, const mesh_init_params <T> &mip, T c2) {

        auto h_max = std::max(mip.hx(), mip.hy());
        return c2 * eta / gamma * h_max;


    }

    template<typename T, typename VeloField>
    T time_step_CFL_new(const VeloField &u, const mesh_init_params <T> &mip, T eps) {

        auto h_max = std::max(mip.hx(), mip.hy());
        auto u_max = u.sol_FEM.first.template lpNorm<Infinity>() + u.sol_FEM.second.
                template lpNorm<Infinity>();
        if (std::abs(u_max) < 1e-15)
            return 1e-8;
        else
            return eps * h_max / (std::abs(u_max));
    }



}


