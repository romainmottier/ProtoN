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

#include "level_set_transport_problem.hpp"
using namespace level_set_transport;

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


namespace computationQuantities {
    template<typename T, typename FiniteSpace, typename Mesh>
    void
    reference_quantities_computation(T &perim_ref, T &area_ref, T &circularity_ref, T &radius, T &x_centre, T &y_centre,
                                     const FiniteSpace &fe_data, const Mesh &msh, size_t degree_curve, T perimeter_initial,
                                     T initial_area, size_t int_refsteps, size_t degree_det_jac_curve) {
// calculus of circle REF in Q^k mesh N x M --> CIRCULARITY REF
        std::cout
                << "------ NOTICE: The REF quantities are the numerical calculation of some important quantities for a circle interface. This is useful in the fixed-pont problem to check the convergence of the flower into the equivalent circle."
                << std::endl;
        auto level_anal_ref = circle_level_set<T>(radius, x_centre, y_centre);
        std::cout << "REF radius = " << radius << std::endl;
        typedef circle_level_set <T> Fonction_REF;
        auto level_set_ref = Level_set_berstein_high_order_interpolation_grad_cont_fast<Mesh, Fonction_REF, FiniteSpace, T>(
                fe_data, level_anal_ref, msh);

        Mesh msh_ref = msh;
        offset_definition(msh_ref);
        detect_node_position3(msh_ref, level_set_ref); // In cuthho_geom
        detect_cut_faces3(msh_ref, level_set_ref); // In cuthho_geom
        detect_cut_cells3(msh_ref, level_set_ref); // In cuthho_geom
        refine_interface_pro3_curve_para(msh_ref, level_set_ref, int_refsteps, degree_curve);
        set_integration_mesh(msh_ref, degree_curve);
        detect_cell_agglo_set(msh_ref, level_set_ref); // Non serve modificarla
        make_neighbors_info_cartesian(msh_ref); // Non serve modificarla
        make_agglomeration_no_double_points(msh_ref, level_set_ref, degree_det_jac_curve);
        set_integration_mesh(msh_ref, degree_curve);

        typedef Level_set_berstein_high_order_interpolation_grad_cont_fast<Mesh, Fonction_REF, FiniteSpace, T> Level_Set_REF;
        auto ls_cell_ref = LS_cell_high_order_grad_cont_fast<T, Mesh, Level_Set_REF, Fonction_REF, FiniteSpace>(
                level_set_ref, msh_ref);


        for (auto &cl: msh_ref.cells) {
            ls_cell_ref.cell_assignment(cl);


            if (location(msh_ref, cl) == element_location::IN_NEGATIVE_SIDE ||
                location(msh_ref, cl) == element_location::ON_INTERFACE) {
                T partial_area = measure(msh_ref, cl, element_location::IN_NEGATIVE_SIDE);
                area_ref += partial_area;

            }
            if (cl.user_data.location == element_location::ON_INTERFACE) {
                perim_ref += measure_interface(msh_ref, cl, element_location::ON_INTERFACE);
//                for(auto interface_point = cl.user_data.interface.begin() ; interface_point < cl.user_data.interface.end() -1 ; interface_point++ )
//                {
//                    perim_ref += ( *(interface_point+1) - *interface_point ).to_vector().norm();
//
//                }
            }
        }

//        T d_a_REF = sqrt(4.0*area_ref/M_PI) ;
//        std::cout<<"AREA REF = " << area_ref <<std::endl;
//        std::cout<<"PERIMETER REF = " << perim_ref <<std::endl;
        std::cout << "Error( perimetre_ref - perimeter_initial ) = " << perim_ref - perimeter_initial << std::endl;
        std::cout << "Error( area_ref - initial_area ) = " << area_ref - initial_area << std::endl;
        circularity_ref = 1.0; // M_PI*d_a_REF/perim_ref ;
        std::cout << "CIRCULARITY REF = " << circularity_ref << std::endl;
//        T area_anal = M_PI*radius*radius ;
//        std::cout<<"Error( area_ref - area_analytic ) = " << area_ref - area_anal <<std::endl;
//
//
//        T perimeter_anal = 2.0*M_PI*radius ;
//        std::cout<<"Error( perimetre_ref - perimeter_anal ) = " << perim_ref - perimeter_anal <<std::endl;


    }

    template<typename Mesh, typename Para_Interface, typename T>
    void
    para_curvature_error(Mesh &msh_i, Para_Interface &para_interface, T curvature_anal) {


        T L1_divergence_error_fin = 0.0;
        T linf_divergence_error_fin = 0.0;
        T l1_divergence_error_fin = 0.0;
        T tot_error = 10;
        size_t counter_interface_pts_fin = 0;
        size_t degree_curvature = para_interface.dd_degree;
        size_t degree_jacobian = para_interface.degree_det;

        for (auto &cl: msh_i.cells) {


            if (cl.user_data.location == element_location::ON_INTERFACE) {

                auto global_cells_i = para_interface.get_global_cells_interface(msh_i, cl);
                auto integration_msh = cl.user_data.integration_msh;
                auto degree_int = degree_curvature + degree_jacobian; // 3*degree_curve -1;
                auto qps = edge_quadrature<T>(degree_int);
                for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                    auto pts = points(integration_msh, integration_msh.cells[i_cell]);
                    size_t global_cl_i = global_cells_i[i_cell];

                    for (auto &qp: qps) {
                        auto t = 0.5 * qp.first.x() + 0.5;
                        auto curv_err = std::abs(
                                std::abs(para_interface.curvature_cont(t, global_cl_i)) - std::abs(curvature_anal));
                        T jacobian = para_interface.jacobian_cont(t, global_cl_i);
                        auto w = 0.5 * qp.second * jacobian;
                        L1_divergence_error_fin += curv_err * w;
                    }

                    for (int i = 0; i < tot_error; i++) {
                        T pos = 0.0 + i / tot_error;
                        T val0 = para_interface.curvature_cont(pos, global_cl_i);
                        T error_curvature = std::abs(std::abs(val0) - std::abs(curvature_anal));

                        linf_divergence_error_fin = std::max(linf_divergence_error_fin, error_curvature);
                        l1_divergence_error_fin += error_curvature;
                        counter_interface_pts_fin++;

                    }
                }


            } // if cut cell

        } // for loop cell


        std::cout << bold << green << "PARAMETRIC INTERFACE ANALYSIS:" << '\n' << reset;
        std::cout << "Number of interface points is " << counter_interface_pts_fin << std::endl;
        std::cout << "The l1 error of PARA-CURVATURE is " << l1_divergence_error_fin / counter_interface_pts_fin
                  << std::endl;

        std::cout << "The linf error of PARA-CURVATURE is " << linf_divergence_error_fin << std::endl;
        std::cout << "The L1 error of the PARA-CURVATURE is " << L1_divergence_error_fin << std::endl;


    }


    template<typename Mesh, typename Level_Set, typename Para_Interface, typename T, typename Velocity>
    void
    check_goal_quantities_final_para_no_plot(Mesh &msh_i, Level_Set &ls_cell, Para_Interface &para_interface,
                                             Velocity &u_projected, T &perimeter, T &d_a, T &area_fin, T &centre_mass_x,
                                             T &centre_mass_y, size_t degree_FEM, T &mass_fin, size_t degree_velocity,
                                             T &l1_divergence_error_fin, T &l2_divergence_error_fin,
                                             T &linf_divergence_error_fin, T &radius, T &L1_divergence_error_fin,
                                             size_t time_step, T &rise_vel0, T &rise_vel1, T &flux_interface,
                                             size_t &counter_interface_pts_fin, size_t degree_curve, size_t n_int,
                                             T &L1_normal_interface_para, T &linf_u_n_para, T &max_u_n_val_para,
                                             T &l1_normal_interface_para, size_t &counter_interface_pts_para,
                                             T &max_curvature, T &min_curvature) {

// PLOTTING OF NORMAL
        typedef typename Mesh::point_type point_type;

//postprocess_output<double> postoutput_vec;
//auto vec_normal_grad_cont_fin = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_continuos_grad_Stokes_final.dat");

        point_type pt_x_dx, pt_x_sx, pt_y_up, pt_y_doen;

//    std::vector<T>  val_u_n_fin ; //val_u_nx_fin , val_u_ny_fin ;
//    std::vector< point<T, 2> > interface_points_plot_fin ;
//    std::vector< std::pair<T,T> > vec_n ; // , velocity_interface , velocity_field , points_vel_field;

        T tot_error = 10;

        size_t degree_curvature = para_interface.dd_degree;
        size_t degree_jacobian = para_interface.degree_det;

//    std::vector< std::pair<T,T> >  velocity_field , points_vel_field ;

        for (auto &cl: msh_i.cells) {
            ls_cell.cell_assignment(cl);
            u_projected.cell_assignment(cl);

            if ((location(msh_i, cl) == element_location::IN_NEGATIVE_SIDE) ||
                (location(msh_i, cl) == element_location::ON_INTERFACE)) {

                T partial_area = measure(msh_i, cl, element_location::IN_NEGATIVE_SIDE);
                area_fin += partial_area;

                size_t max_deg = std::max(degree_velocity, degree_FEM);
                auto qps_fin = integrate(msh_i, cl, max_deg, element_location::IN_NEGATIVE_SIDE);


                for (auto &qp: qps_fin) {
                    mass_fin += qp.second * ls_cell(qp.first);
                    centre_mass_x += qp.second * qp.first.x();
                    centre_mass_y += qp.second * qp.first.y();

                    rise_vel0 += qp.second * u_projected(qp.first).first;
                    rise_vel1 += qp.second * u_projected(qp.first).second;
                }

            }
            if (cl.user_data.location == element_location::ON_INTERFACE) {


//                   auto qps = integrate_interface(msh_i, cl, degree_FEM, element_location::ON_INTERFACE);


                auto global_cells_i = para_interface.get_global_cells_interface(msh_i, cl);
                auto integration_msh = cl.user_data.integration_msh;
                auto degree_int = degree_curvature + degree_jacobian; // 3*degree_curve -1;
                auto qps = edge_quadrature<T>(degree_int);
//            auto qps += integrate_interface(msh_i, cl, degree_curve, element_location::ON_INTERFACE) ;

                for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                    auto pts = points(integration_msh, integration_msh.cells[i_cell]);
                    size_t global_cl_i = global_cells_i[i_cell];

                    for (auto &qp: qps) {
                        auto t = 0.5 * qp.first.x() + 0.5;
                        auto curv_err = std::abs(
                                std::abs(para_interface.curvature_cont(t, global_cl_i)) - std::abs(1.0 / radius));
                        T jacobian = para_interface.jacobian_cont(t, global_cl_i);
                        auto w = 0.5 * qp.second * jacobian;
                        L1_divergence_error_fin += curv_err * w;


                    }
                    for (int i = 0; i < tot_error; i++) {
                        T pos = 0.0 + i / tot_error;
                        T val0 = para_interface.curvature_cont(pos, global_cl_i);
                        T error_curvature = std::abs(std::abs(val0) - std::abs(1.0 / radius));
                        max_curvature = std::max(max_curvature, std::abs(val0));
                        min_curvature = std::min(min_curvature, std::abs(val0));

                        linf_divergence_error_fin = std::max(linf_divergence_error_fin, error_curvature);
                        l1_divergence_error_fin += error_curvature;
                        counter_interface_pts_fin++;


                    }
                }

//                   auto qps_un = integrate_interface(msh_i, cl, degree_FEM + degree_velocity , element_location::ON_INTERFACE);
//                   for(auto& qp:qps_un){
//                       T flux = u_projected(qp.first).first * ls_cell.normal(qp.first)(0) + u_projected(qp.first).second * ls_cell.normal(qp.first)(1) ;
//                       flux_interface += qp.second * flux ;
//
//
//                   }

                auto qps_un = edge_quadrature<T>(degree_jacobian + degree_curvature + degree_velocity);

                for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                    auto pts = points(integration_msh, integration_msh.cells[i_cell]);
                    size_t global_cl_i = global_cells_i[i_cell];

                    for (auto &qp: qps_un) {
                        auto t = 0.5 * qp.first.x() + 0.5;

                        T jacobian = para_interface.jacobian_cont(t, global_cl_i);
                        auto w = 0.5 * qp.second * jacobian;

                        auto p = para_interface(t, global_cl_i);
                        auto pt = typename Mesh::point_type(p(0), p(1));
                        auto u_pt = u_projected(pt);
                        auto interface_n_pt = para_interface.normal_cont(t, global_cl_i);
                        T flux = u_pt.first * interface_n_pt(0) + u_pt.second * interface_n_pt(1);
                        flux_interface += w * flux;

                        L1_normal_interface_para += w * std::abs(flux);
                        linf_u_n_para = std::max(linf_u_n_para, std::abs(flux));
                        if (std::abs(flux) == linf_u_n_para)
                            max_u_n_val_para = flux;

                        l1_normal_interface_para += std::abs(flux);
                        counter_interface_pts_para++;

                    }
                }


                perimeter += measure_interface(msh_i, cl, element_location::ON_INTERFACE);


            }
//            auto pts = equidistriduted_nodes_ordered_bis<T>(msh_i, cl, u_projected.degree_FEM);
//            for(auto& pt : pts)
//            {
//                points_vel_field.push_back( std::make_pair(pt.x() , pt.y() ) ) ;
//                auto u_pt = u_projected(pt);
//                velocity_field.push_back( std::make_pair(u_pt.first , u_pt.second)) ;
//
//            }



        }


//    std::string filename_stokes6 = "velocity_field_"+ std::to_string(time_step) +".3D";
//    std::ofstream interface_file6(filename_stokes6, std::ios::out | std::ios::trunc);
//
//    if(interface_file6)
//    {
//        // instructions
//        interface_file6 << "X   Y   val0   val1" << std::endl;
//        size_t i = 0;
//        for(auto point_vel = points_vel_field.begin() ; point_vel < points_vel_field.end() ; point_vel++ )
//        {
//            //std::cout<<val_u_nx[i]<<std::endl;
//            interface_file6 << (*point_vel).first << "   " <<  (*point_vel).second << "   "
//            << velocity_field[i].first << "   " << velocity_field[i].second<< std::endl;
//
//            i++;
//
//        }
//
//        interface_file6.close();
//    }
//    else
//        std::cerr << "File 'vec_u_n' has not been opened" << std::endl;


//    goal_quantities_time_fast(msh_i , interface_points_plot_fin , val_u_n_fin  , vec_n , time_step);



    }


    template<typename Mesh, typename Level_Set, typename Para_Interface, typename T, typename Velocity>
    void
    check_goal_quantities_final_para(Mesh &msh_i, Level_Set &ls_cell, Para_Interface &para_interface, Velocity &u_projected,
                                     T &perimeter, T &d_a, T &area_fin, T &centre_mass_x, T &centre_mass_y,
                                     size_t degree_FEM, T &mass_fin, size_t degree_velocity, T &l1_divergence_error_fin,
                                     T &l2_divergence_error_fin, T &linf_divergence_error_fin, T &radius,
                                     T &L1_divergence_error_fin, size_t time_step, T &rise_vel0, T &rise_vel1,
                                     T &flux_interface, size_t &counter_interface_pts_fin, size_t degree_curve,
                                     size_t n_int, T &L1_normal_interface_para, T &linf_u_n_para, T &max_u_n_val_para,
                                     T &l1_normal_interface_para, size_t &counter_interface_pts_para) {

// PLOTTING OF NORMAL
        typedef typename Mesh::point_type point_type;

//postprocess_output<double> postoutput_vec;
//auto vec_normal_grad_cont_fin = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_continuos_grad_Stokes_final.dat");



        std::vector <T> val_u_n_fin; //val_u_nx_fin , val_u_ny_fin ;
        std::vector <point<T, 2>> interface_points_plot_fin;
        std::vector <std::pair<T, T>> vec_n; // , velocity_interface , velocity_field , points_vel_field;

        T tot_error = 100;

        size_t degree_curvature = para_interface.dd_degree;
        size_t degree_jacobian = para_interface.degree_det;


        for (auto &cl: msh_i.cells) {
            ls_cell.cell_assignment(cl);
            u_projected.cell_assignment(cl);

            if ((location(msh_i, cl) == element_location::IN_NEGATIVE_SIDE) ||
                (location(msh_i, cl) == element_location::ON_INTERFACE)) {

                T partial_area = measure(msh_i, cl, element_location::IN_NEGATIVE_SIDE);
                area_fin += partial_area;

                size_t max_deg = std::max(degree_velocity, degree_FEM);
                auto qps_fin = integrate(msh_i, cl, max_deg, element_location::IN_NEGATIVE_SIDE);


                for (auto &qp: qps_fin) {
                    mass_fin += qp.second * ls_cell(qp.first);
                    centre_mass_x += qp.second * qp.first.x();
                    centre_mass_y += qp.second * qp.first.y();

                    rise_vel0 += qp.second * u_projected(qp.first).first;
                    rise_vel1 += qp.second * u_projected(qp.first).second;
                }

            }
            if (cl.user_data.location == element_location::ON_INTERFACE) {


//                   auto qps = integrate_interface(msh_i, cl, degree_FEM, element_location::ON_INTERFACE);


                auto global_cells_i = para_interface.get_global_cells_interface(msh_i, cl);
                auto integration_msh = cl.user_data.integration_msh;
                auto degree_int = degree_curvature + degree_jacobian; // 3*degree_curve -1;
                auto qps = edge_quadrature<T>(degree_int);
//            auto qps += integrate_interface(msh_i, cl, degree_curve, element_location::ON_INTERFACE) ;

                for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                    auto pts = points(integration_msh, integration_msh.cells[i_cell]);
                    size_t global_cl_i = global_cells_i[i_cell];

                    for (auto &qp: qps) {
                        auto t = 0.5 * qp.first.x() + 0.5;
                        auto curv_err = std::abs(para_interface.curvature_cont(t, global_cl_i) + 1.0 / radius);
                        T jacobian = para_interface.jacobian_cont(t, global_cl_i);
                        auto w = 0.5 * qp.second * jacobian;
                        L1_divergence_error_fin += curv_err * w;


                    }
                    for (int i = 0; i < tot_error; i++) {
                        T pos = 0.0 + i / tot_error;
                        T val0 = para_interface.curvature_cont(pos, global_cl_i);
                        T error_curvature = std::abs(val0 + 1.0 / radius);

                        linf_divergence_error_fin = std::max(linf_divergence_error_fin, error_curvature);
                        l1_divergence_error_fin += error_curvature;
                        counter_interface_pts_fin++;


                    }
                }

//                   auto qps_un = integrate_interface(msh_i, cl, degree_FEM + degree_velocity , element_location::ON_INTERFACE);
//                   for(auto& qp:qps_un){
//                       T flux = u_projected(qp.first).first * ls_cell.normal(qp.first)(0) + u_projected(qp.first).second * ls_cell.normal(qp.first)(1) ;
//                       flux_interface += qp.second * flux ;
//
//
//                   }

                auto qps_un = edge_quadrature<T>(degree_jacobian + degree_curvature + degree_velocity);

                for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                    auto pts = points(integration_msh, integration_msh.cells[i_cell]);
                    size_t global_cl_i = global_cells_i[i_cell];

                    for (auto &qp: qps_un) {
                        auto t = 0.5 * qp.first.x() + 0.5;

                        T jacobian = para_interface.jacobian_cont(t, global_cl_i);
                        auto w = 0.5 * qp.second * jacobian;

                        auto p = para_interface(t, global_cl_i);
                        auto pt = typename Mesh::point_type(p(0), p(1));
                        T flux = u_projected(pt).first * para_interface.normal_cont(t, global_cl_i)(0) +
                                 u_projected(pt).second * para_interface.normal_cont(t, global_cl_i)(1);
                        flux_interface += w * flux;

                        L1_normal_interface_para += w * std::abs(flux);
                        linf_u_n_para = std::max(linf_u_n_para, std::abs(flux));
                        if (std::abs(flux) == linf_u_n_para)
                            max_u_n_val_para = flux;

                        l1_normal_interface_para += std::abs(flux);
                        counter_interface_pts_para++;

                    }
                }


                perimeter += measure_interface(msh_i, cl, element_location::ON_INTERFACE);

                for (auto interface_point = cl.user_data.interface.begin();
                     interface_point < cl.user_data.interface.end(); interface_point++) {
//                       T segment = ( *(interface_point+1) - *interface_point ).to_vector().norm();
//                       perimeter += segment ;
//                       T val0 = ls_cell.divergence( *interface_point );
//                       T curvature_error = std::abs( std::abs(val0) - 1.0/radius ) ;
//                       l1_divergence_error_fin += curvature_error ;
//                       l2_divergence_error_fin += pow(curvature_error,2) ;
//                       linf_divergence_error_fin = std::max(linf_divergence_error_fin ,  curvature_error);



                    Eigen::Matrix<T, 2, 1> normal_cont_grad = ls_cell.normal(*interface_point);
                    std::pair <T, T> normal_vec_grad_cont = std::make_pair(normal_cont_grad(0), normal_cont_grad(1));

                    vec_n.push_back(normal_vec_grad_cont);

                    T u_n_0 = u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0);
                    T u_n_1 = u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1);

                    interface_points_plot_fin.push_back(*(interface_point));

                    val_u_n_fin.push_back(u_n_0 + u_n_1);

                }


            }


        }

// --------- CHECKING CURVATURE DISC GRAD CONT (THE ONE USED INTO THE CODE) ----------
        postprocess_output <T> postoutput_div2;
        std::string filename_curvature_k0 = "k0_curvature_" + std::to_string(time_step) + ".dat";
        auto test_curv_var_divergence0 = std::make_shared < gnuplot_output_object < double > > (filename_curvature_k0);
        std::string filename_curv_var = "cell_limit_curv_var_" + std::to_string(time_step) + ".dat";
        auto test_curv_var_cell = std::make_shared < gnuplot_output_object < double > > (filename_curv_var);
        std::string filename_curv_var_inner_cl = "inner_cell_limit_curv_var_" + std::to_string(time_step) + ".dat";
        auto test_inner_cell = std::make_shared < gnuplot_output_object < double > > (filename_curv_var_inner_cl);

        bool first_cut_cell_found = FALSE;
        T distance_pts = 0.0;
        point<T, 2> first_point;
        point<T, 2> cell_end_point;
        for (auto &cl: msh_i.cells) {

            if (cl.user_data.location == element_location::ON_INTERFACE) {

                if (!first_cut_cell_found) {
                    ls_cell.cell_assignment(cl);
                    bool agglo_cl =
                            cl.user_data.highlight && ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                    size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                    std::vector <size_t> index_inner_cls;
                    if (agglo_cl) {
                        for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                            index_inner_cls.push_back(i_cl * (cl.user_data.interface.size() - 1) / amount_sub_cls);
                    }


                    size_t pos_index = 0;
                    size_t pos_index_bis = 0;

                    for (auto interface_point = cl.user_data.interface.begin();
                         interface_point < cl.user_data.interface.end(); interface_point++) {
                        T val0 = ls_cell.divergence(*interface_point);

                        point<T, 2> curv_var = point_type(distance_pts, 0.0);
                        if (interface_point == cl.user_data.interface.begin() ||
                            interface_point == (cl.user_data.interface.end() - 1))
                            test_curv_var_cell->add_data(curv_var, val0);


                        if (agglo_cl && pos_index == index_inner_cls[pos_index_bis]) {
                            auto offset_cells = pt_in_subcell_skeleton(ls_cell.level_set.msh, *interface_point, cl);
                            assert(offset_cells.size() == 2);
                            auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                            auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                            T val_skeleton0 = ls_cell.divergence(*interface_point, subcl0);
                            T val_skeleton1 = ls_cell.divergence(*interface_point, subcl1);

                            test_inner_cell->add_data(curv_var, val_skeleton0);
                            test_inner_cell->add_data(curv_var, val_skeleton1);
                            if (pos_index_bis + 1 < index_inner_cls.size())
                                pos_index_bis++;
                        }


                        test_curv_var_divergence0->add_data(curv_var, val0);
                        if (*interface_point == *(cl.user_data.interface.end() - 1))
                            distance_pts += 0.0;
                        else
                            distance_pts += (*(interface_point + 1) - *interface_point).to_vector().norm();
// In the case in which *interface_point == *(cl.user_data.interface.end() -1) I'm in the skeleton and it means that the next point it will be in the same abscisse.
                        pos_index++;
                    }
                    first_cut_cell_found = TRUE;
                    first_point = *cl.user_data.interface.begin();
                    cell_end_point = *(cl.user_data.interface.end() - 1);
                } else if (first_cut_cell_found && !(first_point == cell_end_point)) {
                    for (auto &cl: msh_i.cells) {
                        if ((cl.user_data.location == element_location::ON_INTERFACE) &&
                            (cell_end_point == *cl.user_data.interface.begin()) && !(first_point == cell_end_point)) {
                            ls_cell.cell_assignment(cl);

                            bool agglo_cl = cl.user_data.highlight &&
                                            ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                            size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                            std::vector <size_t> index_inner_cls;
                            if (agglo_cl) {
                                for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                                    index_inner_cls.push_back(i_cl * (cl.user_data.interface.size() - 1) / amount_sub_cls);
                            }
                            size_t pos_index = 0;
                            size_t pos_index_bis = 0;

                            for (auto interface_point = cl.user_data.interface.begin();
                                 interface_point < cl.user_data.interface.end(); interface_point++) {

                                T val0 = ls_cell.divergence(*interface_point);

                                point<T, 2> curv_var = point_type(distance_pts, 0.0);
                                if (interface_point == cl.user_data.interface.begin() ||
                                    interface_point == (cl.user_data.interface.end() - 1))
                                    test_curv_var_cell->add_data(curv_var, val0);

                                test_curv_var_divergence0->add_data(curv_var, val0);

                                if (agglo_cl && pos_index == index_inner_cls[pos_index_bis]) {
                                    auto offset_cells = pt_in_subcell_skeleton(ls_cell.level_set.msh, *interface_point, cl);
                                    assert(offset_cells.size() == 2);
                                    auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                                    auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                                    T val_skeleton0 = ls_cell.divergence(*interface_point, subcl0);
                                    T val_skeleton1 = ls_cell.divergence(*interface_point, subcl1);

                                    test_inner_cell->add_data(curv_var, val_skeleton0);
                                    test_inner_cell->add_data(curv_var, val_skeleton1);
                                    if (pos_index_bis + 1 < index_inner_cls.size())
                                        pos_index_bis++;
                                }


                                if (*interface_point == *(cl.user_data.interface.end() - 1))
                                    distance_pts += 0.0;
                                else
                                    distance_pts += (*(interface_point + 1) - *interface_point).to_vector().norm();
                                pos_index++;
                            }
                            cell_end_point = *(cl.user_data.interface.end() - 1);
                        }

                    }

                } else
                    break;

            }


        }


        postoutput_div2.add_object(test_curv_var_divergence0);
        postoutput_div2.add_object(test_curv_var_cell);
        postoutput_div2.add_object(test_inner_cell);

        postoutput_div2.write();




// --------- CHECKING CURVATURE DISC ----------
        postprocess_output <T> postoutput_div_bis;
        std::string filename_curvature_k2 = "k0_curvature_disc_" + std::to_string(time_step) + ".dat";
        auto test_curv_var_divergence2 = std::make_shared < gnuplot_output_object < double > > (filename_curvature_k2);
        std::string filename_curv_var2 = "cell_limit_disc_" + std::to_string(time_step) + ".dat";
        auto test_curv_var_cell2 = std::make_shared < gnuplot_output_object < double > > (filename_curv_var2);
        std::string filename_curv_var_inner_cl2 = "inner_cell_limit_disc_" + std::to_string(time_step) + ".dat";
        auto test_inner_cell2 = std::make_shared < gnuplot_output_object < double > > (filename_curv_var_inner_cl2);

        bool first_cut_cell_found2 = FALSE;
        T distance_pts2 = 0.0;
        point<T, 2> first_point2;
        point<T, 2> cell_end_point2;
        for (auto &cl: msh_i.cells) {

            if (cl.user_data.location == element_location::ON_INTERFACE) {
                ls_cell.cell_assignment(cl);
                if (!first_cut_cell_found2) {
                    bool agglo_cl =
                            cl.user_data.highlight && ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                    size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                    std::vector <size_t> index_inner_cls;
                    if (agglo_cl) {
                        for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                            index_inner_cls.push_back(i_cl * (cl.user_data.interface.size() - 1) / amount_sub_cls);
                    }


                    size_t pos_index = 0;
                    size_t pos_index_bis = 0;

                    for (auto interface_point = cl.user_data.interface.begin();
                         interface_point < cl.user_data.interface.end(); interface_point++) {
                        T val0 = ls_cell.divergence_disc(*interface_point);

                        point<T, 2> curv_var = point_type(distance_pts2, 0.0);
                        if (interface_point == cl.user_data.interface.begin() ||
                            interface_point == (cl.user_data.interface.end() - 1))
                            test_curv_var_cell2->add_data(curv_var, val0);


                        if (agglo_cl && pos_index == index_inner_cls[pos_index_bis]) {
                            auto offset_cells = pt_in_subcell_skeleton(ls_cell.level_set.msh, *interface_point, cl);
                            assert(offset_cells.size() == 2);
                            auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                            auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                            T val_skeleton0 = ls_cell.divergence_disc(*interface_point, subcl0);
                            T val_skeleton1 = ls_cell.divergence_disc(*interface_point, subcl1);

                            test_inner_cell2->add_data(curv_var, val_skeleton0);
                            test_inner_cell2->add_data(curv_var, val_skeleton1);
                            if (pos_index_bis + 1 < index_inner_cls.size())
                                pos_index_bis++;
                        }


                        test_curv_var_divergence2->add_data(curv_var, val0);
                        if (*interface_point == *(cl.user_data.interface.end() - 1))
                            distance_pts2 += 0.0;
                        else
                            distance_pts2 += (*(interface_point + 1) - *interface_point).to_vector().norm();
// In the case in which *interface_point == *(cl.user_data.interface.end() -1) I'm in the skeleton and it means that the next point it will be in the same abscisse.
                        pos_index++;
                    }
                    first_cut_cell_found2 = TRUE;
                    first_point2 = *cl.user_data.interface.begin();
                    cell_end_point2 = *(cl.user_data.interface.end() - 1);
                } else if (first_cut_cell_found2 && !(first_point2 == cell_end_point2)) {
                    for (auto &cl: msh_i.cells) {
                        if ((cl.user_data.location == element_location::ON_INTERFACE) &&
                            (cell_end_point2 == *cl.user_data.interface.begin()) && !(first_point2 == cell_end_point2)) {
                            ls_cell.cell_assignment(cl);

                            bool agglo_cl = cl.user_data.highlight &&
                                            ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                            size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                            std::vector <size_t> index_inner_cls;
                            if (agglo_cl) {
                                for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                                    index_inner_cls.push_back(i_cl * (cl.user_data.interface.size() - 1) / amount_sub_cls);
                            }
                            size_t pos_index = 0;
                            size_t pos_index_bis = 0;

                            for (auto interface_point = cl.user_data.interface.begin();
                                 interface_point < cl.user_data.interface.end(); interface_point++) {

                                T val0 = ls_cell.divergence_disc(*interface_point);

                                point<T, 2> curv_var = point_type(distance_pts2, 0.0);
                                if (interface_point == cl.user_data.interface.begin() ||
                                    interface_point == (cl.user_data.interface.end() - 1))
                                    test_curv_var_cell2->add_data(curv_var, val0);

                                test_curv_var_divergence2->add_data(curv_var, val0);

                                if (agglo_cl && pos_index == index_inner_cls[pos_index_bis]) {
                                    auto offset_cells = pt_in_subcell_skeleton(ls_cell.level_set.msh, *interface_point, cl);
                                    assert(offset_cells.size() == 2);
                                    auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                                    auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                                    T val_skeleton0 = ls_cell.divergence_disc(*interface_point, subcl0);
                                    T val_skeleton1 = ls_cell.divergence_disc(*interface_point, subcl1);

                                    test_inner_cell2->add_data(curv_var, val_skeleton0);
                                    test_inner_cell2->add_data(curv_var, val_skeleton1);
                                    if (pos_index_bis + 1 < index_inner_cls.size())
                                        pos_index_bis++;
                                }


                                if (*interface_point == *(cl.user_data.interface.end() - 1))
                                    distance_pts2 += 0.0;
                                else
                                    distance_pts2 += (*(interface_point + 1) - *interface_point).to_vector().norm();
                                pos_index++;
                            }
                            cell_end_point2 = *(cl.user_data.interface.end() - 1);
                        }

                    }

                } else
                    break;

            }


        }


        postoutput_div_bis.add_object(test_curv_var_divergence2);
        postoutput_div_bis.add_object(test_curv_var_cell2);
        postoutput_div_bis.add_object(test_inner_cell2);

        postoutput_div_bis.write();


        goal_quantities_time_fast(msh_i, interface_points_plot_fin, val_u_n_fin, vec_n, time_step);


//goal_quantities_time(msh , tot_time, interface_points_plot_fin , val_u_nx_fin , val_u_ny_fin , val_u_n_fin , vec_n , velocity_interface , velocity_field , points_vel_field , time_step ) ;
//goal_quantities_time(msh , tot_time, interface_points_plot_fin , val_u_nx_fin , val_u_ny_fin , val_u_n_fin , interface_normals_fin ) ;

//if(time_step == T_N)
//    testing_level_set_time(msh,level_set_function, tot_time,time_step);



    }

    template<typename Mesh, typename Level_Set, typename T, typename Velocity>
    void
    check_goal_quantities_final(Mesh &msh_i, Level_Set &ls_cell, Velocity &u_projected, T &perimeter, T &d_a, T &area_fin,
                                T &centre_mass_x, T &centre_mass_y, size_t degree_FEM, T &mass_fin, size_t degree_velocity,
                                T &l1_divergence_error_fin, T &l2_divergence_error_fin, T &linf_divergence_error_fin,
                                T &radius, T &L1_divergence_error_fin, size_t time_step, T &rise_vel0, T &rise_vel1,
                                T &flux_interface, size_t &counter_interface_pts_fin, size_t degree_curve, size_t n_int) {

// PLOTTING OF NORMAL
        typedef typename Mesh::point_type point_type;

//postprocess_output<double> postoutput_vec;
//auto vec_normal_grad_cont_fin = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_continuos_grad_Stokes_final.dat");



        std::vector <T> val_u_n_fin; //val_u_nx_fin , val_u_ny_fin ;
        std::vector <point<T, 2>> interface_points_plot_fin;
        std::vector <std::pair<T, T>> vec_n; // , velocity_interface , velocity_field , points_vel_field;





        for (auto &cl: msh_i.cells) {
            ls_cell.cell_assignment(cl);
            u_projected.cell_assignment(cl);

            if ((location(msh_i, cl) == element_location::IN_NEGATIVE_SIDE) ||
                (location(msh_i, cl) == element_location::ON_INTERFACE)) {

                T partial_area = measure(msh_i, cl, element_location::IN_NEGATIVE_SIDE);
                area_fin += partial_area;

                auto qps_fin = integrate(msh_i, cl, 2 * degree_FEM, element_location::IN_NEGATIVE_SIDE);

                for (auto &qp: qps_fin) {
                    mass_fin += qp.second * ls_cell(qp.first);
                    centre_mass_x += qp.second * qp.first.x();
                    centre_mass_y += qp.second * qp.first.y();

                    rise_vel0 += qp.second * u_projected(qp.first).first;
                    rise_vel1 += qp.second * u_projected(qp.first).second;
                }

            }
            if (cl.user_data.location == element_location::ON_INTERFACE) {


                auto qps = integrate_interface(msh_i, cl, degree_FEM, element_location::ON_INTERFACE);
                for (auto &qp: qps) {
                    T curv_err = std::abs(ls_cell.divergence(qp.first) + 1.0 / radius);
                    L1_divergence_error_fin += qp.second * curv_err;
                    linf_divergence_error_fin = std::max(linf_divergence_error_fin, curv_err);
                    l1_divergence_error_fin += curv_err;
                    counter_interface_pts_fin++;
                }

                auto qps_un = integrate_interface(msh_i, cl, degree_FEM + degree_velocity, element_location::ON_INTERFACE);
                for (auto &qp: qps_un) {
                    T flux = u_projected(qp.first).first * ls_cell.normal(qp.first)(0) +
                             u_projected(qp.first).second * ls_cell.normal(qp.first)(1);
                    flux_interface += qp.second * flux;


                }


                perimeter += measure_interface(msh_i, cl, element_location::ON_INTERFACE);

                for (auto interface_point = cl.user_data.interface.begin();
                     interface_point < cl.user_data.interface.end(); interface_point++) {
//                       T segment = ( *(interface_point+1) - *interface_point ).to_vector().norm();
//                       perimeter += segment ;
//                       T val0 = ls_cell.divergence( *interface_point );
//                       T curvature_error = std::abs( std::abs(val0) - 1.0/radius ) ;
//                       l1_divergence_error_fin += curvature_error ;
//                       l2_divergence_error_fin += pow(curvature_error,2) ;
//                       linf_divergence_error_fin = std::max(linf_divergence_error_fin ,  curvature_error);



                    Eigen::Matrix<T, 2, 1> normal_cont_grad = ls_cell.normal(*interface_point);
                    std::pair <T, T> normal_vec_grad_cont = std::make_pair(normal_cont_grad(0), normal_cont_grad(1));

                    vec_n.push_back(normal_vec_grad_cont);

                    T u_n_0 = u_projected(*(interface_point)).first * ls_cell.normal(*(interface_point))(0);
                    T u_n_1 = u_projected(*(interface_point)).second * ls_cell.normal(*(interface_point))(1);

                    interface_points_plot_fin.push_back(*(interface_point));

                    val_u_n_fin.push_back(u_n_0 + u_n_1);

                }


            }


        }

// --------- CHECKING CURVATURE DISC GRAD CONT (THE ONE USED INTO THE CODE) ----------
        postprocess_output <T> postoutput_div2;
        std::string filename_curvature_k0 = "k0_curvature_" + std::to_string(time_step) + ".dat";
        auto test_curv_var_divergence0 = std::make_shared < gnuplot_output_object < double > > (filename_curvature_k0);
        std::string filename_curv_var = "cell_limit_curv_var_" + std::to_string(time_step) + ".dat";
        auto test_curv_var_cell = std::make_shared < gnuplot_output_object < double > > (filename_curv_var);
        std::string filename_curv_var_inner_cl = "inner_cell_limit_curv_var_" + std::to_string(time_step) + ".dat";
        auto test_inner_cell = std::make_shared < gnuplot_output_object < double > > (filename_curv_var_inner_cl);

        bool first_cut_cell_found = FALSE;
        T distance_pts = 0.0;
        point<T, 2> first_point;
        point<T, 2> cell_end_point;
        for (auto &cl: msh_i.cells) {

            if (cl.user_data.location == element_location::ON_INTERFACE) {

                if (!first_cut_cell_found) {
                    ls_cell.cell_assignment(cl);
                    bool agglo_cl =
                            cl.user_data.highlight && ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                    size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                    std::vector <size_t> index_inner_cls;
                    if (agglo_cl) {
                        for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                            index_inner_cls.push_back(i_cl * (cl.user_data.interface.size() - 1) / amount_sub_cls);
                    }


                    size_t pos_index = 0;
                    size_t pos_index_bis = 0;

                    for (auto interface_point = cl.user_data.interface.begin();
                         interface_point < cl.user_data.interface.end(); interface_point++) {
                        T val0 = ls_cell.divergence(*interface_point);

                        point<T, 2> curv_var = point_type(distance_pts, 0.0);
                        if (interface_point == cl.user_data.interface.begin() ||
                            interface_point == (cl.user_data.interface.end() - 1))
                            test_curv_var_cell->add_data(curv_var, val0);


                        if (agglo_cl && pos_index == index_inner_cls[pos_index_bis]) {
                            auto offset_cells = pt_in_subcell_skeleton(ls_cell.level_set.msh, *interface_point, cl);
                            assert(offset_cells.size() == 2);
                            auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                            auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                            T val_skeleton0 = ls_cell.divergence(*interface_point, subcl0);
                            T val_skeleton1 = ls_cell.divergence(*interface_point, subcl1);

                            test_inner_cell->add_data(curv_var, val_skeleton0);
                            test_inner_cell->add_data(curv_var, val_skeleton1);
                            if (pos_index_bis + 1 < index_inner_cls.size())
                                pos_index_bis++;
                        }


                        test_curv_var_divergence0->add_data(curv_var, val0);
                        if (*interface_point == *(cl.user_data.interface.end() - 1))
                            distance_pts += 0.0;
                        else
                            distance_pts += (*(interface_point + 1) - *interface_point).to_vector().norm();
// In the case in which *interface_point == *(cl.user_data.interface.end() -1) I'm in the skeleton and it means that the next point it will be in the same abscisse.
                        pos_index++;
                    }
                    first_cut_cell_found = TRUE;
                    first_point = *cl.user_data.interface.begin();
                    cell_end_point = *(cl.user_data.interface.end() - 1);
                } else if (first_cut_cell_found && !(first_point == cell_end_point)) {
                    for (auto &cl: msh_i.cells) {
                        if ((cl.user_data.location == element_location::ON_INTERFACE) &&
                            (cell_end_point == *cl.user_data.interface.begin()) && !(first_point == cell_end_point)) {
                            ls_cell.cell_assignment(cl);

                            bool agglo_cl = cl.user_data.highlight &&
                                            ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                            size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                            std::vector <size_t> index_inner_cls;
                            if (agglo_cl) {
                                for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                                    index_inner_cls.push_back(i_cl * (cl.user_data.interface.size() - 1) / amount_sub_cls);
                            }
                            size_t pos_index = 0;
                            size_t pos_index_bis = 0;

                            for (auto interface_point = cl.user_data.interface.begin();
                                 interface_point < cl.user_data.interface.end(); interface_point++) {

                                T val0 = ls_cell.divergence(*interface_point);

                                point<T, 2> curv_var = point_type(distance_pts, 0.0);
                                if (interface_point == cl.user_data.interface.begin() ||
                                    interface_point == (cl.user_data.interface.end() - 1))
                                    test_curv_var_cell->add_data(curv_var, val0);

                                test_curv_var_divergence0->add_data(curv_var, val0);

                                if (agglo_cl && pos_index == index_inner_cls[pos_index_bis]) {
                                    auto offset_cells = pt_in_subcell_skeleton(ls_cell.level_set.msh, *interface_point, cl);
                                    assert(offset_cells.size() == 2);
                                    auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                                    auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                                    T val_skeleton0 = ls_cell.divergence(*interface_point, subcl0);
                                    T val_skeleton1 = ls_cell.divergence(*interface_point, subcl1);

                                    test_inner_cell->add_data(curv_var, val_skeleton0);
                                    test_inner_cell->add_data(curv_var, val_skeleton1);
                                    if (pos_index_bis + 1 < index_inner_cls.size())
                                        pos_index_bis++;
                                }


                                if (*interface_point == *(cl.user_data.interface.end() - 1))
                                    distance_pts += 0.0;
                                else
                                    distance_pts += (*(interface_point + 1) - *interface_point).to_vector().norm();
                                pos_index++;
                            }
                            cell_end_point = *(cl.user_data.interface.end() - 1);
                        }

                    }

                } else
                    break;

            }


        }


        postoutput_div2.add_object(test_curv_var_divergence0);
        postoutput_div2.add_object(test_curv_var_cell);
        postoutput_div2.add_object(test_inner_cell);

        postoutput_div2.write();




// --------- CHECKING CURVATURE DISC ----------
        postprocess_output <T> postoutput_div_bis;
        std::string filename_curvature_k2 = "k0_curvature_disc_" + std::to_string(time_step) + ".dat";
        auto test_curv_var_divergence2 = std::make_shared < gnuplot_output_object < double > > (filename_curvature_k2);
        std::string filename_curv_var2 = "cell_limit_disc_" + std::to_string(time_step) + ".dat";
        auto test_curv_var_cell2 = std::make_shared < gnuplot_output_object < double > > (filename_curv_var2);
        std::string filename_curv_var_inner_cl2 = "inner_cell_limit_disc_" + std::to_string(time_step) + ".dat";
        auto test_inner_cell2 = std::make_shared < gnuplot_output_object < double > > (filename_curv_var_inner_cl2);

        bool first_cut_cell_found2 = FALSE;
        T distance_pts2 = 0.0;
        point<T, 2> first_point2;
        point<T, 2> cell_end_point2;
        for (auto &cl: msh_i.cells) {

            if (cl.user_data.location == element_location::ON_INTERFACE) {
                ls_cell.cell_assignment(cl);
                if (!first_cut_cell_found2) {
                    bool agglo_cl =
                            cl.user_data.highlight && ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                    size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                    std::vector <size_t> index_inner_cls;
                    if (agglo_cl) {
                        for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                            index_inner_cls.push_back(i_cl * (cl.user_data.interface.size() - 1) / amount_sub_cls);
                    }


                    size_t pos_index = 0;
                    size_t pos_index_bis = 0;

                    for (auto interface_point = cl.user_data.interface.begin();
                         interface_point < cl.user_data.interface.end(); interface_point++) {
                        T val0 = ls_cell.divergence_disc(*interface_point);

                        point<T, 2> curv_var = point_type(distance_pts2, 0.0);
                        if (interface_point == cl.user_data.interface.begin() ||
                            interface_point == (cl.user_data.interface.end() - 1))
                            test_curv_var_cell2->add_data(curv_var, val0);


                        if (agglo_cl && pos_index == index_inner_cls[pos_index_bis]) {
                            auto offset_cells = pt_in_subcell_skeleton(ls_cell.level_set.msh, *interface_point, cl);
                            assert(offset_cells.size() == 2);
                            auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                            auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                            T val_skeleton0 = ls_cell.divergence_disc(*interface_point, subcl0);
                            T val_skeleton1 = ls_cell.divergence_disc(*interface_point, subcl1);

                            test_inner_cell2->add_data(curv_var, val_skeleton0);
                            test_inner_cell2->add_data(curv_var, val_skeleton1);
                            if (pos_index_bis + 1 < index_inner_cls.size())
                                pos_index_bis++;
                        }


                        test_curv_var_divergence2->add_data(curv_var, val0);
                        if (*interface_point == *(cl.user_data.interface.end() - 1))
                            distance_pts2 += 0.0;
                        else
                            distance_pts2 += (*(interface_point + 1) - *interface_point).to_vector().norm();
// In the case in which *interface_point == *(cl.user_data.interface.end() -1) I'm in the skeleton and it means that the next point it will be in the same abscisse.
                        pos_index++;
                    }
                    first_cut_cell_found2 = TRUE;
                    first_point2 = *cl.user_data.interface.begin();
                    cell_end_point2 = *(cl.user_data.interface.end() - 1);
                } else if (first_cut_cell_found2 && !(first_point2 == cell_end_point2)) {
                    for (auto &cl: msh_i.cells) {
                        if ((cl.user_data.location == element_location::ON_INTERFACE) &&
                            (cell_end_point2 == *cl.user_data.interface.begin()) && !(first_point2 == cell_end_point2)) {
                            ls_cell.cell_assignment(cl);

                            bool agglo_cl = cl.user_data.highlight &&
                                            ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                            size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                            std::vector <size_t> index_inner_cls;
                            if (agglo_cl) {
                                for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                                    index_inner_cls.push_back(i_cl * (cl.user_data.interface.size() - 1) / amount_sub_cls);
                            }
                            size_t pos_index = 0;
                            size_t pos_index_bis = 0;

                            for (auto interface_point = cl.user_data.interface.begin();
                                 interface_point < cl.user_data.interface.end(); interface_point++) {

                                T val0 = ls_cell.divergence_disc(*interface_point);

                                point<T, 2> curv_var = point_type(distance_pts2, 0.0);
                                if (interface_point == cl.user_data.interface.begin() ||
                                    interface_point == (cl.user_data.interface.end() - 1))
                                    test_curv_var_cell2->add_data(curv_var, val0);

                                test_curv_var_divergence2->add_data(curv_var, val0);

                                if (agglo_cl && pos_index == index_inner_cls[pos_index_bis]) {
                                    auto offset_cells = pt_in_subcell_skeleton(ls_cell.level_set.msh, *interface_point, cl);
                                    assert(offset_cells.size() == 2);
                                    auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                                    auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                                    T val_skeleton0 = ls_cell.divergence_disc(*interface_point, subcl0);
                                    T val_skeleton1 = ls_cell.divergence_disc(*interface_point, subcl1);

                                    test_inner_cell2->add_data(curv_var, val_skeleton0);
                                    test_inner_cell2->add_data(curv_var, val_skeleton1);
                                    if (pos_index_bis + 1 < index_inner_cls.size())
                                        pos_index_bis++;
                                }


                                if (*interface_point == *(cl.user_data.interface.end() - 1))
                                    distance_pts2 += 0.0;
                                else
                                    distance_pts2 += (*(interface_point + 1) - *interface_point).to_vector().norm();
                                pos_index++;
                            }
                            cell_end_point2 = *(cl.user_data.interface.end() - 1);
                        }

                    }

                } else
                    break;

            }


        }


        postoutput_div_bis.add_object(test_curv_var_divergence2);
        postoutput_div_bis.add_object(test_curv_var_cell2);
        postoutput_div_bis.add_object(test_inner_cell2);

        postoutput_div_bis.write();


        goal_quantities_time_fast(msh_i, interface_points_plot_fin, val_u_n_fin, vec_n, time_step);


//goal_quantities_time(msh , tot_time, interface_points_plot_fin , val_u_nx_fin , val_u_ny_fin , val_u_n_fin , vec_n , velocity_interface , velocity_field , points_vel_field , time_step ) ;
//goal_quantities_time(msh , tot_time, interface_points_plot_fin , val_u_nx_fin , val_u_ny_fin , val_u_n_fin , interface_normals_fin ) ;

//if(time_step == T_N)
//    testing_level_set_time(msh,level_set_function, tot_time,time_step);



    }




    template<typename Mesh, typename Level_Set, typename T>
    void
    check_disc_curvature(Mesh &msh_i, Level_Set &ls_cell, T curvature_anal, size_t degree_FEM) {


        T L1_divergence_error = 0.0;
        T linf_divergence_error = 0.0;
        T l1_divergence_error = 0.0;

        size_t counter_interface_pts = 0;


        for (auto &cl: msh_i.cells) {
            ls_cell.cell_assignment(cl);


            if (cl.user_data.location == element_location::ON_INTERFACE) {


                auto qps = integrate_interface(msh_i, cl, degree_FEM, element_location::ON_INTERFACE);
                for (auto &qp: qps) {
                    T val = std::abs(std::abs(ls_cell.divergence_disc(qp.first)) - std::abs(curvature_anal));
                    L1_divergence_error += qp.second * val;
                    linf_divergence_error = std::max(linf_divergence_error, val);
                    l1_divergence_error += val;
                    counter_interface_pts++;
                }


            }
        }

        l1_divergence_error /= counter_interface_pts;

        std::cout << "Number of interface points is " << counter_interface_pts << std::endl;
        std::cout << "The l1 error of the LS-DISC-CURVATURE, at INITIAL time is " << l1_divergence_error << std::endl;
        std::cout << "The linf error of the LS-DISC-CURVATURE, at INITIAL time is " << linf_divergence_error << std::endl;

        std::cout << "The L1 error of the LS-DISC-CURVATURE, at INITIAL time is " << L1_divergence_error << std::endl;


    }


    template<typename Mesh, typename Level_Set, typename T>
    void
    check_goal_quantities(Mesh &msh_i, Level_Set &ls_cell, T &perimeter_initial, T &d_a, T &initial_area,
                          T &centre_mass_x_inital, T &centre_mass_y_inital, size_t degree_FEM, T &initial_mass, bool flower,
                          T &l1_divergence_error, T &l2_divergence_error, T &linf_divergence_error, T &radius,
                          T &L1_divergence_error, bool ellipse, size_t degree_curve, size_t n_int) {


        typedef typename Mesh::point_type point_type;



//postprocess_output<double> postoutput_vec;
//auto vec_normal_grad_cont = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_continuos_grad_Stokes.dat");

        postprocess_output<double> postoutput_div2;
        std::string filename_curvature_k0 = "k0_curvature_initial.dat";
        auto test_curv_var_divergence0 = std::make_shared < gnuplot_output_object < double > > (filename_curvature_k0);

        std::string filename_curv_var = "cell_limit_curv_var_initial.dat";
        auto test_curv_var_cell = std::make_shared < gnuplot_output_object < double > > (filename_curv_var);

        std::string filename_curv_var_inner_cl = "inner_cell_limit_curv_var_initial.dat";
        auto test_inner_cell = std::make_shared < gnuplot_output_object < double > > (filename_curv_var_inner_cl);

        std::vector <point<T, 2>> interface_points_plot;
        std::vector <std::pair<T, T>> interface_normals;
        size_t counter_interface_pts = 0;

// CHECKING OF AREA, MASS, CENTRE OF MASS, PERIMETER, CURVATURE ERRORS
        for (auto &cl: msh_i.cells) {
            ls_cell.cell_assignment(cl);

            if (location(msh_i, cl) == element_location::IN_NEGATIVE_SIDE ||
                location(msh_i, cl) == element_location::ON_INTERFACE) {
                T partial_area = measure(msh_i, cl, element_location::IN_NEGATIVE_SIDE);

                initial_area += partial_area;
                auto qps = integrate(msh_i, cl, degree_FEM, element_location::IN_NEGATIVE_SIDE);
                for (auto &qp: qps) {
                    initial_mass += qp.second * ls_cell(qp.first);
                    centre_mass_x_inital += qp.second * qp.first.x();
                    centre_mass_y_inital += qp.second * qp.first.y();
                }
            }

            if (cl.user_data.location == element_location::ON_INTERFACE) {
                perimeter_initial += measure_interface(msh_i, cl, element_location::ON_INTERFACE);

                auto qps = integrate_interface(msh_i, cl, degree_FEM, element_location::ON_INTERFACE);
                for (auto &qp: qps) {
                    T val = std::abs(std::abs(ls_cell.divergence(qp.first)) - std::abs(1.0 / radius));
                    L1_divergence_error += qp.second * val;
                    linf_divergence_error = std::max(linf_divergence_error, val);
                    l1_divergence_error += val;
                    counter_interface_pts++;
                }


                for (auto interface_point = cl.user_data.interface.begin();
                     interface_point < cl.user_data.interface.end(); interface_point++) {


//                T val0 = ls_cell.divergence( *interface_point );
//                T error_curvature = std::abs( val0 + 1.0/radius) ;
//                l1_divergence_error += error_curvature;
//                l2_divergence_error += pow(error_curvature,2) ;
//                linf_divergence_error = std::max(linf_divergence_error , error_curvature ) ;

                    Eigen::Matrix<T, 2, 1> normal_grad_cont = ls_cell.normal(*interface_point);
                    std::pair <T, T> normal_vec_grad_cont = std::make_pair(normal_grad_cont(0), normal_grad_cont(1));
                    interface_normals.push_back(normal_vec_grad_cont);
                    interface_points_plot.push_back(*(interface_point));

//                counter_interface_pts++;

                }


            }
        }


// PLOTTING OF THE CURVATURE IN A 2d FRAME
        bool first_cut_cell_found = FALSE;
        T distance_pts = 0.0;
        point<T, 2> first_point;
        point<T, 2> cell_end_point;

// --------- CHECKING CURVATURE DISC GRAD CONT ----------
        for (auto &cl: msh_i.cells) {

            if (cl.user_data.location == element_location::ON_INTERFACE) {
                ls_cell.cell_assignment(cl);
                if (!first_cut_cell_found) {
                    bool agglo_cl =
                            cl.user_data.highlight && ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                    size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                    std::vector <size_t> index_inner_cls;
                    if (agglo_cl) {
                        for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                            index_inner_cls.push_back(i_cl * (cl.user_data.interface.size() - 1) / amount_sub_cls);
                    }
//                if( agglo_cl && amount_sub_cls == 2 )
//                    index_inner_cls.push_back( (cl.user_data.interface.size() - 1)/amount_sub_cls );
//                if( agglo_cl && amount_sub_cls == 3 ){
//                    index_inner_cls.push_back( (cl.user_data.interface.size() - 1)/amount_sub_cls );
//                    index_inner_cls.push_back( 2.0*(cl.user_data.interface.size() - 1)/amount_sub_cls );
//                }
                    size_t pos_index = 0;
                    size_t pos_index_bis = 0;

                    for (auto interface_point = cl.user_data.interface.begin();
                         interface_point < cl.user_data.interface.end(); interface_point++) {
                        T val0 = ls_cell.divergence(*interface_point);

                        point<T, 2> curv_var = point_type(distance_pts, 0.0);
                        if (interface_point == cl.user_data.interface.begin() ||
                            interface_point == (cl.user_data.interface.end() - 1))
                            test_curv_var_cell->add_data(curv_var, val0);


                        if (agglo_cl && pos_index == index_inner_cls[pos_index_bis]) {
                            auto offset_cells = pt_in_subcell_skeleton(ls_cell.level_set.msh, *interface_point, cl);
                            assert(offset_cells.size() == 2);
                            auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                            auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                            T val_skeleton0 = ls_cell.divergence(*interface_point, subcl0);
                            T val_skeleton1 = ls_cell.divergence(*interface_point, subcl1);

                            test_inner_cell->add_data(curv_var, val_skeleton0);
                            test_inner_cell->add_data(curv_var, val_skeleton1);
                            if (pos_index_bis + 1 < index_inner_cls.size())
                                pos_index_bis++;
                        }

                        test_curv_var_divergence0->add_data(curv_var, val0);
                        if (*interface_point == *(cl.user_data.interface.end() - 1))
                            distance_pts += 0.0;
                        else
                            distance_pts += (*(interface_point + 1) - *interface_point).to_vector().norm();
// In the case in which *interface_point == *(cl.user_data.interface.end() -1) I'm in the skeleton and it means that the next point it will be in the same abscisse.
                        pos_index++;
                    }
                    first_cut_cell_found = TRUE;
                    first_point = *cl.user_data.interface.begin();
                    cell_end_point = *(cl.user_data.interface.end() - 1);
                } else if (first_cut_cell_found && !(first_point == cell_end_point)) {
                    for (auto &cl: msh_i.cells) {
                        if ((cl.user_data.location == element_location::ON_INTERFACE) &&
                            (cell_end_point == *cl.user_data.interface.begin()) && !(first_point == cell_end_point)) {
                            ls_cell.cell_assignment(cl);

                            bool agglo_cl = cl.user_data.highlight &&
                                            ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                            size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                            std::vector <size_t> index_inner_cls;
                            if (agglo_cl) {
                                for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                                    index_inner_cls.push_back(i_cl * (cl.user_data.interface.size() - 1) / amount_sub_cls);
                            }
                            size_t pos_index = 0;
                            size_t pos_index_bis = 0;

                            for (auto interface_point = cl.user_data.interface.begin();
                                 interface_point < cl.user_data.interface.end(); interface_point++) {

                                T val0 = ls_cell.divergence(*interface_point);

                                point<T, 2> curv_var = point_type(distance_pts, 0.0);
                                if (interface_point == cl.user_data.interface.begin() ||
                                    interface_point == (cl.user_data.interface.end() - 1))
                                    test_curv_var_cell->add_data(curv_var, val0);

                                test_curv_var_divergence0->add_data(curv_var, val0);

                                if (agglo_cl && pos_index == index_inner_cls[pos_index_bis]) {
                                    auto offset_cells = pt_in_subcell_skeleton(ls_cell.level_set.msh, *interface_point, cl);
                                    assert(offset_cells.size() == 2);
                                    auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                                    auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                                    T val_skeleton0 = ls_cell.divergence(*interface_point, subcl0);
                                    T val_skeleton1 = ls_cell.divergence(*interface_point, subcl1);

                                    test_inner_cell->add_data(curv_var, val_skeleton0);
                                    test_inner_cell->add_data(curv_var, val_skeleton1);
                                    if (pos_index_bis + 1 < index_inner_cls.size())
                                        pos_index_bis++;
                                }


                                if (*interface_point == *(cl.user_data.interface.end() - 1))
                                    distance_pts += 0.0;
                                else
                                    distance_pts += (*(interface_point + 1) - *interface_point).to_vector().norm();
                                pos_index++;
                            }
                            cell_end_point = *(cl.user_data.interface.end() - 1);
                        }

                    }

                } else
                    break;

            }


        }
        postoutput_div2.add_object(test_curv_var_divergence0);
        postoutput_div2.add_object(test_curv_var_cell);
        postoutput_div2.add_object(test_inner_cell);

        postoutput_div2.write();

        std::cout << "--------> First Point curvilinear variable = " << first_point << std::endl;

// --------- CHECKING CURVATURE DISC ----------
        postprocess_output <T> postoutput_div_bis;
        std::string filename_curvature_k2 = "k0_curvature_disc_initial.dat";
        auto test_curv_var_divergence2 = std::make_shared < gnuplot_output_object < double > > (filename_curvature_k2);

        std::string filename_curv_var2 = "cell_limit_disc_initial.dat";
        auto test_curv_var_cell2 = std::make_shared < gnuplot_output_object < double > > (filename_curv_var2);

        std::string filename_curv_var_inner_cl2 = "inner_cell_limit_disc_initial.dat";
        auto test_inner_cell2 = std::make_shared < gnuplot_output_object < double > > (filename_curv_var_inner_cl2);


        bool first_cut_cell_found2 = FALSE;
        T distance_pts2 = 0.0;
        point<T, 2> first_point2;
        point<T, 2> cell_end_point2;
        for (auto &cl: msh_i.cells) {

            if (cl.user_data.location == element_location::ON_INTERFACE) {
                ls_cell.cell_assignment(cl);
                if (!first_cut_cell_found2) {
                    bool agglo_cl =
                            cl.user_data.highlight && ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                    size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                    std::vector <size_t> index_inner_cls;
                    if (agglo_cl) {
                        for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                            index_inner_cls.push_back(i_cl * (cl.user_data.interface.size() - 1) / amount_sub_cls);
                    }


                    size_t pos_index = 0;
                    size_t pos_index_bis = 0;

                    for (auto interface_point = cl.user_data.interface.begin();
                         interface_point < cl.user_data.interface.end(); interface_point++) {
                        T val0 = ls_cell.divergence_disc(*interface_point);

                        point<T, 2> curv_var = point_type(distance_pts2, 0.0);
                        if (interface_point == cl.user_data.interface.begin() ||
                            interface_point == (cl.user_data.interface.end() - 1))
                            test_curv_var_cell2->add_data(curv_var, val0);


                        if (agglo_cl && pos_index == index_inner_cls[pos_index_bis]) {
                            auto offset_cells = pt_in_subcell_skeleton(ls_cell.level_set.msh, *interface_point, cl);
                            assert(offset_cells.size() == 2);
                            auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                            auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                            T val_skeleton0 = ls_cell.divergence_disc(*interface_point, subcl0);
                            T val_skeleton1 = ls_cell.divergence_disc(*interface_point, subcl1);

                            test_inner_cell2->add_data(curv_var, val_skeleton0);
                            test_inner_cell2->add_data(curv_var, val_skeleton1);
                            if (pos_index_bis + 1 < index_inner_cls.size())
                                pos_index_bis++;
                        }


                        test_curv_var_divergence2->add_data(curv_var, val0);
                        if (*interface_point == *(cl.user_data.interface.end() - 1))
                            distance_pts2 += 0.0;
                        else
                            distance_pts2 += (*(interface_point + 1) - *interface_point).to_vector().norm();
// In the case in which *interface_point == *(cl.user_data.interface.end() -1) I'm in the skeleton and it means that the next point it will be in the same abscisse.
                        pos_index++;
                    }
                    first_cut_cell_found2 = TRUE;
                    first_point2 = *cl.user_data.interface.begin();
                    cell_end_point2 = *(cl.user_data.interface.end() - 1);
                } else if (first_cut_cell_found2 && !(first_point2 == cell_end_point2)) {
                    for (auto &cl: msh_i.cells) {
                        if ((cl.user_data.location == element_location::ON_INTERFACE) &&
                            (cell_end_point2 == *cl.user_data.interface.begin()) && !(first_point2 == cell_end_point2)) {
                            ls_cell.cell_assignment(cl);

                            bool agglo_cl = cl.user_data.highlight &&
                                            ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                            size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                            std::vector <size_t> index_inner_cls;
                            if (agglo_cl) {
                                for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                                    index_inner_cls.push_back(i_cl * (cl.user_data.interface.size() - 1) / amount_sub_cls);
                            }
                            size_t pos_index = 0;
                            size_t pos_index_bis = 0;

                            for (auto interface_point = cl.user_data.interface.begin();
                                 interface_point < cl.user_data.interface.end(); interface_point++) {

                                T val0 = ls_cell.divergence_disc(*interface_point);

                                point<T, 2> curv_var = point_type(distance_pts2, 0.0);
                                if (interface_point == cl.user_data.interface.begin() ||
                                    interface_point == (cl.user_data.interface.end() - 1))
                                    test_curv_var_cell2->add_data(curv_var, val0);

                                test_curv_var_divergence2->add_data(curv_var, val0);

                                if (agglo_cl && pos_index == index_inner_cls[pos_index_bis]) {
                                    auto offset_cells = pt_in_subcell_skeleton(ls_cell.level_set.msh, *interface_point, cl);
                                    assert(offset_cells.size() == 2);
                                    auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                                    auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                                    T val_skeleton0 = ls_cell.divergence_disc(*interface_point, subcl0);
                                    T val_skeleton1 = ls_cell.divergence_disc(*interface_point, subcl1);

                                    test_inner_cell2->add_data(curv_var, val_skeleton0);
                                    test_inner_cell2->add_data(curv_var, val_skeleton1);
                                    if (pos_index_bis + 1 < index_inner_cls.size())
                                        pos_index_bis++;
                                }


                                if (*interface_point == *(cl.user_data.interface.end() - 1))
                                    distance_pts2 += 0.0;
                                else
                                    distance_pts2 += (*(interface_point + 1) - *interface_point).to_vector().norm();
                                pos_index++;
                            }
                            cell_end_point2 = *(cl.user_data.interface.end() - 1);
                        }

                    }

                } else
                    break;

            }


        }


        postoutput_div_bis.add_object(test_curv_var_divergence2);
        postoutput_div_bis.add_object(test_curv_var_cell2);
        postoutput_div_bis.add_object(test_inner_cell2);

        postoutput_div_bis.write();


        if (!flower)  // Circular case (to calcualte the Curvature error I use the analytic radius)
        {
            l1_divergence_error /= counter_interface_pts;

//        l2_divergence_error = sqrt(l2_divergence_error/counter_interface_pts);
            std::cout << "Number of interface points is " << counter_interface_pts << std::endl;
            std::cout << "The l1 error of the CURVATURE at the INTERFACE, at INITIAL time is " << l1_divergence_error
                      << std::endl;
//        std::cout<<"The l2 error of the CURVATURE at the INTERFACE, at INITIAL time is " << l2_divergence_error <<std::endl;
            std::cout << "The linf error of the CURVATURE at the INTERFACE, at INITIAL time is " << linf_divergence_error
                      << std::endl;

            std::cout << "The L1 error of the CURVATURE at the INTERFACE, at INITIAL time is " << L1_divergence_error
                      << std::endl;

        }

        std::cout << bold << yellow << '\n' << "Initial time, AREA  = " << initial_area << reset << std::endl;
//    std::cout << "Initial time, MASS  = "<< initial_mass   << std::endl;
        std::cout << "Initial time, CENTRE OF MASS  = ( " << centre_mass_x_inital / initial_area << " , "
                  << centre_mass_y_inital / initial_area << " )." << std::endl;

        d_a = sqrt(4.0 * initial_area / M_PI);
//    std::cout<<"Initial time, PERIMETER = " << perimeter_initial <<std::endl;
        std::cout << "Initial time, CIRCULARITY (OLD) = " << M_PI * d_a / perimeter_initial << std::endl;
        std::cout << "Initial time, CIRCULARITY = " << 4.0 * M_PI * initial_area / (perimeter_initial * perimeter_initial)
                  << std::endl;

        T perimeter_anal = 2.0 * M_PI * radius;
//    std::cout<<"Error( perimetre NEW - perimeter_anal ) = " << perimeter_initial - perimeter_anal <<std::endl;





        if (flower || ellipse) // FLOWER O ELLIPSE CASE: the radius is calculated by the area conservation
        {
//        std::cout<<"OLD radius = " << radius <<std::endl;
            l1_divergence_error = 0., l2_divergence_error = 0.;
            L1_divergence_error = 0.;
            T linf_divergence_error = -10.;
            radius = sqrt(initial_area / M_PI);
//        std::cout<<"FROM CONSERVATION OF AREA -> radius = " << radius <<std::endl;
            for (auto &cl: msh_i.cells) {
                ls_cell.cell_assignment(cl);

                if (cl.user_data.location == element_location::ON_INTERFACE) {
                    auto qps = integrate_interface(msh_i, cl, degree_FEM, element_location::ON_INTERFACE);
                    for (auto &qp: qps) {
                        T val = std::abs(std::abs(ls_cell.divergence(qp.first)) - 1.0 / radius);
                        L1_divergence_error += qp.second * val;
                        linf_divergence_error = std::max(linf_divergence_error, val);
                        l1_divergence_error += val;


                    }

//                for(auto interface_point = cl.user_data.interface.begin() ; interface_point < cl.user_data.interface.end() ; interface_point++ )
//                {
//
//
//                    T val0 = ls_cell.divergence( *interface_point );
//                    T error_curvature = std::abs( val0 + 1.0/radius) ;
//                    l1_divergence_error += error_curvature;
//                    l2_divergence_error += pow(error_curvature,2) ;
//                    linf_divergence_error = std::max(linf_divergence_error , error_curvature ) ;
//
//                }
                }
            }
            l1_divergence_error /= counter_interface_pts;

//        l2_divergence_error = sqrt(l2_divergence_error/counter_interface_pts);

            std::cout << "Number of interface points is " << counter_interface_pts << std::endl;
            std::cout << "The l1 error of the CURVATURE at the INTERFACE, at INITIAL time is " << l1_divergence_error
                      << std::endl;
//        std::cout<<"The l2 error of the CURVATURE at the INTERFACE, at INITIAL time is " << l2_divergence_error <<std::endl;
            std::cout << "The linf error of the CURVATURE at the INTERFACE, at INITIAL time is " << linf_divergence_error
                      << std::endl;
            std::cout << "The L1 error of the CURVATURE at the INTERFACE, at INITIAL time is " << L1_divergence_error
                      << std::endl;

        }


    }




    template<typename Mesh, typename Level_Set, typename T = typename Mesh::coordinate_type>
    void
    plot_curvature_normal_vs_curv_abscisse_PARAMETRIC(Mesh &msh_i, Level_Set &ls_cell, size_t degree_curve, size_t n_int,
                                                      size_t time_step, size_t degree_curvature) {


        typedef typename Mesh::point_type point_type;


        bool l2proj_para = false;

        bool l2proj = true;
        bool avg = false;
        bool disc = false;


        Interface_parametrisation_mesh1d_global <Mesh> curve(msh_i, degree_curve, degree_curvature);

// *********************** DERIVATIVE / NORMAL PARA *************************//
//------------- L2 cont curvature from parametric interface  r ---------- //
        curve.make_L2_proj_para_derivative(msh_i);

//---------------------------- L2 global Normal from LS  ----------------------- //
        if (l2proj) {
            if (!disc)
                curve.make_L2_proj_para_normal(msh_i, ls_cell);
            else
                curve.make_L2_proj_para_normal_disc(msh_i, ls_cell);
        }
//---------------------------- Avg Normal from LS  ---------------------------- //
        if (avg) {
            if (!disc)
                curve.make_avg_L2_local_proj_para_normal(msh_i, ls_cell);
            else
                curve.make_avg_L2_local_proj_para_normal_disc(msh_i, ls_cell);
        }


// *********************** CURVATURE PARA *************************//

//------------- L2 cont curvature from parametric interface  r ---------- //
        if (l2proj_para)
            curve.make_L2_proj_para_curvature(msh_i);



//---------------------------- L2 global Curvature from LS  ----------------------- //
        if (l2proj) {
            if (!disc)
                curve.make_L2_proj_para_curvature(msh_i, ls_cell);
            else
                curve.make_L2_proj_para_curvature_disc(msh_i, ls_cell);
        }
//---------------------------- Avg Curvature from LS  ---------------------------- //
        if (avg) {
            if (!disc)
                curve.make_avg_L2_local_proj_para_curvature(msh_i, ls_cell);
            else
                curve.make_avg_L2_local_proj_para_curvature_disc(msh_i, ls_cell);

        }


        Matrix<T, 2, 1> ref_vec;
        ref_vec(0) = 1.0;
        ref_vec(1) = 0.0;
//postprocess_output<double> postoutput_vec;
//auto vec_normal_grad_cont = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_continuos_grad_Stokes.dat");






        postprocess_output<double> postoutput_div_para;
        std::string filename_curvature_k0 = "curvature_para_2d_" + std::to_string(time_step) + ".dat";
        auto test_curvature_para = std::make_shared < gnuplot_output_object < double > > (filename_curvature_k0);

        std::string filename_curv_var_para = "cell_limit_curv_var_para_cont.dat";
        auto test_curv_var_para = std::make_shared < gnuplot_output_object < double > > (filename_curv_var_para);

//    auto test_jacobian = std::make_shared< gnuplot_output_object<double> >("jacobian_para_cont.dat");
//    auto test_jacobian_cl = std::make_shared< gnuplot_output_object<double> >("jacobian_para_cont_cells.dat");

        std::string filename_curv_var_inner_cl = "inner_cell_limit_para_cont.dat";
        auto test_inner_cell = std::make_shared < gnuplot_output_object < double > > (filename_curv_var_inner_cl);


        T distance_pts_para = 0.0;
        bool first_cut_cell_found = FALSE;
        point<T, 2> first_point;
        point<T, 2> cell_end_point;
        T tot = degree_curve; // *(pow(2,n_int)); // degree_curve - 1 ; //degree_curve - 2 ; // 10
        for (auto &cl: msh_i.cells) {

            if (cl.user_data.location == element_location::ON_INTERFACE) {
//            std::cout<<"cell = "<<offset(msh_i, cl);
                auto global_cells_i = curve.get_global_cells_interface(msh_i, cl);
                auto integration_msh = cl.user_data.integration_msh;
                if (!first_cut_cell_found) {
                    bool agglo_cl =
                            cl.user_data.highlight && ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                    size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                    std::vector <size_t> index_inner_cls;
                    if (agglo_cl) {
                        for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                            index_inner_cls.push_back(i_cl * (integration_msh.cells.size()) / amount_sub_cls);
                    }

                    size_t pos_index_bis = 0;

                    for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                        size_t global_cl_i = global_cells_i[i_cell];
//                    std::cout<<"i_cell = "<<i_cell<<" , global global_cl_i = "<<global_cl_i<<std::endl;
                        auto pts = points(integration_msh, integration_msh.cells[i_cell]);

//                    if( i_cell == 0)
//                        first_point = pts[0] ;
//                    if( i_cell == integration_msh.cells.size()-1)
//                        cell_end_point =  pts[1] ;

                        for (T i = 0.0; i <= tot; i++) {
                            T pos = 0.0 + i / tot;

                            T val0 = curve.curvature_cont(pos, global_cl_i);
//                        std::cout<<"pos = "<<pos<<" , pt = "<<curve(pos, pts , degree_curve)<<" , k = "<<val0<<std::endl;
                            T pj = curve.jacobian_cont(pos, global_cl_i);

//                        std::cout<<"distance_pts_para = "<<distance_pts_para<<std::endl;
                            point<T, 2> curv_var = point_type(distance_pts_para, 0.0);

                            if (agglo_cl && i_cell == index_inner_cls[pos_index_bis] && i == 0) {

//                            auto pts_old = points(integration_msh,integration_msh.cells[i_cell-1]);
                                T valold = curve.curvature_cont(1.0, global_cl_i - 1);

                                test_inner_cell->add_data(curv_var, val0);
                                test_inner_cell->add_data(curv_var, valold);
                                if (pos_index_bis + 1 < index_inner_cls.size())
                                    pos_index_bis++;
                            }


                            if ((i_cell == 0 && pos == 0.0) || (i_cell == integration_msh.cells.size() - 1 && pos == 1.0)) {
                                test_curv_var_para->add_data(curv_var, val0);
//                            test_jacobian_cl->add_data(curv_var,pj );
//                            std::cout<<"interface cell bdry -> YES "<<std::endl;
                            }

                            test_curvature_para->add_data(curv_var, val0);
//                        test_jacobian->add_data(curv_var,pj );
                            T dist;
                            if (pos == 1)
                                dist = 0.0;
                            else
                                dist = (curve(pos + 1.0 / tot, pts, degree_curve) - curve(pos, pts, degree_curve)).norm();

                            distance_pts_para += dist;


                        }

                    }
                    first_cut_cell_found = TRUE;
                    first_point = *cl.user_data.interface.begin();
                    cell_end_point = *(cl.user_data.interface.end() - 1);
                } else if (first_cut_cell_found && !(first_point == cell_end_point)) {
                    for (auto &cl: msh_i.cells) {


                        if ((cl.user_data.location == element_location::ON_INTERFACE) &&
                            (cell_end_point == *cl.user_data.interface.begin()) && !(first_point == cell_end_point)) {
//                        std::cout<<"cell = "<<offset(msh_i, cl);
//                        std::cout<<"first_point =  "<<first_point<<std::endl;
//                        std::cout<<"*cl.user_data.interface.begin() =  "<<*cl.user_data.interface.begin()<<std::endl;
//                        std::cout<<"cell_end_point =  "<<cell_end_point<<std::endl;
                            auto integration_msh = cl.user_data.integration_msh;
                            auto global_cells_i = curve.get_global_cells_interface(msh_i, cl);
                            bool agglo_cl = cl.user_data.highlight &&
                                            ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                            size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                            std::vector <size_t> index_inner_cls;
                            if (agglo_cl) {
                                for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                                    index_inner_cls.push_back(i_cl * (integration_msh.cells.size()) / amount_sub_cls);
                            }

                            size_t pos_index_bis = 0;


                            for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                                auto pts = points(integration_msh, integration_msh.cells[i_cell]);
                                size_t global_cl_i = global_cells_i[i_cell];
//                            std::cout<<"i_cell = "<<i_cell<<" , global global_cl_i = "<<global_cl_i<<std::endl;
//                            if( i_cell == integration_msh.cells.size()-1)
//                                cell_end_point =  pts[1] ;

                                for (T i = 0.0; i <= tot; i++) {
                                    T pos = 0.0 + i / tot;
                                    T val0 = curve.curvature_cont(pos, global_cl_i);
                                    T pj = curve.jacobian_cont(pos, global_cl_i);

//                                std::cout<<"pos = "<<pos<<" , pt = "<<curve(pos, pts , degree_curve)<<" , k = "<<val0<<std::endl;
//                                std::cout<<"distance_pts_para = "<<distance_pts_para<<std::endl;
                                    point<T, 2> curv_var = point_type(distance_pts_para, 0.0);
                                    if (agglo_cl && i_cell == index_inner_cls[pos_index_bis] && i == 0) {

//                                    auto pts_old = points(integration_msh,integration_msh.cells[i_cell-1]);
                                        T valold = curve.curvature_cont(1.0, global_cl_i - 1);

                                        test_inner_cell->add_data(curv_var, val0);
                                        test_inner_cell->add_data(curv_var, valold);
                                        if (pos_index_bis + 1 < index_inner_cls.size())
                                            pos_index_bis++;
                                    }


                                    if ((i_cell == 0 && pos == 0.0) ||
                                        (i_cell == integration_msh.cells.size() - 1 && pos == 1.0)) {
                                        test_curv_var_para->add_data(curv_var, val0);
//                                    test_jacobian_cl->add_data(curv_var,pj );
//                                    std::cout<<"interface cell bdry -> YES "<<std::endl;
                                    }

                                    test_curvature_para->add_data(curv_var, val0);
//                                test_jacobian->add_data(curv_var,pj );

                                    T dist;
                                    if (pos == 1)
                                        dist = 0.0;
                                    else
                                        dist = (curve(pos + 1.0 / tot, pts, degree_curve) -
                                                curve(pos, pts, degree_curve)).norm();

                                    distance_pts_para += dist;


                                }

                            }
                            cell_end_point = *(cl.user_data.interface.end() - 1);
//                        std::cout<<"cell_end_point =  "<<cell_end_point<<std::endl;
                        }

                    }

                } else
                    break;

            }


        }
        postoutput_div_para.add_object(test_curvature_para);
        postoutput_div_para.add_object(test_curv_var_para);

//    postoutput_div_para.add_object(test_jacobian);
//    postoutput_div_para.add_object(test_jacobian_cl);
        postoutput_div_para.add_object(test_inner_cell);

        postoutput_div_para.write();


    }



    template<typename T>
    void
    plotting_in_time_new(const std::vector <T> &time_vec, const std::vector <T> &area_time,
                         const std::vector <T> &l1_err_u_n_time, const std::vector <T> &linf_err_u_n_time,
                         const std::vector <T> &max_val_u_n_time, const std::vector <T> &l1_err_curvature_time,
                         const std::vector <T> &linf_err_curvature_time, T dt,
                         const std::vector <std::pair<T, T>> &min_max_vec, const std::vector <T> &flux_interface_time,
                         const std::vector <std::pair<T, T>> &rise_velocity_time,
                         const std::vector <std::pair<T, T>> &centre_mass_err_time, const std::vector <T> &perimeter_time,
                         const std::vector <T> &circularity_time, T circularity_ref, T perimetre_ref, T area_ref, T radius,
                         const std::vector <T> &L1_err_u_n_time, const std::vector <T> &l1_err_u_n_time_para,
                         const std::vector <T> &linf_err_u_n_time_para, const std::vector <T> &L1_err_u_n_time_para,
                         const std::vector <T> &max_val_u_n_time_para, const std::vector <T> &linf_der_time_interface,
                         const std::vector <T> &eccentricity_vec, std::string &folder) {

        postprocess_output <T> postoutput;

//    auto testref0  = std::make_shared< gnuplot_output_object_time<T> >("area_ref_time.dat");
//    auto testref1  = std::make_shared< gnuplot_output_object_time<T> >("perimeter_ref_time.dat");
//    auto testref2  = std::make_shared< gnuplot_output_object_time<T> >("circularity_ref_time.dat");

        T area_analytic = M_PI * radius * radius;
        T perimeter_analytic = 2.0 * M_PI * radius;
//    auto testanal0  = std::make_shared< gnuplot_output_object_time<T> >("area_anal_time.dat");
//    auto testanal1  = std::make_shared< gnuplot_output_object_time<T> >("perimeter_anal_time.dat");


        auto test0 = std::make_shared < gnuplot_output_object_time < T > > (folder + "area_time.dat");
        auto test1 = std::make_shared < gnuplot_output_object_time < T > > (folder + "lpiccolo1_err_u_n_time.dat");
        auto test2 = std::make_shared < gnuplot_output_object_time < T > > (folder + "linf_err_u_n_time.dat");

        auto test_der_int = std::make_shared < gnuplot_output_object_time < T > > (folder + "linf_der_interface_time.dat");

//    auto test1_counter  = std::make_shared< gnuplot_output_object_time<T> >("lpiccolo1_err_u_n_time_counter.dat");
//    auto test2_counter  = std::make_shared< gnuplot_output_object_time<T> >("linf_err_u_n_time_counter.dat");

//    auto test3  = std::make_shared< gnuplot_output_object_time<T> >("max_val_u_n_time.dat");
        auto test4 = std::make_shared < gnuplot_output_object_time < T > > (folder + "l1_err_curvature_time.dat");
        auto test5 = std::make_shared < gnuplot_output_object_time < T > > (folder + "linf_err_curvature_time.dat");
//    auto test4_counter  = std::make_shared< gnuplot_output_object_time<T> >("l1_err_curvature_time_counter.dat");
//    auto test5_counter  = std::make_shared< gnuplot_output_object_time<T> >("linf_err_curvature_time_counter.dat");

        auto test_new0 = std::make_shared < gnuplot_output_object_time < T > > (folder + "Lgrande1_err_u_n_time.dat");
//    auto test_new1  = std::make_shared< gnuplot_output_object_time<T> >("lpiccolo1_err_u_n_time_para.dat");
//    auto test_new2  = std::make_shared< gnuplot_output_object_time<T> >("linf_err_u_n_time_para.dat");
//    auto test_new3  = std::make_shared< gnuplot_output_object_time<T> >("Lgrande1_err_u_n_time_para.dat");
//    auto test_new4  = std::make_shared< gnuplot_output_object_time<T> >("max_val_u_n_time_para.dat");

//    auto test_new0_counter  = std::make_shared< gnuplot_output_object_time<T> >("Lgrande1_err_u_n_time_counter.dat");
//    auto test_new1_counter  = std::make_shared< gnuplot_output_object_time<T> >("lpiccolo1_err_u_n_time_para_counter.dat");
//    auto test_new2_counter  = std::make_shared< gnuplot_output_object_time<T> >("linf_err_u_n_time_para_counter.dat");
//    auto test_new3_counter  = std::make_shared< gnuplot_output_object_time<T> >("Lgrande1_err_u_n_time_para_counter.dat");
//    auto test_new4_counter  = std::make_shared< gnuplot_output_object_time<T> >("max_val_u_n_time_para_counter.dat");




//    auto test0b  = std::make_shared< gnuplot_output_object_time<T> >("area_time_normalised.dat");

//    auto test1b  = std::make_shared< gnuplot_output_object_time<T> >("l1_err_u_n_time_normalised.dat");
//    auto test2b  = std::make_shared< gnuplot_output_object_time<T> >("linf_err_u_n_time_normalised.dat");
//    auto test3b  = std::make_shared< gnuplot_output_object_time<T> >("max_val_u_n_time_normalised.dat");
//    auto test4b  = std::make_shared< gnuplot_output_object_time<T> >("l1_err_curvature_time_normalised.dat");
//    auto test5b  = std::make_shared< gnuplot_output_object_time<T> >("linf_err_curvature_time_normalised.dat");
// //auto test4c  = std::make_shared< gnuplot_output_object_time<T> >("l1_l1_err_curvature_err_time.dat");

        auto test_dt = std::make_shared < gnuplot_output_object_time < T > > (folder + "dt_M.dat");
        auto test_e = std::make_shared < gnuplot_output_object_time < T > > (folder + "eccentricity.dat");

        auto test0c = std::make_shared < gnuplot_output_object_time < T > > (folder + "area_time_err.dat");
        auto test0c_counter = std::make_shared < gnuplot_output_object_time < T > > (folder + "area_time_err_counter.dat");

//    auto testm0  = std::make_shared< gnuplot_output_object_time<T> >("min_time.dat");
//    auto testm1  = std::make_shared< gnuplot_output_object_time<T> >("min_normalised_time.dat");
//    auto testM0  = std::make_shared< gnuplot_output_object_time<T> >("max_time.dat");
//    auto testM1  = std::make_shared< gnuplot_output_object_time<T> >("max_normalised_time.dat");
        auto testflux = std::make_shared < gnuplot_output_object_time < T > > (folder + "flux_interface_time.dat");
//    auto testvelx  = std::make_shared< gnuplot_output_object_time<T> >("rise_velocity_err_x_time.dat");
//    auto testvely  = std::make_shared< gnuplot_output_object_time<T> >("rise_velocity_err_y_time.dat");
        auto testvel = std::make_shared < gnuplot_output_object_time < T > > (folder + "rise_velocity_err_time.dat");
//    auto testvel_counter  = std::make_shared< gnuplot_output_object_time<T> >("rise_velocity_err_time_counter.dat");
        auto testcomx = std::make_shared < gnuplot_output_object_time < T > > (folder + "centre_mass_err_x_time.dat");
        auto testcomy = std::make_shared < gnuplot_output_object_time < T > > (folder + "centre_mass_err_y_time.dat");
        auto testcom = std::make_shared < gnuplot_output_object_time < T > > (folder + "centre_mass_err_time.dat");


        auto testper0 = std::make_shared < gnuplot_output_object_time < T > > (folder + "perimeter_time.dat");
//    auto testper1  = std::make_shared< gnuplot_output_object_time<T> >("perimeter_normalised_time.dat");
//    auto testper2  = std::make_shared< gnuplot_output_object_time<T> >("perimeter_err_time.dat");
//    auto testper2_counter  = std::make_shared< gnuplot_output_object_time<T> >("perimeter_err_time_counter.dat");

        auto testcirc = std::make_shared < gnuplot_output_object_time < T > > (folder + "circularity_time.dat");
        auto testcirc1 = std::make_shared < gnuplot_output_object_time < T > > (folder + "circularity_error_time.dat");
        auto testcirc1_counter =
                std::make_shared < gnuplot_output_object_time < T > > (folder + "circularity_error_time_counter.dat");

        size_t tot = l1_err_u_n_time.size();

//    testm0->add_data(time_vec[0] ,min_max_vec[0].first );
//    testm1->add_data(time_vec[0] ,1.0 );
//    testM0->add_data(time_vec[0] ,min_max_vec[0].second );
//    testM1->add_data(time_vec[0] ,1.0 );

        testcomx->add_data(time_vec[0], 0.0);
        testcomy->add_data(time_vec[0], 0.0);
        testcom->add_data(time_vec[0], 0.0);


        testper0->add_data(time_vec[0], perimeter_time[0]);
//    testper1->add_data(time_vec[0] , 1.0 );
//    testper2->add_data(time_vec[0] , std::abs(perimeter_time[0] - perimetre_ref)/perimetre_ref );
//    testper2_counter->add_data(0 , std::abs(perimeter_time[0] - perimetre_ref)/perimetre_ref );

        testcirc->add_data(time_vec[0], circularity_time[0]);
        testcirc1->add_data(time_vec[0], std::abs(circularity_time[0] - circularity_ref) / circularity_ref);
        testcirc1_counter->add_data(0, std::abs(circularity_time[0] - circularity_ref) / circularity_ref);

        test0c->add_data(time_vec[0], 0.0);
        test0c_counter->add_data(0, 0.0);
//test4c->add_data(time_vec[0] , 0.0 );


        test0->add_data(time_vec[0], area_time[0]);
// ADD 8/01/21
        test1->add_data(time_vec[0], l1_err_u_n_time[0]);
        test2->add_data(time_vec[0], linf_err_u_n_time[0]);
//    test1_counter->add_data(0 ,l1_err_u_n_time[0] );
//    test2_counter->add_data(0 ,linf_err_u_n_time[0] );
//    test3->add_data(time_vec[0] ,max_val_u_n_time[0] );
// UP TO HERE
        test4->add_data(time_vec[0], l1_err_curvature_time[0]);
        test5->add_data(time_vec[0], linf_err_curvature_time[0]);
//    test4_counter->add_data(0 ,l1_err_curvature_time[0] );
//    test5_counter->add_data(0 ,linf_err_curvature_time[0] );

        test_new0->add_data(time_vec[0], L1_err_u_n_time[0]);
//    test_new1->add_data(time_vec[0] ,l1_err_u_n_time_para[0] );
//    test_new2->add_data(time_vec[0] ,linf_err_u_n_time_para[0] );
//    test_new3->add_data(time_vec[0] ,L1_err_u_n_time_para[0] );
//    test_new4->add_data(time_vec[0] ,max_val_u_n_time_para[0] );
//
//    test_new0_counter->add_data(0 ,L1_err_u_n_time[0] );
//    test_new1_counter->add_data(0 ,l1_err_u_n_time_para[0] );
//    test_new2_counter->add_data(0 ,linf_err_u_n_time_para[0] );
//    test_new3_counter->add_data(0,L1_err_u_n_time_para[0] );
//    test_new4_counter->add_data(0 ,max_val_u_n_time_para[0] );
//
//
//
//    test0b->add_data(time_vec[0] ,area_time[0]/area_time[0] );
// ADD 8/01/21
//    test1b->add_data(time_vec[0] ,l1_err_u_n_time[0]/l1_err_u_n_time[0] );
//    test2b->add_data(time_vec[0] ,linf_err_u_n_time[0]/linf_err_u_n_time[0] );
//    test3b->add_data(time_vec[0] ,max_val_u_n_time[0]/max_val_u_n_time[0]  );
// UP TO HERE
//    test4b->add_data(time_vec[0] ,l1_err_curvature_time[0]/l1_err_curvature_time[0]  );
//    test5b->add_data(time_vec[0] ,linf_err_curvature_time[0]/linf_err_curvature_time[0]  );
//
//    testref0->add_data(time_vec[0] , area_ref );
//    testref1->add_data(time_vec[0] , perimetre_ref );
//    testref2->add_data(time_vec[0] , circularity_ref );

//    testanal0->add_data(time_vec[0] , area_analytic );
//    testanal1->add_data(time_vec[0] , perimeter_analytic );


        testflux->add_data(time_vec[0], flux_interface_time[0]);
//    testvelx ->add_data(time_vec[0] , std::abs(rise_velocity_time[0].first) );
//    testvely ->add_data(time_vec[0] , std::abs(rise_velocity_time[0].second) );
        testvel->add_data(time_vec[0], std::abs(rise_velocity_time[0].first) + std::abs(rise_velocity_time[0].second));
//    testvel_counter->add_data(0, std::abs(rise_velocity_time[0].first) + std::abs(rise_velocity_time[0].second) );
        test_dt->add_data(time_vec[0], 0.0);
        test_e->add_data(time_vec[0], eccentricity_vec[0]);
        test_der_int->add_data(time_vec[0], 0.0);
        for (size_t i = 0; i < tot - 1; i++) {
            test0->add_data(time_vec[i + 1], area_time[i + 1]);
            test1->add_data(time_vec[i + 1], l1_err_u_n_time[i + 1]);
            test2->add_data(time_vec[i + 1], linf_err_u_n_time[i + 1]);
            test_der_int->add_data(time_vec[i + 1], linf_der_time_interface[i]);

//        test1_counter->add_data(i+1 ,l1_err_u_n_time[i+1] );
//        test2_counter->add_data(i+1 ,linf_err_u_n_time[i+1] );
//        test3->add_data(time_vec[i+1] ,max_val_u_n_time[i+1] );
////        test1->add_data(time_vec[i+1] ,l1_err_u_n_time[i] );
////        test2->add_data(time_vec[i+1] ,linf_err_u_n_time[i] );
////        test1_counter->add_data(i+1 ,l1_err_u_n_time[i] );
////        test2_counter->add_data(i+1 ,linf_err_u_n_time[i] );
////        test3->add_data(time_vec[i+1] ,max_val_u_n_time[i] );
            test4->add_data(time_vec[i + 1], l1_err_curvature_time[i + 1]);
            test5->add_data(time_vec[i + 1], linf_err_curvature_time[i + 1]);
//        test4_counter->add_data(i+1 ,l1_err_curvature_time[i+1] );
//        test5_counter->add_data(i+1 ,linf_err_curvature_time[i+1] );

//        test0b->add_data(time_vec[i+1] ,area_time[i+1]/area_time[0] );
//
//        test1b->add_data(time_vec[i+1] ,l1_err_u_n_time[i+1]/l1_err_u_n_time[0] );
//        test2b->add_data(time_vec[i+1] ,linf_err_u_n_time[i+1]/linf_err_u_n_time[0] );
//        test3b->add_data(time_vec[i+1] ,max_val_u_n_time[i+1]/max_val_u_n_time[0]  );
////        test1b->add_data(time_vec[i+1] ,l1_err_u_n_time[i]/l1_err_u_n_time[0] );
////        test2b->add_data(time_vec[i+1] ,linf_err_u_n_time[i]/linf_err_u_n_time[0] );
////        test3b->add_data(time_vec[i+1] ,max_val_u_n_time[i]/max_val_u_n_time[0]  );
//        test4b->add_data(time_vec[i+1] ,l1_err_curvature_time[i+1]/l1_err_curvature_time[0] );
//        test5b->add_data(time_vec[i+1] ,linf_err_curvature_time[i+1]/linf_err_curvature_time[0]  );

            test_new0->add_data(time_vec[i + 1], L1_err_u_n_time[i + 1]);
//        test_new1->add_data(time_vec[i+1] ,l1_err_u_n_time_para[i+1] );
//        test_new2->add_data(time_vec[i+1] ,linf_err_u_n_time_para[i+1] );
//        test_new3->add_data(time_vec[i+1] ,L1_err_u_n_time_para[i+1] );
//        test_new4->add_data(time_vec[i+1] ,max_val_u_n_time_para[i+1] );

//        test_new0_counter->add_data(i+1 ,L1_err_u_n_time[i+1] );
//        test_new1_counter->add_data(i+1 ,l1_err_u_n_time_para[i+1] );
//        test_new2_counter->add_data(i+1 ,linf_err_u_n_time_para[i+1] );
//        test_new3_counter->add_data(i+1,L1_err_u_n_time_para[i+1] );
//        test_new4_counter->add_data(i+1 ,max_val_u_n_time_para[i+1] );

            testcomx->add_data(time_vec[i + 1],
                               std::abs(centre_mass_err_time[i + 1].first - centre_mass_err_time[0].first) /
                               centre_mass_err_time[0].first);
            testcomy->add_data(time_vec[i + 1],
                               std::abs(centre_mass_err_time[i + 1].second - centre_mass_err_time[0].second) /
                               centre_mass_err_time[0].second);
            testcom->add_data(time_vec[i + 1], std::abs(centre_mass_err_time[i + 1].first - centre_mass_err_time[0].first) /
                                               centre_mass_err_time[0].first + std::abs(
                    centre_mass_err_time[i + 1].second - centre_mass_err_time[0].second) / centre_mass_err_time[0].second);


//        testm0->add_data(time_vec[i+1] , min_max_vec[i+1].first );
//        testm1->add_data(time_vec[i+1] ,min_max_vec[i+1].first/min_max_vec[0].first );
//        testM0->add_data(time_vec[i+1] , min_max_vec[i+1].second );
//        testM1->add_data(time_vec[i+1] ,min_max_vec[i+1].second/min_max_vec[0].second );


            testper0->add_data(time_vec[i + 1], perimeter_time[i + 1]);
//        testper1->add_data(time_vec[i+1] , perimeter_time[i+1]/perimeter_time[0] );
//        testper2->add_data(time_vec[i+1] , std::abs( perimeter_time[i+1] - perimetre_ref) / perimetre_ref );
//        testper2_counter->add_data(i+1 , std::abs( perimeter_time[i+1] - perimetre_ref) / perimetre_ref );

            testcirc->add_data(time_vec[i + 1], circularity_time[i + 1]);
            testcirc1->add_data(time_vec[i + 1], std::abs(circularity_time[i + 1] - circularity_ref) / circularity_ref);
            testcirc1_counter->add_data(i + 1, std::abs(circularity_time[i + 1] - circularity_ref) / circularity_ref);

            test0c->add_data(time_vec[i + 1], std::abs(area_time[i + 1] - area_time[0]) / area_time[0]);
            test0c_counter->add_data(i + 1, std::abs(area_time[i + 1] - area_time[0]) / area_time[0]);
//test4c->add_data(time_vec[i+1] , std::abs(l1_err_curvature_time[i+1] - l1_err_curvature_time[0] )/l1_err_curvature_time[0]);

            testflux->add_data(time_vec[i + 1], flux_interface_time[i + 1]);
//        testvelx ->add_data(time_vec[i+1] , std::abs(rise_velocity_time[i+1].first) );
//        testvely ->add_data(time_vec[i+1] , std::abs(rise_velocity_time[i+1].second) );
            testvel->add_data(time_vec[i + 1],
                              std::abs(rise_velocity_time[i + 1].first) + std::abs(rise_velocity_time[i + 1].second));
//        testvel_counter->add_data(i+1 , std::abs(rise_velocity_time[i+1].first) + std::abs(rise_velocity_time[i+1].second) );

//        testref0->add_data(time_vec[i+1] , area_ref );
//        testref1->add_data(time_vec[i+1] , perimetre_ref );
//        testref2->add_data(time_vec[i+1] , circularity_ref );
//
//        testanal0->add_data(time_vec[i+1] , area_analytic );
//        testanal1->add_data(time_vec[i+1] , perimeter_analytic );

//std::cout<<"time_vec[i] = "<<time_vec[i]<<", time_vec[i+1] = "<<time_vec[i+1]<<", dt = "<<dt<<", error = "<<std::abs( std::abs( time_vec[i+1] - time_vec[i] ) - dt )<<std::endl;
//        if( std::abs( std::abs( time_vec[i+1] - time_vec[i] ) - dt ) > 1e-10 ){
//
//            test_dt->add_data(time_vec[i+1] , 0.0 );
//        }
            test_dt->add_data(time_vec[i + 1], time_vec[i + 1] - time_vec[i]);
            test_e->add_data(time_vec[i + 1], eccentricity_vec[i]);

        }


        postoutput.add_object(test0);
        postoutput.add_object(test1);
        postoutput.add_object(test2);
        postoutput.add_object(test_der_int);


//    postoutput.add_object(test1_counter);
//    postoutput.add_object(test2_counter);


//    postoutput.add_object(test3);
        postoutput.add_object(test4);
        postoutput.add_object(test5);
//    postoutput.add_object(test4_counter);
//    postoutput.add_object(test5_counter);

        postoutput.add_object(test_new0);
//    postoutput.add_object(test_new1);
//    postoutput.add_object(test_new2);
//    postoutput.add_object(test_new3);
//    postoutput.add_object(test_new4);
//
//    postoutput.add_object(test_new0_counter);
//    postoutput.add_object(test_new1_counter);
//    postoutput.add_object(test_new2_counter);
//    postoutput.add_object(test_new3_counter);
//    postoutput.add_object(test_new4_counter);
//
//
//
//
//    postoutput.add_object(test0b);
//    postoutput.add_object(test1b);
//    postoutput.add_object(test2b);
//    postoutput.add_object(test3b);
//    postoutput.add_object(test4b);
//    postoutput.add_object(test5b);

        postoutput.add_object(testcomx);
        postoutput.add_object(testcomy);
        postoutput.add_object(testcom);

//    postoutput.add_object(testm0);
//    postoutput.add_object(testm1);
//    postoutput.add_object(testM0);
//    postoutput.add_object(testM1);

        postoutput.add_object(testper0);
//    postoutput.add_object(testper1);
//    postoutput.add_object(testper2);
//    postoutput.add_object(testper2_counter);


        postoutput.add_object(testcirc);
        postoutput.add_object(testcirc1);
        postoutput.add_object(testcirc1_counter);

        postoutput.add_object(test0c);
        postoutput.add_object(test0c_counter);
//postoutput.add_object(test4c);

        postoutput.add_object(testflux);
//    postoutput.add_object(testvelx);
//    postoutput.add_object(testvely);
        postoutput.add_object(testvel);
//    postoutput.add_object(testvel_counter);


//    postoutput.add_object(testref0);
//    postoutput.add_object(testref1);
//    postoutput.add_object(testref2);

//    postoutput.add_object(testanal0);
//    postoutput.add_object(testanal1);


        postoutput.add_object(test_dt);
        postoutput.add_object(test_e);

        postoutput.write();

    }


    template<typename Mesh, typename Parametric_Interface, typename T = typename Mesh::coordinate_type>
    void
    plot_curvature_normal_vs_curv_abscisse_PARAMETRIC(Mesh &msh_i, size_t degree_curve, size_t n_int, size_t time_step,
                                                      size_t degree_curvature, Parametric_Interface &curve,
                                                      std::string &folder) {


        typedef typename Mesh::point_type point_type;


        Matrix<T, 2, 1> ref_vec;
        ref_vec(0) = 1.0;
        ref_vec(1) = 0.0;
//postprocess_output<double> postoutput_vec;
//auto vec_normal_grad_cont = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_continuos_grad_Stokes.dat");






        postprocess_output<double> postoutput_div_para;
        std::string filename_curvature_k0 = folder + "curvature_filtered_para_2d_" + std::to_string(time_step) + ".dat";
        auto test_curvature_para = std::make_shared < gnuplot_output_object < double > > (filename_curvature_k0);

        std::string filename_curv_var_para =
                folder + "cell_limit_curv_filter_var_para_" + std::to_string(time_step) + ".dat";
        auto test_curv_var_para = std::make_shared < gnuplot_output_object < double > > (filename_curv_var_para);

//    auto test_jacobian = std::make_shared< gnuplot_output_object<double> >("jacobian_para_cont.dat");
//    auto test_jacobian_cl = std::make_shared< gnuplot_output_object<double> >("jacobian_para_cont_cells.dat");

        std::string filename_curv_var_inner_cl =
                folder + "inner_cell_limit_curv_filter_para_" + std::to_string(time_step) + ".dat";
        auto test_inner_cell = std::make_shared < gnuplot_output_object < double > > (filename_curv_var_inner_cl);


        T distance_pts_para = 0.0;
        bool first_cut_cell_found = FALSE;
        point<T, 2> first_point;
        point<T, 2> cell_end_point;
        T tot = degree_curve; //*pow(2,n_int); // degree_curve - 1 ; //degree_curve - 2 ; // 10
        for (auto &cl: msh_i.cells) {

            if (cl.user_data.location == element_location::ON_INTERFACE) {
//            std::cout<<"cell = "<<offset(msh_i, cl);
                auto global_cells_i = curve.get_global_cells_interface(msh_i, cl);
                auto integration_msh = cl.user_data.integration_msh;
                if (!first_cut_cell_found) {
                    bool agglo_cl =
                            cl.user_data.highlight && ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                    size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                    std::vector <size_t> index_inner_cls;
                    if (agglo_cl) {
                        for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                            index_inner_cls.push_back(i_cl * (integration_msh.cells.size()) / amount_sub_cls);
                    }

                    size_t pos_index_bis = 0;

                    for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                        size_t global_cl_i = global_cells_i[i_cell];
//                    std::cout<<"i_cell = "<<i_cell<<" , global global_cl_i = "<<global_cl_i<<std::endl;
                        auto pts = points(integration_msh, integration_msh.cells[i_cell]);

//                    if( i_cell == 0)
//                        first_point = pts[0] ;
//                    if( i_cell == integration_msh.cells.size()-1)
//                        cell_end_point =  pts[1] ;

                        for (T i = 0.0; i <= tot; i++) {
                            T pos = 0.0 + i / tot;

                            T val0 = curve.curvature_cont(pos, global_cl_i);
//                        std::cout<<"pos = "<<pos<<" , pt = "<<curve(pos, pts , degree_curve)<<" , k = "<<val0<<std::endl;
                            T pj = curve.jacobian_cont(pos, global_cl_i);

//                        std::cout<<"distance_pts_para = "<<distance_pts_para<<std::endl;
                            point<T, 2> curv_var = point_type(distance_pts_para, 0.0);

                            if (agglo_cl && i_cell == index_inner_cls[pos_index_bis] && i == 0) {

                                size_t global_cl_i_prev;
                                if (global_cl_i == 0)
                                    global_cl_i_prev = curve.ndof_dd - 1;
                                else
                                    global_cl_i_prev = global_cl_i - 1;
//                            auto pts_old = points(integration_msh,integration_msh.cells[i_cell-1]);
                                T valold = curve.curvature_cont(1.0, global_cl_i_prev);

                                test_inner_cell->add_data(curv_var, val0);
                                test_inner_cell->add_data(curv_var, valold);
                                if (pos_index_bis + 1 < index_inner_cls.size())
                                    pos_index_bis++;
                            }


                            if ((i_cell == 0 && pos == 0.0) || (i_cell == integration_msh.cells.size() - 1 && pos == 1.0)) {
                                test_curv_var_para->add_data(curv_var, val0);
//                            test_jacobian_cl->add_data(curv_var,pj );
//                            std::cout<<"interface cell bdry -> YES "<<std::endl;
                            }

                            test_curvature_para->add_data(curv_var, val0);
//                        test_jacobian->add_data(curv_var,pj );
                            T dist;
                            if (pos == 1)
                                dist = 0.0;
                            else
                                dist = (curve(pos + 1.0 / tot, pts, degree_curve) - curve(pos, pts, degree_curve)).norm();

                            distance_pts_para += dist;


                        }

                    }
                    first_cut_cell_found = TRUE;
                    first_point = *cl.user_data.interface.begin();
                    cell_end_point = *(cl.user_data.interface.end() - 1);
                } else if (first_cut_cell_found && !(first_point == cell_end_point)) {
                    for (auto &cl: msh_i.cells) {


                        if ((cl.user_data.location == element_location::ON_INTERFACE) &&
                            (cell_end_point == *cl.user_data.interface.begin()) && !(first_point == cell_end_point)) {
//                        std::cout<<"cell = "<<offset(msh_i, cl);
//                        std::cout<<"first_point =  "<<first_point<<std::endl;
//                        std::cout<<"*cl.user_data.interface.begin() =  "<<*cl.user_data.interface.begin()<<std::endl;
//                        std::cout<<"cell_end_point =  "<<cell_end_point<<std::endl;
                            auto integration_msh = cl.user_data.integration_msh;
                            auto global_cells_i = curve.get_global_cells_interface(msh_i, cl);
                            bool agglo_cl = cl.user_data.highlight &&
                                            ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                            size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                            std::vector <size_t> index_inner_cls;
                            if (agglo_cl) {
                                for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                                    index_inner_cls.push_back(i_cl * (integration_msh.cells.size()) / amount_sub_cls);
                            }

                            size_t pos_index_bis = 0;


                            for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                                auto pts = points(integration_msh, integration_msh.cells[i_cell]);
                                size_t global_cl_i = global_cells_i[i_cell];
//                            std::cout<<"i_cell = "<<i_cell<<" , global global_cl_i = "<<global_cl_i<<std::endl;
//                            if( i_cell == integration_msh.cells.size()-1)
//                                cell_end_point =  pts[1] ;

                                for (T i = 0.0; i <= tot; i++) {
                                    T pos = 0.0 + i / tot;
                                    T val0 = curve.curvature_cont(pos, global_cl_i);
                                    T pj = curve.jacobian_cont(pos, global_cl_i);

//                                std::cout<<"pos = "<<pos<<" , pt = "<<curve(pos, pts , degree_curve)<<" , k = "<<val0<<std::endl;
//                                std::cout<<"distance_pts_para = "<<distance_pts_para<<std::endl;
                                    point<T, 2> curv_var = point_type(distance_pts_para, 0.0);
                                    if (agglo_cl && i_cell == index_inner_cls[pos_index_bis] && i == 0) {

//                                    auto pts_old = points(integration_msh,integration_msh.cells[i_cell-1]);
                                        size_t global_cl_i_prev;
                                        if (global_cl_i == 0)
                                            global_cl_i_prev = curve.ndof_dd - 1;
                                        else
                                            global_cl_i_prev = global_cl_i - 1;
                                        T valold = curve.curvature_cont(1.0, global_cl_i_prev);

                                        test_inner_cell->add_data(curv_var, val0);
                                        test_inner_cell->add_data(curv_var, valold);
                                        if (pos_index_bis + 1 < index_inner_cls.size())
                                            pos_index_bis++;
                                    }


                                    if ((i_cell == 0 && pos == 0.0) ||
                                        (i_cell == integration_msh.cells.size() - 1 && pos == 1.0)) {
                                        test_curv_var_para->add_data(curv_var, val0);
//                                    test_jacobian_cl->add_data(curv_var,pj );
//                                    std::cout<<"interface cell bdry -> YES "<<std::endl;
                                    }

                                    test_curvature_para->add_data(curv_var, val0);
//                                test_jacobian->add_data(curv_var,pj );

                                    T dist;
                                    if (pos == 1)
                                        dist = 0.0;
                                    else
                                        dist = (curve(pos + 1.0 / tot, pts, degree_curve) -
                                                curve(pos, pts, degree_curve)).norm();

                                    distance_pts_para += dist;


                                }

                            }
                            cell_end_point = *(cl.user_data.interface.end() - 1);
//                        std::cout<<"cell_end_point =  "<<cell_end_point<<std::endl;
                        }

                    }

                } else
                    break;

            }


        }
        postoutput_div_para.add_object(test_curvature_para);
        postoutput_div_para.add_object(test_curv_var_para);

//    postoutput_div_para.add_object(test_jacobian);
//    postoutput_div_para.add_object(test_jacobian_cl);
        postoutput_div_para.add_object(test_inner_cell);

        postoutput_div_para.write();


    }



    template<typename Mesh, typename Parametric_Interface, typename T = typename Mesh::coordinate_type>
    void
    plot_curvature_normal_vs_curv_abscisse_PARAMETRIC(const Mesh &msh, size_t n_int, const Parametric_Interface &curve,
                                                      size_t time_step) {


        typedef typename Mesh::point_type point_type;


        Matrix<T, 2, 1> ref_vec;
        ref_vec(0) = 1.0;
        ref_vec(1) = 0.0;
//postprocess_output<double> postoutput_vec;
//auto vec_normal_grad_cont = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_continuos_grad_Stokes.dat");



        size_t dd_degree = curve.dd_degree;
        size_t degree_curve = curve.basis_degree;


        postprocess_output<double> postoutput_div_para;
        std::string filename_curvature_k0 = "curvature_para_Hp_2d_" + std::to_string(time_step) + ".dat";
        auto test_curvature_para = std::make_shared < gnuplot_output_object < double > > (filename_curvature_k0);

        std::string filename_curv_var_para = "cell_limit_curv_var_para_Hp_" + std::to_string(time_step) + ".dat";
        auto test_curv_var_para = std::make_shared < gnuplot_output_object < double > > (filename_curv_var_para);

//    auto test_jacobian = std::make_shared< gnuplot_output_object<double> >("jacobian_para_cont.dat");
//    auto test_jacobian_cl = std::make_shared< gnuplot_output_object<double> >("jacobian_para_cont_cells.dat");

        std::string filename_curv_var_inner_cl = "inner_cell_limit_para_Hp_" + std::to_string(time_step) + ".dat";
        auto test_inner_cell = std::make_shared < gnuplot_output_object < double > > (filename_curv_var_inner_cl);


        T distance_pts_para = 0.0;
        bool first_cut_cell_found = FALSE;
        point<T, 2> first_point;
        point<T, 2> cell_end_point;
        T tot = degree_curve; // degree_curve - 1 ; //degree_curve - 2 ; // 10

        for (auto &cl: msh.cells) {

            if (cl.user_data.location == element_location::ON_INTERFACE) {
//            std::cout<<"cell = "<<offset(msh_i, cl);
                auto global_cells_i = curve.get_global_cells_interface(msh, cl);
                auto integration_msh = cl.user_data.integration_msh;
                if (!first_cut_cell_found) {
                    bool agglo_cl =
                            cl.user_data.highlight && ((cl.user_data.interface.size() > pow(2, n_int) * dd_degree + 1));
                    size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                    std::vector <size_t> index_inner_cls;
                    if (agglo_cl) {
                        for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                            index_inner_cls.push_back(i_cl * (integration_msh.cells.size()) / amount_sub_cls);
                    }

                    size_t pos_index_bis = 0;

                    for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                        size_t global_cl_i = global_cells_i[i_cell];
//                    std::cout<<"i_cell = "<<i_cell<<" , global global_cl_i = "<<global_cl_i<<std::endl;
                        auto pts = points(integration_msh, integration_msh.cells[i_cell]);

//                    if( i_cell == 0)
//                        first_point = pts[0] ;
//                    if( i_cell == integration_msh.cells.size()-1)
//                        cell_end_point =  pts[1] ;

                        for (T i = 0.0; i <= tot; i++) {
                            T pos = 0.0 + i / tot;

                            T val0 = curve.curvature_cont(pos, global_cl_i);
//                        std::cout<<"pos = "<<pos<<" , pt = "<<curve(pos, pts , degree_curve)<<" , k = "<<val0<<std::endl;
//                            T pj = curve.jacobian_cont(pos , global_cl_i ) ;

//                        std::cout<<"distance_pts_para = "<<distance_pts_para<<std::endl;
                            point<T, 2> curv_var = point_type(distance_pts_para, 0.0);

                            if (agglo_cl && i_cell == index_inner_cls[pos_index_bis] && i == 0) {
                                size_t global_cl_i_prev;
                                if (global_cl_i == 0)
                                    global_cl_i_prev = curve.ndof_dd - 1;
                                else
                                    global_cl_i_prev = global_cl_i - 1;
//                             auto pts_old = points(integration_msh,integration_msh.cells[i_cell-1]);
                                T valold = curve.curvature_cont(1.0, global_cl_i_prev);

                                test_inner_cell->add_data(curv_var, val0);
                                test_inner_cell->add_data(curv_var, valold);
                                if (pos_index_bis + 1 < index_inner_cls.size())
                                    pos_index_bis++;
                            }


                            if ((i_cell == 0 && pos == 0.0) || (i_cell == integration_msh.cells.size() - 1 && pos == 1.0)) {
                                test_curv_var_para->add_data(curv_var, val0);
//                            test_jacobian_cl->add_data(curv_var,pj );
//                            std::cout<<"interface cell bdry -> YES "<<std::endl;
                            }

                            test_curvature_para->add_data(curv_var, val0);
//                        test_jacobian->add_data(curv_var,pj );
                            T dist;
                            if (pos == 1)
                                dist = 0.0;
                            else
                                dist = (curve(pos + 1.0 / tot, pts, dd_degree) - curve(pos, pts, dd_degree)).norm();

                            distance_pts_para += dist;


                        }

                    }
                    first_cut_cell_found = TRUE;
                    first_point = *cl.user_data.interface.begin();
                    cell_end_point = *(cl.user_data.interface.end() - 1);
                } else if (first_cut_cell_found && !(first_point == cell_end_point)) {
                    for (auto &cl: msh.cells) {


                        if ((cl.user_data.location == element_location::ON_INTERFACE) &&
                            (cell_end_point == *cl.user_data.interface.begin()) && !(first_point == cell_end_point)) {
//                        std::cout<<"cell = "<<offset(msh_i, cl);
//                        std::cout<<"first_point =  "<<first_point<<std::endl;
//                        std::cout<<"*cl.user_data.interface.begin() =  "<<*cl.user_data.interface.begin()<<std::endl;
//                        std::cout<<"cell_end_point =  "<<cell_end_point<<std::endl;
                            auto integration_msh = cl.user_data.integration_msh;
                            auto global_cells_i = curve.get_global_cells_interface(msh, cl);
                            bool agglo_cl = cl.user_data.highlight &&
                                            ((cl.user_data.interface.size() > pow(2, n_int) * dd_degree + 1));
                            size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                            std::vector <size_t> index_inner_cls;
                            if (agglo_cl) {
                                for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                                    index_inner_cls.push_back(i_cl * (integration_msh.cells.size()) / amount_sub_cls);
                            }

                            size_t pos_index_bis = 0;


                            for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                                auto pts = points(integration_msh, integration_msh.cells[i_cell]);
                                size_t global_cl_i = global_cells_i[i_cell];
//                            std::cout<<"i_cell = "<<i_cell<<" , global global_cl_i = "<<global_cl_i<<std::endl;
//                            if( i_cell == integration_msh.cells.size()-1)
//                                cell_end_point =  pts[1] ;

                                for (T i = 0.0; i <= tot; i++) {
                                    T pos = 0.0 + i / tot;
                                    T val0 = curve.curvature_cont(pos, global_cl_i);
//                                    T pj = curve.jacobian_cont(pos , global_cl_i ) ;

//                                std::cout<<"pos = "<<pos<<" , pt = "<<curve(pos, pts , degree_curve)<<" , k = "<<val0<<std::endl;
//                                std::cout<<"distance_pts_para = "<<distance_pts_para<<std::endl;
                                    point<T, 2> curv_var = point_type(distance_pts_para, 0.0);
                                    if (agglo_cl && i_cell == index_inner_cls[pos_index_bis] && i == 0) {

//                                    auto pts_old = points(integration_msh,integration_msh.cells[i_cell-1]);
                                        size_t global_cl_i_prev;
                                        if (global_cl_i == 0)
                                            global_cl_i_prev = curve.ndof_dd - 1;
                                        else
                                            global_cl_i_prev = global_cl_i - 1;

                                        T valold = curve.curvature_cont(1.0, global_cl_i_prev);

                                        test_inner_cell->add_data(curv_var, val0);
                                        test_inner_cell->add_data(curv_var, valold);
                                        if (pos_index_bis + 1 < index_inner_cls.size())
                                            pos_index_bis++;
                                    }


                                    if ((i_cell == 0 && pos == 0.0) ||
                                        (i_cell == integration_msh.cells.size() - 1 && pos == 1.0)) {
                                        test_curv_var_para->add_data(curv_var, val0);
//                                    test_jacobian_cl->add_data(curv_var,pj );
//                                    std::cout<<"interface cell bdry -> YES "<<std::endl;
                                    }

                                    test_curvature_para->add_data(curv_var, val0);
//                                test_jacobian->add_data(curv_var,pj );

                                    T dist;
                                    if (pos == 1)
                                        dist = 0.0;
                                    else
                                        dist = (curve(pos + 1.0 / tot, pts, dd_degree) - curve(pos, pts, dd_degree)).norm();

                                    distance_pts_para += dist;


                                }

                            }
                            cell_end_point = *(cl.user_data.interface.end() - 1);
//                        std::cout<<"cell_end_point =  "<<cell_end_point<<std::endl;
                        }

                    }

                } else
                    break;

            }


        }
        postoutput_div_para.add_object(test_curvature_para);
        postoutput_div_para.add_object(test_curv_var_para);

//    postoutput_div_para.add_object(test_jacobian);
//    postoutput_div_para.add_object(test_jacobian_cl);
        postoutput_div_para.add_object(test_inner_cell);

        postoutput_div_para.write();


    }


    template<typename Mesh, typename Level_Set, typename T = typename Mesh::coordinate_type>
    void
    plot_curvature_normal_vs_curv_abscisse(Mesh &msh_i, Level_Set &ls_cell, size_t degree_curve, size_t n_int,
                                           size_t time_step, std::string &folder) {


        typedef typename Mesh::point_type point_type;

        Matrix<T, 2, 1> ref_vec;
        ref_vec(0) = 1.0;
        ref_vec(1) = 0.0;
//postprocess_output<double> postoutput_vec;
//auto vec_normal_grad_cont = std::make_shared< gnuplot_output_object_vec<double> >("normal_interface_continuos_grad_Stokes.dat");

        postprocess_output<double> postoutput_div2;

        std::string filename_curvature_k0 = folder + "curvature_2d_" + std::to_string(time_step) + ".dat";
        auto test_curv_var_divergence0 = std::make_shared < gnuplot_output_object < double > > (filename_curvature_k0);

        std::string filename_normal_k0 = folder + "normal_2d_" + std::to_string(time_step) + ".dat";
        auto test_var_normal0 = std::make_shared < gnuplot_output_object < double > > (filename_normal_k0);
// ----
        std::string filename_curv_var = folder + "cell_limit_curvature_" + std::to_string(time_step) + ".dat";
        auto test_curv_var_cell = std::make_shared < gnuplot_output_object < double > > (filename_curv_var);

        std::string filename_norm_var = folder + "cell_limit_normal_" + std::to_string(time_step) + ".dat";
        auto test_norm_var_cell = std::make_shared < gnuplot_output_object < double > > (filename_norm_var);
// ----
        std::string filename_curv_var_inner_cl =
                folder + "inner_cell_limit_curvature_" + std::to_string(time_step) + ".dat";
        auto test_inner_cell_curv = std::make_shared < gnuplot_output_object < double > > (filename_curv_var_inner_cl);

        std::string filename_norm_var_inner_cl = folder + "inner_cell_limit_normal_" + std::to_string(time_step) + ".dat";
        auto test_inner_cell_norm = std::make_shared < gnuplot_output_object < double > > (filename_norm_var_inner_cl);

        std::vector <point<T, 2>> interface_points_plot;
        std::vector <std::pair<T, T>> interface_normals;
        size_t counter_interface_pts = 0;




// PLOTTING OF THE CURVATURE and NORMALE IN A 2d FRAME
        bool first_cut_cell_found = FALSE;
        T distance_pts = 0.0;
        point<T, 2> first_point;
        point<T, 2> cell_end_point;

// --------- CHECKING CURVATURE DISC LS ----------
        for (auto &cl: msh_i.cells) {

            if (cl.user_data.location == element_location::ON_INTERFACE) {
                ls_cell.cell_assignment(cl);
                if (!first_cut_cell_found) {
                    bool agglo_cl =
                            cl.user_data.highlight && ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                    size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                    std::vector <size_t> index_inner_cls;
                    if (agglo_cl) {
                        for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                            index_inner_cls.push_back(i_cl * (cl.user_data.interface.size() - 1) / amount_sub_cls);
                    }
//                if( agglo_cl && amount_sub_cls == 2 )
//                    index_inner_cls.push_back( (cl.user_data.interface.size() - 1)/amount_sub_cls );
//                if( agglo_cl && amount_sub_cls == 3 ){
//                    index_inner_cls.push_back( (cl.user_data.interface.size() - 1)/amount_sub_cls );
//                    index_inner_cls.push_back( 2.0*(cl.user_data.interface.size() - 1)/amount_sub_cls );
//                }
                    size_t pos_index = 0;
                    size_t pos_index_bis = 0;

                    for (auto interface_point = cl.user_data.interface.begin();
                         interface_point < cl.user_data.interface.end(); interface_point++) {
                        T disc0 = ls_cell.divergence(*interface_point);
                        auto norm0 = ls_cell.normal(*interface_point);
                        T angle_norm = acos(norm0.dot(ref_vec));
                        if (norm0[1] < 0)
                            angle_norm = 2.0 * M_PI - angle_norm;
//                    if(norm0[0] > 0 && norm0[1] < 0  )
//                        angle_norm = 4.0*M_PI - angle_norm ;
////                        angle_norm += 3.0/2.0*M_PI ;
//                    if(norm0[0] < 0 && norm0[1] < 0  )
//                        angle_norm = 4.0*M_PI - angle_norm ;
////                        angle_norm += 1.0/2.0*M_PI ;

                        point<T, 2> curv_var = point_type(distance_pts, 0.0);
                        if (interface_point == cl.user_data.interface.begin() ||
                            interface_point == (cl.user_data.interface.end() - 1)) {
                            test_curv_var_cell->add_data(curv_var, disc0);
                            test_norm_var_cell->add_data(curv_var, angle_norm);

                        }
                        if (agglo_cl && pos_index == index_inner_cls[pos_index_bis]) {
                            auto offset_cells = pt_in_subcell_skeleton(ls_cell.level_set.msh, *interface_point, cl);
                            assert(offset_cells.size() == 2);
                            auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                            auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                            T val_skeleton0 = ls_cell.divergence(*interface_point, subcl0);
                            T val_skeleton1 = ls_cell.divergence(*interface_point, subcl1);

                            auto norm_sk0 = ls_cell.normal(*interface_point, subcl0);
                            auto norm_sk1 = ls_cell.normal(*interface_point, subcl1);
                            T angle_norm0 = acos(norm_sk0.dot(ref_vec));
                            T angle_norm1 = acos(norm_sk1.dot(ref_vec));
//                         if(norm_sk0[0] > 0 && norm_sk0[1] < 0  )
//                             angle_norm0 += 3.0/2.0*M_PI ;
//                         if(norm_sk0[0] < 0 && norm_sk0[1] < 0  )
//                             angle_norm0 += 1.0/2.0*M_PI ;

                            if (norm_sk0[1] < 0)
                                angle_norm0 = 2.0 * M_PI - angle_norm0;

                            if (norm_sk1[1] < 0)
                                angle_norm1 = 2.0 * M_PI - angle_norm1;

//                         if(norm_sk1[0] > 0 && norm_sk1[1] < 0  )
//                             angle_norm1 += 3.0/2.0*M_PI ;
//                         if(norm_sk1[0] < 0 && norm_sk1[1] < 0  )
//                             angle_norm1 += 1.0/2.0*M_PI ;

                            test_inner_cell_curv->add_data(curv_var, val_skeleton0);
                            test_inner_cell_curv->add_data(curv_var, val_skeleton1);

                            test_inner_cell_norm->add_data(curv_var, angle_norm0);
                            test_inner_cell_norm->add_data(curv_var, angle_norm1);

                            if (pos_index_bis + 1 < index_inner_cls.size())
                                pos_index_bis++;
                        }

                        test_curv_var_divergence0->add_data(curv_var, disc0);
                        test_var_normal0->add_data(curv_var, angle_norm);
                        if (*interface_point == *(cl.user_data.interface.end() - 1))
                            distance_pts += 0.0;
                        else
                            distance_pts += (*(interface_point + 1) - *interface_point).to_vector().norm();
// In the case in which *interface_point == *(cl.user_data.interface.end() -1) I'm in the skeleton and it means that the next point it will be in the same abscisse.
                        pos_index++;
                    }
                    first_cut_cell_found = TRUE;
                    first_point = *cl.user_data.interface.begin();
                    cell_end_point = *(cl.user_data.interface.end() - 1);
                } else if (first_cut_cell_found && !(first_point == cell_end_point)) {
                    for (auto &cl: msh_i.cells) {
                        if ((cl.user_data.location == element_location::ON_INTERFACE) &&
                            (cell_end_point == *cl.user_data.interface.begin()) && !(first_point == cell_end_point)) {
                            ls_cell.cell_assignment(cl);

                            bool agglo_cl = cl.user_data.highlight &&
                                            ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                            size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                            std::vector <size_t> index_inner_cls;
                            if (agglo_cl) {
                                for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                                    index_inner_cls.push_back(i_cl * (cl.user_data.interface.size() - 1) / amount_sub_cls);
                            }
                            size_t pos_index = 0;
                            size_t pos_index_bis = 0;

                            for (auto interface_point = cl.user_data.interface.begin();
                                 interface_point < cl.user_data.interface.end(); interface_point++) {

                                T curv0 = ls_cell.divergence(*interface_point);
                                auto norm0 = ls_cell.normal(*interface_point);
                                T angle_norm = acos(norm0.dot(ref_vec));
//                            if(norm0[0] > 0 && norm0[1] < 0  )
//                                angle_norm += 3.0/2.0*M_PI ;
//                            if(norm0[0] < 0 && norm0[1] < 0  )
//                                angle_norm += 1.0/2.0*M_PI ;

                                if (norm0[1] < 0)
                                    angle_norm = 2.0 * M_PI - angle_norm;

                                point<T, 2> curv_var = point_type(distance_pts, 0.0);
                                if (interface_point == cl.user_data.interface.begin() ||
                                    interface_point == (cl.user_data.interface.end() - 1)) {
                                    test_curv_var_cell->add_data(curv_var, curv0);
                                    test_norm_var_cell->add_data(curv_var, angle_norm);

                                }
                                test_curv_var_divergence0->add_data(curv_var, curv0);
                                test_var_normal0->add_data(curv_var, angle_norm);

                                if (agglo_cl && pos_index == index_inner_cls[pos_index_bis]) {
                                    auto offset_cells = pt_in_subcell_skeleton(ls_cell.level_set.msh, *interface_point, cl);
                                    assert(offset_cells.size() == 2);
                                    auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                                    auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                                    T val_skeleton0 = ls_cell.divergence(*interface_point, subcl0);
                                    T val_skeleton1 = ls_cell.divergence(*interface_point, subcl1);

                                    auto norm_sk0 = ls_cell.normal(*interface_point, subcl0);
                                    auto norm_sk1 = ls_cell.normal(*interface_point, subcl1);
                                    T angle_norm0 = acos(norm_sk0.dot(ref_vec));
                                    T angle_norm1 = acos(norm_sk1.dot(ref_vec));
//                                if(norm_sk0[0] > 0 && norm_sk0[1] < 0  )
//                                    angle_norm0 += 3.0/2.0*M_PI ;
//                                if(norm_sk0[0] < 0 && norm_sk0[1] < 0  )
//                                    angle_norm0 += 1.0/2.0*M_PI ;
//
//                                if(norm_sk1[0] > 0 && norm_sk1[1] < 0  )
//                                    angle_norm1 += 3.0/2.0*M_PI ;
//                                if(norm_sk1[0] < 0 && norm_sk1[1] < 0  )
//                                    angle_norm1 += 1.0/2.0*M_PI ;

                                    if (norm_sk0[1] < 0)
                                        angle_norm0 = 2.0 * M_PI - angle_norm0;

                                    if (norm_sk1[1] < 0)
                                        angle_norm1 = 2.0 * M_PI - angle_norm1;


                                    test_inner_cell_curv->add_data(curv_var, val_skeleton0);
                                    test_inner_cell_curv->add_data(curv_var, val_skeleton1);

                                    test_inner_cell_norm->add_data(curv_var, angle_norm0);
                                    test_inner_cell_norm->add_data(curv_var, angle_norm1);

                                    if (pos_index_bis + 1 < index_inner_cls.size())
                                        pos_index_bis++;
                                }


                                if (*interface_point == *(cl.user_data.interface.end() - 1))
                                    distance_pts += 0.0;
                                else
                                    distance_pts += (*(interface_point + 1) - *interface_point).to_vector().norm();
                                pos_index++;
                            }
                            cell_end_point = *(cl.user_data.interface.end() - 1);
                        }

                    }

                } else
                    break;

            }


        }
        postoutput_div2.add_object(test_curv_var_divergence0);
//    postoutput_div2.add_object(test_curv_var_cell);
//    postoutput_div2.add_object(test_inner_cell_curv);

        postoutput_div2.add_object(test_var_normal0);
//    postoutput_div2.add_object(test_norm_var_cell);
//    postoutput_div2.add_object(test_inner_cell_norm);

        postoutput_div2.write();


    }


    template<typename Mesh, typename Curve, typename T>
    void
    plotting_para_curvature_cont_time_fast(Mesh &msh_i, Curve &curve, size_t degree_curve, size_t degree_FEM, T radius,
                                           size_t time_step, size_t n_int) {

        typedef typename Mesh::point_type point_type;
//    T tot = degree_curve ; // 100
        T tot_error = 10;


        postprocess_output<double> postoutput_div_para;

        std::string filename_curvature_para = "k_curvature_para_cont" + std::to_string(time_step) + ".dat";
        auto test_curvature_para = std::make_shared < gnuplot_output_object < double > > (filename_curvature_para);

        std::string filename_curv_var_para = "cell_limit_curv_var_para_cont" + std::to_string(time_step) + ".dat";
        auto test_curv_var_para = std::make_shared < gnuplot_output_object < double > > (filename_curv_var_para);


        std::string filename_curv_var_inner_cl = "inner_cell_limit_para_cont" + std::to_string(time_step) + ".dat";
        auto test_inner_cell = std::make_shared < gnuplot_output_object < double > > (filename_curv_var_inner_cl);


        T distance_pts_para = 0.0;
        bool first_cut_cell_found = FALSE;
        point<T, 2> first_point;
        point<T, 2> cell_end_point;
        T tot = 10; // degree_curve; //*pow(2,n_int) ; // degree_curve - 1 ; //degree_curve - 2 ; // 10
        for (auto &cl: msh_i.cells) {

            if (cl.user_data.location == element_location::ON_INTERFACE) {
//            std::cout<<"cell = "<<offset(msh_i, cl);
                auto global_cells_i = curve.get_global_cells_interface(msh_i, cl);
                auto integration_msh = cl.user_data.integration_msh;
                if (!first_cut_cell_found) {
                    bool agglo_cl =
                            cl.user_data.highlight && ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                    size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                    std::vector <size_t> index_inner_cls;
                    if (agglo_cl) {
                        for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                            index_inner_cls.push_back(i_cl * (integration_msh.cells.size()) / amount_sub_cls);
                    }

                    size_t pos_index_bis = 0;

                    for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                        size_t global_cl_i = global_cells_i[i_cell];
//                    std::cout<<"i_cell = "<<i_cell<<" , global global_cl_i = "<<global_cl_i<<std::endl;
                        auto pts = points(integration_msh, integration_msh.cells[i_cell]);

//                    if( i_cell == 0)
//                        first_point = pts[0] ;
//                    if( i_cell == integration_msh.cells.size()-1)
//                        cell_end_point =  pts[1] ;

                        for (T i = 0.0; i <= tot; i++) {
                            T pos = 0.0 + i / tot;

                            T val0 = curve.curvature_cont(pos, global_cl_i);
//                        std::cout<<"pos = "<<pos<<" , pt = "<<curve(pos, pts , degree_curve)<<" , k = "<<val0<<std::endl;
                            T pj = curve.jacobian_cont(pos, global_cl_i);

//                        std::cout<<"distance_pts_para = "<<distance_pts_para<<std::endl;
                            point<T, 2> curv_var = point_type(distance_pts_para, 0.0);

                            if (agglo_cl && i_cell == index_inner_cls[pos_index_bis] && i == 0) {

//                            auto pts_old = points(integration_msh,integration_msh.cells[i_cell-1]);
                                T valold = curve.curvature_cont(1.0, global_cl_i - 1);

                                test_inner_cell->add_data(curv_var, val0);
                                test_inner_cell->add_data(curv_var, valold);
                                if (pos_index_bis + 1 < index_inner_cls.size())
                                    pos_index_bis++;
                            }

                            if ((i_cell == 0 && pos == 0.0) || (i_cell == integration_msh.cells.size() - 1 && pos == 1.0)) {
                                test_curv_var_para->add_data(curv_var, val0);
//                            test_jacobian_cl->add_data(curv_var,pj );
//                            std::cout<<"interface cell bdry -> YES "<<std::endl;
                            }

                            test_curvature_para->add_data(curv_var, val0);
//                        test_jacobian->add_data(curv_var,pj );
                            T dist;
                            if (pos == 1)
                                dist = 0.0;
                            else
                                dist = (curve(pos + 1.0 / tot, pts, degree_curve) - curve(pos, pts, degree_curve)).norm();

                            distance_pts_para += dist;


                        }

                    }
                    first_cut_cell_found = TRUE;
                    first_point = *cl.user_data.interface.begin();
                    cell_end_point = *(cl.user_data.interface.end() - 1);
                } else if (first_cut_cell_found && !(first_point == cell_end_point)) {
                    for (auto &cl: msh_i.cells) {


                        if ((cl.user_data.location == element_location::ON_INTERFACE) &&
                            (cell_end_point == *cl.user_data.interface.begin()) && !(first_point == cell_end_point)) {
//                        std::cout<<"cell = "<<offset(msh_i, cl);
//                        std::cout<<"first_point =  "<<first_point<<std::endl;
//                        std::cout<<"*cl.user_data.interface.begin() =  "<<*cl.user_data.interface.begin()<<std::endl;
//                        std::cout<<"cell_end_point =  "<<cell_end_point<<std::endl;
                            auto integration_msh = cl.user_data.integration_msh;
                            auto global_cells_i = curve.get_global_cells_interface(msh_i, cl);

                            bool agglo_cl = cl.user_data.highlight &&
                                            ((cl.user_data.interface.size() > pow(2, n_int) * degree_curve + 1));
                            size_t amount_sub_cls = cl.user_data.offset_subcells.size();
                            std::vector <size_t> index_inner_cls;
                            if (agglo_cl) {
                                for (size_t i_cl = 1; i_cl < amount_sub_cls; i_cl++)
                                    index_inner_cls.push_back(i_cl * (integration_msh.cells.size()) / amount_sub_cls);
                            }

                            size_t pos_index_bis = 0;

                            for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                                auto pts = points(integration_msh, integration_msh.cells[i_cell]);
                                size_t global_cl_i = global_cells_i[i_cell];
//                            std::cout<<"i_cell = "<<i_cell<<" , global global_cl_i = "<<global_cl_i<<std::endl;
//                            if( i_cell == integration_msh.cells.size()-1)
//                                cell_end_point =  pts[1] ;

                                for (T i = 0.0; i <= tot; i++) {
                                    T pos = 0.0 + i / tot;
                                    T val0 = curve.curvature_cont(pos, global_cl_i);
                                    T pj = curve.jacobian_cont(pos, global_cl_i);

//                                std::cout<<"pos = "<<pos<<" , pt = "<<curve(pos, pts , degree_curve)<<" , k = "<<val0<<std::endl;
//                                std::cout<<"distance_pts_para = "<<distance_pts_para<<std::endl;
                                    point<T, 2> curv_var = point_type(distance_pts_para, 0.0);

                                    if (agglo_cl && i_cell == index_inner_cls[pos_index_bis] && i == 0) {

//                                    auto pts_old = points(integration_msh,integration_msh.cells[i_cell-1]);
                                        T valold = curve.curvature_cont(1.0, global_cl_i - 1);

                                        test_inner_cell->add_data(curv_var, val0);
                                        test_inner_cell->add_data(curv_var, valold);
                                        if (pos_index_bis + 1 < index_inner_cls.size())
                                            pos_index_bis++;
                                    }

                                    if ((i_cell == 0 && pos == 0.0) ||
                                        (i_cell == integration_msh.cells.size() - 1 && pos == 1.0)) {
                                        test_curv_var_para->add_data(curv_var, val0);
//                                    test_jacobian_cl->add_data(curv_var,pj );
//                                    std::cout<<"interface cell bdry -> YES "<<std::endl;
                                    }

                                    test_curvature_para->add_data(curv_var, val0);
//                                test_jacobian->add_data(curv_var,pj );

                                    T dist;
                                    if (pos == 1)
                                        dist = 0.0;
                                    else
                                        dist = (curve(pos + 1.0 / tot, pts, degree_curve) -
                                                curve(pos, pts, degree_curve)).norm();

                                    distance_pts_para += dist;


                                }

                            }
                            cell_end_point = *(cl.user_data.interface.end() - 1);
//                        std::cout<<"cell_end_point =  "<<cell_end_point<<std::endl;
                        }

                    }

                } else
                    break;

            }


        }
        postoutput_div_para.add_object(test_curvature_para);
        postoutput_div_para.add_object(test_curv_var_para);

//    postoutput_div_para.add_object(test_jacobian);
//    postoutput_div_para.add_object(test_jacobian_cl);
        postoutput_div_para.add_object(test_inner_cell);

        postoutput_div_para.write();


    }



    template<typename T>
    void
    plotting_in_time_complete(const std::vector <T> &time_vec, const std::vector <T> &area_time,
                              const std::vector <T> &l1_err_u_n_time, const std::vector <T> &linf_err_u_n_time,
                              const std::vector <T> &max_val_u_n_time, const std::vector <T> &l1_err_curvature_time,
                              const std::vector <T> &linf_err_curvature_time, T dt,
                              const std::vector <std::pair<T, T>> &min_max_vec, const std::vector <T> &flux_interface_time,
                              const std::vector <std::pair<T, T>> &rise_velocity_time,
                              const std::vector <std::pair<T, T>> &centre_mass_err_time,
                              const std::vector <T> &perimeter_time, const std::vector <T> &circularity_time,
                              T circularity_ref, T perimetre_ref, T area_ref, T radius) {
        std::cout << "IT does not work, I have some vector small without first iteration!!" << std::endl;
        postprocess_output <T> postoutput;

        auto testref0 = std::make_shared < gnuplot_output_object_time < T > > ("area_ref_time.dat");
        auto testref1 = std::make_shared < gnuplot_output_object_time < T > > ("perimeter_ref_time.dat");
        auto testref2 = std::make_shared < gnuplot_output_object_time < T > > ("circularity_ref_time.dat");

        T area_analytic = M_PI * radius * radius;
        T perimeter_analytic = 2.0 * M_PI * radius;
        auto testanal0 = std::make_shared < gnuplot_output_object_time < T > > ("area_anal_time.dat");
        auto testanal1 = std::make_shared < gnuplot_output_object_time < T > > ("perimeter_anal_time.dat");


        auto test0 = std::make_shared < gnuplot_output_object_time < T > > ("area_time.dat");
        auto test1 = std::make_shared < gnuplot_output_object_time < T > > ("l1_err_u_n_time.dat");
        auto test2 = std::make_shared < gnuplot_output_object_time < T > > ("linf_err_u_n_time.dat");
        auto test1_counter = std::make_shared < gnuplot_output_object_time < T > > ("l1_err_u_n_time_counter.dat");
        auto test2_counter = std::make_shared < gnuplot_output_object_time < T > > ("linf_err_u_n_time_counter.dat");

        auto test3 = std::make_shared < gnuplot_output_object_time < T > > ("max_val_u_n_time.dat");
        auto test4 = std::make_shared < gnuplot_output_object_time < T > > ("l1_err_curvature_time.dat");
        auto test5 = std::make_shared < gnuplot_output_object_time < T > > ("linf_err_curvature_time.dat");
        auto test4_counter = std::make_shared < gnuplot_output_object_time < T > > ("l1_err_curvature_time_counter.dat");
        auto test5_counter = std::make_shared < gnuplot_output_object_time < T > > ("linf_err_curvature_time_counter.dat");


        auto test0b = std::make_shared < gnuplot_output_object_time < T > > ("area_time_normalised.dat");

        auto test1b = std::make_shared < gnuplot_output_object_time < T > > ("l1_err_u_n_time_normalised.dat");
        auto test2b = std::make_shared < gnuplot_output_object_time < T > > ("linf_err_u_n_time_normalised.dat");
        auto test3b = std::make_shared < gnuplot_output_object_time < T > > ("max_val_u_n_time_normalised.dat");
        auto test4b = std::make_shared < gnuplot_output_object_time < T > > ("l1_err_curvature_time_normalised.dat");
        auto test5b = std::make_shared < gnuplot_output_object_time < T > > ("linf_err_curvature_time_normalised.dat");
//auto test4c  = std::make_shared< gnuplot_output_object_time<T> >("l1_l1_err_curvature_err_time.dat");

        auto test_dt = std::make_shared < gnuplot_output_object_time < T > > ("dt_M.dat");

        auto test0c = std::make_shared < gnuplot_output_object_time < T > > ("area_time_err.dat");
        auto test0c_counter = std::make_shared < gnuplot_output_object_time < T > > ("area_time_err_counter.dat");

        auto testm0 = std::make_shared < gnuplot_output_object_time < T > > ("min_time.dat");
        auto testm1 = std::make_shared < gnuplot_output_object_time < T > > ("min_normalised_time.dat");
        auto testM0 = std::make_shared < gnuplot_output_object_time < T > > ("max_time.dat");
        auto testM1 = std::make_shared < gnuplot_output_object_time < T > > ("max_normalised_time.dat");
        auto testflux = std::make_shared < gnuplot_output_object_time < T > > ("flux_interface_time.dat");
        auto testvelx = std::make_shared < gnuplot_output_object_time < T > > ("rise_velocity_err_x_time.dat");
        auto testvely = std::make_shared < gnuplot_output_object_time < T > > ("rise_velocity_err_y_time.dat");
        auto testvel = std::make_shared < gnuplot_output_object_time < T > > ("rise_velocity_err_time.dat");
        auto testvel_counter = std::make_shared < gnuplot_output_object_time < T > > ("rise_velocity_err_time_counter.dat");
        auto testcomx = std::make_shared < gnuplot_output_object_time < T > > ("centre_mass_err_x_time.dat");
        auto testcomy = std::make_shared < gnuplot_output_object_time < T > > ("centre_mass_err_y_time.dat");
        auto testcom = std::make_shared < gnuplot_output_object_time < T > > ("centre_mass_err_time.dat");


        auto testper0 = std::make_shared < gnuplot_output_object_time < T > > ("perimeter_time.dat");
        auto testper1 = std::make_shared < gnuplot_output_object_time < T > > ("perimeter_normalised_time.dat");
        auto testper2 = std::make_shared < gnuplot_output_object_time < T > > ("perimeter_err_time.dat");
        auto testper2_counter = std::make_shared < gnuplot_output_object_time < T > > ("perimeter_err_time_counter.dat");

        auto testcirc = std::make_shared < gnuplot_output_object_time < T > > ("circularity_time.dat");
        auto testcirc1 = std::make_shared < gnuplot_output_object_time < T > > ("circularity_error_time.dat");
        auto testcirc1_counter =
                std::make_shared < gnuplot_output_object_time < T > > ("circularity_error_time_counter.dat");

        size_t tot = l1_err_u_n_time.size();

        testm0->add_data(time_vec[0], min_max_vec[0].first);
        testm1->add_data(time_vec[0], 1.0);
        testM0->add_data(time_vec[0], min_max_vec[0].second);
        testM1->add_data(time_vec[0], 1.0);

        testcomx->add_data(time_vec[0], 0.0);
        testcomy->add_data(time_vec[0], 0.0);
        testcom->add_data(time_vec[0], 0.0);


        testper0->add_data(time_vec[0], perimeter_time[0]);
        testper1->add_data(time_vec[0], 1.0);
        testper2->add_data(time_vec[0], std::abs(perimeter_time[0] - perimetre_ref) / perimetre_ref);
        testper2_counter->add_data(0, std::abs(perimeter_time[0] - perimetre_ref) / perimetre_ref);

        testcirc->add_data(time_vec[0], circularity_time[0]);
        testcirc1->add_data(time_vec[0], std::abs(circularity_time[0] - circularity_ref) / circularity_ref);
        testcirc1_counter->add_data(0, std::abs(circularity_time[0] - circularity_ref) / circularity_ref);

        test0c->add_data(time_vec[0], 0.0);
        test0c_counter->add_data(0, 0.0);
//test4c->add_data(time_vec[0] , 0.0 );


        test0->add_data(time_vec[0], area_time[0]);
// ADD 8/01/21
        test1->add_data(time_vec[0], l1_err_u_n_time[0]);
        test2->add_data(time_vec[0], linf_err_u_n_time[0]);
        test1_counter->add_data(0, l1_err_u_n_time[0]);
        test2_counter->add_data(0, linf_err_u_n_time[0]);
        test3->add_data(time_vec[0], max_val_u_n_time[0]);
// UP TO HERE
        test4->add_data(time_vec[0], l1_err_curvature_time[0]);
        test5->add_data(time_vec[0], linf_err_curvature_time[0]);
        test4_counter->add_data(0, l1_err_curvature_time[0]);
        test5_counter->add_data(0, linf_err_curvature_time[0]);

        test0b->add_data(time_vec[0], area_time[0] / area_time[0]);
// ADD 8/01/21
        test1b->add_data(time_vec[0], l1_err_u_n_time[0] / l1_err_u_n_time[0]);
        test2b->add_data(time_vec[0], linf_err_u_n_time[0] / linf_err_u_n_time[0]);
        test3b->add_data(time_vec[0], max_val_u_n_time[0] / max_val_u_n_time[0]);
// UP TO HERE
        test4b->add_data(time_vec[0], l1_err_curvature_time[0] / l1_err_curvature_time[0]);
        test5b->add_data(time_vec[0], linf_err_curvature_time[0] / linf_err_curvature_time[0]);

        testref0->add_data(time_vec[0], area_ref);
        testref1->add_data(time_vec[0], perimetre_ref);
        testref2->add_data(time_vec[0], circularity_ref);

        testanal0->add_data(time_vec[0], area_analytic);
        testanal1->add_data(time_vec[0], perimeter_analytic);


        testflux->add_data(time_vec[0], flux_interface_time[0]);
        testvelx->add_data(time_vec[0], std::abs(rise_velocity_time[0].first));
        testvely->add_data(time_vec[0], std::abs(rise_velocity_time[0].second));
        testvel->add_data(time_vec[0], std::abs(rise_velocity_time[0].first) + std::abs(rise_velocity_time[0].second));
        testvel_counter->add_data(0, std::abs(rise_velocity_time[0].first) + std::abs(rise_velocity_time[0].second));

        for (size_t i = 0; i < tot; i++) {
            test0->add_data(time_vec[i + 1], area_time[i + 1]);
            test1->add_data(time_vec[i + 1], l1_err_u_n_time[i + 1]);
            test2->add_data(time_vec[i + 1], linf_err_u_n_time[i + 1]);
            test1_counter->add_data(i + 1, l1_err_u_n_time[i + 1]);
            test2_counter->add_data(i + 1, linf_err_u_n_time[i + 1]);
            test3->add_data(time_vec[i + 1], max_val_u_n_time[i + 1]);
//        test1->add_data(time_vec[i+1] ,l1_err_u_n_time[i] );
//        test2->add_data(time_vec[i+1] ,linf_err_u_n_time[i] );
//        test1_counter->add_data(i+1 ,l1_err_u_n_time[i] );
//        test2_counter->add_data(i+1 ,linf_err_u_n_time[i] );
//        test3->add_data(time_vec[i+1] ,max_val_u_n_time[i] );
            test4->add_data(time_vec[i + 1], l1_err_curvature_time[i + 1]);
            test5->add_data(time_vec[i + 1], linf_err_curvature_time[i + 1]);
            test4_counter->add_data(i + 1, l1_err_curvature_time[i + 1]);
            test5_counter->add_data(i + 1, linf_err_curvature_time[i + 1]);

            test0b->add_data(time_vec[i + 1], area_time[i + 1] / area_time[0]);

            test1b->add_data(time_vec[i + 1], l1_err_u_n_time[i + 1] / l1_err_u_n_time[0]);
            test2b->add_data(time_vec[i + 1], linf_err_u_n_time[i + 1] / linf_err_u_n_time[0]);
            test3b->add_data(time_vec[i + 1], max_val_u_n_time[i + 1] / max_val_u_n_time[0]);
//        test1b->add_data(time_vec[i+1] ,l1_err_u_n_time[i]/l1_err_u_n_time[0] );
//        test2b->add_data(time_vec[i+1] ,linf_err_u_n_time[i]/linf_err_u_n_time[0] );
//        test3b->add_data(time_vec[i+1] ,max_val_u_n_time[i]/max_val_u_n_time[0]  );
            test4b->add_data(time_vec[i + 1], l1_err_curvature_time[i + 1] / l1_err_curvature_time[0]);
            test5b->add_data(time_vec[i + 1], linf_err_curvature_time[i + 1] / linf_err_curvature_time[0]);


            testcomx->add_data(time_vec[i + 1],
                               std::abs(centre_mass_err_time[i + 1].first - centre_mass_err_time[0].first) /
                               centre_mass_err_time[0].first);
            testcomy->add_data(time_vec[i + 1],
                               std::abs(centre_mass_err_time[i + 1].second - centre_mass_err_time[0].second) /
                               centre_mass_err_time[0].second);
            testcom->add_data(time_vec[i + 1], std::abs(centre_mass_err_time[i + 1].first - centre_mass_err_time[0].first) /
                                               centre_mass_err_time[0].first + std::abs(
                    centre_mass_err_time[i + 1].second - centre_mass_err_time[0].second) / centre_mass_err_time[0].second);


            testm0->add_data(time_vec[i + 1], min_max_vec[i + 1].first);
            testm1->add_data(time_vec[i + 1], min_max_vec[i + 1].first / min_max_vec[0].first);
            testM0->add_data(time_vec[i + 1], min_max_vec[i + 1].second);
            testM1->add_data(time_vec[i + 1], min_max_vec[i + 1].second / min_max_vec[0].second);


            testper0->add_data(time_vec[i + 1], perimeter_time[i + 1]);
            testper1->add_data(time_vec[i + 1], perimeter_time[i + 1] / perimeter_time[0]);
            testper2->add_data(time_vec[i + 1], std::abs(perimeter_time[i + 1] - perimetre_ref) / perimetre_ref);
            testper2_counter->add_data(i + 1, std::abs(perimeter_time[i + 1] - perimetre_ref) / perimetre_ref);

            testcirc->add_data(time_vec[i + 1], circularity_time[i + 1]);
            testcirc1->add_data(time_vec[i + 1], std::abs(circularity_time[i + 1] - circularity_ref) / circularity_ref);
            testcirc1_counter->add_data(i + 1, std::abs(circularity_time[i + 1] - circularity_ref) / circularity_ref);

            test0c->add_data(time_vec[i + 1], std::abs(area_time[i + 1] - area_time[0]) / area_time[0]);
            test0c_counter->add_data(i + 1, std::abs(area_time[i + 1] - area_time[0]) / area_time[0]);
//test4c->add_data(time_vec[i+1] , std::abs(l1_err_curvature_time[i+1] - l1_err_curvature_time[0] )/l1_err_curvature_time[0]);

            testflux->add_data(time_vec[i + 1], flux_interface_time[i + 1]);
            testvelx->add_data(time_vec[i + 1], std::abs(rise_velocity_time[i + 1].first));
            testvely->add_data(time_vec[i + 1], std::abs(rise_velocity_time[i + 1].second));
            testvel->add_data(time_vec[i + 1],
                              std::abs(rise_velocity_time[i + 1].first) + std::abs(rise_velocity_time[i + 1].second));
            testvel_counter->add_data(i + 1, std::abs(rise_velocity_time[i + 1].first) +
                                             std::abs(rise_velocity_time[i + 1].second));

            testref0->add_data(time_vec[i + 1], area_ref);
            testref1->add_data(time_vec[i + 1], perimetre_ref);
            testref2->add_data(time_vec[i + 1], circularity_ref);

            testanal0->add_data(time_vec[i + 1], area_analytic);
            testanal1->add_data(time_vec[i + 1], perimeter_analytic);

//std::cout<<"time_vec[i] = "<<time_vec[i]<<", time_vec[i+1] = "<<time_vec[i+1]<<", dt = "<<dt<<", error = "<<std::abs( std::abs( time_vec[i+1] - time_vec[i] ) - dt )<<std::endl;
            if (std::abs(std::abs(time_vec[i + 1] - time_vec[i]) - dt) > 1e-10) {
                test_dt->add_data(time_vec[i], 0.0);
                test_dt->add_data(time_vec[i + 1], 0.0);
            }
        }


        postoutput.add_object(test0);
        postoutput.add_object(test1);
        postoutput.add_object(test2);
        postoutput.add_object(test1_counter);
        postoutput.add_object(test2_counter);


        postoutput.add_object(test3);
        postoutput.add_object(test4);
        postoutput.add_object(test5);
        postoutput.add_object(test4_counter);
        postoutput.add_object(test5_counter);

        postoutput.add_object(test0b);
        postoutput.add_object(test1b);
        postoutput.add_object(test2b);
        postoutput.add_object(test3b);
        postoutput.add_object(test4b);
        postoutput.add_object(test5b);

        postoutput.add_object(testcomx);
        postoutput.add_object(testcomy);
        postoutput.add_object(testcom);

        postoutput.add_object(testm0);
        postoutput.add_object(testm1);
        postoutput.add_object(testM0);
        postoutput.add_object(testM1);

        postoutput.add_object(testper0);
        postoutput.add_object(testper1);
        postoutput.add_object(testper2);
        postoutput.add_object(testper2_counter);


        postoutput.add_object(testcirc);
        postoutput.add_object(testcirc1);
        postoutput.add_object(testcirc1_counter);

        postoutput.add_object(test0c);
        postoutput.add_object(test0c_counter);
//postoutput.add_object(test4c);

        postoutput.add_object(testflux);
        postoutput.add_object(testvelx);
        postoutput.add_object(testvely);
        postoutput.add_object(testvel);
        postoutput.add_object(testvel_counter);


        postoutput.add_object(testref0);
        postoutput.add_object(testref1);
        postoutput.add_object(testref2);

        postoutput.add_object(testanal0);
        postoutput.add_object(testanal1);


        postoutput.add_object(test_dt);

        postoutput.write();

    }

    template<typename T>
    void
    plotting_in_time_new(const std::vector <T> &time_vec, const std::vector <T> &area_time,
                         const std::vector <T> &l1_err_u_n_time, const std::vector <T> &linf_err_u_n_time,
                         const std::vector <T> &max_val_u_n_time, const std::vector <T> &l1_err_curvature_time,
                         const std::vector <T> &linf_err_curvature_time, T dt,
                         const std::vector <std::pair<T, T>> &min_max_vec, const std::vector <T> &flux_interface_time,
                         const std::vector <std::pair<T, T>> &rise_velocity_time,
                         const std::vector <std::pair<T, T>> &centre_mass_err_time, const std::vector <T> &perimeter_time,
                         const std::vector <T> &circularity_time, T circularity_ref, T perimetre_ref, T area_ref, T radius,
                         const std::vector <T> &L1_err_u_n_time, const std::vector <T> &l1_err_u_n_time_para,
                         const std::vector <T> &linf_err_u_n_time_para, const std::vector <T> &L1_err_u_n_time_para,
                         const std::vector <T> &max_val_u_n_time_para) {

        postprocess_output <T> postoutput;

//    auto testref0  = std::make_shared< gnuplot_output_object_time<T> >("area_ref_time.dat");
//    auto testref1  = std::make_shared< gnuplot_output_object_time<T> >("perimeter_ref_time.dat");
//    auto testref2  = std::make_shared< gnuplot_output_object_time<T> >("circularity_ref_time.dat");

        T area_analytic = M_PI * radius * radius;
        T perimeter_analytic = 2.0 * M_PI * radius;
//    auto testanal0  = std::make_shared< gnuplot_output_object_time<T> >("area_anal_time.dat");
//    auto testanal1  = std::make_shared< gnuplot_output_object_time<T> >("perimeter_anal_time.dat");


        auto test0 = std::make_shared < gnuplot_output_object_time < T > > ("area_time.dat");
        auto test1 = std::make_shared < gnuplot_output_object_time < T > > ("lpiccolo1_err_u_n_time.dat");
        auto test2 = std::make_shared < gnuplot_output_object_time < T > > ("linf_err_u_n_time.dat");

        auto test_der_int = std::make_shared < gnuplot_output_object_time < T > > ("linf_der_interface_time.dat");

//    auto test1_counter  = std::make_shared< gnuplot_output_object_time<T> >("lpiccolo1_err_u_n_time_counter.dat");
//    auto test2_counter  = std::make_shared< gnuplot_output_object_time<T> >("linf_err_u_n_time_counter.dat");

//    auto test3  = std::make_shared< gnuplot_output_object_time<T> >("max_val_u_n_time.dat");
        auto test4 = std::make_shared < gnuplot_output_object_time < T > > ("l1_err_curvature_time.dat");
        auto test5 = std::make_shared < gnuplot_output_object_time < T > > ("linf_err_curvature_time.dat");
//    auto test4_counter  = std::make_shared< gnuplot_output_object_time<T> >("l1_err_curvature_time_counter.dat");
//    auto test5_counter  = std::make_shared< gnuplot_output_object_time<T> >("linf_err_curvature_time_counter.dat");

        auto test_new0 = std::make_shared < gnuplot_output_object_time < T > > ("Lgrande1_err_u_n_time.dat");
//    auto test_new1  = std::make_shared< gnuplot_output_object_time<T> >("lpiccolo1_err_u_n_time_para.dat");
//    auto test_new2  = std::make_shared< gnuplot_output_object_time<T> >("linf_err_u_n_time_para.dat");
//    auto test_new3  = std::make_shared< gnuplot_output_object_time<T> >("Lgrande1_err_u_n_time_para.dat");
//    auto test_new4  = std::make_shared< gnuplot_output_object_time<T> >("max_val_u_n_time_para.dat");

//    auto test_new0_counter  = std::make_shared< gnuplot_output_object_time<T> >("Lgrande1_err_u_n_time_counter.dat");
//    auto test_new1_counter  = std::make_shared< gnuplot_output_object_time<T> >("lpiccolo1_err_u_n_time_para_counter.dat");
//    auto test_new2_counter  = std::make_shared< gnuplot_output_object_time<T> >("linf_err_u_n_time_para_counter.dat");
//    auto test_new3_counter  = std::make_shared< gnuplot_output_object_time<T> >("Lgrande1_err_u_n_time_para_counter.dat");
//    auto test_new4_counter  = std::make_shared< gnuplot_output_object_time<T> >("max_val_u_n_time_para_counter.dat");




//    auto test0b  = std::make_shared< gnuplot_output_object_time<T> >("area_time_normalised.dat");

//    auto test1b  = std::make_shared< gnuplot_output_object_time<T> >("l1_err_u_n_time_normalised.dat");
//    auto test2b  = std::make_shared< gnuplot_output_object_time<T> >("linf_err_u_n_time_normalised.dat");
//    auto test3b  = std::make_shared< gnuplot_output_object_time<T> >("max_val_u_n_time_normalised.dat");
//    auto test4b  = std::make_shared< gnuplot_output_object_time<T> >("l1_err_curvature_time_normalised.dat");
//    auto test5b  = std::make_shared< gnuplot_output_object_time<T> >("linf_err_curvature_time_normalised.dat");
// //auto test4c  = std::make_shared< gnuplot_output_object_time<T> >("l1_l1_err_curvature_err_time.dat");

        auto test_dt = std::make_shared < gnuplot_output_object_time < T > > ("dt_M.dat");
        auto test_e = std::make_shared < gnuplot_output_object_time < T > > ("eccentricity.dat");

        auto test0c = std::make_shared < gnuplot_output_object_time < T > > ("area_time_err.dat");
        auto test0c_counter = std::make_shared < gnuplot_output_object_time < T > > ("area_time_err_counter.dat");

//    auto testm0  = std::make_shared< gnuplot_output_object_time<T> >("min_time.dat");
//    auto testm1  = std::make_shared< gnuplot_output_object_time<T> >("min_normalised_time.dat");
//    auto testM0  = std::make_shared< gnuplot_output_object_time<T> >("max_time.dat");
//    auto testM1  = std::make_shared< gnuplot_output_object_time<T> >("max_normalised_time.dat");
        auto testflux = std::make_shared < gnuplot_output_object_time < T > > ("flux_interface_time.dat");
//    auto testvelx  = std::make_shared< gnuplot_output_object_time<T> >("rise_velocity_err_x_time.dat");
//    auto testvely  = std::make_shared< gnuplot_output_object_time<T> >("rise_velocity_err_y_time.dat");
        auto testvel = std::make_shared < gnuplot_output_object_time < T > > ("rise_velocity_err_time.dat");
//    auto testvel_counter  = std::make_shared< gnuplot_output_object_time<T> >("rise_velocity_err_time_counter.dat");
        auto testcomx = std::make_shared < gnuplot_output_object_time < T > > ("centre_mass_err_x_time.dat");
        auto testcomy = std::make_shared < gnuplot_output_object_time < T > > ("centre_mass_err_y_time.dat");
        auto testcom = std::make_shared < gnuplot_output_object_time < T > > ("centre_mass_err_time.dat");


        auto testper0 = std::make_shared < gnuplot_output_object_time < T > > ("perimeter_time.dat");
//    auto testper1  = std::make_shared< gnuplot_output_object_time<T> >("perimeter_normalised_time.dat");
//    auto testper2  = std::make_shared< gnuplot_output_object_time<T> >("perimeter_err_time.dat");
//    auto testper2_counter  = std::make_shared< gnuplot_output_object_time<T> >("perimeter_err_time_counter.dat");

        auto testcirc = std::make_shared < gnuplot_output_object_time < T > > ("circularity_time.dat");
        auto testcirc1 = std::make_shared < gnuplot_output_object_time < T > > ("circularity_error_time.dat");
        auto testcirc1_counter =
                std::make_shared < gnuplot_output_object_time < T > > ("circularity_error_time_counter.dat");

        size_t tot = l1_err_u_n_time.size();

//    testm0->add_data(time_vec[0] ,min_max_vec[0].first );
//    testm1->add_data(time_vec[0] ,1.0 );
//    testM0->add_data(time_vec[0] ,min_max_vec[0].second );
//    testM1->add_data(time_vec[0] ,1.0 );

        testcomx->add_data(time_vec[0], 0.0);
        testcomy->add_data(time_vec[0], 0.0);
        testcom->add_data(time_vec[0], 0.0);


        testper0->add_data(time_vec[0], perimeter_time[0]);
//    testper1->add_data(time_vec[0] , 1.0 );
//    testper2->add_data(time_vec[0] , std::abs(perimeter_time[0] - perimetre_ref)/perimetre_ref );
//    testper2_counter->add_data(0 , std::abs(perimeter_time[0] - perimetre_ref)/perimetre_ref );

        testcirc->add_data(time_vec[0], circularity_time[0]);
        testcirc1->add_data(time_vec[0], std::abs(circularity_time[0] - circularity_ref) / circularity_ref);
        testcirc1_counter->add_data(0, std::abs(circularity_time[0] - circularity_ref) / circularity_ref);

        test0c->add_data(time_vec[0], 0.0);
        test0c_counter->add_data(0, 0.0);
//test4c->add_data(time_vec[0] , 0.0 );


        test0->add_data(time_vec[0], area_time[0]);
// ADD 8/01/21
        test1->add_data(time_vec[0], l1_err_u_n_time[0]);
        test2->add_data(time_vec[0], linf_err_u_n_time[0]);
//    test1_counter->add_data(0 ,l1_err_u_n_time[0] );
//    test2_counter->add_data(0 ,linf_err_u_n_time[0] );
//    test3->add_data(time_vec[0] ,max_val_u_n_time[0] );
// UP TO HERE
        test4->add_data(time_vec[0], l1_err_curvature_time[0]);
        test5->add_data(time_vec[0], linf_err_curvature_time[0]);
//    test4_counter->add_data(0 ,l1_err_curvature_time[0] );
//    test5_counter->add_data(0 ,linf_err_curvature_time[0] );

        test_new0->add_data(time_vec[0], L1_err_u_n_time[0]);
//    test_new1->add_data(time_vec[0] ,l1_err_u_n_time_para[0] );
//    test_new2->add_data(time_vec[0] ,linf_err_u_n_time_para[0] );
//    test_new3->add_data(time_vec[0] ,L1_err_u_n_time_para[0] );
//    test_new4->add_data(time_vec[0] ,max_val_u_n_time_para[0] );
//
//    test_new0_counter->add_data(0 ,L1_err_u_n_time[0] );
//    test_new1_counter->add_data(0 ,l1_err_u_n_time_para[0] );
//    test_new2_counter->add_data(0 ,linf_err_u_n_time_para[0] );
//    test_new3_counter->add_data(0,L1_err_u_n_time_para[0] );
//    test_new4_counter->add_data(0 ,max_val_u_n_time_para[0] );
//
//
//
//    test0b->add_data(time_vec[0] ,area_time[0]/area_time[0] );
// ADD 8/01/21
//    test1b->add_data(time_vec[0] ,l1_err_u_n_time[0]/l1_err_u_n_time[0] );
//    test2b->add_data(time_vec[0] ,linf_err_u_n_time[0]/linf_err_u_n_time[0] );
//    test3b->add_data(time_vec[0] ,max_val_u_n_time[0]/max_val_u_n_time[0]  );
// UP TO HERE
//    test4b->add_data(time_vec[0] ,l1_err_curvature_time[0]/l1_err_curvature_time[0]  );
//    test5b->add_data(time_vec[0] ,linf_err_curvature_time[0]/linf_err_curvature_time[0]  );
//
//    testref0->add_data(time_vec[0] , area_ref );
//    testref1->add_data(time_vec[0] , perimetre_ref );
//    testref2->add_data(time_vec[0] , circularity_ref );

//    testanal0->add_data(time_vec[0] , area_analytic );
//    testanal1->add_data(time_vec[0] , perimeter_analytic );


        testflux->add_data(time_vec[0], flux_interface_time[0]);
//    testvelx ->add_data(time_vec[0] , std::abs(rise_velocity_time[0].first) );
//    testvely ->add_data(time_vec[0] , std::abs(rise_velocity_time[0].second) );
        testvel->add_data(time_vec[0], std::abs(rise_velocity_time[0].first) + std::abs(rise_velocity_time[0].second));
//    testvel_counter->add_data(0, std::abs(rise_velocity_time[0].first) + std::abs(rise_velocity_time[0].second) );
        test_dt->add_data(time_vec[0], 0.0);
//test_e->add_data(time_vec[0] , eccentricity_vec[0] );
        test_der_int->add_data(time_vec[0], 0.0);
        for (size_t i = 0; i < tot - 1; i++) {
            test0->add_data(time_vec[i + 1], area_time[i + 1]);
            test1->add_data(time_vec[i + 1], l1_err_u_n_time[i + 1]);
            test2->add_data(time_vec[i + 1], linf_err_u_n_time[i + 1]);
//test_der_int->add_data(time_vec[i+1] ,linf_der_time_interface[i] );

//        test1_counter->add_data(i+1 ,l1_err_u_n_time[i+1] );
//        test2_counter->add_data(i+1 ,linf_err_u_n_time[i+1] );
//        test3->add_data(time_vec[i+1] ,max_val_u_n_time[i+1] );
////        test1->add_data(time_vec[i+1] ,l1_err_u_n_time[i] );
////        test2->add_data(time_vec[i+1] ,linf_err_u_n_time[i] );
////        test1_counter->add_data(i+1 ,l1_err_u_n_time[i] );
////        test2_counter->add_data(i+1 ,linf_err_u_n_time[i] );
////        test3->add_data(time_vec[i+1] ,max_val_u_n_time[i] );
            test4->add_data(time_vec[i + 1], l1_err_curvature_time[i + 1]);
            test5->add_data(time_vec[i + 1], linf_err_curvature_time[i + 1]);
//        test4_counter->add_data(i+1 ,l1_err_curvature_time[i+1] );
//        test5_counter->add_data(i+1 ,linf_err_curvature_time[i+1] );

//        test0b->add_data(time_vec[i+1] ,area_time[i+1]/area_time[0] );
//
//        test1b->add_data(time_vec[i+1] ,l1_err_u_n_time[i+1]/l1_err_u_n_time[0] );
//        test2b->add_data(time_vec[i+1] ,linf_err_u_n_time[i+1]/linf_err_u_n_time[0] );
//        test3b->add_data(time_vec[i+1] ,max_val_u_n_time[i+1]/max_val_u_n_time[0]  );
////        test1b->add_data(time_vec[i+1] ,l1_err_u_n_time[i]/l1_err_u_n_time[0] );
////        test2b->add_data(time_vec[i+1] ,linf_err_u_n_time[i]/linf_err_u_n_time[0] );
////        test3b->add_data(time_vec[i+1] ,max_val_u_n_time[i]/max_val_u_n_time[0]  );
//        test4b->add_data(time_vec[i+1] ,l1_err_curvature_time[i+1]/l1_err_curvature_time[0] );
//        test5b->add_data(time_vec[i+1] ,linf_err_curvature_time[i+1]/linf_err_curvature_time[0]  );

            test_new0->add_data(time_vec[i + 1], L1_err_u_n_time[i + 1]);
//        test_new1->add_data(time_vec[i+1] ,l1_err_u_n_time_para[i+1] );
//        test_new2->add_data(time_vec[i+1] ,linf_err_u_n_time_para[i+1] );
//        test_new3->add_data(time_vec[i+1] ,L1_err_u_n_time_para[i+1] );
//        test_new4->add_data(time_vec[i+1] ,max_val_u_n_time_para[i+1] );

//        test_new0_counter->add_data(i+1 ,L1_err_u_n_time[i+1] );
//        test_new1_counter->add_data(i+1 ,l1_err_u_n_time_para[i+1] );
//        test_new2_counter->add_data(i+1 ,linf_err_u_n_time_para[i+1] );
//        test_new3_counter->add_data(i+1,L1_err_u_n_time_para[i+1] );
//        test_new4_counter->add_data(i+1 ,max_val_u_n_time_para[i+1] );

            testcomx->add_data(time_vec[i + 1],
                               std::abs(centre_mass_err_time[i + 1].first - centre_mass_err_time[0].first) /
                               centre_mass_err_time[0].first);
            testcomy->add_data(time_vec[i + 1],
                               std::abs(centre_mass_err_time[i + 1].second - centre_mass_err_time[0].second) /
                               centre_mass_err_time[0].second);
            testcom->add_data(time_vec[i + 1], std::abs(centre_mass_err_time[i + 1].first - centre_mass_err_time[0].first) /
                                               centre_mass_err_time[0].first + std::abs(
                    centre_mass_err_time[i + 1].second - centre_mass_err_time[0].second) / centre_mass_err_time[0].second);


//        testm0->add_data(time_vec[i+1] , min_max_vec[i+1].first );
//        testm1->add_data(time_vec[i+1] ,min_max_vec[i+1].first/min_max_vec[0].first );
//        testM0->add_data(time_vec[i+1] , min_max_vec[i+1].second );
//        testM1->add_data(time_vec[i+1] ,min_max_vec[i+1].second/min_max_vec[0].second );


            testper0->add_data(time_vec[i + 1], perimeter_time[i + 1]);
//        testper1->add_data(time_vec[i+1] , perimeter_time[i+1]/perimeter_time[0] );
//        testper2->add_data(time_vec[i+1] , std::abs( perimeter_time[i+1] - perimetre_ref) / perimetre_ref );
//        testper2_counter->add_data(i+1 , std::abs( perimeter_time[i+1] - perimetre_ref) / perimetre_ref );

            testcirc->add_data(time_vec[i + 1], circularity_time[i + 1]);
            testcirc1->add_data(time_vec[i + 1], std::abs(circularity_time[i + 1] - circularity_ref) / circularity_ref);
            testcirc1_counter->add_data(i + 1, std::abs(circularity_time[i + 1] - circularity_ref) / circularity_ref);

            test0c->add_data(time_vec[i + 1], std::abs(area_time[i + 1] - area_time[0]) / area_time[0]);
            test0c_counter->add_data(i + 1, std::abs(area_time[i + 1] - area_time[0]) / area_time[0]);
//test4c->add_data(time_vec[i+1] , std::abs(l1_err_curvature_time[i+1] - l1_err_curvature_time[0] )/l1_err_curvature_time[0]);

            testflux->add_data(time_vec[i + 1], flux_interface_time[i + 1]);
//        testvelx ->add_data(time_vec[i+1] , std::abs(rise_velocity_time[i+1].first) );
//        testvely ->add_data(time_vec[i+1] , std::abs(rise_velocity_time[i+1].second) );
            testvel->add_data(time_vec[i + 1],
                              std::abs(rise_velocity_time[i + 1].first) + std::abs(rise_velocity_time[i + 1].second));
//        testvel_counter->add_data(i+1 , std::abs(rise_velocity_time[i+1].first) + std::abs(rise_velocity_time[i+1].second) );

//        testref0->add_data(time_vec[i+1] , area_ref );
//        testref1->add_data(time_vec[i+1] , perimetre_ref );
//        testref2->add_data(time_vec[i+1] , circularity_ref );
//
//        testanal0->add_data(time_vec[i+1] , area_analytic );
//        testanal1->add_data(time_vec[i+1] , perimeter_analytic );

//std::cout<<"time_vec[i] = "<<time_vec[i]<<", time_vec[i+1] = "<<time_vec[i+1]<<", dt = "<<dt<<", error = "<<std::abs( std::abs( time_vec[i+1] - time_vec[i] ) - dt )<<std::endl;
//        if( std::abs( std::abs( time_vec[i+1] - time_vec[i] ) - dt ) > 1e-10 ){
//
//            test_dt->add_data(time_vec[i+1] , 0.0 );
//        }
            test_dt->add_data(time_vec[i + 1], time_vec[i + 1] - time_vec[i]);
//test_e->add_data(time_vec[i+1] , eccentricity_vec[i] );

        }


        postoutput.add_object(test0);
        postoutput.add_object(test1);
        postoutput.add_object(test2);
        postoutput.add_object(test_der_int);


//    postoutput.add_object(test1_counter);
//    postoutput.add_object(test2_counter);


//    postoutput.add_object(test3);
        postoutput.add_object(test4);
        postoutput.add_object(test5);
//    postoutput.add_object(test4_counter);
//    postoutput.add_object(test5_counter);

        postoutput.add_object(test_new0);
//    postoutput.add_object(test_new1);
//    postoutput.add_object(test_new2);
//    postoutput.add_object(test_new3);
//    postoutput.add_object(test_new4);
//
//    postoutput.add_object(test_new0_counter);
//    postoutput.add_object(test_new1_counter);
//    postoutput.add_object(test_new2_counter);
//    postoutput.add_object(test_new3_counter);
//    postoutput.add_object(test_new4_counter);
//
//
//
//
//    postoutput.add_object(test0b);
//    postoutput.add_object(test1b);
//    postoutput.add_object(test2b);
//    postoutput.add_object(test3b);
//    postoutput.add_object(test4b);
//    postoutput.add_object(test5b);

        postoutput.add_object(testcomx);
        postoutput.add_object(testcomy);
        postoutput.add_object(testcom);

//    postoutput.add_object(testm0);
//    postoutput.add_object(testm1);
//    postoutput.add_object(testM0);
//    postoutput.add_object(testM1);

        postoutput.add_object(testper0);
//    postoutput.add_object(testper1);
//    postoutput.add_object(testper2);
//    postoutput.add_object(testper2_counter);


        postoutput.add_object(testcirc);
        postoutput.add_object(testcirc1);
        postoutput.add_object(testcirc1_counter);

        postoutput.add_object(test0c);
        postoutput.add_object(test0c_counter);
//postoutput.add_object(test4c);

        postoutput.add_object(testflux);
//    postoutput.add_object(testvelx);
//    postoutput.add_object(testvely);
        postoutput.add_object(testvel);
//    postoutput.add_object(testvel_counter);


//    postoutput.add_object(testref0);
//    postoutput.add_object(testref1);
//    postoutput.add_object(testref2);

//    postoutput.add_object(testanal0);
//    postoutput.add_object(testanal1);


        postoutput.add_object(test_dt);
        postoutput.add_object(test_e);

        postoutput.write();

    }


    template<typename T>
    void
    plotting_in_time(const std::vector <T> &time_vec, const std::vector <T> &area_time,
                     const std::vector <T> &l1_err_u_n_time, const std::vector <T> &linf_err_u_n_time,
                     const std::vector <T> &max_val_u_n_time, const std::vector <T> &l1_err_curvature_time,
                     const std::vector <T> &linf_err_curvature_time, T dt) {

        postprocess_output <T> postoutput;

        auto test0 = std::make_shared < gnuplot_output_object_time < double > > ("area_time.dat");
        auto test1 = std::make_shared < gnuplot_output_object_time < double > > ("l1_err_u_n_time.dat");
        auto test2 = std::make_shared < gnuplot_output_object_time < double > > ("linf_err_u_n_time.dat");
        auto test3 = std::make_shared < gnuplot_output_object_time < double > > ("max_val_u_n_time.dat");
        auto test4 = std::make_shared < gnuplot_output_object_time < double > > ("l1_err_curvature_time.dat");
        auto test5 = std::make_shared < gnuplot_output_object_time < double > > ("linf_err_curvature_time.dat");

        auto test0b = std::make_shared < gnuplot_output_object_time < double > > ("area_time_normalised.dat");
        auto test1b = std::make_shared < gnuplot_output_object_time < double > > ("l1_err_u_n_time_normalised.dat");
        auto test2b = std::make_shared < gnuplot_output_object_time < double > > ("linf_err_u_n_time_normalised.dat");
        auto test3b = std::make_shared < gnuplot_output_object_time < double > > ("max_val_u_n_time_normalised.dat");
        auto test4b = std::make_shared < gnuplot_output_object_time < double > > ("l1_err_curvature_time_normalised.dat");
        auto test5b = std::make_shared < gnuplot_output_object_time < double > > ("linf_err_curvature_time_normalised.dat");

        auto test_dt = std::make_shared < gnuplot_output_object_time < double > > ("dt_M.dat");


        size_t tot = l1_err_u_n_time.size();
        test0->add_data(time_vec[0], area_time[0]);
        test4->add_data(time_vec[0], l1_err_curvature_time[0]);
        test5->add_data(time_vec[0], linf_err_curvature_time[0]);

        test0b->add_data(time_vec[0], area_time[0] / area_time[0]);
        test4b->add_data(time_vec[0], l1_err_curvature_time[0] / l1_err_curvature_time[0]);
        test5b->add_data(time_vec[0], linf_err_curvature_time[0] / linf_err_curvature_time[0]);

        for (size_t i = 0; i < tot; i++) {
            test0->add_data(time_vec[i + 1], area_time[i + 1]);
            test1->add_data(time_vec[i + 1], l1_err_u_n_time[i]);
            test2->add_data(time_vec[i + 1], linf_err_u_n_time[i]);
            test3->add_data(time_vec[i + 1], max_val_u_n_time[i]);
            test4->add_data(time_vec[i + 1], l1_err_curvature_time[i + 1]);
            test5->add_data(time_vec[i + 1], linf_err_curvature_time[i + 1]);

            test0b->add_data(time_vec[i + 1], area_time[i + 1] / area_time[0]);
            test1b->add_data(time_vec[i + 1], l1_err_u_n_time[i] / l1_err_u_n_time[0]);
            test2b->add_data(time_vec[i + 1], linf_err_u_n_time[i] / linf_err_u_n_time[0]);
            test3b->add_data(time_vec[i + 1], max_val_u_n_time[i] / max_val_u_n_time[0]);
            test4b->add_data(time_vec[i + 1], l1_err_curvature_time[i + 1] / l1_err_curvature_time[0]);
            test5b->add_data(time_vec[i + 1], linf_err_curvature_time[i + 1] / linf_err_curvature_time[0]);

//std::cout<<"time_vec[i] = "<<time_vec[i]<<", time_vec[i+1] = "<<time_vec[i+1]<<", dt = "<<dt<<", error = "<<std::abs( std::abs( time_vec[i+1] - time_vec[i] ) - dt )<<std::endl;
            if (std::abs(std::abs(time_vec[i + 1] - time_vec[i]) - dt) > 1e-10) {
                test_dt->add_data(time_vec[i], 0.0);
                test_dt->add_data(time_vec[i + 1], 0.0);
            }
        }


        postoutput.add_object(test0);
        postoutput.add_object(test1);
        postoutput.add_object(test2);
        postoutput.add_object(test3);
        postoutput.add_object(test4);
        postoutput.add_object(test5);
        postoutput.add_object(test0b);
        postoutput.add_object(test1b);
        postoutput.add_object(test2b);
        postoutput.add_object(test3b);
        postoutput.add_object(test4b);
        postoutput.add_object(test5b);

        postoutput.add_object(test_dt);

        postoutput.write();

    }



/// Useful to plot level set pre FE transport problem
/// in cuthho_export.hpp
    template<typename Mesh, typename Function>
    void
    output_mesh_info2_pre_FEM(const Mesh &msh, const Function &level_set_function) {
        using RealType = typename Mesh::coordinate_type;

/************** OPEN SILO DATABASE **************/
        silo_database silo;
        silo.create("cuthho_meshinfo_preFEM_Stokes.silo");
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

/************** MAKE A SILO VARIABLE FOR CELL HIGHLIGHT **************/
        std::vector <RealType> highlight_markers;
        for (auto &cl: msh.cells) {
            if (cl.user_data.highlight)
                highlight_markers.push_back(1.0);
            else
                highlight_markers.push_back(0.0);

        }
        silo.add_variable("mesh", "highlighted_cells", highlight_markers.data(), highlight_markers.size(),
                          zonal_variable_t);

/************** MAKE A SILO VARIABLE FOR LEVEL SET FUNCTION **************/
        std::vector <RealType> level_set_vals;
// for (auto& pt : msh.points)
//    level_set_vals.push_back( level_set_function(pt) );
        for (auto &n: msh.nodes)
            level_set_vals.push_back(level_set_function(n));

        silo.add_variable("mesh", "level_set", level_set_vals.data(), level_set_vals.size(), nodal_variable_t);

/************** MAKE A SILO VARIABLE FOR NODE POSITIONING **************/
        std::vector <RealType> node_pos;
        for (auto &n: msh.nodes)
            node_pos.push_back(location(msh, n) == element_location::IN_POSITIVE_SIDE ? +1.0 : -1.0);
        silo.add_variable("mesh", "node_pos", node_pos.data(), node_pos.size(), nodal_variable_t);

        std::vector <RealType> cell_set;
        for (auto &cl: msh.cells) {
            RealType r;

            switch (cl.user_data.agglo_set) {
                case cell_agglo_set::UNDEF:
                    r = 0.0;
                    break;

                case cell_agglo_set::T_OK:
                    r = 1.0;
                    break;

                case cell_agglo_set::T_KO_NEG:
                    r = 2.0;
                    break;

                case cell_agglo_set::T_KO_POS:
                    r = 3.0;
                    break;

            }

            cell_set.push_back(r);
        }
        silo.add_variable("mesh", "agglo_set", cell_set.data(), cell_set.size(), zonal_variable_t);

        silo.close();

/*************  MAKE AN OUTPUT FOR THE INTERSECTION POINTS *************/
        std::vector <RealType> int_pts_x;
        std::vector <RealType> int_pts_y;

        for (auto &fc: msh.faces) {
            if (fc.user_data.location != element_location::ON_INTERFACE) continue;

            RealType x = fc.user_data.intersection_point.x();
            RealType y = fc.user_data.intersection_point.y();

            int_pts_x.push_back(x);
            int_pts_y.push_back(y);
        }

        std::ofstream points_file_pre_FEM("int_points_pre_FEM.3D", std::ios::out | std::ios::trunc);

        if (points_file_pre_FEM) {
// instructions
            points_file_pre_FEM << "X   Y   Z   val" << std::endl;

            for (size_t i = 0; i < int_pts_x.size(); i++) {
                points_file_pre_FEM << int_pts_x[i] << "   " << int_pts_y[i]
                                    << "   0.0     0.0" << std::endl;
            }

            points_file_pre_FEM.close();
        } else
            std::cerr << "points_file_pre_FEM has not been opened" << std::endl;


/*************  MAKE AN OUTPUT FOR THE INTERFACE *************/
        std::vector <RealType> int_x;
        std::vector <RealType> int_y;

        for (auto &cl: msh.cells) {
            if (cl.user_data.location != element_location::ON_INTERFACE) continue;

            for (size_t i = 0; i < cl.user_data.interface.size(); i++) {
                RealType x = cl.user_data.interface.at(i).x();
                RealType y = cl.user_data.interface.at(i).y();

                int_x.push_back(x);
                int_y.push_back(y);
            }
        }
        std::ofstream interface_file_preFEM("interface_pre_FEM_Stokes.3D", std::ios::out | std::ios::trunc);

        if (interface_file_preFEM) {
// instructions
            interface_file_preFEM << "X   Y   Z   val" << std::endl;

            for (size_t i = 0; i < int_x.size(); i++) {
                interface_file_preFEM << int_x[i] << "   " << int_y[i]
                                      << "   0.0     0.0" << std::endl;
            }

            interface_file_preFEM.close();
        } else
            std::cerr << "interface_file_preFEM has not been opened" << std::endl;
    }


    template<typename Mesh, typename VEC, typename T>
    void
    goal_quantities_time_fast_para(const Mesh &msh, const VEC &interface_gamma, const VEC &tangent_gamma,
                                   const VEC &normal_gamma, const std::vector <T> &curvature_gamma) {

        std::string filename_stokes4 = "tangent_gamma_para.3D";
        std::ofstream interface_file4(filename_stokes4, std::ios::out | std::ios::trunc);

        if (interface_file4) {
// instructions
            interface_file4 << "X   Y   val0   val1" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_gamma.begin();
                 interface_point < interface_gamma.end(); interface_point++) {
// << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                interface_file4 << (*interface_point)(0) << "   " << (*interface_point)(1) << "   "
                                << tangent_gamma[i](0) << "   " << tangent_gamma[i](1) << std::endl;

                i++;

            }

            interface_file4.close();
        } else
            std::cerr << "File 'tangent_gamma' has not been opened" << std::endl;

        std::string filename_stokes0 = "normal_gamma_para.3D";
        std::ofstream interface_file0(filename_stokes0, std::ios::out | std::ios::trunc);

        if (interface_file0) {
// instructions
            interface_file0 << "X   Y   val0   val1" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_gamma.begin();
                 interface_point < interface_gamma.end(); interface_point++) {
// << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                interface_file0 << (*interface_point)(0) << "   " << (*interface_point)(1) << "   "
                                << normal_gamma[i](0) << "   " << normal_gamma[i](1) << std::endl;

                i++;

            }

            interface_file0.close();
        } else
            std::cerr << "File 'normal_gamma' has not been opened" << std::endl;

        std::string filename_stokes1 = "curvature_gamma_para.3D";
        std::ofstream interface_file1(filename_stokes1, std::ios::out | std::ios::trunc);

        if (interface_file1) {
// instructions
            interface_file1 << "X   Y   val0" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_gamma.begin();
                 interface_point < interface_gamma.end(); interface_point++) {
//<< std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                interface_file1 << (*interface_point)(0) << "   " << (*interface_point)(1) << "   "
                                << curvature_gamma[i] << std::endl;

                i++;

            }

            interface_file1.close();
        } else
            std::cerr << "File 'curvature_gamma' has not been opened" << std::endl;


    }


    template<typename Mesh, typename VEC, typename T>
    void
    goal_quantities_time_fast_para_cont(const Mesh &msh, const VEC &interface_gamma, const VEC &tangent_gamma,
                                        const VEC &normal_gamma, const std::vector <T> &curvature_gamma) {

        std::string filename_stokes4 = "tangent_gamma_para_cont.3D";
        std::ofstream interface_file4(filename_stokes4, std::ios::out | std::ios::trunc);

        if (interface_file4) {
// instructions
            interface_file4 << "X   Y   val0   val1" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_gamma.begin();
                 interface_point < interface_gamma.end(); interface_point++) {
// << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                interface_file4 << (*interface_point)(0) << "   " << (*interface_point)(1) << "   "
                                << tangent_gamma[i](0) << "   " << tangent_gamma[i](1) << std::endl;

                i++;

            }

            interface_file4.close();
        } else
            std::cerr << "File 'tangent_gamma' has not been opened" << std::endl;

        std::string filename_stokes0 = "normal_gamma_para_cont.3D";
        std::ofstream interface_file0(filename_stokes0, std::ios::out | std::ios::trunc);

        if (interface_file0) {
// instructions
            interface_file0 << "X   Y   val0   val1" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_gamma.begin();
                 interface_point < interface_gamma.end(); interface_point++) {
// << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                interface_file0 << (*interface_point)(0) << "   " << (*interface_point)(1) << "   "
                                << normal_gamma[i](0) << "   " << normal_gamma[i](1) << std::endl;

                i++;

            }

            interface_file0.close();
        } else
            std::cerr << "File 'normal_gamma' has not been opened" << std::endl;

        std::string filename_stokes1 = "curvature_gamma_para_cont.3D";
        std::ofstream interface_file1(filename_stokes1, std::ios::out | std::ios::trunc);

        if (interface_file1) {
// instructions
            interface_file1 << "X   Y   val0" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_gamma.begin();
                 interface_point < interface_gamma.end(); interface_point++) {
//std::cout<<val_u_nx[i]<<std::endl;  << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                interface_file1 << (*interface_point)(0) << "   " << (*interface_point)(1) << "   "
                                << curvature_gamma[i] << std::endl;

                i++;

            }

            interface_file1.close();
        } else
            std::cerr << "File 'curvature_gamma' has not been opened" << std::endl;


    }


    template<typename Mesh, typename VEC, typename T>
    void
    goal_quantities_time_fast(const Mesh &msh, const VEC &interface_gamma, const VEC &tangent_gamma,
                              const VEC &normal_gamma, const std::vector <T> &curvature_gamma) {

        std::string filename_stokes4 = "tangent_gamma.3D";
        std::ofstream interface_file4(filename_stokes4, std::ios::out | std::ios::trunc);

        if (interface_file4) {
// instructions
            interface_file4 << "X   Y   val0   val1" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_gamma.begin();
                 interface_point < interface_gamma.end(); interface_point++) {
//std::cout<<val_u_nx[i]<<std::endl; << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                interface_file4 << (*interface_point)(0) << "   " << (*interface_point)(1) << "   "
                                << tangent_gamma[i](0) << "   " << tangent_gamma[i](1) << std::endl;

                i++;

            }

            interface_file4.close();
        } else
            std::cerr << "File 'tangent_gamma' has not been opened" << std::endl;

        std::string filename_stokes0 = "normal_gamma.3D";
        std::ofstream interface_file0(filename_stokes0, std::ios::out | std::ios::trunc);

        if (interface_file0) {
// instructions
            interface_file0 << "X   Y   val0   val1" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_gamma.begin();
                 interface_point < interface_gamma.end(); interface_point++) {
//std::cout<<val_u_nx[i]<<std::endl; << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                interface_file0 << (*interface_point)(0) << "   " << (*interface_point)(1) << "   "
                                << normal_gamma[i](0) << "   " << normal_gamma[i](1) << std::endl;

                i++;

            }

            interface_file0.close();
        } else
            std::cerr << "File 'normal_gamma' has not been opened" << std::endl;

        std::string filename_stokes1 = "curvature_gamma.3D";
        std::ofstream interface_file1(filename_stokes1, std::ios::out | std::ios::trunc);

        if (interface_file1) {
// instructions
            interface_file1 << "X   Y   val0" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_gamma.begin();
                 interface_point < interface_gamma.end(); interface_point++) {
//std::cout<<val_u_nx[i]<<std::endl; << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
                interface_file1 << (*interface_point)(0) << "   " << (*interface_point)(1) << "   "
                                << curvature_gamma[i] << std::endl;

                i++;

            }

            interface_file1.close();
        } else
            std::cerr << "File 'curvature_gamma' has not been opened" << std::endl;


    }

    template<typename Mesh, typename VEC, typename T>
    void
    goal_quantities_time_fast(const Mesh &msh, const VEC &interface_points, const std::vector <T> &val_u_n,
                              const std::vector <std::pair<T, T>> &vec_n, size_t time_step, std::string &folder) {

        std::string filename_stokes4 = folder + "vec_u_n_" + std::to_string(time_step) + ".3D";
        std::ofstream interface_file4(filename_stokes4, std::ios::out | std::ios::trunc);

        if (interface_file4) {
// instructions
            interface_file4 << "X   Y   val0   val1" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_points.begin();
                 interface_point < interface_points.end(); interface_point++) {
//std::cout<<val_u_nx[i]<<std::endl;
                interface_file4 << (*interface_point).x() << "   " << (*interface_point).y() << "   "
                                << val_u_n[i] * vec_n[i].first << "   " << val_u_n[i] * vec_n[i].second << std::endl;

                i++;

            }

            interface_file4.close();
        } else
            std::cerr << "File 'vec_u_n' has not been opened" << std::endl;

        std::string filename_stokes7 = folder + "vec_normal_" + std::to_string(time_step) + ".3D";
        std::ofstream interface_file7(filename_stokes7, std::ios::out | std::ios::trunc);

        if (interface_file7) {
// instructions
            interface_file7 << "X   Y   val0   val1" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_points.begin();
                 interface_point < interface_points.end(); interface_point++) {
//std::cout<<val_u_nx[i]<<std::endl;
                interface_file7 << (*interface_point).x() << "   " << (*interface_point).y() << "   "
                                << vec_n[i].first << "   " << vec_n[i].second << std::endl;

                i++;

            }

            interface_file7.close();
        } else
            std::cerr << "File 'vec_normal_' has not been opened" << std::endl;

    }

    template<typename Mesh, typename Level_Set, typename Velocity, typename T = typename Mesh::coordinate_type>
    void
    plot_u_n_interface(Mesh &msh_i, Level_Set &ls_cell, Velocity &u_projected, size_t time_step, std::string &folder) {


        std::vector <T> val_u_n_fin; //val_u_nx_fin , val_u_ny_fin ;
        std::vector <point<T, 2>> interface_points_plot_fin;
        std::vector <std::pair<T, T>> vec_n; // , velocity_interface , velocity_field , points_vel_field;



        for (auto &cl: msh_i.cells) {


            if (cl.user_data.location == element_location::ON_INTERFACE) {

                ls_cell.cell_assignment(cl);
                u_projected.cell_assignment(cl);

                for (auto interface_point = cl.user_data.interface.begin();
                     interface_point < cl.user_data.interface.end(); interface_point++) {
                    Eigen::Matrix<T, 2, 1> normal_cont_grad = ls_cell.normal(*interface_point);
                    std::pair <T, T> normal_vec_grad_cont = std::make_pair(normal_cont_grad(0), normal_cont_grad(1));

                    vec_n.push_back(normal_vec_grad_cont);

                    auto u_pt = u_projected(*(interface_point));
                    auto ls_n_pt = ls_cell.normal(*(interface_point));

                    T u_n_0 = u_pt.first * ls_n_pt(0);
                    T u_n_1 = u_pt.second * ls_n_pt(1);

                    interface_points_plot_fin.push_back(*(interface_point));

                    val_u_n_fin.push_back(u_n_0 + u_n_1);


                }


            }


        }


        goal_quantities_time_fast(msh_i, interface_points_plot_fin, val_u_n_fin, vec_n, time_step, folder);


    }


    template<typename Mesh, typename VEC, typename T>
    void
    goal_quantities_time(const Mesh &msh, const VEC &interface_points, const std::vector <T> &val_u_n,
                         const std::vector <std::pair<T, T>> &vec_u_n, const std::vector <std::pair<T, T>> &velocity_field,
                         const std::vector <std::pair<T, T>> &points_vel_field, size_t time_step) {
/*
    std::string filename_stokes0 = "val_u_nx_" + std::to_string(time_step) + ".3D";
    std::ofstream interface_file0(filename_stokes0, std::ios::out | std::ios::trunc);

    if(interface_file0)
    {
        // instructions
        interface_file0 << "X   Y   Z   val" << std::endl;
        size_t i = 0;
        for(auto interface_point = interface_points.begin() ; interface_point < interface_points.end() ; interface_point++ )
        {
            //std::cout<<val_u_nx[i]<<std::endl;
            interface_file0 << (*interface_point).x() << "   " <<  (*interface_point).y() << "   "
            << val_u_nx[i] << "    0.0"<< std::endl;

            i++;

        }

        interface_file0.close();
    }
    else
        std::cerr << "File 'val_u_nx' has not been opened" << std::endl;

    std::string filename_stokes1 = "val_u_ny_" + std::to_string(time_step) + ".3D";
    std::ofstream interface_file1(filename_stokes1, std::ios::out | std::ios::trunc);

       if(interface_file1)
       {
           // instructions
           interface_file1 << "X   Y   Z   val" << std::endl;
           size_t i = 0;
           for(auto interface_point = interface_points.begin() ; interface_point < interface_points.end() ; interface_point++ )
           {
               //std::cout<<val_u_ny[i]<<std::endl;
               interface_file1 << (*interface_point).x() << "   " <<  (*interface_point).y() << "   "
               << val_u_ny[i] << "    0.0"<< std::endl;
               i++;
           }

           interface_file1.close();
       }
       else
           std::cerr << "File 'val_u_ny' has not been opened" << std::endl;


    std::string filename_stokes2 = "val_u_n_" + std::to_string(time_step) + ".3D";
    std::ofstream interface_file2(filename_stokes2, std::ios::out | std::ios::trunc);

    if(interface_file2)
    {
        // instructions
        interface_file2 << "X   Y   Z   val" << std::endl;
        size_t i = 0;
        for(auto interface_point = interface_points.begin() ; interface_point < interface_points.end() ; interface_point++ )
        {
            //std::cout<<val_u_n[i]<<std::endl;
            interface_file2 << (*interface_point).x() << "   " <<  (*interface_point).y() << "   "
            << val_u_n[i] << "    0.0"<< std::endl;
            i++;
        }

        interface_file2.close();
    }
    else
        std::cerr << "File 'val_u_n' has not been opened" << std::endl;



    postprocess_output<double> postoutput_vec_u_n;
    std::string filename_stokes3 = "vec_u_n_" + std::to_string(time_step) + ".dat";
    auto test_vec_u_n = std::make_shared< gnuplot_output_object_vec<double> >(filename_stokes3);

    size_t i = 0;
    for(auto interface_point = interface_points.begin() ; interface_point < interface_points.end() ; interface_point++ )
    {
        test_vec_u_n->add_data(*interface_point,std::make_pair( val_u_n[i]*vec_u_n[i].first , val_u_n[i]*vec_u_n[i].second ) );
        i++;
    }

    postoutput_vec_u_n.add_object(test_vec_u_n);
    postoutput_vec_u_n.write();

   */
        std::string filename_stokes4 = "vec_u_n_" + std::to_string(time_step) + ".3D";
        std::ofstream interface_file4(filename_stokes4, std::ios::out | std::ios::trunc);

        if (interface_file4) {
// instructions
            interface_file4 << "X   Y   val0   val1" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_points.begin();
                 interface_point < interface_points.end(); interface_point++) {
//std::cout<<val_u_nx[i]<<std::endl;
                interface_file4 << (*interface_point).x() << "   " << (*interface_point).y() << "   "
                                << val_u_n[i] * vec_u_n[i].first << "   " << val_u_n[i] * vec_u_n[i].second << std::endl;

                i++;

            }

            interface_file4.close();
        } else
            std::cerr << "File 'vec_u_n' has not been opened" << std::endl;

/*
    std::string filename_stokes5 = "velocity_interface_" + std::to_string(time_step) + ".3D";
    std::ofstream interface_file5(filename_stokes5, std::ios::out | std::ios::trunc);

    if(interface_file5)
    {
        // instructions
        interface_file5 << "X   Y   val0   val1" << std::endl;
        size_t i = 0;
        for(auto interface_point = interface_points.begin() ; interface_point < interface_points.end() ; interface_point++ )
        {
            //std::cout<<val_u_nx[i]<<std::endl;
            interface_file5 << (*interface_point).x() << "   " <<  (*interface_point).y() << "   "
            << velocity_interface[i].first << "   " << velocity_interface[i].second<< std::endl;

            i++;

        }

        interface_file5.close();
    }
    else
        std::cerr << "File 'vec_u_n' has not been opened" << std::endl;

     */
//    std::string filename_stokes6 = "velocity_field_" + std::to_string(time_step) + ".3D";
        std::string filename_stokes6 = "velocity_field_.3D";
        std::ofstream interface_file6(filename_stokes6, std::ios::out | std::ios::trunc);

        if (interface_file6) {
// instructions
            interface_file6 << "X   Y   val0   val1" << std::endl;
            size_t i = 0;
            for (auto point_vel = points_vel_field.begin(); point_vel < points_vel_field.end(); point_vel++) {
//std::cout<<val_u_nx[i]<<std::endl;
                interface_file6 << (*point_vel).first << "   " << (*point_vel).second << "   "
                                << velocity_field[i].first << "   " << velocity_field[i].second << std::endl;

                i++;

            }

            interface_file6.close();
        } else
            std::cerr << "File 'vec_u_n' has not been opened" << std::endl;
/*
    std::string filename_stokes7 = "vec_normal_" + std::to_string(time_step) + ".3D";
    std::ofstream interface_file7(filename_stokes7, std::ios::out | std::ios::trunc);

    if(interface_file7)
    {
        // instructions
        interface_file7 << "X   Y   val0   val1" << std::endl;
        size_t i = 0;
        for(auto interface_point = interface_points.begin() ; interface_point < interface_points.end() ; interface_point++ )
        {
            //std::cout<<val_u_nx[i]<<std::endl;
            interface_file7 << (*interface_point).x() << "   " <<  (*interface_point).y() << "   "
            << vec_u_n[i].first << "   " << vec_u_n[i].second<< std::endl;

            i++;

        }

        interface_file7.close();
    }
    else
        std::cerr << "File 'vec_normal_' has not been opened" << std::endl;
    */
    }


    template<typename Mesh, typename VEC, typename T>
    void
    goal_quantities_time(const Mesh &msh, T time, const VEC &interface_points, const std::vector <T> &val_u_nx,
                         const std::vector <T> &val_u_ny, const std::vector <T> &val_u_n,
                         const std::vector <std::pair<T, T>> &vec_u_n,
                         const std::vector <std::pair<T, T>> &velocity_interface,
                         const std::vector <std::pair<T, T>> &velocity_field,
                         const std::vector <std::pair<T, T>> &points_vel_field, size_t time_step) {
/*
    std::string filename_stokes0 = "val_u_nx_" + std::to_string(time_step) + ".3D";
    std::ofstream interface_file0(filename_stokes0, std::ios::out | std::ios::trunc);

    if(interface_file0)
    {
        // instructions
        interface_file0 << "X   Y   Z   val" << std::endl;
        size_t i = 0;
        for(auto interface_point = interface_points.begin() ; interface_point < interface_points.end() ; interface_point++ )
        {
            //std::cout<<val_u_nx[i]<<std::endl;
            interface_file0 << (*interface_point).x() << "   " <<  (*interface_point).y() << "   "
            << val_u_nx[i] << "    0.0"<< std::endl;

            i++;

        }

        interface_file0.close();
    }
    else
        std::cerr << "File 'val_u_nx' has not been opened" << std::endl;

    std::string filename_stokes1 = "val_u_ny_" + std::to_string(time_step) + ".3D";
    std::ofstream interface_file1(filename_stokes1, std::ios::out | std::ios::trunc);

       if(interface_file1)
       {
           // instructions
           interface_file1 << "X   Y   Z   val" << std::endl;
           size_t i = 0;
           for(auto interface_point = interface_points.begin() ; interface_point < interface_points.end() ; interface_point++ )
           {
               //std::cout<<val_u_ny[i]<<std::endl;
               interface_file1 << (*interface_point).x() << "   " <<  (*interface_point).y() << "   "
               << val_u_ny[i] << "    0.0"<< std::endl;
               i++;
           }

           interface_file1.close();
       }
       else
           std::cerr << "File 'val_u_ny' has not been opened" << std::endl;


    std::string filename_stokes2 = "val_u_n_" + std::to_string(time_step) + ".3D";
    std::ofstream interface_file2(filename_stokes2, std::ios::out | std::ios::trunc);

    if(interface_file2)
    {
        // instructions
        interface_file2 << "X   Y   Z   val" << std::endl;
        size_t i = 0;
        for(auto interface_point = interface_points.begin() ; interface_point < interface_points.end() ; interface_point++ )
        {
            //std::cout<<val_u_n[i]<<std::endl;
            interface_file2 << (*interface_point).x() << "   " <<  (*interface_point).y() << "   "
            << val_u_n[i] << "    0.0"<< std::endl;
            i++;
        }

        interface_file2.close();
    }
    else
        std::cerr << "File 'val_u_n' has not been opened" << std::endl;



    postprocess_output<double> postoutput_vec_u_n;
    std::string filename_stokes3 = "vec_u_n_" + std::to_string(time_step) + ".dat";
    auto test_vec_u_n = std::make_shared< gnuplot_output_object_vec<double> >(filename_stokes3);

    size_t i = 0;
    for(auto interface_point = interface_points.begin() ; interface_point < interface_points.end() ; interface_point++ )
    {
        test_vec_u_n->add_data(*interface_point,std::make_pair( val_u_n[i]*vec_u_n[i].first , val_u_n[i]*vec_u_n[i].second ) );
        i++;
    }

    postoutput_vec_u_n.add_object(test_vec_u_n);
    postoutput_vec_u_n.write();

   */
        std::string filename_stokes4 = "vec_u_n_" + std::to_string(time_step) + ".3D";
        std::ofstream interface_file4(filename_stokes4, std::ios::out | std::ios::trunc);

        if (interface_file4) {
// instructions
            interface_file4 << "X   Y   val0   val1" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_points.begin();
                 interface_point < interface_points.end(); interface_point++) {
//std::cout<<val_u_nx[i]<<std::endl;
                interface_file4 << (*interface_point).x() << "   " << (*interface_point).y() << "   "
                                << val_u_n[i] * vec_u_n[i].first << "   " << val_u_n[i] * vec_u_n[i].second << std::endl;

                i++;

            }

            interface_file4.close();
        } else
            std::cerr << "File 'vec_u_n' has not been opened" << std::endl;

/*
    std::string filename_stokes5 = "velocity_interface_" + std::to_string(time_step) + ".3D";
    std::ofstream interface_file5(filename_stokes5, std::ios::out | std::ios::trunc);

    if(interface_file5)
    {
        // instructions
        interface_file5 << "X   Y   val0   val1" << std::endl;
        size_t i = 0;
        for(auto interface_point = interface_points.begin() ; interface_point < interface_points.end() ; interface_point++ )
        {
            //std::cout<<val_u_nx[i]<<std::endl;
            interface_file5 << (*interface_point).x() << "   " <<  (*interface_point).y() << "   "
            << velocity_interface[i].first << "   " << velocity_interface[i].second<< std::endl;

            i++;

        }

        interface_file5.close();
    }
    else
        std::cerr << "File 'vec_u_n' has not been opened" << std::endl;

     */
//    std::string filename_stokes6 = "velocity_field_" + std::to_string(time_step) + ".3D";
        std::string filename_stokes6 = "velocity_field_.3D";
        std::ofstream interface_file6(filename_stokes6, std::ios::out | std::ios::trunc);

        if (interface_file6) {
// instructions
            interface_file6 << "X   Y   val0   val1" << std::endl;
            size_t i = 0;
            for (auto point_vel = points_vel_field.begin(); point_vel < points_vel_field.end(); point_vel++) {
//std::cout<<val_u_nx[i]<<std::endl;
                interface_file6 << (*point_vel).first << "   " << (*point_vel).second << "   "
                                << velocity_field[i].first << "   " << velocity_field[i].second << std::endl;

                i++;

            }

            interface_file6.close();
        } else
            std::cerr << "File 'vec_u_n' has not been opened" << std::endl;
/*
    std::string filename_stokes7 = "vec_normal_" + std::to_string(time_step) + ".3D";
    std::ofstream interface_file7(filename_stokes7, std::ios::out | std::ios::trunc);

    if(interface_file7)
    {
        // instructions
        interface_file7 << "X   Y   val0   val1" << std::endl;
        size_t i = 0;
        for(auto interface_point = interface_points.begin() ; interface_point < interface_points.end() ; interface_point++ )
        {
            //std::cout<<val_u_nx[i]<<std::endl;
            interface_file7 << (*interface_point).x() << "   " <<  (*interface_point).y() << "   "
            << vec_u_n[i].first << "   " << vec_u_n[i].second<< std::endl;

            i++;

        }

        interface_file7.close();
    }
    else
        std::cerr << "File 'vec_normal_' has not been opened" << std::endl;
    */
    }

    template<typename Mesh, typename VEC, typename T>
    void
    goal_quantities_time(const Mesh &msh, T time, const VEC &interface_points, const std::vector <T> &val_u_nx,
                         const std::vector <T> &val_u_ny, const std::vector <T> &val_u_n,
                         const std::vector <std::pair<T, T>> &vec_u_n) {

        std::string filename_stokes0 = "val_u_nx_" + std::to_string(time) + ".3D";
        std::ofstream interface_file0(filename_stokes0, std::ios::out | std::ios::trunc);

        if (interface_file0) {
// instructions
            interface_file0 << "X   Y   Z   val" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_points.begin();
                 interface_point < interface_points.end(); interface_point++) {
//std::cout<<val_u_nx[i]<<std::endl;
                interface_file0 << (*interface_point).x() << "   " << (*interface_point).y() << "   "
                                << val_u_nx[i] << "    0.0" << std::endl;

                i++;

            }

            interface_file0.close();
        } else
            std::cerr << "File 'val_u_nx' has not been opened" << std::endl;

        std::string filename_stokes1 = "val_u_ny_" + std::to_string(time) + ".3D";
        std::ofstream interface_file1(filename_stokes1, std::ios::out | std::ios::trunc);

        if (interface_file1) {
// instructions
            interface_file1 << "X   Y   Z   val" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_points.begin();
                 interface_point < interface_points.end(); interface_point++) {
//std::cout<<val_u_ny[i]<<std::endl;
                interface_file1 << (*interface_point).x() << "   " << (*interface_point).y() << "   "
                                << val_u_ny[i] << "    0.0" << std::endl;
                i++;
            }

            interface_file1.close();
        } else
            std::cerr << "File 'val_u_ny' has not been opened" << std::endl;


        std::string filename_stokes2 = "val_u_n_" + std::to_string(time) + ".3D";
        std::ofstream interface_file2(filename_stokes2, std::ios::out | std::ios::trunc);

        if (interface_file2) {
// instructions
            interface_file2 << "X   Y   Z   val" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_points.begin();
                 interface_point < interface_points.end(); interface_point++) {
//std::cout<<val_u_n[i]<<std::endl;
                interface_file2 << (*interface_point).x() << "   " << (*interface_point).y() << "   "
                                << val_u_n[i] << "    0.0" << std::endl;
                i++;
            }

            interface_file2.close();
        } else
            std::cerr << "File 'val_u_n' has not been opened" << std::endl;


        postprocess_output<double> postoutput_vec_u_n;
        std::string filename_stokes3 = "vec_u_n_" + std::to_string(time) + ".dat";
        auto test_vec_u_n = std::make_shared < gnuplot_output_object_vec < double > > (filename_stokes3);

        size_t i = 0;
        for (auto interface_point = interface_points.begin(); interface_point < interface_points.end(); interface_point++) {
            test_vec_u_n->add_data(*interface_point,
                                   std::make_pair(val_u_n[i] * vec_u_n[i].first, val_u_n[i] * vec_u_n[i].second));
            i++;
        }

        postoutput_vec_u_n.add_object(test_vec_u_n);
        postoutput_vec_u_n.write();


        std::string filename_stokes4 = "vec_u_n_" + std::to_string(time) + ".3D";
        std::ofstream interface_file4(filename_stokes4, std::ios::out | std::ios::trunc);

        if (interface_file4) {
// instructions
            interface_file4 << "X   Y   val0   val1" << std::endl;
            size_t i = 0;
            for (auto interface_point = interface_points.begin();
                 interface_point < interface_points.end(); interface_point++) {
//std::cout<<val_u_nx[i]<<std::endl;
                interface_file4 << (*interface_point).x() << "   " << (*interface_point).y() << "   "
                                << val_u_n[i] * vec_u_n[i].first << "   " << val_u_n[i] * vec_u_n[i].second << std::endl;

                i++;

            }

            interface_file4.close();
        } else
            std::cerr << "File 'vec_u_n' has not been opened" << std::endl;


    }




}

