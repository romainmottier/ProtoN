
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

#include "methods/cuthho_bits/cuthho_export.hpp"
#include "level_set_transport_problem.hpp"
using namespace level_set_transport;

namespace random_functions{



    template<typename T, typename Mesh>
    bool
    pt_in_cell(const Mesh &msh, const point<T, 2> &point_to_find, const typename Mesh::cell_type &cl) {
        auto pts = points(msh, cl);

//std::cout<<"Point to find "<<std::setprecision(15)<<point_to_find.x()<<", "<<point_to_find.y()<<std::endl;

// std::cout<<"Min x "<<std::setprecision(15)<<pts[0].x()<<", max x "<<pts[1].x()<<std::endl;

//std::cout<<"Min y "<<std::setprecision(15)<<pts[1].y()<<", max y "<<pts[2].y()<<std::endl;

        T epsilon = 1e-10;
        if ((pts[0].x() - epsilon) <= point_to_find.x() && (pts[1].x() + epsilon) >= point_to_find.x() &&
            (pts[1].y() - epsilon) <= point_to_find.y() && (pts[2].y() + epsilon) >= point_to_find.y())
            return TRUE;
        else
            return FALSE;

    }

    template<typename T, typename Mesh>
    bool
    pt_in_cell_approximated(const Mesh &msh, const point<T, 2> &point_to_find, const typename Mesh::cell_type &cl) {
        auto pts = points(msh, cl);

//std::cout<<"Point to find "<<std::setprecision(15)<<point_to_find.x()<<", "<<point_to_find.y()<<std::endl;

// std::cout<<"Min x "<<std::setprecision(15)<<pts[0].x()<<", max x "<<pts[1].x()<<std::endl;

//std::cout<<"Min y "<<std::setprecision(15)<<pts[1].y()<<", max y "<<pts[2].y()<<std::endl;

        T epsilon = 1e-3;
        if ((pts[0].x() - epsilon) <= point_to_find.x() && (pts[1].x() + epsilon) >= point_to_find.x() &&
            (pts[1].y() - epsilon) <= point_to_find.y() && (pts[2].y() + epsilon) >= point_to_find.y())
            return TRUE;
        else
            return FALSE;

    }

    template<typename T, typename Mesh>
    size_t
    pt_in_subcell(const Mesh &msh, const point<T, 2> &point_to_find, const typename Mesh::cell_type &agglocl) {

        for (auto &offset_subcells: agglocl.user_data.offset_subcells) {
//std::cout<<"OFFSET ORIGINAL CELL "<<offset_subcells<<std::endl;
            auto cl = msh.cells[offset_subcells];
            if (pt_in_cell(msh, point_to_find, cl)) // pt_in_cell
                return offset_subcells;
        }

        for (auto &offset_subcells: agglocl.user_data.offset_subcells) {
//std::cout<<"OFFSET ORIGINAL CELL "<<offset_subcells<<std::endl;
            auto cl = msh.cells[offset_subcells];
            if (pt_in_cell_approximated(msh, point_to_find, cl)) // pt_in_cell
                return offset_subcells;
        }
// IF IT ARRIVES HERE, IT DIDN'T FIND THE POINT IN THE CELL.
        std::cout << "the point did not find is " << point_to_find << std::endl;
        std::cout << "IT DIDN'T FIND THE POINT IN SUBCELL " << offset(msh, agglocl) << std::endl;
        std::cout << "CELL vertices:" << '\n';
        for (auto &pt: points(msh, agglocl))
            std::cout << " , pt = " << pt;
        std::cout << '\n' << std::endl;
        throw std::invalid_argument("Invalid point-> NOT IN AGGLO_CELL");

    }

    template<typename T, typename Mesh>
    size_t
    pt_in_subcell_APPROX(const Mesh &msh, const point<T, 2> &point_to_find, const typename Mesh::cell_type &agglocl) {

        for (auto &offset_subcells: agglocl.user_data.offset_subcells) {
//std::cout<<"OFFSET ORIGINAL CELL "<<offset_subcells<<std::endl;
            auto cl = msh.cells[offset_subcells];
            if (pt_in_cell_approximated(msh, point_to_find, cl))
                return offset_subcells;
        }
// IF IT ARRIVES HERE, IT DIDN'T FIND THE POINT IN THE CELL.
        std::cout << "the point did not find is " << point_to_find << std::endl;
        std::cout << "IT DIDN'T FIND THE POINT IN SUBCELL(sbagliata) " << offset(msh, agglocl) << std::endl;
        std::cout << "CELL vertices:" << '\n';
        for (auto &pt: points(msh, agglocl))
            std::cout << " , pt = " << pt;
        std::cout << '\n' << std::endl;
        throw std::invalid_argument("Invalid point-> NOT IN AGGLO_CELL");

    }

    template<typename T, typename Mesh>
    std::vector <size_t>
    pt_in_skeleton(const Mesh &msh, const point<T, 2> &point_to_find) {
        std::vector <size_t> cells_offset;
        for (auto &cl: msh.cells) {
            for (auto &offset_subcells: cl.user_data.offset_subcells) {
//std::cout<<"OFFSET ORIGINAL CELL "<<offset_subcells<<std::endl;
                auto cl = msh.cells[offset_subcells];
                if (pt_in_cell(msh, point_to_find, cl))
                    cells_offset.push_back(offset_subcells);

            }
        }
        return cells_offset;
// IF IT ARRIVES HERE, IT DIDN'T FIND THE POINT IN THE CELL.
//    if( cells_offset.size() == 2  )
//        return cells_offset ;
//    else{
//        std::cout<<"IT IS NOT A SKELETON POINT: pt = "<<point_to_find<<". DIDN'T FIND THE POINT IN AGGLO CELL: "<<offset(msh,agglocl)<<std::endl;
//        throw std::invalid_argument("Invalid point-> NOT IN AGGLO_CELL");
//    }

    }


    template<typename T, typename Mesh>
    std::vector <size_t>
    pt_in_subcell_skeleton(const Mesh &msh, const point<T, 2> &point_to_find, const typename Mesh::cell_type &agglocl) {
        std::vector <size_t> cells_offset;
        for (auto &offset_subcells: agglocl.user_data.offset_subcells) {
//std::cout<<"OFFSET ORIGINAL CELL "<<offset_subcells<<std::endl;
            auto cl = msh.cells[offset_subcells];
            if (pt_in_cell(msh, point_to_find, cl))
                cells_offset.push_back(offset_subcells);

        }
        return cells_offset;
// IF IT ARRIVES HERE, IT DIDN'T FIND THE POINT IN THE CELL.
//    if( cells_offset.size() == 2  )
//        return cells_offset ;
//    else{
//        std::cout<<"IT IS NOT A SKELETON POINT: pt = "<<point_to_find<<". DIDN'T FIND THE POINT IN AGGLO CELL: "<<offset(msh,agglocl)<<std::endl;
//        throw std::invalid_argument("Invalid point-> NOT IN AGGLO_CELL");
//    }

    }



    template<typename T>
    std::vector <point<T, 1>>
            reference_nodes(size_t);

    template<typename FonctionD, typename Mesh, typename FonctionA>
    void
    testing_level_set(const Mesh msh, const FonctionD &, const FonctionA &);

    template<typename FonctionD, typename Mesh>
    void
    test_new_method(const Mesh, const FonctionD &, const typename Mesh::cell_type &);


// Qualitative testing of the discrete level set function wrt the analytical one
    template<typename FonctionD, typename Mesh, typename FonctionA>
    void
    gradient_checking1(const Mesh msh, const FonctionD &level_set_disc, const FonctionA &level_set_anal,
                       const typename Mesh::cell_type &cl) {
//typedef typename Mesh::point_type       point_type;

        double derD1x, derD1y, derAx, derAy, derD2x, derD2y;
        Eigen::Matrix<double, 2, 1> derD1, derD2, derA;
        point<double, 2> node;


        auto pts = points(msh, cl);
        for (auto &node: pts) {
            derD1 = level_set_disc.gradient(node);
            derD2 = level_set_disc.gradient(node, msh, cl);
            derA = level_set_anal.gradient(node);

            derD1x = derD1(0);
            derD1y = derD1(1);
            derD2x = derD2(0);
            derD2y = derD2(1);
            derAx = derA(0);
            derAy = derA(1);


/*
            if((derD1x-derD2x)>1e-2)
            {
            std::cout<<"Differnce between two x-evaluation system "<<(derD1x-derD2x)<<std::endl;
            }
        */

            if ((derAx - derD2x) > 1e-2) {
                std::cout << "Differnce between analytic and NEW X-evaluation system " << (derAx - derD2x) << std::endl;
            }

            if ((derAx - derD1x) > 1e-2) {
                std::cout << "Differnce between analytic and OLD X-evaluation system " << (derAx - derD1x) << std::endl;
            }


/*
        if((derD1y-derD2y)>1e-2)
        {
            std::cout<<"Differnce between two y-evaluation system "<<(derD1y-derD2y)<<std::endl;
        }
         */


            if ((derAy - derD2y) > 1e-2) {
                std::cout << "Differnce between analytic and NEW Y-evaluation system " << (derAy - derD2y) << std::endl;
            }

            if ((derAy - derD1y) > 1e-2) {
                std::cout << "Differnce between analytic and OLD Y-evaluation system " << (derAy - derD1y) << std::endl;
            }


        }


    }

// Qualitative testing of the discrete level set function wrt the analytical one
    template<typename FonctionD, typename Mesh, typename FonctionA>
    void
    gradient_checking(const Mesh msh, const FonctionD &level_set_disc, const FonctionA &level_set_anal,
                      const typename Mesh::cell_type &cl) {
//typedef typename Mesh::point_type       point_type;

        double derD1x, derD1y, derAx, derAy; // , derD2x , derD2y ;
        Eigen::Matrix<double, 2, 1> derD1, derD2, derA;
        point<double, 2> node;

        auto pts = points(msh, cl);
        for (auto &node: pts) {
            derD1 = level_set_disc.gradient(node);
// derD2 = level_set_disc.gradient(node,msh,cl);
            derA = level_set_anal.gradient(node);

            derD1x = derD1(0);
            derD1y = derD1(1);
//   derD2x = derD2(0);
//   derD2y = derD2(1);
            derAx = derA(0);
            derAy = derA(1);


/*
            if((derD1x-derD2x)>1e-2)
            {
            std::cout<<"Differnce between two x-evaluation system "<<(derD1x-derD2x)<<std::endl;
            }
        */
/*
        if((derAx-derD2x)>1e-2)
        {
            std::cout<<"Differnce between analytic and NEW X-evaluation system "<<(derAx-derD2x)<<std::endl;
        }
        */
            if ((derAx - derD1x) > 1e-2) {
                std::cout << "Differnce between analytic and OLD X-evaluation system " << (derAx - derD1x) << std::endl;
            }


/*
        if((derD1y-derD2y)>1e-2)
        {
            std::cout<<"Differnce between two y-evaluation system "<<(derD1y-derD2y)<<std::endl;
        }
         */

/*
        if((derAy-derD2y)>1e-2)
        {
            std::cout<<"Differnce between analytic and NEW Y-evaluation system "<<(derAy-derD2y)<<std::endl;
        }
        */
            if ((derAy - derD1y) > 1e-2) {
                std::cout << "Differnce between analytic and OLD Y-evaluation system " << (derAy - derD1y) << std::endl;
            }


        }


    }


    template<typename FonctionD, typename Mesh, typename FonctionA>
    void
    testing_velocity(const Mesh &msh, const FonctionD &vel_disc, const FonctionA &vel_anal) {
//typedef typename Mesh::point_type       point_type;
        postprocess_output<double> postoutput1;

        Eigen::Matrix<double, 2, 1> valueA;
        point<double, 2> node;
        size_t N, M;
        N = 40;
        M = 40;
        auto test_discx = std::make_shared < gnuplot_output_object < double > > ("vel_discX.dat");
        auto test_analx = std::make_shared < gnuplot_output_object < double > > ("vel_analX.dat");
        auto test_discy = std::make_shared < gnuplot_output_object < double > > ("vel_discY.dat");
        auto test_analy = std::make_shared < gnuplot_output_object < double > > ("vel_analY.dat");

        for (auto &cl: msh.cells) {
            auto pts = points(msh, cl);
            for (auto &pt: pts) {
                auto valueD = vel_disc(pt, msh, cl);
                valueA = vel_anal(pt);

                test_discx->add_data(pt, valueD.first);
                test_discy->add_data(pt, valueD.second);
                test_analx->add_data(pt, valueA(0));
                test_analy->add_data(pt, valueA(1));


            }
        }

        postoutput1.add_object(test_discx);
        postoutput1.add_object(test_analx);
        postoutput1.add_object(test_discy);
        postoutput1.add_object(test_analy);


        postoutput1.write();

    }




    template<typename T>
    std::vector <point<T, 1>>
    reference_nodes(size_t degree) {
        auto comp_degree = degree + 1;

        size_t reqd_nodes = comp_degree;

        std::vector <point<T, 1>> ret;
        ret.reserve(reqd_nodes);

        point<T, 1> qp;
        T a1, a2;
        T delta_x;
        switch (reqd_nodes) {
            case 1:
                qp = point<T, 1>({
                                         0.0
                                 });
                ret.push_back(qp);
                return ret;

            case 2:
                qp = point<T, 1>({
                                         1.0
                                 });
                ret.push_back(-qp);
                ret.push_back(qp);
                return ret;

            case 3:
                qp = point<T, 1>({
                                         1.0
                                 });
                ret.push_back(-qp);
                ret.push_back(qp);
                qp = point<T, 1>({0.0});
                ret.push_back(qp);
                return ret;

            case 4:
                a1 = 1.0 / 3.0;
                qp = point<T, 1>({1.0});
                ret.push_back(-qp);
                ret.push_back(qp);
                qp = point<T, 1>({a1});
                ret.push_back(-qp);
                ret.push_back(qp);
                return ret;

            case 5:
// Be carefull in what order data is inserted in ret!
// In Gauss Legendre the first one was 0.0, now is the last one
                a2 = 0.5;
                a1 = 1.0;
                qp = point<T, 1>({a1});
                ret.push_back(-qp);
                ret.push_back(qp);

                qp = point<T, 1>({a2});
                ret.push_back(-qp);
                ret.push_back(qp);

                qp = point<T, 1>({0.0});
                ret.push_back(qp);

                return ret;

            default:

                delta_x = 1.0 / degree;
                a1 = 1.0;
                while (a1 > 0) {
                    qp = point<T, 1>({
                                             a1
                                     });
                    ret.push_back(-qp);
                    ret.push_back(qp);
                    a1 -= delta_x;

                }
                if (a1 == 0) {
                    qp = point<T, 1>({
                                             0.0
                                     });
                    ret.push_back(qp);
                }
                return ret;
        }

        return ret;
    }

    template<typename T, typename Mesh>
    std::vector <point<T, 2>>
    equidistriduted_nodes(const Mesh &msh,
                          const typename Mesh::cell_type &cl,
                          size_t degree) {
        typedef typename Mesh::point_type point_type;

        auto qps = reference_nodes<T>(degree);


        auto pts = points(msh, cl);

        auto v0 = pts[1] - pts[0];
        auto v1 = pts[2] - pts[1];
        auto v2 = pts[3] - pts[2];
        auto v3 = pts[3] - pts[0];

        std::vector <point<T, 2>> ret;

        auto P = [&](T xi, T eta) -> T {
            return 0.25 * pts[0].x() * (1 - xi) * (1 - eta) +
                   0.25 * pts[1].x() * (1 + xi) * (1 - eta) +
                   0.25 * pts[2].x() * (1 + xi) * (1 + eta) +
                   0.25 * pts[3].x() * (1 - xi) * (1 + eta);
        };

        auto Q = [&](T xi, T eta) -> T {
            return 0.25 * pts[0].y() * (1 - xi) * (1 - eta) +
                   0.25 * pts[1].y() * (1 + xi) * (1 - eta) +
                   0.25 * pts[2].y() * (1 + xi) * (1 + eta) +
                   0.25 * pts[3].y() * (1 - xi) * (1 + eta);
        };

        for (auto jtor = qps.begin(); jtor != qps.end(); jtor++) {
            for (auto itor = qps.begin(); itor != qps.end(); itor++) {
                auto qp_x = *itor;
                auto qp_y = *jtor;

                auto xi = qp_x.x();
                auto eta = qp_y.x();

                auto px = P(xi, eta);
                auto py = Q(xi, eta);

                ret.push_back(point_type(px, py));
            }
        }

        return ret;
    }


// Qualitative testing of the discrete level set function wrt the analytical one
    template<typename Fonction, typename Mesh>
    void
    testing_velocity_field_L2projected(const Mesh msh, const Fonction &vel) {
//typedef typename Mesh::point_type       point_type;
        postprocess_output<double> postoutput1;

        auto test_discx = std::make_shared < gnuplot_output_object < double > > ("L2vel_HHOX.dat");
        auto test_discy = std::make_shared < gnuplot_output_object < double > > ("L2vel_HHOY.dat");


        for (auto &cl: msh.cells) {
            auto pts = equidistriduted_nodes_ordered_bis<double, Mesh>(msh, cl, vel.degree_FEM);
            for (auto &pt: pts) {
                auto value = vel(pt, msh, cl);
                test_discx->add_data(pt, value.first);
                test_discy->add_data(pt, value.second);


            }
        }

        postoutput1.add_object(test_discx);
        postoutput1.add_object(test_discy);


        postoutput1.write();

    }

    template<typename Fonction, typename Mesh>
    void
    testing_velocity_field(const Mesh msh, const Fonction &vel) {

        std::string filename_interface_Stokes = "vel_lagrange_disc.3D";

        std::ofstream interface_file(filename_interface_Stokes, std::ios::out | std::ios::trunc);

        if (interface_file) {
// instructions
            interface_file << "X   Y   val0   val1" << std::endl;


        } else
            std::cerr << "Interface_file has not been opened" << std::endl;


        for (auto &cl: msh.cells) {
            auto pts = equidistriduted_nodes_ordered_bis<double, Mesh>(msh, cl, vel.degree_FEM);
            for (auto &pt: pts) {
                auto value = vel(pt, msh, cl);
                interface_file << pt.x() << "   " << pt.y() << "   " << value.first << "   " << value.second << std::endl;


            }
        }

        interface_file.close();

    }


    template<typename Fonction, typename Mesh>
    void
    testing_velocity_field(const Mesh &msh, const Fonction &vel, size_t time_step, std::string &folder) {

        std::string filename_stokes6 = folder + "velocity_field_" + std::to_string(time_step) + ".3D";
        std::ofstream interface_file(filename_stokes6, std::ios::out | std::ios::trunc);

        if (interface_file) {
// instructions
            interface_file << "X   Y   val0   val1" << std::endl;


        } else
            std::cerr << "Interface_file has not been opened" << std::endl;


        for (auto &cl: msh.cells) {
            auto pts = equidistriduted_nodes_ordered_bis<double, Mesh>(msh, cl, vel.degree_FEM);
            for (auto &pt: pts) {
                auto value = vel(pt, msh, cl);
                interface_file << pt.x() << "   " << pt.y() << "   " << value.first << "   " << value.second << std::endl;


            }
        }

        interface_file.close();

    }

    template<typename FonctionD, typename Mesh, typename T>
    void
    testing_level_set_time(Mesh &msh, const FonctionD &level_set_disc, T time, size_t time_step, std::string &folder) {
        typedef typename Mesh::point_type point_type;


        postprocess_output <T> postoutput0;
        std::string filename_FEM = folder + "sol_FEM_t=" + std::to_string(time_step) + ".dat";
        auto test_FEM = std::make_shared < gnuplot_output_object < double > > (filename_FEM);



// CHECK MAX AND MIN
        T ret0 = -10.0;
        T ret1 = 10.0;
        for (auto &cl: msh.cells) {
            auto nodes = equidistriduted_nodes_ordered_bis<T, Mesh>(msh, cl, level_set_disc.degree_FEM);

            for (auto &nd: nodes) {
                auto new_ret = level_set_disc(nd, msh, cl);
                test_FEM->add_data(nd, new_ret);
                ret0 = std::max(new_ret, ret0);
                ret1 = std::min(new_ret, ret1);
            }


        }

        postoutput0.add_object(test_FEM);
        postoutput0.write();

        std::cout << "At initial time: min(phi) = " << level_set_disc.phi_min << ", max(phi) = " << level_set_disc.phi_max
                  << std::endl;

        std::cout << "At time t = " << time << ": min(phi) = " << ret1 << ", max(phi) = " << ret0 << std::endl;

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


    template<typename FonctionD, typename Mesh, typename T>
    void
    testing_level_set_max_min(const Mesh &msh, const FonctionD &level_set_disc, size_t time_step,
                              std::vector <std::pair<T, T>> &min_max_vec) {

// CHECK MAX AND MIN
        T ret0 = -10.0;
        T ret1 = 10.0;


        for (auto &cl: msh.cells) {

            auto nodes = equidistriduted_nodes_ordered_bis<T, Mesh>(msh, cl, level_set_disc.degree_FEM);

            for (auto &nd: nodes) {
                auto new_ret = level_set_disc(nd, msh, cl);
                ret0 = std::max(new_ret, ret0);
                ret1 = std::min(new_ret, ret1);
            }


        }

        min_max_vec.push_back(std::make_pair(ret1, ret0));

//std::cout<<"Initial time: MIN(phi) = "<<level_set_disc.phi_min<<", MAX(phi) = "<<level_set_disc.phi_max<< std::endl;
//std::cout<<"At time t = "<<time<<": MIN(phi) = "<<ret1<<" , MAX(phi) = "<<ret0<< std::endl;

    }



/// Useful to plot level set post FE transport problem
/// in cuthho_export.hpp
    template<typename Mesh, typename Function, typename T>
    void
    output_mesh_info2_time(const Mesh &msh, const Function &level_set_function, T time, size_t time_step,
                           std::string &folder) {
        using RealType = typename Mesh::coordinate_type;

/************** OPEN SILO DATABASE **************/
        silo_database silo;
        std::string filename_silo = folder + "cuthho_meshinfo_Stokes_" + std::to_string(time_step) + ".silo";
        silo.create(filename_silo);
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

        std::ofstream points_file(folder + "int_points.3D", std::ios::out | std::ios::trunc);

        if (points_file) {
// instructions
            points_file << "X   Y   Z   val" << std::endl;

            for (size_t i = 0; i < int_pts_x.size(); i++) {
                points_file << int_pts_x[i] << "   " << int_pts_y[i]
                            << "   0.0     0.0" << std::endl;
            }

            points_file.close();
        } else
            std::cerr << "Points_file has not been opened" << std::endl;


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

        std::string filename_interface_Stokes = folder + "interface_Stokes_" + std::to_string(time_step) + ".3D";

        std::ofstream interface_file(filename_interface_Stokes, std::ios::out | std::ios::trunc);

        if (interface_file) {
// instructions
            interface_file << "X   Y   Z   val" << std::endl;

            for (size_t i = 0; i < int_x.size(); i++) {
                interface_file << int_x[i] << "   " << int_y[i]
                               << "   0.0     0.0" << std::endl;
            }

            interface_file.close();
        } else
            std::cerr << "Interface_file has not been opened" << std::endl;
    }


/// Useful to plot level set post FE transport problem
/// in cuthho_export.hpp
    template<typename Mesh, typename Function>
    void
    output_mesh_info_ls_l_n(const Mesh &msh, const Function &ls) {
// Plotting of parametric interface (l,n)
// l = degree
// n = # of cells
        using RealType = typename Mesh::coordinate_type;



/*************  MAKE AN OUTPUT FOR THE INTERFACE *************/
        std::vector <RealType> int_x;
        std::vector <RealType> int_y;

        auto interface0 = ls.interface0;
        auto interface1 = ls.interface1;
        interface0.rows();
        auto nOfRow = interface0.rows();
        auto nOfCol = interface0.cols();
        for (auto i = 0; i < nOfRow; i++) {
            for (auto j = 0; j < nOfCol; j++) {
                RealType x = interface0(i, j);
                RealType y = interface1(i, j);

                int_x.push_back(x);
                int_y.push_back(y);

            }
        }


        std::string filename_interface_Stokes = "parametric_interface_Stokes.3D";

        std::ofstream interface_file(filename_interface_Stokes, std::ios::out | std::ios::trunc);

        if (interface_file) {
// instructions
            interface_file << "X   Y   Z   val" << std::endl;

            for (size_t i = 0; i < int_x.size(); i++) {
                interface_file << int_x[i] << "   " << int_y[i]
                               << "   0.0     0.0" << std::endl;
            }

            interface_file.close();
        } else
            std::cerr << "Interface_file has not been opened" << std::endl;


        std::vector <RealType> int_x1;
        std::vector <RealType> int_y1;


        for (auto &cl: msh.cells) {


            if (cl.user_data.location == element_location::ON_INTERFACE) {

                auto global_cells_i = ls.get_global_cells_interface(msh, cl);
                auto integration_msh = cl.user_data.integration_msh;


                auto qps_un = edge_quadrature<RealType>(100);

                for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                    auto pts = points(integration_msh, integration_msh.cells[i_cell]);
                    size_t global_cl_i = global_cells_i[i_cell];

                    for (auto &qp: qps_un) {
                        auto t = 0.5 * qp.first.x() + 0.5;

                        auto p = ls(t, global_cl_i);

                        int_x1.push_back(p(0));
                        int_y1.push_back(p(1));


                    }
                }
            }
        }


        std::string filename_interface_Stokes1 = "parametric_interface_Stokes_multipts.3D";

        std::ofstream interface_file1(filename_interface_Stokes1, std::ios::out | std::ios::trunc);

        if (interface_file1) {
// instructions
            interface_file1 << "X   Y   Z   val" << std::endl;

            for (size_t i = 0; i < int_x1.size(); i++) {
                interface_file1 << int_x1[i] << "   " << int_y1[i]
                                << "   0.0     0.0" << std::endl;
            }

            interface_file1.close();
        } else
            std::cerr << "Interface_file has not been opened" << std::endl;


    }

/// Useful to plot level set post FE transport problem
/// in cuthho_export.hpp
    template<typename Mesh, typename Function, typename T>
    void
    output_mesh_info2_time_fixed_mesh(const Mesh &msh, const Function &level_set_function, T time, size_t time_step) {
        using RealType = typename Mesh::coordinate_type;



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

        std::string filename_interface_Stokes = "interface_Stokes_" + std::to_string(time_step) + ".3D";

        std::ofstream interface_file(filename_interface_Stokes, std::ios::out | std::ios::trunc);

        if (interface_file) {
// instructions
            interface_file << "X   Y   Z   val" << std::endl;

            for (size_t i = 0; i < int_x.size(); i++) {
                interface_file << int_x[i] << "   " << int_y[i]
                               << "   0.0     0.0" << std::endl;
            }

            interface_file.close();
        } else
            std::cerr << "Interface_file has not been opened" << std::endl;
    }

/// Useful to plot level set post FE transport problem
/// in cuthho_export.hpp
    template<typename Mesh, typename Function>
    void
    output_mesh_info2(const Mesh &msh, const Function &level_set_function) {
        using RealType = typename Mesh::coordinate_type;

/************** OPEN SILO DATABASE **************/
        silo_database silo;
        silo.create("cuthho_meshinfo_Stokes.silo");
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

        std::ofstream points_file("int_points.3D", std::ios::out | std::ios::trunc);

        if (points_file) {
// instructions
            points_file << "X   Y   Z   val" << std::endl;

            for (size_t i = 0; i < int_pts_x.size(); i++) {
                points_file << int_pts_x[i] << "   " << int_pts_y[i]
                            << "   0.0     0.0" << std::endl;
            }

            points_file.close();
        } else
            std::cerr << "Points_file has not been opened" << std::endl;


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
        std::ofstream interface_file("interface_Stokes.3D", std::ios::out | std::ios::trunc);

        if (interface_file) {
// instructions
            interface_file << "X   Y   Z   val" << std::endl;

            for (size_t i = 0; i < int_x.size(); i++) {
                interface_file << int_x[i] << "   " << int_y[i]
                               << "   0.0     0.0" << std::endl;
            }

            interface_file.close();
        } else
            std::cerr << "Interface_file has not been opened" << std::endl;
    }

/// New version of find_zero_crossing for discret level set functions
/// in cuthho_geom.hpp
    template<typename T, typename Function, typename Mesh>
    point<T, 2>
    find_zero_crossing_on_face(const point<T, 2> &p0, const point<T, 2> &p1, const Function &level_set_function,
                               const T &threshold, const Mesh &msh, const typename Mesh::face_type &fc) {
/* !!! We assume that the level set function *has* a zero crossing
     * between p0 and p1 !!! */

// I SHOULD CHECK THAT pm IS ALWAYS ON THE FACE p0-p1 ???????
        auto pa = p0;
        auto pb = p1;
        auto pm = (pa + pb) / 2.0;
        auto pm_prev = pm;

        T x_diff_sq, y_diff_sq;

/* A threshold of 1/10000 the diameter of the element is considered
     * acceptable. Since with 24 iterations we reduce the error by 16384
     * and the worst case is that the two points are at the opposite sides
     * of the element, we put 30 as limit. */
        size_t max_iter = 50;

        do {
            auto la = level_set_function(pa, msh, fc);
            auto lb = level_set_function(pb, msh, fc);
            auto lm = level_set_function(pm, msh, fc);

            if ((lb >= 0 && lm >= 0) || (lb < 0 && lm < 0)) {   /* intersection is between pa and pm */
                pm_prev = pm;
                pb = pm;
                pm = (pa + pb) / 2.0;
            } else {   /* intersection is between pm and pb */
                pm_prev = pm;
                pa = pm;
                pm = (pa + pb) / 2.0;
            }

            x_diff_sq = (pm_prev.x() - pm.x()) * (pm_prev.x() - pm.x());
            y_diff_sq = (pm_prev.y() - pm.y()) * (pm_prev.y() - pm.y());

        } while ((sqrt(x_diff_sq + y_diff_sq) > threshold) && max_iter--);

        return pm;

/* Affine zero crossing was like that: */
//auto t = l0/(l0-l1);
//auto ip = (pts[1] - pts[0]) * t + pts[0];
    }

/// New version of find_zero_crossing for discret level set functions
/// in cuthho_geom.hpp
    template<typename T, typename Function, typename Mesh>
    point<T, 2>
    find_zero_crossing_on_face3(const point<T, 2> &p0, const point<T, 2> &p1, const Function &level_set_function,
                                const T &threshold, const Mesh &msh, const typename Mesh::face_type &fc) {
/* !!! We assume that the level set function *has* a zero crossing
     * between p0 and p1 !!! */

// I SHOULD CHECK THAT pm IS ALWAYS ON THE FACE p0-p1 ???????
        auto pa = p0;
        auto pb = p1;
        auto pm = (pa + pb) / 2.0;
        auto pm_prev = pm;
        T iso_val_interface = level_set_function.iso_val_interface;
        T x_diff_sq, y_diff_sq;

/* A threshold of 1/10000 the diameter of the element is considered
     * acceptable. Since with 24 iterations we reduce the error by 16384
     * and the worst case is that the two points are at the opposite sides
     * of the element, we put 30 as limit. */
        size_t max_iter = 50;

        do {
//auto la = level_set_function(pa,msh,fc);
            auto lb = level_set_function(pb, msh, fc);
            auto lm = level_set_function(pm, msh, fc);

            if ((lb >= iso_val_interface && lm >= iso_val_interface) ||
                (lb < iso_val_interface && lm < iso_val_interface)) {   /* intersection is between pa and pm */
                pm_prev = pm;
                pb = pm;
                pm = (pa + pb) / 2.0;
            } else {   /* intersection is between pm and pb */
                pm_prev = pm;
                pa = pm;
                pm = (pa + pb) / 2.0;
            }

            x_diff_sq = (pm_prev.x() - pm.x()) * (pm_prev.x() - pm.x());
            y_diff_sq = (pm_prev.y() - pm.y()) * (pm_prev.y() - pm.y());

        } while ((sqrt(x_diff_sq + y_diff_sq) > threshold) && max_iter--);

        return pm;

/* Affine zero crossing was like that: */
//auto t = l0/(l0-l1);
//auto ip = (pts[1] - pts[0]) * t + pts[0];
    }

/// New version of find_zero_crossing for discret level set functions
/// in cuthho_geom.hpp
    template<typename T, typename Function, typename Mesh>
    point<T, 2>
    find_zero_crossing_in_cell(const point<T, 2> &p0, const point<T, 2> &p1, const Function &level_set_function,
                               const T &threshold, const Mesh &msh, const typename Mesh::cell_type &cl) {
/* !!! We assume that the level set function *has* a zero crossing
     * between p0 and p1 !!! */

// I SHOULD CHECK THAT pm IS ALWAYS IN THE CELL ???????
        auto pa = p0;
        auto pb = p1;
        auto pm = (pa + pb) / 2.0;
        auto pm_prev = pm;

        T x_diff_sq, y_diff_sq;

/* A threshold of 1/10000 the diameter of the element is considered
     * acceptable. Since with 24 iterations we reduce the error by 16384
     * and the worst case is that the two points are at the opposite sides
     * of the element, we put 30 as limit. */
        size_t max_iter = 50; // ERA 50, METTO 100

        do {
            auto la = level_set_function(pa, msh, cl);
            auto lb = level_set_function(pb, msh, cl);
            auto lm = level_set_function(pm, msh, cl);

            if ((lb >= 0 && lm >= 0) || (lb < 0 && lm < 0)) {   /* intersection is between pa and pm */
                pm_prev = pm;
                pb = pm;
                pm = (pa + pb) / 2.0;
            } else {   /* intersection is between pm and pb */
                pm_prev = pm;
                pa = pm;
                pm = (pa + pb) / 2.0;
            }

            x_diff_sq = (pm_prev.x() - pm.x()) * (pm_prev.x() - pm.x());
            y_diff_sq = (pm_prev.y() - pm.y()) * (pm_prev.y() - pm.y());

        } while ((sqrt(x_diff_sq + y_diff_sq) > threshold) && max_iter--);

        return pm;

/* Affine zero crossing was like that: */
//auto t = l0/(l0-l1);
//auto ip = (pts[1] - pts[0]) * t + pts[0];
    }

/// New version of find_zero_crossing for discret level set functions
/// in cuthho_geom.hpp
    template<typename T, typename Function, typename Mesh>
    point<T, 2>
    find_zero_crossing_in_cell3(const point<T, 2> &p0, const point<T, 2> &p1, const Function &level_set_function,
                                const T &threshold, const Mesh &msh, const typename Mesh::cell_type &cl) {
/* !!! We assume that the level set function *has* a zero crossing
     * between p0 and p1 !!! */

// I SHOULD CHECK THAT pm IS ALWAYS IN THE CELL ???????
        auto pa = p0;
        auto pb = p1;
        auto pm = (pa + pb) / 2.0;
        auto pm_prev = pm;
        T iso_val_interface = level_set_function.iso_val_interface;
        T x_diff_sq, y_diff_sq;

/* A threshold of 1/10000 the diameter of the element is considered
     * acceptable. Since with 24 iterations we reduce the error by 16384
     * and the worst case is that the two points are at the opposite sides
     * of the element, we put 30 as limit. */
        size_t max_iter = 50; // ERA 50, METTO 100

        do {
//auto la = level_set_function(pa,msh,cl);
            auto lb = level_set_function(pb, msh, cl);
            auto lm = level_set_function(pm, msh, cl);

            if ((lb >= iso_val_interface && lm >= iso_val_interface) ||
                (lb < iso_val_interface && lm < iso_val_interface)) {   /* intersection is between pa and pm */
                pm_prev = pm;
                pb = pm;
                pm = (pa + pb) / 2.0;
            } else {   /* intersection is between pm and pb */
                pm_prev = pm;
                pa = pm;
                pm = (pa + pb) / 2.0;
            }

            x_diff_sq = (pm_prev.x() - pm.x()) * (pm_prev.x() - pm.x());
            y_diff_sq = (pm_prev.y() - pm.y()) * (pm_prev.y() - pm.y());

        } while ((sqrt(x_diff_sq + y_diff_sq) > threshold) && max_iter--);

        return pm;

/* Affine zero crossing was like that: */
//auto t = l0/(l0-l1);
//auto ip = (pts[1] - pts[0]) * t + pts[0];
    }


/// New version of detect_node_position for discret level functions
/// in cuthho_geom.hpp
    template<typename T, size_t ET, typename Function>
    void
    detect_node_position2(cuthho_mesh <T, ET> &msh, const Function &level_set_function) {
        for (auto &n: msh.nodes) {
//auto pt = points(msh, n); //deleted by Stefano
//if ( level_set_function(pt) < 0 ) //deleted by Stefano

            T value_node = level_set_function(n);
            if (std::abs(value_node) < 1e-17) {
                std::cout << "In detect_node_position2 -> ATTENTION, INTERFACE ON A NODE!" << std::endl;
                auto pt = points(msh, n);
                value_node = level_set_function(pt);
            }


            if (value_node < 0) // add by Stefano
                n.user_data.location = element_location::IN_NEGATIVE_SIDE;
            else
                n.user_data.location = element_location::IN_POSITIVE_SIDE;


        }
    }


/// New version of detect_node_position for discret level functions -> USING interface = 1/2
/// in cuthho_geom.hpp
    template<typename T, size_t ET, typename Function>
    void
    detect_node_position3(cuthho_mesh <T, ET> &msh, const Function &level_set_function) {

        timecounter tc;
        tc.tic();
        T iso_val_interface = level_set_function.iso_val_interface;
//std::cout<<"In 'detect_node' -> iso_val_interface = "<<iso_val_interface<<std::endl;

        for (auto &n: msh.nodes) {
//auto pt = points(msh, n); //deleted by Stefano
//if ( level_set_function(pt) < 0 ) //deleted by Stefano

            T value_node = level_set_function(n);

            if (std::abs(value_node - iso_val_interface) < 1e-17) {
                std::cout << "In detect_node_position3 -> ATTENTION, INTERFACE ON A NODE!" << std::endl;
                auto pt = points(msh, n);
                value_node = level_set_function(pt);
            }


            if (value_node < iso_val_interface) { // add by Stefano
                n.user_data.location = element_location::IN_NEGATIVE_SIDE;
//std::cout<<"n.user_data.location = IN_NEGATIVE_SIDE"<<std::endl;
            } else {
                n.user_data.location = element_location::IN_POSITIVE_SIDE;
//std::cout<<"n.user_data.location = IN_POSITIVE_SIDE"<<std::endl;
            }
//std::cout<<"Value_node = "<< value_node<<std::endl;

        }

        tc.toc();
//std::cout << bold << yellow << "detect_node_position3, time resolution: " << tc << " seconds" << reset << std::endl;
    }

/// New version of detect_node_position for discret level functions -> USING interface = 1/2
/// in cuthho_geom.hpp
/*
template<typename T, size_t ET, typename Function>
void
detect_node_position3_parallel(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    timecounter tc ;
    tc.tic();
    T iso_val_interface = level_set_function.iso_val_interface ;
    std::cout<<"In 'detect_node' -> iso_val_interface = "<<iso_val_interface<<std::endl;
#ifdef HAVE_INTEL_TBB
    size_t n_nodes = msh.nodes.size();
    std::cout<<" I m in parallel zone"<<std::endl;
    tbb::parallel_for(size_t(0), size_t(n_nodes), size_t(1),
    [&msh,&level_set_function,&iso_val_interface] (size_t & cell_ind){
        auto& n = msh.nodes[cell_ind];
        T value_node = level_set_function(n);

        if( std::abs(value_node - iso_val_interface) <  1e-17 ){
            std::cout<<"In detect_node_position2 -> ATTENTION, INTERFACE ON A NODE!"<<std::endl;
            auto pt = points(msh, n);
            value_node = level_set_function(pt) ;
        }



        if ( value_node < iso_val_interface ){ // add by Stefano
            n.user_data.location = element_location::IN_NEGATIVE_SIDE;
                //std::cout<<"n.user_data.location = IN_NEGATIVE_SIDE"<<std::endl;
        }
        else{
            n.user_data.location = element_location::IN_POSITIVE_SIDE;
                //std::cout<<"n.user_data.location = IN_POSITIVE_SIDE"<<std::endl;
        }

    }
    );

    tc.toc();
    //std::cout << "detect_node_position3_parallel, time resolution: " << tc << " seconds" << std::endl;
#else

    for (auto& n : msh.nodes)
    {
        //auto pt = points(msh, n); //deleted by Stefano
        //if ( level_set_function(pt) < 0 ) //deleted by Stefano

        T value_node = level_set_function(n);

        if( std::abs(value_node - iso_val_interface) <  1e-17 ){
            std::cout<<"In detect_node_position2 -> ATTENTION, INTERFACE ON A NODE!"<<std::endl;
            auto pt = points(msh, n);
            value_node = level_set_function(pt) ;
        }


        //std::cout<<"value_node SEQUENTIAL = "<<level_set_function(n)<<std::endl;
        if ( value_node < iso_val_interface ){ // add by Stefano
            n.user_data.location = element_location::IN_NEGATIVE_SIDE;
            //std::cout<<"n.user_data.location = IN_NEGATIVE_SIDE"<<std::endl;
        }
        else{
            n.user_data.location = element_location::IN_POSITIVE_SIDE;
            //std::cout<<"n.user_data.location = IN_POSITIVE_SIDE"<<std::endl;
        }
        //std::cout<<"Value_node = "<< value_node<<std::endl;

    }


#endif

}
*/
/// New version of detect_cut_faces for discret level functions
/// in cuthho_geom.hpp
    template<typename T, size_t ET, typename Function>
    void
    detect_cut_faces2(cuthho_mesh <T, ET> &msh, const Function &level_set_function) {
        for (auto &fc: msh.faces) {
            auto pts = points(msh, fc);
/*
        if( (pts[0].x() == 0.375) && (pts[0].y() == 0.5) )
        {
            std::cout<<"CASE pt0!!"<<std::endl;
            std::cout<<"AND  pts[1] = "<<pts[1]<<std::endl;
            std::cout<<"level_set_function(pts[0]) = "<<level_set_function(pts[0])<< " , level_set_function(pts[1]) = "<<level_set_function(pts[1])<<std::endl;
             std::cout<<"level_set_function(pts[0],msh,fc) = "<<level_set_function(pts[0],msh,fc)<< " , level_set_function(pts[1],msh,fc) = "<<level_set_function(pts[1],msh,fc)<<'\n'<<std::endl;


        }
        if( (pts[1].x() == 0.375) && (pts[1].y() == 0.5) )
        {
            std::cout<<"CASE pt1!!"<<std::endl;
            std::cout<<"AND  pts[0] = "<<pts[0]<<std::endl;
            std::cout<<"level_set_function(pts[0]) = "<<level_set_function(pts[0])<< " , level_set_function(pts[1]) = "<<level_set_function(pts[1])<<std::endl;
             std::cout<<"level_set_function(pts[0],msh,fc) = "<<level_set_function(pts[0],msh,fc)<< " , level_set_function(pts[1],msh,fc) = "<<level_set_function(pts[1],msh,fc)<<'\n'<<std::endl;


        }
        */
//auto l0 = level_set_function(pts[0]);      //deleted by Stefano
//auto l1 = level_set_function(pts[1]);       //deleted by Stefano

            auto l0 = level_set_function(pts[0], msh, fc);      // add by Stefano
            auto l1 = level_set_function(pts[1], msh, fc);       // add by Stefano

// In the case doubt, I don't care the value, just the sign and I assign the same sign of the other. JUST ONE OF THE TWO SHOULD BE SO SMALL
            if ((std::abs(l0) < 1e-17) && (std::abs(l1) < 1e-17))
                std::cout << "STOP --> CASE DOUBT: l0 = " << l0 << " and l1 = " << l1 << std::endl;

            else if (std::abs(l0) < 1e-17) {
                std::cout << "The node " << pts[0] << " is very close to the interface." << std::endl;
                l0 = level_set_function(pts[0]);
                std::cout << "l0 = " << l0 << " ,l1 = " << l1 << '\n' << std::endl;
            } else if (std::abs(l1) < 1e-17) {
                std::cout << "The node " << pts[1] << " is very close to the interface." << std::endl;
                l1 = level_set_function(pts[1]);
                std::cout << "l0 = " << l0 << " ,l1 = " << l1 << '\n' << std::endl;
            }
/*
        if( ((pts[1].x() == 0.375) && (pts[1].y() == 0.5) ) ||(pts[0].x() == 0.375) && (pts[0].y() == 0.5))
        {
        std::cout<<"l0 = "<<l0<< " ,l1 = "<<l1<<'\n'<<std::endl;
        }
        */
            if (l0 >= 0 && l1 >= 0) {
                fc.user_data.location = element_location::IN_POSITIVE_SIDE;
                continue;
            }

            if (l0 < 0 && l1 < 0) {
                fc.user_data.location = element_location::IN_NEGATIVE_SIDE;
                continue;
            }


            auto threshold = diameter(msh, fc) / 1e20;
//auto pm = find_zero_crossing(pts[0], pts[1], level_set_function, threshold);
            auto pm = find_zero_crossing_on_face(pts[0], pts[1], level_set_function, threshold, msh, fc);

/* If node 0 is in the negative region, mark it as node inside, otherwise mark node 1 */
            fc.user_data.node_inside = (l0 < 0) ? 0 : 1;
            fc.user_data.location = element_location::ON_INTERFACE;
            fc.user_data.intersection_point = pm;
        }
    }

/// New version of detect_cut_faces for discret level functions -> USING interface = 1/2
/// in cuthho_geom.hpp
    template<typename T, size_t ET, typename Function>
    void
    detect_cut_faces3(cuthho_mesh <T, ET> &msh, const Function &level_set_function) {
        T iso_val_interface = level_set_function.iso_val_interface;
//std::cout<<"In 'detect_cut_face3'--> iso_val_interface = "<<iso_val_interface<<std::endl;
        for (auto &fc: msh.faces) {
            auto pts = points(msh, fc);
/*
        if( (pts[0].x() == 0.375) && (pts[0].y() == 0.5) )
        {
            std::cout<<"CASE pt0!!"<<std::endl;
            std::cout<<"AND  pts[1] = "<<pts[1]<<std::endl;
            std::cout<<"level_set_function(pts[0]) = "<<level_set_function(pts[0])<< " , level_set_function(pts[1]) = "<<level_set_function(pts[1])<<std::endl;
             std::cout<<"level_set_function(pts[0],msh,fc) = "<<level_set_function(pts[0],msh,fc)<< " , level_set_function(pts[1],msh,fc) = "<<level_set_function(pts[1],msh,fc)<<'\n'<<std::endl;


        }
        if( (pts[1].x() == 0.375) && (pts[1].y() == 0.5) )
        {
            std::cout<<"CASE pt1!!"<<std::endl;
            std::cout<<"AND  pts[0] = "<<pts[0]<<std::endl;
            std::cout<<"level_set_function(pts[0]) = "<<level_set_function(pts[0])<< " , level_set_function(pts[1]) = "<<level_set_function(pts[1])<<std::endl;
             std::cout<<"level_set_function(pts[0],msh,fc) = "<<level_set_function(pts[0],msh,fc)<< " , level_set_function(pts[1],msh,fc) = "<<level_set_function(pts[1],msh,fc)<<'\n'<<std::endl;


        }
        */
//auto l0 = level_set_function(pts[0]);      //deleted by Stefano
//auto l1 = level_set_function(pts[1]);       //deleted by Stefano

            auto l0 = level_set_function(pts[0], msh, fc);      // add by Stefano
            auto l1 = level_set_function(pts[1], msh, fc);       // add by Stefano
//        std::cout<<"level_set_function(pts[0]) = "<<level_set_function(pts[0])<<" vs l0 = "<<l0<<std::endl;
//        std::cout<<"level_set_function(pts[1]) = "<<level_set_function(pts[1])<<" vs l1 = "<<l1<<'\n'<<std::endl;

// In the case doubt, I don't care the value, just the sign and I assign the same sign of the other. JUST ONE OF THE TWO SHOULD BE SO SMALL
            if ((std::abs(l0 - iso_val_interface) < 1e-17) && (std::abs(l1 - iso_val_interface) < 1e-17))
                std::cout << "STOP --> CASE DOUBT: l0 = " << l0 << " and l1 = " << l1 << std::endl;

            else if (std::abs(l0 - iso_val_interface) < 1e-17) {
                std::cout << "The node " << pts[0] << " is very close to the interface." << std::endl;
                l0 = level_set_function(pts[0]);
                std::cout << "l0 = " << l0 << " ,l1 = " << l1 << '\n' << std::endl;
            } else if (std::abs(l1 - iso_val_interface) < 1e-17) {
                std::cout << "The node " << pts[1] << " is very close to the interface." << std::endl;
                l1 = level_set_function(pts[1]);
                std::cout << "l0 = " << l0 << " ,l1 = " << l1 << '\n' << std::endl;
            }
/*
        if( ((pts[1].x() == 0.375) && (pts[1].y() == 0.5) ) ||(pts[0].x() == 0.375) && (pts[0].y() == 0.5))
        {
        std::cout<<"l0 = "<<l0<< " ,l1 = "<<l1<<'\n'<<std::endl;
        }
        */
            if (l0 >= iso_val_interface && l1 >= iso_val_interface) {
                fc.user_data.location = element_location::IN_POSITIVE_SIDE;
                continue;
            }

            if (l0 < iso_val_interface && l1 < iso_val_interface) {
                fc.user_data.location = element_location::IN_NEGATIVE_SIDE;
                continue;
            }


            auto threshold = diameter(msh, fc) / 1e20;
//auto pm = find_zero_crossing(pts[0], pts[1], level_set_function, threshold);
            auto pm = find_zero_crossing_on_face3(pts[0], pts[1], level_set_function, threshold, msh, fc);
//std::cout<<"pm = "<<pm<< " and level_set_function = "<<level_set_function(pm,msh,fc)<<std::endl;
/* If node 0 is in the negative region, mark it as node inside, otherwise mark node 1 */
            fc.user_data.node_inside = (l0 < iso_val_interface) ? 0 : 1;
            fc.user_data.location = element_location::ON_INTERFACE;
            fc.user_data.intersection_point = pm;
        }
    }

/// New version of detect_cut_cells for discret level functions
/// in cuthho_geom.hpp
    template<typename T, size_t ET, typename Function>
    void
    detect_cut_cells2(cuthho_mesh <T, ET> &msh, const Function &level_set_function) {
        std::cout << "I AM IN DETECT CUT CELL2!!!!" << std::endl;
//typedef typename cuthho_mesh<T, ET>::face_type  face_type;
        typedef typename cuthho_mesh<T, ET>::point_type point_type;
//typedef typename cuthho_mesh<T, ET>::cell_type cell_type;

        size_t cell_i = 0;
        for (auto &cl: msh.cells) {
            auto fcs = faces(msh, cl);
            std::array<std::pair<size_t, point_type>, 2> cut_faces;

            size_t k = 0;
            for (size_t i = 0; i < fcs.size(); i++) {
                if (is_cut(msh, fcs[i]))
                    cut_faces.at(k++) = std::make_pair(i, fcs[i].user_data.intersection_point);
            }

/* If a face is cut, the cells that own the face are cut. Is this
         * unconditionally true? It should...fortunately this isn't avionics
         * software */

            if (k == 0) {

                auto is_positive = [&](const point_type &pt) -> bool {
                    return level_set_function(pt) > 0;
                };


                auto pts = points(msh, cl);

                if (std::all_of(pts.begin(), pts.end(), is_positive))
                    cl.user_data.location = element_location::IN_POSITIVE_SIDE;
                else
                    cl.user_data.location = element_location::IN_NEGATIVE_SIDE;




/*
            auto pts = points(msh, cl);
            auto pt = pts.begin();
            size_t counter = 0;
            while( ( pt!= pts.end() ) && ( level_set_function(*pt,msh,cl) > 0 ) )
            {
                counter++;
                pt++;

            }

            if ( counter == pts.size() )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
            */

            }
//MODIFICARE QUAAAA
            if (k == 2) {
                cl.user_data.location = element_location::ON_INTERFACE;
                auto p0 = cut_faces[0].second;
                auto p1 = cut_faces[1].second;
                auto pt = p1 - p0;
                auto pn = p0 + point<T, 2>(-pt.y(), pt.x());
                auto pn_prova = (p0 + p1) / 2.0 + 0.5 * point<T, 2>(-pt.y(), pt.x());
//if(offset(msh,cl)== 119)
//    std::cout<<"p0 = "<<p0<< " , p1 ="<<p1<<std::endl;
// PRIMA ERA DA p0 ->   MODIFCATO, ora è pt  medio!
/*
            if( !pt_in_cell(msh, pn, cl) )
            {
                std::cout<<"I chose another pn to ordering interface_points in 'detect_cut_cells2'."<<std::endl;
                T m_half = ( ps1.y() - pm.y() )/( ps1.x() - pm.x() );
                T q = pm.y() - m_half * pm.x() ;
                auto pt_bdry = search_boundary( msh , cl , pm , m_half , q , lm , level_set_function ) ;
                auto lm_bdry = level_set_function( pt_bdry , msh , cl );
            }
            */
                if (offset(msh, cl) == 137 || offset(msh, cl) == 138 || offset(msh, cl) == 134 || offset(msh, cl) == 103) {
                    std::cout << yellow << bold << "offset(msh,cl) = " << offset(msh, cl) << reset << std::endl;
                    auto pn_bis = (p0 + p1) / 2.0 + point<T, 2>(-pt.y(), pt.x());
                    std::cout << "pn_bis = " << pn_bis << " , level_set_function(pn_bis,msh,cl) ="
                              << level_set_function(pn_bis, msh, cl) << std::endl;
                    auto pn_bis0 = (p0 + p1) / 2.0 + 0.5 * point<T, 2>(-pt.y(), pt.x());
                    std::cout << "pn_bis0 = " << pn_bis0 << " , level_set_function(pn_bis0,msh,cl) ="
                              << level_set_function(pn_bis0, msh, cl) << std::endl;
                    auto pn_bis1 = p0 + 0.5 * point<T, 2>(-pt.y(), pt.x());
                    std::cout << "pn_bis1 = " << pn_bis1 << " , level_set_function(pn_bis1,msh,cl) ="
                              << level_set_function(pn_bis1, msh, cl) << '\n' << std::endl;

                    std::cout << "pn = " << pn << " , p0 = " << p0 << " , p1 = " << p1 << std::endl;
                    std::cout << "level_set_function(pn,msh,cl) = " << level_set_function(pn, msh, cl)
                              << " , level_set_function(p0,msh,cl) = " << level_set_function(p0, msh, cl)
                              << " , level_set_function(pn,msh,cl) = " << level_set_function(p1, msh, cl) << std::endl;
                    std::cout << "p0 - point<T,2>(-pt.y(), pt.x()) = " << p0 - point<T, 2>(-pt.y(), pt.x())
                              << " , level_set_function(p0 - point<T,2>(-pt.y(), pt.x()),msh,cl) = "
                              << level_set_function(p0 - point<T, 2>(-pt.y(), pt.x()), msh, cl) << std::endl;
                }


                if (!(std::signbit(level_set_function(pn, msh, cl)) ==
                      std::signbit(level_set_function(pn_prova, msh, cl)))) {
                    pn = pn_prova;
                    std::cout << "pn = " << pn << " , pn_prova = " << pn_prova << " , level_set_function(pn,msh,cl) = "
                              << level_set_function(pn, msh, cl) << " , level_set_function(pn_prova,msh,cl) = "
                              << level_set_function(pn_prova, msh, cl) << std::endl;
                }

                if (level_set_function(pn, msh, cl) >= 0) {
                    cl.user_data.p0 = p1;
                    cl.user_data.p1 = p0;
                } else {
                    cl.user_data.p0 = p0;
                    cl.user_data.p1 = p1;
                }

                cl.user_data.interface.push_back(cl.user_data.p0);
                cl.user_data.interface.push_back(cl.user_data.p1);
            }

            if (k != 0 && k != 2) {
                auto pts = points(msh, cl);
                std::cout << "Point[0] = " << pts[0] << " , point[1] = " << pts[1] << " , point[2] = " << pts[2]
                          << " , point[3] = " << pts[3] << std::endl;
                std::cout << "level_set_function(p0) = " << level_set_function(pts[0], msh, cl)
                          << " , level_set_function(p1) = " << level_set_function(pts[1], msh, cl)
                          << " , level_set_function(p2) = " << level_set_function(pts[2], msh, cl)
                          << " , level_set_function(p3) = " << level_set_function(pts[3], msh, cl) << std::endl;
                for (size_t i = 0; i < fcs.size(); i++) {
                    if (is_cut(msh, fcs[i]))
                        std::cout << "fcs[i].user_data.intersection_point = " << fcs[i].user_data.intersection_point
                                  << std::endl;
                }

                std::cout << "ERROR: in cut cell " << cell_i << " there are k = " << k << " cuts!!!!" << std::endl;
                throw std::logic_error(" --> Invalid number of cuts in cell");

            }

            cell_i++;
        }
    }

/// New version of detect_cut_cells for discret level functions -> USING INTERFACE = 1/2
/// in cuthho_geom.hpp
    template<typename T, size_t ET, typename Function>
    void
    detect_cut_cells3(cuthho_mesh <T, ET> &msh, const Function &level_set_function) {
//std::cout<<"I AM IN DETECT CUT CELL3!!!!"<<std::endl;
        timecounter tc;
        tc.tic();
//typedef typename cuthho_mesh<T, ET>::face_type  face_type;
        typedef typename cuthho_mesh<T, ET>::point_type point_type;
//typedef typename cuthho_mesh<T, ET>::cell_type cell_type;
        T iso_val_interface = level_set_function.iso_val_interface;
//std::cout<<"iso_val_interface = "<<iso_val_interface<<std::endl;
        size_t cell_i = 0;
        for (auto &cl: msh.cells) {
            auto fcs = faces(msh, cl);
            std::array<std::pair<size_t, point_type>, 2> cut_faces;

            size_t k = 0;
            for (size_t i = 0; i < fcs.size(); i++) {
                if (is_cut(msh, fcs[i]))
                    cut_faces.at(k++) = std::make_pair(i, fcs[i].user_data.intersection_point);
            }

/* If a face is cut, the cells that own the face are cut. Is this
         * unconditionally true? It should...fortunately this isn't avionics
         * software */

            if (k == 0) {

                auto is_positive = [&](const point_type &pt) -> bool {
                    return level_set_function(pt, msh, cl) > iso_val_interface;
                };


                auto pts = points(msh, cl);

                if (std::all_of(pts.begin(), pts.end(), is_positive))
                    cl.user_data.location = element_location::IN_POSITIVE_SIDE;
                else
                    cl.user_data.location = element_location::IN_NEGATIVE_SIDE;




/*
            auto pts = points(msh, cl);
            auto pt = pts.begin();
            size_t counter = 0;
            while( ( pt!= pts.end() ) && ( level_set_function(*pt,msh,cl) > 0 ) )
            {
                counter++;
                pt++;

            }

            if ( counter == pts.size() )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
            */

            }
//MODIFICARE QUAAAA
            if (k == 2) {
                cl.user_data.location = element_location::ON_INTERFACE;
                auto p0 = cut_faces[0].second;
                auto p1 = cut_faces[1].second;
                auto pt = p1 - p0;
                auto pn = p0 + point<T, 2>(-pt.y(), pt.x());
                auto pn_prova = (p0 + p1) / 2.0 + 0.5 * point<T, 2>(-pt.y(), pt.x());
//if(offset(msh,cl)== 119)
//    std::cout<<"p0 = "<<p0<< " , p1 ="<<p1<<std::endl;
// PRIMA ERA DA p0 ->   MODIFCATO, ora è pt  medio!
/*
            if( !pt_in_cell(msh, pn, cl) )
            {
                std::cout<<"I chose another pn to ordering interface_points in 'detect_cut_cells2'."<<std::endl;
                T m_half = ( ps1.y() - pm.y() )/( ps1.x() - pm.x() );
                T q = pm.y() - m_half * pm.x() ;
                auto pt_bdry = search_boundary( msh , cl , pm , m_half , q , lm , level_set_function ) ;
                auto lm_bdry = level_set_function( pt_bdry , msh , cl );
            }
            */
/*
            if(offset(msh,cl)== 137 || offset(msh,cl)== 138 || offset(msh,cl)== 134||offset(msh,cl)== 103){
                std::cout<<yellow<<bold<<"offset(msh,cl) = "<<offset(msh,cl)<<reset<<std::endl;
                auto pn_bis = (p0+p1)/2.0 + point<T,2>(-pt.y(), pt.x());
                std::cout<<"pn_bis = "<<pn_bis<< " , level_set_function(pn_bis,msh,cl) ="<<level_set_function(pn_bis,msh,cl) <<std::endl;
                auto pn_bis0 = (p0+p1)/2.0 + 0.5* point<T,2>(-pt.y(), pt.x());
                std::cout<<"pn_bis0 = "<<pn_bis0<< " , level_set_function(pn_bis0,msh,cl) ="<<level_set_function(pn_bis0,msh,cl) <<std::endl;
                auto pn_bis1 = p0 + 0.5 * point<T,2>(-pt.y(), pt.x());
                std::cout<<"pn_bis1 = "<<pn_bis1<< " , level_set_function(pn_bis1,msh,cl) ="<<level_set_function(pn_bis1,msh,cl)<<'\n' <<std::endl;

                std::cout<<"pn = "<<pn<< " , p0 = "<<p0<< " , p1 = "<<p1<<std::endl;
                std::cout<<"level_set_function(pn,msh,cl) = "<<level_set_function(pn,msh,cl)<< " , level_set_function(p0,msh,cl) = "<<level_set_function(p0,msh,cl)<< " , level_set_function(pn,msh,cl) = "<<level_set_function(p1,msh,cl)<<std::endl;
                std::cout<<"p0 - point<T,2>(-pt.y(), pt.x()) = "<<p0 - point<T,2>(-pt.y(), pt.x())<< " , level_set_function(p0 - point<T,2>(-pt.y(), pt.x()),msh,cl) = "<<level_set_function(p0 - point<T,2>(-pt.y(), pt.x()),msh,cl)<<std::endl;
            }
            */

                if (!(std::signbit(level_set_function(pn, msh, cl) - iso_val_interface) ==
                      std::signbit(level_set_function(pn_prova, msh, cl) - iso_val_interface))) {
                    std::cout << "p0 = " << p0 << " , p1 = " << p1 << std::endl;
                    std::cout << "pn = " << pn << " , pn_prova = " << pn_prova << " , level_set_function(pn,msh,cl) = "
                              << level_set_function(pn, msh, cl) << " , level_set_function(pn_prova,msh,cl) = "
                              << level_set_function(pn_prova, msh, cl) << std::endl;
                    pn = pn_prova;
                }

                if (level_set_function(pn, msh, cl) >= iso_val_interface) {
                    cl.user_data.p0 = p1;
                    cl.user_data.p1 = p0;
                } else {
                    cl.user_data.p0 = p0;
                    cl.user_data.p1 = p1;
                }

                cl.user_data.interface.push_back(cl.user_data.p0);
                cl.user_data.interface.push_back(cl.user_data.p1);
            }

            if (k != 0 && k != 2) {
                auto pts = points(msh, cl);
                std::cout << "Point[0] = " << pts[0] << " , point[1] = " << pts[1] << " , point[2] = " << pts[2]
                          << " , point[3] = " << pts[3] << std::endl;
                std::cout << "level_set_function(p0) = " << level_set_function(pts[0], msh, cl)
                          << " , level_set_function(p1) = " << level_set_function(pts[1], msh, cl)
                          << " , level_set_function(p2) = " << level_set_function(pts[2], msh, cl)
                          << " , level_set_function(p3) = " << level_set_function(pts[3], msh, cl) << std::endl;
                for (size_t i = 0; i < fcs.size(); i++) {
                    if (is_cut(msh, fcs[i]))
                        std::cout << "fcs[i].user_data.intersection_point = " << fcs[i].user_data.intersection_point
                                  << std::endl;
                }

                std::cout << "ERROR: in cut cell " << cell_i << " there are k = " << k << " cuts!!!!" << std::endl;
                throw std::logic_error(" --> Invalid number of cuts in cell");

            }

            cell_i++;
        }
        tc.toc();
//std::cout << bold << yellow << "detect_cut_cells3, time resolution: " << tc << " seconds" << reset << std::endl;
    }

/*
template<typename T, size_t ET, typename Function>
void
detect_cut_cells3_parallelized(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    std::cout<<"I AM IN DETECT CUT CELL3 PARALLELELIZED !!!!"<<std::endl;
    timecounter tc;
    tc.tic();
    //typedef typename cuthho_mesh<T, ET>::face_type  face_type;
    typedef typename cuthho_mesh<T, ET>::point_type point_type;
    //typedef typename cuthho_mesh<T, ET>::cell_type cell_type;
    T iso_val_interface = level_set_function.iso_val_interface ;
    std::cout<<"iso_val_interface = "<<iso_val_interface<<std::endl;


#ifdef HAVE_INTEL_TBB
    size_t n_cells = msh.cells.size();
    std::cout<<" I m in parallel zone"<<std::endl;
    tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
    [&] (size_t & cell_ind){
        auto& cl = msh.cells[cell_ind];

        auto fcs = faces(msh, cl);
        std::array< std::pair<size_t, point_type>, 2 >  cut_faces;

        size_t k = 0;
        for (size_t i = 0; i < fcs.size(); i++)
        {
            if ( is_cut(msh, fcs[i]) )
                cut_faces.at(k++) = std::make_pair(i, fcs[i].user_data.intersection_point);
        }




        if (k == 0)
        {

            auto is_positive = [&](const point_type& pt) -> bool {
            return level_set_function(pt) > iso_val_interface;
            };


            auto pts = points(msh, cl);

            if ( std::all_of(pts.begin(), pts.end(), is_positive) )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;






        }
        //MODIFICARE QUAAAA
        if (k == 2)
        {
            cl.user_data.location = element_location::ON_INTERFACE;
            auto p0 = cut_faces[0].second;
            auto p1 = cut_faces[1].second;
            auto pt = p1 - p0;
            auto pn = p0 + point<T,2>(-pt.y(), pt.x());
            auto pn_prova = (p0+p1)/2.0 + 0.5*point<T,2>(-pt.y(), pt.x());
            //if(offset(msh,cl)== 119)
            //    std::cout<<"p0 = "<<p0<< " , p1 ="<<p1<<std::endl;
            // PRIMA ERA DA p0 ->   MODIFCATO, ora è pt  medio!


            if( !(signbit(level_set_function(pn,msh,cl)-iso_val_interface) == signbit(level_set_function(pn_prova,msh,cl) -iso_val_interface) ) ){
                std::cout<<"p0 = "<<p0<< " , p1 = "<<p1<< std::endl;
                std::cout<<"pn = "<<pn<< " , pn_prova = "<<pn_prova<< " , level_set_function(pn,msh,cl) = "<<level_set_function(pn,msh,cl)<< " , level_set_function(pn_prova,msh,cl) = "<<level_set_function(pn_prova,msh,cl) <<std::endl;
                pn = pn_prova ;
            }

            if ( level_set_function(pn,msh,cl) >= iso_val_interface )
            {
                cl.user_data.p0 = p1;
                cl.user_data.p1 = p0;
            }
            else
            {
                cl.user_data.p0 = p0;
                cl.user_data.p1 = p1;
            }

            cl.user_data.interface.push_back(cl.user_data.p0);
            cl.user_data.interface.push_back(cl.user_data.p1);
        }

        if ( k != 0 && k != 2 ){
            auto pts = points(msh,cl);
            std::cout<<"Point[0] = "<<pts[0]<<" , point[1] = "<<pts[1]<<" , point[2] = "<<pts[2]<<" , point[3] = "<<pts[3]<<std::endl;
            std::cout<<"level_set_function(p0) = "<<level_set_function(pts[0],msh,cl) << " , level_set_function(p1) = "<<level_set_function(pts[1],msh,cl)<< " , level_set_function(p2) = "<<level_set_function(pts[2],msh,cl)<< " , level_set_function(p3) = "<<level_set_function(pts[3],msh,cl)<<std::endl;
            for (size_t i = 0; i < fcs.size(); i++)
            {
                if ( is_cut(msh, fcs[i]) )
                    std::cout<<"fcs[i].user_data.intersection_point = "<<fcs[i].user_data.intersection_point<<std::endl;
            }

            std::cout<<"ERROR: in cut cell "<<cell_ind<<" there are k = "<<k<<" cuts!!!!"<<std::endl;
            throw std::logic_error(" --> Invalid number of cuts in cell");

        }


    });
    tc.toc();
    std::cout << bold << yellow << "detect_cut_cells3_parallelized, time resolution: " << tc << " seconds" << reset << std::endl;
#else

    size_t cell_i = 0;
    for (auto& cl : msh.cells)
    {
        auto fcs = faces(msh, cl);
        std::array< std::pair<size_t, point_type>, 2 >  cut_faces;

        size_t k = 0;
        for (size_t i = 0; i < fcs.size(); i++)
        {
            if ( is_cut(msh, fcs[i]) )
                cut_faces.at(k++) = std::make_pair(i, fcs[i].user_data.intersection_point);
        }



        if (k == 0)
        {

            auto is_positive = [&](const point_type& pt) -> bool {
            return level_set_function(pt) > iso_val_interface;
            };


            auto pts = points(msh, cl);

            if ( std::all_of(pts.begin(), pts.end(), is_positive) )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;





        }
        //MODIFICARE QUAAAA
        if (k == 2)
        {
            cl.user_data.location = element_location::ON_INTERFACE;
            auto p0 = cut_faces[0].second;
            auto p1 = cut_faces[1].second;
            auto pt = p1 - p0;
            auto pn = p0 + point<T,2>(-pt.y(), pt.x());
            auto pn_prova = (p0+p1)/2.0 + 0.5*point<T,2>(-pt.y(), pt.x());
            //if(offset(msh,cl)== 119)
            //    std::cout<<"p0 = "<<p0<< " , p1 ="<<p1<<std::endl;
            // PRIMA ERA DA p0 ->   MODIFCATO, ora è pt  medio!

            if( !(signbit(level_set_function(pn,msh,cl)-iso_val_interface) == signbit(level_set_function(pn_prova,msh,cl) -iso_val_interface) ) ){
                std::cout<<"p0 = "<<p0<< " , p1 = "<<p1<< std::endl;
                std::cout<<"pn = "<<pn<< " , pn_prova = "<<pn_prova<< " , level_set_function(pn,msh,cl) = "<<level_set_function(pn,msh,cl)<< " , level_set_function(pn_prova,msh,cl) = "<<level_set_function(pn_prova,msh,cl) <<std::endl;
                pn = pn_prova ;
            }

            if ( level_set_function(pn,msh,cl) >= iso_val_interface )
            {
                cl.user_data.p0 = p1;
                cl.user_data.p1 = p0;
            }
            else
            {
                cl.user_data.p0 = p0;
                cl.user_data.p1 = p1;
            }

            cl.user_data.interface.push_back(cl.user_data.p0);
            cl.user_data.interface.push_back(cl.user_data.p1);
        }

        if ( k != 0 && k != 2 ){
            auto pts = points(msh,cl);
            std::cout<<"Point[0] = "<<pts[0]<<" , point[1] = "<<pts[1]<<" , point[2] = "<<pts[2]<<" , point[3] = "<<pts[3]<<std::endl;
            std::cout<<"level_set_function(p0) = "<<level_set_function(pts[0],msh,cl) << " , level_set_function(p1) = "<<level_set_function(pts[1],msh,cl)<< " , level_set_function(p2) = "<<level_set_function(pts[2],msh,cl)<< " , level_set_function(p3) = "<<level_set_function(pts[3],msh,cl)<<std::endl;
            for (size_t i = 0; i < fcs.size(); i++)
            {
                if ( is_cut(msh, fcs[i]) )
                  std::cout<<"fcs[i].user_data.intersection_point = "<<fcs[i].user_data.intersection_point<<std::endl;
            }

            std::cout<<"ERROR: in cut cell "<<cell_i<<" there are k = "<<k<<" cuts!!!!"<<std::endl;
            throw std::logic_error(" --> Invalid number of cuts in cell");

        }

        cell_i++;
    }

#endif
}
*/
/// New version of refine_interface for discret level functions

    template<typename T, size_t ET, typename Function>
    void
    refine_interface2(cuthho_mesh <T, ET> &msh, typename cuthho_mesh<T, ET>::cell_type &cl,
                      const Function &level_set_function, size_t min, size_t max) {
        if ((max - min) < 2)
            return;

        typedef typename cuthho_mesh<T, ET>::point_type point_type;

        size_t mid = (max + min) / 2;
        auto p0 = cl.user_data.interface.at(min);
        auto p1 = cl.user_data.interface.at(max);
        auto pm = (p0 + p1) / 2.0;
        auto pt = p1 - p0;
        auto pn = point_type(-pt.y(), pt.x());
        auto ps1 = pm + pn;
        auto ps2 = pm - pn;

        auto lm = level_set_function(pm, msh, cl);
        auto ls1 = level_set_function(ps1, msh, cl);
        auto ls2 = level_set_function(ps2, msh, cl);

        point_type ip;
// std::cout<<"the node of interface are "<<p0<<" and "<<p1<<". I search pm= "<<pm<<" in which phi = "<<lm<<" and ps1 e ps2 "<<ps1<<" and "<<ps2<<"equal to "<<ls1<<" , "<<ls2<<std::endl;
        if (!((lm >= 0 && ls1 >= 0) || (lm < 0 && ls1 < 0))) {
            auto threshold = diameter(msh, cl) / 1e20;
            ip = find_zero_crossing_in_cell(pm, ps1, level_set_function, threshold, msh, cl);
        } else if (!((lm >= 0 && ls2 >= 0) || (lm < 0 && ls2 < 0))) {
            auto threshold = diameter(msh, cl) / 1e20;
            ip = find_zero_crossing_in_cell(pm, ps2, level_set_function, threshold, msh, cl);
        } else
            throw std::logic_error("interface not found in search range");

        cl.user_data.interface.at(mid) = ip;

        refine_interface2(msh, cl, level_set_function, min, mid);
        refine_interface2(msh, cl, level_set_function, mid, max);
    }

    template<typename T, size_t ET, typename Function>
    void
    refine_interface2(cuthho_mesh <T, ET> &msh, const Function &level_set_function, size_t levels) {
        if (levels == 0)
            return;

        size_t interface_points = iexp_pow(2, levels);

        for (auto &cl: msh.cells) {
            if (!is_cut(msh, cl))
                continue;


            std::cout << yellow << bold << "--------------------> CELL = " << offset(msh, cl) << "<--------------------"
                      << reset << std::endl;
            cl.user_data.interface.resize(interface_points + 1);
            cl.user_data.interface.at(0) = cl.user_data.p0;
            cl.user_data.interface.at(interface_points) = cl.user_data.p1;

            refine_interface2(msh, cl, level_set_function, 0, interface_points);

            std::cout << "LIMIT CELL " << offset(msh, cl) << " are:" << std::endl;
            std::cout << "pt[0] = " << points(msh, cl)[0] << " , pt[1] = " << points(msh, cl)[1] << " , pt[2] = "
                      << points(msh, cl)[2] << " , pt[3] = " << points(msh, cl)[3] << std::endl;

            for (size_t i_int = 0; i_int < interface_points + 1; i_int++)
                std::cout << "refined points are p = " << cl.user_data.interface.at(i_int) << std::endl;
            std::cout << "--------------------> CELL = " << offset(msh, cl) << "<--------------------" << std::endl;
        }
    }


    template<typename T, size_t ET, typename Function>
    typename cuthho_mesh<T, ET>::point_type
    search_boundary(cuthho_mesh <T, ET> &msh, typename cuthho_mesh<T, ET>::cell_type &cl,
                    typename cuthho_mesh<T, ET>::point_type &p_init, T m, T q, T lm, const Function &level_set) {
        typedef typename cuthho_mesh<T, ET>::point_type point_type;
        auto pts = points(msh, cl);

        point_type pt_tmp0 = point_type(pts[0].x(), m * pts[0].x() + q);
        point_type pt_tmp1 = point_type(pts[1].x(), m * pts[1].x() + q);
        point_type pt_tmp2 = point_type((pts[1].y() - q) / m, pts[1].y());
        point_type pt_tmp3 = point_type((pts[2].y() - q) / m, pts[2].y());
/*
    if( offset(msh,cl) == 1029 || offset(msh,cl) == 1082 )
    {
        std::cout<<yellow<<bold<<"search_boundary"<<reset<<std::endl;
        std::cout<<"pt_tmp0 = "<<pt_tmp0<<std::endl;
        std::cout<<"pt_tmp1 = "<<pt_tmp1<<std::endl;
        std::cout<<"pt_tmp2 = "<<pt_tmp2<<std::endl;
        std::cout<<"pt_tmp3 = "<<pt_tmp3<<std::endl;
    }
    */
        auto ls0 = level_set(pt_tmp0, msh, cl);
        auto ls1 = level_set(pt_tmp1, msh, cl);
        auto ls2 = level_set(pt_tmp2, msh, cl);
        auto ls3 = level_set(pt_tmp3, msh, cl);

        if (pt_in_cell(msh, pt_tmp0, cl) && (!((lm >= 0 && ls0 >= 0) || (lm < 0 && ls0 < 0))))
            return pt_tmp0;
        if (pt_in_cell(msh, pt_tmp1, cl) && (!((lm >= 0 && ls1 >= 0) || (lm < 0 && ls1 < 0))))
            return pt_tmp1;
        if (pt_in_cell(msh, pt_tmp2, cl) && (!((lm >= 0 && ls2 >= 0) || (lm < 0 && ls2 < 0))))
            return pt_tmp2;
        if (pt_in_cell(msh, pt_tmp3, cl) && (!((lm >= 0 && ls3 >= 0) || (lm < 0 && ls3 < 0))))
            return pt_tmp3;
        else {
            std::cout << "In cell = " << offset(msh, cl) << " points(msh,cl)[0] = " << points(msh, cl)[0]
                      << " points(msh,cl)[1] = " << points(msh, cl)[1] << " points(msh,cl)[2] = " << points(msh, cl)[2]
                      << " points(msh,cl)[3] = " << points(msh, cl)[3] << std::endl;
            std::cout << "m = " << m << " --> q = " << q << std::endl;
            std::cout << "p_init = " << p_init << " --> pt_tmp0 = " << pt_tmp0 << " , pt_tmp1 = " << pt_tmp1
                      << " , pt_tmp2 = " << pt_tmp2 << " , pt_tmp3 = " << pt_tmp3 << std::endl;
            std::cout << "ls0 = " << ls0 << " , ls1 = " << ls1 << " , ls2 = " << ls2 << " , ls3 = " << ls3 << " AND lm = "
                      << lm << std::endl;
            std::cout << "pt_in_cell( pt_tmp0 ) = " << pt_in_cell(msh, pt_tmp0, cl) << " , pt_in_cell( pt_tmp1 ) = "
                      << pt_in_cell(msh, pt_tmp1, cl) << " , pt_in_cell( pt_tmp2 ) = " << pt_in_cell(msh, pt_tmp2, cl)
                      << " , pt_in_cel( pt_tmp3 ) = " << pt_in_cell(msh, pt_tmp3, cl) << std::endl;
            T pp = pts[0].x();
            T dist = std::abs(pp - p_init.x()) / 10.0;
            std::cout << "DIST = " << dist << " and pp = " << pp << " and p_init.x() = " << p_init.x() << std::endl;
            point_type p0 = point_type(pp + dist, m * (pp - dist) + q);
            point_type p1 = point_type(pp + (dist * 2), m * (pp + (dist * 2)) + q);
            point_type p2 = point_type(pp + (dist * 3), m * (pp + (dist * 3)) + q);
            point_type p3 = point_type(pp + (dist * 4), m * (pp + (dist * 4)) + q);
            point_type p4 = point_type(pp + (dist * 5), m * (pp + (dist * 5)) + q);
            point_type p5 = point_type(pp + (dist * 6), m * (pp + (dist * 6)) + q);
            point_type p6 = point_type(pp + (dist * 7), m * (pp + (dist * 7)) + q);
            point_type p7 = point_type(pp + (dist * 8), m * (pp + (dist * 8)) + q);
            point_type p8 = point_type(pp + (dist * 9), m * (pp + (dist * 9)) + q);
            std::cout << "p0 = " << p0 << " , level_set = " << level_set(p0, msh, cl) << " , p1 = " << p1
                      << " , level_set = " << level_set(p1, msh, cl) << " , p2 = " << p2 << " , level_set = "
                      << level_set(p2, msh, cl) << " , p3 = " << p3 << " , level_set = " << level_set(p3, msh, cl)
                      << " ,p4 = " << p4 << " , level_set = " << level_set(p4, msh, cl) << " ,p5 = " << p5
                      << " , level_set = " << level_set(p5, msh, cl) << " , p6 = " << p6 << " , level_set = "
                      << level_set(p6, msh, cl) << ", p7 = " << p7 << " , level_set = " << level_set(p7, msh, cl)
                      << " , p8 = " << p8 << " , level_set = " << level_set(p8, msh, cl) << std::endl;

//throw std::logic_error("search_boundary not find -> Stefano");
//return p_init ;

            point_type ret;
            T val_min = 1e10;
            if (pt_in_cell(msh, pt_tmp0, cl) && std::abs(ls0) < val_min) {
                val_min = std::abs(ls0);
                ret = pt_tmp0;

            }
            if (pt_in_cell(msh, pt_tmp1, cl) && std::abs(ls1) < val_min) {
                val_min = std::abs(ls1);
                ret = pt_tmp1;

            }
            if (pt_in_cell(msh, pt_tmp2, cl) && std::abs(ls2) < val_min) {
                val_min = std::abs(ls2);
                ret = pt_tmp2;

            }
            if (pt_in_cell(msh, pt_tmp3, cl) && std::abs(ls3) < val_min) {
                val_min = std::abs(ls3);
                ret = pt_tmp3;

            }
            return ret;

        }


    }


    template<typename T, size_t ET>
    typename cuthho_mesh<T, ET>::point_type
    search_boundary(cuthho_mesh <T, ET> &msh, typename cuthho_mesh<T, ET>::cell_type &cl,
                    typename cuthho_mesh<T, ET>::point_type &p_init, T m, T q) {
        typedef typename cuthho_mesh<T, ET>::point_type point_type;
        auto pts = points(msh, cl);

        point_type pt_tmp0 = point_type(pts[0].x(), m * pts[0].x() + q);
        point_type pt_tmp1 = point_type(pts[1].x(), m * pts[1].x() + q);
        point_type pt_tmp2 = point_type((pts[1].y() - q) / m, pts[1].y());
        point_type pt_tmp3 = point_type((pts[2].y() - q) / m, pts[2].y());


        if (pt_in_cell(msh, pt_tmp0, cl) && !(p_init == pt_tmp0))
            return pt_tmp0;
        if (pt_in_cell(msh, pt_tmp1, cl) && !(p_init == pt_tmp1))
            return pt_tmp1;
        if (pt_in_cell(msh, pt_tmp2, cl) && !(p_init == pt_tmp2))
            return pt_tmp2;
        if (pt_in_cell(msh, pt_tmp3, cl) && !(p_init == pt_tmp3))
            return pt_tmp3;


    }


    template<typename T, size_t ET, typename Function>
    void
    refine_interface_pro(cuthho_mesh <T, ET> &msh, typename cuthho_mesh<T, ET>::cell_type &cl,
                         const Function &level_set_function, size_t min, size_t max) {
        if ((max - min) < 2)
            return;

        typedef typename cuthho_mesh<T, ET>::point_type point_type;

        size_t mid = (max + min) / 2;
        auto p0 = cl.user_data.interface.at(min);
        auto p1 = cl.user_data.interface.at(max);
        auto pm = (p0 + p1) / 2.0;
        auto pt = p1 - p0;
        auto pn = point_type(-pt.y(), pt.x());
        auto ps1 = pm + pn;
        auto ps2 = pm - pn;


        auto lm = level_set_function(pm, msh, cl);
        auto ls1 = level_set_function(ps1, msh, cl);
        auto ls2 = level_set_function(ps2, msh, cl);

// CASE  MAX PT on the boudary
        T m_half = (ps1.y() - pm.y()) / (ps1.x() - pm.x());
        T q = pm.y() - m_half * pm.x();
        if (offset(msh, cl) == 119) {
            std::cout << yellow << bold << "CELL 119" << reset << std::endl;
            std::cout << "p0 = " << p0 << " , p1 = " << p1 << std::endl;
            std::cout << "ps1.y() = " << ps1.y() << " , pm.y() = " << pm.y() << std::endl;
            std::cout << "ps1.x() = " << ps1.x() << " , pm.x() = " << pm.x() << std::endl;
            std::cout << "ps1.x() = " << ps1.x() << " , pm.x() = " << pm.x() << std::endl;
        }
/*
    if( offset(msh,cl) == 118 )
    {
        T m_half_bis = ( ps2.y() - pm.y() )/( ps2.x() - pm.x() );
        T q_bis = pm.y() - m_half * pm.x() ;
        std::cout<<yellow<<bold<<"CELL 118"<<reset<<std::endl;
        std::cout<<"p0 = "<<p0 << " , p1 = "<<p1<<std::endl;
        std::cout<<"m_half = "<<m_half << " , m_half_bis = "<<m_half_bis<<std::endl;
        std::cout<<"q = "<<q << " , q_bis = "<<q_bis<<std::endl;

        std::cout<<"pm = "<<pm << " , level_set_function(pm) = "<<lm<<std::endl;
        std::cout<<"ps1 = "<<ps1 << " , level_set_function(ps1) = "<<ls1<<std::endl;
        std::cout<<"ps2 = "<<ps2 << " , level_set_function(ps2) = "<<ls2<<std::endl;
    }
    */
        auto pt_bdry = search_boundary(msh, cl, pm, m_half, q, lm, level_set_function);
        auto lm_bdry = level_set_function(pt_bdry, msh, cl);
/*
    if( offset(msh,cl) == 118 )
        std::cout<<"pt_bdry = "<<pt_bdry << " , level_set_function(lm_bdry) = "<<lm_bdry<<std::endl;
    */
//std::cout<<"pm = "<<pm << " , level_set_function(pm) = "<<lm<<std::endl;
//std::cout<<"ps1 = "<<ps1 << " , level_set_function(ps1) = "<<ls1<<std::endl;
//std::cout<<"ps2 = "<<ps2 << " , level_set_function(ps2) = "<<ls2<<std::endl;


        point_type ip;
// std::cout<<"the node of interface are "<<p0<<" and "<<p1<<". I search pm= "<<pm<<" in which phi = "<<lm<<" and ps1 e ps2 "<<ps1<<" and "<<ps2<<"equal to "<<ls1<<" , "<<ls2<<std::endl;
        if (pt_in_cell(msh, ps1, cl) && (!((lm >= 0 && ls1 >= 0) || (lm < 0 && ls1 < 0)))) {
            auto threshold = diameter(msh, cl) / 1e20;
//auto threshold = diameter(msh, cl) / 1e10;
            ip = find_zero_crossing_in_cell(pm, ps1, level_set_function, threshold, msh, cl);
//std::cout<<"OLD 1"<<std::endl;
        } else if (pt_in_cell(msh, ps2, cl) && (!((lm >= 0 && ls2 >= 0) || (lm < 0 && ls2 < 0)))) {
            auto threshold = diameter(msh, cl) / 1e20;
//auto threshold = diameter(msh, cl) / 1e10;
            ip = find_zero_crossing_in_cell(pm, ps2, level_set_function, threshold, msh, cl);
//std::cout<<"OLD 2"<<std::endl;
        } else if (pt_in_cell(msh, pt_bdry, cl) && (!((lm >= 0 && lm_bdry >= 0) || (lm < 0 && lm_bdry < 0)))) {
            auto threshold = diameter(msh, cl) / 1e20;
//auto threshold = diameter(msh, cl) / 1e10;
            ip = find_zero_crossing_in_cell(pm, pt_bdry, level_set_function, threshold, msh, cl);
//std::cout<<"BDRY NEW"<<std::endl;
        } else {
//throw std::logic_error("interface not found in search range");
//std::cout<<yellow<<bold<< "In cell "<<offset(msh,cl)<<" ---> implementing linear approximation. INTERFACE NOT FOUND."<<reset<<std::endl;
//ip = pm;
            std::cout << yellow << bold << "In cell " << offset(msh, cl)
                      << " ---> implementing MINIMISATION ERROR APPROXIMATION. INTERFACE NOT FOUND." << reset << std::endl;
            point_type ret;
            T val_min = 1e10;
            if (pt_in_cell(msh, ps1, cl) && std::abs(ls1) < val_min) {
                val_min = std::abs(ls1);
                ret = ps1;
                std::cout << "ps1 = " << ps1 << " , ls1 = " << ls1 << std::endl;

            }
            if (pt_in_cell(msh, ps2, cl) && std::abs(ls2) < val_min) {
                val_min = std::abs(ls2);
                ret = ps2;
                std::cout << "ps2 = " << ps2 << " , ls2 = " << ls2 << std::endl;
            }
            if (pt_in_cell(msh, pt_bdry, cl) && std::abs(lm_bdry) < val_min) {
                val_min = std::abs(lm_bdry);
                ret = pt_bdry;
                std::cout << "ppt_bdrys1 = " << pt_bdry << " , lm_bdry = " << lm_bdry << std::endl;
            }
            if (pt_in_cell(msh, pm, cl) && std::abs(lm) < val_min) {
                val_min = std::abs(lm);
                ret = pm;
                std::cout << "pm = " << ps1 << " , lm = " << ls1 << std::endl;
            }
            std::cout << "ret = " << ret << std::endl;
            ip = ret;

        }
/*
    if( offset(msh,cl) == 118 )
        std::cout<<"POINT INTERFACE Ip = "<<ip <<  " in pos = "<<mid<<std::endl;
    */
        cl.user_data.interface.at(mid) = ip;

        refine_interface_pro(msh, cl, level_set_function, min, mid);
        refine_interface_pro(msh, cl, level_set_function, mid, max);
    }

    template<typename T, size_t ET, typename Function>
    void
    refine_interface_pro(cuthho_mesh <T, ET> &msh, const Function &level_set_function, size_t levels) {
        if (levels == 0)
            return;


        size_t interface_points = iexp_pow(2, levels);

        for (auto &cl: msh.cells) {
            if (!is_cut(msh, cl))
                continue;

/*
        if( offset(msh,cl) == 118 )
        {
        std::cout<<yellow<<bold<<"--------------------> CELL = "<<offset(msh,cl)<<" <--------------------"<<reset<<std::endl;
        size_t counter = 0;
        for (auto& nd :  nodes(msh,cl) )
        {

            if( nd.user_data.location == element_location::IN_NEGATIVE_SIDE ){
                std::cout<<"NEGATIVE -> nd = "<<nd.ptid << " --> pt = "<<points(msh,cl)[counter] << std::endl;
            }
            else{
                std::cout<<"POSITIVE -> nd = "<<nd.ptid << " --> pt = "<<points(msh,cl)[counter] << std::endl;
            }
            counter++;
            //std::cout<<"nd = "<<nd.ptid<<std::endl;
        }
        std::cout<<"INTERFACE_0 = "<<cl.user_data.p0 << " , INTERFACE_1 = "<<cl.user_data.p1 << std::endl;
        }
        */
            cl.user_data.interface.resize(interface_points + 1);
            cl.user_data.interface.at(0) = cl.user_data.p0;
            cl.user_data.interface.at(interface_points) = cl.user_data.p1;

            refine_interface_pro(msh, cl, level_set_function, 0, interface_points);
/*
        if( offset(msh,cl) == 118 )
        {
        for(size_t i_int = 0 ; i_int < interface_points + 1 ; i_int++ )
            std::cout<<"refined points are p = "<<cl.user_data.interface.at(i_int)<<std::endl;

        std::cout<<"--------------------> FINE CELL <--------------------"<<std::endl;
        }
        */
        }
    }

    template<typename T, size_t ET, typename Function>
    typename cuthho_mesh<T, ET>::point_type
    search_boundary3(cuthho_mesh <T, ET> &msh, typename cuthho_mesh<T, ET>::cell_type &cl,
                     typename cuthho_mesh<T, ET>::point_type &p_init, T m, T q, T lm, const Function &level_set,
                     T iso_val_interface) {
        typedef typename cuthho_mesh<T, ET>::point_type point_type;
        auto pts = points(msh, cl);

        point_type pt_tmp0 = point_type(pts[0].x(), m * pts[0].x() + q);
        point_type pt_tmp1 = point_type(pts[1].x(), m * pts[1].x() + q);
        point_type pt_tmp2 = point_type((pts[1].y() - q) / m, pts[1].y());
        point_type pt_tmp3 = point_type((pts[2].y() - q) / m, pts[2].y());
/*
    if( offset(msh,cl) == 1029 || offset(msh,cl) == 1082 )
    {
        std::cout<<yellow<<bold<<"search_boundary"<<reset<<std::endl;
        std::cout<<"pt_tmp0 = "<<pt_tmp0<<std::endl;
        std::cout<<"pt_tmp1 = "<<pt_tmp1<<std::endl;
        std::cout<<"pt_tmp2 = "<<pt_tmp2<<std::endl;
        std::cout<<"pt_tmp3 = "<<pt_tmp3<<std::endl;
    }
    */
        auto ls0 = level_set(pt_tmp0, msh, cl);
        auto ls1 = level_set(pt_tmp1, msh, cl);
        auto ls2 = level_set(pt_tmp2, msh, cl);
        auto ls3 = level_set(pt_tmp3, msh, cl);

        if (pt_in_cell(msh, pt_tmp0, cl) && (!((lm >= iso_val_interface && ls0 >= iso_val_interface) ||
                                               (lm < iso_val_interface && ls0 < iso_val_interface))))
            return pt_tmp0;
        if (pt_in_cell(msh, pt_tmp1, cl) && (!((lm >= iso_val_interface && ls1 >= iso_val_interface) ||
                                               (lm < iso_val_interface && ls1 < iso_val_interface))))
            return pt_tmp1;
        if (pt_in_cell(msh, pt_tmp2, cl) && (!((lm >= iso_val_interface && ls2 >= iso_val_interface) ||
                                               (lm < iso_val_interface && ls2 < iso_val_interface))))
            return pt_tmp2;
        if (pt_in_cell(msh, pt_tmp3, cl) && (!((lm >= iso_val_interface && ls3 >= iso_val_interface) ||
                                               (lm < iso_val_interface && ls3 < iso_val_interface))))
            return pt_tmp3;
        else {
            std::cout << "In cell = " << offset(msh, cl) << " points(msh,cl)[0] = " << points(msh, cl)[0]
                      << " points(msh,cl)[1] = " << points(msh, cl)[1] << " points(msh,cl)[2] = " << points(msh, cl)[2]
                      << " points(msh,cl)[3] = " << points(msh, cl)[3] << std::endl;
            std::cout << "m = " << m << " --> q = " << q << std::endl;
            std::cout << "p_init = " << p_init << " --> pt_tmp0 = " << pt_tmp0 << " , pt_tmp1 = " << pt_tmp1
                      << " , pt_tmp2 = " << pt_tmp2 << " , pt_tmp3 = " << pt_tmp3 << std::endl;
            std::cout << "ls0 = " << ls0 << " , ls1 = " << ls1 << " , ls2 = " << ls2 << " , ls3 = " << ls3 << " AND lm = "
                      << lm << std::endl;
            std::cout << "pt_in_cell( pt_tmp0 ) = " << pt_in_cell(msh, pt_tmp0, cl) << " , pt_in_cell( pt_tmp1 ) = "
                      << pt_in_cell(msh, pt_tmp1, cl) << " , pt_in_cell( pt_tmp2 ) = " << pt_in_cell(msh, pt_tmp2, cl)
                      << " , pt_in_cel( pt_tmp3 ) = " << pt_in_cell(msh, pt_tmp3, cl) << std::endl;
            T pp = pts[0].x();
            T dist = std::abs(pp - p_init.x()) / 10.0;
            std::cout << "DIST = " << dist << " and pp = " << pp << " and p_init.x() = " << p_init.x() << std::endl;
            point_type p0 = point_type(pp + dist, m * (pp - dist) + q);
            point_type p1 = point_type(pp + (dist * 2), m * (pp + (dist * 2)) + q);
            point_type p2 = point_type(pp + (dist * 3), m * (pp + (dist * 3)) + q);
            point_type p3 = point_type(pp + (dist * 4), m * (pp + (dist * 4)) + q);
            point_type p4 = point_type(pp + (dist * 5), m * (pp + (dist * 5)) + q);
            point_type p5 = point_type(pp + (dist * 6), m * (pp + (dist * 6)) + q);
            point_type p6 = point_type(pp + (dist * 7), m * (pp + (dist * 7)) + q);
            point_type p7 = point_type(pp + (dist * 8), m * (pp + (dist * 8)) + q);
            point_type p8 = point_type(pp + (dist * 9), m * (pp + (dist * 9)) + q);
            std::cout << "p0 = " << p0 << " , level_set = " << level_set(p0, msh, cl) << " , p1 = " << p1
                      << " , level_set = " << level_set(p1, msh, cl) << " , p2 = " << p2 << " , level_set = "
                      << level_set(p2, msh, cl) << " , p3 = " << p3 << " , level_set = " << level_set(p3, msh, cl)
                      << " ,p4 = " << p4 << " , level_set = " << level_set(p4, msh, cl) << " ,p5 = " << p5
                      << " , level_set = " << level_set(p5, msh, cl) << " , p6 = " << p6 << " , level_set = "
                      << level_set(p6, msh, cl) << ", p7 = " << p7 << " , level_set = " << level_set(p7, msh, cl)
                      << " , p8 = " << p8 << " , level_set = " << level_set(p8, msh, cl) << std::endl;

//throw std::logic_error("search_boundary not find -> Stefano");
//return p_init ;

            point_type ret;
            T val_min = 1e10;
            if (pt_in_cell(msh, pt_tmp0, cl) && std::abs(ls0 - iso_val_interface) < val_min) {
                val_min = std::abs(ls0);
                ret = pt_tmp0;

            }
            if (pt_in_cell(msh, pt_tmp1, cl) && std::abs(ls1 - iso_val_interface) < val_min) {
                val_min = std::abs(ls1);
                ret = pt_tmp1;

            }
            if (pt_in_cell(msh, pt_tmp2, cl) && std::abs(ls2 - iso_val_interface) < val_min) {
                val_min = std::abs(ls2);
                ret = pt_tmp2;

            }
            if (pt_in_cell(msh, pt_tmp3, cl) && std::abs(ls3 - iso_val_interface) < val_min) {
                val_min = std::abs(ls3);
                ret = pt_tmp3;

            }
            return ret;

        }


    }


// USING INTERFACE = 1/2
    template<typename T, size_t ET, typename Function>
    void
    refine_interface_pro3(cuthho_mesh <T, ET> &msh, typename cuthho_mesh<T, ET>::cell_type &cl,
                          const Function &level_set_function, size_t min, size_t max) {
        if ((max - min) < 2)
            return;

        typedef typename cuthho_mesh<T, ET>::point_type point_type;
        T iso_val_interface = level_set_function.iso_val_interface;
        size_t mid = (max + min) / 2;
        auto p0 = cl.user_data.interface.at(min);
        auto p1 = cl.user_data.interface.at(max);
        auto pm = (p0 + p1) / 2.0;
        auto pt = p1 - p0;
        auto pn = point_type(-pt.y(), pt.x());
        auto ps1 = pm + pn;
        auto ps2 = pm - pn;


        auto lm = level_set_function(pm, msh, cl);
        auto ls1 = level_set_function(ps1, msh, cl);
        auto ls2 = level_set_function(ps2, msh, cl);

// CASE  MAX PT on the boudary
        T m_half = (ps1.y() - pm.y()) / (ps1.x() - pm.x());
        T q = pm.y() - m_half * pm.x();
/*
    if( offset(msh,cl) == 119 )
    {
        std::cout<<yellow<<bold<<"CELL 119"<<reset<<std::endl;
        std::cout<<"p0 = "<<p0 << " , p1 = "<<p1<<std::endl;
        std::cout<<"ps1.y() = "<<ps1.y() << " , pm.y() = "<<pm.y()<<std::endl;
        std::cout<<"ps1.x() = "<<ps1.x() << " , pm.x() = "<<pm.x()<<std::endl;
        std::cout<<"ps1.x() = "<<ps1.x() << " , pm.x() = "<<pm.x()<<std::endl;
    }

    if( offset(msh,cl) == 118 )
    {
        T m_half_bis = ( ps2.y() - pm.y() )/( ps2.x() - pm.x() );
        T q_bis = pm.y() - m_half * pm.x() ;
        std::cout<<yellow<<bold<<"CELL 118"<<reset<<std::endl;
        std::cout<<"p0 = "<<p0 << " , p1 = "<<p1<<std::endl;
        std::cout<<"m_half = "<<m_half << " , m_half_bis = "<<m_half_bis<<std::endl;
        std::cout<<"q = "<<q << " , q_bis = "<<q_bis<<std::endl;

        std::cout<<"pm = "<<pm << " , level_set_function(pm) = "<<lm<<std::endl;
        std::cout<<"ps1 = "<<ps1 << " , level_set_function(ps1) = "<<ls1<<std::endl;
        std::cout<<"ps2 = "<<ps2 << " , level_set_function(ps2) = "<<ls2<<std::endl;
    }
    */
        auto pt_bdry = search_boundary3(msh, cl, pm, m_half, q, lm, level_set_function, iso_val_interface);
        auto lm_bdry = level_set_function(pt_bdry, msh, cl);
/*
    if( offset(msh,cl) == 118 )
        std::cout<<"pt_bdry = "<<pt_bdry << " , level_set_function(lm_bdry) = "<<lm_bdry<<std::endl;
    */
//std::cout<<"pm = "<<pm << " , level_set_function(pm) = "<<lm<<std::endl;
//std::cout<<"ps1 = "<<ps1 << " , level_set_function(ps1) = "<<ls1<<std::endl;
//std::cout<<"ps2 = "<<ps2 << " , level_set_function(ps2) = "<<ls2<<std::endl;


        point_type ip;
// std::cout<<"the node of interface are "<<p0<<" and "<<p1<<". I search pm= "<<pm<<" in which phi = "<<lm<<" and ps1 e ps2 "<<ps1<<" and "<<ps2<<"equal to "<<ls1<<" , "<<ls2<<std::endl;
        if (pt_in_cell(msh, ps1, cl) && (!((lm >= iso_val_interface && ls1 >= iso_val_interface) ||
                                           (lm < iso_val_interface && ls1 < iso_val_interface)))) {
            auto threshold = diameter(msh, cl) / 1e20;
//auto threshold = diameter(msh, cl) / 1e10;
            ip = find_zero_crossing_in_cell3(pm, ps1, level_set_function, threshold, msh, cl);
//std::cout<<"OLD 1"<<std::endl;
        } else if (pt_in_cell(msh, ps2, cl) && (!((lm >= iso_val_interface && ls2 >= iso_val_interface) ||
                                                  (lm < iso_val_interface && ls2 < iso_val_interface)))) {
            auto threshold = diameter(msh, cl) / 1e20;
//auto threshold = diameter(msh, cl) / 1e10;
            ip = find_zero_crossing_in_cell3(pm, ps2, level_set_function, threshold, msh, cl);
//std::cout<<"OLD 2"<<std::endl;
        } else if (pt_in_cell(msh, pt_bdry, cl) && (!((lm >= iso_val_interface && lm_bdry >= iso_val_interface) ||
                                                      (lm < iso_val_interface && lm_bdry < iso_val_interface)))) {
            auto threshold = diameter(msh, cl) / 1e20;
//auto threshold = diameter(msh, cl) / 1e10;
            ip = find_zero_crossing_in_cell3(pm, pt_bdry, level_set_function, threshold, msh, cl);
//std::cout<<"BDRY NEW"<<std::endl;
        } else {
//throw std::logic_error("interface not found in search range");
//std::cout<<yellow<<bold<< "In cell "<<offset(msh,cl)<<" ---> implementing linear approximation. INTERFACE NOT FOUND."<<reset<<std::endl;
//ip = pm;
            std::cout << yellow << bold << "In cell " << offset(msh, cl)
                      << " ---> implementing MINIMISATION ERROR APPROXIMATION. INTERFACE NOT FOUND." << reset << std::endl;
            point_type ret;
            T val_min = 1e10;
            if (pt_in_cell(msh, ps1, cl) && std::abs(ls1 - iso_val_interface) < val_min) {
                val_min = std::abs(ls1);
                ret = ps1;
                std::cout << "ps1 = " << ps1 << " , ls1 = " << ls1 << std::endl;

            }
            if (pt_in_cell(msh, ps2, cl) && std::abs(ls2 - iso_val_interface) < val_min) {
                val_min = std::abs(ls2);
                ret = ps2;
                std::cout << "ps2 = " << ps2 << " , ls2 = " << ls2 << std::endl;
            }
            if (pt_in_cell(msh, pt_bdry, cl) && std::abs(lm_bdry - iso_val_interface) < val_min) {
                val_min = std::abs(lm_bdry);
                ret = pt_bdry;
                std::cout << "ppt_bdrys1 = " << pt_bdry << " , lm_bdry = " << lm_bdry << std::endl;
            }
            if (pt_in_cell(msh, pm, cl) && std::abs(lm - iso_val_interface) < val_min) {
                val_min = std::abs(lm);
                ret = pm;
                std::cout << "pm = " << ps1 << " , lm = " << ls1 << std::endl;
            }
            std::cout << "ret = " << ret << std::endl;
            ip = ret;

        }
/*
    if( offset(msh,cl) == 118 )
        std::cout<<"POINT INTERFACE Ip = "<<ip <<  " in pos = "<<mid<<std::endl;
    */
        cl.user_data.interface.at(mid) = ip;

        refine_interface_pro3(msh, cl, level_set_function, min, mid);
        refine_interface_pro3(msh, cl, level_set_function, mid, max);
    }


    template<typename T, size_t ET, typename Function>
    void
    refine_interface_pro3(cuthho_mesh <T, ET> &msh, const Function &level_set_function, size_t levels) {
        if (levels == 0)
            return;


        size_t interface_points = iexp_pow(2, levels);

        for (auto &cl: msh.cells) {
            if (!is_cut(msh, cl))
                continue;

/*
        if( offset(msh,cl) == 118 )
        {
        std::cout<<yellow<<bold<<"--------------------> CELL = "<<offset(msh,cl)<<" <--------------------"<<reset<<std::endl;
        size_t counter = 0;
        for (auto& nd :  nodes(msh,cl) )
        {

            if( nd.user_data.location == element_location::IN_NEGATIVE_SIDE ){
                std::cout<<"NEGATIVE -> nd = "<<nd.ptid << " --> pt = "<<points(msh,cl)[counter] << std::endl;
            }
            else{
                std::cout<<"POSITIVE -> nd = "<<nd.ptid << " --> pt = "<<points(msh,cl)[counter] << std::endl;
            }
            counter++;
            //std::cout<<"nd = "<<nd.ptid<<std::endl;
        }
        std::cout<<"INTERFACE_0 = "<<cl.user_data.p0 << " , INTERFACE_1 = "<<cl.user_data.p1 << std::endl;
        }
        */
            cl.user_data.interface.resize(interface_points + 1);
            cl.user_data.interface.at(0) = cl.user_data.p0;
            cl.user_data.interface.at(interface_points) = cl.user_data.p1;

            refine_interface_pro3(msh, cl, level_set_function, 0, interface_points);
/*
        if( offset(msh,cl) == 118 )
        {
        for(size_t i_int = 0 ; i_int < interface_points + 1 ; i_int++ )
            std::cout<<"refined points are p = "<<cl.user_data.interface.at(i_int)<<std::endl;

        std::cout<<"--------------------> FINE CELL <--------------------"<<std::endl;
        }
        */
        }
    }


    template<typename T, size_t ET, typename Function, typename ITERATOR>
    void
    refine_interface_pro3_disp(cuthho_mesh <T, ET> &msh, typename cuthho_mesh<T, ET>::cell_type &cl,
                               const Function &level_set_function, ITERATOR first_pt, ITERATOR second_pt, int &counter,
                               int counter_fin, int curve_degree, T &fraz) {

        if ((counter_fin - counter) == 1)
            return;

        typedef typename cuthho_mesh<T, ET>::point_type point_type;
        T iso_val_interface = level_set_function.iso_val_interface;

        auto p0 = *first_pt;
        auto p1 = *second_pt;

        auto pm = p0 + fraz * (p1 - p0);

//std::cout<<"first_pt = "<<*first_pt<<" ,  second_pt = "<< *second_pt<< " , pm = "<<pm<<" , fraz = "<<fraz<<std::endl;
        fraz += 1.0 / curve_degree;
        auto pt = p1 - p0;
        auto pn = point_type(-pt.y(), pt.x());
        auto ps1 = pm + pn;
        auto ps2 = pm - pn;


        auto lm = level_set_function(pm, msh, cl);
        auto ls1 = level_set_function(ps1, msh, cl);
        auto ls2 = level_set_function(ps2, msh, cl);

// CASE  MAX PT on the boudary
        T m_half = (ps1.y() - pm.y()) / (ps1.x() - pm.x());
        T q = pm.y() - m_half * pm.x();


        auto pt_bdry = search_boundary3(msh, cl, pm, m_half, q, lm, level_set_function, iso_val_interface);
        auto lm_bdry = level_set_function(pt_bdry, msh, cl);
/*
    if( offset(msh,cl) == 118 )
        std::cout<<"pt_bdry = "<<pt_bdry << " , level_set_function(lm_bdry) = "<<lm_bdry<<std::endl;
    */
//std::cout<<"pm = "<<pm << " , level_set_function(pm) = "<<lm<<std::endl;
//std::cout<<"ps1 = "<<ps1 << " , level_set_function(ps1) = "<<ls1<<std::endl;
//std::cout<<"ps2 = "<<ps2 << " , level_set_function(ps2) = "<<ls2<<std::endl;


        point_type ip;
// std::cout<<"the node of interface are "<<p0<<" and "<<p1<<". I search pm= "<<pm<<" in which phi = "<<lm<<" and ps1 e ps2 "<<ps1<<" and "<<ps2<<"equal to "<<ls1<<" , "<<ls2<<std::endl;
        if (pt_in_cell(msh, ps1, cl) && (!((lm >= iso_val_interface && ls1 >= iso_val_interface) ||
                                           (lm < iso_val_interface && ls1 < iso_val_interface)))) {
            auto threshold = diameter(msh, cl) / 1e20;
//auto threshold = diameter(msh, cl) / 1e10;
            ip = find_zero_crossing_in_cell3(pm, ps1, level_set_function, threshold, msh, cl);
//std::cout<<"OLD 1"<<std::endl;
        } else if (pt_in_cell(msh, ps2, cl) && (!((lm >= iso_val_interface && ls2 >= iso_val_interface) ||
                                                  (lm < iso_val_interface && ls2 < iso_val_interface)))) {
            auto threshold = diameter(msh, cl) / 1e20;
//auto threshold = diameter(msh, cl) / 1e10;
            ip = find_zero_crossing_in_cell3(pm, ps2, level_set_function, threshold, msh, cl);
//std::cout<<"OLD 2"<<std::endl;
        } else if (pt_in_cell(msh, pt_bdry, cl) && (!((lm >= iso_val_interface && lm_bdry >= iso_val_interface) ||
                                                      (lm < iso_val_interface && lm_bdry < iso_val_interface)))) {
            auto threshold = diameter(msh, cl) / 1e20;
//auto threshold = diameter(msh, cl) / 1e10;
            ip = find_zero_crossing_in_cell3(pm, pt_bdry, level_set_function, threshold, msh, cl);
//std::cout<<"BDRY NEW"<<std::endl;
        } else {
//throw std::logic_error("interface not found in search range");
//std::cout<<yellow<<bold<< "In cell "<<offset(msh,cl)<<" ---> implementing linear approximation. INTERFACE NOT FOUND."<<reset<<std::endl;
//ip = pm;
            std::cout << yellow << bold << "In cell " << offset(msh, cl)
                      << " ---> implementing MINIMISATION ERROR APPROXIMATION. INTERFACE NOT FOUND." << reset << std::endl;
            point_type ret;
            T val_min = 1e10;
            if (pt_in_cell(msh, ps1, cl) && std::abs(ls1 - iso_val_interface) < val_min) {
                val_min = std::abs(ls1);
                ret = ps1;
                std::cout << "ps1 = " << ps1 << " , ls1 = " << ls1 << std::endl;

            }
            if (pt_in_cell(msh, ps2, cl) && std::abs(ls2 - iso_val_interface) < val_min) {
                val_min = std::abs(ls2);
                ret = ps2;
                std::cout << "ps2 = " << ps2 << " , ls2 = " << ls2 << std::endl;
            }
            if (pt_in_cell(msh, pt_bdry, cl) && std::abs(lm_bdry - iso_val_interface) < val_min) {
                val_min = std::abs(lm_bdry);
                ret = pt_bdry;
                std::cout << "ppt_bdrys1 = " << pt_bdry << " , lm_bdry = " << lm_bdry << std::endl;
            }
            if (pt_in_cell(msh, pm, cl) && std::abs(lm - iso_val_interface) < val_min) {
                val_min = std::abs(lm);
                ret = pm;
                std::cout << "pm = " << ps1 << " , lm = " << ls1 << std::endl;
            }
            std::cout << "ret = " << ret << std::endl;
            ip = ret;

        }
        counter++;
        cl.user_data.interface.at(counter) = ip;

        refine_interface_pro3_disp(msh, cl, level_set_function, first_pt, second_pt, counter, counter_fin, curve_degree,
                                   fraz);
    }

    template<typename T, size_t ET, typename Function>
    void
    refine_interface_pro3_disp_subcell(cuthho_mesh <T, ET> &msh, typename cuthho_mesh<T, ET>::cell_type &cl,
                                       const Function &level_set_function, std::vector <point<T, 2>> &interface_tmp,
                                       size_t min, size_t max) {
        if ((max - min) < 2)
            return;

        typedef typename cuthho_mesh<T, ET>::point_type point_type;
        T iso_val_interface = level_set_function.iso_val_interface;
        size_t mid = (max + min) / 2;
        auto p0 = interface_tmp.at(min);
        auto p1 = interface_tmp.at(max);
        auto pm = (p0 + p1) / 2.0;
        auto pt = p1 - p0;
        auto pn = point_type(-pt.y(), pt.x());
        auto ps1 = pm + pn;
        auto ps2 = pm - pn;


        auto lm = level_set_function(pm, msh, cl);
        auto ls1 = level_set_function(ps1, msh, cl);
        auto ls2 = level_set_function(ps2, msh, cl);

// CASE  MAX PT on the boudary
        T m_half = (ps1.y() - pm.y()) / (ps1.x() - pm.x());
        T q = pm.y() - m_half * pm.x();

/*
    if( offset(msh,cl) == 118 )
    {
        T m_half_bis = ( ps2.y() - pm.y() )/( ps2.x() - pm.x() );
        T q_bis = pm.y() - m_half * pm.x() ;
        std::cout<<yellow<<bold<<"CELL 118"<<reset<<std::endl;
        std::cout<<"p0 = "<<p0 << " , p1 = "<<p1<<std::endl;
        std::cout<<"m_half = "<<m_half << " , m_half_bis = "<<m_half_bis<<std::endl;
        std::cout<<"q = "<<q << " , q_bis = "<<q_bis<<std::endl;

        std::cout<<"pm = "<<pm << " , level_set_function(pm) = "<<lm<<std::endl;
        std::cout<<"ps1 = "<<ps1 << " , level_set_function(ps1) = "<<ls1<<std::endl;
        std::cout<<"ps2 = "<<ps2 << " , level_set_function(ps2) = "<<ls2<<std::endl;
    }
    */
        auto pt_bdry = search_boundary3(msh, cl, pm, m_half, q, lm, level_set_function, iso_val_interface);
        auto lm_bdry = level_set_function(pt_bdry, msh, cl);
/*
    if( offset(msh,cl) == 118 )
        std::cout<<"pt_bdry = "<<pt_bdry << " , level_set_function(lm_bdry) = "<<lm_bdry<<std::endl;
    */
//std::cout<<"pm = "<<pm << " , level_set_function(pm) = "<<lm<<std::endl;
//std::cout<<"ps1 = "<<ps1 << " , level_set_function(ps1) = "<<ls1<<std::endl;
//std::cout<<"ps2 = "<<ps2 << " , level_set_function(ps2) = "<<ls2<<std::endl;


        point_type ip;
// std::cout<<"the node of interface are "<<p0<<" and "<<p1<<". I search pm= "<<pm<<" in which phi = "<<lm<<" and ps1 e ps2 "<<ps1<<" and "<<ps2<<"equal to "<<ls1<<" , "<<ls2<<std::endl;
        if (pt_in_cell(msh, ps1, cl) && (!((lm >= iso_val_interface && ls1 >= iso_val_interface) ||
                                           (lm < iso_val_interface && ls1 < iso_val_interface)))) {
            auto threshold = diameter(msh, cl) / 1e20;
//auto threshold = diameter(msh, cl) / 1e10;
            ip = find_zero_crossing_in_cell3(pm, ps1, level_set_function, threshold, msh, cl);
//std::cout<<"OLD 1"<<std::endl;
        } else if (pt_in_cell(msh, ps2, cl) && (!((lm >= iso_val_interface && ls2 >= iso_val_interface) ||
                                                  (lm < iso_val_interface && ls2 < iso_val_interface)))) {
            auto threshold = diameter(msh, cl) / 1e20;
//auto threshold = diameter(msh, cl) / 1e10;
            ip = find_zero_crossing_in_cell3(pm, ps2, level_set_function, threshold, msh, cl);
//std::cout<<"OLD 2"<<std::endl;
        } else if (pt_in_cell(msh, pt_bdry, cl) && (!((lm >= iso_val_interface && lm_bdry >= iso_val_interface) ||
                                                      (lm < iso_val_interface && lm_bdry < iso_val_interface)))) {
            auto threshold = diameter(msh, cl) / 1e20;
//auto threshold = diameter(msh, cl) / 1e10;
            ip = find_zero_crossing_in_cell3(pm, pt_bdry, level_set_function, threshold, msh, cl);
//std::cout<<"BDRY NEW"<<std::endl;
        } else {
//throw std::logic_error("interface not found in search range");
//std::cout<<yellow<<bold<< "In cell "<<offset(msh,cl)<<" ---> implementing linear approximation. INTERFACE NOT FOUND."<<reset<<std::endl;
//ip = pm;
            std::cout << yellow << bold << "In cell " << offset(msh, cl)
                      << " ---> implementing MINIMISATION ERROR APPROXIMATION. INTERFACE NOT FOUND." << reset << std::endl;
            point_type ret;
            T val_min = 1e10;
            if (pt_in_cell(msh, ps1, cl) && std::abs(ls1 - iso_val_interface) < val_min) {
                val_min = std::abs(ls1);
                ret = ps1;
                std::cout << "ps1 = " << ps1 << " , ls1 = " << ls1 << std::endl;

            }
            if (pt_in_cell(msh, ps2, cl) && std::abs(ls2 - iso_val_interface) < val_min) {
                val_min = std::abs(ls2);
                ret = ps2;
                std::cout << "ps2 = " << ps2 << " , ls2 = " << ls2 << std::endl;
            }
            if (pt_in_cell(msh, pt_bdry, cl) && std::abs(lm_bdry - iso_val_interface) < val_min) {
                val_min = std::abs(lm_bdry);
                ret = pt_bdry;
                std::cout << "ppt_bdrys1 = " << pt_bdry << " , lm_bdry = " << lm_bdry << std::endl;
            }
            if (pt_in_cell(msh, pm, cl) && std::abs(lm - iso_val_interface) < val_min) {
                val_min = std::abs(lm);
                ret = pm;
                std::cout << "pm = " << ps1 << " , lm = " << ls1 << std::endl;
            }
            std::cout << "ret = " << ret << std::endl;
            ip = ret;

        }
/*
    if( offset(msh,cl) == 118 )
        std::cout<<"POINT INTERFACE Ip = "<<ip <<  " in pos = "<<mid<<std::endl;
    */
        interface_tmp.at(mid) = ip;

        refine_interface_pro3_disp_subcell(msh, cl, level_set_function, interface_tmp, min, mid);
        refine_interface_pro3_disp_subcell(msh, cl, level_set_function, interface_tmp, mid, max);
    }

    template<typename T, size_t ET, typename Function>
    void
    refine_interface_pro3_curve_para(cuthho_mesh <T, ET> &msh, const Function &level_set_function, size_t levels,
                                     size_t degree_curve) {

        size_t interface_points;

        if (levels == 0 && degree_curve == 1)
            return;

//if (levels == 0 && degree_curve % 2 == 0 )
//   interface_points = degree_curve+1 ;

        interface_points = iexp_pow(2, levels) * degree_curve;

        for (auto &cl: msh.cells) {
            if (!is_cut(msh, cl))
                continue;


            cl.user_data.interface.resize(interface_points + 1);
            cl.user_data.interface.at(0) = cl.user_data.p0;
            cl.user_data.interface.at(interface_points) = cl.user_data.p1;
            if (degree_curve % 2 == 0)
                refine_interface_pro3(msh, cl, level_set_function, 0, interface_points);
            else {
//std::cout<<"--> Attention it works just for P3!!! "<<std::endl;
                size_t num_subcl = iexp_pow(2, levels);
                std::vector <point<T, 2>> interface_tmp(num_subcl + 1);
                interface_tmp.at(0) = cl.user_data.p0;
                interface_tmp.at(num_subcl) = cl.user_data.p1;
                if (levels > 0) {
                    refine_interface_pro3_disp_subcell(msh, cl, level_set_function, interface_tmp, 0, num_subcl);
                }
                int counter = 0;
//std::cout<<"cl.user_data.p0 = "<<cl.user_data.p0<<" ,  cl.user_data.p1 = "<< cl.user_data.p1<<std::endl;
//std::cout<<"interface_tmp[0] = "<<interface_tmp[0]<<" , interface_tmp[1] = "<< interface_tmp[1]<<" , interface_tmp[2] = "<< interface_tmp[2]<<std::endl;
                for (auto first_pt = interface_tmp.begin(); first_pt < interface_tmp.end() - 1; first_pt++) {
                    auto second_pt = first_pt + 1;
                    int counter_fin = counter + degree_curve;
                    T fraz = 1.0 / degree_curve;
//std::cout<<"first_pt = "<<*first_pt<<" ,  second_pt = "<< *second_pt<<std::endl;
                    refine_interface_pro3_disp(msh, cl, level_set_function, first_pt, second_pt, counter, counter_fin,
                                               degree_curve, fraz);

                    counter++;
                    if (counter < interface_points)
                        cl.user_data.interface.at(counter) = *second_pt;
                }
            }
        }
    }


    template<typename T, size_t ET, typename Function>
    void
    refine_interface_angle(cuthho_mesh <T, ET> &msh, typename cuthho_mesh<T, ET>::cell_type &cl,
                           const Function &level_set_function, size_t min, size_t max,
                           typename cuthho_mesh<T, ET>::point_type &p_init, T h, bool pos, int multiplicity, T angle0,
                           T angle1) {
        if ((max - min) < 2)
            return;

        typedef typename cuthho_mesh<T, ET>::point_type point_type;


        T angle_half = (angle0 + angle1) / 2.0;
        T m_half = tan(angle_half);

        std::cout << bold << yellow << "CHECK ANGLE --------> " << reset << "angle0 = " << angle0 * 180 / M_PI
                  << " , angle1 = " << angle1 * 180 / M_PI << " and angle_half = " << angle_half * 180 / M_PI << std::endl;

/*
    // In the case h is not long enough!
    T h_max = 0.0;
    if(multiplicity > 1  )
    {
        T cateto_min = 2*m_half/h ;
        h_max = std::max( h , sqrt( pow(cateto_min,2) + pow(h,2)/4.0 ) );
    }
    */

// CASE h:
        T val = sqrt(pow(h, 2) / (1 + pow(m_half, 2)));

        T x_new0 = p_init.x() + val;
        T y_new0 = p_init.y() + (x_new0 - p_init.x()) * m_half;
        point_type pt_half0 = point_type(x_new0, y_new0);

        auto lm0 = level_set_function(pt_half0, msh, cl);

        T x_new1 = p_init.x() - val;
        T y_new1 = p_init.y() + (x_new1 - p_init.x()) * m_half;
        point_type pt_half1 = point_type(x_new1, y_new1);

        auto lm1 = level_set_function(pt_half1, msh, cl);




// CASE h_max = h*sqrt(2)
        T h_max = h * sqrt(2.0);
        T val_max = sqrt(pow(h_max, 2) / (1 + pow(m_half, 2)));
        T x_new_max0 = p_init.x() + val_max;
        T y_new_max0 = p_init.y() + (x_new_max0 - p_init.x()) * m_half;
        T x_new_max1 = p_init.x() - val_max;
        T y_new_max1 = p_init.y() + (x_new_max1 - p_init.x()) * m_half;


        point_type pt_half_max0 = point_type(x_new_max0, y_new_max0);
        point_type pt_half_max1 = point_type(x_new_max1, y_new_max1);

        auto lm_max0 = level_set_function(pt_half_max0, msh, cl);
        auto lm_max1 = level_set_function(pt_half_max1, msh, cl);

// CASE h_min = h/2
        T h_min = h / 2.0;
        T val_min = sqrt(pow(h_min, 2) / (1 + pow(m_half, 2)));
        T x_new_min0 = p_init.x() + val_min;
        T y_new_min0 = p_init.y() + (x_new_min0 - p_init.x()) * m_half;
        T x_new_min1 = p_init.x() - val_min;
        T y_new_min1 = p_init.y() + (x_new_min1 - p_init.x()) * m_half;


        point_type pt_half_min0 = point_type(x_new_min0, y_new_min0);
        point_type pt_half_min1 = point_type(x_new_min1, y_new_min1);

        auto lm_min0 = level_set_function(pt_half_min0, msh, cl);
        auto lm_min1 = level_set_function(pt_half_min1, msh, cl);



// CASE PT on the boudary
        T q = p_init.y() - m_half * p_init.x();

        auto pt_bdry = search_boundary(msh, cl, p_init, m_half, q);
        auto lm_bdry = level_set_function(pt_bdry, msh, cl);


        size_t mid = (max + min) / 2;
        point_type ip;
        auto p0 = cl.user_data.interface.at(min);
        auto p1 = cl.user_data.interface.at(max);
        auto pm = (p0 + p1) / 2.0;
//std::cout<<"p_init = "<<p_init<< " level_set(p_init) = "<<level_set_function(p_init,msh,cl)<<std::endl;


        if (pt_in_cell(msh, pt_half0, cl) && !((lm0 >= 0 && pos == TRUE) || (lm0 < 0 && pos == FALSE))) {

            auto threshold = diameter(msh, cl) / 1e20;
            ip = find_zero_crossing_in_cell(p_init, pt_half0, level_set_function, threshold, msh, cl);
            std::cout << "NORMAL + --> pt_half0 = " << pt_half0 << " and lm0 = " << lm0 << " ------> ip = " << ip
                      << std::endl;
        } else if (pt_in_cell(msh, pt_half1, cl) && !((lm1 >= 0 && pos == TRUE) || (lm1 < 0 && pos == FALSE))) {

            auto threshold = diameter(msh, cl) / 1e20;
            ip = find_zero_crossing_in_cell(p_init, pt_half1, level_set_function, threshold, msh, cl);
            std::cout << "NORMAL - --> pt_half1 = " << pt_half1 << " and lm1 = " << lm1 << " ------> ip = " << ip
                      << std::endl;

        } else if (pt_in_cell(msh, pt_bdry, cl) && !((lm_bdry >= 0 && pos == TRUE) || (lm_bdry < 0 && pos == FALSE))) {
//std::cout<<"CHECK IF MIN POINT (WITH -) IS IN CELL"<<std::endl;

            auto threshold = diameter(msh, cl) / 1e20;
            ip = find_zero_crossing_in_cell(p_init, pt_bdry, level_set_function, threshold, msh, cl);
            std::cout << "BDRY - --> pt_bdry = " << pt_bdry << " and lm_bdry = " << lm_bdry << " --> ip = " << ip
                      << std::endl;

        } else if (pt_in_cell(msh, pt_half_max0, cl) && !((lm_max0 >= 0 && pos == TRUE) || (lm_max0 < 0 && pos == FALSE))) {
//std::cout<<"CHECK IF MAX POINT (WITH +) IS IN CELL"<<std::endl;

            auto threshold = diameter(msh, cl) / 1e20;
            ip = find_zero_crossing_in_cell(p_init, pt_half_max0, level_set_function, threshold, msh, cl);
            std::cout << "MAX + --> pt_max0 = " << pt_half_max0 << " and lm_max0 = " << lm_max0 << " --> ip = " << ip
                      << std::endl;

        } else if (pt_in_cell(msh, pt_half_max1, cl) && !((lm_max1 >= 0 && pos == TRUE) || (lm_max1 < 0 && pos == FALSE))) {
//std::cout<<"CHECK IF MAX POINT (WITH -) IS IN CELL"<<std::endl;

            auto threshold = diameter(msh, cl) / 1e20;
            ip = find_zero_crossing_in_cell(p_init, pt_half_max1, level_set_function, threshold, msh, cl);
            std::cout << "MAX - --> pt_max1 = " << pt_half_max1 << " and lm_max1 = " << lm_max1 << " --> ip = " << ip
                      << std::endl;

        } else if (pt_in_cell(msh, pt_half_min0, cl) && !((lm_min0 >= 0 && pos == TRUE) || (lm_min0 < 0 && pos == FALSE))) {
//std::cout<<"CHECK IF MIN POINT (WITH +) IS IN CELL"<<std::endl;

            auto threshold = diameter(msh, cl) / 1e20;
            ip = find_zero_crossing_in_cell(p_init, pt_half_min0, level_set_function, threshold, msh, cl);
            std::cout << "MIN + --> pt_min0 = " << pt_half_min0 << " and lm_min0 = " << lm_min0 << " --> ip = " << ip
                      << std::endl;

        } else if (pt_in_cell(msh, pt_half_min1, cl) && !((lm_min1 >= 0 && pos == TRUE) || (lm_min1 < 0 && pos == FALSE))) {
//std::cout<<"CHECK IF MIN POINT (WITH -) IS IN CELL"<<std::endl;

            auto threshold = diameter(msh, cl) / 1e20;
            ip = find_zero_crossing_in_cell(p_init, pt_half_min1, level_set_function, threshold, msh, cl);
            std::cout << "MIN - --> pt_min1 = " << pt_half_min1 << " and lm_min1 = " << lm_min1 << " --> ip = " << ip
                      << std::endl;

        } else {
// IN THE CASE I DON'T FIND THE POINT I CONSIDER IT LINEAR

            std::cout << "-----> ATTENTION: INTERFACE_REFINE3-> POINT DID NOT FIND, LINEAR APPROXIMATION EMPLOYED!"
                      << std::endl;
            std::cout << "p_init = " << p_init << " level_set(p_init) = " << level_set_function(p_init, msh, cl)
                      << std::endl;
            std::cout << "CASE + : pt_half0 = " << pt_half0 << " and lm0 = " << lm0 << std::endl;
            std::cout << "CASE - : pt_half1 = " << pt_half1 << " and lm1 = " << lm1 << std::endl;
            std::cout << "CASE MAX+: pt_half_max0 = " << pt_half_max0 << " and lm_max0 = " << lm_max0 << std::endl;
            std::cout << "CASE MAX-: pt_half_max1 = " << pt_half_max1 << " and lm_max1 = " << lm_max1 << std::endl;
            std::cout << "CASE MIN +: pt_half_min0 = " << pt_half_min0 << " and lm_min0 = " << lm_min0 << std::endl;
            std::cout << "CASE MIN -: pt_half_min1 = " << pt_half_min1 << " and lm_min1 = " << lm_min1 << std::endl;
            std::cout << "CASE BDRY: pt_bdry = " << pt_bdry << " and lm_bdry = " << lm_bdry << std::endl;
            std::cout << "--> ip = pm =  " << pm << std::endl;
            std::cout << "ATTENTION: INTERFACE_REFINE3-> POINT DID NOT FIND, LINEAR APPROXIMATION EMPLOYED! <-------"
                      << std::endl;
            ip = pm;
        }


        cl.user_data.interface.at(mid) = ip;

        refine_interface_angle(msh, cl, level_set_function, min, mid, p_init, h, pos, multiplicity, angle0, angle_half);
        refine_interface_angle(msh, cl, level_set_function, mid, max, p_init, h, pos, multiplicity, angle_half, angle1);
    }


    template<typename T, size_t ET, typename Function>
    void
    refine_interface_angle(cuthho_mesh <T, ET> &msh, const Function &level_set_function, size_t levels) {
        if (levels == 0)
            return;

        typedef typename cuthho_mesh<T, ET>::point_type point_type;
//typedef typename cuthho_mesh<T, ET>::node_type node_type;
        size_t interface_points = iexp_pow(2, levels);

        for (auto &cl: msh.cells) {
            if (!is_cut(msh, cl))
                continue;

            cl.user_data.interface.resize(interface_points + 1);
            cl.user_data.interface.at(0) = cl.user_data.p0;
            cl.user_data.interface.at(interface_points) = cl.user_data.p1;
            std::cout << yellow << bold << "--------------------> CELL = " << offset(msh, cl) << " <--------------------"
                      << reset << std::endl;
// ADDED BY STEFANO
            point_type pos0 = cl.user_data.p0;
            point_type pos1 = cl.user_data.p1;

            bool positive = TRUE;
            int multiplicity = 1;
            std::vector <size_t> position_neg, position_pos;
            T angle0, angle1;
            point_type pm;

            size_t counter = 0;
            for (auto &nd: nodes(msh, cl)) {

                if (nd.user_data.location == element_location::IN_NEGATIVE_SIDE) {
                    position_neg.push_back(counter);
                } else {

                    position_pos.push_back(counter);
                }
                counter++;
//std::cout<<"nd = "<<nd.ptid<<std::endl;
            }

/// FIND  EXTREME  ANGLES
            if (position_neg.size() == 1) {

                pm = points(msh, cl)[position_neg[0]];
                std::cout << "POINT" << pm << " , position_neg = " << position_neg[0] << std::endl;
                positive = FALSE;
                if (position_neg[0] == 0) {
                    angle0 = 0.0 * M_PI;
                    angle1 = M_PI / 2.0;
                } else if (position_neg[0] == 1) {
                    angle0 = M_PI / 2.0;
                    angle1 = M_PI;
                } else if (position_neg[0] == 2) {
                    angle0 = M_PI;
                    angle1 = 3.0 * M_PI / 2.0;
                } else if (position_neg[0] == 3) {
                    angle0 = 3.0 * M_PI / 2.0;
                    angle1 = 2.0 * M_PI;
                } else {
                    throw std::logic_error("POSITION ANGLE REFINE INTERFACE WRONG");
                }
            } else if (position_pos.size() == 1) {
                std::cout << yellow << bold << "CASO POSITIVO MA TOGLIEREI E METTEREI SOLO CASI NEGATIVI!!!" << reset
                          << std::endl;
                pm = points(msh, cl)[position_pos[0]];
                std::cout << "POINT" << pm << " , position_pos = " << position_pos[0] << std::endl;
                if (position_pos[0] == 0) {
                    angle0 = 0.0 * M_PI;
                    angle1 = M_PI / 2.0;
                } else if (position_pos[0] == 1) {
                    angle0 = M_PI / 2.0;
                    angle1 = M_PI;
                } else if (position_pos[0] == 2) {
                    angle0 = M_PI;
                    angle1 = 3.0 * M_PI / 2.0;
                } else if (position_pos[0] == 3) {
                    angle0 = 3.0 * M_PI / 2.0;
                    angle1 = 2.0 * M_PI;
                } else {
                    throw std::logic_error("POSITION ANGLE REFINE INTERFACE WRONG");
                }

            } else {
//std::cout<<"sono qua 4 NEW"<<std::endl;
//if(plus_close(p0,))
// MULTIPLICITY 2 -> #pos_point = #neg_point = 2
                if (position_neg[0] == 0 && position_neg[1] == 1) {
                    positive = FALSE;
                    pm = (points(msh, cl)[position_neg[0]] + points(msh, cl)[position_neg[1]]) / 2.0;
                    std::cout << "POINT " << pm << " , position_neg[0][1] --> " << position_neg[0] << " , "
                              << position_neg[1] << std::endl;
                    multiplicity = 2;
                    angle0 = atan((pos0.y() - pm.y()) / (pos0.x() - pm.x()));
                    angle1 = atan((pos1.y() - pm.y()) / (pos1.x() - pm.x()));
                    if (angle0 < 0)
                        angle0 += M_PI;
                    else
                        angle1 += M_PI;
                } else if (position_neg[0] == 1 && position_neg[1] == 2) {
                    positive = FALSE;
                    pm = (points(msh, cl)[position_neg[0]] + points(msh, cl)[position_neg[1]]) / 2.0;
                    std::cout << "POINT " << pm << " , position_neg[0][1] --> " << position_neg[0] << " , "
                              << position_neg[1] << std::endl;
                    multiplicity = 2;
                    angle0 = M_PI + atan((pos0.y() - pm.y()) / (pos0.x() - pm.x()));
                    angle1 = M_PI + atan((pos1.y() - pm.y()) / (pos1.x() - pm.x()));

                } else if (position_neg[0] == 2 && position_neg[1] == 3) {
                    positive = FALSE;
                    pm = (points(msh, cl)[position_neg[0]] + points(msh, cl)[position_neg[1]]) / 2.0;
                    std::cout << "POINT " << pm << " , position_neg[0][1] --> " << position_neg[0] << " , "
                              << position_neg[1] << std::endl;
                    multiplicity = 2;
                    angle0 = atan((pos0.y() - pm.y()) / (pos0.x() - pm.x()));
                    angle1 = atan((pos1.y() - pm.y()) / (pos1.x() - pm.x()));
                    if (angle0 > 0)
                        angle0 += M_PI;
                    else
                        angle1 += M_PI;

                    if (angle0 < 0)
                        angle0 = 2.0 * M_PI + angle0;
                    else
                        angle1 = 2.0 * M_PI + angle1;

                } else if (position_neg[0] == 0 && position_neg[1] == 3) {
                    positive = FALSE;
                    pm = (points(msh, cl)[position_neg[0]] + points(msh, cl)[position_neg[1]]) / 2.0;
                    std::cout << "POINT " << pm << " , position_neg[0][1] --> " << position_neg[0] << " , "
                              << position_neg[1] << std::endl;
                    multiplicity = 2;
                    angle0 = atan((pos0.y() - pm.y()) / (pos0.x() - pm.x()));
                    angle1 = atan((pos1.y() - pm.y()) / (pos1.x() - pm.x()));

                } else if (position_neg[0] == 0 && position_neg[1] == 2) {
//positive = FALSE ;
                    pm = points(msh, cl)[1];
                    std::cout << "POINT " << pm << " , position_neg[0][1] --> " << position_neg[0] << " , "
                              << position_neg[1] << std::endl;
//multiplicity = 2 ;
                    angle0 = M_PI / 2.0;
                    angle1 = M_PI;

                } else if (position_neg[0] == 1 && position_neg[1] == 3) {
//positive = FALSE ;
                    pm = points(msh, cl)[0];
                    std::cout << "POINT " << pm << " , position_neg[0][1] --> " << position_neg[0] << " , "
                              << position_neg[1] << std::endl;
//multiplicity = 2 ;
                    angle0 = 0.0 * M_PI;
                    angle1 = M_PI / 2.0;

                } else {
                    throw std::logic_error("POSITION ANGLE REFINE INTERFACE WRONG");
                }
            }


            if (angle0 > angle1) {
                T tmp = angle1;
                angle1 = angle0;
                angle0 = tmp;
            }
//std::cout<<"CHECK ANGLE --------> angle0 = "<<angle0<< " and angle1 = "<<angle1<<std::endl;


            auto checK_sign = level_set_function(pm, msh, cl);

            if ((positive == FALSE && std::signbit(checK_sign) == 0) ||
                (positive == TRUE && std::signbit(checK_sign) == 1)) {
                std::cout << "LEVEL SET(Pm) = " << checK_sign << " and sign used is = " << positive << " and signbit is "
                          << std::signbit(checK_sign) << std::endl;
                throw std::logic_error("HO FATTO ERRORE IN POSITIVE SIGN CHECKING");
            }


            T h = std::min(level_set_function.params.hx(), level_set_function.params.hy());


            refine_interface_angle(msh, cl, level_set_function, 0, interface_points, pm, h, positive, multiplicity, angle0,
                                   angle1);


            std::cout << "LIMIT CELL " << offset(msh, cl) << " are:" << std::endl;
            std::cout << "pt[0] = " << points(msh, cl)[0] << " , pt[1] = " << points(msh, cl)[1] << " , pt[2] = "
                      << points(msh, cl)[2] << " , pt[3] = " << points(msh, cl)[3] << std::endl;

            for (size_t i_int = 0; i_int < interface_points + 1; i_int++)
                std::cout << "refined points are p = " << cl.user_data.interface.at(i_int) << std::endl;
            std::cout << "--------------------> CELL = " << offset(msh, cl) << "<--------------------" << std::endl;
        }

    }



    template<typename T, typename Mesh>
    bool
    pt_in_cell(const Mesh &msh, const point<T, 2> &, const typename Mesh::cell_type &);

    template<typename T, typename Mesh>
    std::vector <point<T, 2>>
    equidistriduted_nodes(const Mesh &, const typename Mesh::cell_type &, size_t);

    template<typename T, typename Mesh>
    std::vector <point<T, 2>>
    equidistriduted_nodes_subcell(const Mesh &,
                                  const typename Mesh::cell_type &,
                                  size_t, const std::vector <size_t> &);





    int binomial_coeff_fx(int n, int k) {

        int C[n + 1][k + 1];
        int i, j;

// Caculate value of Binomial Coefficient
// in bottom up manner
        for (i = 0; i <= n; i++) {
            for (j = 0; j <= std::min(i, k); j++) {
// Base Cases
                if (j == 0 || j == i)
                    C[i][j] = 1;

// Calculate value using previously
// stored values
                else
                    C[i][j] = C[i - 1][j - 1] +
                              C[i - 1][j];
            }
        }

        return C[n][k];
    }


    template<typename FonctionD, typename Mesh>
    void
    testing_level_set(const Mesh &msh, const FonctionD &level_set_disc, std::string &folder) {
        typedef typename Mesh::point_type point_type;
        postprocess_output<double> postoutput1;
        double valueD, derDx, derDy;
        Eigen::Matrix<double, 2, 1> derD;
        point<double, 2> node;
        size_t N, M;

        N = 4; //80 points to see also the interface!!!
        M = 4; //80 points to see also the interface!!!
        auto test_disc = std::make_shared < gnuplot_output_object < double > > (folder + "testing_interface_disc.dat");


        auto test_disc_gradX = std::make_shared < gnuplot_output_object < double > > (folder + "testing_der_discX.dat");

        auto test_disc_gradY = std::make_shared < gnuplot_output_object < double > > (folder + "testing_der_discY.dat");

        auto test_profile_disc = std::make_shared < gnuplot_output_object < double > > (folder + "test_profile_disc.dat");

        auto test_profile_obliq_disc =
                std::make_shared < gnuplot_output_object < double > > (folder + "test_profile_obliq.dat");


        auto test_disc_grad = std::make_shared < gnuplot_output_object < double > > (folder + "testing_der_disc_fin.dat");

        for (auto &cl: msh.cells) {
            auto pts = points(msh, cl);
            auto pt0_x = pts[0].x();
            auto pt1_x = pts[1].x();
            auto pt0_y = pts[0].y();
            auto pt1_y = pts[3].y();
            for (size_t i = 0; i <= N; i++) {
                double px = pt0_x + i * ((pt1_x - pt0_x) / N);
                for (size_t j = 0; j <= M; j++) {

                    double py = pt0_y + j * ((pt1_y - pt0_y) / M);
                    node = point_type(px, py);
                    valueD = level_set_disc(node, msh, cl);
                    derD = level_set_disc.gradient(node, msh, cl);


                    derDx = derD(0);
                    derDy = derD(1);


                    test_disc->add_data(node, valueD);

                    test_disc_gradX->add_data(node, derDx);
                    test_disc_gradY->add_data(node, derDy);
                    test_disc_grad->add_data(node, sqrt(derDx * derDx + derDy * derDy));

                }


            }
        }

        for (size_t i = 0; i <= 100; i++) {
            double px = i * ((1.0) / 100.0);
            node = point_type(px, 0.5);
            valueD = level_set_disc(node);
            test_profile_disc->add_data(node, valueD);
        }

        for (size_t i = 0; i <= 100; i++) {
            double px = i * ((1.0) / 100.0);
            node = point_type(px, px);
            valueD = level_set_disc(node);
            test_profile_obliq_disc->add_data(node, valueD);
        }


        postoutput1.add_object(test_disc);

        postoutput1.add_object(test_disc_grad);
        postoutput1.add_object(test_disc_gradX);

        postoutput1.add_object(test_disc_gradY);

        postoutput1.add_object(test_profile_disc);

        postoutput1.add_object(test_profile_obliq_disc);


        postoutput1.write();

    }

    template<typename FonctionD, typename Mesh>
    void
    testing_level_set_mapped(const Mesh &msh, const FonctionD &level_set_disc) {
        typedef typename Mesh::point_type point_type;
        postprocess_output<double> postoutput1;
        double valueD, derDx, derDy;
        Eigen::Matrix<double, 2, 1> derD;
        point<double, 2> node;
        size_t N, M;

        N = 4; //80 points to see also the interface!!!
        M = 4; //80 points to see also the interface!!!
        auto test_disc = std::make_shared < gnuplot_output_object < double > > ("testing_interface_disc_mapped.dat");


        auto test_disc_gradX = std::make_shared < gnuplot_output_object < double > > ("testing_der_discX_mapped.dat");

        auto test_disc_gradY = std::make_shared < gnuplot_output_object < double > > ("testing_der_discY_mapped.dat");

        auto test_profile_disc = std::make_shared < gnuplot_output_object < double > > ("test_profile_disc_mapped.dat");

        for (auto &cl: msh.cells) {
            auto pts = points(msh, cl);
            auto pt0_x = pts[0].x();
            auto pt1_x = pts[1].x();
            auto pt0_y = pts[0].y();
            auto pt1_y = pts[3].y();
            for (size_t i = 0; i <= N; i++) {
                double px = pt0_x + i * ((pt1_x - pt0_x) / N);
                for (size_t j = 0; j <= M; j++) {

                    double py = pt0_y + j * ((pt1_y - pt0_y) / M);
                    node = point_type(px, py);
                    valueD = level_set_disc(node, msh, cl);
                    derD = level_set_disc.gradient(node, msh, cl);


                    derDx = derD(0);
                    derDy = derD(1);


                    test_disc->add_data(node, valueD);

                    test_disc_gradX->add_data(node, derDx);
                    test_disc_gradY->add_data(node, derDy);


                }


            }
        }

        for (size_t i = 0; i <= 100; i++) {
            double px = i * ((1.0) / 100.0);
            node = point_type(px, 0.5);
            valueD = level_set_disc(node);
            test_profile_disc->add_data(node, valueD);
        }

        postoutput1.add_object(test_disc);


        postoutput1.add_object(test_disc_gradX);

        postoutput1.add_object(test_disc_gradY);

        postoutput1.add_object(test_profile_disc);


        postoutput1.write();

    }

    template<typename FonctionD, typename Mesh>
    void
    testing_level_set_inverse_mapped(const Mesh &msh, const FonctionD &level_set_disc) {
        typedef typename Mesh::point_type point_type;
        postprocess_output<double> postoutput1;
        double valueD, derDx, derDy;
        Eigen::Matrix<double, 2, 1> derD;
        point<double, 2> node;
        size_t N, M;

        N = 4; //80 points to see also the interface!!!
        M = 4; //80 points to see also the interface!!!
        auto test_disc =
                std::make_shared < gnuplot_output_object < double > > ("testing_interface_disc_inverse_mapped.dat");


        auto test_disc_gradX =
                std::make_shared < gnuplot_output_object < double > > ("testing_der_discX_inverse_mapped.dat");

        auto test_disc_gradY =
                std::make_shared < gnuplot_output_object < double > > ("testing_der_discY_inverse_mapped.dat");

        auto test_profile_disc =
                std::make_shared < gnuplot_output_object < double > > ("test_profile_disc_inverse_mapped.dat");

        for (auto &cl: msh.cells) {
            auto pts = points(msh, cl);
            auto pt0_x = pts[0].x();
            auto pt1_x = pts[1].x();
            auto pt0_y = pts[0].y();
            auto pt1_y = pts[3].y();
            for (size_t i = 0; i <= N; i++) {
                double px = pt0_x + i * ((pt1_x - pt0_x) / N);
                for (size_t j = 0; j <= M; j++) {

                    double py = pt0_y + j * ((pt1_y - pt0_y) / M);
                    node = point_type(px, py);
                    valueD = level_set_disc(node, msh, cl);
                    derD = level_set_disc.gradient(node, msh, cl);


                    derDx = derD(0);
                    derDy = derD(1);


                    test_disc->add_data(node, valueD);

                    test_disc_gradX->add_data(node, derDx);
                    test_disc_gradY->add_data(node, derDy);


                }


            }
        }

        for (size_t i = 0; i <= 100; i++) {
            double px = i * ((1.0) / 100.0);
            node = point_type(px, 0.5);
            valueD = level_set_disc(node);
            test_profile_disc->add_data(node, valueD);
        }

        postoutput1.add_object(test_disc);


        postoutput1.add_object(test_disc_gradX);

        postoutput1.add_object(test_disc_gradY);

        postoutput1.add_object(test_profile_disc);


        postoutput1.write();

    }

    template<typename FonctionD, typename Mesh, typename FonctionA>
    void
    testing_level_set(const Mesh msh, const FonctionD &level_set_disc, const FonctionA &level_set_anal) {
        typedef typename Mesh::point_type point_type;
        postprocess_output<double> postoutput1;
        double valueD, valueA, derDx, derDy, derAx, derAy, value_profile, value_profile_anal;
        Eigen::Matrix<double, 2, 1> derD, derA;
        point<double, 2> node;
        size_t N, M;
        std::cout << "In testing_level_set I need 80x80 points to see the interface. Now is 40x40, faster." << std::endl;
        N = 4; //80 points to see also the interface!!!
        M = 4; //80 points to see also the interface!!!
        auto test_disc = std::make_shared < gnuplot_output_object < double > > ("testing_interface_disc.dat");
        auto test_anal = std::make_shared < gnuplot_output_object < double > > ("testing_interface_anal.dat");

        auto test_disc_gradX = std::make_shared < gnuplot_output_object < double > > ("testing_der_discX.dat");
        auto test_anal_gradX = std::make_shared < gnuplot_output_object < double > > ("testing_der_analX.dat");

        auto test_disc_gradY = std::make_shared < gnuplot_output_object < double > > ("testing_der_discY.dat");
        auto test_anal_gradY = std::make_shared < gnuplot_output_object < double > > ("testing_der_analY.dat");

        auto test_anal_grad = std::make_shared < gnuplot_output_object < double > > ("testing_der_anal.dat");
        auto test_disc_grad = std::make_shared < gnuplot_output_object < double > > ("testing_der_disc.dat");

        auto test_profile_disc = std::make_shared < gnuplot_output_object < double > > ("test_profile_disc.dat");
        auto test_profile_anal = std::make_shared < gnuplot_output_object < double > > ("test_profile_anal.dat");

        auto pt_c_x = level_set_anal.alpha;
        auto pt_c_y = level_set_anal.beta;
        auto pt_c = point_type(pt_c_x, pt_c_y);
        double py = 0.5;
//double iso_val_interface = level_set_disc.iso_val_interface ;
        for (auto &cl: msh.cells) {
            auto pts = points(msh, cl);
            auto pt0_x = pts[0].x();
            auto pt1_x = pts[1].x();
            auto pt0_y = pts[0].y();
            auto pt1_y = pts[3].y();

            if (pt_in_cell(msh, pt_c, cl)) {
                test_disc_gradX->add_data(node, derDx);
                test_anal_gradX->add_data(node, derAx);
                test_disc_gradY->add_data(node, derDy);
                test_anal_gradY->add_data(node, derAy);

                test_anal_grad->add_data(node, sqrt(derAx * derAx + derAy * derAy));
                test_disc_grad->add_data(node, sqrt(derDx * derDx + derDy * derDy));
            }

            for (size_t i = 0; i <= N; i++) {
                double px = pt0_x + i * ((pt1_x - pt0_x) / N);
                for (size_t j = 0; j <= M; j++) {
                    double py = pt0_y + j * ((pt1_y - pt0_y) / M);
                    node = point_type(px, py);

                    valueA = level_set_anal(node);
                    valueD = level_set_disc(node, msh, cl);


                    test_disc->add_data(node, valueD);
                    test_anal->add_data(node, valueA);


                    derA = level_set_anal.gradient(node);
                    derD = level_set_disc.gradient(node, msh, cl);
                    derDx = derD(0);
                    derDy = derD(1);
                    derAx = derA(0);
                    derAy = derA(1);

                    test_disc_gradX->add_data(node, derDx);
                    test_anal_gradX->add_data(node, derAx);
                    test_disc_gradY->add_data(node, derDy);
                    test_anal_gradY->add_data(node, derAy);

                    test_anal_grad->add_data(node, sqrt(derAx * derAx + derAy * derAy));
                    test_disc_grad->add_data(node, sqrt(derDx * derDx + derDy * derDy));

                }

                node = point_type(px, py);
                value_profile = level_set_disc(node, msh, cl);
                test_profile_disc->add_data(node, value_profile);
                value_profile_anal = level_set_anal(node);
                test_profile_anal->add_data(node, value_profile_anal);
            }

        }


        postoutput1.add_object(test_disc);
        postoutput1.add_object(test_anal);

        postoutput1.add_object(test_disc_gradX);
        postoutput1.add_object(test_anal_gradX);
        postoutput1.add_object(test_disc_gradY);
        postoutput1.add_object(test_anal_gradY);

        postoutput1.add_object(test_anal_grad);
        postoutput1.add_object(test_disc_grad);


        postoutput1.add_object(test_profile_disc);
        postoutput1.add_object(test_profile_anal);

        postoutput1.write();

    }


    template<typename FonctionD, typename Mesh, typename FonctionA>
    void
    testing_level_set_disc(const Mesh msh, const FonctionD &level_set_disc, const FonctionA &level_set_anal) {
        typedef typename Mesh::point_type point_type;
        postprocess_output<double> postoutput1;
        double valueD, valueA, derDx, derDy, derAx, derAy, value_profile, value_profile_anal;
        Eigen::Matrix<double, 2, 1> derD, derA;
        point<double, 2> node;
        size_t N, M;
        std::cout << "In testing_level_set I need 80x80 points to see the interface. Now is 40x40, faster." << std::endl;
        N = 4; //80 points to see also the interface!!!
        M = 4; //80 points to see also the interface!!!
        auto test_disc = std::make_shared < gnuplot_output_object < double > > ("testing_interface_disc.dat");
        auto test_anal = std::make_shared < gnuplot_output_object < double > > ("testing_interface_disc_tmp.dat");

        auto test_disc_gradX = std::make_shared < gnuplot_output_object < double > > ("testing_der_discX.dat");
        auto test_anal_gradX = std::make_shared < gnuplot_output_object < double > > ("testing_der_disc_tmp_X.dat");

        auto test_disc_gradY = std::make_shared < gnuplot_output_object < double > > ("testing_der_discY.dat");
        auto test_anal_gradY = std::make_shared < gnuplot_output_object < double > > ("testing_der_disc_tmp_Y.dat");

        auto test_profile_disc = std::make_shared < gnuplot_output_object < double > > ("test_profile_disc.dat");
        auto test_profile_anal = std::make_shared < gnuplot_output_object < double > > ("test_profile_disc_tmp.dat");


        double py = 0.5;
//double iso_val_interface = level_set_disc.iso_val_interface ;
        for (auto &cl: msh.cells) {
            auto pts = points(msh, cl);
            auto pt0_x = pts[0].x();
            auto pt1_x = pts[1].x();
            auto pt0_y = pts[0].y();
            auto pt1_y = pts[3].y();

            for (size_t i = 0; i <= N; i++) {
                double px = pt0_x + i * ((pt1_x - pt0_x) / N);
                for (size_t j = 0; j <= M; j++) {
                    double py = pt0_y + j * ((pt1_y - pt0_y) / M);
                    node = point_type(px, py);

                    valueA = level_set_anal(node, msh, cl);
                    valueD = level_set_disc(node, msh, cl);


                    test_disc->add_data(node, valueD);
                    test_anal->add_data(node, valueA);


                    derA = level_set_anal.gradient(node, msh, cl);
                    derD = level_set_disc.gradient(node, msh, cl);
                    derDx = derD(0);
                    derDy = derD(1);
                    derAx = derA(0);
                    derAy = derA(1);

                    test_disc_gradX->add_data(node, derDx);
                    test_anal_gradX->add_data(node, derAx);
                    test_disc_gradY->add_data(node, derDy);
                    test_anal_gradY->add_data(node, derAy);


                }

                node = point_type(px, py);
                value_profile = level_set_disc(node, msh, cl);
                test_profile_disc->add_data(node, value_profile);
                value_profile_anal = level_set_anal(node, msh, cl);
                test_profile_anal->add_data(node, value_profile_anal);
            }

        }


        postoutput1.add_object(test_disc);
        postoutput1.add_object(test_anal);

        postoutput1.add_object(test_disc_gradX);
        postoutput1.add_object(test_anal_gradX);
        postoutput1.add_object(test_disc_gradY);
        postoutput1.add_object(test_anal_gradY);

        postoutput1.add_object(test_profile_disc);
        postoutput1.add_object(test_profile_anal);

        postoutput1.write();

    }


    template<typename FonctionD, typename Mesh>
    void
    testing_level_set2_bis(const Mesh msh, const FonctionD &level_set_disc) {
        typedef typename Mesh::point_type point_type;
        postprocess_output<double> postoutput1;
        double valueD; // , derDx , derDy ;
        Eigen::Matrix<double, 2, 1> derD;
        point<double, 2> node;
        size_t N, M;
        std::cout
                << "In testing_level_set2 I need 80x80 points to see the interface. Now is 40x40, faster. No analytic solution."
                << std::endl;
        N = 40; //80 points to see also the interface!!!
        M = 40; //80 points to see also the interface!!!
        double valueD_gamma;
        auto test_disc = std::make_shared < gnuplot_output_object < double > > ("testing_interface_fin_disc_bis.dat");

        auto test_gamma_disc = std::make_shared < gnuplot_output_object < double > > ("test_gamma_disc_bis.dat");

        auto test_profile_disc = std::make_shared < gnuplot_output_object < double > > ("test_profile_disc_fin_bis.dat");

        double iso_val_interface = level_set_disc.iso_val_interface;

        for (size_t i = 0; i <= N; i++) {
            for (size_t j = 0; j <= M; j++) {
                double px = i * (1.0 / N);
                double py = j * (1.0 / M);
                node = point_type(px, py);

                if (std::abs(level_set_disc(node) - iso_val_interface) < 1 * 1e-3) {
                    valueD_gamma = 1;
                } else
                    valueD_gamma = 0;

                valueD = level_set_disc(node);


                if (py == 0.5) {
                    test_profile_disc->add_data(node, valueD);

                }


                test_disc->add_data(node, valueD);

                test_gamma_disc->add_data(node, valueD_gamma);


            }

        }
        postoutput1.add_object(test_disc);

        postoutput1.add_object(test_gamma_disc);

        postoutput1.add_object(test_profile_disc);

        postoutput1.write();

    }

    template<typename FonctionD, typename Mesh>
    void
    testing_level_set2(const Mesh &msh, const FonctionD &level_set_disc) {
        typedef typename Mesh::point_type point_type;
        postprocess_output<double> postoutput1;
        double valueD, value_profile;
        point<double, 2> node;
        size_t N, M;
        std::cout
                << "In testing_level_set2 I need 80x80 points to see the interface. Now is 40x40, faster. No analytic solution."
                << std::endl;
        N = 4; //80 points to see also the interface!!!
        M = 4; //80 points to see also the interface!!!

        auto test_disc = std::make_shared < gnuplot_output_object < double > > ("testing_interface_fin_disc.dat");

//auto test_gamma_disc = std::make_shared< gnuplot_output_object<double> >("test_gamma_disc.dat");

        auto test_profile_disc = std::make_shared < gnuplot_output_object < double > > ("test_profile_disc_fin.dat");

        double py = 0.5;
//double iso_val_interface = level_set_disc.iso_val_interface ;
        for (auto &cl: msh.cells) {
            auto pts = points(msh, cl);
            auto pt0_x = pts[0].x();
            auto pt1_x = pts[1].x();
            auto pt0_y = pts[0].y();
            auto pt1_y = pts[3].y();

            for (size_t i = 0; i <= N; i++) {
                double px = pt0_x + i * ((pt1_x - pt0_x) / N);
                for (size_t j = 0; j <= M; j++) {
                    double py = pt0_y + j * ((pt1_y - pt0_y) / M);
                    node = point_type(px, py);

                    valueD = level_set_disc(node, msh, cl);
                    test_disc->add_data(node, valueD);


                }

                node = point_type(px, py);
                value_profile = level_set_disc(node, msh, cl);
                test_profile_disc->add_data(node, value_profile);
            }

        }
        postoutput1.add_object(test_disc);

        postoutput1.add_object(test_profile_disc);

        postoutput1.write();

    }

    template<typename FonctionD, typename Mesh>
    void
    testing_level_set2(const Mesh msh, const FonctionD &level_set_disc, const FonctionD &level_set_fin) {
        typedef typename Mesh::point_type point_type;
        postprocess_output<double> postoutput1;
        double valueD, valueA; //, derDx , derDy , derAx , derAy;
        Eigen::Matrix<double, 2, 1> derD, derA;
        point<double, 2> node;
        size_t N, M;
        std::cout << "In testing_level_set2 I need 80x80 points to see the interface. Now is 40x40, faster." << std::endl;
        N = 40; //80 points to see also the interface!!!
        M = 40; //80 points to see also the interface!!!
        double valueA_gamma, valueD_gamma;
        auto test_disc = std::make_shared < gnuplot_output_object < double > > ("testing_interface_fin_disc.dat");
        auto test_anal = std::make_shared < gnuplot_output_object < double > > ("testing_interface_fin_anal.dat");
        auto test_gamma_disc = std::make_shared < gnuplot_output_object < double > > ("test_gamma_disc.dat");
        auto test_gamma_anal = std::make_shared < gnuplot_output_object < double > > ("test_gamma_anal.dat");

        auto test_profile_disc = std::make_shared < gnuplot_output_object < double > > ("test_profile_disc_fin.dat");
        auto test_profile_anal = std::make_shared < gnuplot_output_object < double > > ("test_profile_anal_fin.dat");
/*
    auto test_disc_gradX  = std::make_shared< gnuplot_output_object<double> >("testing_der_discX.dat");
    auto test_anal_gradX = std::make_shared< gnuplot_output_object<double> >("testing_der_analX.dat");

    auto test_disc_gradY  = std::make_shared< gnuplot_output_object<double> >("testing_der_discY.dat");
    auto test_anal_gradY = std::make_shared< gnuplot_output_object<double> >("testing_der_analY.dat");
    */
        for (size_t i = 0; i <= N; i++) {
            for (size_t j = 0; j <= M; j++) {
                double px = i * (1.0 / N);
                double py = j * (1.0 / M);
                node = point_type(px, py);

                if (std::abs(level_set_disc(node)) < 1 * 1e-3) {
                    valueD_gamma = 1;
                } else
                    valueD_gamma = 0;

                valueD = level_set_disc(node);

                if (std::abs(level_set_fin(node)) < 1 * 1e-3) {
                    valueA_gamma = 1;
                } else
                    valueA_gamma = 0;

                valueA = level_set_fin(node);


                if (py == 0.5) {
                    test_profile_disc->add_data(node, valueD);
                    test_profile_anal->add_data(node, valueA);
                }

/*
            derD = level_set_disc.gradient(node);
            derA = level_set_fin.gradient(node);


            derDx = derD(0);
            derDy = derD(1);
            derAx = derA(0);
            derAy = derA(1);
            */
                test_disc->add_data(node, valueD);
                test_anal->add_data(node, valueA);
                test_gamma_disc->add_data(node, valueD_gamma);
                test_gamma_anal->add_data(node, valueA_gamma);
/*
            test_disc_gradX->add_data(node,derDx);
            test_anal_gradX->add_data(node,derAx);
            test_disc_gradY->add_data(node,derDy);
            test_anal_gradY->add_data(node,derAy);
            */
            }

        }
        postoutput1.add_object(test_disc);
        postoutput1.add_object(test_anal);
        postoutput1.add_object(test_gamma_disc);
        postoutput1.add_object(test_gamma_anal);
        postoutput1.add_object(test_profile_disc);
        postoutput1.add_object(test_profile_anal);
/*
    postoutput1.add_object(test_disc_gradX);
    postoutput1.add_object(test_anal_gradX);
    postoutput1.add_object(test_disc_gradY);
    postoutput1.add_object(test_anal_gradY);
    */
        postoutput1.write();

    }

// Qualitative testing of the discrete level set function wrt the analytical one
    template<typename FonctionD, typename Mesh>
    void
    testing_level_set_mesh2(const Mesh msh, const FonctionD &level_set_disc, const FonctionD &level_set_fin) {
        typedef typename Mesh::point_type point_type;
        postprocess_output<double> postoutput1;
        double valueD, valueA; //, derDx , derDy , derAx , derAy;
        Eigen::Matrix<double, 2, 1> derD, derA;
        point<double, 2> node;
        size_t N, M;
        N = 40;
        M = 40;
        auto test_disc = std::make_shared < gnuplot_output_object < double > > ("testing_interface_fin_disc_mesh2.dat");
        auto test_anal = std::make_shared < gnuplot_output_object < double > > ("testing_interface_fin_anal_mesh2.dat");
/*
    auto test_disc_gradX  = std::make_shared< gnuplot_output_object<double> >("testing_der_discX.dat");
    auto test_anal_gradX = std::make_shared< gnuplot_output_object<double> >("testing_der_analX.dat");

    auto test_disc_gradY  = std::make_shared< gnuplot_output_object<double> >("testing_der_discY.dat");
    auto test_anal_gradY = std::make_shared< gnuplot_output_object<double> >("testing_der_analY.dat");
    */
        for (size_t i = 0; i <= N; i++) {
            for (size_t j = 0; j <= M; j++) {
                double px = i * (1.0 / N);
                double py = j * (1.0 / M);
                node = point_type(px, py);
/*
            if (std::abs( level_set_disc(node)) <1e-2 ) {
                valueD = 1;
            }
            else
                valueD = 0;
            */
                valueD = level_set_disc(node);
/*
            if (std::abs( level_set_anal(node)) <1e-2 ) {
                valueA = 1;
            }
            else
                valueA = 0;
            */
                valueA = level_set_fin(node);
/*
            derD = level_set_disc.gradient(node);
            derA = level_set_fin.gradient(node);


            derDx = derD(0);
            derDy = derD(1);
            derAx = derA(0);
            derAy = derA(1);
            */
                test_disc->add_data(node, valueD);
                test_anal->add_data(node, valueA);
/*
            test_disc_gradX->add_data(node,derDx);
            test_anal_gradX->add_data(node,derAx);
            test_disc_gradY->add_data(node,derDy);
            test_anal_gradY->add_data(node,derAy);
            */
            }

        }
        postoutput1.add_object(test_disc);
        postoutput1.add_object(test_anal);
/*
    postoutput1.add_object(test_disc_gradX);
    postoutput1.add_object(test_anal_gradX);
    postoutput1.add_object(test_disc_gradY);
    postoutput1.add_object(test_anal_gradY);
    */
        postoutput1.write();

    }

// Qualitative testing of the discrete level set function wrt the analytical one
    template<typename FonctionD, typename Mesh>
    void
    testing_level_set_mesh1(const Mesh msh, const FonctionD &level_set_disc, const FonctionD &level_set_fin) {
        typedef typename Mesh::point_type point_type;
        postprocess_output<double> postoutput1;
        double valueD, valueA; //, derDx , derDy , derAx , derAy;
        Eigen::Matrix<double, 2, 1> derD, derA;
        point<double, 2> node;
        size_t N, M;
        N = 40;
        M = 40;
        auto test_disc = std::make_shared < gnuplot_output_object < double > > ("testing_interface_fin_disc_mesh1.dat");
        auto test_anal = std::make_shared < gnuplot_output_object < double > > ("testing_interface_fin_anal_mesh1.dat");
/*
    auto test_disc_gradX  = std::make_shared< gnuplot_output_object<double> >("testing_der_discX.dat");
    auto test_anal_gradX = std::make_shared< gnuplot_output_object<double> >("testing_der_analX.dat");

    auto test_disc_gradY  = std::make_shared< gnuplot_output_object<double> >("testing_der_discY.dat");
    auto test_anal_gradY = std::make_shared< gnuplot_output_object<double> >("testing_der_analY.dat");
    */
        for (size_t i = 0; i <= N; i++) {
            for (size_t j = 0; j <= M; j++) {
                double px = i * (1.0 / N);
                double py = j * (1.0 / M);
                node = point_type(px, py);
/*
            if (std::abs( level_set_disc(node)) <1e-2 ) {
                valueD = 1;
            }
            else
                valueD = 0;
            */
                valueD = level_set_disc(node);
/*
            if (std::abs( level_set_anal(node)) <1e-2 ) {
                valueA = 1;
            }
            else
                valueA = 0;
            */
                valueA = level_set_fin(node);
/*
            derD = level_set_disc.gradient(node);
            derA = level_set_fin.gradient(node);


            derDx = derD(0);
            derDy = derD(1);
            derAx = derA(0);
            derAy = derA(1);
            */
                test_disc->add_data(node, valueD);
                test_anal->add_data(node, valueA);
/*
            test_disc_gradX->add_data(node,derDx);
            test_anal_gradX->add_data(node,derAx);
            test_disc_gradY->add_data(node,derDy);
            test_anal_gradY->add_data(node,derAy);
            */
            }

        }
        postoutput1.add_object(test_disc);
        postoutput1.add_object(test_anal);
/*
    postoutput1.add_object(test_disc_gradX);
    postoutput1.add_object(test_anal_gradX);
    postoutput1.add_object(test_disc_gradY);
    postoutput1.add_object(test_anal_gradY);
    */
        postoutput1.write();

    }

// Qualitative testing of the discrete level set function wrt the analytical one
    template<typename FonctionD, typename Mesh>
    void
    testing_level_set_mesh0(const Mesh msh, const FonctionD &level_set_disc, const FonctionD &level_set_fin) {
        typedef typename Mesh::point_type point_type;
        postprocess_output<double> postoutput1;
        double valueD, valueA; //, derDx , derDy , derAx , derAy;
        Eigen::Matrix<double, 2, 1> derD, derA;
        point<double, 2> node;
        size_t N, M;
        N = 40;
        M = 40;
        auto test_disc = std::make_shared < gnuplot_output_object < double > > ("testing_interface_fin_disc_mesh0.dat");
        auto test_anal = std::make_shared < gnuplot_output_object < double > > ("testing_interface_fin_anal_mesh0.dat");
/*
    auto test_disc_gradX  = std::make_shared< gnuplot_output_object<double> >("testing_der_discX.dat");
    auto test_anal_gradX = std::make_shared< gnuplot_output_object<double> >("testing_der_analX.dat");

    auto test_disc_gradY  = std::make_shared< gnuplot_output_object<double> >("testing_der_discY.dat");
    auto test_anal_gradY = std::make_shared< gnuplot_output_object<double> >("testing_der_analY.dat");
    */
        for (size_t i = 0; i <= N; i++) {
            for (size_t j = 0; j <= M; j++) {
                double px = i * (1.0 / N);
                double py = j * (1.0 / M);
                node = point_type(px, py);
/*
            if (std::abs( level_set_disc(node)) <1e-2 ) {
                valueD = 1;
            }
            else
                valueD = 0;
            */
                valueD = level_set_disc(node);
/*
            if (std::abs( level_set_anal(node)) <1e-2 ) {
                valueA = 1;
            }
            else
                valueA = 0;
            */
                valueA = level_set_fin(node);
/*
            derD = level_set_disc.gradient(node);
            derA = level_set_fin.gradient(node);


            derDx = derD(0);
            derDy = derD(1);
            derAx = derA(0);
            derAy = derA(1);
            */
                test_disc->add_data(node, valueD);
                test_anal->add_data(node, valueA);
/*
            test_disc_gradX->add_data(node,derDx);
            test_anal_gradX->add_data(node,derAx);
            test_disc_gradY->add_data(node,derDy);
            test_anal_gradY->add_data(node,derAy);
            */
            }

        }
        postoutput1.add_object(test_disc);
        postoutput1.add_object(test_anal);
/*
    postoutput1.add_object(test_disc_gradX);
    postoutput1.add_object(test_anal_gradX);
    postoutput1.add_object(test_disc_gradY);
    postoutput1.add_object(test_anal_gradY);
    */
        postoutput1.write();

    }

// Qualitative testing of the discrete level set function wrt the analytical one
    template<typename FonctionD, typename Mesh, typename FonctionA>
    void
    test_new_method(const Mesh msh, const FonctionD &level_set_disc, const FonctionA &level_set_anal,
                    const typename Mesh::cell_type &cl) {
//typedef typename Mesh::point_type       point_type;
        double valueD1, valueD2, valueA;

        auto pts = points(msh, cl);


        timecounter tc1;
        tc1.tic();
        for (auto &pt: pts) {
            valueD1 = level_set_disc(pt, msh, cl);
// valueA = level_set_anal(pt);
// valueD2 = level_set_disc(pt);
// std::cout<<"Differnce between two evaluation system "<<(valueD1-valueD2)<<std::endl;
//  std::cout<<"Im in test_new_method";
        }


        tc1.toc();
        std::cout << bold << yellow << "Time for the new evaluation: " << tc1 << " seconds" << reset << std::endl;

        tc1.tic();
        for (auto &pt: pts) {
            valueD2 = level_set_disc(pt);
//  std::cout<<"Im in test_new_method";
        }
        tc1.toc();
        std::cout << bold << yellow << "Time for the old evaluation: " << tc1 << " seconds" << reset << std::endl;

        tc1.tic();
        for (auto &pt: pts) {
            valueA = level_set_anal(pt);
        }
        tc1.toc();
        std::cout << bold << yellow << "Time for the analytic evaluation: " << tc1 << " seconds" << reset << std::endl;


    }

// Qualitative testing of the discrete level set function wrt the analytical one
    template<typename FonctionD, typename Mesh>
    void
    time_NEWdiscrete_testing(const Mesh msh, const FonctionD &level_set_disc, const typename Mesh::cell_type &cl) {
//typedef typename Mesh::point_type       point_type;
        double valueD1; //, valueD2 ,valueA ;

        auto pts = points(msh, cl);


        for (auto &pt: pts) {
            valueD1 = level_set_disc(pt, msh, cl);

        }


    }

// Qualitative testing of the discrete level set function wrt the analytical one
    template<typename FonctionD, typename Mesh>
    void
    time_OLDdiscrete_testing(const Mesh msh, const FonctionD &level_set_disc, const typename Mesh::cell_type &cl) {
//typedef typename Mesh::point_type       point_type;
        double valueD1; //, valueA ; // valueD2 ,

        auto pts = points(msh, cl);


        for (auto &pt: pts) {
            valueD1 = level_set_disc(pt);

        }


    }

    template<typename FonctionA, typename Mesh>
    void
    time_analytic_testing(const Mesh msh, const FonctionA &level_set_anal, const typename Mesh::cell_type &cl) {
//typedef typename Mesh::point_type       point_type;
        double valueA; // valueD1 , valueD2 ,

        auto pts = points(msh, cl);


        for (auto &pt: pts) {
            valueA = level_set_anal(pt);

        }


    }


    template<typename FonctionD, typename Mesh>
    void
    time_face_testing(const Mesh msh, const FonctionD &level_set_disc, const typename Mesh::face_type &fc) {

//typedef typename Mesh::point_type       point_type;
        double valueD1;// , valueD2 ,valueA ;

        auto pts = points(msh, fc);


        for (auto &pt: pts) {
            valueD1 = level_set_disc(pt, msh, fc);

        }


    }

    template<typename FonctionA, typename Mesh>
    void
    time_faceANALITIC_testing(const Mesh msh, const FonctionA &level_set_anal, const typename Mesh::face_type &fc) {

//typedef typename Mesh::point_type       point_type;
        double valueA; // valueD1 , valueD2 ,

        auto pts = points(msh, fc);


        for (auto &pt: pts) {
            valueA = level_set_anal(pt);

        }


    }


// Qualitative testing of the discrete level set function wrt the analytical one
    template<typename FonctionD, typename Mesh, typename FonctionA>
    void
    test_new_method(const Mesh msh, const FonctionD &level_set_disc, const FonctionA &level_set_anal,
                    const typename Mesh::cell_type &cl, const typename Mesh::face_type &fc) {
//typedef typename Mesh::point_type       point_type;
        double valueD1, valueA, valueD3; //, valueD2

        auto pts = points(msh, fc);


        timecounter tc1;
        tc1.tic();
        for (auto &pt: pts) {
            valueD1 = level_set_disc(pt, msh, fc);
            valueA = level_set_anal(pt);
//valueD2 = level_set_disc(pt,msh,cl);
            valueD3 = level_set_disc(pt);

            std::cout << "Differnce between FACE and OLD evaluation system " << (valueD1 - valueD3) << std::endl;
            std::cout << "Error between analytic and face evaluation " << (valueD1 - valueA) << std::endl;
        }


    }


    template<typename Mesh, typename Fonction, typename T = typename Mesh::coordinate_type, typename VEC>
    void Lp_space_Tfin_error_FEM(const Fonction &level_set_final, const Fonction &level_set_initial, const Mesh &msh,
                                 size_t degree, double p, VEC &error) {
        T errorLp = 0.;
        for (auto &cl: msh.cells) {
            auto qps = integrate(msh, cl, 2 * degree + 2);
            for (auto &qp: qps) {
                auto diff_val = std::abs(level_set_final(qp.first, msh, cl) - level_set_initial(qp.first, msh, cl));
                errorLp += qp.second * pow(diff_val, p);
            }
//std::cout<<"The L^2 error squared in cell "<<offset(msh,cl)<<" is "<< errorL2 <<std::endl;
        }
//std::cout<<"The L^"<<p<<" error is "<< pow (errorLp , 1.0/p )<<std::endl;
        error.push_back(pow(errorLp, 1.0 / p));
    }


    template<typename Mesh, typename Fonction, typename T = typename Mesh::coordinate_type>
    void Lp_space_Tfin_error_FEM(const Fonction &level_set_final, const Fonction &level_set_initial, const Mesh &msh,
                                 size_t degree, double p) {
        T errorLp = 0.;
        for (auto &cl: msh.cells) {
            auto qps = integrate(msh, cl, 2 * degree + 2);
            for (auto &qp: qps) {
                auto diff_val = std::abs(level_set_final(qp.first, msh, cl) - level_set_initial(qp.first, msh, cl));
                errorLp += qp.second * pow(diff_val, p);
            }
//std::cout<<"The L^2 error squared in cell "<<offset(msh,cl)<<" is "<< errorL2 <<std::endl;
        }
        std::cout << "The L^" << p << " error is " << pow(errorLp, 1.0 / p) << std::endl;
    }


    template<typename Mesh, typename Fonction, typename T = typename Mesh::coordinate_type>
    T
    Linf_error_FEM_new(const Fonction &level_set_final, const Fonction &level_set_initial, const Mesh &msh, size_t degree) {
        T errorLinf = ((level_set_final.sol_HHO - level_set_initial.sol_HHO).cwiseAbs()).maxCoeff();
        return errorLinf;

    }


    template<typename Mesh, typename Fonction, typename T = typename Mesh::coordinate_type>
    T W1inf_error_FEM_new(const Fonction &level_set_final, const Fonction &level_set_initial, const Mesh &msh,
                          size_t degree) {
        T errorLinf0 = 0.0, errorLinf1 = 0.0;

        for (auto &cl: msh.cells) {
            auto pts = equidistriduted_nodes_ordered_bis<T, Mesh>(msh, cl, degree);
//auto pts = points(msh,cl);
            for (auto &pt: pts) {
                auto diff_val0 = std::abs(
                        level_set_final.gradient(pt, msh, cl)(0) - level_set_initial.gradient(pt, msh, cl)(0));
                errorLinf0 = std::max(errorLinf0, diff_val0);
                auto diff_val1 = std::abs(
                        level_set_final.gradient(pt, msh, cl)(1) - level_set_initial.gradient(pt, msh, cl)(1));
                errorLinf1 = std::max(errorLinf1, diff_val1);
            }

        }
        return errorLinf0 + errorLinf1;

    }


    template<typename Mesh, typename Fonction, typename T = typename Mesh::coordinate_type>
    T Lp_space_error_FEM(const Fonction &level_set_final, const Fonction &level_set_initial, const Mesh &msh, size_t degree,
                         double p) {
// L^p in space ; l^q in time
        T errorLp = 0.;
        for (auto &cl: msh.cells) {
            auto qps = integrate(msh, cl, 2 * degree + 2); // what orders?
            for (auto &qp: qps) {
                auto diff_val = std::abs(level_set_final(qp.first, msh, cl) - level_set_initial(qp.first, msh, cl));
                errorLp += qp.second * pow(diff_val, p);
            }
//std::cout<<"The L^2 error squared in cell "<<offset(msh,cl)<<" is "<< errorL2 <<std::endl;
        }
        errorLp = pow(errorLp, 1.0 / p);
//std::cout<<"The L^2 error is "<<sqrt( errorL2 )<<std::endl;
        return errorLp;
    }

    template<typename Mesh, typename Fonction, typename T = typename Mesh::coordinate_type>
    T W1p_error_FEM(const Fonction &level_set_final, const Fonction &level_set_initial, const Mesh &msh, size_t degree,
                    double p) {
        T errorH1 = 0.;
        for (auto &cl: msh.cells) {
            auto qps = integrate(msh, cl, 2 * degree + 2); // what orders?
            for (auto &qp: qps) {
                auto diff_val0 = std::abs(
                        level_set_final.gradient(qp.first, msh, cl)(0) - level_set_initial.gradient(qp.first, msh, cl)(0));
                auto diff_val1 = std::abs(
                        level_set_final.gradient(qp.first, msh, cl)(1) - level_set_initial.gradient(qp.first, msh, cl)(1));
                errorH1 += qp.second * (pow(diff_val0, p) + pow(diff_val1, p));
            }
//std::cout<<"The L^2 error squared in cell "<<offset(msh,cl)<<" is "<< errorL2 <<std::endl;
        }
        errorH1 = pow(errorH1, 1.0 / p);
//std::cout<<"The L^2 error is "<<sqrt( errorL2 )<<std::endl;
        return errorH1;
    }


    template<typename Mesh, typename Fonction, typename T = typename Mesh::coordinate_type>
    T Linf_error_FEM(const Fonction &level_set_final, const Fonction &level_set_initial, const Mesh &msh, size_t degree) {
        T errorLinf = ((level_set_final.vertices - level_set_initial.vertices).cwiseAbs()).maxCoeff();
        return errorLinf;

    }

    template<typename Mesh, typename Fonction, typename T = typename Mesh::coordinate_type>
    T W1inf_error_FEM(const Fonction &level_set_final, const Fonction &level_set_initial, const Mesh &msh, size_t degree) {
        T errorLinf0 = 0, errorLinf1 = 0;

        for (auto &cl: msh.cells) {
            auto pts = points(msh, cl);
            for (auto &pt: pts) {
                auto diff_val0 = std::abs(
                        level_set_final.gradient(pt, msh, cl)(0) - level_set_initial.gradient(pt, msh, cl)(0));
                errorLinf0 = std::max(errorLinf0, diff_val0);
                auto diff_val1 = std::abs(
                        level_set_final.gradient(pt, msh, cl)(1) - level_set_initial.gradient(pt, msh, cl)(1));
                errorLinf1 = std::max(errorLinf1, diff_val1);
            }

        }
        return errorLinf0 + errorLinf1;

    }

}