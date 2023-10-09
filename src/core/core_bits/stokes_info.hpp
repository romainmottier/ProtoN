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

namespace stokes_info{


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


}