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

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <sstream>
#include <random>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/SparseExtra>



using namespace Eigen;

#include "core/core"
#include "core/solvers"
#include "methods/hho"

#include "sol/sol.hpp"

#include "dataio/silo_io.hpp"



int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Please specify mesh file name." << std::endl;
        return 1; 
    }

    using T = double;

    poly_mesh<T> msh(argv[1]);


    auto rhs_fun = [](const typename poly_mesh<T>::point_type& pt) -> T {
        return 2.0 * M_PI * M_PI * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };

    auto sol_fun = [](const typename poly_mesh<T>::point_type& pt) -> T {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };

    size_t k = 0;

    hho_degree_info hdi(k, k);

    auto assembler = make_assembler(msh, hdi);
    for (auto& cl : msh.cells)
    {
        auto gr = make_hho_laplacian(msh, cl, hdi);

        Matrix<T, Dynamic, Dynamic> stab;
        stab = make_hho_fancy_stabilization(msh, cl, gr.first, hdi);

        Matrix<T, Dynamic, Dynamic> lc = gr.second + stab;
        Matrix<T, Dynamic, 1> f = make_rhs(msh, cl, hdi.cell_degree(), rhs_fun);
        assembler.assemble(msh, cl, lc, f, sol_fun);
    }

    assembler.finalize();

    Matrix<T, Dynamic, 1> sol;

    SparseLU<SparseMatrix<T>>  solver;

    solver.analyzePattern(assembler.LHS);
    solver.factorize(assembler.LHS);
    sol = solver.solve(assembler.RHS);

    auto cbs = cell_basis<poly_mesh<T>, T>::size(hdi.cell_degree());

    silo_database silo;
    silo.create("polymesh.silo");
    silo.add_mesh(msh, "mesh");
    silo.close();

    std::ofstream ofs("pi.dat");

    size_t cell_i = 0;
    T error = 0.0;
    for (auto& cl : msh.cells)
    {
        Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, cl, hdi.cell_degree());
        Matrix<T, Dynamic, 1> f = make_rhs(msh, cl, hdi.cell_degree(), sol_fun);
        Matrix<T, Dynamic, 1> u = mass.llt().solve(f);

        Matrix<T, Dynamic, 1> uh = sol.block(cell_i*cbs, 0, cbs, 1);

        Matrix<T, Dynamic, 1> diff = u - uh;

        error += diff.dot(mass*diff);

        auto qpts = integrate(msh, cl, 5);
        for (auto& qp : qpts)
        {
            ofs << qp.first.x() << " " << qp.first.y() << " " << cell_i << std::endl;
        }


        cell_i++;
    }

    ofs.close();

    std::cout << error << std::endl;

    return 0;
}



