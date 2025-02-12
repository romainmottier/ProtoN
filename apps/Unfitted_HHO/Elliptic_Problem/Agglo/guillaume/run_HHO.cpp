
template<typename Mesh, typename testType, typename meth>
test_info<typename Mesh::coordinate_type>
run_cuthho_interface(const Mesh& msh, size_t degree, meth method, testType test_case) {

    using RealType = typename Mesh::coordinate_type;

    auto sol_fun = test_case.sol_fun;
    auto sol_grad = test_case.sol_grad;
    auto bcs_fun = test_case.bcs_fun;

    timecounter tc;

    bool sc = true; // static condensation


    /************** ASSEMBLE PROBLEM **************/
    hho_degree_info hdi(degree+1, degree);

    tc.tic();
    auto assembler = make_interface_assembler(msh, bcs_fun, hdi);
    auto assembler_sc = make_interface_condensed_assembler(msh, bcs_fun, hdi);
    for (auto& cl : msh.cells) {
        auto contrib = method.make_contrib(msh, cl, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;

        if (sc)
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
#if 0
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
//#if 0
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
//#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    /************** POSTPROCESS **************/


    postprocess_output<RealType>  postoutput;

    auto uT_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT.dat");
    auto diff_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_diff.dat");


    std::vector<RealType>   solution_uT;

    tc.tic();
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
    size_t      cell_i   = 0;
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
            if( sc )
            {
                locdata_n = assembler_sc.take_local_data(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                locdata_p = assembler_sc.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
            {
                locdata_n = assembler.take_local_data(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                locdata_p = assembler.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }

            cell_dofs_n = locdata_n.head(cbs);
            cell_dofs_p = locdata_p.head(cbs);


            auto qps_n = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs_n(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad(qp.first) - grad).dot(sol_grad(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs_n.dot(t_phi);
                uT_gp->add_data(qp.first, v);
                
                /* Compute L2-error */
                L2_error += qp.second * (sol_fun(qp.first) - v) * (sol_fun(qp.first) - v);
            }
            
            
            auto qps_p = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs_p(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad(qp.first) - grad).dot(sol_grad(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs_p.dot(t_phi);
                uT_gp->add_data(qp.first, v);

                /* Compute L2-error */
                L2_error += qp.second * (sol_fun(qp.first) - v) * (sol_fun(qp.first) - v);
            }
        }
        else
        {
            if( sc )
            {
                locdata = assembler_sc.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
                locdata = assembler.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            cell_dofs = locdata.head(cbs);

            auto qps = integrate(msh, cl, 2*hdi.cell_degree());
            for (auto& qp : qps)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad(qp.first) - grad).dot(sol_grad(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs.dot(t_phi);
                uT_gp->add_data(qp.first, v);

                /* Compute L2-error */
                L2_error += qp.second * (sol_fun(qp.first) - v) * (sol_fun(qp.first) - v);
            }
        }

        cell_i++;
    }

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:           " << std::sqrt(L2_error) << std::endl;

    postoutput.add_object(uT_gp);
    postoutput.add_object(diff_gp);
    postoutput.write();



    test_info<RealType> TI;
    TI.H1 = std::sqrt(H1_error);
    TI.L2 = std::sqrt(L2_error);

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


    return TI;
}
