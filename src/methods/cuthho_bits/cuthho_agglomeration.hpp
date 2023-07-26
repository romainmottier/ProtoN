/*
 *       /\        Matteo Cicuttin (C) 2017,2018; Guillaume Delay 2018,2019
 *      /__\       matteo.cicuttin@enpc.fr        guillaume.delay@enpc.fr
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


template<typename T, size_t ET>
void
detect_cell_agglo_set(cuthho_mesh<T, ET>& msh)
{
    const T threshold = 0.3;
    const T threshold_cells = 0.3;

    for (auto& cl : msh.cells)
    {
        auto fcs = faces(msh, cl);
        auto pts = points(msh, cl);
        auto nds = nodes(msh, cl);

        if (fcs.size() != 4)
            throw std::invalid_argument("This works only on quads for now");

        if( !is_cut(msh, cl) )
        {
            cl.user_data.agglo_set = cell_agglo_set::T_OK;
            continue;
        }

//        std::cout<<"Cell: "<<offset(msh,cl)<<" negative area = "<<measure(msh, cl, element_location::IN_NEGATIVE_SIDE)<< " , positive area = "<< measure(msh, cl, element_location::IN_POSITIVE_SIDE)<<" max threshold area = "<< threshold_cells * measure(msh, cl)<<std::endl;
        
        if( measure(msh, cl, element_location::IN_NEGATIVE_SIDE)
            < threshold_cells * measure(msh, cl) )
        {
            cl.user_data.agglo_set = cell_agglo_set::T_KO_NEG;
            continue;
        }
        else if( measure(msh, cl, element_location::IN_POSITIVE_SIDE)
            < threshold_cells * measure(msh, cl) )
        {
            cl.user_data.agglo_set = cell_agglo_set::T_KO_POS;
            continue;
        }

        /* If it is a quadrilateral we have 6 possible configurations of the
         * element-cut intersection. */

        auto agglo_set_single_node = [&](size_t n) -> void
        {
            auto f1 = (n == 0) ? fcs.size()-1 : n-1;
            auto f2 = n;

            auto ma = measure(msh, fcs[f1]);
            auto pa = (pts[n] - fcs[f1].user_data.intersection_point);
            auto da = pa.to_vector().norm() / ma;

            auto mb = measure(msh, fcs[f2]);
            auto pb = (pts[n] - fcs[f2].user_data.intersection_point);
            auto db = pb.to_vector().norm() / mb;

            assert(da >= 0 && da <= 1);
            assert(db >= 0 && db <= 1);

            if ( std::min(da, db) > threshold )
            {
                cl.user_data.agglo_set = cell_agglo_set::T_OK;
                return;
            }

            if ( location(msh, nds[n]) == element_location::IN_NEGATIVE_SIDE )
                cl.user_data.agglo_set = cell_agglo_set::T_KO_NEG;
            else
                cl.user_data.agglo_set = cell_agglo_set::T_KO_POS;
        };
        
        auto agglo_set_double_node = [&](size_t f1, size_t f2) -> void
        {
            assert ( (f1 == 0 && f2 == 2) || ( f1 == 1 && f2 == 3 ) );

            auto n1 = f1;
            auto n2 = (f2+1) % fcs.size();

            auto ma = measure(msh, fcs[f1]);
            auto pa = (pts[n1] - fcs[f1].user_data.intersection_point);
            auto da = pa.to_vector().norm() / ma;

            auto mb = measure(msh, fcs[f2]);
            auto pb = (pts[n2] - fcs[f2].user_data.intersection_point);
            auto db = pb.to_vector().norm() / mb;

            auto m1 = std::max(da, db);
            auto m2 = std::max(1-da, 1-db);

            if ( std::min(m1, m2) > threshold )
            {
                cl.user_data.agglo_set = cell_agglo_set::T_OK;
                return;
            }

            if ( location(msh, nds[n1]) == element_location::IN_NEGATIVE_SIDE )
                cl.user_data.agglo_set = (m1 <= threshold) ? cell_agglo_set::T_KO_NEG : cell_agglo_set::T_KO_POS;
            else
                cl.user_data.agglo_set = (m2 <= threshold) ? cell_agglo_set::T_KO_NEG : cell_agglo_set::T_KO_POS;
        };
        // for all the faces of the cell cl
        for (size_t i = 0; i < fcs.size(); i++)
        {
            auto f1 = i;
            auto f2 = (i+1) % fcs.size();
            auto n = (i+1) % fcs.size();

            if ( is_cut(msh, fcs[f1]) && is_cut(msh, fcs[f2]) )
                agglo_set_single_node(n);

        }

        if ( is_cut(msh, fcs[0]) && is_cut(msh, fcs[2]) )
            agglo_set_double_node(0,2);

        if ( is_cut(msh, fcs[1]) && is_cut(msh, fcs[3]) )
            agglo_set_double_node(1,3);

    }
}


template<typename T, size_t ET, typename Function>
void
detect_cell_agglo_set(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    //typedef typename cuthho_mesh<T, ET>::face_type  face_type;
    //typedef typename cuthho_mesh<T, ET>::point_type point_type;

    const T threshold = 0.3;
    const T threshold_cells = 0.3;

    for (auto& cl : msh.cells)
    {
        auto fcs = faces(msh, cl);
        auto pts = points(msh, cl);
        auto nds = nodes(msh, cl);

        if (fcs.size() != 4)
            throw std::invalid_argument("This works only on quads for now");

        if( !is_cut(msh, cl) )
        {
            cl.user_data.agglo_set = cell_agglo_set::T_OK;
            continue;
        }

        //// another criterion on the area of the cell
//        std::cout<<"measure(msh, cl, element_location::IN_NEGATIVE_SIDE) = "<<measure(msh, cl, element_location::IN_NEGATIVE_SIDE)<<std::endl;
//         std::cout<<"measure OLD (msh, cl, element_location::IN_NEGATIVE_SIDE) = "<<measure_old(msh, cl, element_location::IN_NEGATIVE_SIDE)<<std::endl;
//        std::cout<<"measure(msh, cl, element_location::IN_POSITIVE_SIDE) = "<<measure(msh, cl, element_location::IN_POSITIVE_SIDE)<<std::endl;
//        std::cout<<"measure OLD (msh, cl, element_location::IN_POSITIVE_SIDE) = "<<measure_old(msh, cl, element_location::IN_POSITIVE_SIDE)<<std::endl;
//        std::cout<<"measure(msh, cl) = "<<measure(msh, cl)<<std::endl;
        
//        std::cout<<"Cell: "<<offset(msh,cl)<<" negative area = "<<measure(msh, cl, element_location::IN_NEGATIVE_SIDE)<< " , positive area = "<< measure(msh, cl, element_location::IN_POSITIVE_SIDE)<<" max threshold area = "<< threshold_cells * measure(msh, cl)<<std::endl;
        
        if( measure(msh, cl, element_location::IN_NEGATIVE_SIDE)
            < threshold_cells * measure(msh, cl) )
        {
            cl.user_data.agglo_set = cell_agglo_set::T_KO_NEG;
            continue;
        }
        else if( measure(msh, cl, element_location::IN_POSITIVE_SIDE)
            < threshold_cells * measure(msh, cl) )
        {
            cl.user_data.agglo_set = cell_agglo_set::T_KO_POS;
            continue;
        }

        /* If it is a quadrilateral we have 6 possible configurations of the
         * element-cut intersection. */

        auto agglo_set_single_node = [&](size_t n) -> void
        {
            auto f1 = (n == 0) ? fcs.size()-1 : n-1;
            auto f2 = n;

            auto ma = measure(msh, fcs[f1]);
            auto pa = (pts[n] - fcs[f1].user_data.intersection_point);
            auto da = pa.to_vector().norm() / ma;

            auto mb = measure(msh, fcs[f2]);
            auto pb = (pts[n] - fcs[f2].user_data.intersection_point);
            auto db = pb.to_vector().norm() / mb;

            assert(da >= 0 && da <= 1);
            assert(db >= 0 && db <= 1);

            if ( std::min(da, db) > threshold )
            {
                cl.user_data.agglo_set = cell_agglo_set::T_OK;
                return;
            }

            if ( location(msh, nds[n]) == element_location::IN_NEGATIVE_SIDE )
                cl.user_data.agglo_set = cell_agglo_set::T_KO_NEG;
            else
                cl.user_data.agglo_set = cell_agglo_set::T_KO_POS;
        };
        
        auto agglo_set_double_node = [&](size_t f1, size_t f2) -> void
        {
            assert ( (f1 == 0 && f2 == 2) || ( f1 == 1 && f2 == 3 ) );

            auto n1 = f1;
            auto n2 = (f2+1) % fcs.size();

            auto ma = measure(msh, fcs[f1]);
            auto pa = (pts[n1] - fcs[f1].user_data.intersection_point);
            auto da = pa.to_vector().norm() / ma;

            auto mb = measure(msh, fcs[f2]);
            auto pb = (pts[n2] - fcs[f2].user_data.intersection_point);
            auto db = pb.to_vector().norm() / mb;

            auto m1 = std::max(da, db);
            auto m2 = std::max(1-da, 1-db);

            if ( std::min(m1, m2) > threshold )
            {
                cl.user_data.agglo_set = cell_agglo_set::T_OK;
                return;
            }

            if ( location(msh, nds[n1]) == element_location::IN_NEGATIVE_SIDE )
                cl.user_data.agglo_set = (m1 <= threshold) ? cell_agglo_set::T_KO_NEG : cell_agglo_set::T_KO_POS;
            else
                cl.user_data.agglo_set = (m2 <= threshold) ? cell_agglo_set::T_KO_NEG : cell_agglo_set::T_KO_POS;
        };
        // for all the faces of the cell cl
        for (size_t i = 0; i < fcs.size(); i++)
        {
            auto f1 = i;
            auto f2 = (i+1) % fcs.size();
            auto n = (i+1) % fcs.size();

            if ( is_cut(msh, fcs[f1]) && is_cut(msh, fcs[f2]) )
                agglo_set_single_node(n);

        }

        if ( is_cut(msh, fcs[0]) && is_cut(msh, fcs[2]) )
            agglo_set_double_node(0,2);

        if ( is_cut(msh, fcs[1]) && is_cut(msh, fcs[3]) )
            agglo_set_double_node(1,3);
/*
        if ( is_cut(msh, fcs[0]) && is_cut(msh, fcs[3]) )
            agglo_set_case_1();
        else if ( is_cut(msh, fcs[0]) && is_cut(msh, fcs[1]) )
            agglo_set_case_2();
        else if ( is_cut(msh, fcs[1]) && is_cut(msh, fcs[2]) )
            agglo_set_case_3();
        else if ( is_cut(msh, fcs[2]) && is_cut(msh, fcs[3]) )
            agglo_set_case_4();
        else if ( is_cut(msh, fcs[0]) && is_cut(msh, fcs[2]) )
            agglo_set_case_5();
        else if ( is_cut(msh, fcs[1]) && is_cut(msh, fcs[3]) )
            agglo_set_case_6();
            */
    }
}


/* this creates Delta(T) */
// two neighbors have at least one common face
template<typename T, size_t ET>
void
make_neighbors_info(cuthho_mesh<T, ET>& msh)
{
    for (size_t i = 0; i < msh.cells.size(); i++)
    {
        auto fc_i = faces(msh,msh.cells[i]);
        for (size_t j = i+1; j < msh.cells.size(); j++)
        {
            auto &cl1 = msh.cells.at(i);
            auto &cl2 = msh.cells.at(j);

            // two f_neighbors have at least one common face
            bool are_f_neighbors = false;
            auto fc_j = faces(msh,msh.cells[j]);
            for (size_t i_face = 0; i_face < fc_i.size(); i_face++)
                for (size_t j_face = 0; j_face < fc_j.size(); j_face++)
                    if ( fc_i[i_face] == fc_j[j_face] )
                        are_f_neighbors = true;

            if (are_f_neighbors)
            {
                auto ofs_cl1 = offset(msh, cl1);
                auto ofs_cl2 = offset(msh, cl2);

                cl1.user_data.f_neighbors.insert(ofs_cl2);
                cl2.user_data.f_neighbors.insert(ofs_cl1);
            }
            else
            {
                // two d_neighbors have at least one common node
                bool are_d_neighbors = false;
                for (size_t ip = 0; ip < cl1.ptids.size(); ip++)
                    for (size_t jp = 0; jp < cl2.ptids.size(); jp++)
                        if ( cl1.ptids[ip] == cl2.ptids[jp] )
                            are_d_neighbors = true;

                if (are_d_neighbors)
                {
                    auto ofs_cl1 = offset(msh, cl1);
                    auto ofs_cl2 = offset(msh, cl2);

                    cl1.user_data.d_neighbors.insert(ofs_cl2);
                    cl2.user_data.d_neighbors.insert(ofs_cl1);
                }

            }
        }
    }

    /*
    for (auto& cl : msh.cells)
    {
        for (auto& n : cl.user_data.neighbors)
            std::cout << n << " ";
        std::cout << std::endl;
    }
    */
}


//// version for cartesian meshes -> very quick
/* this creates Delta(T) */
// there are at least two row and two columns of cells
template<typename T, size_t ET>
void
make_neighbors_info_cartesian(cuthho_mesh<T, ET>& msh)
{
    std::cout << "WARNING : make_neighbors_info_cartesian "
              << "works for cartesian meshes only !!"
              << std::endl;

    size_t N = sqrt(msh.cells.size());

    //////////////////  face neighbors  ///////////////////
    // first row of cells -> look left
    for (size_t i = 1; i < N; i++)
    {
        auto &cl1 = msh.cells.at(i);
        auto &cl2 = msh.cells.at(i-1);

        cl1.user_data.f_neighbors.insert(i-1);
        cl2.user_data.f_neighbors.insert(i);
    }

    // other rows of cells
    for (size_t j = 1; j < N; j++)
    {
        // first cell of the row -> look bottom
        auto &cl1 = msh.cells.at( j*N );
        auto &cl2 = msh.cells.at( (j-1)*N );
        cl1.user_data.f_neighbors.insert( (j-1)*N );
        cl2.user_data.f_neighbors.insert( j*N );

        // other cells -> look left and bottom
        for (size_t i = 1; i < N; i++)
        {
            auto &cl_c = msh.cells.at( j*N + i ); // current
            auto &cl_l = msh.cells.at( j*N + i - 1 ); // left
            auto &cl_b = msh.cells.at( (j-1)*N + i ); // bottom

            cl_c.user_data.f_neighbors.insert( j*N + i - 1 );
            cl_c.user_data.f_neighbors.insert( (j-1)*N + i );

            cl_l.user_data.f_neighbors.insert( j*N + i );
            cl_b.user_data.f_neighbors.insert( j*N + i );
        }
    }

    //////////////////////  diagonal neighbors  //////////////////////
    // first row of cells -> look left top
    for (size_t i = 1; i < N; i++)
    {
        auto &cl1 = msh.cells.at(i);
        auto &cl2 = msh.cells.at(N + i-1);

        cl1.user_data.d_neighbors.insert(N + i-1);
        cl2.user_data.d_neighbors.insert(i);
    }

    // other rows of cells
    for (size_t j = 1; j < N-1; j++)
    {
        // first cell of the row -> nothing to do
        // other cells -> look left top and left bottom
        for (size_t i = 1; i < N; i++)
        {
            auto &cl_c = msh.cells.at( j*N + i ); // current
            auto &cl_l = msh.cells.at( (j+1)*N + i - 1 ); // left top
            auto &cl_b = msh.cells.at( (j-1)*N + i - 1 ); // left bottom

            cl_c.user_data.d_neighbors.insert( (j+1)*N + i - 1 );
            cl_c.user_data.d_neighbors.insert( (j-1)*N + i - 1 );

            cl_l.user_data.d_neighbors.insert( j*N + i );
            cl_b.user_data.d_neighbors.insert( j*N + i );
        }
    }

    // last row -> look left bottom
    for (size_t i = 1; i < N; i++)
    {
        auto &cl1 = msh.cells.at( (N-1)*N + i);
        auto &cl2 = msh.cells.at( (N-2)*N + i-1);

        cl1.user_data.d_neighbors.insert((N-2)*N + i-1);
        cl2.user_data.d_neighbors.insert((N-1)*N + i);
    }
}



//////////////  MERGE_CELLS_FACE
/// merge cl1 and cl2 through the common face fc
// output : new cell
template<typename cell_type, typename face_type>
cell_type
merge_cells_face(cell_type cl1, cell_type cl2, face_type com_f)
{
    cell_type ret;

    size_t f_pt1 = com_f.ptids[0];
    size_t f_pt2 = com_f.ptids[1];

    // list of points
    std::vector<size_t> pts1, pts2;
    // in order to be consistent with the cell representation, start with the smallest index
    if(cl1.ptids[0] < cl2.ptids[0])
    {
        pts1 = cl1.ptids;
        pts2 = cl2.ptids;
    }
    else
    {
        pts1 = cl2.ptids;
        pts2 = cl1.ptids;
    }

    // write points of pts1 until we reach the common face
    size_t ref_pt = pts1[0];
    size_t cp = 0;
    ret.ptids.push_back(ref_pt);

    bool on_face = false;
    if(ref_pt == f_pt1 || ref_pt == f_pt2) on_face = true;

    while(!on_face)
    {
        cp++;
        ref_pt = pts1[cp];
        ret.ptids.push_back(ref_pt);
        if(ref_pt == f_pt1 || ref_pt == f_pt2) on_face = true;
    }

    // look for the corresponding point in pts2
    for(size_t i = 0; i < pts2.size(); i++)
    {
        if(ref_pt == pts2[i])
        {
            cp = i;
            break;
        }
    }
    // write points of pts2 until we reach once more the common face
    on_face = false;
    while( !on_face )
    {
        cp = (cp + 1) % pts2.size();
        ref_pt = pts2[cp];
        ret.ptids.push_back(ref_pt);
        if(ref_pt == f_pt1 || ref_pt == f_pt2) on_face = true;
    }
    // look for the corresponding point in pts1
    for(size_t i=0; i < pts1.size(); i++)
    {
        if(ref_pt == pts1[i])
        {
            cp = i;
            break;
        }
    }

    // finish to write the points of pts1
    cp++;
    while(cp < pts1.size())
    {
        ret.ptids.push_back(pts1[cp]);
        cp++;
    }

    return ret;
}



//////////////  MERGE_CELLS_TWO_FACES
/// merge cl1 and cl2 through the list com_f of common faces
// output : new cell
template<typename cell_type, typename face_type>
cell_type
merge_cells_two_faces(cell_type cl1, cell_type cl2, typename std::vector<face_type> com_f)
{
    cell_type ret;

    assert( com_f.size() == 2 );

    size_t f1_pt1 = com_f[0].ptids[0];
    size_t f1_pt2 = com_f[0].ptids[1];
    size_t f2_pt1 = com_f[1].ptids[0];
    size_t f2_pt2 = com_f[1].ptids[1];

    std::set<size_t> com_pts;
    if(f1_pt1 == f2_pt1 || f1_pt2 == f2_pt1)
    {
        com_pts.insert(f1_pt1);
        com_pts.insert(f1_pt2);
        com_pts.insert(f2_pt2);
    }
    else if(f1_pt1 == f2_pt2 || f1_pt2 == f2_pt2)
    {
        com_pts.insert(f1_pt1);
        com_pts.insert(f1_pt2);
        com_pts.insert(f2_pt1);
    }
    else
        throw std::logic_error("com_f : no common points");

    // list of points
    std::vector<size_t> pts1, pts2;
    // in order to be consistent with the cell representation, start with the smallest index
    if(cl1.ptids[0] < cl2.ptids[0])
    {
        pts1 = cl1.ptids;
        pts2 = cl2.ptids;
    }
    else
    {
        pts1 = cl2.ptids;
        pts2 = cl1.ptids;
    }

    // write points of pts1 until we reach the common face
    size_t ref_pt = pts1[0];
    size_t next_pt= pts1[1];
    size_t cp = 0;
    ret.ptids.push_back(ref_pt);

    bool on_face = false;
    if( com_pts.find(ref_pt) != com_pts.end() && com_pts.find(next_pt) != com_pts.end() )
        on_face = true;

    while(!on_face)
    {
        cp++;
        ref_pt = next_pt;
        next_pt= pts1[(cp+1)%pts1.size()];
        ret.ptids.push_back(ref_pt);
        if( com_pts.find(ref_pt) != com_pts.end() && com_pts.find(next_pt) != com_pts.end() )
            on_face = true;
    }

    // look for the corresponding point in pts2
    for(size_t i = 0; i < pts2.size(); i++)
    {
        if(ref_pt == pts2[i])
        {
            cp = i;
            break;
        }
    }

    // write points of pts2 until we reach once more the common face
    on_face = false;
    while( !on_face )
    {
        cp = (cp + 1) % pts2.size();
        ref_pt = pts2[cp];
        next_pt= pts2[(cp+1) % pts2.size()];
        ret.ptids.push_back(ref_pt);
        if( com_pts.find(ref_pt) != com_pts.end() && com_pts.find(next_pt) != com_pts.end() )
            on_face = true;
    }
    // look for the corresponding point in pts1
    for(size_t i=0; i < pts1.size(); i++)
    {
        if(ref_pt == pts1[i])
        {
            cp = i;
            break;
        }
    }

    // finish to write the points of pts1
    cp++;
    while(cp < pts1.size())
    {
        ret.ptids.push_back(pts1[cp]);
        cp++;
    }

    return ret;
}


//////////////  MERGE_CELLS_DIAG
/// merge cl1 and cl2 through the common node com_n
// output : new cell
template<typename cell_type>
cell_type
merge_cells_diag(cell_type cl1, cell_type cl2, size_t com_n)
{
    // abort the process if both cells are on the interface
    // (this case is not yet supported for the list of points on the interface)
    if( cl1.user_data.location == element_location::ON_INTERFACE
        && cl2.user_data.location == element_location::ON_INTERFACE )
        throw std::invalid_argument("Cannot merge diagonally two cells on the interface.");

    cell_type ret;

    // list of points
    std::vector<size_t> pts1, pts2;
    // in order to be consistent with the cell representation, start with the smallest index
    if(cl1.ptids[0] < cl2.ptids[0])
    {
        pts1 = cl1.ptids;
        pts2 = cl2.ptids;
    }
    else
    {
        pts1 = cl2.ptids;
        pts2 = cl1.ptids;
    }

    // write points of pts1 until we reach the common face
    size_t ref_pt = pts1[0];
    size_t cp = 0;
    ret.ptids.push_back(ref_pt);

    bool on_interface = false;
    if(ref_pt == com_n) on_interface = true;

    while(!on_interface)
    {
        cp++;
        ref_pt = pts1[cp];
        ret.ptids.push_back(ref_pt);
        if(ref_pt == com_n) on_interface = true;
    }

    // look for the corresponding point in pts2
    for(size_t i = 0; i < pts2.size(); i++)
    {
        if(ref_pt == pts2[i])
        {
            cp = i;
            break;
        }
    }
    // write points of pts2 until we reach once more the common node
    on_interface = false;
    while( !on_interface )
    {
        cp = (cp + 1) % pts2.size();
        ref_pt = pts2[cp];
        ret.ptids.push_back(ref_pt);
        if(ref_pt == com_n) on_interface = true;
    }

    // look for the corresponding point in pts1
    for(size_t i=0; i < pts1.size(); i++)
    {
        if(ref_pt == pts1[i])
        {
            cp = i;
            break;
        }
    }

    // finish to write the points of pts1
    cp++;
    while(cp < pts1.size())
    {
        ret.ptids.push_back(pts1[cp]);
        cp++;
    }

    return ret;
}

//////////////  MERGE_CELLS
/// merge cl1 and cl2
// output : the agglomerated cell + list of faces to withdraw
/////  For the moment we can merge only cells that have at least a common face
/////  This procedure currently cannot be iterated
template<typename Mesh>
std::pair<   typename Mesh::cell_type, std::vector<typename Mesh::face_type> >
merge_cells(Mesh& msh, const typename Mesh::cell_type cl1,
            const typename Mesh::cell_type cl2)
{
    //////////////////  TESTS ON INPUTS  //////////////////
    // verify that the two cells are different
    if(cl1 == cl2)
        throw std::invalid_argument("Cannot merge a cell with itself.");

    // identify the common faces
    const auto fcs1 = faces(msh, cl1);
    const auto fcs2 = faces(msh, cl2);

    std::vector<typename Mesh::face_type> com_faces;
    for(size_t i = 0; i < fcs1.size(); i++)
    {
        const auto fc1 = fcs1[i];
        for(size_t j = 0; j < fcs2.size(); j++)
        {
            const auto fc2 = fcs2[j];
            if(fc1 == fc2) com_faces.push_back(fc1);
        }
    }

    // identify the common nodes
    std::set<size_t> com_nodes;
    size_t com_n;
    for(size_t i = 0; i < cl1.ptids.size(); i++)
    {
        for(size_t j = 0; j < cl2.ptids.size(); j++)
        {
            if(cl1.ptids[i] == cl2.ptids[j])
            {
                com_nodes.insert(cl1.ptids[i]);
                com_n = cl1.ptids[i];
            }
        }
    }

    // choose the agglomeration technique
    typename Mesh::cell_type cl;
    if(com_faces.size() == 0)
    {
        std::cout << "com nodes nb = " << com_nodes.size() << std::endl;
        if( com_nodes.size() == 1 )
        {
            cl = merge_cells_diag(cl1, cl2, com_n);
        }
        else
            throw std::invalid_argument("The cells have no common faces.");
    }
    else if( com_faces.size() == 1 )
    {
        assert(com_nodes.size() == 2); // only the nodes of the common face
        cl = merge_cells_face(cl1, cl2, com_faces[0]);
    }
    else if( com_faces.size() == 2 )
    {
        std::cout << "com nodes nb = " << com_nodes.size() << std::endl;
        assert(com_nodes.size() == 3); // only the case with two adjascent common faces
        cl = merge_cells_two_faces(cl1, cl2, com_faces);
    }
    if(com_faces.size() > 2)
        throw std::invalid_argument("The cells have too many common faces.");

    //////////////    COMPLETE THE MERGED CELL   //////////////

    ///////////// build the cell_cuthho_info (using the sub_cells info)
    // location
    if(cl1.user_data.location == element_location::ON_INTERFACE || cl2.user_data.location == element_location::ON_INTERFACE)
        cl.user_data.location = element_location::ON_INTERFACE;
    else if(cl1.user_data.location == element_location::IN_NEGATIVE_SIDE)
        cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
    else if(cl1.user_data.location == element_location::IN_POSITIVE_SIDE)
        cl.user_data.location = element_location::IN_POSITIVE_SIDE;
    else
        throw std::logic_error("we shouldn't arrive here (cuthho_info) !!!");

    // agglo_set ---> NOT DONE : needed to iterate the agglomeration procedure
    cl.user_data.agglo_set = cell_agglo_set::T_OK;

    // p0, p1 and interface
    bool cut1 = cl1.user_data.location == element_location::ON_INTERFACE;
    bool cut2 = cl2.user_data.location == element_location::ON_INTERFACE;
    if(cut1 && !cut2)
    {
        cl.user_data.interface = cl1.user_data.interface;
        cl.user_data.p0 = cl1.user_data.p0;
        cl.user_data.p1 = cl1.user_data.p1;
    }
    else if(!cut1 && cut2)
    {
        cl.user_data.interface = cl2.user_data.interface;
        cl.user_data.p0 = cl2.user_data.p0;
        cl.user_data.p1 = cl2.user_data.p1;
    }
    else if(cut1 && cut2)
    {
        if(cl1.user_data.p0[0] == cl2.user_data.p1[0] &&
           cl1.user_data.p0[1] == cl2.user_data.p1[1])
        {
            cl.user_data.interface = cl2.user_data.interface;
            for(size_t i = 0; i < cl1.user_data.interface.size(); i++ )
            {
                cl.user_data.interface.push_back(cl1.user_data.interface[i]);
            }
            cl.user_data.p0 = cl2.user_data.p0;
            cl.user_data.p1 = cl1.user_data.p1;
        }
        else if(cl2.user_data.p0[0] == cl1.user_data.p1[0] &&
                cl2.user_data.p0[1] == cl1.user_data.p1[1])
        {
            cl.user_data.interface = cl1.user_data.interface;
            for(size_t i = 0; i < cl2.user_data.interface.size(); i++ )
            {
                cl.user_data.interface.push_back(cl2.user_data.interface[i]);
            }
            cl.user_data.p0 = cl1.user_data.p0;
            cl.user_data.p1 = cl2.user_data.p1;
        }
        else{
            std::cout<<"cl1 = "<<offset(msh,cl1)<<" , cl2 = "<<offset(msh,cl2)<<std::endl;
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
            std::cout<<"POINT INTERFACE_0 = "<<cl.user_data.p0 << " , INTERFACE_1 = "<<cl.user_data.p1 << std::endl;
            
            std::cout<<"cl1 = "<<offset(msh,cl1)<<" , cl2 = "<<offset(msh,cl2)<<std::endl;
            std::cout<<"cl2.user_data.p0[0] = "<<cl2.user_data.p0[0]<<" , cl2.user_data.p0[1] = "<<cl2.user_data.p0[1]<<std::endl;
            std::cout<<"cl2.user_data.p1[0] = "<<cl2.user_data.p1[0]<<" , cl2.user_data.p1[1] = "<<cl2.user_data.p1[1]<<std::endl;
            
            std::cout<<"cl1.user_data.p0[0] = "<<cl1.user_data.p0[0]<<" , cl1.user_data.p0[1] = "<<cl1.user_data.p0[1]<<std::endl;
            std::cout<<"cl1.user_data.p1[0] = "<<cl1.user_data.p1[0]<<" , cl1.user_data.p1[1] = "<<cl1.user_data.p1[1]<<std::endl;
            std::cout<<"cut1 = "<<cut1<<" , cut2 = "<<cut2<<std::endl;
            throw std::logic_error("we shouldn't arrive here (interface) !!!");
        }
    }
    // distorted -> has to be updated for more general merges (if a node is withdrawn)
    if(cl1.user_data.distorted || cl2.user_data.distorted )
        cl.user_data.distorted = true;


    // for tests
    cl.user_data.highlight = true;


    // integration -> save composite quadrature
   //  std::cout<<"WARNING: in agglomerated cells the integrations points are saved at priori (for a high degree). COMPUTATIONALLY USELESS."<<std::endl;
    //std::cout<<"----------> SONO IN AGGLO in cell"<<offset(msh,cl)<<std::endl;
    size_t degree_max = 18 ; //8; //////// VERY IMPORTANT !!!!!!! -> max deg for quadratures = 8
  //   std::cout << bold << yellow << "Before integrate 1" << reset << std::endl;
    auto integration1_n = integrate(msh, cl1, degree_max, element_location::IN_NEGATIVE_SIDE);
  //  std::cout << bold << yellow << "Before integrate 2" << reset << std::endl;
    auto integration1_p = integrate(msh, cl1, degree_max, element_location::IN_POSITIVE_SIDE);
   //std::cout << bold << yellow << "Before integrate 3" << reset << std::endl;
    auto integration2_n = integrate(msh, cl2, degree_max, element_location::IN_NEGATIVE_SIDE);
 //   std::cout << bold << yellow << "Before integrate 4" << reset << std::endl;
    auto integration2_p = integrate(msh, cl2, degree_max, element_location::IN_POSITIVE_SIDE);
    //std::cout<<"SONO IN AGGLO FINE <----------"<<std::endl;
    cl.user_data.integration_n = integration1_n;
    cl.user_data.integration_p = integration1_p;
    for(size_t i = 0; i < integration2_n.size(); i++)
        cl.user_data.integration_n.push_back( integration2_n.at(i) );

    for(size_t i = 0; i < integration2_p.size(); i++)
        cl.user_data.integration_p.push_back( integration2_p.at(i) );

    // neighbors ---> NOT DONE : needed to iterate the agglomeration procedure

    return std::make_pair(cl, com_faces);
}

template<typename Mesh>
std::pair<   typename Mesh::cell_type, std::vector<typename Mesh::face_type> >
merge_cells_no_double_pts(Mesh& msh, const typename Mesh::cell_type cl1,
            const typename Mesh::cell_type cl2 , size_t degree_det_jac_curve)
{
    //////////////////  TESTS ON INPUTS  //////////////////
    // verify that the two cells are different
    if(cl1 == cl2)
        throw std::invalid_argument("Cannot merge a cell with itself.");

    // identify the common faces
    const auto fcs1 = faces(msh, cl1);
    const auto fcs2 = faces(msh, cl2);

    std::vector<typename Mesh::face_type> com_faces;
    for(size_t i = 0; i < fcs1.size(); i++)
    {
        const auto fc1 = fcs1[i];
        for(size_t j = 0; j < fcs2.size(); j++)
        {
            const auto fc2 = fcs2[j];
            if(fc1 == fc2) com_faces.push_back(fc1);
        }
    }

    // identify the common nodes
    std::set<size_t> com_nodes;
    size_t com_n;
    for(size_t i = 0; i < cl1.ptids.size(); i++)
    {
        for(size_t j = 0; j < cl2.ptids.size(); j++)
        {
            if(cl1.ptids[i] == cl2.ptids[j])
            {
                com_nodes.insert(cl1.ptids[i]);
                com_n = cl1.ptids[i];
            }
        }
    }

    // choose the agglomeration technique
    typename Mesh::cell_type cl;
    if(com_faces.size() == 0)
    {
        std::cout << "com nodes nb = " << com_nodes.size() << std::endl;
        if( com_nodes.size() == 1 )
        {
            cl = merge_cells_diag(cl1, cl2, com_n);
        }
        else
            throw std::invalid_argument("The cells have no common faces.");
    }
    else if( com_faces.size() == 1 )
    {
        assert(com_nodes.size() == 2); // only the nodes of the common face
        cl = merge_cells_face(cl1, cl2, com_faces[0]);
    }
    else if( com_faces.size() == 2 )
    {
        std::cout << "com nodes nb = " << com_nodes.size() << std::endl;
        assert(com_nodes.size() == 3); // only the case with two adjascent common faces
        cl = merge_cells_two_faces(cl1, cl2, com_faces);
    }
    if(com_faces.size() > 2)
        throw std::invalid_argument("The cells have too many common faces.");

    //////////////    COMPLETE THE MERGED CELL   //////////////

    ///////////// build the cell_cuthho_info (using the sub_cells info)
    // location
    if(cl1.user_data.location == element_location::ON_INTERFACE || cl2.user_data.location == element_location::ON_INTERFACE)
        cl.user_data.location = element_location::ON_INTERFACE;
    else if(cl1.user_data.location == element_location::IN_NEGATIVE_SIDE)
        cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
    else if(cl1.user_data.location == element_location::IN_POSITIVE_SIDE)
        cl.user_data.location = element_location::IN_POSITIVE_SIDE;
    else
        throw std::logic_error("we shouldn't arrive here (cuthho_info) !!!");

    // agglo_set ---> NOT DONE : needed to iterate the agglomeration procedure
    cl.user_data.agglo_set = cell_agglo_set::T_OK;

    // p0, p1 and interface
    bool cut1 = cl1.user_data.location == element_location::ON_INTERFACE;
    bool cut2 = cl2.user_data.location == element_location::ON_INTERFACE;
    if(cut1 && !cut2)
    {
        cl.user_data.interface = cl1.user_data.interface;
        cl.user_data.p0 = cl1.user_data.p0;
        cl.user_data.p1 = cl1.user_data.p1;
    }
    else if(!cut1 && cut2)
    {
        cl.user_data.interface = cl2.user_data.interface;
        cl.user_data.p0 = cl2.user_data.p0;
        cl.user_data.p1 = cl2.user_data.p1;
    }
    else if(cut1 && cut2)
    {
        if(cl1.user_data.p0[0] == cl2.user_data.p1[0] &&
           cl1.user_data.p0[1] == cl2.user_data.p1[1])
        {
            cl.user_data.interface = cl2.user_data.interface;
            for(size_t i = 1; i < cl1.user_data.interface.size(); i++ )
            {
                cl.user_data.interface.push_back(cl1.user_data.interface[i]);
            }
            cl.user_data.p0 = cl2.user_data.p0;
            cl.user_data.p1 = cl1.user_data.p1;
        }
        else if(cl2.user_data.p0[0] == cl1.user_data.p1[0] &&
                cl2.user_data.p0[1] == cl1.user_data.p1[1])
        {
            cl.user_data.interface = cl1.user_data.interface;
            for(size_t i = 1; i < cl2.user_data.interface.size(); i++ )
            {
                cl.user_data.interface.push_back(cl2.user_data.interface[i]);
            }
            cl.user_data.p0 = cl1.user_data.p0;
            cl.user_data.p1 = cl2.user_data.p1;
        }
        else{
            std::cout<<"cl1 = "<<offset(msh,cl1)<<" , cl2 = "<<offset(msh,cl2)<<std::endl;
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
            std::cout<<"POINT INTERFACE_0 = "<<cl.user_data.p0 << " , INTERFACE_1 = "<<cl.user_data.p1 << std::endl;
            
            std::cout<<"cl1 = "<<offset(msh,cl1)<<" , cl2 = "<<offset(msh,cl2)<<std::endl;
            std::cout<<"cl2.user_data.p0[0] = "<<cl2.user_data.p0[0]<<" , cl2.user_data.p0[1] = "<<cl2.user_data.p0[1]<<std::endl;
            std::cout<<"cl2.user_data.p1[0] = "<<cl2.user_data.p1[0]<<" , cl2.user_data.p1[1] = "<<cl2.user_data.p1[1]<<std::endl;
            
            std::cout<<"cl1.user_data.p0[0] = "<<cl1.user_data.p0[0]<<" , cl1.user_data.p0[1] = "<<cl1.user_data.p0[1]<<std::endl;
            std::cout<<"cl1.user_data.p1[0] = "<<cl1.user_data.p1[0]<<" , cl1.user_data.p1[1] = "<<cl1.user_data.p1[1]<<std::endl;
            std::cout<<"cut1 = "<<cut1<<" , cut2 = "<<cut2<<std::endl;
            throw std::logic_error("we shouldn't arrive here (interface) !!!");
        }
    }
    // distorted -> has to be updated for more general merges (if a node is withdrawn)
    if(cl1.user_data.distorted || cl2.user_data.distorted )
        cl.user_data.distorted = true;


    // for tests
    cl.user_data.highlight = true;


    // integration -> save composite quadrature
   //  std::cout<<"WARNING: in agglomerated cells the integrations points are saved at priori (for a high degree). COMPUTATIONALLY USELESS."<<std::endl;
    //std::cout<<"----------> SONO IN AGGLO in cell"<<offset(msh,cl)<<std::endl;
    
//    std::cout<<"cl1 = "<<offset(msh,cl1)<<std::endl;
//    auto pts1 = points( msh , cl1);
//    for(auto& pt: pts1)
//        std::cout<<"Cell pt = "<<'\n'<<pt<<'\n'<<std::endl;
//    
//    std::cout<<"cl2 = "<<offset(msh,cl2)<<std::endl;
//        auto pts2 = points( msh , cl2);
//    for(auto& pt: pts2)
//        std::cout<<"Cell pt = "<<'\n'<<pt<<std::endl;
//    size_t degree_j = degree_det_jacobian(cl1.user_data.integration_msh.degree_curve) ;
    size_t degree_max = 18 - degree_det_jacobian(cl1.user_data.integration_msh.degree_curve) ; //8; //////// VERY IMPORTANT !!!!!!! -> max deg for quadratures = 8
  //   std::cout << bold << yellow << "Before integrate 1" << reset << std::endl;
    auto integration1_n = integrate(msh, cl1, degree_max, element_location::IN_NEGATIVE_SIDE);
  //  std::cout << bold << yellow << "Before integrate 2" << reset << std::endl;
    auto integration1_p = integrate(msh, cl1, degree_max, element_location::IN_POSITIVE_SIDE);
   //std::cout << bold << yellow << "Before integrate 3" << reset << std::endl;
    auto integration2_n = integrate(msh, cl2, degree_max, element_location::IN_NEGATIVE_SIDE);
 //   std::cout << bold << yellow << "Before integrate 4" << reset << std::endl;
    auto integration2_p = integrate(msh, cl2, degree_max, element_location::IN_POSITIVE_SIDE);
    //std::cout<<"SONO IN AGGLO FINE <----------"<<std::endl;
    cl.user_data.integration_n = integration1_n;
    cl.user_data.integration_p = integration1_p;
    for(size_t i = 0; i < integration2_n.size(); i++)
        cl.user_data.integration_n.push_back( integration2_n.at(i) );

    for(size_t i = 0; i < integration2_p.size(); i++)
        cl.user_data.integration_p.push_back( integration2_p.at(i) );

    // neighbors ---> NOT DONE : needed to iterate the agglomeration procedure

    return std::make_pair(cl, com_faces);
}


////// loc_agglo
// container for information about local agglomerations
// main_cell is defined when they are at least 3 cells
template<typename cell_type>
class loc_agglo
{
public:
    cell_type main_cell;
    std::vector<cell_type> cells;

    cell_type new_cell;

    loc_agglo( size_t offset1, size_t offset2, cell_type cl1, cell_type cl2, cell_type n_cell)
    {
        // check that the two cells are neighbors
        auto pts1 = cl1.ptids;
        auto pts2 = cl2.ptids;
        std::vector<size_t> com_nodes;
        for(size_t i = 0; i < pts1.size(); i++)
        {
            for(size_t j = 0; j < pts2.size(); j++)
            {
                if(pts1[i] == pts2[j]) com_nodes.push_back(pts1[i]);
            }
        }

        assert( com_nodes.size() > 0 );

        // init
        cells.push_back( cl1 );
        cells.push_back( cl2 );

        new_cell = n_cell;
        new_cell.user_data.offset_subcells.push_back(offset1);
        new_cell.user_data.offset_subcells.push_back(offset2);
    }

    bool is_agglo_possible(size_t offset)
    {
        // when there are 2 cells, the main cell is not defined -> check all cells
        if( cells.size() == 2 )
        {
            return (cells[0].user_data.f_neighbors.find(offset) != cells[0].user_data.f_neighbors.end())
                || (cells[1].user_data.f_neighbors.find(offset) != cells[1].user_data.f_neighbors.end())
                || (cells[0].user_data.d_neighbors.find(offset) != cells[0].user_data.d_neighbors.end())
                || (cells[1].user_data.d_neighbors.find(offset) != cells[1].user_data.d_neighbors.end());
        }

        // else the main cell is defined -> can agglomerate only if it is a neighbor of that cell
        return (main_cell.user_data.f_neighbors.find(offset) != main_cell.user_data.f_neighbors.end())
            || (main_cell.user_data.d_neighbors.find(offset) != main_cell.user_data.d_neighbors.end());
    }

    bool is_in(cell_type cl)
    {
        bool ret = false;
        for(size_t i = 0; i < cells.size(); i++)
        {
            if( cells[i] == cl)
            {
                ret = true;
                break;
            }
        }
        return ret;
    }

    void add_cell(cell_type cl_added, size_t offset, cell_type new_new_cell)
    {
        assert( is_agglo_possible(offset) );
        
        
        // check that the cell is not already in the agglomeration
        if( is_in(cl_added) )
            throw std::logic_error("Cell already agglomerated !!");
            
        // if there are only two cells, we also have to define the main cell
        if( cells.size() == 2 )
        {
            if ( cells[0].user_data.f_neighbors.find(offset) != cells[0].user_data.f_neighbors.end()
                 || cells[0].user_data.d_neighbors.find(offset) != cells[0].user_data.d_neighbors.end() )
                main_cell = cells[0];
            else
                main_cell = cells[1];
        }

        cells.push_back(cl_added);
        // Adding the subcells offeset to the new agglomerated cell
        // Stefano from here
        for( auto& i:new_cell.user_data.offset_subcells )
            new_new_cell.user_data.offset_subcells.push_back(i);
        
        new_new_cell.user_data.offset_subcells.push_back(offset);
        // Stefano to here
        new_cell = new_new_cell; // I don't know if I am passing all the properties
       // std::cout<<"Adding a cell in agglomeration: CHECK SUBCELLS OF new_cell!!!"<<std::endl;
       // std::cout<<"Number of subcells "<< new_cell.user_data.offset_subcells.size()<<std::endl;
    }
};

////// find_good_neighbor
template<typename Mesh>
typename Mesh::cell_type
find_good_neighbor(const Mesh& msh, const typename Mesh::cell_type cl,
                   const element_location where)
{
    typename Mesh::cell_type best_neigh = cl;

    element_location other_where;
    if(where == element_location::IN_NEGATIVE_SIDE)
        other_where = element_location::IN_POSITIVE_SIDE;
    else
        other_where = element_location::IN_NEGATIVE_SIDE;

    // look for a neighboring cell to merge with
    typename Mesh::coordinate_type area = 1000000;

    auto f_neigh = cl.user_data.f_neighbors;

    for (std::set<size_t>::iterator it = f_neigh.begin(); it != f_neigh.end(); ++it)
    {
        auto cl_n = msh.cells[*it];

        // if cl_n is on the wrong size -> do not consider it
        if (location(msh, cl_n) != where
            && location(msh, cl_n) != element_location::ON_INTERFACE)
            continue;


        // if cl_n is a small cut of the same size -> do not consider it
        if (where == element_location::IN_NEGATIVE_SIDE && cl_n.user_data.agglo_set == cell_agglo_set::T_KO_NEG)
            continue;
        if (where == element_location::IN_POSITIVE_SIDE && cl_n.user_data.agglo_set == cell_agglo_set::T_KO_POS)
            continue;

        // search for the "best" neighbor -> the one with the smallest volume in other_where
        if( area > measure(msh, cl_n, other_where) )
        {
            area = measure(msh, cl_n, other_where);
            best_neigh = cl_n;
        }
    }

    if(best_neigh == cl) // no possible face agglomerations
    {
        // look for a diagonal agglomeration
        auto d_neigh = cl.user_data.d_neighbors;

        for (std::set<size_t>::iterator it = d_neigh.begin(); it != d_neigh.end(); ++it)
        {
            auto cl_n = msh.cells[*it];

            // if cl_n is on the wrong size or cut -> do not consider it
            if (location(msh, cl_n) != where)
                continue;

            // search for the "best" neighbor -> the one with the smallest volume
            if( area > measure(msh, cl_n, other_where) )
            {
                area = measure(msh, cl_n, other_where);
                best_neigh = cl_n;
            }
        }
    }

    if(best_neigh == cl)
    {
        throw std::logic_error("No possible agglomerations !!!");
    }

    
    return best_neigh;
}

/////// check_corner
// do another agglomeration if the agglomerated cell is cut four times
// needed for the case of a square interface
// cl_agglo : a cell that is already agglomerated with other cells
// cl : a cell that is not yet agglomerated
// return: a boolean: whether another agglomeration is needed
//         a cell: the cell added to the agglomeration
template<typename Mesh>
std::pair< bool, typename Mesh::cell_type >
check_corner(Mesh& msh, typename Mesh::cell_type cl_agglo, typename Mesh::cell_type cl)
{
    typename Mesh::cell_type ret_cl;
    
    // check is another agglo is needed
    bool need_other_agglo = true;

    if( cl_agglo.user_data.p0.x() == cl.user_data.p1.x()
        && cl_agglo.user_data.p0.y() == cl.user_data.p1.y() )
        need_other_agglo = false;
    if( cl_agglo.user_data.p1.x() == cl.user_data.p0.x()
        && cl_agglo.user_data.p1.y() == cl.user_data.p0.y() )
        need_other_agglo = false;
    if( cl_agglo.user_data.p0.x() == cl.user_data.p0.x()
        && cl_agglo.user_data.p0.y() == cl.user_data.p0.y() )
        need_other_agglo = false;
    if( cl_agglo.user_data.p1.x() == cl.user_data.p1.x()
        && cl_agglo.user_data.p1.y() == cl.user_data.p1.y() )
        need_other_agglo = false;

    if(!need_other_agglo)
        return std::make_pair(false, ret_cl);

    std::cout << "agglomerate one more cell !!" << std::endl;

    // look for a cell that shares two intersection points with cl_agglo and cl
    // this cell is searched among the face neighbors
    auto f_neigh = cl.user_data.f_neighbors;
    typename Mesh::cell_type added_cell;
    bool found_added_cell = false;
    size_t offset_added_cell;
    for (std::set<size_t>::iterator it = f_neigh.begin(); it != f_neigh.end(); ++it)
    {
        auto cl_f = msh.cells[*it];
        if(location(msh, cl_f) != element_location::ON_INTERFACE)
            continue;

        if( (  cl_f.user_data.p0.x() == cl.user_data.p0.x()
               && cl_f.user_data.p0.y() == cl.user_data.p0.y() )
            || (cl_f.user_data.p1.x() == cl.user_data.p0.x()
                && cl_f.user_data.p1.y() == cl.user_data.p0.y() )
            || (cl_f.user_data.p0.x() == cl.user_data.p1.x()
                && cl_f.user_data.p0.y() == cl.user_data.p1.y() )
            || (cl_f.user_data.p1.x() == cl.user_data.p1.x()
                && cl_f.user_data.p1.y() == cl.user_data.p1.y() )
            )
        { // this cell is connected with cl
            if( (  cl_f.user_data.p0.x() == cl_agglo.user_data.p0.x()
                   && cl_f.user_data.p0.y() == cl_agglo.user_data.p0.y() )
                || (cl_f.user_data.p1.x() == cl_agglo.user_data.p0.x()
                    && cl_f.user_data.p1.y() == cl_agglo.user_data.p0.y() )
                || (cl_f.user_data.p0.x() == cl_agglo.user_data.p1.x()
                    && cl_f.user_data.p0.y() == cl_agglo.user_data.p1.y() )
                || (cl_f.user_data.p1.x() == cl_agglo.user_data.p1.x()
                    && cl_f.user_data.p1.y() == cl_agglo.user_data.p1.y() )
                )
            { // this cell is connected with cl_agglo
                added_cell = cl_f;
                offset_added_cell = *it;
                found_added_cell = true;
                break;
            }
        }
    }
    
    if(!found_added_cell)    
        throw std::logic_error("added_cell not found !!");

    return std::make_pair(true, added_cell);

}


////// output_agglo_lists
// output a .okc file
// can plot arrows with visit
// represent the lists of agglomerations
template<typename Mesh>
void
output_agglo_lists(Mesh& msh, std::vector<int> table_neg, std::vector<int> table_pos,
                   std::string file)
{
    // number of arrows
    size_t nb_arrows = 0;
    for(size_t i=0; i<table_neg.size(); i++)
    {
        if(table_neg.at(i) != -1)
            nb_arrows++;
        if(table_pos.at(i) != -1)
            nb_arrows++;
    }

    // initiate the output file
    std::ofstream output(file, std::ios::out | std::ios::trunc);
    if( !output )
        std::cerr << "agglo output file has not been opened" << std::endl;

    output << "5 " << nb_arrows << " 12" << std::endl;
    output << "x" << std::endl;
    output << "y" << std::endl;
    output << "z" << std::endl;
    output << "u" << std::endl;
    output << "v" << std::endl;

    output << "0.  1.  10" << std::endl;
    output << "0.  1.  10" << std::endl;
    output << "0.  0.  10" << std::endl;
    output << "-1  1.  10" << std::endl;
    output << "-1  1.  10" << std::endl;


    // loop on the cells
    size_t cp = 0;
    for (auto cl : msh.cells)
    {
        size_t TN = table_neg.at(cp);
        if( TN != -1)
        {
            auto bar_cl = barycenter(msh,cl);
            auto bar_neigh = barycenter(msh,msh.cells.at(TN));

            auto vect = bar_neigh - bar_cl;

            output << bar_cl[0] << "   " << bar_cl[1] << "   0.   "
                   << vect[0] << "   " << vect[1] << std::endl;
        }


        size_t TP = table_pos.at(cp);
        if( TP != -1)
        {
            auto bar_cl = barycenter(msh,cl);
            auto bar_neigh = barycenter(msh,msh.cells.at(TP));

            auto vect = bar_neigh - bar_cl;

            output << bar_cl[0] << "   " << bar_cl[1] << "   0.   "
                   << vect[0] << "   " << vect[1] << std::endl;
        }
        cp++;
    }

    output.close();
}

////// make_agglomeration
// the main agglomeration routine
// currently, the mesh obtained must have convex cells
// obtained by merging sub_cells with one face in common
// for non-convex cells -> modify measure and integrate
// for merging more general cells -> modify merge_cells
// for diagonal cells -> modify make_neighbors_info
// no iterations after merging bad cells once
template<typename Mesh, typename Function>
void
make_agglomeration(Mesh& msh, const Function& level_set_function)
{
    // initiate lists to store the agglomeration infos
    std::vector<int> agglo_table_neg, agglo_table_pos;
    size_t nb_cells = msh.cells.size();
    agglo_table_neg.resize(nb_cells);
    agglo_table_pos.resize(nb_cells);
    
    for(size_t i=0; i < nb_cells; i++)
    {
        agglo_table_neg.at(i) = -1;
        agglo_table_pos.at(i) = -1;
    }
    std::cout<<"prova="<<std::endl;
    ///////////////////////   LOOK FOR NEIGHBORS  ////////////////
    size_t nb_step1 = 0;
    size_t nb_step2 = 0;
    //size_t nb_ko_neg_cls = 0;
    //size_t nb_ko_pos_cls = 0;
    // start the process for domain 1, and then domain 2
    for(size_t domain=1; domain < 3; domain++)
    {
        // loop on the cells
        for (auto cl : msh.cells)
        {
            element_location where;

            if(domain == 1)
                where = element_location::IN_NEGATIVE_SIDE;
            else if(domain == 2)
                where = element_location::IN_POSITIVE_SIDE;
            else 
                throw std::logic_error("pb with domain");
                
            if(cl.user_data.agglo_set == cell_agglo_set::T_OK)
                continue;
            else if(cl.user_data.agglo_set == cell_agglo_set::UNDEF)
                throw std::logic_error("UNDEF agglo_set");
            else if(cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG
                    && where != element_location::IN_NEGATIVE_SIDE)
                continue;
            else if(cl.user_data.agglo_set == cell_agglo_set::T_KO_POS
                    && where != element_location::IN_POSITIVE_SIDE)
                continue;


            // if cl is already agglomerated : no need for further agglomerations
            bool already_agglo = false;
            size_t offset_cl = offset(msh, cl);
            for (size_t i = 0; i < agglo_table_pos.size(); i++)
            {
                if( agglo_table_pos.at(i) == offset_cl || agglo_table_neg.at(i) == offset_cl )
                {
                    already_agglo = true;
                    break;
                }
            }
            if( already_agglo )
                continue;


            typename Mesh::cell_type best_neigh = find_good_neighbor(msh, cl, where);

            auto f_neigh = cl.user_data.f_neighbors;

            // prepare agglomeration of cells cl and best_neigh
            size_t offset1 = offset(msh,cl);
            size_t offset2 = offset(msh,best_neigh);

            if(where == element_location::IN_NEGATIVE_SIDE)
            {
                agglo_table_neg.at(offset1) = offset2;
                nb_step1++;
            }
            else
            {
                agglo_table_pos.at(offset1) = offset2;
                nb_step2++;
            }
            
            //if(domain==1)
            //    nb_ko_neg_cls++;
            //if (domain==2)
            //    nb_ko_pos_cls++;
            
        }

        if(domain == 1)
            output_agglo_lists(msh, agglo_table_neg, agglo_table_pos, "agglo_one.okc");
        if(domain == 2)
            output_agglo_lists(msh, agglo_table_neg, agglo_table_pos, "agglo_two.okc");
        
    }
    
    //std::cout<<"Nb of KO_NEG cells "<<nb_ko_neg_cls<<" and of KO_POS "<<nb_ko_pos_cls
    //std::cout<<"Nb of agglomerated cells in neg side "<<nb_step1<<" and in pos side "<<nb_step2<<std::endl;
    //////////////   CHANGE THE AGGLO FOR THE CELLS OF DOMAIN 1 THAT ARE TARGETTED ///////
    size_t nb_step3 = 0;
    for (auto cl : msh.cells)
    {
        if(cl.user_data.agglo_set != cell_agglo_set::T_KO_NEG)
            continue;

        size_t offset1 = offset(msh,cl);

        // are there cells that try to agglomerate with cl ?
        bool agglo = false;
        size_t cl2_offset;
        for(size_t i = 0; i < agglo_table_pos.size(); i++)
        {
            if(agglo_table_pos.at(i) == offset1)
            {
                agglo = true;
                cl2_offset = i;
                break;
            }
        }
        if(!agglo)
            continue;

        // at this point cl2_offset tries to agglomerate with cl
        size_t cl1_agglo = agglo_table_neg.at(offset1);
        
        // -> check that no one tries to agglomerate with cl1_agglo
        agglo = false;
        for(size_t i = 0; i < agglo_table_neg.size(); i++)
        {
            if( i == offset1)
                continue;

            if(agglo_table_neg.at(i) == cl1_agglo)
            {
                agglo = true;
                break;
            }
        }
        if(!agglo && msh.cells.at(cl1_agglo).user_data.agglo_set == cell_agglo_set::T_KO_POS)
            continue;

        // at this point we risk chain agglomerations
        // -> remove the target of cl
        nb_step3++;
        agglo_table_neg.at(offset1) = -1;
    }
    output_agglo_lists(msh, agglo_table_neg, agglo_table_pos, "agglo_three.okc");
    //std::cout<<"Nb of removed cells in neg side "<<nb_step3<<std::endl;
    
    ///////////////////  BUILD LOCAL AGGLOMERATIONS  //////////////////
    std::vector< loc_agglo<typename Mesh::cell_type> > loc_agglos;
    std::vector<typename Mesh::face_type> removed_faces;
    std::vector<typename Mesh::cell_type> removed_cells;
    for (auto cl : msh.cells)
    {
        auto offset_cl = offset(msh, cl);

        bool agglo = false;
        size_t offset_neigh;
        if(agglo_table_pos.at(offset_cl) != -1)
        {
            offset_neigh = agglo_table_pos.at(offset_cl);
            agglo = true;
        }
        if(agglo_table_neg.at(offset_cl) != -1)
        {
            offset_neigh = agglo_table_neg.at(offset_cl);
            agglo = true;
        }

        if(!agglo)
            continue;

        typename Mesh::cell_type neigh = msh.cells.at(offset_neigh);

        
        // test if one of the two cells is already agglomerated
        size_t agglo_offset_cl, agglo_offset_neigh;
        bool already_agglo_cl = false;
        for (size_t i = 0; i < loc_agglos.size(); i++)
        {
            if( loc_agglos.at(i).is_in(cl) )
            {
                already_agglo_cl = true;
                agglo_offset_cl = i;
                break;
            }
        }
        bool already_agglo_neigh = false;
        for (size_t i = 0; i < loc_agglos.size(); i++)
        {
            if( loc_agglos.at(i).is_in(neigh) )
            {
                already_agglo_neigh = true;
                agglo_offset_neigh = i;
                break;
            }
        }

        // the two cells can not be both already agglomerated in different agglomeration sets
        if(already_agglo_cl && already_agglo_neigh)
        {
            if(agglo_offset_neigh != agglo_offset_cl)
            {
                throw std::logic_error("Both cells already agglomerated !!");
            }
            else
                std::cout << "agglo already done" << std::endl;
            // else : the two cells are already agglomerated together : DO NOTHING
        }
        else if(!already_agglo_cl && !already_agglo_neigh)
        {
            
            //std::cout<<"The cell number "<<offset(msh,cl)<<" is aggloremated with the cell num "<<offset(msh,neigh)<<std::endl;
           
            // create a new local agglomeration
            auto MC = merge_cells(msh, cl, neigh);

            loc_agglos.push_back( loc_agglo<typename Mesh::cell_type>(offset(msh,cl),offset(msh,neigh),cl, neigh, MC.first) );

            for(size_t i=0; i<MC.second.size(); i++)
            {
                removed_faces.push_back( MC.second[i] );
            }
            removed_cells.push_back(cl);
            removed_cells.push_back(neigh);
        }
        else // only one cell is already agglomerated
        {
            //std::cout<<"The cell number "<<offset(msh,cl)<<" is aggloremated with the cell num "<<offset(msh,neigh)<<std::endl;
            typename Mesh::cell_type cl1, cl2;
            size_t offset_cl2 , agglo_offset ;
            if(already_agglo_cl)
            {
                removed_cells.push_back(neigh);
                cl2 = neigh;
                offset_cl2 = offset_neigh;
                agglo_offset = agglo_offset_cl;
            }
            else if(already_agglo_neigh)
            {
                removed_cells.push_back(cl);
                cl2 = cl;
                offset_cl2 = offset_cl;
                agglo_offset = agglo_offset_neigh;
            }

            // get the agglomerated cell
            cl1 = loc_agglos.at(agglo_offset).new_cell;
            
            // check if we need one more agglomeration here
            auto CC = check_corner(msh, cl1, cl2);
            if(CC.first)
            {
                auto offset_added_cell = offset(msh, CC.second);
                auto MC_bis = merge_cells(msh, CC.second, cl1);
                loc_agglos.at(agglo_offset).add_cell(CC.second, offset_added_cell,
                                                         MC_bis.first);
                for(size_t i=0; i<MC_bis.second.size(); i++)
                {
                    removed_faces.push_back( MC_bis.second[i] );
                }
                removed_cells.push_back(CC.second);
            }

            // end the merge procedure
            auto MC = merge_cells(msh, cl2, loc_agglos.at(agglo_offset).new_cell);

            loc_agglos.at(agglo_offset).add_cell(cl2, offset_cl2, MC.first);

            for(size_t i=0; i<MC.second.size(); i++)
            {
                removed_faces.push_back( MC.second[i] );
            }
        }
    }
    
    //////////////////////   UPDATE THE MESH   ////////////////////////
    size_t nb_cells_before = msh.cells.size();
    size_t nb_cells_ok = 0;
    size_t nb_cells_ko1 = 0;
    size_t nb_cells_ko2 = 0;
    size_t nb_cut_before = 0;
    for (auto& cl : msh.cells)
    {
        if( location(msh, cl) == element_location::ON_INTERFACE )
            nb_cut_before++;
        else
            continue;

        if( cl.user_data.agglo_set == cell_agglo_set::T_OK )
            nb_cells_ok++;
        else if( cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG )
            nb_cells_ko1++;
        else if( cl.user_data.agglo_set == cell_agglo_set::T_KO_POS )
            nb_cells_ko2++;
        else
            throw std::logic_error("We should not arrive here !!");
    }

    // remove the agglomerated cells
    typename std::vector<typename Mesh::cell_type>::iterator it_RC;
    for(it_RC = removed_cells.begin(); it_RC != removed_cells.end(); it_RC++) {
        msh.cells.erase(std::remove(begin(msh.cells), end(msh.cells), *it_RC ), end(msh.cells));
    }

    
    // add new cells
    for (size_t i = 0; i < loc_agglos.size(); i++)
    {
        msh.cells.push_back(loc_agglos.at(i).new_cell);
    }

    // sort the new list of cells
    std::sort(msh.cells.begin(), msh.cells.end());
    
    // Check out if subcells work correctly
    
    /*
    for(auto&cl:msh.cells)
    {
        //if(cl.user_data.offset_subcells.size()>1){
            std::cout<<"The subcells of "<<offset(msh,cl)<<" are: ";
            for(auto& i: cl.user_data.offset_subcells)
                std::cout<<i<<", ";
            std::cout<<std::endl;
        //}
        
    }
    */
    
    // remove faces
    typename std::vector<typename Mesh::face_type>::iterator it_RF;
    for(it_RF = removed_faces.begin(); it_RF != removed_faces.end(); it_RF++) {
        msh.faces.erase(std::remove(begin(msh.faces), end(msh.faces), *it_RF ), end(msh.faces));
    }

    // sort the new list of faces
    std::sort(msh.faces.begin(), msh.faces.end());

    size_t nb_cells_after = msh.cells.size();
    size_t nb_cut_after = 0;
    for (auto& cl : msh.cells)
    {
        if( location(msh, cl) == element_location::ON_INTERFACE )
            nb_cut_after++;
    }

    ////////////  output some info
    std::ofstream output_cells("output_cells.txt", std::ios::out | std::ios::trunc);
    output_cells << " NB_CELLS_BEFORE = " << nb_cells_before << std::endl;
    output_cells << " NB_CELLS_AFTER = " << nb_cells_after << std::endl;
    output_cells << " NB_CUT_CELLS_BEFORE = " << nb_cut_before << std::endl;
    output_cells << " NB_CUT_CELLS_AFTER = " << nb_cut_after << std::endl;

    output_cells << " NB_CELLS_OK = " << nb_cells_ok << std::endl;
    output_cells << " NB_CELLS_KO1 = " << nb_cells_ko1 << std::endl;
    output_cells << " NB_CELLS_KO2 = " << nb_cells_ko2 << std::endl;

    output_cells << " NB_CELLS_STEP_1 = " << nb_step1 << std::endl;
    output_cells << " NB_CELLS_STEP_2 = " << nb_step2 << std::endl;
    output_cells << " NB_CELLS_STEP_3 = " << nb_step3 << std::endl;

    output_cells.close();
}


template<typename Mesh, typename Function>
void
make_agglomeration_no_double_points(Mesh& msh, const Function& level_set_function, size_t degree_det_jac_curve)
{
    // initiate lists to store the agglomeration infos
    std::vector<int> agglo_table_neg, agglo_table_pos;
    size_t nb_cells = msh.cells.size();
    agglo_table_neg.resize(nb_cells);
    agglo_table_pos.resize(nb_cells);
    
    for(size_t i=0; i < nb_cells; i++)
    {
        agglo_table_neg.at(i) = -1;
        agglo_table_pos.at(i) = -1;
    }

    ///////////////////////   LOOK FOR NEIGHBORS  ////////////////
    size_t nb_step1 = 0;
    size_t nb_step2 = 0;
    //size_t nb_ko_neg_cls = 0;
    //size_t nb_ko_pos_cls = 0;
    // start the process for domain 1, and then domain 2
    for(size_t domain=1; domain < 3; domain++)
    {
        // loop on the cells
        for (auto cl : msh.cells)
        {
            element_location where;

            if(domain == 1)
                where = element_location::IN_NEGATIVE_SIDE;
            else if(domain == 2)
                where = element_location::IN_POSITIVE_SIDE;
            else
                throw std::logic_error("pb with domain");
                
            if(cl.user_data.agglo_set == cell_agglo_set::T_OK)
                continue;
            else if(cl.user_data.agglo_set == cell_agglo_set::UNDEF)
                throw std::logic_error("UNDEF agglo_set");
            else if(cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG
                    && where != element_location::IN_NEGATIVE_SIDE)
                continue;
            else if(cl.user_data.agglo_set == cell_agglo_set::T_KO_POS
                    && where != element_location::IN_POSITIVE_SIDE)
                continue;


            // if cl is already agglomerated : no need for further agglomerations
            bool already_agglo = false;
            size_t offset_cl = offset(msh, cl);
            for (size_t i = 0; i < agglo_table_pos.size(); i++)
            {
                if( agglo_table_pos.at(i) == offset_cl || agglo_table_neg.at(i) == offset_cl )
                {
                    already_agglo = true;
                    break;
                }
            }
            if( already_agglo )
                continue;


            typename Mesh::cell_type best_neigh = find_good_neighbor(msh, cl, where);

            auto f_neigh = cl.user_data.f_neighbors;

            // prepare agglomeration of cells cl and best_neigh
            size_t offset1 = offset(msh,cl);
            size_t offset2 = offset(msh,best_neigh);

            if(where == element_location::IN_NEGATIVE_SIDE)
            {
                agglo_table_neg.at(offset1) = offset2;
                nb_step1++;
            }
            else
            {
                agglo_table_pos.at(offset1) = offset2;
                nb_step2++;
            }
            
            //if(domain==1)
            //    nb_ko_neg_cls++;
            //if (domain==2)
            //    nb_ko_pos_cls++;
            
        }

        if(domain == 1)
            output_agglo_lists(msh, agglo_table_neg, agglo_table_pos, "agglo_one.okc");
        if(domain == 2)
            output_agglo_lists(msh, agglo_table_neg, agglo_table_pos, "agglo_two.okc");
        
    }
    
    //std::cout<<"Nb of KO_NEG cells "<<nb_ko_neg_cls<<" and of KO_POS "<<nb_ko_pos_cls
    //std::cout<<"Nb of agglomerated cells in neg side "<<nb_step1<<" and in pos side "<<nb_step2<<std::endl;
    //////////////   CHANGE THE AGGLO FOR THE CELLS OF DOMAIN 1 THAT ARE TARGETTED ///////
    size_t nb_step3 = 0;
    for (auto cl : msh.cells)
    {
        if(cl.user_data.agglo_set != cell_agglo_set::T_KO_NEG)
            continue;

        size_t offset1 = offset(msh,cl);

        // are there cells that try to agglomerate with cl ?
        bool agglo = false;
        size_t cl2_offset;
        for(size_t i = 0; i < agglo_table_pos.size(); i++)
        {
            if(agglo_table_pos.at(i) == offset1)
            {
                agglo = true;
                cl2_offset = i;
                break;
            }
        }
        if(!agglo)
            continue;

        // at this point cl2_offset tries to agglomerate with cl
        size_t cl1_agglo = agglo_table_neg.at(offset1);
        
        // -> check that no one tries to agglomerate with cl1_agglo
        agglo = false;
        for(size_t i = 0; i < agglo_table_neg.size(); i++)
        {
            if( i == offset1)
                continue;

            if(agglo_table_neg.at(i) == cl1_agglo)
            {
                agglo = true;
                break;
            }
        }
        if(!agglo && msh.cells.at(cl1_agglo).user_data.agglo_set == cell_agglo_set::T_KO_POS)
            continue;

        // at this point we risk chain agglomerations
        // -> remove the target of cl
        nb_step3++;
        agglo_table_neg.at(offset1) = -1;
    }
    output_agglo_lists(msh, agglo_table_neg, agglo_table_pos, "agglo_three.okc");
    //std::cout<<"Nb of removed cells in neg side "<<nb_step3<<std::endl;
    
    ///////////////////  BUILD LOCAL AGGLOMERATIONS  //////////////////
    std::vector< loc_agglo<typename Mesh::cell_type> > loc_agglos;
    std::vector<typename Mesh::face_type> removed_faces;
    std::vector<typename Mesh::cell_type> removed_cells;
    for (auto cl : msh.cells)
    {
        auto offset_cl = offset(msh, cl);

        bool agglo = false;
        size_t offset_neigh;
        if(agglo_table_pos.at(offset_cl) != -1)
        {
            offset_neigh = agglo_table_pos.at(offset_cl);
            agglo = true;
        }
        if(agglo_table_neg.at(offset_cl) != -1)
        {
            offset_neigh = agglo_table_neg.at(offset_cl);
            agglo = true;
        }

        if(!agglo)
            continue;

        typename Mesh::cell_type neigh = msh.cells.at(offset_neigh);

        
        // test if one of the two cells is already agglomerated
        size_t agglo_offset_cl, agglo_offset_neigh;
        bool already_agglo_cl = false;
        for (size_t i = 0; i < loc_agglos.size(); i++)
        {
            if( loc_agglos.at(i).is_in(cl) )
            {
                already_agglo_cl = true;
                agglo_offset_cl = i;
                break;
            }
        }
        bool already_agglo_neigh = false;
        for (size_t i = 0; i < loc_agglos.size(); i++)
        {
            if( loc_agglos.at(i).is_in(neigh) )
            {
                already_agglo_neigh = true;
                agglo_offset_neigh = i;
                break;
            }
        }

        // the two cells can not be both already agglomerated in different agglomeration sets
        if(already_agglo_cl && already_agglo_neigh)
        {
            if(agglo_offset_neigh != agglo_offset_cl)
            {
                throw std::logic_error("Both cells already agglomerated !!");
            }
            else
                std::cout << "agglo already done" << std::endl;
            // else : the two cells are already agglomerated together : DO NOTHING
        }
        else if(!already_agglo_cl && !already_agglo_neigh)
        {
            
            //std::cout<<"The cell number "<<offset(msh,cl)<<" is aggloremated with the cell num "<<offset(msh,neigh)<<std::endl;
           
            // create a new local agglomeration
            auto MC = merge_cells_no_double_pts(msh, cl, neigh,degree_det_jac_curve);

            loc_agglos.push_back( loc_agglo<typename Mesh::cell_type>(offset(msh,cl),offset(msh,neigh),cl, neigh, MC.first) );

            for(size_t i=0; i<MC.second.size(); i++)
            {
                removed_faces.push_back( MC.second[i] );
            }
            removed_cells.push_back(cl);
            removed_cells.push_back(neigh);
        }
        else // only one cell is already agglomerated
        {
            //std::cout<<"The cell number "<<offset(msh,cl)<<" is aggloremated with the cell num "<<offset(msh,neigh)<<std::endl;
            typename Mesh::cell_type cl1, cl2;
            size_t offset_cl2 , agglo_offset ;
            if(already_agglo_cl)
            {
                removed_cells.push_back(neigh);
                cl2 = neigh;
                offset_cl2 = offset_neigh;
                agglo_offset = agglo_offset_cl;
            }
            else if(already_agglo_neigh)
            {
                removed_cells.push_back(cl);
                cl2 = cl;
                offset_cl2 = offset_cl;
                agglo_offset = agglo_offset_neigh;
            }

            // get the agglomerated cell
            cl1 = loc_agglos.at(agglo_offset).new_cell;
            
            // check if we need one more agglomeration here
            auto CC = check_corner(msh, cl1, cl2);
            if(CC.first)
            {
                auto offset_added_cell = offset(msh, CC.second);
                auto MC_bis = merge_cells_no_double_pts(msh, CC.second, cl1,degree_det_jac_curve);
                loc_agglos.at(agglo_offset).add_cell(CC.second, offset_added_cell,
                                                         MC_bis.first);
                for(size_t i=0; i<MC_bis.second.size(); i++)
                {
                    removed_faces.push_back( MC_bis.second[i] );
                }
                removed_cells.push_back(CC.second);
            }

            // end the merge procedure
            auto MC = merge_cells_no_double_pts(msh, cl2, loc_agglos.at(agglo_offset).new_cell,degree_det_jac_curve);

            loc_agglos.at(agglo_offset).add_cell(cl2, offset_cl2, MC.first);

            for(size_t i=0; i<MC.second.size(); i++)
            {
                removed_faces.push_back( MC.second[i] );
            }
        }
    }
    
    //////////////////////   UPDATE THE MESH   ////////////////////////
    size_t nb_cells_before = msh.cells.size();
    size_t nb_cells_ok = 0;
    size_t nb_cells_ko1 = 0;
    size_t nb_cells_ko2 = 0;
    size_t nb_cut_before = 0;
    for (auto& cl : msh.cells)
    {
        if( location(msh, cl) == element_location::ON_INTERFACE )
            nb_cut_before++;
        else
            continue;

        if( cl.user_data.agglo_set == cell_agglo_set::T_OK )
            nb_cells_ok++;
        else if( cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG )
            nb_cells_ko1++;
        else if( cl.user_data.agglo_set == cell_agglo_set::T_KO_POS )
            nb_cells_ko2++;
        else
            throw std::logic_error("We should not arrive here !!");
    }

    // remove the agglomerated cells
    typename std::vector<typename Mesh::cell_type>::iterator it_RC;
    for(it_RC = removed_cells.begin(); it_RC != removed_cells.end(); it_RC++) {
        msh.cells.erase(std::remove(begin(msh.cells), end(msh.cells), *it_RC ), end(msh.cells));
    }

    
    // add new cells
    for (size_t i = 0; i < loc_agglos.size(); i++)
    {
        msh.cells.push_back(loc_agglos.at(i).new_cell);
    }

    // sort the new list of cells
    std::sort(msh.cells.begin(), msh.cells.end());
    
    // Check out if subcells work correctly
    
    /*
    for(auto&cl:msh.cells)
    {
        //if(cl.user_data.offset_subcells.size()>1){
            std::cout<<"The subcells of "<<offset(msh,cl)<<" are: ";
            for(auto& i: cl.user_data.offset_subcells)
                std::cout<<i<<", ";
            std::cout<<std::endl;
        //}
        
    }
    */
    
    // remove faces
    typename std::vector<typename Mesh::face_type>::iterator it_RF;
    for(it_RF = removed_faces.begin(); it_RF != removed_faces.end(); it_RF++) {
        msh.faces.erase(std::remove(begin(msh.faces), end(msh.faces), *it_RF ), end(msh.faces));
    }

    // sort the new list of faces
    std::sort(msh.faces.begin(), msh.faces.end());

    size_t nb_cells_after = msh.cells.size();
    size_t nb_cut_after = 0;
    for (auto& cl : msh.cells)
    {
        if( location(msh, cl) == element_location::ON_INTERFACE )
            nb_cut_after++;
    }

    ////////////  output some info
    std::ofstream output_cells("output_cells.txt", std::ios::out | std::ios::trunc);
    output_cells << " NB_CELLS_BEFORE = " << nb_cells_before << std::endl;
    output_cells << " NB_CELLS_AFTER = " << nb_cells_after << std::endl;
    output_cells << " NB_CUT_CELLS_BEFORE = " << nb_cut_before << std::endl;
    output_cells << " NB_CUT_CELLS_AFTER = " << nb_cut_after << std::endl;

    output_cells << " NB_CELLS_OK = " << nb_cells_ok << std::endl;
    output_cells << " NB_CELLS_KO1 = " << nb_cells_ko1 << std::endl;
    output_cells << " NB_CELLS_KO2 = " << nb_cells_ko2 << std::endl;

    output_cells << " NB_CELLS_STEP_1 = " << nb_step1 << std::endl;
    output_cells << " NB_CELLS_STEP_2 = " << nb_step2 << std::endl;
    output_cells << " NB_CELLS_STEP_3 = " << nb_step3 << std::endl;

    output_cells.close();
}



template<typename Mesh>
void
make_agglomeration_no_double_points(Mesh& msh,  size_t degree_det_jac_curve)
{
    // initiate lists to store the agglomeration infos
    std::vector<int> agglo_table_neg, agglo_table_pos;
    size_t nb_cells = msh.cells.size();
    agglo_table_neg.resize(nb_cells);
    agglo_table_pos.resize(nb_cells);
    
    for(size_t i=0; i < nb_cells; i++)
    {
        agglo_table_neg.at(i) = -1;
        agglo_table_pos.at(i) = -1;
    }

    ///////////////////////   LOOK FOR NEIGHBORS  ////////////////
    size_t nb_step1 = 0;
    size_t nb_step2 = 0;
    //size_t nb_ko_neg_cls = 0;
    //size_t nb_ko_pos_cls = 0;
    // start the process for domain 1, and then domain 2
    for(size_t domain=1; domain < 3; domain++)
    {
        // loop on the cells
        for (auto cl : msh.cells)
        {
            element_location where;

            if(domain == 1)
                where = element_location::IN_NEGATIVE_SIDE;
            else if(domain == 2)
                where = element_location::IN_POSITIVE_SIDE;
            else
                throw std::logic_error("pb with domain");
                
            if(cl.user_data.agglo_set == cell_agglo_set::T_OK)
                continue;
            else if(cl.user_data.agglo_set == cell_agglo_set::UNDEF)
                throw std::logic_error("UNDEF agglo_set");
            else if(cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG
                    && where != element_location::IN_NEGATIVE_SIDE)
                continue;
            else if(cl.user_data.agglo_set == cell_agglo_set::T_KO_POS
                    && where != element_location::IN_POSITIVE_SIDE)
                continue;


            // if cl is already agglomerated : no need for further agglomerations
            bool already_agglo = false;
            size_t offset_cl = offset(msh, cl);
            for (size_t i = 0; i < agglo_table_pos.size(); i++)
            {
                if( agglo_table_pos.at(i) == offset_cl || agglo_table_neg.at(i) == offset_cl )
                {
                    already_agglo = true;
                    break;
                }
            }
            if( already_agglo )
                continue;


            typename Mesh::cell_type best_neigh = find_good_neighbor(msh, cl, where);

            auto f_neigh = cl.user_data.f_neighbors;

            // prepare agglomeration of cells cl and best_neigh
            size_t offset1 = offset(msh,cl);
            size_t offset2 = offset(msh,best_neigh);

            if(where == element_location::IN_NEGATIVE_SIDE)
            {
                agglo_table_neg.at(offset1) = offset2;
                nb_step1++;
            }
            else
            {
                agglo_table_pos.at(offset1) = offset2;
                nb_step2++;
            }
            
            //if(domain==1)
            //    nb_ko_neg_cls++;
            //if (domain==2)
            //    nb_ko_pos_cls++;
            
        }

        if(domain == 1)
            output_agglo_lists(msh, agglo_table_neg, agglo_table_pos, "agglo_one.okc");
        if(domain == 2)
            output_agglo_lists(msh, agglo_table_neg, agglo_table_pos, "agglo_two.okc");
        
    }
    
    //std::cout<<"Nb of KO_NEG cells "<<nb_ko_neg_cls<<" and of KO_POS "<<nb_ko_pos_cls
    //std::cout<<"Nb of agglomerated cells in neg side "<<nb_step1<<" and in pos side "<<nb_step2<<std::endl;
    //////////////   CHANGE THE AGGLO FOR THE CELLS OF DOMAIN 1 THAT ARE TARGETTED ///////
    size_t nb_step3 = 0;
    for (auto cl : msh.cells)
    {
        if(cl.user_data.agglo_set != cell_agglo_set::T_KO_NEG)
            continue;

        size_t offset1 = offset(msh,cl);

        // are there cells that try to agglomerate with cl ?
        bool agglo = false;
        size_t cl2_offset;
        for(size_t i = 0; i < agglo_table_pos.size(); i++)
        {
            if(agglo_table_pos.at(i) == offset1)
            {
                agglo = true;
                cl2_offset = i;
                break;
            }
        }
        if(!agglo)
            continue;

        // at this point cl2_offset tries to agglomerate with cl
        size_t cl1_agglo = agglo_table_neg.at(offset1);
        
        // -> check that no one tries to agglomerate with cl1_agglo
        agglo = false;
        for(size_t i = 0; i < agglo_table_neg.size(); i++)
        {
            if( i == offset1)
                continue;

            if(agglo_table_neg.at(i) == cl1_agglo)
            {
                agglo = true;
                break;
            }
        }
        if(!agglo && msh.cells.at(cl1_agglo).user_data.agglo_set == cell_agglo_set::T_KO_POS)
            continue;

        // at this point we risk chain agglomerations
        // -> remove the target of cl
        nb_step3++;
        agglo_table_neg.at(offset1) = -1;
    }
    output_agglo_lists(msh, agglo_table_neg, agglo_table_pos, "agglo_three.okc");
    //std::cout<<"Nb of removed cells in neg side "<<nb_step3<<std::endl;
    
    ///////////////////  BUILD LOCAL AGGLOMERATIONS  //////////////////
    std::vector< loc_agglo<typename Mesh::cell_type> > loc_agglos;
    std::vector<typename Mesh::face_type> removed_faces;
    std::vector<typename Mesh::cell_type> removed_cells;
    for (auto cl : msh.cells)
    {
        auto offset_cl = offset(msh, cl);

        bool agglo = false;
        size_t offset_neigh;
        if(agglo_table_pos.at(offset_cl) != -1)
        {
            offset_neigh = agglo_table_pos.at(offset_cl);
            agglo = true;
        }
        if(agglo_table_neg.at(offset_cl) != -1)
        {
            offset_neigh = agglo_table_neg.at(offset_cl);
            agglo = true;
        }

        if(!agglo)
            continue;

        typename Mesh::cell_type neigh = msh.cells.at(offset_neigh);

        
        // test if one of the two cells is already agglomerated
        size_t agglo_offset_cl, agglo_offset_neigh;
        bool already_agglo_cl = false;
        for (size_t i = 0; i < loc_agglos.size(); i++)
        {
            if( loc_agglos.at(i).is_in(cl) )
            {
                already_agglo_cl = true;
                agglo_offset_cl = i;
                break;
            }
        }
        bool already_agglo_neigh = false;
        for (size_t i = 0; i < loc_agglos.size(); i++)
        {
            if( loc_agglos.at(i).is_in(neigh) )
            {
                already_agglo_neigh = true;
                agglo_offset_neigh = i;
                break;
            }
        }

        // the two cells can not be both already agglomerated in different agglomeration sets
        if(already_agglo_cl && already_agglo_neigh)
        {
            if(agglo_offset_neigh != agglo_offset_cl)
            {
                throw std::logic_error("Both cells already agglomerated !!");
            }
            else
                std::cout << "agglo already done" << std::endl;
            // else : the two cells are already agglomerated together : DO NOTHING
        }
        else if(!already_agglo_cl && !already_agglo_neigh)
        {
            
            //std::cout<<"The cell number "<<offset(msh,cl)<<" is aggloremated with the cell num "<<offset(msh,neigh)<<std::endl;
           
            // create a new local agglomeration
            auto MC = merge_cells_no_double_pts(msh, cl, neigh,degree_det_jac_curve);

            loc_agglos.push_back( loc_agglo<typename Mesh::cell_type>(offset(msh,cl),offset(msh,neigh),cl, neigh, MC.first) );

            for(size_t i=0; i<MC.second.size(); i++)
            {
                removed_faces.push_back( MC.second[i] );
            }
            removed_cells.push_back(cl);
            removed_cells.push_back(neigh);
        }
        else // only one cell is already agglomerated
        {
            //std::cout<<"The cell number "<<offset(msh,cl)<<" is aggloremated with the cell num "<<offset(msh,neigh)<<std::endl;
            typename Mesh::cell_type cl1, cl2;
            size_t offset_cl2 , agglo_offset ;
            if(already_agglo_cl)
            {
                removed_cells.push_back(neigh);
                cl2 = neigh;
                offset_cl2 = offset_neigh;
                agglo_offset = agglo_offset_cl;
            }
            else if(already_agglo_neigh)
            {
                removed_cells.push_back(cl);
                cl2 = cl;
                offset_cl2 = offset_cl;
                agglo_offset = agglo_offset_neigh;
            }

            // get the agglomerated cell
            cl1 = loc_agglos.at(agglo_offset).new_cell;
            
            // check if we need one more agglomeration here
            auto CC = check_corner(msh, cl1, cl2);
            if(CC.first)
            {
                auto offset_added_cell = offset(msh, CC.second);
                auto MC_bis = merge_cells_no_double_pts(msh, CC.second, cl1,degree_det_jac_curve);
                loc_agglos.at(agglo_offset).add_cell(CC.second, offset_added_cell,
                                                         MC_bis.first);
                for(size_t i=0; i<MC_bis.second.size(); i++)
                {
                    removed_faces.push_back( MC_bis.second[i] );
                }
                removed_cells.push_back(CC.second);
            }

            // end the merge procedure
            auto MC = merge_cells_no_double_pts(msh, cl2, loc_agglos.at(agglo_offset).new_cell,degree_det_jac_curve);

            loc_agglos.at(agglo_offset).add_cell(cl2, offset_cl2, MC.first);

            for(size_t i=0; i<MC.second.size(); i++)
            {
                removed_faces.push_back( MC.second[i] );
            }
        }
    }
    
    //////////////////////   UPDATE THE MESH   ////////////////////////
    size_t nb_cells_before = msh.cells.size();
    size_t nb_cells_ok = 0;
    size_t nb_cells_ko1 = 0;
    size_t nb_cells_ko2 = 0;
    size_t nb_cut_before = 0;
    for (auto& cl : msh.cells)
    {
        if( location(msh, cl) == element_location::ON_INTERFACE )
            nb_cut_before++;
        else
            continue;

        if( cl.user_data.agglo_set == cell_agglo_set::T_OK )
            nb_cells_ok++;
        else if( cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG )
            nb_cells_ko1++;
        else if( cl.user_data.agglo_set == cell_agglo_set::T_KO_POS )
            nb_cells_ko2++;
        else
            throw std::logic_error("We should not arrive here !!");
    }

    // remove the agglomerated cells
    typename std::vector<typename Mesh::cell_type>::iterator it_RC;
    for(it_RC = removed_cells.begin(); it_RC != removed_cells.end(); it_RC++) {
        msh.cells.erase(std::remove(begin(msh.cells), end(msh.cells), *it_RC ), end(msh.cells));
    }

    
    // add new cells
    for (size_t i = 0; i < loc_agglos.size(); i++)
    {
        msh.cells.push_back(loc_agglos.at(i).new_cell);
    }

    // sort the new list of cells
    std::sort(msh.cells.begin(), msh.cells.end());
    
    // Check out if subcells work correctly
    
    /*
    for(auto&cl:msh.cells)
    {
        //if(cl.user_data.offset_subcells.size()>1){
            std::cout<<"The subcells of "<<offset(msh,cl)<<" are: ";
            for(auto& i: cl.user_data.offset_subcells)
                std::cout<<i<<", ";
            std::cout<<std::endl;
        //}
        
    }
    */
    
    // remove faces
    typename std::vector<typename Mesh::face_type>::iterator it_RF;
    for(it_RF = removed_faces.begin(); it_RF != removed_faces.end(); it_RF++) {
        msh.faces.erase(std::remove(begin(msh.faces), end(msh.faces), *it_RF ), end(msh.faces));
    }

    // sort the new list of faces
    std::sort(msh.faces.begin(), msh.faces.end());

    size_t nb_cells_after = msh.cells.size();
    size_t nb_cut_after = 0;
    for (auto& cl : msh.cells)
    {
        if( location(msh, cl) == element_location::ON_INTERFACE )
            nb_cut_after++;
    }

    ////////////  output some info
    std::ofstream output_cells("output_cells.txt", std::ios::out | std::ios::trunc);
    output_cells << " NB_CELLS_BEFORE = " << nb_cells_before << std::endl;
    output_cells << " NB_CELLS_AFTER = " << nb_cells_after << std::endl;
    output_cells << " NB_CUT_CELLS_BEFORE = " << nb_cut_before << std::endl;
    output_cells << " NB_CUT_CELLS_AFTER = " << nb_cut_after << std::endl;

    output_cells << " NB_CELLS_OK = " << nb_cells_ok << std::endl;
    output_cells << " NB_CELLS_KO1 = " << nb_cells_ko1 << std::endl;
    output_cells << " NB_CELLS_KO2 = " << nb_cells_ko2 << std::endl;

    output_cells << " NB_CELLS_STEP_1 = " << nb_step1 << std::endl;
    output_cells << " NB_CELLS_STEP_2 = " << nb_step2 << std::endl;
    output_cells << " NB_CELLS_STEP_3 = " << nb_step3 << std::endl;

    output_cells.close();
}
