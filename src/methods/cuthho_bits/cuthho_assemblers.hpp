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

////////////////////////  STATIC CONDENSATION  //////////////////////////
template<typename T>
std::pair<   Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, 1>  >
static_condensation_compute(const Matrix<T, Dynamic, Dynamic> lhs, const Matrix<T, Dynamic, 1> rhs,
                            const size_t cell_size, const size_t face_size)
{
    size_t size_tot = cell_size + face_size;
    assert(lhs.cols() == size_tot && lhs.rows() == size_tot);
    assert(rhs.rows() == size_tot || rhs.rows() == cell_size);

    Matrix<T, Dynamic, Dynamic> lhs_sc = Matrix<T, Dynamic, Dynamic>::Zero(face_size, face_size);
    Matrix<T, Dynamic, 1> rhs_sc = Matrix<T, Dynamic, 1>::Zero(face_size);


    // sub--lhs
    Matrix<T, Dynamic, Dynamic> K_TT = lhs.topLeftCorner(cell_size, cell_size);
    Matrix<T, Dynamic, Dynamic> K_TF = lhs.topRightCorner(cell_size, face_size);
    Matrix<T, Dynamic, Dynamic> K_FT = lhs.bottomLeftCorner(face_size, cell_size);
    Matrix<T, Dynamic, Dynamic> K_FF = lhs.bottomRightCorner(face_size, face_size);

    // sub--rhs
    Matrix<T, Dynamic, 1> cell_rhs = Matrix<T, Dynamic, 1>::Zero(cell_size);
    Matrix<T, Dynamic, 1> face_rhs = Matrix<T, Dynamic, 1>::Zero(face_size);
    if(rhs.rows() == cell_size)
        cell_rhs = rhs;
    else
    {
        cell_rhs = rhs.head(cell_size);
        face_rhs = rhs.tail(face_size);
    }

    // static condensation
    auto K_TT_ldlt = K_TT.ldlt();
    Matrix<T, Dynamic, Dynamic> AL = K_TT_ldlt.solve(K_TF);
    Matrix<T, Dynamic, 1> bL = K_TT_ldlt.solve(cell_rhs);

    lhs_sc = K_FF - K_FT * AL;
    rhs_sc = face_rhs - K_FT * bL;

    return std::make_pair(lhs_sc, rhs_sc);
}

template<typename T>
Matrix<T, Dynamic, 1>
static_condensation_recover(const Matrix<T, Dynamic, Dynamic> lhs, const Matrix<T, Dynamic, 1> rhs,
                            const size_t cell_size, const size_t face_size,
                            const Matrix<T, Dynamic, 1> solF)
{
    size_t size_tot = cell_size + face_size;
    assert(lhs.cols() == size_tot && lhs.rows() == size_tot);
    assert(rhs.rows() == size_tot || rhs.rows() == cell_size);
    assert(solF.rows() == face_size);

    // sub--lhs
    Matrix<T, Dynamic, Dynamic> K_TT = lhs.topLeftCorner(cell_size, cell_size);
    Matrix<T, Dynamic, Dynamic> K_TF = lhs.topRightCorner(cell_size, face_size);

    // sub--rhs
    Matrix<T, Dynamic, 1> cell_rhs = Matrix<T, Dynamic, 1>::Zero(cell_size);
    cell_rhs = rhs.head(cell_size);

    // recover cell solution
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(size_tot);

    ret.head(cell_size) = K_TT.ldlt().solve(cell_rhs - K_TF*solF);
    ret.tail(face_size) = solF;

    return ret;
}

/////////////////////////////  ASSEMBLY  INDEX ///////////////////////
// used in all assemblers
class assembly_index
{
    size_t  idx;
    bool    assem;

public:
    assembly_index(size_t i, bool as)
        : idx(i), assem(as)
        {}

    operator size_t() const
        {
            if (!assem)
                throw std::logic_error("Invalid assembly_index");

            return idx;
        }

    bool assemble() const
        {
            return assem;
        }

    friend std::ostream& operator<<(std::ostream& os, const assembly_index& as)
        {
            os << "(" << as.idx << "," << as.assem << ")";
            return os;
        }
};


/******************************************************************************************/
/*******************                                               ************************/
/*******************               SCALAR ASSEMBLERS               ************************/
/*******************                                               ************************/
/******************************************************************************************/


template<typename Mesh, typename Function>
class virt_scalar_assembler
{
    using T = typename Mesh::coordinate_type;

protected:
    std::vector< Triplet<T> >           triplets;
    std::vector< Triplet<T> >           triplets_mass;
    std::vector<size_t>                 face_table;
    std::vector<size_t>                 cell_table;

    hho_degree_info                     di;
    Function                            dir_func;

    element_location loc_zone; // IN_NEGATIVE_SIDE or IN_POSITIVE_SIDE for fictitious problem
                               // ON_INTERFACE for the interface problem
    size_t num_cells, num_other_faces, loc_cbs;

public:

    SparseMatrix<T>         LHS;
    SparseMatrix<T>         MASS;
    Matrix<T, Dynamic, 1>   RHS;


    virt_scalar_assembler(const Mesh& msh, const Function& dirichlet_bf, hho_degree_info hdi)
        : dir_func(dirichlet_bf), di(hdi)
    {
    }

    size_t
    face_SOL_offset(const Mesh& msh, const typename Mesh::face_type& fc)
    {
        auto facdeg = di.face_degree();
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto cbs = loc_cbs; // cbs = 0 if static condensation

        auto face_offset = offset(msh, fc);
        return num_cells * cbs + face_table.at(face_offset) * fbs;
    }

    std::vector<assembly_index>
    init_asm_map(const Mesh& msh, const typename Mesh::cell_type& cl)
    {
        bool double_unknowns = ( location(msh, cl) == element_location::ON_INTERFACE
                                 && loc_zone == element_location::ON_INTERFACE );

        std::vector<assembly_index> asm_map;

        auto facdeg = di.face_degree();
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto f_dofs = num_faces * fbs;
        auto cbs = loc_cbs;
        auto loc_size = cbs + f_dofs;
        if( double_unknowns )
            loc_size = 2 * loc_size;
        asm_map.reserve( loc_size );

        size_t cell_offset = cell_table.at( offset(msh, cl) );
        size_t cell_LHS_offset = cell_offset * cbs;

        if( double_unknowns )
            cbs = 2 * cbs;

        for (size_t i = 0; i < cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );


        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_LHS_offset = face_SOL_offset(msh, fc);

            bool in_dom = true;
            if( loc_zone != element_location::ON_INTERFACE )
            {
                element_location loc_fc = location(msh, fc);
                in_dom = (loc_fc == element_location::ON_INTERFACE ||
                          loc_fc == loc_zone);
            }

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET
                && in_dom;

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );
        }

        if( double_unknowns )
        {
            for (size_t face_i = 0; face_i < num_faces; face_i++)
            {
                auto fc = fcs[face_i];
                auto d = (location(msh, fc) == element_location::ON_INTERFACE) ? fbs : 0;
                auto face_LHS_offset = face_SOL_offset(msh, fc) + d;

                bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
            if ( dirichlet )
                    std::cout << "Dirichlet boundary on cut cell detected." << std::endl;
//                    throw std::invalid_argument("Dirichlet boundary on cut cell not supported.");

                for (size_t i = 0; i < fbs; i++)
                    asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );
            }
        }

        return asm_map;
    }
    
    std::vector<assembly_index>
    init_asm_map_extended(const Mesh& msh, const typename Mesh::cell_type& cl)
    {
        bool double_unknowns = ( location(msh, cl) == element_location::ON_INTERFACE
                                 && loc_zone == element_location::ON_INTERFACE );

        std::vector<assembly_index> asm_map;

        // DOFS
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        asm_map.reserve(cl.user_data.local_dofs);

        // CELL OFFSET
        size_t cell_offset = cell_table.at(offset(msh, cl)); 
        size_t cell_LHS_offset = cell_offset*cbs;
        auto cbs_cut = 2*cbs; // CELL DEGREES OF FREEDOM OF DEPENDENT CELLS 

        ///////////////////////////////////////////// ASSEMBLY OF THE DOFS OF THE CURRENT CELLS 
        {
            // CELL DOFS
            auto cbs_loc = cbs; 
            if(is_cut(msh,cl))
                cbs_loc = 2*cbs;
            for (size_t i = 0; i < cbs_loc; i++)
                asm_map.push_back(assembly_index(cell_LHS_offset+i, true));
            // FACES DOFS
            for (size_t face_i = 0; face_i < num_faces; face_i++) {
                auto fc = fcs[face_i];
                auto face_LHS_offset = face_SOL_offset(msh, fc);
                bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
                for (size_t i = 0; i < fbs; i++)
                    asm_map.push_back(assembly_index(face_LHS_offset+i, !dirichlet));
            }
            // ASSEMBLY OF THE FACES IN THE POSITIVE SIDE IF THE CELL IS CUT 
            if(is_cut(msh,cl)) {
                for (size_t face_i = 0; face_i < num_faces; face_i++) {
                    auto fc = fcs[face_i];
                    auto d = (location(msh, fc) == element_location::ON_INTERFACE) ? fbs : 0;
                    auto face_LHS_offset = face_SOL_offset(msh, fc) + d;
                    bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
                    if (dirichlet)
                        std::cout << "Dirichlet boundary on cut cell detected." << std::endl;
                    for (size_t i = 0; i < fbs; i++)
                        asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );
                }
            }
        }

        ///////////////////////////////////////////// ASSEMBLY OF THE PAIRED DOFS IF THE CURRENT CELL IS ILL-CUT 
        ///////////////////////////////////////////// PAIRED CELL = WHICH STABILIZES THE CURRENT CELL 
        if (cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG || cl.user_data.agglo_set == cell_agglo_set::T_KO_POS) {
            // CELL DOFS
            auto cbs_loc = cbs; 
            auto paired_cl = msh.cells[cl.user_data.paired_cell];
            cell_offset = cell_table.at(offset(msh, paired_cl)); 
            cell_LHS_offset = cell_offset*cbs;
            if(is_cut(msh,paired_cl))
                cbs_loc = 2*cbs;
            for (size_t i = 0; i < cbs_loc; i++) 
                asm_map.push_back(assembly_index(cell_LHS_offset+i, true));
            // FACES DOFS
            fcs = faces(msh, paired_cl);
            for (size_t face_i = 0; face_i < num_faces; face_i++) {
                auto fc = fcs[face_i];
                auto face_LHS_offset = face_SOL_offset(msh, fc);
                bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
                for (size_t i = 0; i < fbs; i++)
                    asm_map.push_back(assembly_index(face_LHS_offset+i, !dirichlet));
            }
            // ASSEMBLY OF THE FACES IN THE POSITIVE SIDE IF THE CELL IS CUT 
            if(is_cut(msh,paired_cl)) {
                for (size_t face_i = 0; face_i < num_faces; face_i++) {
                    auto fc = fcs[face_i];
                    auto d = (location(msh, fc) == element_location::ON_INTERFACE) ? fbs : 0;
                    auto face_LHS_offset = face_SOL_offset(msh, fc) + d;
                    bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
                    if (dirichlet)
                        std::cout << "Dirichlet boundary on cut cell detected." << std::endl;
                    for (size_t i = 0; i < fbs; i++)
                        asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );
                }
            }
        }

        ///////////////////////////////////////////// ASSEMBLY OF THE DEPENDEND DOFS 
        // DEPENDENT CELLS = CELLS STABILIZED BY THE CURRENT CELL
        // LOOP OVER DEPENDENT CELLS
        std::set<std::set<size_t>> dp_cell_domains;
        dp_cell_domains.insert(cl.user_data.dependent_cells_neg);
        dp_cell_domains.insert(cl.user_data.dependent_cells_pos);
        for (auto &where : dp_cell_domains) {
            for (auto &dp_cl : where) {
                // CELL DOFS
                auto dp_cell = msh.cells[dp_cl];
                cell_offset = cell_table.at(offset(msh, dp_cell)); 
                cell_LHS_offset = cell_offset*cbs;
                for (size_t i = 0; i < cbs_cut; i++)
                    asm_map.push_back(assembly_index(cell_LHS_offset+i, true));
                // FACES DOFS
                fcs = faces(msh, dp_cell);
                for (size_t face_i = 0; face_i < num_faces; face_i++) {
                    auto fc = fcs[face_i];
                    auto face_LHS_offset = face_SOL_offset(msh, fc);
                    bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
                    for (size_t i = 0; i < fbs; i++)
                        asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet));
                }
                // ASSEMBLY OF THE FACES IN THE POSITIVE SIDE 
                for (size_t face_i = 0; face_i < num_faces; face_i++) {
                    auto fc = fcs[face_i];
                    auto d = (location(msh, fc) == element_location::ON_INTERFACE) ? fbs : 0;
                    auto face_LHS_offset = face_SOL_offset(msh, fc) + d;
                    bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
                    if (dirichlet)
                        std::cout << "Dirichlet boundary on cut cell detected." << std::endl;
                    for (size_t i = 0; i < fbs; i++)
                        asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );
                }
            }
        }

        return asm_map;
    }

    size_t
    n_dof(const Mesh& msh, const typename Mesh::cell_type& cl)
    {
        bool double_unknowns = ( location(msh, cl) == element_location::ON_INTERFACE
                                 && loc_zone == element_location::ON_INTERFACE );

        auto facdeg = di.face_degree();
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto f_dofs = num_faces * fbs;
        auto cbs = loc_cbs;
        auto loc_size = cbs + f_dofs;
        if( double_unknowns )
            loc_size = 2 * loc_size;
        return loc_size;

    }

    Matrix<T, Dynamic, 1>
    get_dirichlet_data(const Mesh& msh, const typename Mesh::cell_type& cl)
    {
        bool double_unknowns = ( location(msh, cl) == element_location::ON_INTERFACE
                                 && loc_zone == element_location::ON_INTERFACE );

        auto facdeg = di.face_degree();
        auto fbs = face_basis<Mesh,T>::size(facdeg);
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto f_dofs = num_faces * fbs;

        auto cbs = loc_cbs;
        auto loc_size = cbs + f_dofs;

        if( double_unknowns )
            loc_size *= 2;

        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero( loc_size );

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_LHS_offset = face_SOL_offset(msh, fc);

            bool in_dom = true;
            if( loc_zone != element_location::ON_INTERFACE );
            {
                element_location loc_fc = location(msh, fc);
                bool in_dom = (loc_fc == element_location::ON_INTERFACE ||
                               loc_fc == loc_zone);
            }

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET && in_dom;

            if( dirichlet && double_unknowns )
                std::cout << "Dirichlet boundary on cut cell detected." << std::endl;
//                throw std::invalid_argument("Dirichlet boundary on cut cell not supported.");

            if (dirichlet && loc_zone == element_location::ON_INTERFACE )
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> loc_rhs = make_rhs(msh, fc, facdeg, dir_func);
                dirichlet_data.block(cbs + face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(loc_rhs);
            }
            if (dirichlet && loc_zone != element_location::ON_INTERFACE )
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg, loc_zone);
                Matrix<T, Dynamic, 1> loc_rhs = make_rhs(msh, fc, facdeg, loc_zone, dir_func);
                dirichlet_data.block(cbs + face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(loc_rhs);
            }
        }

        return dirichlet_data;
    }

    Matrix<T, Dynamic, 1>
    get_dirichlet_data_extended(const Mesh& msh, const typename Mesh::cell_type& cl) {

        // DOFS
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto local_dofs = cbs + num_faces*fbs;
        if (is_cut(msh,cl))
            local_dofs = 2*local_dofs;

        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero(local_dofs);

        // LOOP OVER FACES
        for (size_t face_i = 0; face_i < num_faces; face_i++) {
            
            auto fc = fcs[face_i];
            auto face_LHS_offset = face_SOL_offset(msh, fc);

            bool in_dom = true;
            if (loc_zone != element_location::ON_INTERFACE) {
                element_location loc_fc = location(msh, fc);
                bool in_dom = (loc_fc == element_location::ON_INTERFACE ||
                               loc_fc == loc_zone);
            }

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET && in_dom;

            if (dirichlet && is_cut(msh,cl))
                std::cout << "Dirichlet boundary on cut cell detected." << std::endl;

            if (dirichlet && loc_zone == element_location::ON_INTERFACE ) {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> loc_rhs = make_rhs(msh, fc, facdeg, dir_func);
                dirichlet_data.block(cbs + face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(loc_rhs);
            }
            if (dirichlet && loc_zone != element_location::ON_INTERFACE ) {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg, loc_zone);
                Matrix<T, Dynamic, 1> loc_rhs = make_rhs(msh, fc, facdeg, loc_zone, dir_func);
                dirichlet_data.block(cbs + face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(loc_rhs);
            }
        }

        return dirichlet_data;
    }

    void
    assemble_bis(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        if( !(location(msh, cl) == loc_zone
              || location(msh, cl) == element_location::ON_INTERFACE
              || loc_zone == element_location::ON_INTERFACE ) )
            return;
        
        auto asm_map = init_asm_map(msh, cl);
        auto dirichlet_data = get_dirichlet_data(msh, cl);

        assert( asm_map.size() == lhs.rows() && asm_map.size() == lhs.cols() );

        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs(i,j)) );
                else
                    RHS[asm_map[i]] -= lhs(i,j)*dirichlet_data(j);
            }
        }
        // RHS
        for (size_t i = 0; i < rhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            RHS[asm_map[i]] += rhs(i);
        }
    }
         
    void
    assemble_bis_extended(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        if( !(location(msh, cl) == loc_zone
              || location(msh, cl) == element_location::ON_INTERFACE
              || loc_zone == element_location::ON_INTERFACE ) )
            return;
        
        auto asm_map = init_asm_map_extended(msh, cl);
        auto dirichlet_data = get_dirichlet_data_extended(msh, cl);
        assert(asm_map.size() == lhs.rows() && asm_map.size() == lhs.cols());

        // ASSEMBLY OF STIFFNESS MATRIX
        for (size_t i = 0; i < lhs.rows(); i++) {
            if (!asm_map[i].assemble())
                continue;
            for (size_t j = 0; j < lhs.cols(); j++) {
                if (asm_map[j].assemble())
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs(i,j)) );
                else {
	                int itmp=asm_map[i];
                    RHS(itmp) -= lhs(i,j)*dirichlet_data(j);
		        }
            }
        }
        
        // ASSEMBLY OF THE RHS
        for (size_t i = 0; i < rhs.rows(); i++) {
            if (!asm_map[i].assemble())
                continue;
            int itmp   = asm_map[i]; 
            RHS(itmp) += rhs(i);       
        }
    }
                  
    void
    assemble_rhs_bis(const Mesh& msh, const typename Mesh::cell_type& cl, const Matrix<T, Dynamic, 1>& rhs)
    {
        if( !(location(msh, cl) == loc_zone
              || location(msh, cl) == element_location::ON_INTERFACE
              || loc_zone == element_location::ON_INTERFACE ) )
            return;

        auto asm_map = init_asm_map(msh, cl);
        // RHS
        for (size_t i = 0; i < rhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            RHS[asm_map[i]] += rhs(i);
        }
    }

    void
    assemble_bis_mass(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const Matrix<T, Dynamic, Dynamic>& mass)
    {
        if( !(location(msh, cl) == loc_zone
              || location(msh, cl) == element_location::ON_INTERFACE
              || loc_zone == element_location::ON_INTERFACE ) )
            return;

        auto asm_map = init_asm_map(msh, cl);
        auto dirichlet_data = get_dirichlet_data(msh, cl);

        assert( asm_map.size() == mass.rows() && asm_map.size() == mass.cols() );

        // MASS
        for (size_t i = 0; i < mass.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < mass.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets_mass.push_back( Triplet<T>(asm_map[i], asm_map[j], mass(i,j)) );
            }
        }
    }
            
    Matrix<T, Dynamic, 1>
    get_solF(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, 1>& solution) {

        bool double_unknowns = ( location(msh, cl) == element_location::ON_INTERFACE
                                 && loc_zone == element_location::ON_INTERFACE );

        auto facdeg = di.face_degree();
        auto fbs = face_basis<Mesh,T>::size(facdeg);
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces*fbs;
        
        if( double_unknowns )
            f_dofs = 2 * f_dofs;

        Matrix<T, Dynamic, 1> solF = Matrix<T, Dynamic, 1>::Zero( f_dofs );

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            if( loc_zone != element_location::ON_INTERFACE )
            {
                auto loc_fc = location(msh, fc);
                if( !(loc_fc == element_location::ON_INTERFACE || loc_fc == loc_zone) )
                    continue;
            }

            auto face_LHS_offset = face_SOL_offset(msh, fc);
            if ( location(msh, fc) == element_location::ON_INTERFACE
                 && loc_zone == element_location::ON_INTERFACE )
            {
                // we assume that there is not boundary condition on cut cells (for interface pb)
                solF.block(face_i*fbs, 0, fbs, 1) = solution.block(face_LHS_offset, 0, fbs, 1);
                solF.block( (num_faces+face_i)*fbs, 0, fbs, 1)
                    = solution.block(face_LHS_offset + fbs, 0, fbs, 1);
                continue;
            }

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, facdeg, dir_func);
                solF.block(face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs);
                continue;
            }
            if( location(msh, cl) == element_location::ON_INTERFACE &&
                location(msh, fc) == element_location::IN_POSITIVE_SIDE &&
                loc_zone == element_location::ON_INTERFACE )
            {
                solF.block((num_faces+face_i)*fbs, 0, fbs, 1)
                    = solution.block(face_LHS_offset, 0, fbs, 1);
                continue;
            }
            //else
            solF.block(face_i*fbs, 0, fbs, 1) = solution.block(face_LHS_offset, 0, fbs, 1);
        }
        return solF;
    }

    Matrix<T, Dynamic, 1>
    get_solF_extended(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, 1>& solution) {

        bool double_unknowns = ( location(msh, cl) == element_location::ON_INTERFACE
                                 && loc_zone == element_location::ON_INTERFACE );

        auto facdeg = di.face_degree();
        auto fbs = face_basis<Mesh,T>::size(facdeg);
        std::vector< typename Mesh::face_type > fcs;
        size_t num_faces;

        if (location(msh, cl) == element_location::ON_INTERFACE) {
            if (cl.user_data.agglo_set == cell_agglo_set::T_OK) {
                // Adding the faces of the dependent terms
                fcs = faces(msh, cl);
                size_t num_faces_neg = 0;
                auto nb_dp_cl_neg = cl.user_data.dependent_cells_neg.size();
                auto dependent_cells_neg = cl.user_data.dependent_cells_neg;
                for (auto& dp_cl : dependent_cells_neg) {
                    auto dp_cell = msh.cells[dp_cl];
                    auto fcs_dp = faces(msh, dp_cell);
                    fcs.insert(fcs.end(), fcs_dp.begin(), fcs_dp.end());
                    num_faces_neg++;
                }
                size_t num_faces_pos = 0;
                auto nb_dp_cl_pos = cl.user_data.dependent_cells_pos.size(); // Number of dependent cells 
                auto dependent_cells_pos = cl.user_data.dependent_cells_pos;
                for (auto& dp_cl : dependent_cells_pos) {
                    auto dp_cell = msh.cells[dp_cl];
                    auto fcs_dp = faces(msh, dp_cell);
                    fcs.insert(fcs.end(), fcs_dp.begin(), fcs_dp.end());
                    num_faces_pos++;
                }
                num_faces = fcs.size();
            }
            if (cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG) {
                // Adding the faces of the dependent terms
                fcs = faces(msh, cl);
                auto nb_dp_cl = cl.user_data.dependent_cells_pos.size(); // Number of dependent cells 
                auto dependent_cells = cl.user_data.dependent_cells_pos;
                for (auto& dp_cl : dependent_cells) {
                    auto dp_cell = msh.cells[dp_cl];
                    auto fcs_dp = faces(msh, dp_cell);
                    fcs.insert(fcs.end(), fcs_dp.begin(), fcs_dp.end());
                }
                num_faces = fcs.size();
            }
            if (cl.user_data.agglo_set == cell_agglo_set::T_KO_POS) {
                // Adding the faces of the dependent terms
                fcs = faces(msh, cl);
                auto nb_dp_cl = cl.user_data.dependent_cells_neg.size(); // Number of dependent cells 
                auto dependent_cells = cl.user_data.dependent_cells_neg;
                for (auto& dp_cl : dependent_cells) {
                    auto dp_cell = msh.cells[dp_cl];
                    auto fcs_dp = faces(msh, dp_cell);
                    fcs.insert(fcs.end(), fcs_dp.begin(), fcs_dp.end());
                }
                num_faces = fcs.size();
            }
        }
        else {
            // Adding the faces of the dependent terms
            fcs = faces(msh, cl);
            auto nb_dp_cl = cl.user_data.dependent_cells_neg.size();
            auto dependent_cells = cl.user_data.dependent_cells_neg;
            if (cl.user_data.location == element_location::IN_POSITIVE_SIDE) {
                nb_dp_cl = cl.user_data.dependent_cells_pos.size(); // Number of dependent cells 
                dependent_cells = cl.user_data.dependent_cells_pos;
            }
            for (auto& dp_cl : dependent_cells) {
                auto dp_cell = msh.cells[dp_cl];
                auto fcs_dp = faces(msh, dp_cell);
                auto ns_dp  = normals(msh, dp_cell);
                fcs.insert(fcs.end(), fcs_dp.begin(), fcs_dp.end());
            }
            num_faces = fcs.size();
        }
        size_t f_dofs = num_faces*fbs;
        
        // if( double_unknowns )
        //     f_dofs = 2 * f_dofs;

        Matrix<T, Dynamic, 1> solF = Matrix<T, Dynamic, 1>::Zero( f_dofs );

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            if( loc_zone != element_location::ON_INTERFACE )
            {
                auto loc_fc = location(msh, fc);
                if( !(loc_fc == element_location::ON_INTERFACE || loc_fc == loc_zone) )
                    continue;
            }

            auto face_LHS_offset = face_SOL_offset(msh, fc);
            if ( location(msh, fc) == element_location::ON_INTERFACE
                 && loc_zone == element_location::ON_INTERFACE )
            {
                // we assume that there is not boundary condition on cut cells (for interface pb)
                solF.block(face_i*fbs, 0, fbs, 1) = solution.block(face_LHS_offset, 0, fbs, 1);
                solF.block( (num_faces+face_i)*fbs, 0, fbs, 1)
                    = solution.block(face_LHS_offset + fbs, 0, fbs, 1);
                continue;
            }

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, facdeg, dir_func);
                solF.block(face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs);
                continue;
            }
            if( location(msh, cl) == element_location::ON_INTERFACE &&
                location(msh, fc) == element_location::IN_POSITIVE_SIDE &&
                loc_zone == element_location::ON_INTERFACE )
            {
                solF.block((num_faces+face_i)*fbs, 0, fbs, 1)
                    = solution.block(face_LHS_offset, 0, fbs, 1);
                continue;
            }
            //else
            solF.block(face_i*fbs, 0, fbs, 1) = solution.block(face_LHS_offset, 0, fbs, 1);
        }
        return solF;
    }

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
        MASS.setFromTriplets( triplets_mass.begin(), triplets_mass.end() );
        triplets_mass.clear();
    }
};


//////////////////////////////   FICTITIOUS DOMAIN ASSEMBLERS   /////////////////////////////
template<typename Mesh, typename Function>
class virt_fict_assembler : public virt_scalar_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;

public:

    virt_fict_assembler(const Mesh& msh, const Function& dirichlet_bf,
                        hho_degree_info hdi, element_location where)
        : virt_scalar_assembler<Mesh, Function>(msh, dirichlet_bf, hdi)
    {
        if( where != element_location::IN_NEGATIVE_SIDE
            && where != element_location::IN_POSITIVE_SIDE )
            throw std::invalid_argument("Choose the location in NEGATIVE/POSITIVE side.");
        this->loc_zone = where;

        auto is_removed = [&](const typename Mesh::face_type& fc) -> bool {
            bool is_dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
            auto loc = location(msh,fc);
            bool is_where = (loc == where || loc == element_location::ON_INTERFACE);
            return is_dirichlet || (!is_where);
        };

        auto num_all_faces = msh.faces.size();
        auto num_removed_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_removed);
        this->num_other_faces = num_all_faces - num_removed_faces;

        this->face_table.resize( num_all_faces );

        size_t compressed_offset = 0;
        for (size_t i = 0; i < num_all_faces; i++)
        {
            auto fc = msh.faces[i];
            if ( !is_removed(fc) )
            {
                this->face_table.at(i) = compressed_offset;
                compressed_offset++;
            }
        }

        this->cell_table.resize( msh.cells.size() );
        compressed_offset = 0;
        for (size_t i = 0; i < msh.cells.size(); i++)
        {
            auto cl = msh.cells[i];
            if (location(msh, cl) == where || location(msh, cl) == element_location::ON_INTERFACE)
            {
                this->cell_table.at(i) = compressed_offset;
                compressed_offset++;
            }
        }
        this->num_cells = compressed_offset;
    }
};

/////////////////////////////////////////
template<typename Mesh, typename Function>
class fict_assembler : public virt_fict_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;

public:
    fict_assembler(const Mesh& msh, const Function& dirichlet_bf,
                    hho_degree_info hdi, element_location where)
        : virt_fict_assembler<Mesh, Function>(msh, dirichlet_bf, hdi, where)
    {
        this->loc_cbs = cell_basis<Mesh,T>::size( this->di.cell_degree() );

        auto fbs = face_basis<Mesh,T>::size( this->di.face_degree() );
        auto system_size = this->loc_cbs * this->num_cells + fbs * this->num_other_faces;

        this->LHS = SparseMatrix<T>( system_size, system_size );
        this->RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        this->assemble_bis(msh, cl, lhs, rhs);
    }

    //// take_local_data
    Matrix<T, Dynamic, 1>
    take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution)
    {
        auto loc_cl = location(msh, cl);
        if( !(loc_cl == element_location::ON_INTERFACE || loc_cl == this->loc_zone) )
            throw std::logic_error("Bad cell !!");

        auto solF = this->get_solF(msh, cl, solution);

        auto cbs = this->loc_cbs;

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero( cbs + solF.size() );

        auto cell_offset = this->cell_table.at( offset(msh, cl) );
        auto cell_SOL_offset    = cell_offset * cbs;

        ret.head( cbs ) = solution.block(cell_SOL_offset, 0, cbs, 1);
        ret.tail( solF.size() ) = solF;

        return ret;
    }
};


template<typename Mesh, typename Function>
auto make_fict_assembler(const Mesh& msh, const Function dirichlet_bf,
                          hho_degree_info hdi, element_location where)
{
    return fict_assembler<Mesh, Function>(msh, dirichlet_bf, hdi, where);
}

/////////////////////////////////////////
template<typename Mesh, typename Function>
class fict_condensed_assembler : public virt_fict_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;

    std::vector< Matrix<T, Dynamic, Dynamic> > loc_LHS;
    std::vector< Matrix<T, Dynamic, 1> >       loc_RHS;

public:
    fict_condensed_assembler(const Mesh& msh, const Function& dirichlet_bf,
                              hho_degree_info hdi, element_location where)
        : virt_fict_assembler<Mesh, Function>(msh, dirichlet_bf, hdi, where)
    {
        this->loc_cbs = 0;

        auto fbs = face_basis<Mesh,T>::size( this->di.face_degree() );
        auto system_size = fbs * this->num_other_faces;

        this->LHS = SparseMatrix<T>( system_size, system_size );
        this->RHS = Matrix<T, Dynamic, 1>::Zero( system_size );

        loc_LHS.resize( this->num_cells );
        loc_RHS.resize( this->num_cells );
    }

    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        if( !(location(msh, cl) == this->loc_zone ||
              location(msh, cl) == element_location::ON_INTERFACE) )
            return;

        // save local matrices
        size_t cell_offset = this->cell_table.at( offset(msh, cl) );
        loc_LHS.at( cell_offset ) = lhs;
        loc_RHS.at( cell_offset ) = rhs;

        auto cbs = cell_basis<Mesh,T>::size( this->di.cell_degree() );
        auto fbs = face_basis<Mesh,T>::size( this->di.face_degree() );
        auto num_faces = faces(msh, cl).size();
        size_t f_dofs = num_faces * fbs;

        // static condensation
        auto mat_sc = static_condensation_compute(lhs, rhs, cbs, f_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_sc = mat_sc.first;
        Matrix<T, Dynamic, 1> rhs_sc = mat_sc.second;

        this->assemble_bis(msh, cl, lhs_sc, rhs_sc);
    }

    //// take_local_data
    Matrix<T, Dynamic, 1>
    take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution)
    {
        auto loc_cl = location(msh, cl);
        if( !(loc_cl == element_location::ON_INTERFACE || loc_cl == this->loc_zone) )
            throw std::logic_error("Bad cell !!");

        auto solF = this->get_solF(msh, cl, solution);

        auto cbs = cell_basis<Mesh,T>::size( this->di.cell_degree() );
        auto fbs = face_basis<Mesh,T>::size( this->di.face_degree() );
        auto num_faces = faces(msh, cl).size();
        size_t f_dofs = num_faces * fbs;

        size_t offset_cl = this->cell_table.at( offset(msh, cl) );

        return static_condensation_recover(loc_LHS.at(offset_cl), loc_RHS.at(offset_cl), cbs, f_dofs, solF);
    }
};

template<typename Mesh, typename Function>
auto make_fict_condensed_assembler(const Mesh& msh, const Function dirichlet_bf,
                                    hho_degree_info hdi, element_location where)
{
    return fict_condensed_assembler<Mesh, Function>(msh, dirichlet_bf, hdi, where);
}


////////////////////////////////  INTERFACE ASSEMBLERS  /////////////////////////

template<typename Mesh, typename Function>
class virt_interface_assembler : public virt_scalar_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;

public:

    virt_interface_assembler(const Mesh& msh, const Function& dirichlet_bf, hho_degree_info hdi)
        : virt_scalar_assembler<Mesh, Function>(msh, dirichlet_bf, hdi)
    {
        this->loc_zone = element_location::ON_INTERFACE;

        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
        };

        size_t loc_num_cells = 0; /* counts cells with dup. unknowns */
        for (auto& cl : msh.cells)
        {
            this->cell_table.push_back( loc_num_cells );
            if (location(msh, cl) == element_location::ON_INTERFACE)
                loc_num_cells += 2;
            else
                loc_num_cells += 1;
        }
        this->num_cells = loc_num_cells; // 
        assert(this->cell_table.size() == msh.cells.size());

        size_t num_all_faces = 0; /* counts faces with dup. unknowns */
        for (auto& fc : msh.faces)
        {
            if (location(msh, fc) == element_location::ON_INTERFACE)
                num_all_faces += 2;
            else
                num_all_faces += 1;
        }
            
        size_t num_dirichlet_faces = 0; /* counts faces with dup. unknowns */
        for (auto& fc : msh.faces)
        {
            if(fc.is_boundary && fc.bndtype == boundary::DIRICHLET){
                if (location(msh, fc) == element_location::ON_INTERFACE)
                        num_dirichlet_faces += 2;
                    else
                        num_dirichlet_faces += 1;
            }
        }

        /* We assume that cut cells can not have dirichlet faces */
//        auto num_dirichlet_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_dirichlet);
        this->num_other_faces = num_all_faces - num_dirichlet_faces;

        this->face_table.resize( msh.faces.size() );

        size_t compressed_offset = 0;
        for (size_t i = 0; i < msh.faces.size(); i++)
        {
            auto fc = msh.faces.at(i);
            if ( !is_dirichlet(fc) )
            {
                this->face_table.at(i) = compressed_offset;
                if ( location(msh, fc) == element_location::ON_INTERFACE )
                    compressed_offset += 2;
                else
                    compressed_offset += 1;
            }
        }
    }
};


////////////////////////////////////////////////

template<typename Mesh, typename Function>
class interface_assembler : public virt_interface_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;

public:

    interface_assembler(const Mesh& msh, const Function& dirichlet_bf, hho_degree_info hdi)
        : virt_interface_assembler<Mesh, Function>(msh, dirichlet_bf, hdi)
    {
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        this->loc_cbs = cbs;

        auto system_size = cbs * this->num_cells + fbs * this->num_other_faces;

        this->LHS = SparseMatrix<T>( system_size, system_size );
        this->RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        this->assemble_bis(msh, cl, lhs, rhs);
    }


    Matrix<T, Dynamic, 1>
    take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution,
                    element_location where)
    {
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto cell_offset        = offset(msh, cl);
        size_t cell_SOL_offset;
        if ( location(msh, cl) == element_location::ON_INTERFACE )
        {
            if (where == element_location::IN_NEGATIVE_SIDE)
                cell_SOL_offset = this->cell_table.at(cell_offset) * cbs;
            else if (where == element_location::IN_POSITIVE_SIDE)
                cell_SOL_offset = this->cell_table.at(cell_offset) * cbs + cbs;
            else
                throw std::invalid_argument("Invalid location");
        }
        else
        {
            cell_SOL_offset = this->cell_table.at(cell_offset) * cbs;
        }

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + num_faces*fbs);
        ret.block(0, 0, cbs, 1) = solution.block(cell_SOL_offset, 0, cbs, 1);


        auto solF = this->get_solF(msh, cl, solution);
        if(where == element_location::IN_NEGATIVE_SIDE)
            ret.tail(num_faces * fbs) = solF.head(num_faces * fbs);
        else
            ret.tail(num_faces * fbs) = solF.tail(num_faces * fbs);

        return ret;
    }
};
            
template<typename Mesh, typename Function>
auto make_interface_assembler(const Mesh& msh, Function dirichlet_bf, hho_degree_info hdi)
{
    return interface_assembler<Mesh, Function>(msh, dirichlet_bf, hdi);
}

// ELLIPTIC PROBLEM - FIRST ORDER FORMULATION
template<typename Mesh, typename Function>
class one_field_interface_assembler : public virt_interface_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;
    std::vector< size_t > m_elements_with_bc_eges;

public:
    
    one_field_interface_assembler(const Mesh& msh, const Function& dirichlet_bf, hho_degree_info hdi)
        : virt_interface_assembler<Mesh, Function>(msh, dirichlet_bf, hdi)
    {
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        this->loc_cbs = cbs;
        auto system_size = cbs * this->num_cells + fbs * this->num_other_faces;

        this->LHS = SparseMatrix<T>(system_size, system_size);
        this->RHS = Matrix<T, Dynamic, 1>::Zero( system_size);
        this->MASS = SparseMatrix<T>(system_size, system_size);
        
        // std::cout << "LHS.size() = " << this->LHS.size() << std::endl;
        // std::cout << "RHS.size() = " << this->RHS.size() << std::endl;
        // std::cout << "MASS.size() = " << this->MASS.size() << std::endl;

        classify_cells(msh);
    }

    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        this->assemble_bis(msh, cl, lhs, rhs);
    }

    void
    assemble_extended(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        this->assemble_bis_extended(msh, cl, lhs, rhs);
    }
           
    void
    assemble_rhs(const Mesh& msh, const typename Mesh::cell_type& cl, const Matrix<T, Dynamic, 1>& rhs)
    {
        this->assemble_rhs_bis(msh, cl, rhs);
    }
            
    void
    assemble_mass(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& mass)
    {
        this->assemble_bis_mass(msh, cl, mass);
    }

    void classify_cells(const Mesh& msh){

        m_elements_with_bc_eges.clear();
        size_t cell_ind = 0;
        for (auto& cell : msh.cells)
        {
            auto face_list = faces(msh, cell);
            for (size_t face_i = 0; face_i < face_list.size(); face_i++)
            {
                auto fc = face_list[face_i];
//                auto fc_id = msh.lookup(fc);
                bool is_dirichlet_Q = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
//                && in_dom;
//                bool is_dirichlet_Q = m_bnd.is_dirichlet_face(fc_id);
                if (is_dirichlet_Q)
                {
                    m_elements_with_bc_eges.push_back(cell_ind);
                    break;
                }
            }
            cell_ind++;
        }
    }

    Matrix<T, Dynamic, 1>
    take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution,
                    element_location where = element_location::UNDEF)
    {
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto cell_offset        = offset(msh, cl);
        size_t cell_SOL_offset;
        if ( location(msh, cl) == element_location::ON_INTERFACE )
        {
            if (where == element_location::IN_NEGATIVE_SIDE)
                cell_SOL_offset = this->cell_table.at(cell_offset) * cbs;
            else if (where == element_location::IN_POSITIVE_SIDE)
                cell_SOL_offset = this->cell_table.at(cell_offset) * cbs + cbs;
            else
                throw std::invalid_argument("Invalid location");
        }
        else
        {
            cell_SOL_offset = this->cell_table.at(cell_offset) * cbs;
        }

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + num_faces*fbs);
        ret.block(0, 0, cbs, 1) = solution.block(cell_SOL_offset, 0, cbs, 1);


        auto solF = this->get_solF(msh, cl, solution);
        if(where == element_location::IN_NEGATIVE_SIDE)
            ret.tail(num_faces * fbs) = solF.head(num_faces * fbs);
        else
            ret.tail(num_faces * fbs) = solF.tail(num_faces * fbs);

        return ret;
    }
            
    Matrix<T, Dynamic, 1>
    take_local_data_extended(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution,
                    element_location where = element_location::UNDEF)
    {
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto cell_offset = offset(msh, cl);
        size_t cell_SOL_offset;
        std::vector< typename Mesh::face_type > fcs;
        size_t num_faces;

        if (location(msh, cl) == element_location::ON_INTERFACE) {
            if (cl.user_data.agglo_set == cell_agglo_set::T_OK) {
                std::cout << "TOK" << std::endl;
                if (where == element_location::IN_NEGATIVE_SIDE)
                    cell_SOL_offset = this->cell_table.at(cell_offset)*cbs;
                else if (where == element_location::IN_POSITIVE_SIDE)
                    cell_SOL_offset = this->cell_table.at(cell_offset)*cbs + cbs;
                else  throw std::invalid_argument("Invalid location");
            }
            if (cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG) {
                std::cout << "TKOneg" << std::endl;
                if (where == element_location::IN_NEGATIVE_SIDE)
                    cell_SOL_offset = this->cell_table.at(cell_offset)*cbs;
                else if (where == element_location::IN_POSITIVE_SIDE)
                    cell_SOL_offset = this->cell_table.at(cell_offset)*cbs + cbs;
                else  throw std::invalid_argument("Invalid location");
            }
            if (cl.user_data.agglo_set == cell_agglo_set::T_KO_POS) {
                std::cout << "TKOpos" << std::endl;
                if (where == element_location::IN_NEGATIVE_SIDE)
                    cell_SOL_offset = this->cell_table.at(cell_offset)*cbs;
                else if (where == element_location::IN_POSITIVE_SIDE)
                    cell_SOL_offset = this->cell_table.at(cell_offset)*cbs + cbs;
                else  throw std::invalid_argument("Invalid location");
            }
        }
        else {
            cell_SOL_offset = this->cell_table.at(cell_offset) * cbs;
            // Adding the faces of the dependent terms
            fcs = faces(msh, cl);
            auto nb_dp_cl = cl.user_data.dependent_cells_neg.size();
            auto dependent_cells = cl.user_data.dependent_cells_neg;
            if (cl.user_data.location == element_location::IN_POSITIVE_SIDE) {
                nb_dp_cl = cl.user_data.dependent_cells_pos.size(); // Number of dependent cells 
                dependent_cells = cl.user_data.dependent_cells_pos;
            }
            for (auto& dp_cl : dependent_cells) {
                auto dp_cell = msh.cells[dp_cl];
                auto fcs_dp = faces(msh, dp_cell);
                auto ns_dp  = normals(msh, dp_cell);
                fcs.insert(fcs.end(), fcs_dp.begin(), fcs_dp.end());
            }
            num_faces = fcs.size();
        }

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + num_faces*fbs);
        ret.block(0, 0, cbs, 1) = solution.block(cell_SOL_offset, 0, cbs, 1);

        // auto solF = this->get_solF_extended(msh, cl, solution);
        // if(where == element_location::IN_NEGATIVE_SIDE) {
        //     ret.tail(num_faces * fbs) = solF.head(num_faces * fbs);
        // }
        // else {
        //     ret.tail(num_faces * fbs) = solF.tail(num_faces * fbs);
        // }

        return ret;
    }
                        
    Matrix<T, Dynamic, 1>
    gather_cell_dof(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution,
                    element_location where)
    {
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto cell_offset        = offset(msh, cl);
        size_t cell_SOL_offset;
        if ( location(msh, cl) == element_location::ON_INTERFACE )
        {
            if (where == element_location::IN_NEGATIVE_SIDE)
                cell_SOL_offset = this->cell_table.at(cell_offset) * cbs;
            else if (where == element_location::IN_POSITIVE_SIDE)
                cell_SOL_offset = this->cell_table.at(cell_offset) * cbs + cbs;
            else
                throw std::invalid_argument("Invalid location");
        }
        else
        {
            cell_SOL_offset = this->cell_table.at(cell_offset) * cbs;
        }
        return solution.block(cell_SOL_offset, 0, cbs, 1);
    }
            
    void project_over_cells(const Mesh& msh, hho_degree_info hho_di, Matrix<T, Dynamic, 1> & x_glob, std::function<T(const typename Mesh::point_type& )> scal_fun){
        
        for (auto& cl : msh.cells)
        {
            if( location(msh, cl) != element_location::ON_INTERFACE ){
                project_over_uncutcells(msh, cl, hho_di, x_glob, scal_fun);
            } else{ // on interface
                project_over_cutcells(msh, cl, hho_di, x_glob, scal_fun);
            }
        }
    }
        
    void project_over_uncutcells(const Mesh& msh, const typename Mesh::cell_type& cl, hho_degree_info hho_di, Matrix<T, Dynamic, 1> & x_glob, std::function<T(const typename Mesh::point_type& )> scal_fun){
            
        Matrix<T, Dynamic, 1> x_proj_dof = project_function(msh, cl, hho_di, scal_fun);
            
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);
        auto cell_offset        = offset(msh, cl);
        size_t cell_SOL_offset = this->cell_table.at(cell_offset) * cbs;
        x_glob.block(cell_SOL_offset, 0, cbs, 1) = x_proj_dof.block(0, 0, cbs, 1);
    }
    
    void project_over_cutcells(const Mesh& msh, const typename Mesh::cell_type& cl, hho_degree_info hho_di, Matrix<T, Dynamic, 1> & x_glob, std::function<T(const typename Mesh::point_type& )> scal_fun){
            
        Matrix<T, Dynamic, 1> x_neg_proj_dof = project_function(msh, cl, hho_di, element_location::IN_NEGATIVE_SIDE, scal_fun);
            
        Matrix<T, Dynamic, 1> x_pos_proj_dof = project_function(msh, cl, hho_di, element_location::IN_POSITIVE_SIDE, scal_fun);
            
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);
        auto cell_offset        = offset(msh, cl);
        size_t cell_SOL_offset = this->cell_table.at(cell_offset) * cbs;
        x_glob.block(cell_SOL_offset, 0, cbs, 1) = x_neg_proj_dof.block(0, 0, cbs, 1);
        x_glob.block(cell_SOL_offset+cbs, 0, cbs, 1) = x_pos_proj_dof.block(0, 0, cbs, 1);
    
    }
    std::vector<std::pair<size_t,size_t>> compute_cell_basis_data(const Mesh& msh){
        size_t n_cells =  msh.cells.size();
        std::vector<std::pair<size_t,size_t>> cell_basis_data;
        cell_basis_data.reserve(n_cells);
        size_t cell_ind = 0;
        for(auto& cl : msh.cells) {
            bool double_unknowns = ( location(msh, cl) == element_location::ON_INTERFACE);
            auto cbs = this->loc_cbs;
            if( double_unknowns ){
                cbs *= 2;
            }
            cell_basis_data.push_back(std::make_pair(cell_ind, cbs));
            cell_ind++;
        }
        
        return cell_basis_data;
    }

    auto compute_dofs_data(Mesh& msh, hho_degree_info hdi){

        size_t nb_cells =  msh.cells.size();

        auto celdeg = hdi.cell_degree();
        auto facdeg = hdi.face_degree();
        auto cbs   = cell_basis<Mesh,T>::size(celdeg);
        auto fbs   = face_basis<Mesh,T>::size(facdeg);

        // Filling the structure cell dofs 
        auto offset_ddl = 0;
        for(size_t i=0; i < nb_cells; i++) {
            auto& cl = msh.cells[i];
            auto num_faces = faces(msh, cl).size();
            auto dofs = 0;
            if (!is_cut(msh, cl)) {
                // std::cout << "UNCUT CELL: " << offset(msh, cl) << std::endl;
                dofs += cbs + num_faces*fbs; // DOFS OF THE CURRENT CELL
                // std::cout << "Dependant cells:   ";
                for (auto &dp_cl : cl.user_data.dependent_cells_neg) {
                    // std::cout << dp_cl << "   ";
                    num_faces = faces(msh, msh.cells[dp_cl]).size();
                    dofs += 2*(cbs + num_faces*fbs); // ADDING DOFS OF BOTH SIDES OF THE DEPENDENT CELLS
                }
                // std::cout << std::endl;
                // std::cout << "Dependant cells:   ";
                for (auto &dp_cl : cl.user_data.dependent_cells_pos) {
                    // std::cout << dp_cl << "   ";
                    num_faces = faces(msh, msh.cells[dp_cl]).size();
                    dofs += 2*(cbs + num_faces*fbs); // ADDING DOFS OF BOTH SIDES OF THE DEPENDENT CELLS
                }
                cl.user_data.local_dofs = dofs;
                // std::cout << std::endl;
            }
            else {
                // std::cout << "CUT CELL: " << offset(msh, cl) << std::endl;    
                dofs += 2*(cbs + num_faces*fbs); // DOFS OF THE CURRENT CELL
                // std::cout << "Negative dependant cells:   ";
                for (auto &dp_cl : cl.user_data.dependent_cells_neg) {
                    // std::cout << dp_cl << "   ";
                    num_faces = faces(msh, msh.cells[dp_cl]).size();
                    dofs += 2*(cbs + num_faces*fbs); // ADDING DOFS OF BOTH SIDES OF THE DEPENDENT CELLS
                }
                // std::cout << std::endl;
                // std::cout << "Positive dependant cells:   ";
                for (auto &dp_cl : cl.user_data.dependent_cells_pos) {
                    // std::cout << dp_cl << "   ";
                    num_faces = faces(msh, msh.cells[dp_cl]).size();
                    dofs += 2*(cbs + num_faces*fbs); // ADDING DOFS OF BOTH SIDES OF THE DEPENDENT CELLS
                }
                // std::cout << std::endl;
                if (cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG || cl.user_data.agglo_set == cell_agglo_set::T_KO_POS) {
                    num_faces = faces(msh, msh.cells[cl.user_data.paired_cell]).size();
                    if (!is_cut(msh,msh.cells[cl.user_data.paired_cell])) {
                        dofs += cbs + num_faces*fbs; // ADING THE DOFS OF THE PAIRED CELL FOR THE COMPUTATION OF JUMP IN THE CURRENT CELL
                    }
                    else {
                        dofs += 2*(cbs + num_faces*fbs); // ADING THE DOFS OF THE PAIRED CELL FOR THE COMPUTATION OF JUMP IN THE CURRENT CELL
                    }
                }
                cl.user_data.local_dofs = dofs;
            }
            // std::cout << "local dofs = " << cl.user_data.local_dofs << std::endl << std::endl;
        }
        // Vérif n_dofs
        auto n_dofs = 0;
        auto verif_dofs = 0;
        auto cp_dp = 0;
        for(size_t i=0; i < nb_cells; i++) {
            auto cl = msh.cells[i];
            auto celdeg = hdi.cell_degree();
            auto facdeg = hdi.face_degree();
            auto cbs   = cell_basis<Mesh,T>::size(celdeg);
            auto fbs   = face_basis<Mesh,T>::size(facdeg);
            auto local_dofs = 0;
            if (!is_cut(msh, cl)) {
                local_dofs = cbs;
            }
            else {
                local_dofs = 2*cbs;
            }
            verif_dofs += local_dofs;
            n_dofs += cl.user_data.local_dofs;
            cp_dp += cl.user_data.dependent_cells_neg.size() + cl.user_data.dependent_cells_pos.size();
        }
        // std::cout << "verif_dofs = " << verif_dofs << std::endl;
        // std::cout << "n_dofs = "     << n_dofs     << std::endl;
        // std::cout << "minus  = " << n_dofs-cp_dp*3*(cbs+4*fbs) << std::endl;
        // std::cout << "COMPUTE_DOFS_DATA OK" << std::endl;

        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
        };

        for(size_t i=0; i < msh.faces.size(); i++) {
            auto fbs = face_basis<Mesh,T>::size(facdeg);
            if (is_dirichlet(msh.faces[i])) {
                continue;
            }
            else if (!is_cut(msh, msh.faces[i])) {
                verif_dofs += fbs;
            } 
            else {
                verif_dofs += 2*fbs;
            }


        }

        return verif_dofs;
    }


    Matrix<T, Dynamic, 1> make_projection_operator(const Mesh& msh, hho_degree_info hho_di, size_t system_size, std::function<T(const typename Mesh::point_type& )> scal_fun){
        
        Matrix<T, Dynamic, 1> x_glob = Matrix<T, Dynamic, 1>::Zero(system_size, 1);
 
        // Loop on uncut, cut cells, and good side of ill-cut cells 
        for (const auto& cl : msh.cells) {
            auto celdeg = this->di.cell_degree();
            auto facdeg = this->di.face_degree();
            auto cbs = cell_basis<Mesh,T>::size(celdeg);
            auto fbs = face_basis<Mesh,T>::size(facdeg);
            auto offset_cl = offset(msh,cl);
            auto dofs = cl.user_data.local_dofs;
            // std::cout << "cell = " << offset(msh, cl) << std::endl;
            // std::cout << "offset = " << offset_cl << std::endl;
            // std::cout << "dofs = " << dofs << std::endl << std::endl;
            // if (!is_cut(msh, cl) && cl.user_data.location == element_location::IN_NEGATIVE_SIDE) {    
            //     Matrix<T, Dynamic, 1> x_proj_dof = project_function_uncut(msh, cl, hho_di, element_location::IN_NEGATIVE_SIDE, scal_fun);
            //     x_glob.block(offset_cl, 0, dofs, 1) = x_proj_dof.block(0, 0, dofs, 1);
            //     // std::cout << x_glob.block(offset_cl, 0, dofs, 1) << std::endl << std::endl;
            // }
            // else if (!is_cut(msh, cl) && cl.user_data.location == element_location::IN_POSITIVE_SIDE) {   
            //     Matrix<T, Dynamic, 1> x_proj_dof = project_function_uncut(msh, cl, hho_di, element_location::IN_POSITIVE_SIDE, scal_fun);
            //     x_glob.block(offset_cl, 0, dofs, 1) = x_proj_dof.block(0, 0, dofs, 1);
            //     // std::cout << x_glob.block(offset_cl, 0, dofs, 1) << std::endl << std::endl;
            // }
            // else if (cl.user_data.agglo_set == cell_agglo_set::T_OK) {     
            //     Matrix<T, Dynamic, 1> x_proj_dof = project_function_TOK(msh, cl, hho_di, scal_fun);
            //     x_glob.block(offset_cl, 0, dofs, 1) = x_proj_dof.block(0, 0, dofs, 1);
            //     // std::cout << x_glob.block(offset_cl, 0, dofs, 1) << std::endl << std::endl;
            // }
            // else {
            //     if (cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG) { 
            //         Matrix<T, Dynamic, 1> x_proj_dof = project_function_TKOibar(msh, cl, hho_di, element_location::IN_POSITIVE_SIDE, scal_fun);
            //         x_glob.block(offset_cl, 0, dofs, 1) = x_proj_dof.block(0, 0, dofs, 1);
            //         // std::cout << x_glob.block(offset_cl, 0, dofs, 1) << std::endl << std::endl;
            //     }
            //     else { 
            //         Matrix<T, Dynamic, 1> x_proj_dof = project_function_TKOibar(msh, cl, hho_di, element_location::IN_NEGATIVE_SIDE, scal_fun);
            //         x_glob.block(offset_cl, 0, dofs, 1) = x_proj_dof.block(0, 0, dofs, 1);
            //         // std::cout << x_glob.block(offset_cl, 0, dofs, 1) << std::endl << std::endl;
            //     }
            // }
        }

        // std::cout << std::endl << "System size = " << x_glob.size() << std::endl;
        // std::cout << "MAKE_PROJECTION_OPERATOR ok" << std::endl;

        return x_glob;
    }

};

template<typename Mesh, typename Function>
auto make_one_field_interface_assembler(const Mesh& msh, Function dirichlet_bf, hho_degree_info hdi)
{
    return one_field_interface_assembler<Mesh, Function>(msh, dirichlet_bf, hdi);
}

// POLYNOMIAL EXTENSION ASSEMBLER - ELLIPTIC PROBLEM - MIXED FORMULATION
template<typename Mesh, typename Function>
class two_fields_interface_assembler : public virt_interface_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;
    std::vector< size_t > m_elements_with_bc_eges;
public:
            


    two_fields_interface_assembler(const Mesh& msh, const Function& dirichlet_bf, hho_degree_info hdi)
        : virt_interface_assembler<Mesh, Function>(msh, dirichlet_bf, hdi)
    {
        auto celdeg = this->di.cell_degree();
        auto graddeg = this->di.grad_degree();
        auto facdeg = this->di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto rbs = vector_cell_basis<Mesh,T>::size(graddeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        this->loc_cbs = (cbs+rbs);
        auto system_size = (cbs+rbs) * this->num_cells + fbs * this->num_other_faces;

        this->LHS = SparseMatrix<T>( system_size, system_size );
        this->RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
        this->MASS = SparseMatrix<T>( system_size, system_size );
        
        classify_cells(msh);
    }

    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        this->assemble_bis(msh, cl, lhs, rhs);
    }
            
    void
    assemble_rhs(const Mesh& msh, const typename Mesh::cell_type& cl, const Matrix<T, Dynamic, 1>& rhs)
    {
        this->assemble_rhs_bis(msh, cl, rhs);
    }
            
    void
    assemble_mass(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& mass)
    {
        this->assemble_bis_mass(msh, cl, mass);
    }

    void classify_cells(const Mesh& msh){

        m_elements_with_bc_eges.clear();
        size_t cell_ind = 0;
        for (auto& cell : msh.cells)
        {
            auto face_list = faces(msh, cell);
            for (size_t face_i = 0; face_i < face_list.size(); face_i++)
            {
                auto fc = face_list[face_i];
                bool is_dirichlet_Q = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
                if (is_dirichlet_Q)
                {
                    m_elements_with_bc_eges.push_back(cell_ind);
                    break;
                }
            }
            cell_ind++;
        }
    }

    Matrix<T, Dynamic, 1>
    take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution,
                    element_location where)
    {
        auto celdeg = this->di.cell_degree();
        auto graddeg = this->di.grad_degree();
        auto facdeg = this->di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto rbs = vector_cell_basis<Mesh,T>::size(graddeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto cell_offset        = offset(msh, cl);
        size_t cell_SOL_offset;
        if ( location(msh, cl) == element_location::ON_INTERFACE )
        {
            if (where == element_location::IN_NEGATIVE_SIDE)
                cell_SOL_offset = this->cell_table.at(cell_offset) * (cbs+rbs);
            else if (where == element_location::IN_POSITIVE_SIDE)
                cell_SOL_offset = this->cell_table.at(cell_offset) * (cbs+rbs) + (cbs+rbs);
            else
                throw std::invalid_argument("Invalid location");
        }
        else
        {
            cell_SOL_offset = this->cell_table.at(cell_offset) * (cbs+rbs);
        }

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero((cbs+rbs) + num_faces*fbs);
        ret.block(0, 0, (cbs+rbs), 1) = solution.block(cell_SOL_offset, 0, (cbs+rbs), 1);


        auto solF = this->get_solF(msh, cl, solution);
        if(where == element_location::IN_NEGATIVE_SIDE)
            ret.tail(num_faces * fbs) = solF.head(num_faces * fbs);
        else
            ret.tail(num_faces * fbs) = solF.tail(num_faces * fbs);

        return ret;
    }
            
    Matrix<T, Dynamic, 1>
    gather_cell_dof(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution,
                    element_location where)
    {
        auto celdeg = this->di.cell_degree();
        auto graddeg = this->di.grad_degree();
        auto facdeg = this->di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto rbs = vector_cell_basis<Mesh,T>::size(graddeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto cell_offset        = offset(msh, cl);
        size_t flux_cell_SOL_offset;
        size_t cell_SOL_offset;
        if ( location(msh, cl) == element_location::ON_INTERFACE )
        {
            if (where == element_location::IN_NEGATIVE_SIDE){
                flux_cell_SOL_offset = this->cell_table.at(cell_offset) * (cbs+rbs);
                cell_SOL_offset = this->cell_table.at(cell_offset) * (cbs+rbs) + 2*rbs;
            }
            else if (where == element_location::IN_POSITIVE_SIDE){
                flux_cell_SOL_offset = this->cell_table.at(cell_offset) * (cbs+rbs)+rbs;
                cell_SOL_offset = this->cell_table.at(cell_offset) * (cbs+rbs) + 2*rbs + cbs;
            }
            else
                throw std::invalid_argument("Invalid location");
        }
        else
        {
            flux_cell_SOL_offset = this->cell_table.at(cell_offset) * (cbs+rbs);
            cell_SOL_offset = this->cell_table.at(cell_offset) * (cbs+rbs)+rbs;
        }
        Matrix<T, Dynamic, 1> dof = Matrix<T, Dynamic, 1>::Zero(rbs+cbs,1);
        dof.block(0,0,rbs,1) = solution.block(flux_cell_SOL_offset, 0, rbs, 1);
        dof.block(rbs,0,cbs,1) = solution.block(cell_SOL_offset, 0, cbs, 1);
        return dof;
    }
    
    Matrix<T, Dynamic, 1> project_vec_function(const Mesh& msh, const typename Mesh::cell_type& cell, hho_degree_info hho_di,
                      std::function<Matrix<T, 1, 2>(const typename Mesh::point_type& )> vec_fun, element_location where = element_location::UNDEF){
    
            auto gradeg = hho_di.grad_degree();
            vector_cell_basis<Mesh, T> vec_cell_basis(msh, cell, hho_di.grad_degree());
            auto gbs = vec_cell_basis.size();
        
            Matrix<T, Dynamic, Dynamic> mass;
            Matrix<T, Dynamic, 1> rhs = Matrix<T, Dynamic, 1>::Zero(gbs);
            
            if(element_location::UNDEF == where){
                mass = make_vec_mass_matrix(msh, cell, hho_di);
                const auto qps = integrate(msh, cell, 2*gradeg);
                for (auto& qp : qps)
                {
                  auto t_phi = vec_cell_basis.eval_basis(qp.first);
                  Matrix<T, 1, 2> f_vec = vec_fun(qp.first);
                  for (size_t i = 0; i < gbs; i++){
                  Matrix<T, 2, 1> phi_i = t_phi.block(i, 0, 1, 2).transpose();
                      rhs(i,0) = rhs(i,0) + (qp.second * f_vec*phi_i)(0,0);
                  }
                }
            }else{
                mass = make_vec_mass_matrix(msh, cell, hho_di, where);
                const auto qps = integrate(msh, cell, 2*gradeg, where);
                for (auto& qp : qps)
                {
                  auto t_phi = vec_cell_basis.eval_basis(qp.first);
                  Matrix<T, 1, 2> f_vec = vec_fun(qp.first);
                  for (size_t i = 0; i < gbs; i++){
                  Matrix<T, 2, 1> phi_i = t_phi.block(i, 0, 1, 2).transpose();
                      rhs(i,0) = rhs(i,0) + (qp.second * f_vec*phi_i)(0,0);
                  }
                }
                
//                {
//                    vector_cell_basis<Mesh, T> vec_cell_basis(msh, cell, hho_di.grad_degree());
//                    Matrix<T, Dynamic, 1> x_g_proj_dof = mass.llt().solve(rhs);
//                    const auto qps = integrate(msh, cell, 2*gradeg, where);
//                    for (auto& qp : qps)
//                    {
//
//                        Matrix<T, Dynamic, 1> vec_cell_dof = x_g_proj_dof;
//                        auto t_phi_v = vec_cell_basis.eval_basis( qp.first );
//                        Matrix<T, 1, 2> grad_uh = Matrix<T, 1, 2>::Zero();
//                        for (size_t i = 0; i < t_phi_v.rows(); i++){
//                            grad_uh = grad_uh + vec_cell_dof(i)*t_phi_v.block(i, 0, 1, 2);
//                        }
//                        Matrix<T, 1, 2> f_vec = vec_fun(qp.first);
//                        int aka=0;
//                    }
//                }
                
            }
    
            Matrix<T, Dynamic, 1> x_dof = mass.llt().solve(rhs);
            return x_dof;
    }
        
    void project_over_cells(const Mesh& msh, hho_degree_info hho_di, Matrix<T, Dynamic, 1> & x_glob, std::function<T(const typename Mesh::point_type& )> scal_fun, std::function<Matrix<T, 1, 2>(const typename Mesh::point_type& )> flux_fun){
        size_t n_dof = this->MASS.rows();
        x_glob = Matrix<T, Dynamic, 1>::Zero(n_dof);

        for (auto& cl : msh.cells)
        {
            if( location(msh, cl) != element_location::ON_INTERFACE ){
                project_over_uncutcells(msh, cl, hho_di, x_glob, scal_fun, flux_fun);
            } else{ // on interface
                project_over_cutcells(msh, cl, hho_di, x_glob, scal_fun, flux_fun);
            }
        }
    }
    

            
    void project_over_uncutcells(const Mesh& msh, const typename Mesh::cell_type& cl, hho_degree_info hho_di, Matrix<T, Dynamic, 1> & x_glob, std::function<T(const typename Mesh::point_type& )> scal_fun, std::function<Matrix<T, 1, 2>(const typename Mesh::point_type& )> flux_fun){
            
        Matrix<T, Dynamic, 1> x_g_proj_dof = project_vec_function(msh, cl, hho_di, flux_fun);
        Matrix<T, Dynamic, 1> x_c_proj_dof = project_function(msh, cl, hho_di, scal_fun);
        
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();
        auto gradeg = this->di.grad_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);
        auto gbs = vector_cell_basis<Mesh,T>::size(gradeg);
        auto cell_offset        = offset(msh, cl);
        size_t cell_SOL_offset = this->cell_table.at(cell_offset) * (cbs+gbs);
        
        x_glob.block(cell_SOL_offset, 0, gbs, 1) = x_g_proj_dof;
        x_glob.block(cell_SOL_offset+gbs, 0, cbs, 1) = x_c_proj_dof.block(0, 0, cbs, 1);
        
//        vector_cell_basis<Mesh, T> vec_cell_basis(msh, cl, hho_di.grad_degree());
//        const auto qps = integrate(msh, cl, 2*gradeg);
//        for (auto& qp : qps)
//        {
//
//            Matrix<T, Dynamic, 1> vec_cell_dof = x_g_proj_dof;
//            auto t_phi_v = vec_cell_basis.eval_basis( qp.first );
//            Matrix<T, 1, 2> grad_uh = Matrix<T, 1, 2>::Zero();
//            for (size_t i = 0; i < t_phi_v.rows(); i++){
//                grad_uh = grad_uh + vec_cell_dof(i)*t_phi_v.block(i, 0, 1, 2);
//            }
//            Matrix<T, 1, 2> f_vec = flux_fun(qp.first);
//            int aka=0;
//        }

        
    }
            
    void project_over_cutcells(const Mesh& msh, const typename Mesh::cell_type& cl, hho_degree_info hho_di, Matrix<T, Dynamic, 1> & x_glob, std::function<T(const typename Mesh::point_type& )> scal_fun, std::function<Matrix<T, 1, 2>(const typename Mesh::point_type& )> flux_fun){
            
        Matrix<T, Dynamic, 1> x_neg_g_proj_dof = project_vec_function(msh, cl, hho_di, flux_fun, element_location::IN_NEGATIVE_SIDE);
        Matrix<T, Dynamic, 1> x_neg_proj_dof = project_function(msh, cl, hho_di, element_location::IN_NEGATIVE_SIDE, scal_fun);
            
        Matrix<T, Dynamic, 1> x_pos_g_proj_dof = project_vec_function(msh, cl, hho_di, flux_fun, element_location::IN_POSITIVE_SIDE);
        Matrix<T, Dynamic, 1> x_pos_proj_dof = project_function(msh, cl, hho_di, element_location::IN_POSITIVE_SIDE, scal_fun);
            
            
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();
        auto gradeg = this->di.grad_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);
        auto gbs = vector_cell_basis<Mesh,T>::size(gradeg);
        
        auto cell_offset        = offset(msh, cl);
        size_t cell_SOL_offset = this->cell_table.at(cell_offset) * (cbs+gbs);

        x_glob.block(cell_SOL_offset, 0, gbs, 1) = x_neg_g_proj_dof;
        x_glob.block(cell_SOL_offset+gbs, 0, gbs, 1) = x_pos_g_proj_dof;
        x_glob.block(cell_SOL_offset+2*gbs, 0, cbs, 1) = x_neg_proj_dof.block(0, 0, cbs, 1);
        x_glob.block(cell_SOL_offset+cbs+2*gbs, 0, cbs, 1) = x_pos_proj_dof.block(0, 0, cbs, 1);
    
    }
            
    std::vector<std::pair<size_t,size_t>> compute_cell_basis_data(const Mesh& msh){
        size_t n_cells =  msh.cells.size();
        std::vector<std::pair<size_t,size_t>> cell_basis_data;
        cell_basis_data.reserve(n_cells);
        size_t cell_ind = 0;
        for(auto& cl : msh.cells) {
            bool double_unknowns = ( location(msh, cl) == element_location::ON_INTERFACE);
            auto cbs = this->loc_cbs;
            if( double_unknowns ){
                cbs *= 2;
            }
            cell_basis_data.push_back(std::make_pair(cell_ind, cbs));
            cell_ind++;
        }
        
        return cell_basis_data;
    }
    
    size_t get_n_faces(){
        return this->num_other_faces;
    }
            
};

template<typename Mesh, typename Function>
auto make_two_fields_interface_assembler(const Mesh& msh, Function dirichlet_bf, hho_degree_info hdi)
{
    return two_fields_interface_assembler<Mesh, Function>(msh, dirichlet_bf, hdi);
}

/////////////////////////////////////////////////


template<typename Mesh, typename Function>
class interface_condensed_assembler : public virt_interface_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;
    std::vector< Matrix<T, Dynamic, Dynamic> > loc_LHS;
    std::vector< Matrix<T, Dynamic, 1> > loc_RHS;

public:

    interface_condensed_assembler(const Mesh& msh, const Function& dirichlet_bf,
                                   hho_degree_info hdi)
        : virt_interface_assembler<Mesh, Function>(msh, dirichlet_bf, hdi)
    {
        auto facdeg = this->di.face_degree();
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto system_size = fbs * this->num_other_faces;

        this->LHS = SparseMatrix<T>( system_size, system_size );
        this->RHS = Matrix<T, Dynamic, 1>::Zero( system_size );

        this->loc_cbs = 0;

        loc_LHS.resize( msh.cells.size() );
        loc_RHS.resize( msh.cells.size() );
    }

    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        // save local matrices
        size_t cell_offset = offset(msh, cl);
        loc_LHS.at( cell_offset ) = lhs;
        loc_RHS.at( cell_offset ) = rhs;

        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces * fbs;

        if (location(msh, cl) == element_location::ON_INTERFACE)
        {
            cbs = 2 * cbs;
            f_dofs = 2 * f_dofs;
        }

        // static condensation
        auto mat_sc = static_condensation_compute(lhs, rhs, cbs, f_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_sc = mat_sc.first;
        Matrix<T, Dynamic, 1> rhs_sc = mat_sc.second;

        this->assemble_bis(msh, cl, lhs_sc, rhs_sc);
    } // assemble()

    //// take_local_data
    Matrix<T, Dynamic, 1>
    take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl,
                    const Matrix<T, Dynamic, 1>& solution,
                    const element_location where)
    {
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto f_dofs = num_faces * fbs;

        auto solF = this->get_solF(msh, cl, solution);
        size_t offset_cl = offset(msh, cl);
        auto loc_mat = loc_LHS.at(offset_cl);
        auto loc_rhs = loc_RHS.at(offset_cl);

        // Recover the full solution
        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + f_dofs);

        if( location(msh, cl) == element_location::ON_INTERFACE )
        {
            if( where == element_location::IN_NEGATIVE_SIDE )
            {
                ret.head(cbs) = static_condensation_recover(loc_mat, loc_rhs, 2*cbs, 2*f_dofs, solF).head(cbs);
                ret.tail(num_faces*fbs) = solF.head(f_dofs);
            }

            if( where == element_location::IN_POSITIVE_SIDE )
            {
                ret.head(cbs) = static_condensation_recover(loc_mat, loc_rhs, 2*cbs, 2*f_dofs, solF).block(cbs, 0, cbs, 1);
                ret.tail(num_faces*fbs) = solF.tail(f_dofs);
            }
        }
        else
        {
            ret.head(cbs) = static_condensation_recover(loc_mat, loc_rhs, cbs, f_dofs, solF).head(cbs);
            ret.tail(num_faces*fbs) = solF;
        }

        return ret;
    }
};


template<typename Mesh, typename Function>
auto make_interface_condensed_assembler(const Mesh& msh, Function& dirichlet_bf,
                                         hho_degree_info hdi)
{
    return interface_condensed_assembler<Mesh, Function>(msh, dirichlet_bf, hdi);
}


/******************************************************************************************/
/*******************                                               ************************/
/*******************               STOKES ASSEMBLERS               ************************/
/*******************                                               ************************/
/******************************************************************************************/


////////////////////////  STATIC CONDENSATION  //////////////////////////

template<typename T>
std::pair<   Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, 1>  >
stokes_static_condensation_compute
(const Matrix<T, Dynamic, Dynamic> lhs_A, const Matrix<T, Dynamic, Dynamic> lhs_B,
 const Matrix<T, Dynamic, 1> rhs_A, const Matrix<T, Dynamic, 1> rhs_B,
 const size_t cell_size, const size_t face_size)
{
    using matrix = Matrix<T, Dynamic, Dynamic>;
    using vector = Matrix<T, Dynamic, 1>;

    size_t size_tot = cell_size + face_size;
    size_t p_size = lhs_B.rows();
    assert(lhs_A.cols() == size_tot && lhs_A.rows() == size_tot);
    assert(rhs_A.rows() == size_tot || rhs_A.rows() == cell_size);

    matrix lhs_sc = matrix::Zero(face_size + p_size, face_size + p_size);
    vector rhs_sc = vector::Zero(face_size + p_size);


    // sub--lhs
    matrix K_TT = lhs_A.topLeftCorner(cell_size, cell_size);
    matrix K_TF = lhs_A.topRightCorner(cell_size, face_size);
    matrix K_FT = lhs_A.bottomLeftCorner(face_size, cell_size);
    matrix K_FF = lhs_A.bottomRightCorner(face_size, face_size);
    matrix K_PT = lhs_B.block(0, 0, p_size, cell_size);
    matrix K_PF = lhs_B.block(0, cell_size, p_size, face_size);

    // sub--rhs
    vector cell_rhs = vector::Zero(cell_size);
    vector face_rhs = vector::Zero(face_size);
    if(rhs_A.rows() == cell_size)
        cell_rhs = rhs_A;
    else
    {
        cell_rhs = rhs_A.head(cell_size);
        face_rhs = rhs_A.tail(face_size);
    }

    // static condensation
    auto K_TT_ldlt = K_TT.ldlt();
    matrix AL = K_TT_ldlt.solve(K_TF);
    matrix BL = K_TT_ldlt.solve(K_PT.transpose());
    vector rL = K_TT_ldlt.solve(cell_rhs);

    lhs_sc.block(0, 0, face_size, face_size) = K_FF - K_FT * AL;
    lhs_sc.block(face_size, 0, p_size, face_size) = K_PF - K_PT * AL;
    lhs_sc.block(0, face_size, face_size, p_size)
        = lhs_sc.block(face_size, 0, p_size, face_size).transpose();
    lhs_sc.block(face_size, face_size, p_size, p_size) = - K_PT * BL;

    rhs_sc.head(face_size) = face_rhs - K_FT * rL;
    rhs_sc.tail(p_size) = rhs_B - K_PT * rL;

    return std::make_pair(lhs_sc, rhs_sc);
}


// full static condensation (velocity + pressure)
// in this version we keep only the velocity face dofs
// and one pressure dof per cell (that represents the mean pressure in the cell)
template<typename T>
std::pair<   Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, 1>  >
stokes_full_static_condensation_compute
(const Matrix<T, Dynamic, Dynamic> lhs_A, const Matrix<T, Dynamic, Dynamic> lhs_B,
 const Matrix<T, Dynamic, Dynamic> lhs_C,
 const Matrix<T, Dynamic, 1> rhs_A, const Matrix<T, Dynamic, 1> rhs_B,
 const Matrix<T, Dynamic, 1> mult, const size_t cell_size, const size_t face_size)
{
    using matrix = Matrix<T, Dynamic, Dynamic>;
    using vector = Matrix<T, Dynamic, 1>;

    size_t size_tot = cell_size + face_size;
    size_t p_size = lhs_B.rows();
    assert(lhs_A.cols() == size_tot && lhs_A.rows() == size_tot);
    assert(rhs_A.rows() == size_tot || rhs_A.rows() == cell_size);
    assert(lhs_B.rows() == p_size && lhs_B.cols() == size_tot);
    assert(rhs_B.rows() == p_size);
    assert(lhs_C.rows() == p_size && lhs_C.cols() == p_size);
    assert(mult.rows() == p_size - 1);

    matrix lhs_sc = matrix::Zero(face_size + 1, face_size + 1);
    vector rhs_sc = vector::Zero(face_size + 1);

    // sub--lhs
    matrix K_TT = lhs_A.topLeftCorner(cell_size, cell_size);
    matrix K_TF = lhs_A.topRightCorner(cell_size, face_size);
    matrix K_FT = lhs_A.bottomLeftCorner(face_size, cell_size);
    matrix K_FF = lhs_A.bottomRightCorner(face_size, face_size);
    matrix K_PT = lhs_B.block(0, 0, p_size, cell_size);
    matrix K_PF = lhs_B.block(0, cell_size, p_size, face_size);


    // sub--rhs
    vector cell_rhs = vector::Zero(cell_size);
    vector face_rhs = vector::Zero(face_size);
    if(rhs_A.rows() == cell_size)
        cell_rhs = rhs_A;
    else
    {
        cell_rhs = rhs_A.head(cell_size);
        face_rhs = rhs_A.tail(face_size);
    }

    // compute the new matrices
    matrix tB_PT = K_PT.block(0, 0, 1, cell_size);
    matrix tB_PF = K_PF.block(0, 0, 1, face_size);
    matrix temp_B_PT = K_PT.block(1, 0, p_size-1, cell_size);
    matrix temp_B_PF = K_PF.block(1, 0, p_size-1, face_size);
    matrix hB_PT = mult * tB_PT + temp_B_PT;
    matrix hB_PF = mult * tB_PF + temp_B_PF;

    matrix ttC_pp = lhs_C.block(0, 0, 1, p_size).transpose();
    matrix hhC_pp = lhs_C.block(1, 0, p_size - 1, p_size).transpose() + ttC_pp * mult.transpose();
    matrix C_tptp = ttC_pp.block(0, 0, 1, 1);
    matrix C_tphp = hhC_pp.block(0, 0, 1, p_size - 1);
    matrix C_hptp = ttC_pp.block(1, 0, p_size - 1, 1) + mult * C_tptp;
    matrix C_hphp = hhC_pp.block(1, 0, p_size - 1, p_size - 1) + mult * C_tphp;

    vector rhs_tp = rhs_B.block(0,0,1,1);
    vector rhs_temp = rhs_B.block(1,0,p_size-1,1);
    vector rhs_hp = rhs_temp + mult * rhs_tp;


    ////////////// static condensation
    // invert matrices
    auto K_TT_ldlt = K_TT.ldlt();
    matrix iAhB = K_TT_ldlt.solve(hB_PT.transpose());
    matrix iAK_TF = K_TT_ldlt.solve(K_TF);
    matrix iAtB = K_TT_ldlt.solve(tB_PT.transpose());
    vector iA_rhs_T = K_TT_ldlt.solve(cell_rhs);

    auto iBAB = ( hB_PT * iAhB - C_hphp ).ldlt();
    matrix iBAB_B_PF = iBAB.solve(hB_PF);
    matrix iBAB_B_PT = iBAB.solve(hB_PT);
    matrix iBAB_C_hptp = iBAB.solve(C_hptp);
    vector iBAB_rhs_hp = iBAB.solve(rhs_hp);

    // compute final matrices and rhs
    matrix AFF_1 = K_FF;
    matrix AFF_2 = - K_FT * iAK_TF;
    matrix AFF_3 = ( K_FT * iAhB - hB_PF.transpose() ) * (iBAB_B_PT * iAK_TF - iBAB_B_PF);
    matrix AFF = AFF_1 + AFF_2 + AFF_3;

    matrix BFP_1 = tB_PF.transpose();
    matrix BFP_2 = - K_FT * iAtB;
    matrix BFP_3 = ( hB_PF.transpose() - K_FT * iAhB ) * ( iBAB_C_hptp - iBAB_B_PT * iAtB );
    matrix BFP = BFP_1 + BFP_2 + BFP_3;

    vector RHS_F_1 = face_rhs;
    vector RHS_F_2 = - K_FT * iA_rhs_T;
    vector RHS_F_3 = ( hB_PF.transpose() - K_FT * iAhB ) * ( iBAB_rhs_hp - iBAB_B_PT * iA_rhs_T );
    vector RHS_F = RHS_F_1 + RHS_F_2 + RHS_F_3;

    matrix BPF_1 = tB_PF;
    matrix BPF_2 = - tB_PT * iAK_TF;
    matrix BPF_3 = (C_tphp - tB_PT * iAhB) * ( iBAB_B_PF - iBAB_B_PT * iAK_TF );
    matrix BPF = BPF_1 + BPF_2 + BPF_3;

    matrix CPP_1 = C_tptp;
    matrix CPP_2 = - tB_PT * iAtB;
    matrix CPP_3 = ( C_tphp - tB_PT * iAhB ) * ( iBAB_C_hptp - iBAB_B_PT * iAtB );
    matrix CPP = CPP_1 + CPP_2 + CPP_3;

    vector RHS_P_1 = rhs_tp;
    vector RHS_P_2 = - tB_PT * iA_rhs_T;
    vector RHS_P_3 = ( C_tphp - tB_PT * iAhB) * ( iBAB_rhs_hp - iBAB_B_PT * iA_rhs_T );
    vector RHS_P = RHS_P_1 + RHS_P_2 + RHS_P_3;

    lhs_sc.block(0, 0, face_size, face_size) = AFF;
    lhs_sc.block(0, face_size, face_size, 1) = BFP;
    lhs_sc.block(face_size, 0, 1, face_size) = BPF;
    lhs_sc.block(face_size, face_size, 1, 1) = CPP;

    rhs_sc.head(face_size) = RHS_F;
    rhs_sc.tail(1) = RHS_P;

    return std::make_pair(lhs_sc, rhs_sc);
}


template<typename T>
std::pair<   Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, 1>  >
stokes_full_static_condensation_compute
(const Matrix<T, Dynamic, Dynamic> lhs_A, const Matrix<T, Dynamic, Dynamic> lhs_B,
 const Matrix<T, Dynamic, 1> rhs_A, const Matrix<T, Dynamic, 1> rhs_B,
 const Matrix<T, Dynamic, 1> mult, const size_t cell_size, const size_t face_size)
{
    size_t p_size = lhs_B.rows();
    Matrix<T, Dynamic, Dynamic> lhs_C = Matrix<T, Dynamic, Dynamic>::Zero(p_size, p_size);

    return stokes_full_static_condensation_compute
        (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult, cell_size, face_size);
}



template<typename T>
Matrix<T, Dynamic, 1>
stokes_full_static_condensation_recover_v
(const Matrix<T, Dynamic, Dynamic> lhs_A, const Matrix<T, Dynamic, Dynamic> lhs_B,
 const Matrix<T, Dynamic, Dynamic> lhs_C,
 const Matrix<T, Dynamic, 1> rhs_A, const Matrix<T, Dynamic, 1> rhs_B,
 const Matrix<T, Dynamic, 1> mult,
 const size_t cell_size, const size_t face_size, const Matrix<T, Dynamic, 1> sol_sc)
{
    using matrix = Matrix<T, Dynamic, Dynamic>;
    using vector = Matrix<T, Dynamic, 1>;

    size_t size_tot = cell_size + face_size;
    size_t p_size = lhs_B.rows();
    assert(lhs_A.cols() == size_tot && lhs_A.rows() == size_tot);
    assert(rhs_A.rows() == size_tot || rhs_A.rows() == cell_size);
    assert(lhs_B.rows() == p_size && lhs_B.cols() == size_tot);
    assert(rhs_B.rows() == p_size);
    assert(lhs_C.rows() == p_size && lhs_C.cols() == p_size);
    assert(mult.rows() == p_size - 1);
    assert(sol_sc.rows() == face_size + 1);


    // sub--lhs
    matrix K_TT = lhs_A.topLeftCorner(cell_size, cell_size);
    matrix K_TF = lhs_A.topRightCorner(cell_size, face_size);
    matrix K_TP = lhs_B.topLeftCorner(p_size, cell_size).transpose();
    matrix K_PT = lhs_B.block(0, 0, p_size, cell_size);
    matrix K_PF = lhs_B.block(0, cell_size, p_size, face_size);

    // sub--rhs
    vector cell_rhs = vector::Zero(cell_size);
    cell_rhs = rhs_A.head(cell_size);

    // compute the new matrices
    matrix tB_PT = K_PT.block(0, 0, 1, cell_size);
    matrix tB_PF = K_PF.block(0, 0, 1, face_size);
    matrix temp_B_PT = K_PT.block(1, 0, p_size-1, cell_size);
    matrix temp_B_PF = K_PF.block(1, 0, p_size-1, face_size);
    matrix hB_PT = mult * tB_PT + temp_B_PT;
    matrix hB_PF = mult * tB_PF + temp_B_PF;

    matrix ttC_pp = lhs_C.block(0, 0, 1, p_size).transpose();
    matrix hhC_pp = lhs_C.block(1, 0, p_size - 1, p_size).transpose() + ttC_pp * mult.transpose();
    matrix C_tptp = ttC_pp.block(0, 0, 1, 1);
    matrix C_tphp = hhC_pp.block(0, 0, 1, p_size - 1);
    matrix C_hptp = ttC_pp.block(1, 0, p_size - 1, 1) + mult * C_tptp;
    matrix C_hphp = hhC_pp.block(1, 0, p_size - 1, p_size - 1) + mult * C_tphp;

    vector rhs_tp = rhs_B.block(0,0,1,1);
    vector rhs_temp = rhs_B.block(1,0,p_size-1,1);
    vector rhs_hp = rhs_temp + mult * rhs_tp;

    // recover velocity cell solution
    auto K_TT_ldlt = K_TT.ldlt();
    matrix iAhB = K_TT_ldlt.solve(hB_PT.transpose());
    matrix iAK_TF = K_TT_ldlt.solve(K_TF);
    matrix iAtB = K_TT_ldlt.solve(tB_PT.transpose());
    vector iA_rhs_T = K_TT_ldlt.solve(cell_rhs);

    auto iBAB = ( hB_PT * iAhB - C_hphp ).ldlt();
    matrix iBAB_B_PF = iBAB.solve(hB_PF);
    matrix iBAB_B_PT = iBAB.solve(hB_PT);
    matrix iBAB_C_hptp = iBAB.solve(C_hptp);
    vector iBAB_rhs_hp = iBAB.solve(rhs_hp);

    vector ret = vector::Zero(size_tot);

    vector uF = sol_sc.head(face_size);
    vector solP = sol_sc.tail(1);

    vector uT_1 = iAhB * iBAB_B_PT * (iAK_TF * uF + iAtB * solP - iA_rhs_T);
    vector uT_2 = - iAhB * iBAB_B_PF * uF;
    vector uT_3 = iAhB * iBAB_rhs_hp;
    vector uT_4 = - iAhB * iBAB_C_hptp * solP;
    vector uT_5 = iA_rhs_T - iAK_TF * uF - iAtB * solP;
    vector uT = uT_1 + uT_2 + uT_3 + uT_4 + uT_5;

    ret.head(cell_size) = uT;
    ret.block(cell_size, 0, face_size, 1) = uF;
    return ret;
}


template<typename T>
Matrix<T, Dynamic, 1>
stokes_full_static_condensation_recover_v
(const Matrix<T, Dynamic, Dynamic> lhs_A, const Matrix<T, Dynamic, Dynamic> lhs_B,
 const Matrix<T, Dynamic, 1> rhs_A, const Matrix<T, Dynamic, 1> rhs_B,
 const Matrix<T, Dynamic, 1> mult,
 const size_t cell_size, const size_t face_size, const Matrix<T, Dynamic, 1> sol_sc)
{
    size_t p_size = lhs_B.rows();
    Matrix<T, Dynamic, Dynamic> lhs_C = Matrix<T, Dynamic, Dynamic>::Zero(p_size, p_size);

    return stokes_full_static_condensation_recover_v
        (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult, cell_size, face_size, sol_sc);
}


template<typename T>
Matrix<T, Dynamic, 1>
stokes_full_static_condensation_recover_p
(const Matrix<T, Dynamic, Dynamic> lhs_A, const Matrix<T, Dynamic, Dynamic> lhs_B,
 const Matrix<T, Dynamic, Dynamic> lhs_C,
 const Matrix<T, Dynamic, 1> rhs_A, const Matrix<T, Dynamic, 1> rhs_B,
 const Matrix<T, Dynamic, 1> mult,
 const size_t cell_size, const size_t face_size, const Matrix<T, Dynamic, 1> sol_sc)
{
    using matrix = Matrix<T, Dynamic, Dynamic>;
    using vector = Matrix<T, Dynamic, 1>;

    size_t size_tot = cell_size + face_size;
    size_t p_size = lhs_B.rows();
    assert(lhs_A.cols() == size_tot && lhs_A.rows() == size_tot);
    assert(rhs_A.rows() == size_tot || rhs_A.rows() == cell_size);
    assert(lhs_B.rows() == p_size && lhs_B.cols() == size_tot);
    assert(rhs_B.rows() == p_size);
    assert(lhs_C.rows() == p_size && lhs_C.cols() == p_size);
    assert(mult.rows() == p_size - 1);
    assert(sol_sc.rows() == face_size + 1);

    // sub--lhs
    matrix K_TT = lhs_A.topLeftCorner(cell_size, cell_size);
    matrix K_TF = lhs_A.topRightCorner(cell_size, face_size);
    matrix K_TP = lhs_B.topLeftCorner(p_size, cell_size).transpose();
    matrix K_PT = lhs_B.block(0, 0, p_size, cell_size);
    matrix K_PF = lhs_B.block(0, cell_size, p_size, face_size);

    // sub--rhs
    vector cell_rhs = vector::Zero(cell_size);
    cell_rhs = rhs_A.head(cell_size);

    // compute the new matrices
    matrix tB_PT = K_PT.block(0, 0, 1, cell_size);
    matrix tB_PF = K_PF.block(0, 0, 1, face_size);
    matrix temp_B_PT = K_PT.block(1, 0, p_size-1, cell_size);
    matrix temp_B_PF = K_PF.block(1, 0, p_size-1, face_size);
    matrix hB_PT = mult * tB_PT + temp_B_PT;
    matrix hB_PF = mult * tB_PF + temp_B_PF;

    matrix ttC_pp = lhs_C.block(0, 0, 1, p_size).transpose();
    matrix hhC_pp = lhs_C.block(1, 0, p_size - 1, p_size).transpose() + ttC_pp * mult.transpose();
    matrix C_tptp = ttC_pp.block(0, 0, 1, 1);
    matrix C_tphp = hhC_pp.block(0, 0, 1, p_size - 1);
    matrix C_hptp = ttC_pp.block(1, 0, p_size - 1, 1) + mult * C_tptp;
    matrix C_hphp = hhC_pp.block(1, 0, p_size - 1, p_size - 1) + mult * C_tphp;

    vector rhs_tp = rhs_B.block(0,0,1,1);
    vector rhs_temp = rhs_B.block(1,0,p_size-1,1);
    vector rhs_hp = rhs_temp + mult * rhs_tp;

    // pressure solution
    auto K_TT_ldlt = K_TT.ldlt();
    matrix iAhB = K_TT_ldlt.solve(hB_PT.transpose());
    matrix iAK_TF = K_TT_ldlt.solve(K_TF);
    matrix iAtB = K_TT_ldlt.solve(tB_PT.transpose());
    vector iA_rhs_T = K_TT_ldlt.solve(cell_rhs);

    auto iBAB = ( hB_PT * iAhB - C_hphp ).ldlt();
    matrix iBAB_B_PF = iBAB.solve(hB_PF);
    matrix iBAB_B_PT = iBAB.solve(hB_PT);
    matrix iBAB_C_hptp = iBAB.solve(C_hptp);
    vector iBAB_rhs_hp = iBAB.solve(rhs_hp);

    vector ret = vector::Zero(p_size);

    vector uF = sol_sc.head(face_size);
    vector solP = sol_sc.tail(1);

    vector p_1 = iBAB_B_PF * uF;
    vector p_2 = - iBAB_B_PT * iAK_TF * uF;
    vector p_3 = - iBAB_B_PT * iAtB * solP;
    vector p_4 = iBAB_C_hptp * solP;
    vector p_5 = - iBAB_rhs_hp;
    vector p_6 = iBAB_B_PT * iA_rhs_T;
    vector hP = p_1 + p_2 + p_3 + p_4 + p_5 + p_6;

    ret(0) = solP(0) + mult.dot(hP);
    ret.tail(p_size-1) = hP;
    return ret;
}


template<typename T>
Matrix<T, Dynamic, 1>
stokes_full_static_condensation_recover_p
(const Matrix<T, Dynamic, Dynamic> lhs_A, const Matrix<T, Dynamic, Dynamic> lhs_B,
 const Matrix<T, Dynamic, 1> rhs_A, const Matrix<T, Dynamic, 1> rhs_B,
 const Matrix<T, Dynamic, 1> mult,
 const size_t cell_size, const size_t face_size, const Matrix<T, Dynamic, 1> sol_sc)
{
    size_t p_size = lhs_B.rows();
    Matrix<T, Dynamic, Dynamic> lhs_C = Matrix<T, Dynamic, Dynamic>::Zero(p_size, p_size);

    return stokes_full_static_condensation_recover_p
        (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult, cell_size, face_size, sol_sc);
}


//////////////////////////////  STOKES ASSEMBLERS  ///////////////////////////////


template<typename Mesh, typename Function>
class virt_stokes_assembler
{
    using T = typename Mesh::coordinate_type;

protected:
    std::vector< Triplet<T> >           triplets;
    std::vector<size_t>                 face_table;
    std::vector<size_t>                 cell_table;

    hho_degree_info                     di;
    Function                            dir_func;

    element_location loc_zone; // IN_NEGATIVE_SIDE or IN_POSITIVE_SIDE for fictitious problem
                               // ON_INTERFACE for the interface problem
    size_t num_cells, num_other_faces, loc_cbs, loc_pbs;

public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;


    virt_stokes_assembler(const Mesh& msh, const Function& dirichlet_bf, hho_degree_info hdi)
        : dir_func(dirichlet_bf), di(hdi)
    {
    }

    size_t
    face_SOL_offset(const Mesh& msh, const typename Mesh::face_type& fc)
    {
        auto facdeg = di.face_degree();
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto cbs = loc_cbs; // cbs = 0 if static condensation

        auto face_offset = offset(msh, fc);
        return num_cells * cbs + face_table.at(face_offset) * fbs;
    }

    size_t
    P_SOL_offset(const Mesh& msh, const typename Mesh::cell_type& cl)
    {
        auto facdeg = di.face_degree();
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto cbs = loc_cbs; // cbs = 0 if static condensation

        auto cell_offset = offset(msh, cl);

        if( loc_zone != element_location::ON_INTERFACE || cbs != 0 )
            return num_cells * cbs + num_other_faces * fbs + cell_table.at(cell_offset) * loc_pbs;

        return num_other_faces * fbs + cell_offset;
    }

    std::vector<assembly_index>
    init_asm_map(const Mesh& msh, const typename Mesh::cell_type& cl)
    {
        bool double_unknowns = ( location(msh, cl) == element_location::ON_INTERFACE
                                 && loc_zone == element_location::ON_INTERFACE );

        std::vector<assembly_index> asm_map;

        auto facdeg = di.face_degree();
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto f_dofs = num_faces * fbs;
        auto cbs = loc_cbs;
        auto pbs = loc_pbs;
        auto loc_size = cbs + f_dofs + pbs;
        if( double_unknowns && cbs != 0 )
            loc_size = 2 * loc_size;
        else if( double_unknowns && cbs == 0 )
            loc_size = 2 * (cbs + f_dofs) + pbs;

        asm_map.reserve( loc_size );

        size_t cell_offset = cell_table.at( offset(msh, cl) );
        size_t cell_LHS_offset = cell_offset * cbs;

        if( double_unknowns )
            cbs = 2 * cbs;

        for (size_t i = 0; i < cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );


        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_LHS_offset = face_SOL_offset(msh, fc);

            bool in_dom = true;
            if( loc_zone != element_location::ON_INTERFACE )
            {
                element_location loc_fc = location(msh, fc);
                in_dom = (loc_fc == element_location::ON_INTERFACE ||
                          loc_fc == loc_zone);
            }

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET
                && in_dom;

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );
        }

        if( double_unknowns )
        {
            for (size_t face_i = 0; face_i < num_faces; face_i++)
            {
                auto fc = fcs[face_i];
                auto d = (location(msh, fc) == element_location::ON_INTERFACE) ? fbs : 0;
                auto face_LHS_offset = face_SOL_offset(msh, fc) + d;

                bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
                if ( dirichlet )
                    throw std::invalid_argument("Dirichlet boundary on cut cell not supported.");

                for (size_t i = 0; i < fbs; i++)
                    asm_map.push_back( assembly_index(face_LHS_offset+i, true) );
            }
        }

        size_t P_LHS_offset = P_SOL_offset(msh, cl);
        for (size_t i = 0; i < pbs; i++)
            asm_map.push_back( assembly_index(P_LHS_offset+i, true) );

        if( double_unknowns && cbs != 0 )
            for (size_t i = 0; i < pbs; i++)
                asm_map.push_back( assembly_index(P_LHS_offset+pbs+i, true) );

        return asm_map;
    }

    Matrix<T, Dynamic, 1>
    get_dirichlet_data(const Mesh& msh, const typename Mesh::cell_type& cl)
    {
        bool double_unknowns = ( location(msh, cl) == element_location::ON_INTERFACE
                                 && loc_zone == element_location::ON_INTERFACE );

        auto facdeg = di.face_degree();
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto f_dofs = num_faces * fbs;

        auto cbs = loc_cbs;
        auto loc_size = cbs + f_dofs;

        if( double_unknowns )
            loc_size = 2 * loc_size;

        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero( loc_size );

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_LHS_offset = face_SOL_offset(msh, fc);

            bool in_dom = true;
            if( loc_zone != element_location::ON_INTERFACE );
            {
                element_location loc_fc = location(msh, fc);
                bool in_dom = (loc_fc == element_location::ON_INTERFACE ||
                               loc_fc == loc_zone);
            }

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET
                && in_dom;

            if( dirichlet && double_unknowns )
                throw std::invalid_argument("Dirichlet boundary on cut cell not supported.");

            if (dirichlet && loc_zone == element_location::ON_INTERFACE )
            {
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> loc_rhs = make_vector_rhs(msh, fc, facdeg, dir_func);
                dirichlet_data.block(cbs + face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(loc_rhs);
            }
            if (dirichlet && loc_zone != element_location::ON_INTERFACE )
            {
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg, loc_zone);
                Matrix<T, Dynamic, 1> loc_rhs = make_vector_rhs(msh, fc, facdeg, loc_zone, dir_func);
                dirichlet_data.block(cbs + face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(loc_rhs);
            }
        }

        return dirichlet_data;
    }

    // compute_mult_C -> for the static condensation routines
    Matrix<T, Dynamic, 1>
    compute_mult_C(const Mesh& msh, const typename Mesh::cell_type& cl, size_t pdeg)
    {
        bool double_unknowns = ( location(msh, cl) == element_location::ON_INTERFACE &&
                                 loc_zone == element_location::ON_INTERFACE );

        if( pdeg == 0 && !double_unknowns )
            throw std::invalid_argument("mult_C -> invalid argument.");

        auto pbs = cell_basis<Mesh,T>::size(pdeg);
        size_t p_dofs = pbs;
        if( double_unknowns )
            p_dofs = 2 * p_dofs;
        cell_basis<cuthho_poly_mesh<T>, T> pb(msh, cl, pdeg);
        auto qpsi = integrate(msh, cl, pdeg, element_location::IN_NEGATIVE_SIDE);
        if( loc_zone == element_location::IN_POSITIVE_SIDE || location(msh,cl) == element_location::IN_POSITIVE_SIDE )
            qpsi = integrate(msh, cl, pdeg, element_location::IN_POSITIVE_SIDE);
        Matrix<T, Dynamic, 1> mult_C = Matrix<T, Dynamic, 1>::Zero( p_dofs - 1 );
        T area = 0.0;
        if( pdeg > 0 )
        {
            for (auto& qp : qpsi)
            {
                auto p_phi = pb.eval_basis(qp.first);
                mult_C.head(pbs - 1) -= qp.second * p_phi.tail(pbs - 1);
                area += qp.second;
            }
        }
        else
        {
            for (auto& qp : qpsi)
            {
                area += qp.second;
            }
        }

        if( double_unknowns )
        {
            auto qpsi_p = integrate(msh, cl, pdeg, element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qpsi_p)
            {
                auto p_phi = pb.eval_basis(qp.first);
                mult_C.block(pbs - 1, 0, pbs, 1) -= qp.second * p_phi;
            }
        }

        mult_C = mult_C / area;
        return mult_C;
    }

    void
    assemble_bis(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        if( !(location(msh, cl) == loc_zone
              || location(msh, cl) == element_location::ON_INTERFACE
              || loc_zone == element_location::ON_INTERFACE ) )
            return;

        auto asm_map = init_asm_map(msh, cl);
        auto dirichlet_data = get_dirichlet_data(msh, cl);

        assert( asm_map.size() == lhs.rows() && asm_map.size() == lhs.cols() );

        // LHS
        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs(i,j)) );
                else
                    RHS[asm_map[i]] -= lhs(i,j)*dirichlet_data(j);
            }
        }

        // RHS
        for (size_t i = 0; i < rhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            RHS[asm_map[i]] += rhs(i);
        }


        // null mean pressure condition -> done in each assemble routines
    }
            
    Matrix<T, Dynamic, 1>
    get_solF(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, 1>& solution)
    {
        bool double_unknowns = ( location(msh, cl) == element_location::ON_INTERFACE
                                 && loc_zone == element_location::ON_INTERFACE );

        auto facdeg = di.face_degree();
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces*fbs;
        if( double_unknowns )
            f_dofs = 2 * f_dofs;

        Matrix<T, Dynamic, 1> solF = Matrix<T, Dynamic, 1>::Zero( f_dofs );

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            if( loc_zone != element_location::ON_INTERFACE )
            {
                auto loc_fc = location(msh, fc);
                if( !(loc_fc == element_location::ON_INTERFACE || loc_fc == loc_zone) )
                    continue;
            }

            auto face_LHS_offset = face_SOL_offset(msh, fc);
            if ( location(msh, fc) == element_location::ON_INTERFACE
                 && loc_zone == element_location::ON_INTERFACE )
            {
                // we assume that there is not boundary condition on cut cells (for interface pb)
                solF.block(face_i*fbs, 0, fbs, 1) = solution.block(face_LHS_offset, 0, fbs, 1);
                solF.block( (num_faces+face_i)*fbs, 0, fbs, 1)
                    = solution.block(face_LHS_offset + fbs, 0, fbs, 1);
                continue;
            }

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_vector_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_vector_rhs(msh, fc, facdeg, dir_func);
                solF.block(face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs);
                continue;
            }
            if( location(msh, cl) == element_location::ON_INTERFACE &&
                location(msh, fc) == element_location::IN_POSITIVE_SIDE &&
                loc_zone == element_location::ON_INTERFACE )
            {
                solF.block((num_faces+face_i)*fbs, 0, fbs, 1)
                    = solution.block(face_LHS_offset, 0, fbs, 1);
                continue;
            }
            //else
            solF.block(face_i*fbs, 0, fbs, 1) = solution.block(face_LHS_offset, 0, fbs, 1);
        }
        return solF;
    }

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    }
};

///////////////////////////   STOKES FICTITIOUS DOMAIN  /////////////////////////////////


template<typename Mesh, typename Function>
class virt_stokes_fict_assembler : public virt_stokes_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;

public:

    virt_stokes_fict_assembler(const Mesh& msh, const Function& dirichlet_bf,
                               hho_degree_info hdi, element_location where)
        : virt_stokes_assembler<Mesh, Function>(msh, dirichlet_bf, hdi)
    {
        if( where != element_location::IN_NEGATIVE_SIDE
            && where != element_location::IN_POSITIVE_SIDE )
            throw std::invalid_argument("Choose the location in NEGATIVE/POSITIVE side.");
        this->loc_zone = where;

        auto is_removed = [&](const typename Mesh::face_type& fc) -> bool {
            bool is_dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
            auto loc = location(msh,fc);
            bool is_where = (loc == where || loc == element_location::ON_INTERFACE);
            return is_dirichlet || (!is_where);
        };

        auto num_all_faces = msh.faces.size();
        auto num_removed_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_removed);
        this->num_other_faces = num_all_faces - num_removed_faces;

        this->face_table.resize( num_all_faces );

        size_t compressed_offset = 0;
        for (size_t i = 0; i < num_all_faces; i++)
        {
            auto fc = msh.faces[i];
            if ( !is_removed(fc) )
            {
                this->face_table.at(i) = compressed_offset;
                compressed_offset++;
            }
        }

        this->cell_table.resize( msh.cells.size() );
        compressed_offset = 0;
        for (size_t i = 0; i < msh.cells.size(); i++)
        {
            auto cl = msh.cells[i];
            if (location(msh, cl) == where || location(msh, cl) == element_location::ON_INTERFACE)
            {
                this->cell_table.at(i) = compressed_offset;
                compressed_offset++;
            }
        }
        this->num_cells = compressed_offset;
    }
};

///////////////////////////////////////////////////////////////////


template<typename Mesh, typename Function>
class stokes_fict_assembler : public virt_stokes_fict_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;

public:

    stokes_fict_assembler(const Mesh& msh, Function& dirichlet_bf,
                          hho_degree_info hdi, element_location where)
        : virt_stokes_fict_assembler<Mesh, Function>(msh, dirichlet_bf, hdi, where)
    {

        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = cell_basis<Mesh,T>::size(facdeg);

        this->loc_cbs = cbs;
        this->loc_pbs = pbs;

        auto system_size = (cbs + pbs) * this->num_cells + fbs * this->num_other_faces + 1;

        this->LHS = SparseMatrix<T>( system_size, system_size );
        this->RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs_A, const Matrix<T, Dynamic, Dynamic>& lhs_B,
             const Matrix<T, Dynamic, 1>& rhs_A, const Matrix<T, Dynamic, 1>& rhs_B)
    {
        if( !(location(msh, cl) == this->loc_zone ||
              location(msh, cl) == element_location::ON_INTERFACE) )
            return;

        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();
        auto pdeg = facdeg;

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto f_dofs = num_faces * fbs;

        auto v_size = cbs + f_dofs;
        auto loc_size = v_size + pbs;

        Matrix<T, Dynamic, Dynamic> lhs = Matrix<T, Dynamic, Dynamic>::Zero( loc_size , loc_size );
        lhs.block(0, 0, v_size, v_size) = lhs_A;
        lhs.block(v_size, 0, pbs, v_size) = lhs_B;
        lhs.block(0, v_size, v_size, pbs) = lhs_B.transpose();

        Matrix<T, Dynamic, 1> rhs = Matrix<T, Dynamic, 1>::Zero( loc_size );
        rhs.head( rhs_A.size() ) = rhs_A;
        rhs.tail( pbs ) = rhs_B;

        this->assemble_bis(msh, cl, lhs, rhs);

        // null pressure mean condition
        cell_basis<cuthho_poly_mesh<T>, T> cb(msh, cl, pdeg);
        auto qpsi = integrate(msh, cl, pdeg, this->loc_zone);
        Matrix<T, Dynamic, 1> mult = Matrix<T, Dynamic, 1>::Zero( pbs );
        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);
            mult += qp.second * phi;
        }
        auto mult_offset = (cbs + pbs) * this->num_cells + fbs * this->num_other_faces;

        size_t P_LHS_offset = this->P_SOL_offset(msh, cl);
        for (size_t i = 0; i < mult.rows(); i++)
        {
            this->triplets.push_back( Triplet<T>(P_LHS_offset+i, mult_offset, mult(i)) );
            this->triplets.push_back( Triplet<T>(mult_offset, P_LHS_offset+i, mult(i)) );
        }

    } // assemble()

    Matrix<T, Dynamic, 1>
    take_velocity(const Mesh& msh, const typename Mesh::cell_type& cl,
    const Matrix<T, Dynamic, 1>& solution)
    {
        auto loc_cl = location(msh, cl);
        if( !(loc_cl == element_location::ON_INTERFACE || loc_cl == this->loc_zone) )
            throw std::logic_error("Bad cell !!");

        auto solF = this->get_solF(msh, cl, solution);

        auto cbs = this->loc_cbs;

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + solF.size() );

        auto cell_offset = this->cell_table.at( offset(msh, cl) );
        auto cell_SOL_offset    = cell_offset * cbs;

        ret.head(cbs) = solution.block(cell_SOL_offset, 0, cbs, 1);
        ret.tail( solF.size() ) = solF;

        return ret;
    }


    Matrix<T, Dynamic, 1>
    take_pressure(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& sol)
    {
        auto loc_cl = location(msh, cl);
        if( !(loc_cl == element_location::ON_INTERFACE || loc_cl == this->loc_zone) )
            throw std::logic_error("Bad cell !!");

        auto pres_offset = this->P_SOL_offset(msh, cl);
        Matrix<T, Dynamic, 1> spres = sol.block(pres_offset, 0, this->loc_pbs, 1);
        return spres;
    }

};


template<typename Mesh, typename Function>
auto make_stokes_fict_assembler(const Mesh& msh, Function& dirichlet_bf,
                                hho_degree_info hdi, element_location where)
{
    return stokes_fict_assembler<Mesh,Function>(msh, dirichlet_bf, hdi, where);
}

///////////////////////////////////////////////////////


template<typename Mesh, typename Function>
class stokes_fict_condensed_assembler : public virt_stokes_fict_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;

    std::vector< Matrix<T, Dynamic, Dynamic> > loc_LHS_A, loc_LHS_B;
    std::vector< Matrix<T, Dynamic, 1> > loc_RHS_A, loc_RHS_B;

public:

    stokes_fict_condensed_assembler(const Mesh& msh, Function& dirichlet_bf,
                                    hho_degree_info hdi, element_location where)
        : virt_stokes_fict_assembler<Mesh, Function>(msh, dirichlet_bf, hdi, where)
    {
        loc_LHS_A.resize( this->num_cells );
        loc_LHS_B.resize( this->num_cells );
        loc_RHS_A.resize( this->num_cells );
        loc_RHS_B.resize( this->num_cells );

        auto facdeg = this->di.face_degree();

        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);

        this->loc_cbs = 0;
        this->loc_pbs = 1;

        auto system_size = this->num_cells + fbs * this->num_other_faces + 1;

        this->LHS = SparseMatrix<T>( system_size, system_size );
        this->RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs_A, const Matrix<T, Dynamic, Dynamic>& lhs_B,
             const Matrix<T, Dynamic, 1>& rhs_A, const Matrix<T, Dynamic, 1>& rhs_B)
    {
        if( !(location(msh, cl) == this->loc_zone ||
              location(msh, cl) == element_location::ON_INTERFACE) )
            return;

        // save local matrices
        size_t cell_offset = this->cell_table.at( offset(msh, cl) );
        loc_LHS_A.at( cell_offset ) = lhs_A;
        loc_LHS_B.at( cell_offset ) = lhs_B;
        loc_RHS_A.at( cell_offset ) = rhs_A;
        loc_RHS_B.at( cell_offset ) = rhs_B;

        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();
        auto pdeg = facdeg;

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto f_dofs = num_faces * fbs;

        auto v_size = cbs + f_dofs;
        auto loc_size = v_size + pbs;

        // static condensation
        Matrix<T, Dynamic, Dynamic> lhs_sc;
        Matrix<T, Dynamic, 1> rhs_sc;
        if(facdeg == 0) // condensate only cell velocity dofs
        {
            auto mat_sc = stokes_static_condensation_compute(lhs_A, lhs_B, rhs_A, rhs_B,
                                                             cbs, f_dofs);
            lhs_sc = mat_sc.first;
            rhs_sc = mat_sc.second;
        }
        else // full static condensation
        {
            auto mult_C = this->compute_mult_C(msh, cl, pdeg);
            auto mat_sc = stokes_full_static_condensation_compute
                (lhs_A, lhs_B, rhs_A, rhs_B, mult_C, cbs, f_dofs);
            lhs_sc = mat_sc.first;
            rhs_sc = mat_sc.second;
        }

        this->assemble_bis(msh, cl, lhs_sc, rhs_sc);

        // null pressure mean condition
        auto P_LHS_offset = this->P_SOL_offset(msh, cl);
        auto mult_offset = fbs * this->num_other_faces + this->num_cells;
        auto area = measure(msh, cl, this->loc_zone);
        this->triplets.push_back( Triplet<T>(P_LHS_offset, mult_offset, area) );
        this->triplets.push_back( Triplet<T>(mult_offset, P_LHS_offset, area) );

    } // assemble()

    Matrix<T, Dynamic, 1>
    take_velocity(const Mesh& msh, const typename Mesh::cell_type& cl,
    const Matrix<T, Dynamic, 1>& solution)
    {
        auto loc_cl = location(msh, cl);
        if( !(loc_cl == element_location::ON_INTERFACE || loc_cl == this->loc_zone) )
            throw std::logic_error("Bad cell !!");

        auto solF = this->get_solF(msh, cl, solution);
        auto f_dofs = solF.size();

        auto cbs = vector_cell_basis<Mesh,T>::size( this->di.cell_degree() );

        auto facdeg = this->di.face_degree();

        auto cell_offset = this->cell_table.at( offset(msh, cl) );

        auto loc_mat_A = loc_LHS_A.at( cell_offset );
        auto loc_mat_B = loc_LHS_B.at( cell_offset );
        auto loc_rhs_A = loc_RHS_A.at( cell_offset );
        auto loc_rhs_B = loc_RHS_B.at( cell_offset );

        if(facdeg == 0)
            return static_condensation_recover(loc_mat_A, loc_rhs_A, cbs, f_dofs, solF);

        // at this point facdeg > 0
        Matrix<T, Dynamic, 1> sol_sc = Matrix<T, Dynamic, 1>::Zero(f_dofs + 1);
        sol_sc.head(f_dofs) = solF;
        auto P_LHS_offset = this->P_SOL_offset(msh, cl);
        sol_sc(f_dofs) = solution(P_LHS_offset);

        auto mult_C = this->compute_mult_C(msh, cl, this->di.face_degree() );
        return stokes_full_static_condensation_recover_v
            (loc_mat_A, loc_mat_B, loc_rhs_A, loc_rhs_B, mult_C, cbs, f_dofs, sol_sc);
    }


    Matrix<T, Dynamic, 1>
    take_pressure(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& sol)
    {
        auto loc_cl = location(msh, cl);
        if( !(loc_cl == element_location::ON_INTERFACE || loc_cl == this->loc_zone) )
            throw std::logic_error("Bad cell !!");

        auto facdeg = this->di.face_degree();

        auto pres_offset = this->P_SOL_offset(msh, cl);
        Matrix<T, Dynamic, 1> spres = sol.block(pres_offset, 0, this->loc_pbs, 1);
        if(facdeg == 0)
            return spres;

        // at this point facdeg > 0
        auto solF = this->get_solF(msh, cl, sol);
        auto f_dofs = solF.size();
        Matrix<T, Dynamic, 1> sol_sc = Matrix<T, Dynamic, 1>::Zero(f_dofs + 1);
        sol_sc.head(f_dofs) = solF;
        sol_sc.tail(1) = spres;

        auto cell_offset = this->cell_table.at( offset(msh, cl) );

        auto mult_C = this->compute_mult_C(msh, cl, this->di.face_degree() );
        auto cbs = vector_cell_basis<Mesh,T>::size( this->di.cell_degree() );
        return stokes_full_static_condensation_recover_p
            (loc_LHS_A.at(cell_offset), loc_LHS_B.at(cell_offset),
             loc_RHS_A.at(cell_offset), loc_RHS_B.at(cell_offset), mult_C,
             cbs, f_dofs, sol_sc);
    }

};


template<typename Mesh, typename Function>
auto make_stokes_fict_condensed_assembler(const Mesh& msh, Function& dirichlet_bf,
                                          hho_degree_info hdi, element_location where)
{
    return stokes_fict_condensed_assembler<Mesh,Function>(msh, dirichlet_bf, hdi, where);
}


///////////////////////////////   STOKES INTERFACE  ///////////////////////////////


template<typename Mesh, typename Function>
class virt_stokes_interface_assembler : public virt_stokes_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;

public:

    virt_stokes_interface_assembler(const Mesh& msh, const Function& dirichlet_bf,
                                    hho_degree_info hdi)
        : virt_stokes_assembler<Mesh, Function>(msh, dirichlet_bf, hdi)
    {
        this->loc_zone = element_location::ON_INTERFACE;

        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
        };

        size_t loc_num_cells = 0; /* counts cells with dup. unknowns */
        for (auto& cl : msh.cells)
        {
            this->cell_table.push_back( loc_num_cells );
            if (location(msh, cl) == element_location::ON_INTERFACE)
                loc_num_cells += 2;
            else
                loc_num_cells += 1;
        }
        this->num_cells = loc_num_cells;
        assert(this->cell_table.size() == msh.cells.size());

        size_t num_all_faces = 0; /* counts faces with dup. unknowns */
        for (auto& fc : msh.faces)
        {
            if (location(msh, fc) == element_location::ON_INTERFACE)
                num_all_faces += 2;
            else
                num_all_faces += 1;
        }

        /* We assume that cut cells can not have dirichlet faces */
        auto num_dirichlet_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_dirichlet);
        this->num_other_faces = num_all_faces - num_dirichlet_faces;
        this->face_table.resize( msh.faces.size() );

        size_t compressed_offset = 0;
        for (size_t i = 0; i < msh.faces.size(); i++)
        {
            auto fc = msh.faces.at(i);
            if ( !is_dirichlet(fc) )
            {
                this->face_table.at(i) = compressed_offset;
                if ( location(msh, fc) == element_location::ON_INTERFACE )
                    compressed_offset += 2;
                else
                    compressed_offset += 1;
            }
        }
    }
};

/////////////////////////////////////////////////


template<typename Mesh, typename Function>
class stokes_interface_assembler : public virt_stokes_interface_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;

public:

    stokes_interface_assembler(const Mesh& msh, const Function& dirichlet_bf, hho_degree_info hdi)
        : virt_stokes_interface_assembler<Mesh, Function>(msh, dirichlet_bf, hdi)
    {
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();
        auto pdeg = facdeg;

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        this->loc_cbs = cbs;
        this->loc_pbs = pbs;

        auto system_size = (cbs+pbs) * this->num_cells + fbs * this->num_other_faces + 1;

        this->LHS = SparseMatrix<T>( system_size, system_size );
        this->RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }


    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        bool double_unknowns = (location(msh, cl) == element_location::ON_INTERFACE);

        this->assemble_bis(msh, cl, lhs, rhs);

        // null mean pressure condition
        auto facdeg = this->di.face_degree();
        auto pdeg = facdeg;
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = this->loc_pbs;

        auto p_dofs = pbs;
        if( double_unknowns )
            p_dofs = 2 * p_dofs;

        cell_basis<cuthho_poly_mesh<T>, T> pb(msh, cl, pdeg);
        auto qpsi = integrate(msh, cl, pdeg);
        if( double_unknowns )
            qpsi = integrate(msh, cl, pdeg, element_location::IN_NEGATIVE_SIDE);
        Matrix<T, Dynamic, 1> mult = Matrix<T, Dynamic, 1>::Zero( p_dofs );
        for (auto& qp : qpsi)
        {
            auto phi = pb.eval_basis(qp.first);
            mult.head( pbs ) += qp.second * phi;
        }
        if( double_unknowns )
        {
            auto qpsi_p = integrate(msh, cl, pdeg, element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qpsi_p)
            {
                auto phi = pb.eval_basis(qp.first);
                mult.tail( pbs ) += qp.second * phi;
            }
        }
        auto P_LHS_offset = this->P_SOL_offset(msh, cl);
        auto mult_offset = (this->loc_cbs + this->loc_pbs) * this->num_cells
            + fbs * this->num_other_faces;

        for (size_t i = 0; i < mult.rows(); i++)
        {
            this->triplets.push_back( Triplet<T>(P_LHS_offset+i, mult_offset, mult(i)) );
            this->triplets.push_back( Triplet<T>(mult_offset, P_LHS_offset+i, mult(i)) );
        }

    } // assemble()

    Matrix<T, Dynamic, 1>
    take_velocity(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& solution, element_location where)
    {
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);

        auto cell_offset        = offset(msh, cl);
        size_t cell_SOL_offset;
        if ( location(msh, cl) == element_location::ON_INTERFACE )
        {
            if (where == element_location::IN_NEGATIVE_SIDE)
                cell_SOL_offset = this->cell_table.at(cell_offset) * cbs;
            else if (where == element_location::IN_POSITIVE_SIDE)
                cell_SOL_offset = this->cell_table.at(cell_offset) * cbs + cbs;
            else
                throw std::invalid_argument("Invalid location");
        }
        else
        {
            cell_SOL_offset = this->cell_table.at(cell_offset) * cbs;
        }

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + num_faces*fbs);
        ret.block(0, 0, cbs, 1) = solution.block(cell_SOL_offset, 0, cbs, 1);

        auto solF = this->get_solF(msh, cl, solution);
        if(where == element_location::IN_NEGATIVE_SIDE)
            ret.tail(num_faces * fbs) = solF.head(num_faces * fbs);
        else
            ret.tail(num_faces * fbs) = solF.tail(num_faces * fbs);

        return ret;
    }

    Matrix<T, Dynamic, 1>
    take_pressure(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& sol, element_location where)
    {
        auto pdeg = this->di.face_degree();
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        auto P_LHS_offset = this->P_SOL_offset(msh, cl);
        if( location(msh, cl) == element_location::ON_INTERFACE &&
            where == element_location::IN_POSITIVE_SIDE )
            P_LHS_offset += pbs;

        Matrix<T, Dynamic, 1> spres = sol.block(P_LHS_offset, 0, pbs, 1);
        return spres;
    }
};


template<typename Mesh, typename Function>
auto make_stokes_interface_assembler(const Mesh& msh, Function& dirichlet_bf, hho_degree_info hdi)
{
    return stokes_interface_assembler<Mesh, Function>(msh, dirichlet_bf, hdi);
}

///////////////////////////////////////////////////////



template<typename Mesh, typename Function>
class stokes_interface_condensed_assembler : public virt_stokes_interface_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;

    std::vector< Matrix<T, Dynamic, Dynamic> > loc_LHS;
    std::vector< Matrix<T, Dynamic, 1> > loc_RHS;

public:

    stokes_interface_condensed_assembler(const Mesh& msh, Function& dirichlet_bf,
                                         hho_degree_info hdi)
        : virt_stokes_interface_assembler<Mesh, Function>(msh, dirichlet_bf, hdi)
    {
        this->loc_zone = element_location::ON_INTERFACE;

        loc_LHS.resize( msh.cells.size() );
        loc_RHS.resize( msh.cells.size() );

        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();
        auto pdeg = facdeg;

        this->loc_cbs = 0;
        this->loc_pbs = 1;

        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);

        auto system_size = fbs * this->num_other_faces + msh.cells.size() + 1;

        this->LHS = SparseMatrix<T>( system_size, system_size );
        this->RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs)
    {
        bool double_unknowns = ( location(msh, cl) == element_location::ON_INTERFACE );

        // save local matrices
        size_t cell_offset = offset(msh, cl);
        loc_LHS.at( cell_offset ) = lhs;
        loc_RHS.at( cell_offset ) = rhs;

        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();
        auto pdeg = facdeg;

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto f_dofs = num_faces * fbs;
        auto v_dofs = cbs + f_dofs;
        auto p_dofs = pbs;

        if( double_unknowns )
        {
            cbs = 2 * cbs;
            f_dofs = 2 * f_dofs;
            v_dofs = 2 * v_dofs;
            p_dofs = 2 * p_dofs;
        }

        // static condensation
        Matrix<T, Dynamic, Dynamic> lhs_A = lhs.block(0, 0, v_dofs, v_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_B = lhs.block(v_dofs, 0, p_dofs, v_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_C = lhs.block(v_dofs, v_dofs, p_dofs, p_dofs);
        Matrix<T, Dynamic, 1> rhs_A = rhs.head( v_dofs );
        Matrix<T, Dynamic, 1> rhs_B = rhs.tail( p_dofs );

        Matrix<T, Dynamic, Dynamic> lhs_sc;
        Matrix<T, Dynamic, 1> rhs_sc;
        if( (facdeg == 0) & !double_unknowns ) // condensate only cell velocity dofs
        {
            auto mat_sc = stokes_static_condensation_compute(lhs_A, lhs_B, rhs_A, rhs_B,
                                                             cbs, f_dofs);
            lhs_sc = mat_sc.first;
            rhs_sc = mat_sc.second;
        }
        else // full static condensation
        {
            auto mult_C = this->compute_mult_C(msh, cl, pdeg);

            auto mat_sc = stokes_full_static_condensation_compute
                (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult_C, cbs, f_dofs);
            lhs_sc = mat_sc.first;
            rhs_sc = mat_sc.second;
        }

        this->assemble_bis(msh, cl, lhs_sc, rhs_sc);

        // null mean pressure condition
        auto P_LHS_offset = this->P_SOL_offset(msh, cl);
        auto mult_offset = fbs * this->num_other_faces + msh.cells.size();
        auto area = measure(msh,cl);
        this->triplets.push_back( Triplet<T>(P_LHS_offset, mult_offset, area) );
        this->triplets.push_back( Triplet<T>(mult_offset, P_LHS_offset, area) );

    } // assemble()


    Matrix<T, Dynamic, 1>
    take_velocity(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& solution, element_location where)
    {
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();
        auto pdeg = facdeg;

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        auto cell_offset        = offset(msh, cl);
        auto lhs = loc_LHS.at(cell_offset);
        auto rhs = loc_RHS.at(cell_offset);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces*fbs;

        auto v_dofs = cbs + f_dofs;
        auto p_dofs = pbs;
        auto f_dofs2 = f_dofs;
        auto cbs2 = cbs;
        if( location(msh, cl) == element_location::ON_INTERFACE )
        {
            cbs2 = 2 * cbs;
            f_dofs2 = 2 * f_dofs;
            v_dofs = 2 * v_dofs;
            p_dofs = 2 * p_dofs;
        }

        // Recover the full solution
        Matrix<T, Dynamic, Dynamic> lhs_A = lhs.block(0, 0, v_dofs, v_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_B = lhs.block(v_dofs, 0, p_dofs, v_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_C = lhs.block(v_dofs, v_dofs, p_dofs, p_dofs);
        Matrix<T, Dynamic, 1> rhs_A = rhs.head( v_dofs );
        Matrix<T, Dynamic, 1> rhs_B = rhs.tail( p_dofs );

        auto solF = this->get_solF(msh, cl, solution);
        if( facdeg == 0 && location(msh, cl) != element_location::ON_INTERFACE )
            return static_condensation_recover(lhs_A, rhs_A, cbs, f_dofs, solF);

        // compute sol_sc
        Matrix<T, Dynamic, 1> sol_sc = Matrix<T, Dynamic, 1>::Zero(solF.size() + 1);
        sol_sc.head( solF.size() ) = solF;
        sol_sc( solF.size() ) = solution( this->P_SOL_offset(msh, cl) );

        auto mult_C = this->compute_mult_C(msh, cl, pdeg);
        auto loc_sol = stokes_full_static_condensation_recover_v
            (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult_C, cbs2, f_dofs2, sol_sc);

        if( location(msh, cl) != element_location::ON_INTERFACE )
            return loc_sol;

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + f_dofs);
        if( where == element_location::IN_NEGATIVE_SIDE )
        {
            ret.head(cbs) = loc_sol.head(cbs);
            ret.tail(f_dofs) = solF.head(f_dofs);
        }

        if( where == element_location::IN_POSITIVE_SIDE )
        {
            ret.head(cbs) = loc_sol.block(cbs, 0, cbs, 1);
            ret.tail(f_dofs) = solF.tail(f_dofs);
        }
        return ret;
    }

    Matrix<T, Dynamic, 1>
    take_pressure(const Mesh& msh, const typename Mesh::cell_type& cl,
                  const Matrix<T, Dynamic, 1>& sol, element_location where)
    {
        auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();
        auto pdeg = facdeg;

        auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
        auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
        auto pbs = cell_basis<Mesh,T>::size(pdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        size_t f_dofs = num_faces*fbs;

        auto v_dofs = cbs + f_dofs;
        auto p_dofs = pbs;
        if( location(msh, cl) == element_location::ON_INTERFACE )
        {
            cbs = 2 * cbs;
            f_dofs = 2 * f_dofs;
            v_dofs = 2 * v_dofs;
            p_dofs = 2 * p_dofs;
        }

        auto cell_offset = offset(msh, cl);
        auto lhs = loc_LHS.at(cell_offset);
        auto rhs = loc_RHS.at(cell_offset);

        Matrix<T, Dynamic, Dynamic> lhs_A = lhs.block(0, 0, v_dofs, v_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_B = lhs.block(v_dofs, 0, p_dofs, v_dofs);
        Matrix<T, Dynamic, Dynamic> lhs_C = lhs.block(v_dofs, v_dofs, p_dofs, p_dofs);
        Matrix<T, Dynamic, 1> rhs_A = rhs.head( v_dofs );
        Matrix<T, Dynamic, 1> rhs_B = rhs.tail( p_dofs );

        size_t P_LHS_offset = this->P_SOL_offset(msh, cl);
        if( facdeg == 0 && location(msh, cl) != element_location::ON_INTERFACE )
        {
            Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(pbs);
            ret(0) = sol(P_LHS_offset);
            return ret;
        }

        // compute sol_sc
        auto solF = this->get_solF(msh, cl, sol);
        Matrix<T, Dynamic, 1> sol_sc = Matrix<T, Dynamic, 1>::Zero(solF.size() + 1);
        sol_sc.head( solF.size() ) = solF;
        sol_sc( solF.size() ) = sol(P_LHS_offset);

        auto mult_C = this->compute_mult_C(msh, cl, pdeg);
        auto loc_sol = stokes_full_static_condensation_recover_p
            (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult_C, cbs, f_dofs, sol_sc);

        if( location(msh, cl) != element_location::ON_INTERFACE )
            return loc_sol;

        if( where == element_location::IN_NEGATIVE_SIDE )
            return loc_sol.head(pbs);

        return loc_sol.tail(pbs);
    }
};


template<typename Mesh, typename Function>
auto make_stokes_interface_condensed_assembler(const Mesh& msh, Function& dirichlet_bf,
                                               hho_degree_info hdi)
{
    return stokes_interface_condensed_assembler<Mesh, Function>(msh, dirichlet_bf, hdi);
}

