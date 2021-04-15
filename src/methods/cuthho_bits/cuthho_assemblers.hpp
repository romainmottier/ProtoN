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
static_condensation_compute(const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs,
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
static_condensation_recover(const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs,
                            const size_t cell_size, const size_t face_size,
                            const Matrix<T, Dynamic, 1>& solF)
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
    
    assembly_index(std::pair<size_t,bool>& value)
    : idx(value.first), assem(value.second)
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
    std::vector<size_t>                 face_table;
    std::vector<size_t>                 cell_table;

    hho_degree_info                     di;
    Function                            dir_func;

    element_location loc_zone; // IN_NEGATIVE_SIDE or IN_POSITIVE_SIDE for fictitious problem
                               // ON_INTERFACE for the interface problem
    size_t num_cells, num_other_faces, loc_cbs;

public:
//  Function                            dir_func; // ERA PROTECTED!!!!!!!!!!!!
    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;


    virt_scalar_assembler(const Mesh& msh, const Function& dirichlet_bf, hho_degree_info& hdi)
        : dir_func(dirichlet_bf), di(hdi)
    {
    }

void set_dir_func(const Function& f) {
  dir_func = f;
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
                    throw std::invalid_argument("Dirichlet boundary on cut cell not supported.");

                for (size_t i = 0; i < fbs; i++)
                    asm_map.push_back( assembly_index(face_LHS_offset+i, true) );
            }
        }

        return asm_map;
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
                in_dom = (loc_fc == element_location::ON_INTERFACE ||
                               loc_fc == loc_zone);
            }

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET
                && in_dom;

            if( dirichlet && double_unknowns )
                throw std::invalid_argument("Dirichlet boundary on cut cell not supported.");

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
                    RHS(asm_map[i]) -= lhs(i,j)*dirichlet_data(j);
            }
        }

        // RHS
        for (size_t i = 0; i < rhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            RHS(asm_map[i]) += rhs(i);
        }
    }

    Matrix<T, Dynamic, 1>
    get_solF(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, 1>& solution)
    {
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
            // If I would bdry cond on cut cells I should modify here
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
    }
};


//////////////////////////////   FICTITIOUS DOMAIN ASSEMBLERS   /////////////////////////////



template<typename Mesh, typename Function>
class virt_fict_assembler : public virt_scalar_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;

public:

    virt_fict_assembler(const Mesh& msh, const Function& dirichlet_bf,
                        hho_degree_info& hdi, element_location where)
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
                   hho_degree_info& hdi, element_location where)
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
auto make_fict_assembler(const Mesh& msh, const Function& dirichlet_bf,
                         hho_degree_info& hdi, element_location where)
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
                             hho_degree_info& hdi, element_location where)
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
auto make_fict_condensed_assembler(const Mesh& msh, const Function& dirichlet_bf,
                                   hho_degree_info& hdi, element_location where)
{
    return fict_condensed_assembler<Mesh, Function>(msh, dirichlet_bf, hdi, where);
}


////////////////////////////////  INTERFACE ASSEMBLERS  /////////////////////////



template<typename Mesh, typename Function>
class virt_interface_assembler : public virt_scalar_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;

public:

    virt_interface_assembler(const Mesh& msh, const Function& dirichlet_bf, hho_degree_info& hdi)
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
         std::cout<<"Number of all faces (faces on interface counted twice) =  "<<num_all_faces<<std::endl;
        /* We assume that cut cells can not have dirichlet faces */
        auto num_dirichlet_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_dirichlet);
      //  std::cout<<"Number of dirichlet faces "<<num_dirichlet_faces<<std::endl;
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

    interface_assembler(const Mesh& msh, const Function& dirichlet_bf, hho_degree_info& hdi)
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
auto make_interface_assembler(const Mesh& msh, Function& dirichlet_bf, hho_degree_info& hdi)
{
    return interface_assembler<Mesh, Function>(msh, dirichlet_bf, hdi);
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
                                  hho_degree_info& hdi)
        : virt_interface_assembler<Mesh, Function>(msh, dirichlet_bf, hdi)
    {
        auto facdeg = this->di.face_degree();
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto system_size = fbs * this->num_other_faces;
        std::cout<<"system_size (num of faces - Dirichlet faces) is "<<system_size<<std::endl;
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
        // solF is the sol also in dirichlet fcs
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
                                        hho_degree_info& hdi)
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
(const Matrix<T, Dynamic, Dynamic>& lhs_A, const Matrix<T, Dynamic, Dynamic>& lhs_B,
 const Matrix<T, Dynamic, 1>& rhs_A, const Matrix<T, Dynamic, 1>& rhs_B,
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
(const Matrix<T, Dynamic, Dynamic>& lhs_A, const Matrix<T, Dynamic, Dynamic>& lhs_B,
 const Matrix<T, Dynamic, Dynamic>& lhs_C,
 const Matrix<T, Dynamic, 1>& rhs_A, const Matrix<T, Dynamic, 1>& rhs_B,
 //const Matrix<T, Dynamic, 1> mult, const size_t cell_size, const size_t face_size)
const Matrix<T, Dynamic, 1>& mult, const T coeff_cell,
const size_t cell_size, const size_t face_size)
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
    //matrix tB_PT = K_PT.block(0, 0, 1, cell_size);
    //matrix tB_PF = K_PF.block(0, 0, 1, face_size);
    matrix ttB_PT = K_PT.block(0, 0, 1, cell_size); // ADD NEW
    matrix ttB_PF = K_PF.block(0, 0, 1, face_size); // ADD NEW
    matrix tB_PT = (1.0+coeff_cell) * ttB_PT; // ADD NEW
    matrix tB_PF = (1.0+coeff_cell) * ttB_PF; // ADD NEW
    
    matrix temp_B_PT = K_PT.block(1, 0, p_size-1, cell_size);
    matrix temp_B_PF = K_PF.block(1, 0, p_size-1, face_size);
    /*
    matrix hB_PT = mult * tB_PT + temp_B_PT;
    matrix hB_PF = mult * tB_PF + temp_B_PF;

    matrix ttC_pp = lhs_C.block(0, 0, 1, p_size).transpose();
    matrix hhC_pp = lhs_C.block(1, 0, p_size - 1, p_size).transpose() + ttC_pp * mult.transpose();
    matrix C_tptp = ttC_pp.block(0, 0, 1, 1);
    matrix C_tphp = hhC_pp.block(0, 0, 1, p_size - 1);
    matrix C_hptp = ttC_pp.block(1, 0, p_size - 1, 1) + mult * C_tptp;
    matrix C_hphp = hhC_pp.block(1, 0, p_size - 1, p_size - 1) + mult * C_tphp;

    vector rhs_tp = rhs_B.block(0,0,1,1);
    */
    
    matrix hB_PT = mult * ttB_PT + temp_B_PT;// ADD NEW
    matrix hB_PF = mult * ttB_PF + temp_B_PF;// ADD NEW

    matrix ttC_pp = (1.0+coeff_cell) * lhs_C.block(0, 0, 1, p_size).transpose();// ADD NEW
    matrix hhC_pp = lhs_C.block(1, 0, p_size - 1, p_size).transpose() + lhs_C.block(0, 0, 1, p_size).transpose() * mult.transpose();// ADD NEW
    matrix C_tptp = (1.0/(1.0+coeff_cell)) * ttC_pp.block(0, 0, 1, 1);// ADD NEW
    matrix C_tphp = (1.0/(1.0+coeff_cell)) * hhC_pp.block(0, 0, 1, p_size - 1);// ADD NEW
    matrix C_hptp = ttC_pp.block(1, 0, p_size - 1, 1) + mult * ttC_pp.block(0, 0, 1, 1);// ADD NEW
    matrix C_hphp = hhC_pp.block(1, 0, p_size - 1, p_size - 1) + mult * hhC_pp.block(0, 0, 1, p_size - 1);// ADD NEW

    vector rhs_ttp = rhs_B.block(0,0,1,1); // ADD NEW
    vector rhs_tp = (1.0/(1.0+coeff_cell)) * rhs_ttp; // ADD NEW
    
    vector rhs_temp = rhs_B.block(1,0,p_size-1,1);
    //vector rhs_hp = rhs_temp + mult * rhs_tp;
    vector rhs_hp = rhs_temp + mult * rhs_ttp;  // ADD NEW


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
(const Matrix<T, Dynamic, Dynamic>& lhs_A, const Matrix<T, Dynamic, Dynamic>& lhs_B,
 const Matrix<T, Dynamic, 1>& rhs_A, const Matrix<T, Dynamic, 1>& rhs_B,
 //const Matrix<T, Dynamic, 1> mult, const size_t cell_size, const size_t face_size)
const Matrix<T, Dynamic, 1>& mult, const T coeff_cell,
const size_t cell_size, const size_t face_size)
{
    size_t p_size = lhs_B.rows();
    Matrix<T, Dynamic, Dynamic> lhs_C = Matrix<T, Dynamic, Dynamic>::Zero(p_size, p_size);

    return stokes_full_static_condensation_compute
        (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult, coeff_cell, cell_size, face_size); //ADD NEW
        //(lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult, cell_size, face_size);
}



template<typename T>
Matrix<T, Dynamic, 1>
stokes_full_static_condensation_recover_v
(const Matrix<T, Dynamic, Dynamic>& lhs_A, const Matrix<T, Dynamic, Dynamic>& lhs_B,
 const Matrix<T, Dynamic, Dynamic>& lhs_C,
 const Matrix<T, Dynamic, 1>& rhs_A, const Matrix<T, Dynamic, 1>& rhs_B,
 //const Matrix<T, Dynamic, 1> mult,
 const Matrix<T, Dynamic, 1>& mult, const T coeff_cell,
 const size_t cell_size, const size_t face_size, const Matrix<T, Dynamic, 1>& sol_sc)
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
    //matrix tB_PT = K_PT.block(0, 0, 1, cell_size);
    //matrix tB_PF = K_PF.block(0, 0, 1, face_size);
    matrix ttB_PT = K_PT.block(0, 0, 1, cell_size); // ADD NEW
    matrix ttB_PF = K_PF.block(0, 0, 1, face_size); // ADD NEW
    matrix tB_PT = (1.0+coeff_cell) * ttB_PT; // ADD NEW
    matrix tB_PF = (1.0+coeff_cell) * ttB_PF; // ADD NEW
    
    
    matrix temp_B_PT = K_PT.block(1, 0, p_size-1, cell_size);
    matrix temp_B_PF = K_PF.block(1, 0, p_size-1, face_size);
    
    /*
    matrix hB_PT = mult * tB_PT + temp_B_PT;
    matrix hB_PF = mult * tB_PF + temp_B_PF;

    matrix ttC_pp = lhs_C.block(0, 0, 1, p_size).transpose();
    matrix hhC_pp = lhs_C.block(1, 0, p_size - 1, p_size).transpose() + ttC_pp * mult.transpose();
    matrix C_tptp = ttC_pp.block(0, 0, 1, 1);
    matrix C_tphp = hhC_pp.block(0, 0, 1, p_size - 1);
    matrix C_hptp = ttC_pp.block(1, 0, p_size - 1, 1) + mult * C_tptp;
    matrix C_hphp = hhC_pp.block(1, 0, p_size - 1, p_size - 1) + mult * C_tphp;

    vector rhs_tp = rhs_B.block(0,0,1,1);
    */
    matrix hB_PT = mult * ttB_PT + temp_B_PT; // ADD NEW
    matrix hB_PF = mult * ttB_PF + temp_B_PF; // ADD NEW

    matrix ttC_pp = (1.0+coeff_cell) * lhs_C.block(0, 0, 1, p_size).transpose(); // ADD NEW
    matrix hhC_pp = lhs_C.block(1, 0, p_size - 1, p_size).transpose() + lhs_C.block(0, 0, 1, p_size).transpose() * mult.transpose(); // ADD NEW
    matrix C_tptp = (1.0/(1.0+coeff_cell)) * ttC_pp.block(0, 0, 1, 1); // ADD NEW
    matrix C_tphp = (1.0/(1.0+coeff_cell)) * hhC_pp.block(0, 0, 1, p_size - 1); // ADD NEW
    matrix C_hptp = ttC_pp.block(1, 0, p_size - 1, 1) + mult * ttC_pp.block(0, 0, 1, 1); // ADD NEW
    matrix C_hphp = hhC_pp.block(1, 0, p_size - 1, p_size - 1) + mult * hhC_pp.block(0, 0, 1, p_size - 1); // ADD NEW

    vector rhs_ttp = rhs_B.block(0,0,1,1); // ADD NEW
    vector rhs_tp = (1.0/(1.0+coeff_cell)) * rhs_ttp; // ADD NEW
    
    vector rhs_temp = rhs_B.block(1,0,p_size-1,1);
    
    //vector rhs_hp = rhs_temp + mult * rhs_tp;
    vector rhs_hp = rhs_temp + mult * rhs_ttp; // ADD NEW

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
(const Matrix<T, Dynamic, Dynamic>& lhs_A, const Matrix<T, Dynamic, Dynamic>& lhs_B,
 const Matrix<T, Dynamic, 1>& rhs_A, const Matrix<T, Dynamic, 1>& rhs_B,
 //const Matrix<T, Dynamic, 1> mult,
 const Matrix<T, Dynamic, 1>& mult, const T coeff_cell,
 const size_t cell_size, const size_t face_size, const Matrix<T, Dynamic, 1>& sol_sc)
{
    size_t p_size = lhs_B.rows();
    Matrix<T, Dynamic, Dynamic> lhs_C = Matrix<T, Dynamic, Dynamic>::Zero(p_size, p_size);

    return stokes_full_static_condensation_recover_v
        (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult, coeff_cell, cell_size, face_size, sol_sc);
        //(lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult, cell_size, face_size, sol_sc);
}


template<typename T>
Matrix<T, Dynamic, 1>
stokes_full_static_condensation_recover_p
(const Matrix<T, Dynamic, Dynamic>& lhs_A, const Matrix<T, Dynamic, Dynamic>& lhs_B,
 const Matrix<T, Dynamic, Dynamic>& lhs_C,
 const Matrix<T, Dynamic, 1>& rhs_A, const Matrix<T, Dynamic, 1>& rhs_B,
 //const Matrix<T, Dynamic, 1> mult,
 const Matrix<T, Dynamic, 1>& mult, const T coeff_cell,
 const size_t cell_size, const size_t face_size, const Matrix<T, Dynamic, 1>& sol_sc)
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
    //matrix tB_PT = K_PT.block(0, 0, 1, cell_size);
    //matrix tB_PF = K_PF.block(0, 0, 1, face_size);
    matrix ttB_PT = K_PT.block(0, 0, 1, cell_size); // ADD NEW
    matrix ttB_PF = K_PF.block(0, 0, 1, face_size); // ADD NEW
    matrix tB_PT = (1.0+coeff_cell) * ttB_PT; // ADD NEW
    matrix tB_PF = (1.0+coeff_cell) * ttB_PF; // ADD NEW
    
    matrix temp_B_PT = K_PT.block(1, 0, p_size-1, cell_size);
    matrix temp_B_PF = K_PF.block(1, 0, p_size-1, face_size);
    /*
    matrix hB_PT = mult * tB_PT + temp_B_PT;
    matrix hB_PF = mult * tB_PF + temp_B_PF;

    matrix ttC_pp = lhs_C.block(0, 0, 1, p_size).transpose();
    matrix hhC_pp = lhs_C.block(1, 0, p_size - 1, p_size).transpose() + ttC_pp * mult.transpose();
    matrix C_tptp = ttC_pp.block(0, 0, 1, 1);
    matrix C_tphp = hhC_pp.block(0, 0, 1, p_size - 1);
    matrix C_hptp = ttC_pp.block(1, 0, p_size - 1, 1) + mult * C_tptp;
    matrix C_hphp = hhC_pp.block(1, 0, p_size - 1, p_size - 1) + mult * C_tphp;

    vector rhs_tp = rhs_B.block(0,0,1,1);
    */
    matrix hB_PT = mult * ttB_PT + temp_B_PT;  // ADD NEW
    matrix hB_PF = mult * ttB_PF + temp_B_PF;  // ADD NEW

    matrix ttC_pp = (1.0+coeff_cell) * lhs_C.block(0, 0, 1, p_size).transpose();  // ADD NEW
    matrix hhC_pp = lhs_C.block(1, 0, p_size - 1, p_size).transpose() + lhs_C.block(0, 0, 1, p_size).transpose() * mult.transpose();  // ADD NEW
    matrix C_tptp = (1.0/(1.0+coeff_cell)) * ttC_pp.block(0, 0, 1, 1);  // ADD NEW
    matrix C_tphp = (1.0/(1.0+coeff_cell)) * hhC_pp.block(0, 0, 1, p_size - 1);  // ADD NEW
    matrix C_hptp = ttC_pp.block(1, 0, p_size - 1, 1) + mult * ttC_pp.block(0, 0, 1, 1);  // ADD NEW
    matrix C_hphp = hhC_pp.block(1, 0, p_size - 1, p_size - 1) + mult * hhC_pp.block(0, 0, 1, p_size - 1);  // ADD NEW

    vector rhs_ttp = rhs_B.block(0,0,1,1);  // ADD NEW
    vector rhs_tp = (1.0/(1.0+coeff_cell)) * rhs_ttp;  // ADD NEW
    
    
    vector rhs_temp = rhs_B.block(1,0,p_size-1,1);
    //vector rhs_hp = rhs_temp + mult * rhs_tp;
    vector rhs_hp = rhs_temp + mult * rhs_ttp; // ADD NEW

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

    //ret(0) = solP(0) + mult.dot(hP);
    ret(0) = (1 + coeff_cell) * solP(0) + mult.dot(hP);
    ret.tail(p_size-1) = hP;
    return ret;
}


template<typename T>
Matrix<T, Dynamic, 1>
stokes_full_static_condensation_recover_p
(const Matrix<T, Dynamic, Dynamic>& lhs_A, const Matrix<T, Dynamic, Dynamic>& lhs_B,
 const Matrix<T, Dynamic, 1>& rhs_A, const Matrix<T, Dynamic, 1>& rhs_B,
 //const Matrix<T, Dynamic, 1> mult,
 const Matrix<T, Dynamic, 1>& mult, const T coeff_cell,
 const size_t cell_size, const size_t face_size, const Matrix<T, Dynamic, 1>& sol_sc)
{
    size_t p_size = lhs_B.rows();
    Matrix<T, Dynamic, Dynamic> lhs_C = Matrix<T, Dynamic, Dynamic>::Zero(p_size, p_size);

    return stokes_full_static_condensation_recover_p
        (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult, coeff_cell, cell_size, face_size, sol_sc);
        //(lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult, cell_size, face_size, sol_sc);
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


    virt_stokes_assembler(const Mesh& msh, const Function& dirichlet_bf, hho_degree_info& hdi)
        : dir_func(dirichlet_bf), di(hdi)
    {}

    virt_stokes_assembler( virt_stokes_assembler& other)
    {
        triplets = other.triplets ;
        face_table = other.face_table ;
        cell_table = other.cell_table ;
        di = other.di;
        dir_func = other.dir_func;
        loc_zone = other.loc_zone;
        num_cells = other.num_cells;
        num_other_faces = other.num_other_faces;
        loc_cbs = other.loc_cbs;
        loc_pbs = other.loc_pbs;
        LHS = other.LHS;
        RHS = other.RHS;
        
    }
    
    virt_stokes_assembler( const virt_stokes_assembler& other)
    {
        triplets = other.triplets ;
        face_table = other.face_table ;
        cell_table = other.cell_table ;
        di = other.di;
        dir_func = other.dir_func;
        loc_zone = other.loc_zone;
        num_cells = other.num_cells;
        num_other_faces = other.num_other_faces;
        loc_cbs = other.loc_cbs;
        loc_pbs = other.loc_pbs;
        LHS = other.LHS;
        RHS = other.RHS;
        
    }
    
    void set_dir_func(const Function& f) {
      dir_func = f;
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
            //auto face_LHS_offset = face_SOL_offset(msh, fc);

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
                    RHS(asm_map[i]) -= lhs(i,j)*dirichlet_data(j);
            }
        }

        // RHS
        for (size_t i = 0; i < rhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            RHS(asm_map[i]) += rhs(i);
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
                               hho_degree_info& hdi, element_location where)
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
                          hho_degree_info& hdi, element_location where)
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
                                hho_degree_info& hdi, element_location where)
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
                                    hho_degree_info& hdi, element_location where)
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
        //auto pbs = cell_basis<Mesh,T>::size(pdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto f_dofs = num_faces * fbs;

        //auto v_size = cbs + f_dofs;
        //auto loc_size = v_size + pbs;

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
                (lhs_A, lhs_B, rhs_A, rhs_B, mult_C, 0.0, cbs, f_dofs); // ADD NEW
               // (lhs_A, lhs_B, rhs_A, rhs_B, mult_C, cbs, f_dofs);
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
            (loc_mat_A, loc_mat_B, loc_rhs_A, loc_rhs_B, mult_C, 0.0, cbs, f_dofs, sol_sc); // ADD NEW
            //(loc_mat_A, loc_mat_B, loc_rhs_A, loc_rhs_B, mult_C, cbs, f_dofs, sol_sc);
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
             //loc_RHS_A.at(cell_offset), loc_RHS_B.at(cell_offset), mult_C,
             loc_RHS_A.at(cell_offset), loc_RHS_B.at(cell_offset), mult_C, 0.0,
             cbs, f_dofs, sol_sc);
    }

};


template<typename Mesh, typename Function>
auto make_stokes_fict_condensed_assembler(const Mesh& msh, Function& dirichlet_bf,
                                          hho_degree_info& hdi, element_location where)
{
    return stokes_fict_condensed_assembler<Mesh,Function>(msh, dirichlet_bf, hdi, where);
}


///////////////////////////////   STOKES INTERFACE  ///////////////////////////////


template<typename Mesh, typename Function>
class virt_stokes_interface_assembler : public virt_stokes_assembler<Mesh, Function>
{
    using T = typename Mesh::coordinate_type;

public:
    
    virt_stokes_interface_assembler(virt_stokes_interface_assembler& other ):virt_stokes_assembler<Mesh, Function>(other){}
    virt_stokes_interface_assembler(const virt_stokes_interface_assembler& other ):virt_stokes_assembler<Mesh, Function>(other){}

    virt_stokes_interface_assembler(const Mesh& msh, const Function& dirichlet_bf,
                                    hho_degree_info& hdi)
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

    stokes_interface_assembler(stokes_interface_assembler& other):virt_stokes_interface_assembler<Mesh, Function>(other) {}
    
    stokes_interface_assembler(const stokes_interface_assembler& other):virt_stokes_interface_assembler<Mesh, Function>(other) {}
    
    
    stokes_interface_assembler(const Mesh& msh, const Function& dirichlet_bf, hho_degree_info& hdi)
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
auto make_stokes_interface_assembler(const Mesh& msh, Function& dirichlet_bf, hho_degree_info& hdi)
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

    stokes_interface_condensed_assembler(stokes_interface_condensed_assembler& other):virt_stokes_interface_assembler<Mesh, Function>(other) {
        loc_LHS = other.loc_LHS;
        loc_RHS = other.loc_RHS ;
    }
    
    stokes_interface_condensed_assembler(const stokes_interface_condensed_assembler& other):virt_stokes_interface_assembler<Mesh, Function>(other) {
        loc_LHS = other.loc_LHS;
        loc_RHS = other.loc_RHS ;
    }
    
    
    stokes_interface_condensed_assembler(const Mesh& msh, Function& dirichlet_bf,
                                         hho_degree_info& hdi)
        : virt_stokes_interface_assembler<Mesh, Function>(msh, dirichlet_bf, hdi)
    {
        this->loc_zone = element_location::ON_INTERFACE;

        loc_LHS.resize( msh.cells.size() );
        loc_RHS.resize( msh.cells.size() );

        //auto celdeg = this->di.cell_degree();
        auto facdeg = this->di.face_degree();
        //auto pdeg = facdeg;

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
        if( (facdeg == 0) & (!double_unknowns) ) // condensate only cell velocity dofs
        {
            auto mat_sc = stokes_static_condensation_compute(lhs_A, lhs_B, rhs_A, rhs_B,
                                                             cbs, f_dofs);
            lhs_sc = mat_sc.first;
            rhs_sc = mat_sc.second;
        }
        else // full static condensation
        {
            auto mult_C = this->compute_mult_C(msh, cl, pdeg);

            T coeff_cell = 0.0; // ADD NEW
            if( double_unknowns ) // ADD NEW
            {
                coeff_cell = measure(msh,cl,element_location::IN_POSITIVE_SIDE)
                    / measure(msh,cl,element_location::IN_NEGATIVE_SIDE);
            }

           
            auto mat_sc = stokes_full_static_condensation_compute
                (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult_C, coeff_cell, cbs, f_dofs); // ADD NEW
               // (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult_C, cbs, f_dofs);
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

        T coeff_cell = 0.0; // ADD NEW
        if( location(msh, cl) == element_location::ON_INTERFACE ) // ADD NEW
        {
            coeff_cell = measure(msh,cl,element_location::IN_POSITIVE_SIDE)
                / measure(msh,cl,element_location::IN_NEGATIVE_SIDE);
        }
        
        auto mult_C = this->compute_mult_C(msh, cl, pdeg);
        auto loc_sol = stokes_full_static_condensation_recover_v
            (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult_C, coeff_cell, cbs2, f_dofs2, sol_sc);
        // ADD NEW
           // (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult_C, cbs2, f_dofs2, sol_sc);

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

        T coeff_cell = 0.0; // ADD NEW
        if( location(msh, cl) == element_location::ON_INTERFACE ) // ADD NEW
        {
            coeff_cell = measure(msh,cl,element_location::IN_POSITIVE_SIDE)
                / measure(msh,cl,element_location::IN_NEGATIVE_SIDE);
        }
        
        auto mult_C = this->compute_mult_C(msh, cl, pdeg);
        auto loc_sol = stokes_full_static_condensation_recover_p
            (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult_C, coeff_cell, cbs, f_dofs, sol_sc); // ADD NEW
           // (lhs_A, lhs_B, lhs_C, rhs_A, rhs_B, mult_C, cbs, f_dofs, sol_sc);

        if( location(msh, cl) != element_location::ON_INTERFACE )
            return loc_sol;

        if( where == element_location::IN_NEGATIVE_SIDE )
            return loc_sol.head(pbs);

        return loc_sol.tail(pbs);
    }
};


template<typename Mesh, typename Function>
auto make_stokes_interface_condensed_assembler(const Mesh& msh, Function& dirichlet_bf,
                                               hho_degree_info& hdi)
{
    return stokes_interface_condensed_assembler<Mesh, Function>(msh, dirichlet_bf, hdi);
}
