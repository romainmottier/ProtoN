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

#include <Eigen/Dense>

template<typename T>
T iexp_pow(T x, size_t n)
{
    if (n == 0)
        return 1;

    T y = 1;
    while (n > 1)
    {
        if (n % 2 == 0)
        {
            x = x * x;
            n = n / 2;
        }
        else
        {
            y = x * y;
            x = x * x;
            n = (n-1)/2;
        }
    }

    return x*y;
}


size_t basis_size(size_t k, size_t d)
{
    size_t num = 1;
    size_t den = 1;

    for (size_t i = 1; i <= d; i++)
    {
        num *= k + i;
        den *= i;
    }

    return num/den;
}


/*
template<typename T>
std::vector<point<T,1> >
reference_nodes(size_t degree)
{
    auto comp_degree = degree + 1;

    size_t reqd_nodes = comp_degree;

    std::vector<point<T,1> > ret;
    ret.reserve(reqd_nodes);

    point<T,1>  qp;
    T           a1, a2;
    T           delta_x;
    switch(reqd_nodes)
    {
        case 1:
            qp = point<T,1>({0.0});
            ret.push_back(qp);
            return ret;

        case 2:
            qp = point<T,1>({ 1.0 });
            ret.push_back( -qp );
            ret.push_back( qp );
            return ret;

        case 3:
            qp = point<T,1>({ 1.0 });
            ret.push_back( -qp );
            ret.push_back( qp );
            qp = point<T,1>({0.0});
            ret.push_back( qp );
            return ret;

        case 4:
            a1 = 1.0/3.0;
            qp = point<T,1>({ 1.0 });
            ret.push_back( -qp );
            ret.push_back( qp );
            qp = point<T,1>({ a1 });
            ret.push_back( -qp );
            ret.push_back( qp );
            return ret;

        case 5:
            // Be carefull in what order data is inserted in ret!
            // In Gauss Legendre the first one was 0.0, now is the last one
            a2 = 0.5;
            a1 = 1.0;
            qp = point<T,1>({ a1 });
            ret.push_back( -qp );
            ret.push_back( qp );

            qp = point<T,1>({ a2 });
            ret.push_back( -qp );
            ret.push_back( qp );

            qp = point<T,1>({ 0.0 });
            ret.push_back( qp );

            return ret;

        default:

            delta_x = 1.0/degree;
            a1 = 1.0;
            while (a1>0) {
                qp = point<T,1>({ a1 });
                ret.push_back( -qp );
                ret.push_back( qp );
                a1-=delta_x;

            }
            if(a1==0)
            {
                qp = point<T,1>({0.0});
                ret.push_back( qp );
            }
            return ret;
    }

    return ret;
}



template<typename T,typename Mesh>
std::vector< point<T,2> >
equidistriduted_nodes(const Mesh& msh,
          const typename Mesh::cell_type& cl,
          size_t degree)
{
    typedef typename Mesh::point_type    point_type;

    auto qps = reference_nodes<T>(degree);
    auto pts = points(msh, cl);

    auto v0 = pts[1] - pts[0];
    auto v1 = pts[2] - pts[1];
    auto v2 = pts[3] - pts[2];
    auto v3 = pts[3] - pts[0];

    std::vector< point<T,2> > ret;

    auto P = [&](T xi, T eta) -> T {
        return 0.25 * pts[0].x() * (1-xi)*(1-eta) +
               0.25 * pts[1].x() * (1+xi)*(1-eta) +
               0.25 * pts[2].x() * (1+xi)*(1+eta) +
               0.25 * pts[3].x() * (1-xi)*(1+eta);
    };

    auto Q = [&](T xi, T eta) -> T {
        return 0.25 * pts[0].y() * (1-xi)*(1-eta) +
               0.25 * pts[1].y() * (1+xi)*(1-eta) +
               0.25 * pts[2].y() * (1+xi)*(1+eta) +
               0.25 * pts[3].y() * (1-xi)*(1+eta);
    };

    for (auto jtor = qps.begin(); jtor != qps.end(); jtor++)
    {
        for (auto itor = qps.begin(); itor != qps.end(); itor++)
        {
            auto qp_x = *itor;
            auto qp_y = *jtor;

            auto xi = qp_x.x();
            auto eta = qp_y.x();

            auto px = P(xi, eta);
            auto py = Q(xi, eta);

            ret.push_back( point_type(px, py) );
        }
    }

    return ret;
}


*/
/*
// Lagrangian basis b_kl(x,y) = b_k(x)*b_l(y) over a set of equidistributed 2-dimensional nodes (3D CASE NOT YET IMPLEMENTED)

template<typename Mesh, typename T >
class cell_basis_Lagrangian
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;

    point_type          cell_bar;
    coordinate_type     cell_h;
    size_t              basis_degree, basis_size;
    std::vector<point<T, 2> >          nodes;

public:
    cell_basis_Lagrangian(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree)
    {
        nodes           = equidistriduted_nodes<T,Mesh>(msh, cl, degree);
       
        basis_degree    = degree;
        basis_size      = (basis_degree+1)*(basis_degree+1);
    }

    Matrix<T, Dynamic, 1>
    eval_basis(const point_type& pt)
    {
        Matrix<T, Dynamic, 1> rety = Matrix<T, Dynamic, 1>::Zero(basis_size);
        Matrix<T, Dynamic, 1> retx = Matrix<T, Dynamic, 1>::Zero(basis_size);
        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(basis_size);

        // Per la y, trovo la colonna facendo col = l%(degree+1)
        // scorro su tutta la colonna tmpy = [col:(degree+1): col+(degree+1)*degree]
        // faccio base bl moltiplicando tutti tranne quando tmpy = l
        for ( size_t l = 0; l < basis_size ; l++ )
        {
            size_t col = l%(basis_degree+1);
            T bl = 1.0;
            for (size_t tmpy = col; tmpy <= col+(basis_degree+1)*basis_degree; tmpy+=(basis_degree+1))
            {
                if (tmpy!=l)
                {
                    bl *= ( ( pt.y() - (nodes.at(tmpy)).y() )/ ( (nodes.at(l)).y() - (nodes.at(tmpy)).y() ) );
                }
            }
            rety(l) = bl;
        }
        
        // Per la x, trovo la riga facendo riga = floor(k/(degree+1))
        //scorro su tutta la riga tmpx = [(degree+1)*riga: (degree+1)*(riga+1)-1]
        // faccio base bk moltiplicando tutti tranne quando tmpx = k
         
        for (size_t k = 0 ; k < basis_size ; k++ )
        {
            T bk = 1.0;
            size_t row=floor( k/(basis_degree+1) );
            for (size_t tmpx = (basis_degree+1)*row; tmpx <= (basis_degree+1)*(row+1)-1; tmpx++)
            {
                if (tmpx!=k) {
                    bk *= ( ( pt.x() - (nodes.at(tmpx)).x() )/ ( (nodes.at(k)).x() - (nodes.at(tmpx)).x() ) );
                }
            }
            retx(k) = bk;
        }
        
        for (size_t i = 0; i<basis_size; i++)
        {
            ret(i) = rety(i)*retx(i);
        }
        return ret;
        
    }

    Matrix<T, Dynamic, 2>
    eval_gradients(const point_type& pt)
    {
        // Modified Yves Daoust Algorithm (https://math.stackexchange.com/questions/809927/first-derivative-of-lagrange-polynomial)
        
        Matrix<T, Dynamic, 2> ret = Matrix<T, Dynamic, 2>::Zero(basis_size, 2);
       
        Matrix<T, Dynamic, 1> rety = Matrix<T, Dynamic, 1>::Zero(basis_size);
        Matrix<T, Dynamic, 1> retx = Matrix<T, Dynamic, 1>::Zero(basis_size);
        Matrix<T, Dynamic, 1> sy = Matrix<T, Dynamic, 1>::Zero(basis_size);
        Matrix<T, Dynamic, 1> sx = Matrix<T, Dynamic, 1>::Zero(basis_size);


        // for each l, b_l(y)' = {sum(tmpy!=l)[prod(jy!=l,jy!=tmpy)[x-x_jy]]}/prod(tmpy!=l)[x_l-x_tmpy]
        
        for ( size_t l = 0; l < basis_size ; l++ )
        {
            size_t col = l%(basis_degree+1);
            T bl = 1.0 , bl_der = 1.0 ;
            T sumy = 0.0;
            for (size_t tmpy = col; tmpy <= col+(basis_degree+1)*basis_degree; tmpy+=(basis_degree+1))
            {
                 T sumyy = 1.0 ;
                if (tmpy!=l)
                {

                    bl *= ( ( pt.y() - (nodes.at(tmpy)).y() )/ ( (nodes.at(l)).y() - (nodes.at(tmpy)).y() ) );

                    bl_der *= ( 1.0/ ( (nodes.at(l)).y() - (nodes.at(tmpy)).y() ) );
                    for ( size_t jy = col; jy <= col+(basis_degree+1)*basis_degree; jy+=(basis_degree+1) )
                   {
                        if (jy!=tmpy && jy!=l)
                        {
                            sumyy *= ( pt.y()-(nodes.at(jy)).y() );
                        }
                    }
                    sumy +=sumyy;
                }
            }
            rety(l) = bl;
            sy(l) = bl_der*sumy;
        }
        
        // For the x-derivative of b_k(x), same procedure of b_l(y)'
         
        for (size_t k = 0 ; k < basis_size ; k++ )
        {
            size_t row=floor( k/(basis_degree+1) );
            T bk = 1.0 , bk_der = 1.0 ;
            T sumx = 0.0;
            for (size_t tmpx = (basis_degree+1)*row; tmpx <= (basis_degree+1)*(row+1)-1; tmpx++)
            {
                T sumxx = 1.0 ;
                
                if (tmpx!=k) {
                    
                    bk *= ( ( pt.x() - (nodes.at(tmpx)).x() )/ ( (nodes.at(k)).x() - (nodes.at(tmpx)).x() ) );
                    bk_der *= ( 1.0/ ( (nodes.at(k)).x() - (nodes.at(tmpx)).x() ) );
                    for (size_t jx = (basis_degree+1)*row; jx <= (basis_degree+1)*(row+1)-1; jx++)
                    {
                        if (jx!=tmpx && jx!=k)
                        {
                            sumxx *= ( pt.x()-(nodes.at(jx)).x() );
                        }
                    }
                    sumx += sumxx;
                    
                }
            }
            retx(k) = bk;
            sx(k) = bk_der*sumx;
        }
        
        for (size_t i = 0; i<basis_size; i++)
        {
            ret(i,0) = rety(i)*sx(i);
            ret(i,1) = retx(i)*sy(i);
            
        }
        return ret;
        
    }

    size_t size() const
    {
        return basis_size;
    }

    size_t degree() const
    {
        return basis_degree;
    }

    static size_t size(size_t degree)
    {
        return (degree+1)*(degree+1);
    }
};



*/










//#define POWER_CACHE

template<typename Mesh, typename VT>
class cell_basis
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;

    point_type          cell_bar;
    coordinate_type     cell_h;
    size_t              basis_degree, basis_size;

#ifdef POWER_CACHE
    std::vector<coordinate_type>  power_cache;
#endif

public:
    cell_basis(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree)
    {
        cell_bar        = barycenter(msh, cl);
        cell_h          = diameter(msh, cl);
        basis_degree    = degree;
        basis_size      = (basis_degree+2)*(basis_degree+1)/2;
    }

    Matrix<VT, Dynamic, 1>
    eval_basis(const point_type& pt)
    {
        Matrix<VT, Dynamic, 1> ret = Matrix<VT, Dynamic, 1>::Zero(basis_size);

        auto bx = (pt.x() - cell_bar.x()) / (0.5*cell_h);
        auto by = (pt.y() - cell_bar.y()) / (0.5*cell_h);

#ifdef POWER_CACHE
        if ( power_cache.size() != (basis_degree+1)*2 )
          power_cache.resize( (basis_degree+1)*2);

        power_cache[0] = 1.0;
        power_cache[1] = 1.0;
        for (size_t i = 1; i <= basis_degree; i++)
        {
            power_cache[2*i]    = iexp_pow(bx, i);
            power_cache[2*i+1]  = iexp_pow(by, i);
        }
#endif

        size_t pos = 0;
        for (size_t k = 0; k <= basis_degree; k++)
        {
            for (size_t i = 0; i <= k; i++)
            {
                auto pow_x = k-i;
                auto pow_y = i;
#ifdef POWER_CACHE
                auto bv = power_cache[2*pow_x] * power_cache[2*pow_y+1];
#else
                auto bv = iexp_pow(bx, pow_x) * iexp_pow(by, pow_y);
#endif
                ret(pos++) = bv;
            }
        }

        assert(pos == basis_size);

        return ret;
    }

    Matrix<VT, Dynamic, 2>
    eval_gradients(const point_type& pt)
    {
        Matrix<VT, Dynamic, 2> ret = Matrix<VT, Dynamic, 2>::Zero(basis_size, 2);

        auto bx = (pt.x() - cell_bar.x()) / (0.5*cell_h);
        auto by = (pt.y() - cell_bar.y()) / (0.5*cell_h);
        auto ih = 2.0/cell_h;

#ifdef POWER_CACHE
        if ( power_cache.size() != (basis_degree+1)*2 )
          power_cache.resize( (basis_degree+1)*2);

        power_cache[0] = 1.0;
        power_cache[1] = 1.0;
        for (size_t i = 1; i <= basis_degree; i++)
        {
            power_cache[2*i]    = iexp_pow(bx, i);
            power_cache[2*i+1]  = iexp_pow(by, i);
        }
#endif

        size_t pos = 0;
        for (size_t k = 0; k <= basis_degree; k++)
        {
            for (size_t i = 0; i <= k; i++)
            {
                auto pow_x = k-i;
                auto pow_y = i;
#ifdef POWER_CACHE
                auto px = power_cache[2*pow_x];
                auto py = power_cache[2*pow_y+1];
                auto dx = (pow_x == 0) ? 0 : pow_x*ih*power_cache[2*(pow_x-1)];
                auto dy = (pow_y == 0) ? 0 : pow_y*ih*power_cache[2*(pow_y-1)+1];
#else
                auto px = iexp_pow(bx, pow_x);
                auto py = iexp_pow(by, pow_y);
                auto dx = (pow_x == 0) ? 0 : pow_x*ih*iexp_pow(bx, pow_x-1);
                auto dy = (pow_y == 0) ? 0 : pow_y*ih*iexp_pow(by, pow_y-1);
#endif
                ret(pos,0) = dx*py;
                ret(pos,1) = px*dy;
                pos++;
            }
        }

        assert(pos == basis_size);

        return ret;
    }

    size_t size() const
    {
        return basis_size;
    }

    size_t degree() const
    {
        return basis_degree;
    }
    
    static size_t size(size_t degree)
    {
        return (degree+2)*(degree+1)/2;
    }
};

#if 0
template<template<typename, typename> class Basis, typename CT, typename VT>
class trial_function
{
    const Basis<CT, VT>& basis;

public:
    trial_function(const Basis<CT, VT>& b)
        : basis(b)
    {}
};

template<template<typename, typename> class Basis, typename CT, typename VT>
auto trial(const Basis<CT, VT>& b)
{
    return trial_function<Basis, CT, VT>(b);
}

template<template<typename, typename> class Basis, typename CT, typename VT>
class test_function
{
    const Basis<CT, VT>& basis;

public:
    test_function(const Basis<CT, VT>& b)
        : basis(b)
    {}
};

template<template<typename, typename> class Basis, typename CT, typename VT>
auto test(const Basis<CT, VT>& b)
{
    return test_function<Basis, CT, VT>(b);
}

template<template <typename, typename> class BasisL, template <typename, typename> class BasisR, typename CT, typename VT>
int
operator,(const trial_function<BasisL,CT,VT> trial, const test_function<BasisR,CT,VT>& test)
{
    std::cout << "\\o/  \\o/  \\o/" << std::endl;
    return 42;
}
#endif

template<typename Mesh, typename VT>
class face_basis
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;

    point_type          face_bar;
    point_type          base;
    coordinate_type     face_h;
    size_t              basis_degree, basis_size;

public:
    face_basis(const Mesh& msh, const typename Mesh::face_type& fc, size_t degree)
    {
        face_bar        = barycenter(msh, fc);
        face_h          = diameter(msh, fc);
        basis_degree    = degree;
        basis_size      = degree+1;

        auto pts = points(msh, fc);
        base = face_bar - pts[0];
    }

    Matrix<VT, Dynamic, 1>
    eval_basis(const point_type& pt)
    {
        Matrix<VT, Dynamic, 1> ret = Matrix<VT, Dynamic, 1>::Zero(basis_size);

        auto v = base.to_vector();
        auto t = (pt - face_bar).to_vector();
        auto dot = v.dot(t);
        auto ep = 4.0*dot/(face_h*face_h);

        coordinate_type coeff = sqrt(face_h / 2.0);

        ret(0) = sqrt(1.0 / 2.0) / coeff;
        if( basis_degree == 0)
            return ret;

        ret(1) = ep * sqrt(3.0 / 2.0) / coeff;
        if( basis_degree == 1)
            return ret;

        ret(2) = (3*ep*ep - 1) * sqrt(5.0/8.0) / coeff;
        if( basis_degree == 2)
            return ret;

        ret(3) = (5*ep*ep*ep - 3*ep) * sqrt(7.0 / 8.0) / coeff;
        if( basis_degree == 3)
            return ret;

        throw std::logic_error("bases : we shouldn't be here");
    }

    size_t size() const
    {
        return basis_size;
    }

    size_t degree() const
    {
        return basis_degree;
    }

    static size_t size(size_t degree)
    {
        return degree+1;
    }
};



////////////////////  VECTOR BASIS  /////////////////////////

template<typename Mesh, typename VT>
class vector_cell_basis
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;
    typedef Matrix<VT, 2, 2>                gradient_type;
    typedef Matrix<VT, Dynamic, 2>          function_type;

    point_type          cell_bar;
    coordinate_type     cell_h;
    size_t              basis_degree, basis_size;

    cell_basis<Mesh,VT>          scalar_basis;
    
#ifdef POWER_CACHE
    std::vector<coordinate_type>  power_cache;
#endif

public:
    vector_cell_basis(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree) :
        scalar_basis(msh, cl, degree)
    {
        cell_bar        = barycenter(msh, cl);
        cell_h          = diameter(msh, cl);
        basis_degree    = degree;
        basis_size      = 2*(basis_degree+2)*(basis_degree+1)/2;
    }

    Matrix<VT, Dynamic, 2>
    eval_basis(const point_type& pt)
    {
        Matrix<VT, Dynamic, 2> ret = Matrix<VT, Dynamic, 2>::Zero(basis_size, 2);
        // Matrix<VT, Dynamic, 2> ret;
        
        const auto phi = scalar_basis.eval_basis(pt);

        
        for(size_t i = 0; i < scalar_basis.size(); i++)
        {
            ret(2 * i, 0)     = phi(i);
            ret(2 * i + 1, 1) = phi(i);
        }
        
        assert(2 * scalar_basis.size() == basis_size);

        return ret;
    }

    std::vector<gradient_type>
    eval_gradients(const point_type& pt)
    {
        std::vector<gradient_type> ret;
        ret.reserve(basis_size);

        const function_type dphi = scalar_basis.eval_gradients(pt);

        for (size_t i = 0; i < scalar_basis.size(); i++)
        {
            const Matrix<VT, 1, 2> dphi_i = dphi.row(i);
            gradient_type                   g;

            g        = gradient_type::Zero();
            g.row(0) = dphi_i;
            ret.push_back(g);

            g        = gradient_type::Zero();
            g.row(1) = dphi_i;
            ret.push_back(g);
        }
        
        
        assert(ret.size() == basis_size);

        return ret;
    }

    size_t size() const
    {
        return basis_size;
    }

    size_t degree() const
    {
        return basis_degree;
    }

    static size_t size(size_t degree)
    {
        return 2*(degree+2)*(degree+1)/2;
    }
};




/////////

template<typename Mesh, typename VT>
class vector_face_basis
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;

    point_type          face_bar;
    point_type          base;
    coordinate_type     face_h;
    size_t              basis_degree, basis_size;


    face_basis<Mesh,VT>          scalar_basis;

public:
    vector_face_basis(const Mesh& msh, const typename Mesh::face_type& fc, size_t degree) :
        scalar_basis(msh, fc, degree)
    {
        face_bar        = barycenter(msh, fc);
        face_h          = diameter(msh, fc);
        basis_degree    = degree;
        basis_size      = 2*(degree+1);

        auto pts = points(msh, fc);
        base = face_bar - pts[0];
    }

    Matrix<VT, Dynamic, 2>
    eval_basis(const point_type& pt)
    {
        Matrix<VT, Dynamic, 2> ret = Matrix<VT, Dynamic, 2>::Zero(basis_size, 2);

        const auto psi = scalar_basis.eval_basis(pt);

        for(size_t i = 0; i < scalar_basis.size(); i++)
        {
            ret(2 * i, 0)     = psi(i);
            ret(2 * i + 1, 1) = psi(i);
        }

        assert(2 * scalar_basis.size() == basis_size);

        return ret;
    }

    size_t size() const
    {
        return basis_size;
    }

    size_t degree() const
    {
        return basis_degree;
    }

    static size_t size(size_t degree)
    {
        return 2*(degree+1);
    }
};


////////////////////  MATRIX BASIS  /////////////////////////

template<typename Mesh, typename VT>
class matrix_cell_basis
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;
    typedef Matrix<VT, 2, 2>                matrix_type;
    typedef Matrix<VT, Dynamic, 2>          function_type;

    point_type          cell_bar;
    coordinate_type     cell_h;
    size_t              basis_degree, basis_size;

    cell_basis<Mesh,VT>          scalar_basis;

#ifdef POWER_CACHE
    std::vector<coordinate_type>  power_cache;
#endif

public:
    matrix_cell_basis(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree) :
        scalar_basis(msh, cl, degree)
    {
        cell_bar        = barycenter(msh, cl);
        cell_h          = diameter(msh, cl);
        basis_degree    = degree;
        basis_size      = 2*2*(basis_degree+2)*(basis_degree+1)/2;
    }

    std::vector<matrix_type>
    eval_basis(const point_type& pt)
    {
        std::vector<matrix_type> ret;
        ret.reserve(basis_size);

        const auto phi = scalar_basis.eval_basis(pt);



        for (int k = 0; k < scalar_basis.size(); k++)
        {
            matrix_type fc;

            for (int j = 0; j < 2; j++)
            {
                for (int i = 0; i < 2; i++)
                {
                    fc       = matrix_type::Zero();
                    fc(i, j) = phi(k);
                    ret.push_back(fc);
                }
            }
        }

        assert(2 * 2 * scalar_basis.size() == basis_size);

        return ret;
    }

    size_t size() const
    {
        return basis_size;
    }

    size_t degree() const
    {
        return basis_degree;
    }

    static size_t size(size_t degree)
    {
        return 2*2*(degree+2)*(degree+1)/2;
    }
};


////////////   SYM MATRIX BASIS


template<typename Mesh, typename VT>
class sym_matrix_cell_basis
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;
    typedef Matrix<VT, 2, 2>                matrix_type;
    typedef Matrix<VT, Dynamic, 2>          function_type;

    point_type          cell_bar;
    coordinate_type     cell_h;
    size_t              basis_degree, basis_size;

    cell_basis<Mesh,VT>          scalar_basis;

#ifdef POWER_CACHE
    std::vector<coordinate_type>  power_cache;
#endif

public:
    sym_matrix_cell_basis(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree) :
        scalar_basis(msh, cl, degree)
    {
        cell_bar        = barycenter(msh, cl);
        cell_h          = diameter(msh, cl);
        basis_degree    = degree;
        basis_size      = 3*(basis_degree+2)*(basis_degree+1)/2;
    }

    std::vector<matrix_type>
    eval_basis(const point_type& pt)
    {
        std::vector<matrix_type> ret;
        ret.reserve(basis_size);

        const auto phi = scalar_basis.eval_basis(pt);



        for (int k = 0; k < scalar_basis.size(); k++)
        {
            matrix_type fc;

            fc       = matrix_type::Zero();
            fc(0, 0) = phi(k);
            ret.push_back(fc);

            fc       = matrix_type::Zero();
            fc(1, 0) = phi(k);
            fc(0, 1) = phi(k);
            ret.push_back(fc);

            fc       = matrix_type::Zero();
            fc(1, 1) = phi(k);
            ret.push_back(fc);
        }

        assert(ret.size() == basis_size);
        assert(3 * scalar_basis.size() == basis_size);

        return ret;
    }

    size_t size() const
    {
        return basis_size;
    }

    size_t degree() const
    {
        return basis_degree;
    }

    static size_t size(size_t degree)
    {
        return 3*(degree+2)*(degree+1)/2;
    }
};

