/*
 *       /\        Matteo Cicuttin (C) 2016, 2017, 2018
 *      /__\       matteo.cicuttin@enpc.fr
 *     /_\/_\      École Nationale des Ponts et Chaussées - CERMICS
 *    /\    /\
 *   /__\  /__\    DISK++, a template library for DIscontinuous SKeletal
 *  /_\/_\/_\/_\   methods.
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
#include <array>
#include <stdexcept>
#include <initializer_list>

#include "Eigen/Dense"

using namespace Eigen;

template<typename T, size_t DIM>
using static_vector = Eigen::Matrix<T, DIM, 1>;

template<typename T, size_t DIM>
class point
{
    static_vector<T, DIM>     m_coords;

public:
    typedef T                                   value_type;
    const static size_t                         dimension = DIM;

    point()
    {
        m_coords = static_vector<T, DIM>::Zero(DIM);
    }

    point(const point& other) : m_coords(other.m_coords) {}

    point operator=(const point& other)
    {
        m_coords = other.m_coords;
        return *this;
    }
    
    bool operator==(const point& other)
    {
        if ( m_coords == other.m_coords )
            return 1 ;
        else
            return 0 ;
    }

    template<typename U = T>
    point(const typename std::enable_if<DIM == 1, U>::type& x)
    {
        m_coords(0) = x;
    }

    template<typename U = T>
    point(const typename std::enable_if<DIM == 2, U>::type& x, const U& y)
    {
        m_coords(0) = x;
        m_coords(1) = y;
    }

    template<typename U = T>
    point(const typename std::enable_if<DIM == 3, U>::type& x, const U& y, const U& z)
    {
        m_coords(0) = x;
        m_coords(1) = y;
        m_coords(2) = z;
    }

    T   at(size_t pos) const
    {
        if (pos >= DIM)
            throw std::out_of_range("access out of range");

        return m_coords(pos);
    }

    T&  at(size_t pos)
    {
        if (pos >= DIM)
            throw std::out_of_range("access out of range");

        return m_coords(pos);
    }

    T       operator[](size_t pos) const { return m_coords(pos); }
    T&      operator[](size_t pos)       { return m_coords(pos); }

    point   operator-() const {
        auto ret = -1.0 * (*this);
        return ret;
    }

    template<typename U = T>
    typename std::enable_if<DIM == 1 || DIM == 2 || DIM == 3, U>::type
    x() const { return m_coords(0); }

    template<typename U = T>
    typename std::enable_if<DIM == 1 || DIM == 2 || DIM == 3, U>::type&
    x() { return m_coords(0); }

    template<typename U = T>
    typename std::enable_if<DIM == 2 || DIM == 3, U>::type
    y() const { return m_coords(1); }

    template<typename U = T>
    typename std::enable_if<DIM == 2 || DIM == 3, U>::type&
    y() { return m_coords(1); }

    template<typename U = T>
    typename std::enable_if<DIM == 3, U>::type
    z() const { return m_coords(2); }

    template<typename U = T>
    typename std::enable_if<DIM == 3, U>::type&
    z() { return m_coords(2); }

    auto to_vector() const
    {
        return m_coords;
    }

    friend point operator+(const point& p1, const point& p2)
    {
        point ret;
        ret.m_coords = p1.m_coords + p2.m_coords;
        return ret;
    }

    friend point operator-(const point& p1, const point& p2)
    {
        point ret;
        ret.m_coords = p1.m_coords - p2.m_coords;
        return ret;
    }

    friend point operator*(const point& p, T scalefactor)
    {
        point ret;
        ret.m_coords = p.m_coords * scalefactor;
        return ret;
    }

    friend point operator*(T scalefactor, const point& p)
    {
        return p * scalefactor;
    }

    friend point operator/(const point& p, T scalefactor)
    {
        point ret;
        ret.m_coords = p.m_coords / scalefactor;
        return ret;
    }
};

template<typename T>
T
det(const point<T,2>& p1, const point<T,2>& p2)
{
    return p1.x() * p2.y() - p1.y() * p2.x();
}

template<typename T, size_t DIM>
std::ostream&
operator<<(std::ostream& os, const point<T, DIM>& pt)
{
    os << "( ";
    for (size_t i = 0; i < DIM; i++)
    {
        os << pt[i];

        if (i < DIM-1)
            os << ", ";
    }
    os << " )";
    return os;
}
