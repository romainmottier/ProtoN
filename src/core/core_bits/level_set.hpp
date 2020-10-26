/*
 *       /\        Guillaume Delay 2018,2019
 *      /__\       guillaume.delay@enpc.fr
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


template<typename T>
class level_set
{
   public:
    virtual T operator()(const point<T,2>& pt) const
    {
    }
    
    virtual Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
    }

    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;

        ret = gradient(pt);
        return ret/ret.norm();
    }
   
};

template<typename T>
struct circle_level_set: public level_set<T>
{
    T radius, alpha, beta;

    circle_level_set(T r, T a, T b)
        : radius(r), alpha(a), beta(b)
    {}
    
    circle_level_set(){}
    
    circle_level_set(const circle_level_set& other){
        radius = other.radius ;
        alpha = other.alpha ;
        beta = other.beta ;
        
    }

    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();

        return (x-alpha)*(x-alpha) + (y-beta)*(y-beta) - radius*radius;
    }

    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret(0) = 2*pt.x() - 2*alpha;
        ret(1) = 2*pt.y() - 2*beta;
        return ret;
    }
};


template<typename T>
struct circle_distance_ls: public level_set<T>
{
    T radius, alpha, beta;
    T eps ;
    circle_distance_ls(T r, T a, T b , T eps)
        : radius(r), alpha(a), beta(b) , eps(eps)
    {}

     circle_distance_ls(){}
    
    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();
        //auto val = sqrt((x-alpha)*(x-alpha) + (y-beta)*(y-beta)) - radius ;
        auto val = (x-alpha)*(x-alpha) + (y-beta)*(y-beta) - radius*radius ;
        return 0.5 * ( 1.0 + tanh(val/(2.0*eps)) ) ;
    }

    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        auto x = pt.x();
        auto y = pt.y();
        T val = 1.0/( sqrt( (x-alpha)*(x-alpha) + (y-beta)*(y-beta) ) ) ;
        ret(0) = val * (x-alpha) ;
        ret(1) = val * (y-beta) ;
        return ret;
    }
};

template<typename T>
struct elliptic_distance_ls: public level_set<T>
{
    T radius_a, radius_b, alpha, beta;
    T eps ;
    elliptic_distance_ls(T r_a, T r_b , T a, T b , T eps)
        : radius_a(r_a), radius_b(r_b), alpha(a), beta(b) , eps(eps)
    {}
    
    elliptic_distance_ls(){}

    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();
        return 1.0/( 1.0 + exp( ( ((x-alpha)*(x-alpha))/(radius_a*radius_a) + ((y-beta)*(y-beta))/(radius_b*radius_b) -1.0 )/eps ) ) ;
    }

    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        auto x = pt.x();
        auto y = pt.y();
        T val = 1.0/( 1.0 + exp( ( ((x-alpha)*(x-alpha))/(radius_a*radius_a) + ((y-beta)*(y-beta))/(radius_b*radius_b) -1.0 )/eps ) ) ;
        T val_grad = 1.0/(4.0*eps)*( 1.0-tanh(val/(2.0*eps))*tanh(val/(2.0*eps) ) ) ;
        ret(0) = val_grad * 2.0 / pow(radius_a,2) * (x-alpha) ;
        ret(1) = val_grad * 2.0 / pow(radius_b,2) * (y-beta) ;
        return ret;
    }
};


template<typename T>
struct circle_level_set_new: public level_set<T>
{
    T radius, alpha, beta;
    T gamma = 1.0/16.0;
    circle_level_set_new(T r, T a, T b , T gamma)
        : radius(r), alpha(a), beta(b) , gamma(gamma)
    {}
    
    circle_level_set_new(){}

    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();

        return tanh( (sqrt( (x-alpha)*(x-alpha) + (y-beta)*(y-beta) ) - radius)/gamma ) ;
    }

    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        auto x = pt.x();
        auto y = pt.y();
        T val = 1.0/(pow( (cosh( (sqrt( (x-alpha)*(x-alpha) + (y-beta)*(y-beta) ) - radius)/gamma)),2) * gamma * sqrt( (x-alpha)*(x-alpha) + (y-beta)*(y-beta) ) );
        ret(0) = val * (x-alpha) ;
        ret(1) = val * (y-beta) ;
       
        return ret;
    }
};



template<typename T>
struct elliptic_level_set: public level_set<T>
{
    T radius_a, radius_b, alpha, beta;

    elliptic_level_set(T r_a, T r_b , T a, T b)
        : radius_a(r_a), radius_b(r_b), alpha(a), beta(b)
    {}
    
    elliptic_level_set(){}
    
    elliptic_level_set(const elliptic_level_set& other){
        radius_a = other.radius_a ;
        radius_b = other.radius_b ;
        alpha = other.alpha ;
        beta = other.beta ;
        
    }

    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();

        return radius_b*radius_b*(x-alpha)*(x-alpha) + radius_a*radius_a*(y-beta)*(y-beta) - radius_a*radius_a*radius_b*radius_b ;
    }

    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret(0) = 2*radius_b*radius_b*(pt.x() - alpha);
        ret(1) = 2*radius_a*radius_a*(pt.y() - beta);
        return ret;
    }
};

template<typename T>
struct elliptic_level_set_new: public level_set<T>
{
    T radius_a, radius_b, alpha, beta;
    T gamma = 1.0/16.0;
    elliptic_level_set_new(T r_a, T r_b , T a, T b , T gamma)
        : radius_a(r_a), radius_b(r_b), alpha(a), beta(b) , gamma(gamma)
    {}

    elliptic_level_set_new(){}
    
    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();

        return tanh( ( (x-alpha)*(x-alpha)/(radius_a*radius_a) + (y-beta)*(y-beta)/(radius_b*radius_b)  - 1.0 )/gamma ) ;
    }

    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        auto x = pt.x();
        auto y = pt.y();
        T val = 1.0/(pow( ( ( (x-alpha)*(x-alpha)/(radius_a*radius_a) + (y-beta)*(y-beta)/(radius_b*radius_b)  - 1.0 )/gamma ),2) * gamma );
        ret(0) = 2.0 * val / ( radius_a * radius_a ) * (x-alpha) ;
        ret(1) = 2.0 * val / ( radius_b * radius_b ) * (y-beta) ;
       
        return ret;
    }
};



template<typename T>
class line_level_set: public level_set<T>
{
    T cut_y;

    line_level_set(T cy)
        : cut_y(cy)
    {}

    line_level_set(){}
    
    line_level_set(const line_level_set& other){
        cut_y = other.cut_y ;
        
    }
    
    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();

        return y - cut_y;
    }

    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret(0) = 0;
        ret(1) = 1;
        return ret;
    }
};



template<typename T>
class square_level_set: public level_set<T>
{
    public:
    T y_top, y_bot, x_left, x_right;

    square_level_set(T yt, T yb, T xl, T xr)
        : y_top(yt), y_bot(yb), x_left(xl), x_right(xr)
    {}
    
    square_level_set(){}

    square_level_set(const square_level_set& other){
        y_top = other.y_top ;
        y_bot = other.y_bot ;
        x_left = other.x_left ;
        x_right = other.x_right ;
        
    }
    
    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();

        T in = 1;
        if(x > x_left && x < x_right && y > y_bot && y < y_top)
            in = 1;
        else
            in = -1;

        T dist_x = std::min( abs(x-x_left), abs(x-x_right));
        T dist_y = std::min( abs(y-y_bot), abs(y-y_top));

        
        return - in * std::min(dist_x , dist_y);
    }

    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        

        auto x = pt.x();
        auto y = pt.y();

        T dist = abs(x - x_left);
        ret(0) = -1;
        ret(1) = 0;
        
        if(abs(x - x_right) < dist )
        {
            dist = abs(x - x_right);
            ret(0) = 1;
            ret(1) = 0;
        }
        if(abs(y - y_bot) < dist )
        {
            dist = abs(y - y_bot);
            ret(0) = 0;
            ret(1) = -1;
        }
        if(abs(y - y_top) < dist)
        {
            ret(0) = 0;
            ret(1) = 1;
        }
        
        return ret;
    }
};



template<typename T>
struct flower_level_set: public level_set<T>
{
    T radius, alpha, beta, a;
    size_t N;

    flower_level_set(T r, T al, T b, size_t N_, T a_)
        : radius(r), alpha(al), beta(b), N(N_), a(a_)
    {}

    flower_level_set(){}
    
    flower_level_set(const flower_level_set& other){
        radius = other.radius ;
        alpha = other.alpha ;
        beta = other.beta ;
        a = other.a ;
        
    }
    
    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();

        T theta;
        if(x == alpha && y < beta)
            theta = - M_PI / 2.0;
        else if(x == alpha && y >= beta)
            theta = M_PI / 2.0;
        else
            theta = atan((y-beta)/(x-alpha));

        if(x < alpha)
            theta = theta + M_PI;

        return (x-alpha)*(x-alpha) + (y-beta)*(y-beta) - radius*radius
            - a * std::cos(N*theta);
    }

    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        auto X = pt.x() - alpha;
        auto Y = pt.y() - beta;

        T theta;
        if(X == 0 && Y < 0)
            theta = - M_PI / 2.0;
        else if(X == 0 && Y >= 0)
            theta = M_PI / 2.0;
        else
            theta = atan( Y / X );

        if(pt.x() < alpha)
            theta = theta + M_PI;
        
        ret(0) = 2*X - a * N * std::sin(N * theta) * Y / (X*X + Y*Y);
        ret(1) = 2*Y + a * N * std::sin(N * theta) * X / (X*X + Y*Y);
        return ret;
    }
};



/*

template<typename T, typename Mesh>
bool
pt_in_cell(const Mesh& msh, const point<T,2>& point_to_find, const typename Mesh::cell_type& cl)
{
    bool ret = 0;
    auto pts =points(msh,cl);
    for (size_t i = 0; i < pts.size(); i++)
    {
        if( pts[i].x()>=point_to_find.x() && pts[i].y()>=point_to_find.y() )
            ret = 1;
    }
    return ret;

}
*/

/*
// Ho due possibilità.. se uso values_bis1 uso una vector notation,
//                      se uso values_bis uso la matrix notation -> USO QUESTA PER ORA, MA I TEST CHE HO FATTO PER N,M=32, 64 RILEVANO CHE SONO PRATICAMENTE LO STESSO

template<typename T, typename Mesh ,typename Fonction >
struct projected_level_set: public level_set<T>
{
    std::vector< T > values;
    Eigen::Matrix<T, Dynamic, Dynamic> values_bis; // MATRIX NOTATION
    //Eigen::Matrix<T, Dynamic, 1> values_bis1; // VECTOR NOTATION
    Eigen::Matrix<T, Dynamic, 1> vertices; // saving level_set on vertices mesh
    size_t degree_FEM;
    size_t number_elements;
    Mesh msh;
    size_t      Nx, Ny;
    size_t  last_row_init, last_row_end, number_faces_one_row;
  //  size_t counter_cell , counter_face, num_cell_row;
    
    projected_level_set(const Fonction & level_set, const Mesh & msh, size_t degree_k , const mesh_init_params<T>& params)
        : number_elements((degree_k+1)*(degree_k+1)), msh(msh),degree_FEM(degree_k),Nx(params.Nx),Ny(params.Ny)
    //, values_bis( Eigen::Matrix<T, Dynamic, 1>::Zero(number_elements*msh.cells.size(), 1) )
    {
        
        last_row_init = Ny*(2*Nx+1); // There are 2 faces for each row of cells + Ny
        last_row_end = last_row_init + Nx-1;
        number_faces_one_row = 2*Nx+1; // for each cell I count only the low and sx faces, respectevely 0-1 2-3 4-5 6-7 8 + the last on the right boundary
        vertices = Eigen::Matrix<T, Dynamic, 1>::Zero( ((Nx+1)*(Ny+1)), 1 );
        // MAYBE I COULD DO LIKE BEFORE -> (accelerate??)
        //#ifdef NODES
        
        values_bis= Eigen::Matrix<T, Dynamic, Dynamic>::Zero(number_elements, msh.cells.size());                  // MATRIX NOTATION
        
        //values_bis1= Eigen::Matrix<T, Dynamic, 1>::Zero(number_elements*msh.cells.size(), 1 );      // VECTOR NOTATION
        
        //std::cout<<"Number of cells "<<msh.cells.size()<<std::endl;
        
        // MATRIX NOTATION
        // std::cout<<"Dimension of the basis "<<values_bis.size()<<std::endl;
        // VECTOR NOTATION
        // std::cout<<"Dimension of the basis "<<values_bis1.size()<<std::endl;
         size_t i_global = 0 , i_local=0 , i_vertex=0;
        for(auto& cl:msh.cells)
        {
            auto qps = equidistriduted_nodes<T,Mesh>(msh, cl, degree_FEM);
            i_local = 0;
            for ( const auto & qp : qps)
            {
                values.push_back( level_set(qp) );
                values_bis(i_local,i_global) = level_set(qp) ;  // MATRIX NOTATION
                //values_bis1(i_local+i_global) = level_set(qp) ; // VECTOR NOTATION
                i_vertex = i_global+floor(i_global/Nx);
                if( i_local==0 )
                    vertices(i_vertex) = level_set(qp) ;
                
                if( i_local==1 )
                    vertices(i_vertex+1) = level_set(qp) ;
                
                if( i_local==(degree_FEM+2) )
                    vertices(i_vertex+Nx+2) = level_set(qp) ;
                
                if( i_local==(degree_FEM+1) )
                    vertices(i_vertex+Nx+1) = level_set(qp) ;
                i_local++;
            }
             i_global++;  // MATRIX NOTATION
           //  i_global+=number_elements;       // VECTOR NOTATION
        }
       //#endif
    }
    
    
    T operator()( const typename Mesh::node_type& node ) const
    {
        // Optimised to check the value of the level set only in the vertices
        // std::cout<<"Value in vertices "<<vertices(node.ptid)<<", at position "<<node.ptid<<std::endl;
        return vertices(node.ptid);
        
    }
    
    

    T operator()(const point<T,2>& pt) const
    {
        size_t counter=0;
        for( const auto& cl:msh.cells)
        {
            if( pt_in_cell<T,Mesh>(msh,pt,cl) )
            {
                //cell_basis_Qk<Mesh,T> cb(msh, cl, degree_FEM);
                cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
                T tmp=0;
               
                for(auto i = 0; i<number_elements ; i++)
                {
                    tmp += (values.at(i+counter))*(cb.eval_basis(pt))[i];
                }
                return tmp;
            }
            counter+=number_elements;
        }
        std::cout<<"IF HERE, THERE IS A PROBLEM IN projected_level_set::operator()!!!"<<std::endl;
        return 1e10; //to check if doesn't enter in the loop
    }

    
    T operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        
        // MATRIX NOTATION
        size_t counter = offset(msh,cl);
        //std::cout<<"Value of offset "<<counter<<std::endl;
        //cell_basis_Qk<Mesh,T> cb(msh, cl, degree_FEM);
        cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (values_bis.block(0,counter,number_elements,1)).col(0);
        T tmp = values_cell.dot( cb.eval_basis(pt) );
        return tmp;
        
        // VECTOR NOTATION
        
        //size_t counter = offset(msh,cl)*number_elements;
        //cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
        //auto values_cell = (values_bis1.segment(counter,number_elements));
        //T tmp = values_cell.dot( cb.eval_basis(pt) );
        //return tmp;
        
        
    }
    
 
    T operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::face_type& fc ) const
       {
           
           // MATRIX NOTATION
           auto counter_face = offset(msh,fc);
           size_t counter_cell;
           // da fc devo trovare la cella in cui sono per la base
           //std::cout<<"Face number "<<counter_face<<std::endl;

           // ATTENTION, ALL THIS WORKS IN STRUCTURED QUADRANGULAR MESHES
           
           // Check if I am in the last row, upper faces (ordered differently)
           if(counter_face>=last_row_init && counter_face<=last_row_end)
           {
               counter_cell = (Ny-1)*Nx + counter_face%(last_row_init);
           }
           else
           {
            // Find in what row the face is
             auto  num_cell_row = floor(counter_face/(number_faces_one_row));
               if ( counter_face!= ( (2*Nx)*(num_cell_row+1)+num_cell_row ) )
               {
                // Faces not on the right boudary, to know in what cell are, it is sufficient to analyse the low and left face of each quadrangular cell.
                   counter_cell = floor( (counter_face-num_cell_row)/2.0 );
               }
               else
               {
                // Face on the right boudary,
                   counter_cell = ( num_cell_row+1 )*Nx -1;
               }
           
           }
           //std::cout<<"Face->Cell number "<<counter_cell<<std::endl;
           auto cl = msh.cells.at(counter_cell);
           //cell_basis_Qk<Mesh,T> cb(msh, cl, degree_FEM);
           cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
           auto values_cell = (values_bis.block(0,counter_cell,number_elements,1)).col(0);
           T tmp = values_cell.dot( cb.eval_basis(pt) );
           return tmp;

           
       }
    

    


    
    
    
    
    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        size_t counter=0;
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        for( const auto& cl:msh.cells)
        {
            if(pt_in_cell<T,Mesh>(msh,pt,cl))
            {
                //cell_basis_Qk<Mesh,T> cb(msh, cl, degree_FEM);
                cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
                auto grad_eval =  cb.eval_gradients(pt);
                for(auto i = 0; i<number_elements ; i++)
                {
                    ret(0) += values.at(i+counter)*grad_eval(i,0);
                    ret(1) += values.at(i+counter)*grad_eval(i,1);
                }
                return ret;
            }
            counter+=number_elements;

        }
        std::cout<<"Se compare questo problema in gradient()"<<std::endl;
        ret(0)+=1e10;
        ret(1)+=1e10;
        return ret; //to check if doesn't enter in the loop

    }
};

*/
