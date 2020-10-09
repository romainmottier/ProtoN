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



template<typename T>
std::vector<point<T,1> >
reference_nodes(size_t );


template<typename T>
std::vector<point<T,1> >
reference_nodes_ordered(size_t degree)
{
    auto comp_degree = degree + 1;
    size_t reqd_nodes = comp_degree;

    std::vector<point<T,1> > ret;
    ret.reserve(reqd_nodes);

    point<T,1>  qp , qp_1;
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
            qp_1 = point<T,1>({ 0.0 });
            ret.push_back( qp_1 );
            ret.push_back( qp );

         //   qp = point<T,1>({ 0.0 });
         //   ret.push_back( qp );
            return ret;

        default:

            delta_x = 2.0/degree;
            a1 = 1.0;
            while (a1>1e-10) {
                qp = point<T,1>({ a1 });
                ret.push_back( -qp );
                ret.push_back( qp );
                a1-=delta_x;

            }
            if(a1<1e-10 && a1>-1e-10)
            {
                qp = point<T,1>({0.0});
                ret.push_back( qp );
            }
            std::sort(ret.begin()+2, ret.end() ,[](point<T,1>  a, point<T,1>  b) {
                return a.x() < b.x();
            } );
            return ret;
    }
    return ret;
}


// Lagrangian basis b_kl(x,y) = b_k(x)*b_l(y) over a set of equidistributed 2-dimensional nodes (3D CASE NOT YET IMPLEMENTED)

template<typename T,typename Mesh>
std::vector< point<T,2> >
equidistriduted_nodes_ordered_bis(const Mesh& msh,
          const typename Mesh::cell_type& cl,
          size_t degree)
{
    typedef typename Mesh::point_type    point_type;

    auto qps = reference_nodes_ordered<T>(degree); //Ordering:  0 2 3 4 5 6 ... 1
   
    // 3 -  12 - 11 - 10 - 2
    // 13 - 22 - 21 - 20 - 9
    // 14 - 23 - 24 - 19 - 8
    // 15 - 16 - 17 - 18 - 7
    // 0 -  4 -  5  - 6  - 1

    auto pts = points(msh, cl);
    
    //auto v0 = pts[1] - pts[0];
    //auto v1 = pts[2] - pts[1];
    //auto v2 = pts[3] - pts[2];
    //auto v3 = pts[3] - pts[0];

    std::vector< point<T,2> > ret((degree+1)*(degree+1));

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

    /// ADDING VERTICES:
    
    // (-1,-1)
    auto qp_x = qps[0];
    auto qp_y = qps[0];
    auto xi = qp_x.x();
    auto eta = qp_y.x();
    auto px = P(xi, eta);
    auto py = Q(xi, eta);
    ret[0] = ( point_type(px, py) );
    // (1,-1)
    qp_x = qps[1];
    qp_y = qps[0];
    xi = qp_x.x();
    eta = qp_y.x();
    px = P(xi, eta);
    py = Q(xi, eta);
    ret[1] = ( point_type(px, py) );
    // (1,1)
    qp_x = qps[1];
    qp_y = qps[1];
    xi = qp_x.x();
    eta = qp_y.x();
    px = P(xi, eta);
    py = Q(xi, eta);
    ret[2] = ( point_type(px, py) );
    // (-1,1)
    qp_x = qps[0];
    qp_y = qps[1];
    xi = qp_x.x();
    eta = qp_y.x();
    px = P(xi, eta);
    py = Q(xi, eta);
    ret[3] = ( point_type(px, py) );
    
    /// Counter for each side of the 2D - square : starting to count from a vertice, the position where I save the point in ret is (degree -1)*j , with j the j-esima face.
    
    int  count0 = 4 , count1 = 4 + degree - 1 , count2 = 4 + 2*(degree - 1), count3 = 4 + 3*(degree - 1)  ;
    
    int count_bis0 = 4*degree ; // counter initialisation (USELESS)
    int j_max = floor((degree-1)/2) ; // number of internal layour of points
  //  std::cout<<"j max "<<j_max<<std::endl;
    int pos_right = 100; // initial point from the right from bottom -> up
    int pos_left = 100; // initial point from the left from left -> right
   // size_t i_min = 0;
 //   std::cout<<"inizia ciclo"<<std::endl;
    for (int j = 0 ; j <= j_max ; j++ ) { // for each layout of point
       // bool i_loop = FALSE;
        for(int i = std::max(0,j-1) ; i < degree -1 - j ; i++ ) // I move from into the points over a side of each layout
        {
         
            if( i == std::max(0,j-1) && j > 0) // vertices
            {
                // different pos_left depending on the layout. Especially this rules the y starting point
                if(j == 0)
                    pos_left = 0;
                else if(j == 1)
                    pos_left = 2;
                else
                    pos_left = 2+(j-1);
                           
                //   std::cout<<"pos_left "<<pos_left<<std::endl;
                qp_x = qps[2+i];
                qp_y = qps[pos_left]; //qps[0 + 2*j];
                //   std::cout<<"qp_y "<<qp_y<<std::endl;
                xi = qp_x.x();
                eta = qp_y.x();
                px = P(xi, eta);
                py = Q(xi, eta);
                // Here change counters. No more count0, count1 etc.. Just count_bis0, to have the vertices enumerate one after the other.
                count_bis0 = count0; // counter_bis0 re-initialisation for each loop (first loop)
                //std::cout<<"count bis 0 "<<count_bis0<<std::endl;
                ret[count_bis0] = ( point_type(px, py) ); // adding point bottom layout
                //std::cout<<"count0 is "<<count0<<" , pt0"<<point_type(px, py)<<std::endl;
                //std::cout<<ret[count0]<<std::endl;
                count_bis0++;
                
                if(j==0)
                    pos_right = 1;
                else
                    pos_right = degree + 1 - j;
                // size_t pos = (1 + j*(degree-1));
                // if(pos>degree)
                //     pos -= j*degree ;
                           
                qp_x =  qps[pos_right];
                qp_y = qps[2 + i];
                xi = qp_x.x();
                eta = qp_y.x();
                px = P(xi, eta);
                py = Q(xi, eta);
                ret[count_bis0] = ( point_type(px, py) ); // adding point right layout in count_bis0 pos.
                //std::cout<<"count1 is "<<count1<<" , pt1"<<point_type(px, py)<<std::endl;
                //std::cout<<"count_bis0 is "<<count_bis0<<std::endl;
                //std::cout<<ret[count_bis0]<<std::endl;
                count_bis0++;
                
                qp_x = qps[degree - i];
                qp_y =  qps[pos_right] ;
                xi = qp_x.x();
                eta = qp_y.x();
                px = P(xi, eta);
                py = Q(xi, eta);
                ret[count_bis0] = ( point_type(px, py) ); // adding point top layout in count_bis0 pos.
                //std::cout<<"count_bis0 is "<<count_bis0<<" and pt "<<ret[count_bis0]<<std::endl;
                //std::cout<<"count2 is "<<count2<<" , pt2"<<point_type(px, py)<<std::endl;
                
                count_bis0++;
                qp_x = qps[pos_left];
                qp_y = qps[degree - i];
                xi = qp_x.x();
                eta = qp_y.x();
                px = P(xi, eta);
                py = Q(xi, eta);
                ret[count_bis0] = ( point_type(px, py) ); // adding point left layout in count_bis0 pos.
                //std::cout<<"count_bis0 is "<<count_bis0<<" and pt "<<ret[count_bis0]<<std::endl;
                //std::cout<<"count3 is "<<count3<<" , pt3"<<point_type(px, py)<<std::endl;
                count_bis0++;
                
                // Uploading counters according to count_bis0. I start enumerate "face" unknown for the current layout (IN ELSE HERE BELOW) from the last position count_bis0.
                count0 = count_bis0 ;
                count1 = count0 + std::ceil((degree - 2.0*j)/2.0); //  count0 + (degree - 2); it was
                count2 = count1 + std::ceil((degree - 2.0*j)/2.0);
                count3 = count2 + std::ceil((degree - 2.0*j)/2.0);

            }
            else // NOT vertices -> node in the sides of each layout
            {
                
                //   std::cout<<"i "<<i<<" j "<<j<<std::endl;
                // vertical position where starting for each bottom side layout.
                if(j == 0)
                    pos_left = 0;
                else if(j == 1)
                    pos_left = 2;
                else
                    pos_left = 2+(j-1);
                
                //   std::cout<<"pos_left "<<pos_left<<std::endl;
                qp_x = qps[2+i];
                qp_y = qps[pos_left]; //qps[0 + 2*j];
                //   std::cout<<"qp_y "<<qp_y<<std::endl;
                xi = qp_x.x();
                eta = qp_y.x();
                px = P(xi, eta);
                py = Q(xi, eta);
                ret[count0] = ( point_type(px, py) ); // from left, node of each bottom side layout
                //std::cout<<"count0 is "<<count0<<" , pt0"<<point_type(px, py)<<std::endl;
                //std::cout<<ret[count0]<<std::endl;
                count0++;
                // x-position where to start to increase to get points for each right side layout of points
                if(j==0)
                    pos_right = 1;
                else
                    pos_right = degree + 1 - j;
                // size_t pos = (1 + j*(degree-1));
                // if(pos>degree)
                //     pos -= j*degree ;
            
                qp_x =  qps[pos_right];
                qp_y = qps[2 + i];
                xi = qp_x.x();
                eta = qp_y.x();
                px = P(xi, eta);
                py = Q(xi, eta);
                ret[count1] = ( point_type(px, py) ); // using count1 that allows correct enumeration
                //std::cout<<"count1 is "<<count1<<" , pt1"<<point_type(px, py)<<std::endl;
                //std::cout<<ret[count1]<<std::endl;
                count1++; // count1 bigger than count0. Count1 just enumerate right faces' layout
                
                qp_x = qps[degree - i];
                qp_y =  qps[pos_right] ;
                xi = qp_x.x();
                eta = qp_y.x();
                px = P(xi, eta);
                py = Q(xi, eta);
                ret[count2] = ( point_type(px, py) );
                //std::cout<<"count2 is "<<count2<<" , pt2"<<point_type(px, py)<<std::endl;
                count2++; // count2 just enumerate top faces' layout
                
                qp_x = qps[pos_left];
                qp_y = qps[degree - i];
                xi = qp_x.x();
                eta = qp_y.x();
                px = P(xi, eta);
                py = Q(xi, eta);
                ret[count3] = ( point_type(px, py) );
                //std::cout<<"count3 is "<<count3<<" , pt3"<<point_type(px, py)<<std::endl;
                count3++; // count3 just enumerate left faces' layout
            
                
            }
        }
        // Uploading for the next layout.
        count0 = count3 ;
        count1 = count0 + (degree - 2*(j+1)); //  count0 + (degree - 2); it was
        count2 = count1 + (degree - 2*(j+1));
        count3 = count2 + (degree - 2*(j+1));
        //}
    }
    
    /// Middle point --> the internal node is treated a part of the others. Just even degrees have it.
    if( degree % 2 == 0)
    {
        qp_x = qps[degree - floor((degree-1)/2)];
        qp_y = qps[degree - floor((degree-1)/2)];
        xi = qp_x.x();
        eta = qp_y.x();
        px = P(xi, eta);
        py = Q(xi, eta);
        ret[count0] = ( point_type(px, py) );
        //std::cout<<"counto MIDDLE is "<<count0<<" , pt3"<<point_type(px, py)<<std::endl;
    }
    return ret;
}

template<typename Mesh, typename T >
class cell_basis_Lagrangian_ordered
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;

    point_type          cell_bar;
    coordinate_type     cell_h;
    size_t              basis_degree, basis_size;
    std::vector<point<T, 2> >          nodes;
    std::vector<size_t>         row_indeces , col_indeces ;
    
public:
    cell_basis_Lagrangian_ordered(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree)
    {
        //nodes           = equidistriduted_nodes<T,Mesh>(msh, cl, degree);
        // nodes           = equidistriduted_nodes_ordered<T,Mesh>(msh, cl, degree);
        nodes           = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree);
        basis_degree    = degree;
        basis_size      = (basis_degree+1)*(basis_degree+1);
        set_col_row_indeces();
    
    }
    
    void set_col_row_indeces()
    {
        //std::cout<<"SONO QUA NEL SET INDECES "<<row_indeces.size()<<" and "<<col_indeces.size()<<std::endl;
        // It could be cool to just save once and not call each time. To do that or call the  variable Static(?) or give it just as external fx and initiliaze once.
        if(row_indeces.size()==0 && col_indeces.size()==0 )
        {
            row_indeces.push_back(0);
            row_indeces.push_back(1);
            col_indeces.push_back(0);
            col_indeces.push_back(3);
            for(size_t i = 0 ; i<basis_degree-1 ; i++){
                row_indeces.push_back(4+i);
                col_indeces.push_back(3*basis_degree+1);
            }
            
        }
    }
        
    
    /*
    cell_basis_Lagrangian(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree, const std::vector<size_t>& indeces)
    {
        
        nodes = equidistriduted_nodes_subcell<T,Mesh>(msh, cl, degree, indeces);
        basis_degree    = degree;
        basis_size      = (basis_degree+1)*(basis_degree+1);
        
    }
    */

    Matrix<T, Dynamic, 1>
    eval_basis(const point_type& pt)
    {
        Matrix<T, Dynamic, 1> rety = Matrix<T, Dynamic, 1>::Zero(basis_degree+1);
        Matrix<T, Dynamic, 1> retx = Matrix<T, Dynamic, 1>::Zero(basis_degree+1);
        Matrix<T, Dynamic, 1> ret =  Matrix<T, Dynamic, 1>::Zero(basis_size);

    
        /*
        for(size_t i = 0 ; i<basis_degree+1 ; i++){
            std::cout<<row_indeces[i]<<" , ";
        }
        std::cout<<std::endl;
        for(size_t i = 0 ; i<basis_degree+1 ; i++){
            std::cout<<col_indeces[i]<<" , ";
        }
        std::cout<<std::endl;
        */
        // the indeces of first row and column are already saved. Just need a loop over them and then do the right tensor product maintaining the ordering of the nodes.
        size_t ct = 0;
        for(auto& k : row_indeces)
        {
            T bk = 1;
            for ( auto& j : row_indeces ){
                if(j!=k){
            bk *= ( ( pt.x() - (nodes.at(j)).x() )/ ( (nodes.at(k)).x() - (nodes.at(j)).x() ) );
                }
            }
            
            retx(ct) = bk;
            ct++;
        }
        
        ct = 0;
        for(auto& l : col_indeces)
        {
            T bl = 1;
            for ( auto& j : col_indeces ){
                if(j!=l){
            bl *= ( ( pt.y() - (nodes.at(j)).y() )/ ( (nodes.at(l)).y() - (nodes.at(j)).y() ) );
                }
            }
            rety(ct) = bl;
            ct++;
        }
        
    
    
        // VERTICES FUNCTIONS:
        if( basis_degree == 0){
            ret(0) = rety(0)*retx(0);
            return ret ;
        }
        
        
        ret(0) = rety(0)*retx(0);
        ret(1) = rety(0)*retx(1);
        ret(2) = rety(1)*retx(1);
        ret(3) = rety(1)*retx(0);
        
        if( basis_degree == 1)
            return ret ;
        
        int  count0 = 4 , count1 = 4 + basis_degree - 1 , count2 = 4 + 2*(basis_degree - 1), count3 = 4 + 3*(basis_degree - 1)  ;
        int count_bis0 = 4*basis_degree ; // counter initialisation (USELESS)
        int j_max = floor((basis_degree-1)/2) ; // number of internal layour of points
        int pos_right = 100; // initial point from the right from bottom -> up
        int pos_left = 100; // initial point from the left from left -> right
     
        for (int j = 0 ; j <= j_max ; j++ ) { // for each layout of point
            // bool i_loop = FALSE;
            for(int i = std::max(0,j-1) ; i < basis_degree -1 - j ; i++ ) // I move from left into the points over a side of each layout
            {
                if( i == std::max(0,j-1) && j > 0) // vertices
                {
                    // different pos_left depending on the layout. Especially this rules the y starting point
                    if(j == 0)
                        pos_left = 0;
                    else if(j == 1)
                        pos_left = 2;
                    else
                        pos_left = 2+(j-1);
                                             
                    // Here change counters. No more count0, count1 etc.. Just count_bis0, to have the vertices enumerate one after the other.
                    count_bis0 = count0; // counter_bis0 re-initialisation for each loop (first loop)
                    //std::cout<<"count bis 0 "<<count_bis0<<std::endl;
                    ret(count_bis0) = ( retx(2+i)*rety(pos_left) ); // adding point bottom layout
                    count_bis0++;
                    if(j==0)
                        pos_right = 1;
                    else
                        pos_right = basis_degree + 1 - j;
    
                    ret(count_bis0) = ( retx(pos_right)*rety(2 + i) ); // adding point right layout in count_bis0 pos.
                    count_bis0++;

                    ret(count_bis0) = ( retx(basis_degree - i)*rety(pos_right) ); // adding point top layout in count_bis0 pos.
                                 
                    count_bis0++;
                                 
                    ret(count_bis0) = ( retx(pos_left)*rety(basis_degree - i) );
                                 
                    count_bis0++;
                                  
                    // Uploading counters according to count_bis0. I start enumerate "face" unknown for the current layout (IN ELSE HERE BELOW) from the last position count_bis0.
                    count0 = count_bis0 ;
                    count1 = count0 + std::ceil((basis_degree - 2.0*j)/2.0); //  count0 + (degree - 2); it was
                    count2 = count1 + std::ceil((basis_degree - 2.0*j)/2.0);
                    count3 = count2 + std::ceil((basis_degree - 2.0*j)/2.0);

                }
                else // NOT vertices -> node in the sides of each layout
                {
                    //   std::cout<<"i "<<i<<" j "<<j<<std::endl;
                    // vertical position where starting for each bottom side layout.
                    if(j == 0)
                        pos_left = 0;
                    else if(j == 1)
                        pos_left = 2;
                    else
                        pos_left = 2+(j-1);
                                
                    ret(count0) = ( retx(2+i)*rety(pos_left) );
                    count0++;
                    // x-position where to start to increase to get points for each right side layout of points
                    if(j==0)
                        pos_right = 1;
                    else
                        pos_right = basis_degree + 1 - j;
                  
                    ret(count1) = ( retx(pos_right)*rety(2 + i) );
                    count1++; // count1 bigger than count0. Count1 just enumerate right faces' layout
              
                    ret(count2) = ( retx(basis_degree - i)*rety(pos_right) );
                    count2++; // count2 just enumerate top faces' layout
                                  
                    ret(count3) = ( retx(pos_left)*rety(basis_degree - i) );
                    count3++; // count3 just enumerate left faces' layout
                              
                                  
                }
            }
            // Uploading for the next layout.
            count0 = count3 ;
            count1 = count0 + (basis_degree - 2*(j+1));
            count2 = count1 + (basis_degree - 2*(j+1));
            count3 = count2 + (basis_degree - 2*(j+1));
                     
        }
                      
        /// Middle point --> the internal node is treated a part of the others. Just even degrees have it.
        if( basis_degree % 2 == 0)
            ret(count0) = (retx(basis_degree - floor((basis_degree-1)/2))*rety(basis_degree - floor((basis_degree-1)/2)));
                          
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



int binomial_coeff_fx(int n , int k) {

        int C[n + 1][k + 1];
        int i, j;
      
        // Caculate value of Binomial Coefficient
        // in bottom up manner
        for (i = 0; i <= n; i++)
        {
            for (j = 0; j <= std::min(i, k); j++)
            {
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



template<typename Mesh, typename VT>
class cell_basis_Bernstein
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;

    point_type          cell_bar;
    coordinate_type     cell_h;
    size_t              basis_degree, basis_size;
    coordinate_type min_x , max_x , min_y , max_y ;
    coordinate_type scaling_x , scaling_y ;

#ifdef POWER_CACHE
    std::vector<coordinate_type>  power_cache , power_cache_bis;
    std::vector<size_t>  binomial_coeff , binomial_coeff_bis ;
#endif

public:
    cell_basis_Bernstein(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree)
    {
        //cell_bar        = barycenter(msh, cl);
        //cell_h          = diameter(msh, cl);
        basis_degree    = degree;
        basis_size      = (basis_degree+1)*(basis_degree+1);
        min_x = points(msh,cl)[0].x();
        max_x = points(msh,cl)[1].x();
        min_y = points(msh,cl)[0].y();
        max_y = points(msh,cl)[2].y();
        scaling_x = 1.0/( pow( (max_x - min_x), basis_degree) );
        scaling_y = 1.0/( pow( (max_y - min_y), basis_degree) );
       
    }

    Matrix<VT, Dynamic, 1>
    eval_basis(const point_type& pt)
    {
        Matrix<VT, Dynamic, 1> ret = Matrix<VT, Dynamic, 1>::Zero(basis_size);

        auto bx0 = pt.x() - min_x ;
        auto bx1 = max_x - pt.x() ;
        auto by0 = pt.y() - min_y ;
        auto by1 = max_y - pt.y() ;
        
#ifdef POWER_CACHE
        if ( power_cache.size() != (basis_degree+1)*2 )
          power_cache.resize( (basis_degree+1)*2);
        
        if ( binomial_coeff.size() != (basis_degree+1) )
        binomial_coeff.resize( basis_degree+1 );

        // Creation of the exponential terms
        for (size_t i = 0 ; i <= basis_degree ; i++)
        {
            size_t j = basis_degree - i;
            //if(i == 0)
             //   binomial_coeff[0] = 1.0 ;
            //else if(i == basis_degree)
            //    binomial_coeff[basis_degree+1] = 1.0 ;
           // else
            
            binomial_coeff[i] = binomial_coeff_fx(basis_degree,i);
           
            
            power_cache[2*i]    = binomial_coeff[i]*iexp_pow(bx0, i)*iexp_pow(bx1, j);
            power_cache[2*i+1]  = binomial_coeff[i]*iexp_pow(by0, i)*iexp_pow(by1, j);
          
        }
        
#endif
        size_t pos = 0;
    
        // Case degree FEM = 0
        if (basis_degree == 0){
            VT one = 1.0;
            ret(pos++) = one;
        }
        // Case degree FEM = 1 -> i.e. just vertices
        // Cell ordering:
        // 3 -- 2
        // 0 -- 1
        
        else if(basis_degree == 1){
            for (int pow_y = 0; pow_y <= basis_degree; pow_y +=basis_degree){
                if( pow_y == 0 ) // ordering from sx to dx 0->1
                {
                for (int pow_x = 0; pow_x <= basis_degree; pow_x+=basis_degree){
#ifdef POWER_CACHE
                    auto bv = scaling_x * power_cache[2*pow_x] * scaling_y * power_cache[2*pow_y+1];
#else
                    size_t j_x = basis_degree - pow_x;
                    size_t j_y = basis_degree - pow_y;
                                            
                    auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                    auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                        
                    auto bv = scaling_x * coeff_n_x * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x) * scaling_y * coeff_n_y * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y) ;
#endif
                    ret(pos++) = bv;
                }
                }
                else{ // ordering from dx to sx 2->3
                    for (int pow_x = basis_degree; pow_x >= 0; pow_x-=basis_degree){
#ifdef POWER_CACHE
                    auto bv = scaling_x * power_cache[2*pow_x] * scaling_y * power_cache[2*pow_y+1];
#else
                    size_t j_x = basis_degree - pow_x;
                    size_t j_y = basis_degree - pow_y;
                                                                
                    auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                    auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                                            
                    auto bv = scaling_x * coeff_n_x * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x) * scaling_y * coeff_n_y * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y) ;
#endif
                    ret(pos++) = bv;
                }
                }
            }
        }
        
        else
        {
            // case degree FEM >= 2
            int N = basis_degree ;
            int starter = 0;
            int internal_bases = basis_degree + 1 ; // counting the number of layout of internal nodes: initially there are basis_degree +1 nodes
           
            while(internal_bases > 1) // if there is a deeper layout of nodes
            {
                // Vertices nodes of a quadrangular cell. For each loop of internal_bases I add the vertice of "inner quadrangualar nodes" ( it's not properly true speaking of nodes, being a modal approach; better say the picks of the shape functions).
                for (int pow_y = starter; pow_y <= N; pow_y +=(N-starter)){
                  
                    if(pow_y == starter ) // if bottom line -> from sx to dx
                    {
                       // if not external layout of nodes ( vertices level ) -> I don't take extremes since internal layout has 2 nodes less.
                        for (int pow_x = starter; pow_x <= N; pow_x+=(N-starter)){
#ifdef POWER_CACHE
                        auto bv = scaling_x * power_cache[2*pow_x] * scaling_y * power_cache[2*pow_y+1];
#else
                        size_t j_x = basis_degree - pow_x;
                        size_t j_y = basis_degree - pow_y;
                                                                
                        auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                        auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                                            
                        auto bv = scaling_x * coeff_n_x * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x) * scaling_y * coeff_n_y * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y) ;
#endif
                        ret(pos++) = bv;
                        // std::cout<<"pos vertices bottom is "<<pos<<std::endl;
                        }
                    }
                    else{ // if top line -> from dx to sx
                        for (int pow_x = N; pow_x >= starter; pow_x-=(N-starter)){
#ifdef POWER_CACHE
                        auto bv = scaling_x * power_cache[2*pow_x] * scaling_y * power_cache[2*pow_y+1];
#else
                        size_t j_x = basis_degree - pow_x;
                        size_t j_y = basis_degree - pow_y;
                                                                                    
                        auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                        auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                                                                
                        auto bv = scaling_x * coeff_n_x * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x) * scaling_y * coeff_n_y * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y) ;
#endif
                        ret(pos++) = bv;
                       // std::cout<<"pos vertices upper is "<<pos<<std::endl;
                        }
                        
                    }
                    
                }
                    
                    
                // After the vertices elements, step by faces bases.

                //size_t pos = 4;
                // face 0 (BOTTOM)
                for (size_t pow_x = starter + 1 ; pow_x <= N - 1 ; pow_x++)
                {
                    size_t pow_y = starter;
#ifdef POWER_CACHE
                    auto bv = scaling_x * power_cache[2*pow_x] * scaling_y * power_cache[2*pow_y+1];
#else
                    size_t j_x = basis_degree - pow_x;
                    size_t j_y = basis_degree - pow_y;
                            
                    auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                    auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                           
                    auto bv = scaling_x * coeff_n_x * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x) * scaling_y * coeff_n_y * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y) ;
#endif
                    ret(pos++) = bv;
                   //  std::cout<<"pos face 0 is "<<pos<<std::endl;
                }
       
                // face 1 (RIGHT)
                for (size_t pow_y = starter + 1; pow_y <= N - 1 ; pow_y++)
                {
                    size_t pow_x = N;
        #ifdef POWER_CACHE
                    auto bv = scaling_x * power_cache[2*pow_x] * scaling_y * power_cache[2*pow_y+1];
        #else
                    size_t j_x = basis_degree - pow_x;
                    size_t j_y = basis_degree - pow_y;
                                    
                    auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                    auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                   
                    auto bv = scaling_x * coeff_n_x * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x) * scaling_y * coeff_n_y * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y) ;
        #endif
                    ret(pos++) = bv;
                  //   std::cout<<"pos face 1 is "<<pos<<std::endl;
                }
        
                // face 2 (TOP)
                for (size_t pow_x = N - 1; pow_x >= starter + 1 ; pow_x--)
                {
                    size_t pow_y = N;
        #ifdef POWER_CACHE
                    auto bv = scaling_x * power_cache[2*pow_x] * scaling_y * power_cache[2*pow_y+1];
        #else
                    size_t j_x = basis_degree - pow_x;
                    size_t j_y = basis_degree - pow_y;
                                    
                    auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                    auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                   
                    auto bv = scaling_x * coeff_n_x * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x) * scaling_y * coeff_n_y * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y) ;
        #endif
                    ret(pos++) = bv;
                  //   std::cout<<"pos face 2 is "<<pos<<std::endl;
                }
        
              
                // face 3 (LEFT)
                
                for (size_t pow_y = N - 1; pow_y >= starter + 1 ; pow_y--)
                {
                    size_t pow_x = starter;
        #ifdef POWER_CACHE
                    auto bv = scaling_x * power_cache[2*pow_x] * scaling_y * power_cache[2*pow_y+1];
        #else
                    size_t j_x = basis_degree - pow_x;
                    size_t j_y = basis_degree - pow_y;
                                    
                    auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                    auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                   
                    auto bv = scaling_x * coeff_n_x * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x) * scaling_y * coeff_n_y * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y) ;
        #endif
                    ret(pos++) = bv;
                  //   std::cout<<"pos face 3 is "<<pos<<std::endl;
                }
        
                N--; // for each layout of "nodes", I have un unknown less from right
                starter++; // for each layout of "nodes", I have un unknown less from left
                internal_bases -= 2 ; // each layout I have 2 nodes less than the previous
                
            }
            // for B_k, with k even, odd number of bases: there is a central one.
            if( basis_degree % 2 == 0 )
            {
             //   std::cout<<"N is "<<N<< " and starter is "<<starter<<std::endl;
                assert( N == starter );
                size_t pow_x = starter ;
                size_t pow_y = starter ;
#ifdef POWER_CACHE
                auto bv = scaling_x * power_cache[2*pow_x] * scaling_y * power_cache[2*pow_y+1];
#else
                size_t j_x = basis_degree - pow_x;
                size_t j_y = basis_degree - pow_y;
                                
                auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                            
                auto bv = scaling_x * coeff_n_x * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x) * scaling_y * coeff_n_y * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y) ;
#endif
                ret(pos++) = bv;
               //  std::cout<<"pos %2 is "<<pos<<std::endl;
            }
        }
       // std::cout<<"pos is"<<pos<<" and basis size is "<< basis_size<<std::endl;
        assert(pos == basis_size);

        return ret;
    }

    // Same approach of eval
    Matrix<VT, Dynamic, 2>
    eval_gradients(const point_type& pt)
    {
        Matrix<VT, Dynamic, 2> ret = Matrix<VT, Dynamic, 2>::Zero(basis_size, 2);

        auto bx0 = pt.x() - min_x ;
        auto bx1 = max_x - pt.x() ;
        auto by0 = pt.y() - min_y ;
        auto by1 = max_y - pt.y() ;
        //The stuff below should be in the constructor, but gradient is used less times than eval.
        auto N = basis_degree - 1 ;
        auto coeff_dx = basis_degree/(max_x - min_x); // n/(b-a)
        auto coeff_dy = basis_degree/(max_y - min_y);
        auto scaling_x_bis = 1.0/(pow(max_x - min_x,N) ); // 1/(b-a)^(n-1)
        auto scaling_y_bis = 1.0/(pow(max_y - min_y,N) );
        //std::cout<<"scaling_x_bis "<<scaling_x_bis<<std::endl;
        
#ifdef POWER_CACHE
        if ( power_cache.size() != (basis_degree+1)*2 )
          power_cache.resize( (basis_degree+1)*2) ;
        
        if ( power_cache_bis.size() != (N+1)*2 )
            power_cache_bis.resize( (N+1)*2) ;
        
        if ( binomial_coeff.size() != (basis_degree+1) )
            binomial_coeff.resize( basis_degree+1 ) ;
            
        if ( binomial_coeff_bis.size() != (N+1) )
            binomial_coeff_bis.resize( N+1 ) ;
        
       // Construction of the exponenatial term for bernstein basis B^N and B^(N-1) (useful for derivative)
        for (size_t i = 0; i <= basis_degree ; i++)
        {
            size_t j = basis_degree - i;
            binomial_coeff[i] = binomial_coeff_fx( basis_degree , i );
            power_cache[2*i]    = scaling_x *binomial_coeff[i]*iexp_pow(bx0, i)*iexp_pow(bx1, j);
            power_cache[2*i+1]  = scaling_y *binomial_coeff[i]*iexp_pow(by0, i)*iexp_pow(by1, j);
             
            if( i < basis_degree )
            {
                size_t j_bis = N - i;
                binomial_coeff_bis[i] = binomial_coeff_fx( N , i );
                power_cache_bis[2*i] = scaling_x_bis * binomial_coeff_bis[i]*iexp_pow(bx0, i)*iexp_pow(bx1, j_bis);
                power_cache_bis[2*i+1] = scaling_y_bis * binomial_coeff_bis[i]*iexp_pow(by0, i)*iexp_pow(by1, j_bis);
               
            }
            
        }
#endif

        size_t pos = 0;
        // Case FEM degree = 0
        if (basis_degree == 0){
            VT zero = 0.0;
            ret(pos,0) =  zero;
            ret(pos,1) = zero;
            pos++;
        }
        // Case FEM degree = 1
        else if(basis_degree == 1){
            VT coeffx = 1.0/(max_x - min_x);
            VT coeffy = 1.0/(max_y - min_y);
            //std::cout<<"min x "<<min_x<<"max x "<<max_x<<"min y "<<min_y<<"max y "<<max_y<<std::endl;
            auto px = 0.0 , py = 0.0 ;
            int j = 1; // useful for the sign of the derivative in y
            // Pick just vertices
            for (int pow_y = 0; pow_y <= basis_degree; pow_y +=basis_degree){
                if( pow_y == 0 ) // bottom
                {
                    int i = 1;  // useful for the sign of the derivative in x
                    for (int pow_x = 0; pow_x <= basis_degree; pow_x+=basis_degree){
        #ifdef POWER_CACHE
                    auto bv = scaling_x * power_cache[2*pow_x] * scaling_y * power_cache[2*pow_y+1];
        #else
                    size_t j_x = basis_degree - pow_x;
                    size_t j_y = basis_degree - pow_y;
                                                        
                    auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                    auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                                    
                    px = scaling_x * coeff_n_x * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x) ;
                    py = scaling_y * coeff_n_y * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y) ;
        #endif
                    
                   
                    ret(pos,0) =  py * coeffx * pow(-1,i);
                    ret(pos,1) = px * coeffy * pow(-1,j);
                    //std::cout<<"i is "<<i<< " and pow (-1,i) = "<<pow(-1,i)<<std::endl;
                    pos++;
                    i++;
                    }
                  
                }
                else{
                    int i = 2;
                    for (int pow_x = basis_degree; pow_x >= 0; pow_x-=basis_degree){
        #ifdef POWER_CACHE
                    auto bv = scaling_x * power_cache[2*pow_x] * scaling_y * power_cache[2*pow_y+1];
        #else
                    size_t j_x = basis_degree - pow_x;
                    size_t j_y = basis_degree - pow_y;
                                                                            
                    auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                    auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                    px = scaling_x * coeff_n_x * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x) ;
                    py = scaling_y * coeff_n_y * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y) ;
        #endif
                   
                    ret(pos,0) =  py * coeffx * pow(-1,i);
                    ret(pos,1) = px * coeffy * pow(-1,j);
                  //  std::cout<<"i is "<<i<< " and pow (-1,i) = "<<pow(-1,i)<<std::endl;
                    pos++;
                    i--;
                    }
                }
                j++;
            }
           
        }
        
        else{
            // if degree FEM >= 2
            //std::cout<<"Bernstein Gradient, basis >=2 "<<std::endl;
            int N_partial = basis_degree ;
            int starter = 0;
            int internal_bases = basis_degree + 1 ;
            
            while(internal_bases > 1) // for each layour of internal node
            {
                for (int pow_y = starter; pow_y <= N_partial ; pow_y +=(N_partial-starter)){
                    
                    if(pow_y == starter ) // Bottom side
                    {
                        
                        for (int pow_x = starter; pow_x <= N_partial; pow_x+=(N_partial-starter)){
                            
                            VT ix_bis = pow_x-1; // element i-1 for derivative
                            VT iy_bis = pow_y-1; // element i-1 for derivative
                            
#ifdef POWER_CACHE
                            std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                            auto px = power_cache[2*pow_x];
                            auto py = power_cache[2*pow_y+1];
                            //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                            if ( pow_x == 0 )
                                auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                            else if ( pow_x == basis_degree )
                                auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            else
                                auto dx = coeff_dx*( power_cache_bis[2*ix_bis] - power_cache_bis[2*pow_x] );
                            
                            if ( pow_y == 0 )
                                auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                            else if ( pow_y == basis_degree )
                                auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                            else
                                auto dy = coeff_dy*( power_cache_bis[2*iy_bis+1] - power_cache_bis[2*pow_y+1] );
                            //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                            
#else
                            
                            //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                            size_t j_x = basis_degree - pow_x;
                            size_t j_y = basis_degree - pow_y;
                            //std::cout<<"j_x PRE "<<j_x<<std::endl;
                            auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                            auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                            //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                            //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                            //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                            //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                            //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                            
                            auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                            auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                            //std::cout<<"px PRE "<<px<<std::endl;
                            
                            // DERIVATIVES
                            //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                            //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 ;
                            VT dx = 0.0 , dy = 0.0 ;
                            
                            if ( pow_x == 0 ){
                                //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                dx = -coeff_dx * px_bis0 ;
                                //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                
                            }
                            else if ( pow_x == basis_degree ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis1 = N - ix_bis;
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_dx * px_bis1 ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else{
                                size_t j_x_bis0 = N - pow_x;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                
                                size_t j_x_bis1 = N - ix_bis;
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_dx * (px_bis1 -px_bis0);
                                //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                            }
                            
                            if ( pow_y == 0 ){
                                //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                               // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                dy = -coeff_dy *  py_bis0 ;
                                //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                                
                            }
                            else if ( pow_y == basis_degree ){
                                //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                size_t j_y_bis1 = N - iy_bis;
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_dy * py_bis1  ;
                                //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                            }
                            else{
                                size_t j_y_bis0 = N - pow_y;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                
                                size_t j_y_bis1 = N - iy_bis;
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_dy * ( py_bis1 - py_bis0 );
                            //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                            }
                            
                            
#endif
                            //std::cout<<"pos "<<pos<<std::endl;
                            //std::cout<<"dx "<<dx<<std::endl;
                            //std::cout<<"dy "<<dy<<std::endl;
                            //std::cout<<"px "<<px<<std::endl;
                            //std::cout<<"py "<<py<<std::endl;
                            ret(pos,0) = dx*py;
                            ret(pos,1) = px*dy;
                            pos++;
                            
                            
                        }
                    }
                    else{ // Top side
                        for (int pow_x = N_partial; pow_x >= starter; pow_x-=(N_partial-starter)){
                            VT ix_bis = pow_x-1;
                            VT iy_bis = pow_y-1;
                            
#ifdef POWER_CACHE
                            auto px = power_cache[2*pow_x];
                            auto py = power_cache[2*pow_y+1];
                            if ( pow_x == 0 )
                                auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                            else if ( pow_x == basis_degree )
                                auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            else
                                auto dx = coeff_dx*( power_cache_bis[2*ix_bis] - power_cache_bis[2*pow_x] );
                            
                            if ( pow_y == 0 )
                                auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                            else if ( pow_y == basis_degree )
                                auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                            else
                                auto dy = coeff_dy*( power_cache_bis[2*iy_bis+1] - power_cache_bis[2*pow_y+1] );
#else
                            
                            size_t j_x = basis_degree - pow_x;
                            size_t j_y = basis_degree - pow_y;
                            
                            auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                            auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                            
                            auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                            auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                            
                            // DERIVATIVES
                            //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                            //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 ;
                            VT dx = 0.0 , dy = 0.0 ;
                            
                            if ( pow_x == 0 ){
                                //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                dx = -coeff_dx * px_bis0 ;
                                //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                
                            }
                            else if ( pow_x == basis_degree ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis1 = N - ix_bis;
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_dx * px_bis1 ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else{
                                size_t j_x_bis0 = N - pow_x;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                
                                size_t j_x_bis1 = N - ix_bis;
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_dx * (px_bis1 -px_bis0);
                                //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                            }
                            
                            if ( pow_y == 0 ){
                                //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                               // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                dy = -coeff_dy *  py_bis0 ;
                                //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                                
                            }
                            else if ( pow_y == basis_degree ){
                                //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                size_t j_y_bis1 = N - iy_bis;
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_dy * py_bis1  ;
                                //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                            }
                            else{
                                size_t j_y_bis0 = N - pow_y;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                
                                size_t j_y_bis1 = N - iy_bis;
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_dy * ( py_bis1 - py_bis0 );
                            //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                            }
                            
                            
#endif
                            
                            ret(pos,0) = dx*py;
                            ret(pos,1) = px*dy;
                            pos++;
                            
                            
                        } // loop for top side vertices
                        
                        
                        
                        
                    } // else bottom-top side
                    
                } // end loop for vertices
                
                
                // face 0 (BOTTOM)
                for (size_t pow_x = starter + 1 ; pow_x <= N_partial - 1 ; pow_x++)
                {
                    size_t pow_y = starter;
                    
                    VT ix_bis = pow_x-1;
                    VT iy_bis = pow_y-1;
                    
#ifdef POWER_CACHE
                    auto px = power_cache[2*pow_x];
                    auto py = power_cache[2*pow_y+1];
                    if ( pow_x == 0 )
                        auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                    else if ( pow_x == basis_degree )
                        auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                    else
                        auto dx = coeff_dx*( power_cache_bis[2*ix_bis] - power_cache_bis[2*pow_x] );
                    
                    if ( pow_y == 0 )
                        auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                    else if ( pow_y == basis_degree )
                        auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                    else
                        auto dy = coeff_dy*( power_cache_bis[2*iy_bis+1] - power_cache_bis[2*pow_y+1] );
#else
                    
                    size_t j_x = basis_degree - pow_x;
                    size_t j_y = basis_degree - pow_y;
                    
                    auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                    auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                    
                    auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                    auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                    //std::cout<<"j_x Face0 "<<j_x<<std::endl;
                    //std::cout<<"coeff_n_x Face0 "<<coeff_n_x<<std::endl;
                    //std::cout<<"scaling_x Face0 "<<scaling_x<<std::endl;
                    //std::cout<<"pow_x Face0 "<<pow_x<<std::endl;
                    //std::cout<<"bx0 Face0 "<<bx0<<std::endl;
                    //std::cout<<"bx1 Face0 "<<bx1<<std::endl;
                    
                    
                    // DERIVATIVES
                    //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                    //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 ;
                    VT dx = 0.0 , dy = 0.0 ;
                    
                    if ( pow_x == 0 ){
                        //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                        size_t j_x_bis0 = N - pow_x;
                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                        dx = -coeff_dx * px_bis0 ;
                        //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                        
                    }
                    else if ( pow_x == basis_degree ){
                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                        size_t j_x_bis1 = N - ix_bis;
                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                        dx = coeff_dx * px_bis1 ;
                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                    }
                    else{
                        size_t j_x_bis0 = N - pow_x;
                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                        
                        size_t j_x_bis1 = N - ix_bis;
                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                        dx = coeff_dx * (px_bis1 -px_bis0);
                        //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                    }
                    
                    if ( pow_y == 0 ){
                        //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                        size_t j_y_bis0 = N - pow_y;
                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                       // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                        dy = -coeff_dy *  py_bis0 ;
                        //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                        
                    }
                    else if ( pow_y == basis_degree ){
                        //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                        size_t j_y_bis1 = N - iy_bis;
                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                        dy = coeff_dy * py_bis1  ;
                        //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                    }
                    else{
                        size_t j_y_bis0 = N - pow_y;
                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                        
                        size_t j_y_bis1 = N - iy_bis;
                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                        dy = coeff_dy * ( py_bis1 - py_bis0 );
                    //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                    }
                    
#endif
                    
                    ret(pos,0) = dx*py;
                    ret(pos,1) = px*dy;
                    pos++;
                    
                }
                
                // face 1 (RIGHT)
                for (size_t pow_y = starter + 1; pow_y <= N_partial - 1 ; pow_y++)
                {
                    size_t pow_x = N_partial;
                    VT ix_bis = pow_x-1;
                    VT iy_bis = pow_y-1;
                    
#ifdef POWER_CACHE
                    auto px = power_cache[2*pow_x];
                    auto py = power_cache[2*pow_y+1];
                    if ( pow_x == 0 )
                        auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                    else if ( pow_x == basis_degree )
                        auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                    else
                        auto dx = coeff_dx*( power_cache_bis[2*ix_bis] - power_cache_bis[2*pow_x] );
                    
                    if ( pow_y == 0 )
                        auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                    else if ( pow_y == basis_degree )
                        auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                    else
                        auto dy = coeff_dy*( power_cache_bis[2*iy_bis+1] - power_cache_bis[2*pow_y+1] );
#else
                    
                    size_t j_x = basis_degree - pow_x;
                    size_t j_y = basis_degree - pow_y;
                    
                    auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                    auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                    
                    auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                    auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                    
                    // DERIVATIVES
                    //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                    //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 ;
                    VT dx = 0.0 , dy = 0.0 ;
                    
                    if ( pow_x == 0 ){
                        //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                        size_t j_x_bis0 = N - pow_x;
                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                        dx = -coeff_dx * px_bis0 ;
                        //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                        
                    }
                    else if ( pow_x == basis_degree ){
                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                        size_t j_x_bis1 = N - ix_bis;
                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                        dx = coeff_dx * px_bis1 ;
                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                    }
                    else{
                        size_t j_x_bis0 = N - pow_x;
                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                        
                        size_t j_x_bis1 = N - ix_bis;
                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                        dx = coeff_dx * (px_bis1 -px_bis0);
                        //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                    }
                    
                    if ( pow_y == 0 ){
                        //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                        size_t j_y_bis0 = N - pow_y;
                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                       // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                        dy = -coeff_dy *  py_bis0 ;
                        //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                        
                    }
                    else if ( pow_y == basis_degree ){
                        //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                        size_t j_y_bis1 = N - iy_bis;
                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                        dy = coeff_dy * py_bis1  ;
                        //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                    }
                    else{
                        size_t j_y_bis0 = N - pow_y;
                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                        
                        size_t j_y_bis1 = N - iy_bis;
                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                        dy = coeff_dy * ( py_bis1 - py_bis0 );
                    //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                    }
                    
                    
#endif
                    
                    ret(pos,0) = dx*py;
                    ret(pos,1) = px*dy;
                    pos++;
                }
                
                // face 2 (TOP)
                for (size_t pow_x = N_partial - 1; pow_x >= starter + 1 ; pow_x--)
                {
                    size_t pow_y = N_partial;
                    VT ix_bis = pow_x-1;
                    VT iy_bis = pow_y-1;
                    
#ifdef POWER_CACHE
                    auto px = power_cache[2*pow_x];
                    auto py = power_cache[2*pow_y+1];
                    if ( pow_x == 0 )
                        auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                    else if ( pow_x == basis_degree )
                        auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                    else
                        auto dx = coeff_dx*( power_cache_bis[2*ix_bis] - power_cache_bis[2*pow_x] );
                    
                    if ( pow_y == 0 )
                        auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                    else if ( pow_y == basis_degree )
                        auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                    else
                        auto dy = coeff_dy*( power_cache_bis[2*iy_bis+1] - power_cache_bis[2*pow_y+1] );
#else
                    
                    size_t j_x = basis_degree - pow_x;
                    size_t j_y = basis_degree - pow_y;
                    
                    auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                    auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                    
                    auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                    auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                    
                    // DERIVATIVES
                    //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                    //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 ;
                    VT dx = 0.0 , dy = 0.0 ;
                    
                    if ( pow_x == 0 ){
                        //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                        size_t j_x_bis0 = N - pow_x;
                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                        dx = -coeff_dx * px_bis0 ;
                        //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                        
                    }
                    else if ( pow_x == basis_degree ){
                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                        size_t j_x_bis1 = N - ix_bis;
                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                        dx = coeff_dx * px_bis1 ;
                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                    }
                    else{
                        size_t j_x_bis0 = N - pow_x;
                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                        
                        size_t j_x_bis1 = N - ix_bis;
                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                        dx = coeff_dx * (px_bis1 -px_bis0);
                        //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                    }
                    
                    if ( pow_y == 0 ){
                        //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                        size_t j_y_bis0 = N - pow_y;
                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                       // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                        dy = -coeff_dy *  py_bis0 ;
                        //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                        
                    }
                    else if ( pow_y == basis_degree ){
                        //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                        size_t j_y_bis1 = N - iy_bis;
                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                        dy = coeff_dy * py_bis1  ;
                        //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                    }
                    else{
                        size_t j_y_bis0 = N - pow_y;
                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                        
                        size_t j_y_bis1 = N - iy_bis;
                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                        dy = coeff_dy * ( py_bis1 - py_bis0 );
                    //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                    }
                    
                    
#endif
                    
                    ret(pos,0) = dx*py;
                    ret(pos,1) = px*dy;
                    pos++;
                }
                
                
                // face 3 (LEFT)
                
                for (size_t pow_y = N_partial - 1; pow_y >= starter + 1 ; pow_y--)
                {
                    size_t pow_x = starter;
                    VT ix_bis = pow_x-1;
                    VT iy_bis = pow_y-1;
                    
#ifdef POWER_CACHE
                    auto px = power_cache[2*pow_x];
                    auto py = power_cache[2*pow_y+1];
                    if ( pow_x == 0 )
                        auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                    else if ( pow_x == basis_degree )
                        auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                    else
                        auto dx = coeff_dx*( power_cache_bis[2*ix_bis] - power_cache_bis[2*pow_x] );
                    
                    if ( pow_y == 0 )
                        auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                    else if ( pow_y == basis_degree )
                        auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                    else
                        auto dy = coeff_dy*( power_cache_bis[2*iy_bis+1] - power_cache_bis[2*pow_y+1] );
#else
                    
                    size_t j_x = basis_degree - pow_x;
                    size_t j_y = basis_degree - pow_y;
                    
                    auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                    auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                    
                    auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                    auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                    
                    // DERIVATIVES
                    //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                    //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 ;
                    VT dx = 0.0 , dy = 0.0 ;
                    
                    if ( pow_x == 0 ){
                        //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                        size_t j_x_bis0 = N - pow_x;
                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                        dx = -coeff_dx * px_bis0 ;
                        //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                        
                    }
                    else if ( pow_x == basis_degree ){
                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                        size_t j_x_bis1 = N - ix_bis;
                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                        dx = coeff_dx * px_bis1 ;
                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                    }
                    else{
                        size_t j_x_bis0 = N - pow_x;
                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                        
                        size_t j_x_bis1 = N - ix_bis;
                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                        dx = coeff_dx * (px_bis1 -px_bis0);
                        //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                    }
                    
                    if ( pow_y == 0 ){
                        //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                        size_t j_y_bis0 = N - pow_y;
                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                       // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                        dy = -coeff_dy *  py_bis0 ;
                        //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                        
                    }
                    else if ( pow_y == basis_degree ){
                        //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                        size_t j_y_bis1 = N - iy_bis;
                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                        dy = coeff_dy * py_bis1  ;
                        //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                    }
                    else{
                        size_t j_y_bis0 = N - pow_y;
                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                        
                        size_t j_y_bis1 = N - iy_bis;
                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                        dy = coeff_dy * ( py_bis1 - py_bis0 );
                    //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                    }
                    
                    
#endif
                    
                    ret(pos,0) = dx*py;
                    ret(pos,1) = px*dy;
                    pos++;
                }
                
                N_partial--;
                starter++;
                internal_bases -= 2;
                
            }
            
            // for B_k, with k even, I.E. odd number of bases: there is a central one.
            if( basis_degree % 2 == 0 )
            {
                //  std::cout<<"N is "<<N_partial<< " and starter is "<<starter<<std::endl;
                assert( N_partial == starter );
                size_t pow_x = starter ;
                size_t pow_y = starter ;
                VT ix_bis = pow_x-1;
                VT iy_bis = pow_y-1;
                
#ifdef POWER_CACHE
                auto px = power_cache[2*pow_x];
                auto py = power_cache[2*pow_y+1];
                if ( pow_x == 0 )
                    auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                else if ( pow_x == basis_degree )
                    auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                else
                    auto dx = coeff_dx*( power_cache_bis[2*ix_bis] - power_cache_bis[2*pow_x] );
                
                if ( pow_y == 0 )
                    auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                else if ( pow_y == basis_degree )
                    auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                else
                    auto dy = coeff_dy*( power_cache_bis[2*iy_bis+1] - power_cache_bis[2*pow_y+1] );
#else
                
                size_t j_x = basis_degree - pow_x;
                size_t j_y = basis_degree - pow_y;
                
                auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                
                auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                
                // DERIVATIVES
                //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 ;
                VT dx = 0.0 , dy = 0.0 ;
                
                if ( pow_x == 0 ){
                    //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                    size_t j_x_bis0 = N - pow_x;
                    auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                    auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                    dx = -coeff_dx * px_bis0 ;
                    //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                    
                }
                else if ( pow_x == basis_degree ){
                    //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                    size_t j_x_bis1 = N - ix_bis;
                    auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                    auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                    dx = coeff_dx * px_bis1 ;
                    //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                }
                else{
                    size_t j_x_bis0 = N - pow_x;
                    auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                    auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                    
                    size_t j_x_bis1 = N - ix_bis;
                    auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                    auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                    dx = coeff_dx * (px_bis1 -px_bis0);
                    //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                }
                
                if ( pow_y == 0 ){
                    //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                    size_t j_y_bis0 = N - pow_y;
                    auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                    auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                   // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                    dy = -coeff_dy *  py_bis0 ;
                    //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                    
                }
                else if ( pow_y == basis_degree ){
                    //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                    size_t j_y_bis1 = N - iy_bis;
                    auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                    auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                    dy = coeff_dy * py_bis1  ;
                    //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                }
                else{
                    size_t j_y_bis0 = N - pow_y;
                    auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                    auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                    
                    size_t j_y_bis1 = N - iy_bis;
                    auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                    auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                    dy = coeff_dy * ( py_bis1 - py_bis0 );
                //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                }
                
                
#endif
                
                ret(pos,0) = dx*py;
                ret(pos,1) = px*dy;
                pos++;
            }
        }

          //  std::cout<<"GRADIENTE: pos is"<<pos<<" and basis size is "<< basis_size<<std::endl;
            assert(pos == basis_size);

            return ret;
        

    }

    
    
    
    // Same approach of eval
        Matrix<VT, Dynamic, 1>
        eval_derivative_xy(const point_type& pt)
        {
            Matrix<VT, Dynamic, 1> ret = Matrix<VT, Dynamic, 1>::Zero(basis_size, 1);

            auto bx0 = pt.x() - min_x ;
            auto bx1 = max_x - pt.x() ;
            auto by0 = pt.y() - min_y ;
            auto by1 = max_y - pt.y() ;
            //The stuff below should be in the constructor, but gradient is used less times than eval.
            auto N = basis_degree - 1 ;
            auto coeff_dx = basis_degree/(max_x - min_x); // n/(b-a)
            auto coeff_dy = basis_degree/(max_y - min_y);
            auto scaling_x_bis = 1.0/(pow(max_x - min_x,N) ); // 1/(b-a)^(n-1)
            auto scaling_y_bis = 1.0/(pow(max_y - min_y,N) );
            //std::cout<<"scaling_x_bis "<<scaling_x_bis<<std::endl;
            
    #ifdef POWER_CACHE
            if ( power_cache.size() != (basis_degree+1)*2 )
              power_cache.resize( (basis_degree+1)*2) ;
            
            if ( power_cache_bis.size() != (N+1)*2 )
                power_cache_bis.resize( (N+1)*2) ;
            
            if ( binomial_coeff.size() != (basis_degree+1) )
                binomial_coeff.resize( basis_degree+1 ) ;
                
            if ( binomial_coeff_bis.size() != (N+1) )
                binomial_coeff_bis.resize( N+1 ) ;
            
           // Construction of the exponenatial term for bernstein basis B^N and B^(N-1) (useful for derivative)
            for (size_t i = 0; i <= basis_degree ; i++)
            {
                size_t j = basis_degree - i;
                binomial_coeff[i] = binomial_coeff_fx( basis_degree , i );
                power_cache[2*i]    = scaling_x *binomial_coeff[i]*iexp_pow(bx0, i)*iexp_pow(bx1, j);
                power_cache[2*i+1]  = scaling_y *binomial_coeff[i]*iexp_pow(by0, i)*iexp_pow(by1, j);
                 
                if( i < basis_degree )
                {
                    size_t j_bis = N - i;
                    binomial_coeff_bis[i] = binomial_coeff_fx( N , i );
                    power_cache_bis[2*i] = scaling_x_bis * binomial_coeff_bis[i]*iexp_pow(bx0, i)*iexp_pow(bx1, j_bis);
                    power_cache_bis[2*i+1] = scaling_y_bis * binomial_coeff_bis[i]*iexp_pow(by0, i)*iexp_pow(by1, j_bis);
                   
                }
                
            }
    #endif

            size_t pos = 0;
            // Case FEM degree = 0
            if (basis_degree == 0){
                VT zero = 0.0;
                ret(pos) =  zero;
                
                pos++;
            }
            // Case FEM degree = 1
            else if(basis_degree == 1){
                VT coeffx = 1.0/(max_x - min_x);
                VT coeffy = 1.0/(max_y - min_y);
                //std::cout<<"min x "<<min_x<<"max x "<<max_x<<"min y "<<min_y<<"max y "<<max_y<<std::endl;
                auto px = 0.0 , py = 0.0 ;
                int j = 1; // useful for the sign of the derivative in y
                // Pick just vertices
                for (int pow_y = 0; pow_y <= basis_degree; pow_y +=basis_degree){
                    if( pow_y == 0 ) // bottom
                    {
                        int i = 1;  // useful for the sign of the derivative in x
                        for (int pow_x = 0; pow_x <= basis_degree; pow_x+=basis_degree){
            #ifdef POWER_CACHE
                        auto bv = scaling_x * power_cache[2*pow_x] * scaling_y * power_cache[2*pow_y+1];
            #else
                        size_t j_x = basis_degree - pow_x;
                        size_t j_y = basis_degree - pow_y;
                                                            
                        auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                        auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                                        
                        px = scaling_x * coeff_n_x * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x) ;
                        py = scaling_y * coeff_n_y * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y) ;
            #endif
                        
                       
                        ret(pos) =  coeffy * pow(-1,j) * coeffx * pow(-1,i);
                        //ret(pos,1) = px * coeffy * pow(-1,j);
                        //std::cout<<"i is "<<i<< " and pow (-1,i) = "<<pow(-1,i)<<std::endl;
                        pos++;
                        i++;
                        }
                      
                    }
                    else{
                        int i = 2;
                        for (int pow_x = basis_degree; pow_x >= 0; pow_x-=basis_degree){
            #ifdef POWER_CACHE
                        auto bv = scaling_x * power_cache[2*pow_x] * scaling_y * power_cache[2*pow_y+1];
            #else
                        size_t j_x = basis_degree - pow_x;
                        size_t j_y = basis_degree - pow_y;
                                                                                
                        auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                        auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                        px = scaling_x * coeff_n_x * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x) ;
                        py = scaling_y * coeff_n_y * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y) ;
            #endif
                       
                        ret(pos) =  coeffy * pow(-1,j) * coeffx * pow(-1,i);
                      
                      //  std::cout<<"i is "<<i<< " and pow (-1,i) = "<<pow(-1,i)<<std::endl;
                        pos++;
                        i--;
                        }
                    }
                    j++;
                }
               
            }
            
            else{
                // if degree FEM >= 2
                //std::cout<<"Bernstein Gradient, basis >=2 "<<std::endl;
                int N_partial = basis_degree ;
                int starter = 0;
                int internal_bases = basis_degree + 1 ;
                
                while(internal_bases > 1) // for each layour of internal node
                {
                    for (int pow_y = starter; pow_y <= N_partial ; pow_y +=(N_partial-starter)){
                        
                        if(pow_y == starter ) // Bottom side
                        {
                            
                            for (int pow_x = starter; pow_x <= N_partial; pow_x+=(N_partial-starter)){
                                
                                VT ix_bis = pow_x-1; // element i-1 for derivative
                                VT iy_bis = pow_y-1; // element i-1 for derivative
                                
    #ifdef POWER_CACHE
                                std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                                auto px = power_cache[2*pow_x];
                                auto py = power_cache[2*pow_y+1];
                                //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                                if ( pow_x == 0 )
                                    auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                else if ( pow_x == basis_degree )
                                    auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                else
                                    auto dx = coeff_dx*( power_cache_bis[2*ix_bis] - power_cache_bis[2*pow_x] );
                                
                                if ( pow_y == 0 )
                                    auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                                else if ( pow_y == basis_degree )
                                    auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                                else
                                    auto dy = coeff_dy*( power_cache_bis[2*iy_bis+1] - power_cache_bis[2*pow_y+1] );
                                //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                
    #else
                                
                                //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                                //size_t j_x = basis_degree - pow_x;
                                //size_t j_y = basis_degree - pow_y;
                                //std::cout<<"j_x PRE "<<j_x<<std::endl;
                                //auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                                //auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                                //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                                //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                                //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                                //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                
                                //auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                                //auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                                //std::cout<<"px PRE "<<px<<std::endl;
                                
                                // DERIVATIVES
                                //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                                //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 ;
                                VT dx = 0.0 , dy = 0.0 ;
                                
                                if ( pow_x == 0 ){
                                    //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                    size_t j_x_bis0 = N - pow_x;
                                    auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                    auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                    dx = -coeff_dx * px_bis0 ;
                                    //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                    
                                }
                                else if ( pow_x == basis_degree ){
                                    //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                    size_t j_x_bis1 = N - ix_bis;
                                    auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                    auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                    dx = coeff_dx * px_bis1 ;
                                    //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                }
                                else{
                                    size_t j_x_bis0 = N - pow_x;
                                    auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                    auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                    
                                    size_t j_x_bis1 = N - ix_bis;
                                    auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                    auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                    dx = coeff_dx * (px_bis1 -px_bis0);
                                    //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                                }
                                
                                if ( pow_y == 0 ){
                                    //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                    size_t j_y_bis0 = N - pow_y;
                                    auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                    auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                   // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                    dy = -coeff_dy *  py_bis0 ;
                                    //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                                    
                                }
                                else if ( pow_y == basis_degree ){
                                    //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                    size_t j_y_bis1 = N - iy_bis;
                                    auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                    auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                    dy = coeff_dy * py_bis1  ;
                                    //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                                }
                                else{
                                    size_t j_y_bis0 = N - pow_y;
                                    auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                    auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                    
                                    size_t j_y_bis1 = N - iy_bis;
                                    auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                    auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                    dy = coeff_dy * ( py_bis1 - py_bis0 );
                                //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                                }
                                
                                
    #endif
                                //std::cout<<"pos "<<pos<<std::endl;
                                //std::cout<<"dx "<<dx<<std::endl;
                                //std::cout<<"dy "<<dy<<std::endl;
                                //std::cout<<"px "<<px<<std::endl;
                                //std::cout<<"py "<<py<<std::endl;
                                ret(pos) = dx*dy;
                                //ret(pos,1) = px*dy;
                                pos++;
                                
                                
                            }
                        }
                        else{ // Top side
                            for (int pow_x = N_partial; pow_x >= starter; pow_x-=(N_partial-starter)){
                                VT ix_bis = pow_x-1;
                                VT iy_bis = pow_y-1;
                                
    #ifdef POWER_CACHE
                                auto px = power_cache[2*pow_x];
                                auto py = power_cache[2*pow_y+1];
                                if ( pow_x == 0 )
                                    auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                else if ( pow_x == basis_degree )
                                    auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                else
                                    auto dx = coeff_dx*( power_cache_bis[2*ix_bis] - power_cache_bis[2*pow_x] );
                                
                                if ( pow_y == 0 )
                                    auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                                else if ( pow_y == basis_degree )
                                    auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                                else
                                    auto dy = coeff_dy*( power_cache_bis[2*iy_bis+1] - power_cache_bis[2*pow_y+1] );
    #else
                                
                                //size_t j_x = basis_degree - pow_x;
                                //size_t j_y = basis_degree - pow_y;
                                
                                //auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                                //auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                
                                //auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                                //auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                                
                                // DERIVATIVES
                                //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                                //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 ;
                                VT dx = 0.0 , dy = 0.0 ;
                                
                                if ( pow_x == 0 ){
                                    //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                    size_t j_x_bis0 = N - pow_x;
                                    auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                    auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                    dx = -coeff_dx * px_bis0 ;
                                    //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                    
                                }
                                else if ( pow_x == basis_degree ){
                                    //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                    size_t j_x_bis1 = N - ix_bis;
                                    auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                    auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                    dx = coeff_dx * px_bis1 ;
                                    //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                }
                                else{
                                    size_t j_x_bis0 = N - pow_x;
                                    auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                    auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                    
                                    size_t j_x_bis1 = N - ix_bis;
                                    auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                    auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                    dx = coeff_dx * (px_bis1 -px_bis0);
                                    //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                                }
                                
                                if ( pow_y == 0 ){
                                    //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                    size_t j_y_bis0 = N - pow_y;
                                    auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                    auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                   // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                    dy = -coeff_dy *  py_bis0 ;
                                    //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                                    
                                }
                                else if ( pow_y == basis_degree ){
                                    //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                    size_t j_y_bis1 = N - iy_bis;
                                    auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                    auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                    dy = coeff_dy * py_bis1  ;
                                    //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                                }
                                else{
                                    size_t j_y_bis0 = N - pow_y;
                                    auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                    auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                    
                                    size_t j_y_bis1 = N - iy_bis;
                                    auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                    auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                    dy = coeff_dy * ( py_bis1 - py_bis0 );
                                //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                                }
                                
                                
    #endif
                                
                                ret(pos) = dx*dy;
                                //ret(pos,1) = px*dy;
                                pos++;
                                
                                
                            } // loop for top side vertices
                            
                            
                            
                            
                        } // else bottom-top side
                        
                    } // end loop for vertices
                    
                    
                    // face 0 (BOTTOM)
                    for (size_t pow_x = starter + 1 ; pow_x <= N_partial - 1 ; pow_x++)
                    {
                        size_t pow_y = starter;
                        
                        VT ix_bis = pow_x-1;
                        VT iy_bis = pow_y-1;
                        
    #ifdef POWER_CACHE
                        auto px = power_cache[2*pow_x];
                        auto py = power_cache[2*pow_y+1];
                        if ( pow_x == 0 )
                            auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                        else if ( pow_x == basis_degree )
                            auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        else
                            auto dx = coeff_dx*( power_cache_bis[2*ix_bis] - power_cache_bis[2*pow_x] );
                        
                        if ( pow_y == 0 )
                            auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                        else if ( pow_y == basis_degree )
                            auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                        else
                            auto dy = coeff_dy*( power_cache_bis[2*iy_bis+1] - power_cache_bis[2*pow_y+1] );
    #else
                        
                        //size_t j_x = basis_degree - pow_x;
                        //size_t j_y = basis_degree - pow_y;
                        
                        //auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                        //auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                        
                        //auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                        //auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                        //std::cout<<"j_x Face0 "<<j_x<<std::endl;
                        //std::cout<<"coeff_n_x Face0 "<<coeff_n_x<<std::endl;
                        //std::cout<<"scaling_x Face0 "<<scaling_x<<std::endl;
                        //std::cout<<"pow_x Face0 "<<pow_x<<std::endl;
                        //std::cout<<"bx0 Face0 "<<bx0<<std::endl;
                        //std::cout<<"bx1 Face0 "<<bx1<<std::endl;
                        
                        
                        // DERIVATIVES
                        //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                        //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 ;
                        VT dx = 0.0 , dy = 0.0 ;
                        
                        if ( pow_x == 0 ){
                            //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                            size_t j_x_bis0 = N - pow_x;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            dx = -coeff_dx * px_bis0 ;
                            //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                            
                        }
                        else if ( pow_x == basis_degree ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_x_bis1 = N - ix_bis;
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_dx * px_bis1 ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        }
                        else{
                            size_t j_x_bis0 = N - pow_x;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            
                            size_t j_x_bis1 = N - ix_bis;
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_dx * (px_bis1 -px_bis0);
                            //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                        }
                        
                        if ( pow_y == 0 ){
                            //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                            size_t j_y_bis0 = N - pow_y;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                           // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                            dy = -coeff_dy *  py_bis0 ;
                            //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                            
                        }
                        else if ( pow_y == basis_degree ){
                            //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                            size_t j_y_bis1 = N - iy_bis;
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_dy * py_bis1  ;
                            //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                        }
                        else{
                            size_t j_y_bis0 = N - pow_y;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                            
                            size_t j_y_bis1 = N - iy_bis;
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_dy * ( py_bis1 - py_bis0 );
                        //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                        }
                        
    #endif
                        
                        ret(pos) = dx*dy;
                        //ret(pos,1) = px*dy;
                        pos++;
                        
                    }
                    
                    // face 1 (RIGHT)
                    for (size_t pow_y = starter + 1; pow_y <= N_partial - 1 ; pow_y++)
                    {
                        size_t pow_x = N_partial;
                        VT ix_bis = pow_x-1;
                        VT iy_bis = pow_y-1;
                        
    #ifdef POWER_CACHE
                        auto px = power_cache[2*pow_x];
                        auto py = power_cache[2*pow_y+1];
                        if ( pow_x == 0 )
                            auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                        else if ( pow_x == basis_degree )
                            auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        else
                            auto dx = coeff_dx*( power_cache_bis[2*ix_bis] - power_cache_bis[2*pow_x] );
                        
                        if ( pow_y == 0 )
                            auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                        else if ( pow_y == basis_degree )
                            auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                        else
                            auto dy = coeff_dy*( power_cache_bis[2*iy_bis+1] - power_cache_bis[2*pow_y+1] );
    #else
                        
                        //size_t j_x = basis_degree - pow_x;
                        //size_t j_y = basis_degree - pow_y;
                        
                        //auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                        //auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                        
                        //auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                        //auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                        
                        // DERIVATIVES
                        //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                        //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 ;
                        VT dx = 0.0 , dy = 0.0 ;
                        
                        if ( pow_x == 0 ){
                            //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                            size_t j_x_bis0 = N - pow_x;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            dx = -coeff_dx * px_bis0 ;
                            //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                            
                        }
                        else if ( pow_x == basis_degree ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_x_bis1 = N - ix_bis;
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_dx * px_bis1 ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        }
                        else{
                            size_t j_x_bis0 = N - pow_x;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            
                            size_t j_x_bis1 = N - ix_bis;
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_dx * (px_bis1 -px_bis0);
                            //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                        }
                        
                        if ( pow_y == 0 ){
                            //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                            size_t j_y_bis0 = N - pow_y;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                           // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                            dy = -coeff_dy *  py_bis0 ;
                            //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                            
                        }
                        else if ( pow_y == basis_degree ){
                            //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                            size_t j_y_bis1 = N - iy_bis;
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_dy * py_bis1  ;
                            //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                        }
                        else{
                            size_t j_y_bis0 = N - pow_y;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                            
                            size_t j_y_bis1 = N - iy_bis;
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_dy * ( py_bis1 - py_bis0 );
                        //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                        }
                        
                        
    #endif
                        
                        ret(pos) = dx*dy;
                        //ret(pos,1) = px*dy;
                        pos++;
                    }
                    
                    // face 2 (TOP)
                    for (size_t pow_x = N_partial - 1; pow_x >= starter + 1 ; pow_x--)
                    {
                        size_t pow_y = N_partial;
                        VT ix_bis = pow_x-1;
                        VT iy_bis = pow_y-1;
                        
    #ifdef POWER_CACHE
                        auto px = power_cache[2*pow_x];
                        auto py = power_cache[2*pow_y+1];
                        if ( pow_x == 0 )
                            auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                        else if ( pow_x == basis_degree )
                            auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        else
                            auto dx = coeff_dx*( power_cache_bis[2*ix_bis] - power_cache_bis[2*pow_x] );
                        
                        if ( pow_y == 0 )
                            auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                        else if ( pow_y == basis_degree )
                            auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                        else
                            auto dy = coeff_dy*( power_cache_bis[2*iy_bis+1] - power_cache_bis[2*pow_y+1] );
    #else
                        
                        //size_t j_x = basis_degree - pow_x;
                        //size_t j_y = basis_degree - pow_y;
                        
                        //auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                        //auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                        
                        //auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                        //auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                        
                        // DERIVATIVES
                        //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                        //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 ;
                        VT dx = 0.0 , dy = 0.0 ;
                        
                        if ( pow_x == 0 ){
                            //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                            size_t j_x_bis0 = N - pow_x;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            dx = -coeff_dx * px_bis0 ;
                            //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                            
                        }
                        else if ( pow_x == basis_degree ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_x_bis1 = N - ix_bis;
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_dx * px_bis1 ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        }
                        else{
                            size_t j_x_bis0 = N - pow_x;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            
                            size_t j_x_bis1 = N - ix_bis;
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_dx * (px_bis1 -px_bis0);
                            //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                        }
                        
                        if ( pow_y == 0 ){
                            //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                            size_t j_y_bis0 = N - pow_y;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                           // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                            dy = -coeff_dy *  py_bis0 ;
                            //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                            
                        }
                        else if ( pow_y == basis_degree ){
                            //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                            size_t j_y_bis1 = N - iy_bis;
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_dy * py_bis1  ;
                            //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                        }
                        else{
                            size_t j_y_bis0 = N - pow_y;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                            
                            size_t j_y_bis1 = N - iy_bis;
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_dy * ( py_bis1 - py_bis0 );
                        //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                        }
                        
                        
    #endif
                        
                        ret(pos) = dx*dy;
                        //ret(pos,1) = px*dy;
                        pos++;
                    }
                    
                    
                    // face 3 (LEFT)
                    
                    for (size_t pow_y = N_partial - 1; pow_y >= starter + 1 ; pow_y--)
                    {
                        size_t pow_x = starter;
                        VT ix_bis = pow_x-1;
                        VT iy_bis = pow_y-1;
                        
    #ifdef POWER_CACHE
                        auto px = power_cache[2*pow_x];
                        auto py = power_cache[2*pow_y+1];
                        if ( pow_x == 0 )
                            auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                        else if ( pow_x == basis_degree )
                            auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        else
                            auto dx = coeff_dx*( power_cache_bis[2*ix_bis] - power_cache_bis[2*pow_x] );
                        
                        if ( pow_y == 0 )
                            auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                        else if ( pow_y == basis_degree )
                            auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                        else
                            auto dy = coeff_dy*( power_cache_bis[2*iy_bis+1] - power_cache_bis[2*pow_y+1] );
    #else
                        
                        //size_t j_x = basis_degree - pow_x;
                        //size_t j_y = basis_degree - pow_y;
                        
                        //auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                        //auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                        
                        //auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                        //auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                        
                        // DERIVATIVES
                        //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                        //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 ;
                        VT dx = 0.0 , dy = 0.0 ;
                        
                        if ( pow_x == 0 ){
                            //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                            size_t j_x_bis0 = N - pow_x;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            dx = -coeff_dx * px_bis0 ;
                            //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                            
                        }
                        else if ( pow_x == basis_degree ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_x_bis1 = N - ix_bis;
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_dx * px_bis1 ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        }
                        else{
                            size_t j_x_bis0 = N - pow_x;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            
                            size_t j_x_bis1 = N - ix_bis;
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_dx * (px_bis1 -px_bis0);
                            //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                        }
                        
                        if ( pow_y == 0 ){
                            //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                            size_t j_y_bis0 = N - pow_y;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                           // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                            dy = -coeff_dy *  py_bis0 ;
                            //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                            
                        }
                        else if ( pow_y == basis_degree ){
                            //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                            size_t j_y_bis1 = N - iy_bis;
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_dy * py_bis1  ;
                            //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                        }
                        else{
                            size_t j_y_bis0 = N - pow_y;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                            
                            size_t j_y_bis1 = N - iy_bis;
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_dy * ( py_bis1 - py_bis0 );
                        //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                        }
                        
                        
    #endif
                        
                        ret(pos) = dx*dy;
                        //ret(pos,1) = px*dy;
                        pos++;
                    }
                    
                    N_partial--;
                    starter++;
                    internal_bases -= 2;
                    
                }
                
                // for B_k, with k even, I.E. odd number of bases: there is a central one.
                if( basis_degree % 2 == 0 )
                {
                    //  std::cout<<"N is "<<N_partial<< " and starter is "<<starter<<std::endl;
                    assert( N_partial == starter );
                    size_t pow_x = starter ;
                    size_t pow_y = starter ;
                    VT ix_bis = pow_x-1;
                    VT iy_bis = pow_y-1;
                    
    #ifdef POWER_CACHE
                    auto px = power_cache[2*pow_x];
                    auto py = power_cache[2*pow_y+1];
                    if ( pow_x == 0 )
                        auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                    else if ( pow_x == basis_degree )
                        auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                    else
                        auto dx = coeff_dx*( power_cache_bis[2*ix_bis] - power_cache_bis[2*pow_x] );
                    
                    if ( pow_y == 0 )
                        auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                    else if ( pow_y == basis_degree )
                        auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                    else
                        auto dy = coeff_dy*( power_cache_bis[2*iy_bis+1] - power_cache_bis[2*pow_y+1] );
    #else
                    
                    //size_t j_x = basis_degree - pow_x;
                    //size_t j_y = basis_degree - pow_y;
                    
                    //auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                    //auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                    
                    //auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                    //auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                    
                    // DERIVATIVES
                    //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                    //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 ;
                    VT dx = 0.0 , dy = 0.0 ;
                    
                    if ( pow_x == 0 ){
                        //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                        size_t j_x_bis0 = N - pow_x;
                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                        dx = -coeff_dx * px_bis0 ;
                        //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                        
                    }
                    else if ( pow_x == basis_degree ){
                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                        size_t j_x_bis1 = N - ix_bis;
                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                        dx = coeff_dx * px_bis1 ;
                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                    }
                    else{
                        size_t j_x_bis0 = N - pow_x;
                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                        
                        size_t j_x_bis1 = N - ix_bis;
                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                        dx = coeff_dx * (px_bis1 -px_bis0);
                        //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                    }
                    
                    if ( pow_y == 0 ){
                        //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                        size_t j_y_bis0 = N - pow_y;
                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                       // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                        dy = -coeff_dy *  py_bis0 ;
                        //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                        
                    }
                    else if ( pow_y == basis_degree ){
                        //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                        size_t j_y_bis1 = N - iy_bis;
                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                        dy = coeff_dy * py_bis1  ;
                        //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                    }
                    else{
                        size_t j_y_bis0 = N - pow_y;
                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                        
                        size_t j_y_bis1 = N - iy_bis;
                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                        dy = coeff_dy * ( py_bis1 - py_bis0 );
                    //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                    }
                    
                    
    #endif
                    
                    ret(pos) = dx*dy;
                    //ret(pos,1) = px*dy;
                    pos++;
                }
            }

              //  std::cout<<"GRADIENTE: pos is"<<pos<<" and basis size is "<< basis_size<<std::endl;
                assert(pos == basis_size);

                return ret;
            

        }
    
    
    
    
    // Same approach of eval
    Matrix<VT, Dynamic, 1>
    eval_divergence(const point_type& pt)
    {
                Matrix<VT, Dynamic, 1> ret = Matrix<VT, Dynamic, 1>::Zero(basis_size, 1);

                auto bx0 = pt.x() - min_x ;
                auto bx1 = max_x - pt.x() ;
                auto by0 = pt.y() - min_y ;
                auto by1 = max_y - pt.y() ;
                //The stuff below should be in the constructor, but gradient is used less times than eval.
                auto N = basis_degree - 2 ;
                auto coeff_d2x = basis_degree*(basis_degree-1)/((max_x - min_x)*(max_x - min_x));
                auto coeff_d2y = basis_degree*(basis_degree-1)/((max_y - min_y)*(max_y - min_y));
                auto scaling_x_bis = 1.0/(pow(max_x - min_x,N) ); // 1/(b-a)^(n-1)
                auto scaling_y_bis = 1.0/(pow(max_y - min_y,N) );
                //std::cout<<"scaling_x_bis "<<scaling_x_bis<<std::endl;
                
        #ifdef POWER_CACHE
                if ( power_cache.size() != (basis_degree+1)*2 )
                  power_cache.resize( (basis_degree+1)*2) ;
                
                if ( power_cache_bis.size() != (N+1)*2 )
                    power_cache_bis.resize( (N+1)*2) ;
                
                if ( binomial_coeff.size() != (basis_degree+1) )
                    binomial_coeff.resize( basis_degree+1 ) ;
                    
                if ( binomial_coeff_bis.size() != (N+1) )
                    binomial_coeff_bis.resize( N+1 ) ;
                
               // Construction of the exponenatial term for bernstein basis B^N and B^(N-2) (useful for derivative)
                for (size_t i = 0; i <= basis_degree ; i++)
                {
                    size_t j = basis_degree - i;
                    binomial_coeff[i] = binomial_coeff_fx( basis_degree , i );
                    power_cache[2*i]    = scaling_x *binomial_coeff[i]*iexp_pow(bx0, i)*iexp_pow(bx1, j);
                    power_cache[2*i+1]  = scaling_y *binomial_coeff[i]*iexp_pow(by0, i)*iexp_pow(by1, j);
                     
                    if( i < basis_degree - 1 )
                    {
                        size_t j_bis = N - i;
                        binomial_coeff_bis[i] = binomial_coeff_fx( N , i );
                        power_cache_bis[2*i] = scaling_x_bis * binomial_coeff_bis[i]*iexp_pow(bx0, i)*iexp_pow(bx1, j_bis);
                        power_cache_bis[2*i+1] = scaling_y_bis * binomial_coeff_bis[i]*iexp_pow(by0, i)*iexp_pow(by1, j_bis);
                       
                    }
                    
                }
        #endif

                size_t pos = 0;
                // Case FEM degree = 0
                if (basis_degree == 0 || basis_degree == 1){
                    VT zero = 0.0;
                    while(pos < basis_size){
                        ret(pos) =  zero;
                        pos++;
                    }
                }
                else{
                    // if degree FEM >= 2
                    //std::cout<<"Bernstein Gradient, basis >=2 "<<std::endl;
                    int N_partial = basis_degree ;
                    int starter = 0;
                    int internal_bases = basis_degree + 1 ;
                    
                    while(internal_bases > 1) // for each layout of internal node
                    {
                        for (int pow_y = starter; pow_y <= N_partial ; pow_y +=(N_partial-starter)){
                            
                            if(pow_y == starter ) // Bottom side
                            {
                                
                                for (int pow_x = starter; pow_x <= N_partial; pow_x+=(N_partial-starter))
                                {
                                    
                                    VT ix_bis = pow_x-1; // element i-1 for derivative
                                    VT iy_bis = pow_y-1; // element i-1 for derivative
                                    VT ix2_bis = pow_x-2; // element i-2 for derivative
                                    VT iy2_bis = pow_y-2; // element i-2 for derivative
                                    
        #ifdef POWER_CACHE
                                    std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                                    auto px = power_cache[2*pow_x];
                                    auto py = power_cache[2*pow_y+1];
                                    //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                                    if ( pow_x == 0 )
                                        auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                                    else if ( pow_x == basis_degree )
                                        auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                                    else if ( pow_x == 1 )
                                        auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                                    else if ( pow_x == basis_degree - 1 )
                                        auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                                    else
                                        auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                    
                                    if ( pow_y == 0 )
                                        auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                                    else if ( pow_y == basis_degree )
                                        auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                                    else if ( pow_y == 1 )
                                        auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                                    else if ( pow_y == basis_degree - 1 )
                                        auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                                    else
                                        auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                                    //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                    
        #else
                                    
                                    //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                                    size_t j_x = basis_degree - pow_x;
                                    size_t j_y = basis_degree - pow_y;
                                    //std::cout<<"j_x PRE "<<j_x<<std::endl;
                                    auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                                    auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                    //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                                    //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                                    //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                                    //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                                    //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                    
                                    auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                                    auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                                    //std::cout<<"px PRE "<<px<<std::endl;
                                    
                                    // DERIVATIVES
                                    //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                                    //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                                    VT dx = 0.0 , dy = 0.0 ;
                                    
                                    if ( pow_x == 0 ){
                                        //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - pow_x;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        dx = coeff_d2x * px_bis0 ;
                                        //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                        
                                    }
                                    else if ( pow_x == basis_degree ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis1 = N - ix2_bis;
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * px_bis1 ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else if ( pow_x == 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - pow_x;
                                        size_t j_x_bis1 = N - ix_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                        auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    
                                    else if ( pow_x == basis_degree - 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - ix_bis;
                                        size_t j_x_bis1 = N - ix2_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else{
                                        size_t j_x_bis0 = N - pow_x;
                                        size_t j_x_bis1 = N - ix_bis;
                                        size_t j_x_bis2 = N - ix2_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                        auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                        auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                        
                                        
                                        dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                        //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                                    }
                                    
                                    if ( pow_y == 0 ){
                                        //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                        size_t j_y_bis0 = N - pow_y;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                       // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                        dy = coeff_d2y *  py_bis0 ;
                                        //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                                        
                                    }
                                    else if ( pow_y == basis_degree ){
                                        //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                        size_t j_y_bis1 = N - iy2_bis;
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * py_bis1  ;
                                        //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                                    }
                                    else if ( pow_y == 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_y_bis0 = N - pow_y;
                                        size_t j_y_bis1 = N - iy_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                        auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else if ( pow_y == basis_degree - 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_y_bis0 = N - iy_bis;
                                        size_t j_y_bis1 = N - iy2_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else{
                                        size_t j_y_bis0 = N - pow_y;
                                        size_t j_y_bis1 = N - iy_bis;
                                        size_t j_y_bis2 = N - iy2_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                        auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                        auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                        
                                    
                                        dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                    //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                                    }
                                    
                                    
        #endif
                                    //std::cout<<"pos "<<pos<<std::endl;
                                    //std::cout<<"dx "<<dx<<std::endl;
                                    //std::cout<<"dy "<<dy<<std::endl;
                                    //std::cout<<"px "<<px<<std::endl;
                                    //std::cout<<"py "<<py<<std::endl;
                                    ret(pos) = dx*py + px*dy;
                                    pos++;
                                    
                                    
                                }
                            }
                            else{ // Top side
                                for (int pow_x = N_partial; pow_x >= starter; pow_x-=(N_partial-starter))
                                {
                                    
                                    VT ix_bis = pow_x-1; // element i-1 for derivative
                                    VT iy_bis = pow_y-1; // element i-1 for derivative
                                    VT ix2_bis = pow_x-2; // element i-2 for derivative
                                    VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                
            #ifdef POWER_CACHE
                                                            
                                    std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                                    auto px = power_cache[2*pow_x];
                                    auto py = power_cache[2*pow_y+1];
                                    //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                                    if ( pow_x == 0 )
                                        auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                                    else if ( pow_x == basis_degree )
                                        auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                                    else if ( pow_x == 1 )
                                        auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                                    else if ( pow_x == basis_degree - 1 )
                                        auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                                    else
                                        auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                                                
                                        
                                    if ( pow_y == 0 )
                                        auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                                    else if ( pow_y == basis_degree )
                                        auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                                    else if ( pow_y == 1 )
                                        auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                                    else if ( pow_y == basis_degree - 1 )
                                        auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                                    else
                                        auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                                    //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                                                
            #else
                                                                
                                    //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                                    size_t j_x = basis_degree - pow_x;
                                    size_t j_y = basis_degree - pow_y;
                                    //std::cout<<"j_x PRE "<<j_x<<std::endl;
                                    auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                                    auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                    //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                                    //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                                    //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                                    //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                                    //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                
                                    auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                                    auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                                    //std::cout<<"px PRE "<<px<<std::endl;
                                                                
                                    // DERIVATIVES
                                    //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                                    //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                                    VT dx = 0.0 , dy = 0.0 ;
                                                                
                                    if ( pow_x == 0 ){
                                                                
                                        //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - pow_x;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        dx = coeff_d2x * px_bis0 ;
                                        //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                                                    
                                    }
                                    else if ( pow_x == basis_degree ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis1 = N - ix2_bis;
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * px_bis1 ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else if ( pow_x == 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - pow_x;
                                        size_t j_x_bis1 = N - ix_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                        auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                                                
                                    
                                    else if ( pow_x == basis_degree - 1 ){
                                        
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - ix_bis;
                                        size_t j_x_bis1 = N - ix2_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else{
                                        size_t j_x_bis0 = N - pow_x;
                                        size_t j_x_bis1 = N - ix_bis;
                                        size_t j_x_bis2 = N - ix2_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                        auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                        auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                                                    
                                                                    
                                        dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                        //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                                    }
                                                            
                                                            
                                    if ( pow_y == 0 ){
                                        //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                        size_t j_y_bis0 = N - pow_y;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                        // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                        dy = coeff_d2y *  py_bis0 ;
                                        //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                                                                    
                                    }
                                    else if ( pow_y == basis_degree ){
                                        //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                        size_t j_y_bis1 = N - iy2_bis;
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * py_bis1  ;
                                        //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                                    }
                                    else if ( pow_y == 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_y_bis0 = N - pow_y;
                                        size_t j_y_bis1 = N - iy_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                        auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else if ( pow_y == basis_degree - 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_y_bis0 = N - iy_bis;
                                        size_t j_y_bis1 = N - iy2_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else{
                                        size_t j_y_bis0 = N - pow_y;
                                        size_t j_y_bis1 = N - iy_bis;
                                        size_t j_y_bis2 = N - iy2_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                        auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                        auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                                                    
                                                                
                                        dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                        
                                    }
                                                                
            #endif
                                    //std::cout<<"pos "<<pos<<std::endl;
                                    //std::cout<<"dx "<<dx<<std::endl;
                                    //std::cout<<"dy "<<dy<<std::endl;
                                    //std::cout<<"px "<<px<<std::endl;
                                    //std::cout<<"py "<<py<<std::endl;
                                    ret(pos) = dx*py + px*dy;
                                    pos++;
                                                                
                                                                
                                    
                                    
                                } // loop for top side vertices
                                
                                
                                
                                
                            } // else bottom-top side
                            
                        } // end loop for vertices
                        
                        
                        // face 0 (BOTTOM)
                        for (size_t pow_x = starter + 1 ; pow_x <= N_partial - 1 ; pow_x++)
                        {
                            size_t pow_y = starter;
                            VT ix_bis = pow_x-1; // element i-1 for derivative
                            VT iy_bis = pow_y-1; // element i-1 for derivative
                            VT ix2_bis = pow_x-2; // element i-2 for derivative
                            VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                                
    #ifdef POWER_CACHE
                                                                            
                            std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                            auto px = power_cache[2*pow_x];
                            auto py = power_cache[2*pow_y+1];
                            //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                            if ( pow_x == 0 )
                                auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                            else if ( pow_x == basis_degree )
                                auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                            else if ( pow_x == 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                            else if ( pow_x == basis_degree - 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                            else
                                auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                                                                
                                                    
                            if ( pow_y == 0 )
                                auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                            else if ( pow_y == basis_degree )
                                auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                            else if ( pow_y == 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                            else if ( pow_y == basis_degree - 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                            else
                                auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                            //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                                                                
            #else
                                                                            
                            //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                            size_t j_x = basis_degree - pow_x;
                            size_t j_y = basis_degree - pow_y;
                            //std::cout<<"j_x PRE "<<j_x<<std::endl;
                            auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                            auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                            //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                            //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                            //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                            //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                            //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                                
                            auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                            auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                            //std::cout<<"px PRE "<<px<<std::endl;
                                                                            
                            // DERIVATIVES
                            //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                            //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                            VT dx = 0.0 , dy = 0.0 ;
                                                                                
                            if ( pow_x == 0 ){
                                                                                
                                //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                dx = coeff_d2x * px_bis0 ;
                                //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                                                                    
                            }
                            else if ( pow_x == basis_degree ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * px_bis1 ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else if ( pow_x == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                                                                                
                            else if ( pow_x == basis_degree - 1 ){
                                                    
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_x_bis0 = N - ix_bis;
                            size_t j_x_bis1 = N - ix2_bis;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                            auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            
                            }
                            else{
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                size_t j_x_bis2 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                                                                
                                                                                    
                                dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                            }
                                                                            
                                                                            
                            
                            if ( pow_y == 0 ){
                                //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                dy = coeff_d2y *  py_bis0 ;
                                //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                                                                                    
                            }
                            else if ( pow_y == basis_degree ){
                                //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * py_bis1  ;
                                //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                            }
                            else if ( pow_y == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else if ( pow_y == basis_degree - 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - iy_bis;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else{
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                size_t j_y_bis2 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                                                                    
                                                                                
                                dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                                    
                            }
                                                                                
            #endif
                            //std::cout<<"pos "<<pos<<std::endl;
                            //std::cout<<"dx "<<dx<<std::endl;
                            //std::cout<<"dy "<<dy<<std::endl;
                            //std::cout<<"px "<<px<<std::endl;
                            //std::cout<<"py "<<py<<std::endl;
                            
                            ret(pos) = dx*py + px*dy;
                            pos++;
                                                                                
                                                        
                           
                            
                        }
                        
                        // face 1 (RIGHT) size_t pow_x = N_partial;
                        for (size_t pow_y = starter + 1; pow_y <= N_partial - 1 ; pow_y++)
                        {
                            size_t pow_x = N_partial;
                            VT ix_bis = pow_x-1; // element i-1 for derivative
                            
                            VT iy_bis = pow_y-1; // element i-1 for derivative
                            VT ix2_bis = pow_x-2; // element i-2 for derivative
                            VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                                                        
        #ifdef POWER_CACHE
                                                                                                    
         
                            std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                            auto px = power_cache[2*pow_x];
                            auto py = power_cache[2*pow_y+1];
                            //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                            if ( pow_x == 0 )
                                auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                            else if ( pow_x == basis_degree )
                                auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                            else if ( pow_x == 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                            else if ( pow_x == basis_degree - 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                            else
                                auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                                                                                        
                                                                            
                            if ( pow_y == 0 )
                                auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                            else if ( pow_y == basis_degree )
                                auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                            else if ( pow_y == 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                            else if ( pow_y == basis_degree - 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                            else
                                auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                            //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                                                                                        
            #else
                            //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                            size_t j_x = basis_degree - pow_x;
                            size_t j_y = basis_degree - pow_y;
                            //std::cout<<"j_x PRE "<<j_x<<std::endl;
                            auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                            auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                            //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                            //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                            //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                            //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                            //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                                                        
                            auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                            auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                            //std::cout<<"px PRE "<<px<<std::endl;
                                                                                                    
                            // DERIVATIVES
                            //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                            //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                            VT dx = 0.0 , dy = 0.0 ;
                                                                                                        
                            
                            if ( pow_x == 0 ){
                                //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                dx = coeff_d2x * px_bis0 ;
                                //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                                                                                            
                            }
                            else if ( pow_x == basis_degree ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * px_bis1 ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else if ( pow_x == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                                                                                                        
                            else if ( pow_x == basis_degree - 1 ){
                                                                        
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - ix_bis;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                                    
                            }
                            else{
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                size_t j_x_bis2 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                                                                                        
                                dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                            }
                            
                            if ( pow_y == 0 ){
                                //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                dy = coeff_d2y *  py_bis0 ;
                                //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                            }
                            
                            else if ( pow_y == basis_degree ){
                                //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * py_bis1  ;
                                //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                            }
                            else if ( pow_y == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else if ( pow_y == basis_degree - 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - iy_bis;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else{
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                size_t j_y_bis2 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                                                        
                            }
                                                                                                        
            #endif
                            //std::cout<<"pos "<<pos<<std::endl;
                            //std::cout<<"dx "<<dx<<std::endl;
                            //std::cout<<"dy "<<dy<<std::endl;
                            //std::cout<<"px "<<px<<std::endl;
                            //std::cout<<"py "<<py<<std::endl;
                                            
                            ret(pos) = dx*py + px*dy;
                            pos++;
                        }
                        
                        // face 2 (TOP)
                        for (size_t pow_x = N_partial - 1; pow_x >= starter + 1 ; pow_x--)
                        {
                            size_t pow_y = N_partial;
                            VT ix_bis = pow_x-1; // element i-1 for derivative
                            VT iy_bis = pow_y-1; // element i-1 for derivative
                            VT ix2_bis = pow_x-2; // element i-2 for derivative
                            VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                                                                            
        #ifdef POWER_CACHE
                                                                                                                        
                            std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                            auto px = power_cache[2*pow_x];
                            auto py = power_cache[2*pow_y+1];
                            //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                            if ( pow_x == 0 )
                                auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                            else if ( pow_x == basis_degree )
                                auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                            else if ( pow_x == 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                            else if ( pow_x == basis_degree - 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                            else
                                auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                                                                                                            
                                                                                                
                            if ( pow_y == 0 )
                                auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                            else if ( pow_y == basis_degree )
                                auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                            else if ( pow_y == 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                            else if ( pow_y == basis_degree - 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                            else
                                auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                            //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                                                                                                        
            #else
             
                            //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                            size_t j_x = basis_degree - pow_x;
                            size_t j_y = basis_degree - pow_y;
                            //std::cout<<"j_x PRE "<<j_x<<std::endl;
                            auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                            auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                            //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                            //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                            //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                            //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                            //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                                                                            
                            auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                            auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                            //std::cout<<"px PRE "<<px<<std::endl;
                                                                                                                    
                            //DERIVATIVES
                            //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                            //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                            VT dx = 0.0 , dy = 0.0 ;
                                                                                                                            
                                                
                            if ( pow_x == 0 ){
                                //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                dx = coeff_d2x * px_bis0 ;
                                //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                            }
                             
                            else if ( pow_x == basis_degree ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * px_bis1 ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                             
                            else if ( pow_x == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                                
                            else if ( pow_x == basis_degree - 1 ){
                                                                                            
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - ix_bis;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                                                    
                            }
                             
                            else{
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                size_t j_x_bis2 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                                                                                                        
                                dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                            }
                                                
                            if ( pow_y == 0 ){
                                //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                dy = coeff_d2y *  py_bis0 ;
                                //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                            }
                                                
                            else if ( pow_y == basis_degree ){
                                //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * py_bis1  ;
                                //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                            }
                            else if ( pow_y == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            
                            else if ( pow_y == basis_degree - 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - iy_bis;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else{
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                size_t j_y_bis2 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                                                                            
                            }
                                                                                                                            
            #endif
            
                            //std::cout<<"pos "<<pos<<std::endl;
                            //std::cout<<"dx "<<dx<<std::endl;
                            //std::cout<<"dy "<<dy<<std::endl;
                            //std::cout<<"px "<<px<<std::endl;
                            //std::cout<<"py "<<py<<std::endl;
                                                                
                            ret(pos) = dx*py + px*dy;
                            pos++;
                        }
                        
                        
                        // face 3 (LEFT)
                        
                        for (size_t pow_y = N_partial - 1; pow_y >= starter + 1 ; pow_y--)
                        {
                            size_t pow_x = starter;
                            VT ix_bis = pow_x-1; // element i-1 for derivative
                            VT iy_bis = pow_y-1; // element i-1 for derivative
                            VT ix2_bis = pow_x-2; // element i-2 for derivative
                            VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                                                                            
        #ifdef POWER_CACHE
                                                                                                                        
                             
                            std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                            auto px = power_cache[2*pow_x];
                            auto py = power_cache[2*pow_y+1];
                            //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                            if ( pow_x == 0 )
                                auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                            else if ( pow_x == basis_degree )
                                auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                            else if ( pow_x == 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                            else if ( pow_x == basis_degree - 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                            else
                                auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                                                                                                            
                                                                                                
                            if ( pow_y == 0 )
                                auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                            else if ( pow_y == basis_degree )
                                auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                            else if ( pow_y == 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                            else if ( pow_y == basis_degree - 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                            else
                                auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                            //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                                                                                                            
            #else
                            
                            //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                            size_t j_x = basis_degree - pow_x;
                            size_t j_y = basis_degree - pow_y;
                            //std::cout<<"j_x PRE "<<j_x<<std::endl;
                            auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                            auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                            //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                            //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                            //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                            //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                            //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                                                                            
                            auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                            auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                            //std::cout<<"px PRE "<<px<<std::endl;
                                                                                                                        
                            // DERIVATIVES
                            //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                            //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                            VT dx = 0.0 , dy = 0.0 ;
                                                                                                                            
                                                
                            if ( pow_x == 0 ){
                                //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                dx = coeff_d2x * px_bis0 ;
                                //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                            }
                            
                            else if ( pow_x == basis_degree ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * px_bis1 ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else if ( pow_x == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                                 
                            else if ( pow_x == basis_degree - 1 ){
                                                                                            
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - ix_bis;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                                                        
                            }
                            else{
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                size_t j_x_bis2 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                                                                                                            
                                dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                            }
                                                
                            if ( pow_y == 0 ){
                                //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                dy = coeff_d2y *  py_bis0 ;
                                //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                            }
                                                
                            else if ( pow_y == basis_degree ){
                                //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * py_bis1  ;
                                //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                            }
                            
                            else if ( pow_y == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                                    
                            else if ( pow_y == basis_degree - 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - iy_bis;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else{
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                size_t j_y_bis2 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                                                                            
                            }
                                                                                                                            
                #endif
                            //std::cout<<"pos "<<pos<<std::endl;
                            //std::cout<<"dx "<<dx<<std::endl;
                            //std::cout<<"dy "<<dy<<std::endl;
                            //std::cout<<"px "<<px<<std::endl;
                            //std::cout<<"py "<<py<<std::endl;
                                                                
                            ret(pos) = dx*py + px*dy;
                            pos++;
                        }
                        
                        N_partial--;
                        starter++;
                        internal_bases -= 2;
                        
                    }
                    
                    // for B_k, with k even, I.E. odd number of bases: there is a central one.
                    if( basis_degree % 2 == 0 )
                    {
                        //  std::cout<<"N is "<<N_partial<< " and starter is "<<starter<<std::endl;
                        assert( N_partial == starter );
                        size_t pow_x = starter ;
                        size_t pow_y = starter ;
                                         
                        VT ix_bis = pow_x-1; // element i-1 for derivative
                        VT iy_bis = pow_y-1; // element i-1 for derivative
                        VT ix2_bis = pow_x-2; // element i-2 for derivative
                        VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                                                                                            
        #ifdef POWER_CACHE
                                                                                                                                        
                                             
                        std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                        auto px = power_cache[2*pow_x];
                        auto py = power_cache[2*pow_y+1];
                        //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                        
                        if ( pow_x == 0 )
                            auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                        else if ( pow_x == basis_degree )
                            auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                        else if ( pow_x == 1 )
                            auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                        else if ( pow_x == basis_degree - 1 )
                            auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                        else
                            auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                       
                        if ( pow_y == 0 )
                            auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                        else if ( pow_y == basis_degree )
                            auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                        else if ( pow_y == 1 )
                            auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                        else if ( pow_y == basis_degree - 1 )
                            auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                        else
                            auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                        //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                    
    #else
                        //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                        size_t j_x = basis_degree - pow_x;
                        size_t j_y = basis_degree - pow_y;
                        //std::cout<<"j_x PRE "<<j_x<<std::endl;
                        auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                        auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                        //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                        //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                        //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                        //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                        //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                                                                                            
                        auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                        auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                        //std::cout<<"px PRE "<<px<<std::endl;
                                                                                                                                        
                        // DERIVATIVES
                        //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                        //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                        VT dx = 0.0 , dy = 0.0 ;
                                                                                                
                        if ( pow_x == 0 ){
                            //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                            size_t j_x_bis0 = N - pow_x;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            dx = coeff_d2x * px_bis0 ;
                            //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                        }
                                            
                        else if ( pow_x == basis_degree ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_x_bis1 = N - ix2_bis;
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_d2x * px_bis1 ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        }
                        else if ( pow_x == 1 ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_x_bis0 = N - pow_x;
                            size_t j_x_bis1 = N - ix_bis;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                            auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        }
                                                 
                        else if ( pow_x == basis_degree - 1 ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_x_bis0 = N - ix_bis;
                            size_t j_x_bis1 = N - ix2_bis;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                            auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                                                                    
                        }
                        else{
                            size_t j_x_bis0 = N - pow_x;
                            size_t j_x_bis1 = N - ix_bis;
                            size_t j_x_bis2 = N - ix2_bis;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                            auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                            auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                            auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                          
                            dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                            //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                        }
                                                                
                        
                        if ( pow_y == 0 ){
                            //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                            size_t j_y_bis0 = N - pow_y;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                            // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                            dy = coeff_d2y *  py_bis0 ;
                            //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                        }
                                                                
                        else if ( pow_y == basis_degree ){
                            //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                            size_t j_y_bis1 = N - iy2_bis;
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_d2y * py_bis1  ;
                            //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                        }
                                        
                        else if ( pow_y == 1 ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_y_bis0 = N - pow_y;
                            size_t j_y_bis1 = N - iy_bis;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                            auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        }
                                                    
                        else if ( pow_y == basis_degree - 1 ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_y_bis0 = N - iy_bis;
                            size_t j_y_bis1 = N - iy2_bis;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                            auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        }
                        
                        else{
                            size_t j_y_bis0 = N - pow_y;
                            size_t j_y_bis1 = N - iy_bis;
                            size_t j_y_bis2 = N - iy2_bis;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                            auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                            auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                            auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                            dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                            
                        }
                                                                                                
    #endif
                        //std::cout<<"pos "<<pos<<std::endl;
                        //std::cout<<"dx "<<dx<<std::endl;
                        //std::cout<<"dy "<<dy<<std::endl;
                        //std::cout<<"px "<<px<<std::endl;
                        //std::cout<<"py "<<py<<std::endl;
                                                                                
                        ret(pos) = dx*py + px*dy;
                        pos++;

                       
                        
                    }
                }

                  //  std::cout<<"GRADIENTE: pos is"<<pos<<" and basis size is "<< basis_size<<std::endl;
                    assert(pos == basis_size);

                    return ret;
                

    }
    
    // Same approach of eval
    Matrix<VT, Dynamic, 1>
    eval_double_derivative_x(const point_type& pt)
    {
                Matrix<VT, Dynamic, 1> ret = Matrix<VT, Dynamic, 1>::Zero(basis_size, 1);

                auto bx0 = pt.x() - min_x ;
                auto bx1 = max_x - pt.x() ;
                auto by0 = pt.y() - min_y ;
                auto by1 = max_y - pt.y() ;
                //The stuff below should be in the constructor, but gradient is used less times than eval.
                auto N = basis_degree - 2 ;
                auto coeff_d2x = basis_degree*(basis_degree-1)/((max_x - min_x)*(max_x - min_x));
                auto coeff_d2y = basis_degree*(basis_degree-1)/((max_y - min_y)*(max_y - min_y));
                auto scaling_x_bis = 1.0/(pow(max_x - min_x,N) ); // 1/(b-a)^(n-1)
                auto scaling_y_bis = 1.0/(pow(max_y - min_y,N) );
                //std::cout<<"scaling_x_bis "<<scaling_x_bis<<std::endl;
                
        #ifdef POWER_CACHE
                if ( power_cache.size() != (basis_degree+1)*2 )
                  power_cache.resize( (basis_degree+1)*2) ;
                
                if ( power_cache_bis.size() != (N+1)*2 )
                    power_cache_bis.resize( (N+1)*2) ;
                
                if ( binomial_coeff.size() != (basis_degree+1) )
                    binomial_coeff.resize( basis_degree+1 ) ;
                    
                if ( binomial_coeff_bis.size() != (N+1) )
                    binomial_coeff_bis.resize( N+1 ) ;
                
               // Construction of the exponenatial term for bernstein basis B^N and B^(N-2) (useful for derivative)
                for (size_t i = 0; i <= basis_degree ; i++)
                {
                    size_t j = basis_degree - i;
                    binomial_coeff[i] = binomial_coeff_fx( basis_degree , i );
                    power_cache[2*i]    = scaling_x *binomial_coeff[i]*iexp_pow(bx0, i)*iexp_pow(bx1, j);
                    power_cache[2*i+1]  = scaling_y *binomial_coeff[i]*iexp_pow(by0, i)*iexp_pow(by1, j);
                     
                    if( i < basis_degree - 1 )
                    {
                        size_t j_bis = N - i;
                        binomial_coeff_bis[i] = binomial_coeff_fx( N , i );
                        power_cache_bis[2*i] = scaling_x_bis * binomial_coeff_bis[i]*iexp_pow(bx0, i)*iexp_pow(bx1, j_bis);
                        power_cache_bis[2*i+1] = scaling_y_bis * binomial_coeff_bis[i]*iexp_pow(by0, i)*iexp_pow(by1, j_bis);
                       
                    }
                    
                }
        #endif

                size_t pos = 0;
                // Case FEM degree = 0
                if (basis_degree == 0 || basis_degree == 1){
                    VT zero = 0.0;
                    while(pos < basis_size){
                        ret(pos) =  zero;
                        pos++;
                    }
                }
                else{
                    // if degree FEM >= 2
                    //std::cout<<"Bernstein Gradient, basis >=2 "<<std::endl;
                    int N_partial = basis_degree ;
                    int starter = 0;
                    int internal_bases = basis_degree + 1 ;
                    
                    while(internal_bases > 1) // for each layout of internal node
                    {
                        for (int pow_y = starter; pow_y <= N_partial ; pow_y +=(N_partial-starter)){
                            
                            if(pow_y == starter ) // Bottom side
                            {
                                
                                for (int pow_x = starter; pow_x <= N_partial; pow_x+=(N_partial-starter))
                                {
                                    
                                    VT ix_bis = pow_x-1; // element i-1 for derivative
                                    VT iy_bis = pow_y-1; // element i-1 for derivative
                                    VT ix2_bis = pow_x-2; // element i-2 for derivative
                                    VT iy2_bis = pow_y-2; // element i-2 for derivative
                                    
        #ifdef POWER_CACHE
                                    std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                                    auto px = power_cache[2*pow_x];
                                    auto py = power_cache[2*pow_y+1];
                                    //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                                    if ( pow_x == 0 )
                                        auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                                    else if ( pow_x == basis_degree )
                                        auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                                    else if ( pow_x == 1 )
                                        auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                                    else if ( pow_x == basis_degree - 1 )
                                        auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                                    else
                                        auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                    
                                    if ( pow_y == 0 )
                                        auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                                    else if ( pow_y == basis_degree )
                                        auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                                    else if ( pow_y == 1 )
                                        auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                                    else if ( pow_y == basis_degree - 1 )
                                        auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                                    else
                                        auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                                    //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                    
        #else
                                    
                                    //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                                    //size_t j_x = basis_degree - pow_x;
                                    size_t j_y = basis_degree - pow_y;
                                    //std::cout<<"j_x PRE "<<j_x<<std::endl;
                                    //auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                                    auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                    //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                                    //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                                    //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                                    //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                                    //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                    
                                    //auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                                    auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                                    //std::cout<<"px PRE "<<px<<std::endl;
                                    
                                    // DERIVATIVES
                                    //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                                    //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                                    VT dx = 0.0 , dy = 0.0 ;
                                    
                                    if ( pow_x == 0 ){
                                        //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - pow_x;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        dx = coeff_d2x * px_bis0 ;
                                        //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                        
                                    }
                                    else if ( pow_x == basis_degree ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis1 = N - ix2_bis;
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * px_bis1 ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else if ( pow_x == 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - pow_x;
                                        size_t j_x_bis1 = N - ix_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                        auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    
                                    else if ( pow_x == basis_degree - 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - ix_bis;
                                        size_t j_x_bis1 = N - ix2_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else{
                                        size_t j_x_bis0 = N - pow_x;
                                        size_t j_x_bis1 = N - ix_bis;
                                        size_t j_x_bis2 = N - ix2_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                        auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                        auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                        
                                        
                                        dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                        //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                                    }
                                    
                                    if ( pow_y == 0 ){
                                        //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                        size_t j_y_bis0 = N - pow_y;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                       // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                        dy = coeff_d2y *  py_bis0 ;
                                        //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                                        
                                    }
                                    else if ( pow_y == basis_degree ){
                                        //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                        size_t j_y_bis1 = N - iy2_bis;
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * py_bis1  ;
                                        //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                                    }
                                    else if ( pow_y == 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_y_bis0 = N - pow_y;
                                        size_t j_y_bis1 = N - iy_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                        auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else if ( pow_y == basis_degree - 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_y_bis0 = N - iy_bis;
                                        size_t j_y_bis1 = N - iy2_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else{
                                        size_t j_y_bis0 = N - pow_y;
                                        size_t j_y_bis1 = N - iy_bis;
                                        size_t j_y_bis2 = N - iy2_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                        auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                        auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                        
                                    
                                        dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                    //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                                    }
                                    
                                    
        #endif
                                    //std::cout<<"pos "<<pos<<std::endl;
                                    //std::cout<<"dx "<<dx<<std::endl;
                                    //std::cout<<"dy "<<dy<<std::endl;
                                    //std::cout<<"px "<<px<<std::endl;
                                    //std::cout<<"py "<<py<<std::endl;
                                    ret(pos) = dx*py ; //+ px*dy;
                                    pos++;
                                    
                                    
                                }
                            }
                            else{ // Top side
                                for (int pow_x = N_partial; pow_x >= starter; pow_x-=(N_partial-starter))
                                {
                                    
                                    VT ix_bis = pow_x-1; // element i-1 for derivative
                                    VT iy_bis = pow_y-1; // element i-1 for derivative
                                    VT ix2_bis = pow_x-2; // element i-2 for derivative
                                    VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                
            #ifdef POWER_CACHE
                                                            
                                    std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                                    auto px = power_cache[2*pow_x];
                                    auto py = power_cache[2*pow_y+1];
                                    //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                                    if ( pow_x == 0 )
                                        auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                                    else if ( pow_x == basis_degree )
                                        auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                                    else if ( pow_x == 1 )
                                        auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                                    else if ( pow_x == basis_degree - 1 )
                                        auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                                    else
                                        auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                                                
                                        
                                    if ( pow_y == 0 )
                                        auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                                    else if ( pow_y == basis_degree )
                                        auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                                    else if ( pow_y == 1 )
                                        auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                                    else if ( pow_y == basis_degree - 1 )
                                        auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                                    else
                                        auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                                    //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                                                
            #else
                                                                
                                    //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                                    //size_t j_x = basis_degree - pow_x;
                                    size_t j_y = basis_degree - pow_y;
                                    //std::cout<<"j_x PRE "<<j_x<<std::endl;
                                    //auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                                    auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                    //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                                    //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                                    //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                                    //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                                    //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                
                                    //auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                                    auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                                    //std::cout<<"px PRE "<<px<<std::endl;
                                                                
                                    // DERIVATIVES
                                    //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                                    //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                                    VT dx = 0.0 , dy = 0.0 ;
                                                                
                                    if ( pow_x == 0 ){
                                                                
                                        //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - pow_x;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        dx = coeff_d2x * px_bis0 ;
                                        //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                                                    
                                    }
                                    else if ( pow_x == basis_degree ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis1 = N - ix2_bis;
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * px_bis1 ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else if ( pow_x == 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - pow_x;
                                        size_t j_x_bis1 = N - ix_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                        auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                                                
                                    
                                    else if ( pow_x == basis_degree - 1 ){
                                        
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - ix_bis;
                                        size_t j_x_bis1 = N - ix2_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else{
                                        size_t j_x_bis0 = N - pow_x;
                                        size_t j_x_bis1 = N - ix_bis;
                                        size_t j_x_bis2 = N - ix2_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                        auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                        auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                                                    
                                                                    
                                        dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                        //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                                    }
                                                            
                                                            
                                    if ( pow_y == 0 ){
                                        //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                        size_t j_y_bis0 = N - pow_y;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                        // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                        dy = coeff_d2y *  py_bis0 ;
                                        //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                                                                    
                                    }
                                    else if ( pow_y == basis_degree ){
                                        //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                        size_t j_y_bis1 = N - iy2_bis;
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * py_bis1  ;
                                        //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                                    }
                                    else if ( pow_y == 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_y_bis0 = N - pow_y;
                                        size_t j_y_bis1 = N - iy_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                        auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else if ( pow_y == basis_degree - 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_y_bis0 = N - iy_bis;
                                        size_t j_y_bis1 = N - iy2_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else{
                                        size_t j_y_bis0 = N - pow_y;
                                        size_t j_y_bis1 = N - iy_bis;
                                        size_t j_y_bis2 = N - iy2_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                        auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                        auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                                                    
                                                                
                                        dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                        
                                    }
                                                                
            #endif
                                    //std::cout<<"pos "<<pos<<std::endl;
                                    //std::cout<<"dx "<<dx<<std::endl;
                                    //std::cout<<"dy "<<dy<<std::endl;
                                    //std::cout<<"px "<<px<<std::endl;
                                    //std::cout<<"py "<<py<<std::endl;
                                    ret(pos) = dx*py ; // + px*dy;
                                    pos++;
                                                                
                                                                
                                    
                                    
                                } // loop for top side vertices
                                
                                
                                
                                
                            } // else bottom-top side
                            
                        } // end loop for vertices
                        
                        
                        // face 0 (BOTTOM)
                        for (size_t pow_x = starter + 1 ; pow_x <= N_partial - 1 ; pow_x++)
                        {
                            size_t pow_y = starter;
                            VT ix_bis = pow_x-1; // element i-1 for derivative
                            VT iy_bis = pow_y-1; // element i-1 for derivative
                            VT ix2_bis = pow_x-2; // element i-2 for derivative
                            VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                                
    #ifdef POWER_CACHE
                                                                            
                            std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                            auto px = power_cache[2*pow_x];
                            auto py = power_cache[2*pow_y+1];
                            //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                            if ( pow_x == 0 )
                                auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                            else if ( pow_x == basis_degree )
                                auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                            else if ( pow_x == 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                            else if ( pow_x == basis_degree - 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                            else
                                auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                                                                
                                                    
                            if ( pow_y == 0 )
                                auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                            else if ( pow_y == basis_degree )
                                auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                            else if ( pow_y == 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                            else if ( pow_y == basis_degree - 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                            else
                                auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                            //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                                                                
            #else
                                                                            
                            //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                            //size_t j_x = basis_degree - pow_x;
                            size_t j_y = basis_degree - pow_y;
                            //std::cout<<"j_x PRE "<<j_x<<std::endl;
                            //auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                            auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                            //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                            //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                            //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                            //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                            //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                                
                            //auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                            auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                            //std::cout<<"px PRE "<<px<<std::endl;
                                                                            
                            // DERIVATIVES
                            //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                            //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                            VT dx = 0.0 , dy = 0.0 ;
                                                                                
                            if ( pow_x == 0 ){
                                                                                
                                //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                dx = coeff_d2x * px_bis0 ;
                                //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                                                                    
                            }
                            else if ( pow_x == basis_degree ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * px_bis1 ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else if ( pow_x == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                                                                                
                            else if ( pow_x == basis_degree - 1 ){
                                                    
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_x_bis0 = N - ix_bis;
                            size_t j_x_bis1 = N - ix2_bis;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                            auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            
                            }
                            else{
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                size_t j_x_bis2 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                                                                
                                                                                    
                                dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                            }
                                                                            
                                                                            
                            
                            if ( pow_y == 0 ){
                                //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                dy = coeff_d2y *  py_bis0 ;
                                //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                                                                                    
                            }
                            else if ( pow_y == basis_degree ){
                                //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * py_bis1  ;
                                //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                            }
                            else if ( pow_y == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else if ( pow_y == basis_degree - 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - iy_bis;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else{
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                size_t j_y_bis2 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                                                                    
                                                                                
                                dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                                    
                            }
                                                                                
            #endif
                            //std::cout<<"pos "<<pos<<std::endl;
                            //std::cout<<"dx "<<dx<<std::endl;
                            //std::cout<<"dy "<<dy<<std::endl;
                            //std::cout<<"px "<<px<<std::endl;
                            //std::cout<<"py "<<py<<std::endl;
                            
                            ret(pos) = dx*py ; // + px*dy;
                            pos++;
                                                                                
                                                        
                           
                            
                        }
                        
                        // face 1 (RIGHT) size_t pow_x = N_partial;
                        for (size_t pow_y = starter + 1; pow_y <= N_partial - 1 ; pow_y++)
                        {
                            size_t pow_x = N_partial;
                            VT ix_bis = pow_x-1; // element i-1 for derivative
                            
                            VT iy_bis = pow_y-1; // element i-1 for derivative
                            VT ix2_bis = pow_x-2; // element i-2 for derivative
                            VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                                                        
        #ifdef POWER_CACHE
                                                                                                    
         
                            std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                            auto px = power_cache[2*pow_x];
                            auto py = power_cache[2*pow_y+1];
                            //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                            if ( pow_x == 0 )
                                auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                            else if ( pow_x == basis_degree )
                                auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                            else if ( pow_x == 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                            else if ( pow_x == basis_degree - 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                            else
                                auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                                                                                        
                                                                            
                            if ( pow_y == 0 )
                                auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                            else if ( pow_y == basis_degree )
                                auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                            else if ( pow_y == 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                            else if ( pow_y == basis_degree - 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                            else
                                auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                            //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                                                                                        
            #else
                            //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                            //size_t j_x = basis_degree - pow_x;
                            size_t j_y = basis_degree - pow_y;
                            //std::cout<<"j_x PRE "<<j_x<<std::endl;
                            //auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                            auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                            //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                            //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                            //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                            //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                            //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                                                        
                            //auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                            auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                            //std::cout<<"px PRE "<<px<<std::endl;
                                                                                                    
                            // DERIVATIVES
                            //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                            //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                            VT dx = 0.0 , dy = 0.0 ;
                                                                                                        
                            
                            if ( pow_x == 0 ){
                                //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                dx = coeff_d2x * px_bis0 ;
                                //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                                                                                            
                            }
                            else if ( pow_x == basis_degree ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * px_bis1 ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else if ( pow_x == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                                                                                                        
                            else if ( pow_x == basis_degree - 1 ){
                                                                        
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - ix_bis;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                                    
                            }
                            else{
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                size_t j_x_bis2 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                                                                                        
                                dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                            }
                            
                            if ( pow_y == 0 ){
                                //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                dy = coeff_d2y *  py_bis0 ;
                                //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                            }
                            
                            else if ( pow_y == basis_degree ){
                                //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * py_bis1  ;
                                //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                            }
                            else if ( pow_y == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else if ( pow_y == basis_degree - 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - iy_bis;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else{
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                size_t j_y_bis2 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                                                        
                            }
                                                                                                        
            #endif
                            //std::cout<<"pos "<<pos<<std::endl;
                            //std::cout<<"dx "<<dx<<std::endl;
                            //std::cout<<"dy "<<dy<<std::endl;
                            //std::cout<<"px "<<px<<std::endl;
                            //std::cout<<"py "<<py<<std::endl;
                                            
                            ret(pos) = dx*py ; // + px*dy;
                            pos++;
                        }
                        
                        // face 2 (TOP)
                        for (size_t pow_x = N_partial - 1; pow_x >= starter + 1 ; pow_x--)
                        {
                            size_t pow_y = N_partial;
                            VT ix_bis = pow_x-1; // element i-1 for derivative
                            VT iy_bis = pow_y-1; // element i-1 for derivative
                            VT ix2_bis = pow_x-2; // element i-2 for derivative
                            VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                                                                            
        #ifdef POWER_CACHE
                                                                                                                        
                            std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                            auto px = power_cache[2*pow_x];
                            auto py = power_cache[2*pow_y+1];
                            //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                            if ( pow_x == 0 )
                                auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                            else if ( pow_x == basis_degree )
                                auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                            else if ( pow_x == 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                            else if ( pow_x == basis_degree - 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                            else
                                auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                                                                                                            
                                                                                                
                            if ( pow_y == 0 )
                                auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                            else if ( pow_y == basis_degree )
                                auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                            else if ( pow_y == 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                            else if ( pow_y == basis_degree - 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                            else
                                auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                            //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                                                                                                        
            #else
             
                            //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                            //size_t j_x = basis_degree - pow_x;
                            size_t j_y = basis_degree - pow_y;
                            //std::cout<<"j_x PRE "<<j_x<<std::endl;
                            //auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                            auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                            //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                            //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                            //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                            //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                            //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                                                                            
                            //auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                            auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                            //std::cout<<"px PRE "<<px<<std::endl;
                                                                                                                    
                            //DERIVATIVES
                            //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                            //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                            VT dx = 0.0 , dy = 0.0 ;
                                                                                                                            
                                                
                            if ( pow_x == 0 ){
                                //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                dx = coeff_d2x * px_bis0 ;
                                //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                            }
                             
                            else if ( pow_x == basis_degree ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * px_bis1 ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                             
                            else if ( pow_x == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                                
                            else if ( pow_x == basis_degree - 1 ){
                                                                                            
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - ix_bis;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                                                    
                            }
                             
                            else{
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                size_t j_x_bis2 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                                                                                                        
                                dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                            }
                                                
                            if ( pow_y == 0 ){
                                //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                dy = coeff_d2y *  py_bis0 ;
                                //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                            }
                                                
                            else if ( pow_y == basis_degree ){
                                //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * py_bis1  ;
                                //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                            }
                            else if ( pow_y == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            
                            else if ( pow_y == basis_degree - 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - iy_bis;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else{
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                size_t j_y_bis2 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                                                                            
                            }
                                                                                                                            
            #endif
            
                            //std::cout<<"pos "<<pos<<std::endl;
                            //std::cout<<"dx "<<dx<<std::endl;
                            //std::cout<<"dy "<<dy<<std::endl;
                            //std::cout<<"px "<<px<<std::endl;
                            //std::cout<<"py "<<py<<std::endl;
                                                                
                            ret(pos) = dx*py ; // + px*dy;
                            pos++;
                        }
                        
                        
                        // face 3 (LEFT)
                        
                        for (size_t pow_y = N_partial - 1; pow_y >= starter + 1 ; pow_y--)
                        {
                            size_t pow_x = starter;
                            VT ix_bis = pow_x-1; // element i-1 for derivative
                            VT iy_bis = pow_y-1; // element i-1 for derivative
                            VT ix2_bis = pow_x-2; // element i-2 for derivative
                            VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                                                                            
        #ifdef POWER_CACHE
                                                                                                                        
                             
                            std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                            auto px = power_cache[2*pow_x];
                            auto py = power_cache[2*pow_y+1];
                            //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                            if ( pow_x == 0 )
                                auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                            else if ( pow_x == basis_degree )
                                auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                            else if ( pow_x == 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                            else if ( pow_x == basis_degree - 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                            else
                                auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                                                                                                            
                                                                                                
                            if ( pow_y == 0 )
                                auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                            else if ( pow_y == basis_degree )
                                auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                            else if ( pow_y == 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                            else if ( pow_y == basis_degree - 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                            else
                                auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                            //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                                                                                                            
            #else
                            
                            //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                            //size_t j_x = basis_degree - pow_x;
                            size_t j_y = basis_degree - pow_y;
                            //std::cout<<"j_x PRE "<<j_x<<std::endl;
                            //auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                            auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                            //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                            //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                            //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                            //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                            //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                                                                            
                            //auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                            auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                            //std::cout<<"px PRE "<<px<<std::endl;
                                                                                                                        
                            // DERIVATIVES
                            //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                            //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                            VT dx = 0.0 , dy = 0.0 ;
                                                                                                                            
                                                
                            if ( pow_x == 0 ){
                                //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                dx = coeff_d2x * px_bis0 ;
                                //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                            }
                            
                            else if ( pow_x == basis_degree ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * px_bis1 ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else if ( pow_x == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                                 
                            else if ( pow_x == basis_degree - 1 ){
                                                                                            
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - ix_bis;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                                                        
                            }
                            else{
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                size_t j_x_bis2 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                                                                                                            
                                dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                            }
                                                
                            if ( pow_y == 0 ){
                                //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                dy = coeff_d2y *  py_bis0 ;
                                //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                            }
                                                
                            else if ( pow_y == basis_degree ){
                                //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * py_bis1  ;
                                //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                            }
                            
                            else if ( pow_y == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                                    
                            else if ( pow_y == basis_degree - 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - iy_bis;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else{
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                size_t j_y_bis2 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                                                                            
                            }
                                                                                                                            
                #endif
                            //std::cout<<"pos "<<pos<<std::endl;
                            //std::cout<<"dx "<<dx<<std::endl;
                            //std::cout<<"dy "<<dy<<std::endl;
                            //std::cout<<"px "<<px<<std::endl;
                            //std::cout<<"py "<<py<<std::endl;
                                                                
                            ret(pos) = dx*py ; // + px*dy;
                            pos++;
                        }
                        
                        N_partial--;
                        starter++;
                        internal_bases -= 2;
                        
                    }
                    
                    // for B_k, with k even, I.E. odd number of bases: there is a central one.
                    if( basis_degree % 2 == 0 )
                    {
                        //  std::cout<<"N is "<<N_partial<< " and starter is "<<starter<<std::endl;
                        assert( N_partial == starter );
                        size_t pow_x = starter ;
                        size_t pow_y = starter ;
                                         
                        VT ix_bis = pow_x-1; // element i-1 for derivative
                        VT iy_bis = pow_y-1; // element i-1 for derivative
                        VT ix2_bis = pow_x-2; // element i-2 for derivative
                        VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                                                                                            
        #ifdef POWER_CACHE
                                                                                                                                        
                                             
                        std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                        auto px = power_cache[2*pow_x];
                        auto py = power_cache[2*pow_y+1];
                        //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                        
                        if ( pow_x == 0 )
                            auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                        else if ( pow_x == basis_degree )
                            auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                        else if ( pow_x == 1 )
                            auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                        else if ( pow_x == basis_degree - 1 )
                            auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                        else
                            auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                       
                        if ( pow_y == 0 )
                            auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                        else if ( pow_y == basis_degree )
                            auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                        else if ( pow_y == 1 )
                            auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                        else if ( pow_y == basis_degree - 1 )
                            auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                        else
                            auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                        //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                    
    #else
                        //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                        //size_t j_x = basis_degree - pow_x;
                        size_t j_y = basis_degree - pow_y;
                        //std::cout<<"j_x PRE "<<j_x<<std::endl;
                        //auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                        auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                        //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                        //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                        //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                        //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                        //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                                                                                            
                        //auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                        auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                        //std::cout<<"px PRE "<<px<<std::endl;
                                                                                                                                        
                        // DERIVATIVES
                        //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                        //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                        VT dx = 0.0 , dy = 0.0 ;
                                                                                                
                        if ( pow_x == 0 ){
                            //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                            size_t j_x_bis0 = N - pow_x;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            dx = coeff_d2x * px_bis0 ;
                            //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                        }
                                            
                        else if ( pow_x == basis_degree ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_x_bis1 = N - ix2_bis;
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_d2x * px_bis1 ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        }
                        else if ( pow_x == 1 ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_x_bis0 = N - pow_x;
                            size_t j_x_bis1 = N - ix_bis;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                            auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        }
                                                 
                        else if ( pow_x == basis_degree - 1 ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_x_bis0 = N - ix_bis;
                            size_t j_x_bis1 = N - ix2_bis;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                            auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                                                                    
                        }
                        else{
                            size_t j_x_bis0 = N - pow_x;
                            size_t j_x_bis1 = N - ix_bis;
                            size_t j_x_bis2 = N - ix2_bis;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                            auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                            auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                            auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                          
                            dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                            //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                        }
                                                                
                        
                        if ( pow_y == 0 ){
                            //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                            size_t j_y_bis0 = N - pow_y;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                            // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                            dy = coeff_d2y *  py_bis0 ;
                            //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                        }
                                                                
                        else if ( pow_y == basis_degree ){
                            //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                            size_t j_y_bis1 = N - iy2_bis;
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_d2y * py_bis1  ;
                            //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                        }
                                        
                        else if ( pow_y == 1 ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_y_bis0 = N - pow_y;
                            size_t j_y_bis1 = N - iy_bis;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                            auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        }
                                                    
                        else if ( pow_y == basis_degree - 1 ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_y_bis0 = N - iy_bis;
                            size_t j_y_bis1 = N - iy2_bis;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                            auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        }
                        
                        else{
                            size_t j_y_bis0 = N - pow_y;
                            size_t j_y_bis1 = N - iy_bis;
                            size_t j_y_bis2 = N - iy2_bis;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                            auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                            auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                            auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                            dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                            
                        }
                                                                                                
    #endif
                        //std::cout<<"pos "<<pos<<std::endl;
                        //std::cout<<"dx "<<dx<<std::endl;
                        //std::cout<<"dy "<<dy<<std::endl;
                        //std::cout<<"px "<<px<<std::endl;
                        //std::cout<<"py "<<py<<std::endl;
                                                                                
                        ret(pos) = dx*py ; // + px*dy;
                        pos++;

                       
                        
                    }
                }

                  //  std::cout<<"GRADIENTE: pos is"<<pos<<" and basis size is "<< basis_size<<std::endl;
                    assert(pos == basis_size);

                    return ret;
                

    }
    
    // Same approach of eval
    Matrix<VT, Dynamic, 1>
    eval_double_derivative_y(const point_type& pt)
    {
                Matrix<VT, Dynamic, 1> ret = Matrix<VT, Dynamic, 1>::Zero(basis_size, 1);

                auto bx0 = pt.x() - min_x ;
                auto bx1 = max_x - pt.x() ;
                auto by0 = pt.y() - min_y ;
                auto by1 = max_y - pt.y() ;
                //The stuff below should be in the constructor, but gradient is used less times than eval.
                auto N = basis_degree - 2 ;
                auto coeff_d2x = basis_degree*(basis_degree-1)/((max_x - min_x)*(max_x - min_x));
                auto coeff_d2y = basis_degree*(basis_degree-1)/((max_y - min_y)*(max_y - min_y));
                auto scaling_x_bis = 1.0/(pow(max_x - min_x,N) ); // 1/(b-a)^(n-1)
                auto scaling_y_bis = 1.0/(pow(max_y - min_y,N) );
                //std::cout<<"scaling_x_bis "<<scaling_x_bis<<std::endl;
                
        #ifdef POWER_CACHE
                if ( power_cache.size() != (basis_degree+1)*2 )
                  power_cache.resize( (basis_degree+1)*2) ;
                
                if ( power_cache_bis.size() != (N+1)*2 )
                    power_cache_bis.resize( (N+1)*2) ;
                
                if ( binomial_coeff.size() != (basis_degree+1) )
                    binomial_coeff.resize( basis_degree+1 ) ;
                    
                if ( binomial_coeff_bis.size() != (N+1) )
                    binomial_coeff_bis.resize( N+1 ) ;
                
               // Construction of the exponenatial term for bernstein basis B^N and B^(N-2) (useful for derivative)
                for (size_t i = 0; i <= basis_degree ; i++)
                {
                    size_t j = basis_degree - i;
                    binomial_coeff[i] = binomial_coeff_fx( basis_degree , i );
                    power_cache[2*i]    = scaling_x *binomial_coeff[i]*iexp_pow(bx0, i)*iexp_pow(bx1, j);
                    power_cache[2*i+1]  = scaling_y *binomial_coeff[i]*iexp_pow(by0, i)*iexp_pow(by1, j);
                     
                    if( i < basis_degree - 1 )
                    {
                        size_t j_bis = N - i;
                        binomial_coeff_bis[i] = binomial_coeff_fx( N , i );
                        power_cache_bis[2*i] = scaling_x_bis * binomial_coeff_bis[i]*iexp_pow(bx0, i)*iexp_pow(bx1, j_bis);
                        power_cache_bis[2*i+1] = scaling_y_bis * binomial_coeff_bis[i]*iexp_pow(by0, i)*iexp_pow(by1, j_bis);
                       
                    }
                    
                }
        #endif

                size_t pos = 0;
                // Case FEM degree = 0
                if (basis_degree == 0 || basis_degree == 1){
                    VT zero = 0.0;
                    while(pos < basis_size){
                        ret(pos) =  zero;
                        pos++;
                    }
                }
                else{
                    // if degree FEM >= 2
                    //std::cout<<"Bernstein Gradient, basis >=2 "<<std::endl;
                    int N_partial = basis_degree ;
                    int starter = 0;
                    int internal_bases = basis_degree + 1 ;
                    
                    while(internal_bases > 1) // for each layout of internal node
                    {
                        for (int pow_y = starter; pow_y <= N_partial ; pow_y +=(N_partial-starter)){
                            
                            if(pow_y == starter ) // Bottom side
                            {
                                
                                for (int pow_x = starter; pow_x <= N_partial; pow_x+=(N_partial-starter))
                                {
                                    
                                    VT ix_bis = pow_x-1; // element i-1 for derivative
                                    VT iy_bis = pow_y-1; // element i-1 for derivative
                                    VT ix2_bis = pow_x-2; // element i-2 for derivative
                                    VT iy2_bis = pow_y-2; // element i-2 for derivative
                                    
        #ifdef POWER_CACHE
                                    std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                                    auto px = power_cache[2*pow_x];
                                    auto py = power_cache[2*pow_y+1];
                                    //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                                    if ( pow_x == 0 )
                                        auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                                    else if ( pow_x == basis_degree )
                                        auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                                    else if ( pow_x == 1 )
                                        auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                                    else if ( pow_x == basis_degree - 1 )
                                        auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                                    else
                                        auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                    
                                    if ( pow_y == 0 )
                                        auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                                    else if ( pow_y == basis_degree )
                                        auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                                    else if ( pow_y == 1 )
                                        auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                                    else if ( pow_y == basis_degree - 1 )
                                        auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                                    else
                                        auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                                    //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                    
        #else
                                    
                                    //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                                    size_t j_x = basis_degree - pow_x;
                                    //size_t j_y = basis_degree - pow_y;
                                    //std::cout<<"j_x PRE "<<j_x<<std::endl;
                                    auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                                    //auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                    //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                                    //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                                    //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                                    //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                                    //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                    
                                    auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                                    //auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                                    //std::cout<<"px PRE "<<px<<std::endl;
                                    
                                    // DERIVATIVES
                                    //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                                    //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                                    VT dx = 0.0 , dy = 0.0 ;
                                    
                                    if ( pow_x == 0 ){
                                        //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - pow_x;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        dx = coeff_d2x * px_bis0 ;
                                        //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                        
                                    }
                                    else if ( pow_x == basis_degree ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis1 = N - ix2_bis;
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * px_bis1 ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else if ( pow_x == 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - pow_x;
                                        size_t j_x_bis1 = N - ix_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                        auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    
                                    else if ( pow_x == basis_degree - 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - ix_bis;
                                        size_t j_x_bis1 = N - ix2_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else{
                                        size_t j_x_bis0 = N - pow_x;
                                        size_t j_x_bis1 = N - ix_bis;
                                        size_t j_x_bis2 = N - ix2_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                        auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                        auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                        
                                        
                                        dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                        //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                                    }
                                    
                                    if ( pow_y == 0 ){
                                        //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                        size_t j_y_bis0 = N - pow_y;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                       // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                        dy = coeff_d2y *  py_bis0 ;
                                        //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                                        
                                    }
                                    else if ( pow_y == basis_degree ){
                                        //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                        size_t j_y_bis1 = N - iy2_bis;
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * py_bis1  ;
                                        //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                                    }
                                    else if ( pow_y == 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_y_bis0 = N - pow_y;
                                        size_t j_y_bis1 = N - iy_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                        auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else if ( pow_y == basis_degree - 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_y_bis0 = N - iy_bis;
                                        size_t j_y_bis1 = N - iy2_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else{
                                        size_t j_y_bis0 = N - pow_y;
                                        size_t j_y_bis1 = N - iy_bis;
                                        size_t j_y_bis2 = N - iy2_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                        auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                        auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                        
                                    
                                        dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                    //auto dy=coeff_dy*(power_cache_bis[2*iy_bis+1]-power_cache_bis[2*pow_y+1]);
                                    }
                                    
                                    
        #endif
                                    //std::cout<<"pos "<<pos<<std::endl;
                                    //std::cout<<"dx "<<dx<<std::endl;
                                    //std::cout<<"dy "<<dy<<std::endl;
                                    //std::cout<<"px "<<px<<std::endl;
                                    //std::cout<<"py "<<py<<std::endl;
                                    ret(pos) = px*dy;
                                    pos++;
                                    
                                    
                                }
                            }
                            else{ // Top side
                                for (int pow_x = N_partial; pow_x >= starter; pow_x-=(N_partial-starter))
                                {
                                    
                                    VT ix_bis = pow_x-1; // element i-1 for derivative
                                    VT iy_bis = pow_y-1; // element i-1 for derivative
                                    VT ix2_bis = pow_x-2; // element i-2 for derivative
                                    VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                
            #ifdef POWER_CACHE
                                                            
                                    std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                                    auto px = power_cache[2*pow_x];
                                    auto py = power_cache[2*pow_y+1];
                                    //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                                    if ( pow_x == 0 )
                                        auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                                    else if ( pow_x == basis_degree )
                                        auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                                    else if ( pow_x == 1 )
                                        auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                                    else if ( pow_x == basis_degree - 1 )
                                        auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                                    else
                                        auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                                                
                                        
                                    if ( pow_y == 0 )
                                        auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                                    else if ( pow_y == basis_degree )
                                        auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                                    else if ( pow_y == 1 )
                                        auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                                    else if ( pow_y == basis_degree - 1 )
                                        auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                                    else
                                        auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                                    //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                                                
            #else
                                                                
                                    //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                                    size_t j_x = basis_degree - pow_x;
                                    //size_t j_y = basis_degree - pow_y;
                                    //std::cout<<"j_x PRE "<<j_x<<std::endl;
                                    auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                                    //auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                                    //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                                    //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                                    //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                                    //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                                    //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                
                                    auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                                    //auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                                    //std::cout<<"px PRE "<<px<<std::endl;
                                                                
                                    // DERIVATIVES
                                    //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                                    //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                                    VT dx = 0.0 , dy = 0.0 ;
                                                                
                                    if ( pow_x == 0 ){
                                                                
                                        //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - pow_x;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        dx = coeff_d2x * px_bis0 ;
                                        //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                                                    
                                    }
                                    else if ( pow_x == basis_degree ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis1 = N - ix2_bis;
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * px_bis1 ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else if ( pow_x == 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - pow_x;
                                        size_t j_x_bis1 = N - ix_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                        auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                                                
                                    
                                    else if ( pow_x == basis_degree - 1 ){
                                        
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_x_bis0 = N - ix_bis;
                                        size_t j_x_bis1 = N - ix2_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                        dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else{
                                        size_t j_x_bis0 = N - pow_x;
                                        size_t j_x_bis1 = N - ix_bis;
                                        size_t j_x_bis2 = N - ix2_bis;
                                        auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                        auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                        auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                        auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                        auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                        auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                                                    
                                                                    
                                        dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                        //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                                    }
                                                            
                                                            
                                    if ( pow_y == 0 ){
                                        //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                        size_t j_y_bis0 = N - pow_y;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                        // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                        dy = coeff_d2y *  py_bis0 ;
                                        //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                                                                    
                                    }
                                    else if ( pow_y == basis_degree ){
                                        //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                        size_t j_y_bis1 = N - iy2_bis;
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * py_bis1  ;
                                        //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                                    }
                                    else if ( pow_y == 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_y_bis0 = N - pow_y;
                                        size_t j_y_bis1 = N - iy_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                        auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else if ( pow_y == basis_degree - 1 ){
                                        //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                        size_t j_y_bis0 = N - iy_bis;
                                        size_t j_y_bis1 = N - iy2_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                        dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                        //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                    }
                                    else{
                                        size_t j_y_bis0 = N - pow_y;
                                        size_t j_y_bis1 = N - iy_bis;
                                        size_t j_y_bis2 = N - iy2_bis;
                                        auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                        auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                        auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                        auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                        auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                        auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                                                    
                                                                
                                        dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                        
                                    }
                                                                
            #endif
                                    //std::cout<<"pos "<<pos<<std::endl;
                                    //std::cout<<"dx "<<dx<<std::endl;
                                    //std::cout<<"dy "<<dy<<std::endl;
                                    //std::cout<<"px "<<px<<std::endl;
                                    //std::cout<<"py "<<py<<std::endl;
                                    ret(pos) = px*dy;
                                    pos++;
                                                                
                                                                
                                    
                                    
                                } // loop for top side vertices
                                
                                
                                
                                
                            } // else bottom-top side
                            
                        } // end loop for vertices
                        
                        
                        // face 0 (BOTTOM)
                        for (size_t pow_x = starter + 1 ; pow_x <= N_partial - 1 ; pow_x++)
                        {
                            size_t pow_y = starter;
                            VT ix_bis = pow_x-1; // element i-1 for derivative
                            VT iy_bis = pow_y-1; // element i-1 for derivative
                            VT ix2_bis = pow_x-2; // element i-2 for derivative
                            VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                                
    #ifdef POWER_CACHE
                                                                            
                            std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                            auto px = power_cache[2*pow_x];
                            auto py = power_cache[2*pow_y+1];
                            //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                            if ( pow_x == 0 )
                                auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                            else if ( pow_x == basis_degree )
                                auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                            else if ( pow_x == 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                            else if ( pow_x == basis_degree - 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                            else
                                auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                                                                
                                                    
                            if ( pow_y == 0 )
                                auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                            else if ( pow_y == basis_degree )
                                auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                            else if ( pow_y == 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                            else if ( pow_y == basis_degree - 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                            else
                                auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                            //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                                                                
            #else
                                                                            
                            //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                            size_t j_x = basis_degree - pow_x;
                            //size_t j_y = basis_degree - pow_y;
                            //std::cout<<"j_x PRE "<<j_x<<std::endl;
                            auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                            //auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                            //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                            //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                            //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                            //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                            //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                                
                            auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                            //auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                            //std::cout<<"px PRE "<<px<<std::endl;
                                                                            
                            // DERIVATIVES
                            //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                            //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                            VT dx = 0.0 , dy = 0.0 ;
                                                                                
                            if ( pow_x == 0 ){
                                                                                
                                //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                dx = coeff_d2x * px_bis0 ;
                                //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                                                                    
                            }
                            else if ( pow_x == basis_degree ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * px_bis1 ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else if ( pow_x == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                                                                                
                            else if ( pow_x == basis_degree - 1 ){
                                                    
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_x_bis0 = N - ix_bis;
                            size_t j_x_bis1 = N - ix2_bis;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                            auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            
                            }
                            else{
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                size_t j_x_bis2 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                                                                
                                                                                    
                                dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                            }
                                                                            
                                                                            
                            
                            if ( pow_y == 0 ){
                                //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                dy = coeff_d2y *  py_bis0 ;
                                //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                                                                                    
                            }
                            else if ( pow_y == basis_degree ){
                                //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * py_bis1  ;
                                //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                            }
                            else if ( pow_y == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else if ( pow_y == basis_degree - 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - iy_bis;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else{
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                size_t j_y_bis2 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                                                                    
                                                                                
                                dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                                    
                            }
                                                                                
            #endif
                            //std::cout<<"pos "<<pos<<std::endl;
                            //std::cout<<"dx "<<dx<<std::endl;
                            //std::cout<<"dy "<<dy<<std::endl;
                            //std::cout<<"px "<<px<<std::endl;
                            //std::cout<<"py "<<py<<std::endl;
                            
                            ret(pos) = px*dy;
                            pos++;
                                                                                
                                                        
                           
                            
                        }
                        
                        // face 1 (RIGHT) size_t pow_x = N_partial;
                        for (size_t pow_y = starter + 1; pow_y <= N_partial - 1 ; pow_y++)
                        {
                            size_t pow_x = N_partial;
                            VT ix_bis = pow_x-1; // element i-1 for derivative
                            
                            VT iy_bis = pow_y-1; // element i-1 for derivative
                            VT ix2_bis = pow_x-2; // element i-2 for derivative
                            VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                                                        
        #ifdef POWER_CACHE
                                                                                                    
         
                            std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                            auto px = power_cache[2*pow_x];
                            auto py = power_cache[2*pow_y+1];
                            //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                            if ( pow_x == 0 )
                                auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                            else if ( pow_x == basis_degree )
                                auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                            else if ( pow_x == 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                            else if ( pow_x == basis_degree - 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                            else
                                auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                                                                                        
                                                                            
                            if ( pow_y == 0 )
                                auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                            else if ( pow_y == basis_degree )
                                auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                            else if ( pow_y == 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                            else if ( pow_y == basis_degree - 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                            else
                                auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                            //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                                                                                        
            #else
                            //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                            size_t j_x = basis_degree - pow_x;
                            //size_t j_y = basis_degree - pow_y;
                            //std::cout<<"j_x PRE "<<j_x<<std::endl;
                            auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                            //auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                            //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                            //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                            //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                            //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                            //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                                                        
                            auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                            //auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                            //std::cout<<"px PRE "<<px<<std::endl;
                                                                                                    
                            // DERIVATIVES
                            //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                            //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                            VT dx = 0.0 , dy = 0.0 ;
                                                                                                        
                            
                            if ( pow_x == 0 ){
                                //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                dx = coeff_d2x * px_bis0 ;
                                //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                                                                                                            
                            }
                            else if ( pow_x == basis_degree ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * px_bis1 ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else if ( pow_x == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                                                                                                        
                            else if ( pow_x == basis_degree - 1 ){
                                                                        
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - ix_bis;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                                    
                            }
                            else{
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                size_t j_x_bis2 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                                                                                        
                                dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                            }
                            
                            if ( pow_y == 0 ){
                                //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                dy = coeff_d2y *  py_bis0 ;
                                //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                            }
                            
                            else if ( pow_y == basis_degree ){
                                //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * py_bis1  ;
                                //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                            }
                            else if ( pow_y == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else if ( pow_y == basis_degree - 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - iy_bis;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else{
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                size_t j_y_bis2 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                                                        
                            }
                                                                                                        
            #endif
                            //std::cout<<"pos "<<pos<<std::endl;
                            //std::cout<<"dx "<<dx<<std::endl;
                            //std::cout<<"dy "<<dy<<std::endl;
                            //std::cout<<"px "<<px<<std::endl;
                            //std::cout<<"py "<<py<<std::endl;
                                            
                            ret(pos) = px*dy;
                            pos++;
                        }
                        
                        // face 2 (TOP)
                        for (size_t pow_x = N_partial - 1; pow_x >= starter + 1 ; pow_x--)
                        {
                            size_t pow_y = N_partial;
                            VT ix_bis = pow_x-1; // element i-1 for derivative
                            VT iy_bis = pow_y-1; // element i-1 for derivative
                            VT ix2_bis = pow_x-2; // element i-2 for derivative
                            VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                                                                            
        #ifdef POWER_CACHE
                                                                                                                        
                            std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                            auto px = power_cache[2*pow_x];
                            auto py = power_cache[2*pow_y+1];
                            //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                            if ( pow_x == 0 )
                                auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                            else if ( pow_x == basis_degree )
                                auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                            else if ( pow_x == 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                            else if ( pow_x == basis_degree - 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                            else
                                auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                                                                                                            
                                                                                                
                            if ( pow_y == 0 )
                                auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                            else if ( pow_y == basis_degree )
                                auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                            else if ( pow_y == 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                            else if ( pow_y == basis_degree - 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                            else
                                auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                            //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                                                                                                        
            #else
             
                            //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                            size_t j_x = basis_degree - pow_x;
                            //size_t j_y = basis_degree - pow_y;
                            //std::cout<<"j_x PRE "<<j_x<<std::endl;
                            auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                            //auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                            //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                            //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                            //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                            //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                            //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                                                                            
                            auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                            //auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                            //std::cout<<"px PRE "<<px<<std::endl;
                                                                                                                    
                            //DERIVATIVES
                            //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                            //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                            VT dx = 0.0 , dy = 0.0 ;
                                                                                                                            
                                                
                            if ( pow_x == 0 ){
                                //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                dx = coeff_d2x * px_bis0 ;
                                //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                            }
                             
                            else if ( pow_x == basis_degree ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * px_bis1 ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                             
                            else if ( pow_x == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                                
                            else if ( pow_x == basis_degree - 1 ){
                                                                                            
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - ix_bis;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                                                    
                            }
                             
                            else{
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                size_t j_x_bis2 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                                                                                                        
                                dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                            }
                                                
                            if ( pow_y == 0 ){
                                //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                dy = coeff_d2y *  py_bis0 ;
                                //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                            }
                                                
                            else if ( pow_y == basis_degree ){
                                //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * py_bis1  ;
                                //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                            }
                            else if ( pow_y == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            
                            else if ( pow_y == basis_degree - 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - iy_bis;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else{
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                size_t j_y_bis2 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                                                                            
                            }
                                                                                                                            
            #endif
            
                            //std::cout<<"pos "<<pos<<std::endl;
                            //std::cout<<"dx "<<dx<<std::endl;
                            //std::cout<<"dy "<<dy<<std::endl;
                            //std::cout<<"px "<<px<<std::endl;
                            //std::cout<<"py "<<py<<std::endl;
                                                                
                            ret(pos) = px*dy;
                            pos++;
                        }
                        
                        
                        // face 3 (LEFT)
                        
                        for (size_t pow_y = N_partial - 1; pow_y >= starter + 1 ; pow_y--)
                        {
                            size_t pow_x = starter;
                            VT ix_bis = pow_x-1; // element i-1 for derivative
                            VT iy_bis = pow_y-1; // element i-1 for derivative
                            VT ix2_bis = pow_x-2; // element i-2 for derivative
                            VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                                                                            
        #ifdef POWER_CACHE
                                                                                                                        
                             
                            std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                            auto px = power_cache[2*pow_x];
                            auto py = power_cache[2*pow_y+1];
                            //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                            if ( pow_x == 0 )
                                auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                            else if ( pow_x == basis_degree )
                                auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                            else if ( pow_x == 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                            else if ( pow_x == basis_degree - 1 )
                                auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                            else
                                auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                                                                                                                            
                                                                                                
                            if ( pow_y == 0 )
                                auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                            else if ( pow_y == basis_degree )
                                auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                            else if ( pow_y == 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                            else if ( pow_y == basis_degree - 1 )
                                auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                            else
                                auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                            //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                                                                                                                            
            #else
                            
                            //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                            size_t j_x = basis_degree - pow_x;
                            //size_t j_y = basis_degree - pow_y;
                            //std::cout<<"j_x PRE "<<j_x<<std::endl;
                            auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                            //auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                            //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                            //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                            //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                            //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                            //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                                                                            
                            auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                            //auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                            //std::cout<<"px PRE "<<px<<std::endl;
                                                                                                                        
                            // DERIVATIVES
                            //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                            //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                            VT dx = 0.0 , dy = 0.0 ;
                                                                                                                            
                                                
                            if ( pow_x == 0 ){
                                //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                dx = coeff_d2x * px_bis0 ;
                                //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                            }
                            
                            else if ( pow_x == basis_degree ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * px_bis1 ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else if ( pow_x == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                                 
                            else if ( pow_x == basis_degree - 1 ){
                                                                                            
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_x_bis0 = N - ix_bis;
                                size_t j_x_bis1 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                                dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                                                        
                            }
                            else{
                                size_t j_x_bis0 = N - pow_x;
                                size_t j_x_bis1 = N - ix_bis;
                                size_t j_x_bis2 = N - ix2_bis;
                                auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                                auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                                auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                                auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                                auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                                auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                                                                                                                            
                                dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                                //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                            }
                                                
                            if ( pow_y == 0 ){
                                //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                                dy = coeff_d2y *  py_bis0 ;
                                //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                            }
                                                
                            else if ( pow_y == basis_degree ){
                                //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * py_bis1  ;
                                //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                            }
                            
                            else if ( pow_y == 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                                    
                            else if ( pow_y == basis_degree - 1 ){
                                //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                                size_t j_y_bis0 = N - iy_bis;
                                size_t j_y_bis1 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                                dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                                //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                            }
                            else{
                                size_t j_y_bis0 = N - pow_y;
                                size_t j_y_bis1 = N - iy_bis;
                                size_t j_y_bis2 = N - iy2_bis;
                                auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                                auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                                auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                                auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                                auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                                auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                                dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                                                                                            
                            }
                                                                                                                            
                #endif
                            //std::cout<<"pos "<<pos<<std::endl;
                            //std::cout<<"dx "<<dx<<std::endl;
                            //std::cout<<"dy "<<dy<<std::endl;
                            //std::cout<<"px "<<px<<std::endl;
                            //std::cout<<"py "<<py<<std::endl;
                                                                
                            ret(pos) = px*dy;
                            pos++;
                        }
                        
                        N_partial--;
                        starter++;
                        internal_bases -= 2;
                        
                    }
                    
                    // for B_k, with k even, I.E. odd number of bases: there is a central one.
                    if( basis_degree % 2 == 0 )
                    {
                        //  std::cout<<"N is "<<N_partial<< " and starter is "<<starter<<std::endl;
                        assert( N_partial == starter );
                        size_t pow_x = starter ;
                        size_t pow_y = starter ;
                                         
                        VT ix_bis = pow_x-1; // element i-1 for derivative
                        VT iy_bis = pow_y-1; // element i-1 for derivative
                        VT ix2_bis = pow_x-2; // element i-2 for derivative
                        VT iy2_bis = pow_y-2; // element i-2 for derivative
                                                                                                                                            
        #ifdef POWER_CACHE
                                                                                                                                        
                                             
                        std::cout<<"Bernstein Gradient, basis >=2: POWER_CACHE"<<std::endl;
                        auto px = power_cache[2*pow_x];
                        auto py = power_cache[2*pow_y+1];
                        //std::cout<<"px POWER_CACHE "<<px<<std::endl;
                        
                        if ( pow_x == 0 )
                            auto dx = coeff_d2x * power_cache_bis[2*pow_x] ;
                        else if ( pow_x == basis_degree )
                            auto dx = coeff_d2x * power_cache_bis[2*ix2_bis] ;
                        else if ( pow_x == 1 )
                            auto dx = coeff_d2x * ( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] );
                        else if ( pow_x == basis_degree - 1 )
                            auto dx = coeff_d2x * ( power_cache_bis[2*ix2_bis] -2*power_cache_bis[2*ix_bis] );
                        else
                            auto dx = coeff_d2x*( power_cache_bis[2*pow_x] -2*power_cache_bis[2*ix_bis] + power_cache_bis[2*ix2_bis] );
                       
                        if ( pow_y == 0 )
                            auto dy = coeff_d2y*power_cache_bis[2*pow_y+1] ;
                        else if ( pow_y == basis_degree )
                            auto dy = coeff_d2y*power_cache_bis[2*iy2_bis+1] ;
                        else if ( pow_y == 1 )
                            auto dy = coeff_d2y * ( power_cache_bis[2*pow_y+1] -2*power_cache_bis[2*iy_bis+1] );
                        else if ( pow_y == basis_degree - 1 )
                            auto dy = coeff_d2y * ( power_cache_bis[2*iy2_bis+1] -2*power_cache_bis[2*iy_bis+1] );
                        else
                            auto dy = coeff_d2y*( power_cache_bis[2*pow_y+1] - 2*power_cache_bis[2*iy_bis+1] + power_cache_bis[2*iy2_bis+1] );
                        //std::cout<<"dx POWER_CACHE "<<dx<<std::endl;
                    
    #else
                        //std::cout<<"Bernstein Gradient, basis >=2: NOT CACHE"<<std::endl;
                        size_t j_x = basis_degree - pow_x;
                        //size_t j_y = basis_degree - pow_y;
                        //std::cout<<"j_x PRE "<<j_x<<std::endl;
                        auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                        //auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
                        //std::cout<<"coeff_n_x PRE "<<coeff_n_x<<std::endl;
                        //std::cout<<"scaling_x PRE "<<scaling_x<<std::endl;
                        //std::cout<<"pow_x PRE "<<pow_x<<std::endl;
                        //std::cout<<"bx0 PRE "<<bx0<<std::endl;
                        //std::cout<<"bx1 PRE "<<bx1<<std::endl;
                                                                                                                                            
                        auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                        //auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                        //std::cout<<"px PRE "<<px<<std::endl;
                                                                                                                                        
                        // DERIVATIVES
                        //std::cout<<"pow_x= "<<pow_x<<"pow_y= "<<pow_y<<std::endl;
                        //auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 , px_bis2 = 0.0 , py_bis2 = 0.0 ;
                        VT dx = 0.0 , dy = 0.0 ;
                                                                                                
                        if ( pow_x == 0 ){
                            //std::cout<<"into bottom pow_x == 0: HO b_i^K-1(x)"<<std::endl;
                            size_t j_x_bis0 = N - pow_x;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            dx = coeff_d2x * px_bis0 ;
                            //auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                        }
                                            
                        else if ( pow_x == basis_degree ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_x_bis1 = N - ix2_bis;
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_d2x * px_bis1 ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        }
                        else if ( pow_x == 1 ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_x_bis0 = N - pow_x;
                            size_t j_x_bis1 = N - ix_bis;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                            auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_d2x * (px_bis0 - 2*px_bis1) ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        }
                                                 
                        else if ( pow_x == basis_degree - 1 ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_x_bis0 = N - ix_bis;
                            size_t j_x_bis1 = N - ix2_bis;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,ix_bis);
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix2_bis);
                            auto px_bis0 = scaling_x_bis *  coeff_n_x_bis0 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis0);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis1);
                            dx = coeff_d2x * (px_bis1 - 2*px_bis0) ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                                                                                    
                        }
                        else{
                            size_t j_x_bis0 = N - pow_x;
                            size_t j_x_bis1 = N - ix_bis;
                            size_t j_x_bis2 = N - ix2_bis;
                            auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                            auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                            auto coeff_n_x_bis2 = binomial_coeff_fx(N,ix2_bis);
                            auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                            auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                            auto px_bis2 = scaling_x_bis * coeff_n_x_bis2 * iexp_pow(bx0, ix2_bis) * iexp_pow(bx1, j_x_bis2);
                          
                            dx = coeff_d2x * (px_bis0 -2*px_bis1 +px_bis2);
                            //auto dx = coeff_dx*( power_cache_bis[2*ix_bis]-power_cache_bis[2*pow_x]);
                        }
                                                                
                        
                        if ( pow_y == 0 ){
                            //std::cout<<"into bottom pow_y = 0: HO b_i^K-1(y)"<<std::endl;
                            size_t j_y_bis0 = N - pow_y;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                            // auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                            dy = coeff_d2y *  py_bis0 ;
                            //auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                        }
                                                                
                        else if ( pow_y == basis_degree ){
                            //std::cout<<"into bottom pow_y != basis_degree: HO b_i-1^K-1(y)"<<std::endl;
                            size_t j_y_bis1 = N - iy2_bis;
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_d2y * py_bis1  ;
                            //auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                        }
                                        
                        else if ( pow_y == 1 ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_y_bis0 = N - pow_y;
                            size_t j_y_bis1 = N - iy_bis;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                            auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_d2y * (py_bis0 - 2*py_bis1) ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        }
                                                    
                        else if ( pow_y == basis_degree - 1 ){
                            //std::cout<<"into bottom pow_x != basis_degree: HO b_i-1^K-1(x)"<<std::endl;
                            size_t j_y_bis0 = N - iy_bis;
                            size_t j_y_bis1 = N - iy2_bis;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,iy_bis);
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy2_bis);
                            auto py_bis0 = scaling_y_bis *  coeff_n_y_bis0 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis0);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis1);
                            dy = coeff_d2y * (py_bis1 - 2*py_bis0) ;
                            //auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                        }
                        
                        else{
                            size_t j_y_bis0 = N - pow_y;
                            size_t j_y_bis1 = N - iy_bis;
                            size_t j_y_bis2 = N - iy2_bis;
                            auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                            auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                            auto coeff_n_y_bis2 = binomial_coeff_fx(N,iy2_bis);
                            auto py_bis0 = scaling_y_bis * coeff_n_y_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y_bis0);
                            auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                            auto py_bis2 = scaling_y_bis * coeff_n_y_bis2 * iexp_pow(by0, iy2_bis) * iexp_pow(by1, j_y_bis2);
                            dy = coeff_d2y * ( py_bis0 - 2*py_bis1 + py_bis2 );
                            
                        }
                                                                                                
    #endif
                        //std::cout<<"pos "<<pos<<std::endl;
                        //std::cout<<"dx "<<dx<<std::endl;
                        //std::cout<<"dy "<<dy<<std::endl;
                        //std::cout<<"px "<<px<<std::endl;
                        //std::cout<<"py "<<py<<std::endl;
                                                                                
                        ret(pos) = px*dy;
                        pos++;

                       
                        
                    }
                }

                  //  std::cout<<"GRADIENTE: pos is"<<pos<<" and basis size is "<< basis_size<<std::endl;
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
        return (degree+1)*(degree+1);
    }
};



template<typename Mesh, typename T >
class cell_basis_Lagrangian
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;

    point_type          cell_bar;
    coordinate_type     cell_h;
    size_t              basis_degree, basis_size;
    std::vector<point<T, 2> >          nodes;
    //std::vector<size_t>         indeces;
    
public:
    cell_basis_Lagrangian(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree)
    {
        nodes           = equidistriduted_nodes<T,Mesh>(msh, cl, degree);
       
        basis_degree    = degree;
        basis_size      = (basis_degree+1)*(basis_degree+1);
    }
    
    /*
    cell_basis_Lagrangian(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree, const std::vector<size_t>& indeces)
    {
        
        nodes = equidistriduted_nodes_subcell<T,Mesh>(msh, cl, degree, indeces);
        basis_degree    = degree;
        basis_size      = (basis_degree+1)*(basis_degree+1);
        
    }
    */

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
