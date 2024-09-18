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
struct level_set
{
   public:
    size_t grad_deg = 0 ;
    size_t div_deg = 0;
    virtual T operator()(const point<T,2>& pt) const
    {
        return 0.0;
    }
    
    virtual Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        return ret;
    }

    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;

        ret = gradient(pt);
        return ret/ret.norm();
    }
   
};

template<typename T>
struct rotated_square: public level_set<T>
{
    
    T R ;
    T xc ;
    T yc ;
   
    rotated_square(T xc , T yc , T R )
        : xc(xc), yc(yc),R(R)
    {}
    
    rotated_square(){}
    
    rotated_square(const rotated_square& other){
        xc = other.xc ;
        yc = other.yc ;
        R = other.R;
        
    }

    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();
        
        
        // bottom vertex
        T a_x = xc ;
        T a_y = yc - R ;
        
        // right vertex
        T b_x = xc + R ;
        T b_y = yc ;
        
        // top vertex
        T c_x = xc ;
        T c_y = yc + R ;
        
        
        // left vertex
        T d_x = xc - R;
        T d_y = yc ;
        
       // from bottom vertex to right vertex
        T m1 = (b_y - a_y)/(b_x - a_x) ;
        T c1 = b_y - m1*b_x;
        T y1 = m1*x+c1;
        // inside +, outisde -
        auto side1 = (x >= xc) && (y <= yc) && (y > y1) ;
        
        // from right vertex to top vertex
         m1 = (c_y - b_y)/(c_x - b_x) ;
         c1 = c_y - m1*c_x;
         y1 = m1*x+c1;
         // inside +, outisde -
         auto side2 = (x >= xc) && (y >= yc) && (y < y1) ;
        
        
        // from top vertex to left vertex
         m1 = (d_y - c_y)/(d_x - c_x) ;
         c1 = d_y - m1*d_x;
         y1 = m1*x+c1;
         // inside +, outisde -
         auto side3 = (x <= xc) && (y >= yc) && (y < y1) ;
        
        
        // from left vertex to bottom vertex
         m1 = (a_y - d_y)/(a_x - d_x) ;
         c1 = a_y - m1*a_x;
         y1 = m1*x+c1;
         // inside +, outisde -
         auto side4 = (x <= xc) && (y <= yc) && (y > y1) ;
      
 
        // auto inside = side1 || side2 || side3 || side4;
        
        T in = 1;
        
        if (side1 || side2 || side3 || side4)
            in = 1;
        else
            in = -1;
        
        T dist_x = std::min( abs(x-b_x), abs(x-d_x));
        T dist_y = std::min( abs(y-a_y), abs(y-c_y));
        
        return - in * std::min(dist_x , dist_y);
        
    }

    
    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        auto x = pt.x();
        auto y = pt.y();
       
        T eps = 1e-6;
        int side = 0;
        
        // bottom vertex
        T a_x = xc ;
        T a_y = yc - R ;
        
        // right vertex
        T b_x = xc + R ;
        T b_y = yc ;
        
        // top vertex
        T c_x = xc ;
        T c_y = yc + R ;
        
        
        // left vertex
        T d_x = xc - R;
        T d_y = yc ;
        
       // from bottom vertex to right vertex
        T m1 = (b_y - a_y)/(b_x - a_x) ;
        T c1 = b_y - m1*b_x;
        T y1 = m1*x+c1;
        // inside +, outisde -
        T dist1 = abs(y - y1);
        // auto side1 = true;  (x >= xc) && (y <= yc) && (abs(y - y1) <= eps) ;
        
        
        T dist = dist1 ;
        side = 1;
        
        // from right vertex to top vertex
         m1 = (c_y - b_y)/(c_x - b_x) ;
         c1 = c_y - m1*c_x;
         T y2 = m1*x+c1;
         // inside +, outisde -
        T dist2 = abs(y - y2);
         // auto side2 = (x >= xc) && (y >= yc) && (abs(y - y2) <= eps) ;
        
        if( dist2 < dist ){
            dist = dist2;
            side = 2;
   
        }
        
        // from top vertex to left vertex
         m1 = (d_y - c_y)/(d_x - c_x) ;
         c1 = d_y - m1*d_x;
         T y3 = m1*x+c1;
         // inside +, outisde -
         T dist3 = abs(y - y3);
         // auto side3 = (x <= xc) && (y >= yc) && (abs(y - y3) <= eps) ;
        
        if( dist3 < dist ){
            dist = dist3;
            side = 3;
   
        }
        
        // from left vertex to bottom vertex
         m1 = (a_y - d_y)/(a_x - d_x) ;
         c1 = a_y - m1*a_x;
         T y4 = m1*x+c1;
         // inside +, outisde -
         T dist4 = abs(y - y4);
         // auto side4 = (x <= xc) && (y <= yc) && (abs(y - y4) <= eps) ;
        
        if( dist4 < dist ){
            dist = dist4;
            side = 4;
   
        }

//        T dist = 10.;
//        int side = 0;
//
//        int Nside = 0;
//
//        if( side1 ){
//            dist = abs(y1 - y);
//            side = 1;
//            Nside +=1;
//        }
//        if( side2 ){
//            T tmp_dist = abs(y2 - y);
//            if( tmp_dist < dist)
//            {
//                dist = tmp_dist;
//                side = 2;
//                Nside +=1;
//            }
//
//        }
//        if( side3 ){
//            T tmp_dist = abs(y3 - y);
//            if( tmp_dist < dist)
//            {
//                dist = tmp_dist;
//                side = 3;
//                Nside +=1;
//            }
//        }
//        if( side4 ){
//            T tmp_dist = abs(y4 - y);
//            if( tmp_dist < dist)
//            {
//                dist = tmp_dist;
//                side = 4;
//                Nside +=1;
//            }
//        }
        
        if(side == 0)
            std::cout<<"Error, value not assigned..";
        
//        if( Nside != 1 )
//            std::cout<<"Error, value assigned multiple times..";
            
        if( side == 1 )
        {
            ret(0) = 0.5;
            ret(1) = -0.5;
            ret = ret/ret.norm();
        }
        if( side == 2 )
        {
            ret(0) = 0.5;
            ret(1) = 0.5;
            ret = ret/ret.norm();
        }
        if( side == 3 )
        {
            ret(0) = -0.5;
            ret(1) = 0.5;
            ret = ret/ret.norm();
        }
        if( side == 4 )
        {
            ret(0) = -0.5;
            ret(1) = -0.5;
            ret = ret/ret.norm();
        }
    
        return ret;
        
    }
    

    
};

template<typename T>
struct m_shaped_level_set: public level_set<T>
{
    
    T eps;
    T eps_bndry = 1e-10;
    T pos_sides ;
    m_shaped_level_set(T eps, T eps_bndry, T pos_sides)
        : eps(eps), eps_bndry(eps_bndry),pos_sides(pos_sides)
    {}
    
    m_shaped_level_set(){}
    
    m_shaped_level_set(const m_shaped_level_set& other){
        eps = other.eps ;
        eps_bndry = other.eps_bndry ;
        pos_sides = other.pos_sides;
        
    }

    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();
        
        T a = pos_sides+eps_bndry;
        T b = (1.0-pos_sides)-eps_bndry;
        T c = (0.75)+eps;
        T d = 1.0-eps_bndry;
        
        auto side1 = (y>a);
        auto side2 = (x>a);
        auto side3 = (x<b);
        
        T m1 = (c-d)/(0.5-a);
        T c1 = c - m1*0.5;
        T y1 = m1*x+c1;
        
        auto side5 = (y>c)&&(x<=0.5)&&(y<y1);
        
        
        T m2 = (c-d)/(0.5-b);
        T c2 = c - m2*0.5;
        T y2 = m2*x+c2;
        
        auto side6 = (y>c)&&(x>=0.5)&&(y<y2);
        
        auto side4 = (y<c) || side5 || side6;
        
        T in = 1;
        
        if (side1 && side2 && side3 && side4)
            in = 1;
        else
            in = -1;
        
        T dist_x = std::min( abs(x-a), abs(x-b));
        T dist_y = std::min( abs(y-a), abs(y-d));
        
        return - in * std::min(dist_x , dist_y);
        
    }

    
    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        auto x = pt.x();
        auto y = pt.y();
        
        T epsGrad = 1e-8;
        
        T a = pos_sides+eps_bndry;
        T b = (1.0-pos_sides)-eps_bndry;
        T c = (0.75)+eps;
        T d = 1.0-eps_bndry;
        
        // bottom square
        T x_left = a;
        T x_right = b;
        T y_bot = a;
        T y_top = c;

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
//        if(abs(y - y_top) < dist)
//        {
//            ret(0) = 0;
//            ret(1) = 1;
//        }
        
        if( (x > a + epsGrad) && x <=0.5 && y >= c )// left-top side
        {
            T m1 = (c-d)/(0.5-a);
            T m1N = -1.0/m1;
            T normal1 = m1N*x ;

            ret(0) = x;
            ret(1) = normal1;
            if( ret(1) > 10 || ret(1) < -10 ){
                // degenerate case where M-shaped domain is square
                ret(0) = 0;
                ret(1) = 1;
            }

            ret = ret/ret.norm();

        }

        if(  (x < b-epsGrad) && x > 0.5 && y >= c )// right-top side
        {
            T m2 = (c-d)/(0.5-b);
            T m2N = -1.0/m2;
            T normal2 = m2N*(-x) ;

            ret(0) = -x;
            ret(1) = normal2;

            if( ret(1) > 10 || ret(1) < -10 ){
                // degenerate case where M-shaped domain is square
                ret(0) = 0;
                ret(1) = 1;
            }

            ret = ret/ret.norm();

        }
        
        return ret;
        
    }
    

    
    
    Eigen::Matrix<T,2,1> gradient2(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        

        auto x = pt.x();
        auto y = pt.y();

        T epsGrad = 1e-4;
        
        T a = pos_sides+eps_bndry;
        T b = (1.0-pos_sides)-eps_bndry;
        T c = (0.75)+eps;
        T d = 1.0-eps_bndry;
        
        T distXa = abs(x - a);
        T distXb = abs(x - b);
        T distY  = abs(y - a);

        T isOk = 0;
        
        if( distY < epsGrad )// bottom side
        {
            ret(0) = 0;
            ret(1) = -1;
            isOk +=1 ;
        }
        if( distXa < epsGrad )// left side
        {
            ret(0) = -1 ;
            ret(1) = 0 ;
            isOk +=1 ;
        }
        if( distXb < epsGrad )// right side
        {
            ret(0) = 1 ;
            ret(1) = 0 ;
            isOk +=1 ;
        }
        if( (x > a + epsGrad) && x <=0.5 && y <= d && y >= c )// left-top side
        {
            T m1 = (c-d)/(0.5-a);
            // T c1 = c - m1*0.5;
            T m1N = -1.0/m1;
            
            // T c1N = m1*x+c1 - m1N * x;
            
            T normal1 = m1N*x ; // + c1N;
            
            T norm1 = sqrt(x*x + normal1*normal1);
            
            ret(0) = x; // /norm1;
            ret(1) = normal1; // /norm1;
            
            ret = ret/ret.norm();
            isOk +=1 ;
        }
        
        if(  (x < b-epsGrad) && x > 0.5 && y <= d && y >= c )// right-top side
        {
            T m2 = (c-d)/(0.5-b);
            // T c2 = c - m2*0.5;
            
            T m2N = -1.0/m2;
            
            T normal2 = m2N*(-x) ;
            
            T norm2 = sqrt(x*x + normal2*normal2);
            
            ret(0) = -x; // /norm2;
            ret(1) = normal2; // /norm2;
            
            
//            T retN = ret.norm() ;
            ret = ret/ret.norm();
            
            isOk +=1 ;
          
//            ret(0) = 1 ;
//            ret(1) = 0 ;
        }
        
       
//        if (!isOk)
//            std::cout<<"Error..."<<std::endl;
        
        if (isOk != 1)
            std::cout<<"isOk = "<<isOk<<std::endl;
            
        
        return ret;
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
struct circle_level_set_signed_distance: public level_set<T>
{
    T radius, alpha, beta , eps;
   
    
    circle_level_set_signed_distance(T r, T a, T b , T eps = 0.0)
        : radius(r), alpha(a), beta(b) , eps(eps)
    {}
    
    circle_level_set_signed_distance(){}
    
    circle_level_set_signed_distance(const circle_level_set_signed_distance& other){
        radius = other.radius ;
        alpha = other.alpha ;
        beta = other.beta ;
        eps = other.eps ;
        
    }

    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();
//        return sqrt( (x-alpha)*(x-alpha) + (y-beta)*(y-beta) ) - radius ;
        return sqrt( (x-alpha)*(x-alpha) + (y-beta)*(y-beta) + eps*eps ) - sqrt(radius*radius + eps*eps) ;
    }

    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret ; // ret_pos , ret_neg;
        T x = pt.x();
        T y = pt.y();
        
        ret(0) = (x - alpha)/(sqrt((x-alpha)*(x-alpha) + (y-beta)*(y-beta)+ eps*eps )) ;
        ret(1) = (y - beta)/(sqrt((x-alpha)*(x-alpha) + (y-beta)*(y-beta) + eps*eps )) ;
        
        return ret ;
        
        
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
struct couette_level_set: public level_set<T>
{
    T radius_a, radius_b, alpha, beta;

    couette_level_set(T r_a, T r_b , T a, T b)
        : radius_a(r_a), radius_b(r_b), alpha(a), beta(b)
    {}
    
    couette_level_set(){}
    
    couette_level_set(const couette_level_set& other){
        radius_a = other.radius_a ;
        radius_b = other.radius_b ;
        alpha = other.alpha ;
        beta = other.beta ;
        
    }

    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();
        
        T radius = x*x + y*y ;
        T ret = 0;
        if(radius < 1.0/4.0)
            ret = -(radius_a*radius_a*(x-alpha)*(x-alpha) + radius_a*radius_a*(y-beta)*(y-beta) - radius_a*radius_a*radius_a*radius_a) ;
        else
            ret = radius_b*radius_b*(x-alpha)*(x-alpha) + radius_b*radius_b*(y-beta)*(y-beta) - radius_b*radius_b*radius_b*radius_b ;
            
        

        return ret ;
         
    }

    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        auto x = pt.x();
        auto y = pt.y();
        
        T radius = x*x + y*y ;
        if( radius < 1.0/4.0)
        {
            ret(0) = 2.0*radius_a*radius_a*(x - alpha);
            ret(1) = 2.0*radius_a*radius_a*(y - beta);
        }
        else
        {
            ret(0) = 2.0*radius_b*radius_b*(x - alpha );
            ret(1) = 2.0*radius_b*radius_b*(y - beta );
        }
        
        return ret;
    }
};




template<typename T>
struct elliptic_level_set_signed_distance: public level_set<T>
{
    T radius_a, radius_b, alpha, beta;

    elliptic_level_set_signed_distance(T r_a, T r_b , T a, T b)
        : radius_a(r_a), radius_b(r_b), alpha(a), beta(b)
    {}
    
    elliptic_level_set_signed_distance(){}
    
    elliptic_level_set_signed_distance(const elliptic_level_set_signed_distance& other){
        radius_a = other.radius_a ;
        radius_b = other.radius_b ;
        alpha = other.alpha ;
        beta = other.beta ;
        
    }

    T operator()(const point<T,2>& pt) const
    {
        auto x = pt.x();
        auto y = pt.y();

        return sqrt(radius_b*radius_b*(x-alpha)*(x-alpha) + radius_a*radius_a*(y-beta)*(y-beta)) - sqrt(radius_a*radius_a*radius_b*radius_b) ;
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
struct line_level_set: public level_set<T>
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
struct square_level_set: public level_set<T>
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


template<typename T>
struct flower_level_set_signed_distance: public level_set<T>
{
    T radius, alpha, beta, a;
    size_t N;

    flower_level_set_signed_distance(T r, T al, T b, size_t N_, T a_)
        : radius(r), alpha(al), beta(b), N(N_), a(a_)
    {}

    flower_level_set_signed_distance(){}
    
    flower_level_set_signed_distance(const flower_level_set_signed_distance& other){
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

        return sqrt((x-alpha)*(x-alpha) + (y-beta)*(y-beta)) - sqrt(radius*radius
            - a * std::cos(N*theta + M_PI) );
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
