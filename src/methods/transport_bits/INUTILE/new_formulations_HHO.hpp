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

/// Each mesh has memory of the original nodes: useful to link agglomerated - original mesh
template<typename Mesh>
void offset_definition( Mesh& msh)
{
    //size_t counter = 0;
    for (auto& cl:msh.cells) {
        auto offset_orig = offset(msh,cl);
        cl.user_data.offset_subcells.push_back(offset_orig);
       // counter++;
       // std::cout<<"Initialisation cell num "<< cl.user_data.offset_subcells[offset_orig]<<std::endl;
    }
}


/// New version of find_zero_crossing for discret level set functions
/// in cuthho_geom.hpp
template<typename T, typename Function, typename Mesh>
point<T, 2>
find_zero_crossing_on_face(const point<T,2>& p0, const point<T,2>& p1, const Function& level_set_function,
                   const T& threshold, const Mesh & msh, const typename Mesh::face_type& fc)
{
    /* !!! We assume that the level set function *has* a zero crossing
     * between p0 and p1 !!! */
    
    // I SHOULD CHECK THAT pm IS ALWAYS ON THE FACE p0-p1 ???????
    auto pa = p0;
    auto pb = p1;
    auto pm = (pa+pb)/2.0;
    auto pm_prev = pm;

    T x_diff_sq, y_diff_sq;

    /* A threshold of 1/10000 the diameter of the element is considered
     * acceptable. Since with 24 iterations we reduce the error by 16384
     * and the worst case is that the two points are at the opposite sides
     * of the element, we put 30 as limit. */
    size_t max_iter = 50;

    do {
        auto la = level_set_function(pa,msh,fc);
        auto lb = level_set_function(pb,msh,fc);
        auto lm = level_set_function(pm,msh,fc);

        if ( (lb >= 0 && lm >= 0) || (lb < 0 && lm < 0) )
        {   /* intersection is between pa and pm */
            pm_prev = pm;
            pb = pm;
            pm = (pa+pb)/2.0;
        }
        else
        {   /* intersection is between pm and pb */
            pm_prev = pm;
            pa = pm;
            pm = (pa+pb)/2.0;
        }

        x_diff_sq = (pm_prev.x() - pm.x()) * (pm_prev.x() - pm.x());
        y_diff_sq = (pm_prev.y() - pm.y()) * (pm_prev.y() - pm.y());

    } while ( (sqrt(x_diff_sq + y_diff_sq) > threshold) && max_iter-- );

    return pm;

    /* Affine zero crossing was like that: */
    //auto t = l0/(l0-l1);
    //auto ip = (pts[1] - pts[0]) * t + pts[0];
}

/// New version of find_zero_crossing for discret level set functions
/// in cuthho_geom.hpp
template<typename T, typename Function, typename Mesh>
point<T, 2>
find_zero_crossing_on_face3(const point<T,2>& p0, const point<T,2>& p1, const Function& level_set_function,
                   const T& threshold, const Mesh & msh, const typename Mesh::face_type& fc)
{
    /* !!! We assume that the level set function *has* a zero crossing
     * between p0 and p1 !!! */
    
    // I SHOULD CHECK THAT pm IS ALWAYS ON THE FACE p0-p1 ???????
    auto pa = p0;
    auto pb = p1;
    auto pm = (pa+pb)/2.0;
    auto pm_prev = pm;
    T iso_val_interface = level_set_function.iso_val_interface ;
    T x_diff_sq, y_diff_sq;

    /* A threshold of 1/10000 the diameter of the element is considered
     * acceptable. Since with 24 iterations we reduce the error by 16384
     * and the worst case is that the two points are at the opposite sides
     * of the element, we put 30 as limit. */
    size_t max_iter = 50;

    do {
        //auto la = level_set_function(pa,msh,fc);
        auto lb = level_set_function(pb,msh,fc);
        auto lm = level_set_function(pm,msh,fc);

        if ( (lb >= iso_val_interface && lm >= iso_val_interface) || (lb < iso_val_interface && lm < iso_val_interface) )
        {   /* intersection is between pa and pm */
            pm_prev = pm;
            pb = pm;
            pm = (pa+pb)/2.0;
        }
        else
        {   /* intersection is between pm and pb */
            pm_prev = pm;
            pa = pm;
            pm = (pa+pb)/2.0;
        }

        x_diff_sq = (pm_prev.x() - pm.x()) * (pm_prev.x() - pm.x());
        y_diff_sq = (pm_prev.y() - pm.y()) * (pm_prev.y() - pm.y());

    } while ( (sqrt(x_diff_sq + y_diff_sq) > threshold) && max_iter-- );

    return pm;

    /* Affine zero crossing was like that: */
    //auto t = l0/(l0-l1);
    //auto ip = (pts[1] - pts[0]) * t + pts[0];
}

/// New version of find_zero_crossing for discret level set functions
/// in cuthho_geom.hpp
template<typename T, typename Function, typename Mesh >
point<T, 2>
find_zero_crossing_in_cell(const point<T,2>& p0, const point<T,2>& p1, const Function& level_set_function,
                   const T& threshold, const Mesh & msh, const typename Mesh::cell_type& cl)
{
    /* !!! We assume that the level set function *has* a zero crossing
     * between p0 and p1 !!! */
    
    // I SHOULD CHECK THAT pm IS ALWAYS IN THE CELL ???????
    auto pa = p0;
    auto pb = p1;
    auto pm = (pa+pb)/2.0;
    auto pm_prev = pm;

    T x_diff_sq, y_diff_sq;

    /* A threshold of 1/10000 the diameter of the element is considered
     * acceptable. Since with 24 iterations we reduce the error by 16384
     * and the worst case is that the two points are at the opposite sides
     * of the element, we put 30 as limit. */
    size_t max_iter = 50; // ERA 50, METTO 100

    do {
        auto la = level_set_function(pa,msh,cl);
        auto lb = level_set_function(pb,msh,cl);
        auto lm = level_set_function(pm,msh,cl);

        if ( (lb >= 0 && lm >= 0) || (lb < 0 && lm < 0) )
        {   /* intersection is between pa and pm */
            pm_prev = pm;
            pb = pm;
            pm = (pa+pb)/2.0;
        }
        else
        {   /* intersection is between pm and pb */
            pm_prev = pm;
            pa = pm;
            pm = (pa+pb)/2.0;
        }

        x_diff_sq = (pm_prev.x() - pm.x()) * (pm_prev.x() - pm.x());
        y_diff_sq = (pm_prev.y() - pm.y()) * (pm_prev.y() - pm.y());

    } while ( (sqrt(x_diff_sq + y_diff_sq) > threshold) && max_iter-- );

    return pm;

    /* Affine zero crossing was like that: */
    //auto t = l0/(l0-l1);
    //auto ip = (pts[1] - pts[0]) * t + pts[0];
}

/// New version of find_zero_crossing for discret level set functions
/// in cuthho_geom.hpp
template<typename T, typename Function, typename Mesh >
point<T, 2>
find_zero_crossing_in_cell3(const point<T,2>& p0, const point<T,2>& p1, const Function& level_set_function,
                   const T& threshold, const Mesh & msh, const typename Mesh::cell_type& cl)
{
    /* !!! We assume that the level set function *has* a zero crossing
     * between p0 and p1 !!! */
    
    // I SHOULD CHECK THAT pm IS ALWAYS IN THE CELL ???????
    auto pa = p0;
    auto pb = p1;
    auto pm = (pa+pb)/2.0;
    auto pm_prev = pm;
    T iso_val_interface = level_set_function.iso_val_interface ;
    T x_diff_sq, y_diff_sq;

    /* A threshold of 1/10000 the diameter of the element is considered
     * acceptable. Since with 24 iterations we reduce the error by 16384
     * and the worst case is that the two points are at the opposite sides
     * of the element, we put 30 as limit. */
    size_t max_iter = 50; // ERA 50, METTO 100

    do {
        //auto la = level_set_function(pa,msh,cl);
        auto lb = level_set_function(pb,msh,cl);
        auto lm = level_set_function(pm,msh,cl);

        if ( (lb >= iso_val_interface && lm >= iso_val_interface) || (lb < iso_val_interface && lm < iso_val_interface) )
        {   /* intersection is between pa and pm */
            pm_prev = pm;
            pb = pm;
            pm = (pa+pb)/2.0;
        }
        else
        {   /* intersection is between pm and pb */
            pm_prev = pm;
            pa = pm;
            pm = (pa+pb)/2.0;
        }

        x_diff_sq = (pm_prev.x() - pm.x()) * (pm_prev.x() - pm.x());
        y_diff_sq = (pm_prev.y() - pm.y()) * (pm_prev.y() - pm.y());

    } while ( (sqrt(x_diff_sq + y_diff_sq) > threshold) && max_iter-- );

    return pm;

    /* Affine zero crossing was like that: */
    //auto t = l0/(l0-l1);
    //auto ip = (pts[1] - pts[0]) * t + pts[0];
}



/// New version of detect_node_position for discret level functions
/// in cuthho_geom.hpp
template<typename T, size_t ET, typename Function>
void
detect_node_position2(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    for (auto& n : msh.nodes)
    {
        //auto pt = points(msh, n); //deleted by Stefano
        //if ( level_set_function(pt) < 0 ) //deleted by Stefano
        
        T value_node = level_set_function(n);
        if( std::abs(value_node) < 1e-17 ){
            std::cout<<"In detect_node_position2 -> ATTENTION, INTERFACE ON A NODE!"<<std::endl;
            auto pt = points(msh, n);
            value_node = level_set_function(pt) ;
        }
        
        
        
        if ( value_node < 0 ) // add by Stefano
            n.user_data.location = element_location::IN_NEGATIVE_SIDE;
        else
            n.user_data.location = element_location::IN_POSITIVE_SIDE;
        
        
        
    }
}


/// New version of detect_node_position for discret level functions -> USING interface = 1/2
/// in cuthho_geom.hpp
template<typename T, size_t ET, typename Function>
void
detect_node_position3(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    
    timecounter tc;
    tc.tic();
    T iso_val_interface = level_set_function.iso_val_interface ;
    std::cout<<"In 'detect_node' -> iso_val_interface = "<<iso_val_interface<<std::endl;

    for (auto& n : msh.nodes)
    {
        //auto pt = points(msh, n); //deleted by Stefano
        //if ( level_set_function(pt) < 0 ) //deleted by Stefano
         
        T value_node = level_set_function(n);
        
        if( std::abs(value_node - iso_val_interface) <  1e-17 ){
            std::cout<<"In detect_node_position2 -> ATTENTION, INTERFACE ON A NODE!"<<std::endl;
            auto pt = points(msh, n);
            value_node = level_set_function(pt) ;
        }
        
        
        
        if ( value_node < iso_val_interface ){ // add by Stefano
            n.user_data.location = element_location::IN_NEGATIVE_SIDE;
            //std::cout<<"n.user_data.location = IN_NEGATIVE_SIDE"<<std::endl;
        }
        else{
            n.user_data.location = element_location::IN_POSITIVE_SIDE;
            //std::cout<<"n.user_data.location = IN_POSITIVE_SIDE"<<std::endl;
        }
        //std::cout<<"Value_node = "<< value_node<<std::endl;
        
    }
    
    tc.toc();
    std::cout << bold << yellow << "detect_node_position3, time resolution: " << tc << " seconds" << reset << std::endl;
}

/// New version of detect_node_position for discret level functions -> USING interface = 1/2
/// in cuthho_geom.hpp
template<typename T, size_t ET, typename Function>
void
detect_node_position3_parallel(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    timecounter tc ;
    tc.tic();
    T iso_val_interface = level_set_function.iso_val_interface ;
    std::cout<<"In 'detect_node' -> iso_val_interface = "<<iso_val_interface<<std::endl;
#ifdef HAVE_INTEL_TBB
    size_t n_nodes = msh.nodes.size();
    std::cout<<" I m in parallel zone"<<std::endl;
    tbb::parallel_for(size_t(0), size_t(n_nodes), size_t(1),
    [&msh,&level_set_function,&iso_val_interface] (size_t & cell_ind){
        auto& n = msh.nodes[cell_ind];
        T value_node = level_set_function(n);
            
        if( std::abs(value_node - iso_val_interface) <  1e-17 ){
            std::cout<<"In detect_node_position2 -> ATTENTION, INTERFACE ON A NODE!"<<std::endl;
            auto pt = points(msh, n);
            value_node = level_set_function(pt) ;
        }
            
            
            
        if ( value_node < iso_val_interface ){ // add by Stefano
            n.user_data.location = element_location::IN_NEGATIVE_SIDE;
                //std::cout<<"n.user_data.location = IN_NEGATIVE_SIDE"<<std::endl;
        }
        else{
            n.user_data.location = element_location::IN_POSITIVE_SIDE;
                //std::cout<<"n.user_data.location = IN_POSITIVE_SIDE"<<std::endl;
        }
                
    }
    );
    /*
    for (auto& n : msh.nodes)
    {
        //T value_node = level_set_function(n);
        //std::cout<<"value_node SEQUENTIAL = "<<level_set_function(n)<<std::endl;
        if ( n.user_data.location == element_location::IN_NEGATIVE_SIDE ){ // add by Stefano
            //n.user_data.location = element_location::IN_NEGATIVE_SIDE;
            std::cout<<"n.user_data.location = IN_NEGATIVE_SIDE"<<std::endl;
            std::cout<<"--->value_node SEQUENTIAL = "<<level_set_function(n)<<std::endl;
        }
        else{
            //n.user_data.location = element_location::IN_POSITIVE_SIDE;
                std::cout<<"n.user_data.location = IN_POSITIVE_SIDE"<<std::endl;
                std::cout<<"--------------------->value_node SEQUENTIAL = "<<level_set_function(n)<<std::endl;
        }
    }
    */
    tc.toc();
    std::cout << bold << yellow << "detect_node_position3_parallel, time resolution: " << tc << " seconds" << reset << std::endl;
#else
    
    for (auto& n : msh.nodes)
    {
        //auto pt = points(msh, n); //deleted by Stefano
        //if ( level_set_function(pt) < 0 ) //deleted by Stefano
         
        T value_node = level_set_function(n);
        
        if( std::abs(value_node - iso_val_interface) <  1e-17 ){
            std::cout<<"In detect_node_position2 -> ATTENTION, INTERFACE ON A NODE!"<<std::endl;
            auto pt = points(msh, n);
            value_node = level_set_function(pt) ;
        }
        
        
        //std::cout<<"value_node SEQUENTIAL = "<<level_set_function(n)<<std::endl;
        if ( value_node < iso_val_interface ){ // add by Stefano
            n.user_data.location = element_location::IN_NEGATIVE_SIDE;
            //std::cout<<"n.user_data.location = IN_NEGATIVE_SIDE"<<std::endl;
        }
        else{
            n.user_data.location = element_location::IN_POSITIVE_SIDE;
            //std::cout<<"n.user_data.location = IN_POSITIVE_SIDE"<<std::endl;
        }
        //std::cout<<"Value_node = "<< value_node<<std::endl;
        
    }
    
    
#endif
     
}

/// New version of detect_cut_faces for discret level functions
/// in cuthho_geom.hpp
template<typename T, size_t ET, typename Function>
void
detect_cut_faces2(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    for (auto& fc : msh.faces)
    {
        auto pts = points(msh, fc);
        /*
        if( (pts[0].x() == 0.375) && (pts[0].y() == 0.5) )
        {
            std::cout<<"CASE pt0!!"<<std::endl;
            std::cout<<"AND  pts[1] = "<<pts[1]<<std::endl;
            std::cout<<"level_set_function(pts[0]) = "<<level_set_function(pts[0])<< " , level_set_function(pts[1]) = "<<level_set_function(pts[1])<<std::endl;
             std::cout<<"level_set_function(pts[0],msh,fc) = "<<level_set_function(pts[0],msh,fc)<< " , level_set_function(pts[1],msh,fc) = "<<level_set_function(pts[1],msh,fc)<<'\n'<<std::endl;
           
            
        }
        if( (pts[1].x() == 0.375) && (pts[1].y() == 0.5) )
        {
            std::cout<<"CASE pt1!!"<<std::endl;
            std::cout<<"AND  pts[0] = "<<pts[0]<<std::endl;
            std::cout<<"level_set_function(pts[0]) = "<<level_set_function(pts[0])<< " , level_set_function(pts[1]) = "<<level_set_function(pts[1])<<std::endl;
             std::cout<<"level_set_function(pts[0],msh,fc) = "<<level_set_function(pts[0],msh,fc)<< " , level_set_function(pts[1],msh,fc) = "<<level_set_function(pts[1],msh,fc)<<'\n'<<std::endl;
           
            
        }
        */
        //auto l0 = level_set_function(pts[0]);      //deleted by Stefano
        //auto l1 = level_set_function(pts[1]);       //deleted by Stefano
        
        auto l0 = level_set_function(pts[0],msh,fc);      // add by Stefano
        auto l1 = level_set_function(pts[1],msh,fc);       // add by Stefano
        
        // In the case doubt, I don't care the value, just the sign and I assign the same sign of the other. JUST ONE OF THE TWO SHOULD BE SO SMALL
        if( (std::abs(l0) < 1e-17) && (std::abs(l1) < 1e-17) )
            std::cout<<"STOP --> CASE DOUBT: l0 = "<<l0<< " and l1 = "<<l1<<std::endl;
        
        else if( std::abs(l0) < 1e-17 ){
            std::cout<<"The node "<<pts[0]<<" is very close to the interface."<<std::endl;
            l0 = level_set_function(pts[0]) ;
            std::cout<<"l0 = "<<l0<< " ,l1 = "<<l1<<'\n'<<std::endl;
        }
        else if( std::abs(l1) < 1e-17 ){
            std::cout<<"The node "<<pts[1]<<" is very close to the interface."<<std::endl;
            l1 = level_set_function(pts[1]) ;
            std::cout<<"l0 = "<<l0<< " ,l1 = "<<l1<<'\n'<<std::endl;
        }
        /*
        if( ((pts[1].x() == 0.375) && (pts[1].y() == 0.5) ) ||(pts[0].x() == 0.375) && (pts[0].y() == 0.5))
        {
        std::cout<<"l0 = "<<l0<< " ,l1 = "<<l1<<'\n'<<std::endl;
        }
        */
        if (l0 >= 0 && l1 >= 0)
        {
            fc.user_data.location = element_location::IN_POSITIVE_SIDE;
            continue;
        }

        if (l0 < 0 && l1 < 0)
        {
            fc.user_data.location = element_location::IN_NEGATIVE_SIDE;
            continue;
        }
        
        

        auto threshold = diameter(msh, fc) / 1e20;
        //auto pm = find_zero_crossing(pts[0], pts[1], level_set_function, threshold);
        auto pm = find_zero_crossing_on_face(pts[0], pts[1], level_set_function, threshold,msh,fc);

        /* If node 0 is in the negative region, mark it as node inside, otherwise mark node 1 */
        fc.user_data.node_inside = ( l0 < 0 ) ? 0 : 1;
        fc.user_data.location = element_location::ON_INTERFACE;
        fc.user_data.intersection_point = pm;
    }
}

/// New version of detect_cut_faces for discret level functions -> USING interface = 1/2
/// in cuthho_geom.hpp
template<typename T, size_t ET, typename Function>
void
detect_cut_faces3(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    T iso_val_interface = level_set_function.iso_val_interface ;
    std::cout<<"In 'detect_cut_face3'--> iso_val_interface = "<<iso_val_interface<<std::endl;
    for (auto& fc : msh.faces)
    {
        auto pts = points(msh, fc);
        /*
        if( (pts[0].x() == 0.375) && (pts[0].y() == 0.5) )
        {
            std::cout<<"CASE pt0!!"<<std::endl;
            std::cout<<"AND  pts[1] = "<<pts[1]<<std::endl;
            std::cout<<"level_set_function(pts[0]) = "<<level_set_function(pts[0])<< " , level_set_function(pts[1]) = "<<level_set_function(pts[1])<<std::endl;
             std::cout<<"level_set_function(pts[0],msh,fc) = "<<level_set_function(pts[0],msh,fc)<< " , level_set_function(pts[1],msh,fc) = "<<level_set_function(pts[1],msh,fc)<<'\n'<<std::endl;
           
            
        }
        if( (pts[1].x() == 0.375) && (pts[1].y() == 0.5) )
        {
            std::cout<<"CASE pt1!!"<<std::endl;
            std::cout<<"AND  pts[0] = "<<pts[0]<<std::endl;
            std::cout<<"level_set_function(pts[0]) = "<<level_set_function(pts[0])<< " , level_set_function(pts[1]) = "<<level_set_function(pts[1])<<std::endl;
             std::cout<<"level_set_function(pts[0],msh,fc) = "<<level_set_function(pts[0],msh,fc)<< " , level_set_function(pts[1],msh,fc) = "<<level_set_function(pts[1],msh,fc)<<'\n'<<std::endl;
           
            
        }
        */
        //auto l0 = level_set_function(pts[0]);      //deleted by Stefano
        //auto l1 = level_set_function(pts[1]);       //deleted by Stefano
        
        auto l0 = level_set_function(pts[0],msh,fc);      // add by Stefano
        auto l1 = level_set_function(pts[1],msh,fc);       // add by Stefano
        
        // In the case doubt, I don't care the value, just the sign and I assign the same sign of the other. JUST ONE OF THE TWO SHOULD BE SO SMALL
        if( (std::abs(l0-iso_val_interface) < 1e-17) && (std::abs(l1-iso_val_interface) < 1e-17) )
            std::cout<<"STOP --> CASE DOUBT: l0 = "<<l0<< " and l1 = "<<l1<<std::endl;
        
        else if( std::abs(l0-iso_val_interface) < 1e-17 ){
            std::cout<<"The node "<<pts[0]<<" is very close to the interface."<<std::endl;
            l0 = level_set_function(pts[0]) ;
            std::cout<<"l0 = "<<l0<< " ,l1 = "<<l1<<'\n'<<std::endl;
        }
        else if( std::abs(l1-iso_val_interface) < 1e-17 ){
            std::cout<<"The node "<<pts[1]<<" is very close to the interface."<<std::endl;
            l1 = level_set_function(pts[1]) ;
            std::cout<<"l0 = "<<l0<< " ,l1 = "<<l1<<'\n'<<std::endl;
        }
        /*
        if( ((pts[1].x() == 0.375) && (pts[1].y() == 0.5) ) ||(pts[0].x() == 0.375) && (pts[0].y() == 0.5))
        {
        std::cout<<"l0 = "<<l0<< " ,l1 = "<<l1<<'\n'<<std::endl;
        }
        */
        if (l0 >= iso_val_interface && l1 >= iso_val_interface)
        {
            fc.user_data.location = element_location::IN_POSITIVE_SIDE;
            continue;
        }

        if (l0 < iso_val_interface && l1 < iso_val_interface)
        {
            fc.user_data.location = element_location::IN_NEGATIVE_SIDE;
            continue;
        }
        
        

        auto threshold = diameter(msh, fc) / 1e20;
        //auto pm = find_zero_crossing(pts[0], pts[1], level_set_function, threshold);
        auto pm = find_zero_crossing_on_face3(pts[0], pts[1], level_set_function, threshold,msh,fc);
        //std::cout<<"pm = "<<pm<< " and level_set_function = "<<level_set_function(pm,msh,fc)<<std::endl;
        /* If node 0 is in the negative region, mark it as node inside, otherwise mark node 1 */
        fc.user_data.node_inside = ( l0 < iso_val_interface ) ? 0 : 1;
        fc.user_data.location = element_location::ON_INTERFACE;
        fc.user_data.intersection_point = pm;
    }
}

/// New version of detect_cut_cells for discret level functions
/// in cuthho_geom.hpp
template<typename T, size_t ET, typename Function>
void
detect_cut_cells2(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    std::cout<<"I AM IN DETECT CUT CELL2!!!!"<<std::endl;
    //typedef typename cuthho_mesh<T, ET>::face_type  face_type;
    typedef typename cuthho_mesh<T, ET>::point_type point_type;
    //typedef typename cuthho_mesh<T, ET>::cell_type cell_type;

    size_t cell_i = 0;
    for (auto& cl : msh.cells)
    {
        auto fcs = faces(msh, cl);
        std::array< std::pair<size_t, point_type>, 2 >  cut_faces;

        size_t k = 0;
        for (size_t i = 0; i < fcs.size(); i++)
        {
            if ( is_cut(msh, fcs[i]) )
                cut_faces.at(k++) = std::make_pair(i, fcs[i].user_data.intersection_point);
        }

        /* If a face is cut, the cells that own the face are cut. Is this
         * unconditionally true? It should...fortunately this isn't avionics
         * software */

        if (k == 0)
        {
            
            auto is_positive = [&](const point_type& pt) -> bool {
            return level_set_function(pt) > 0;
            };
            
            
            auto pts = points(msh, cl);
            
            if ( std::all_of(pts.begin(), pts.end(), is_positive) )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
     
            
            
            
            /*
            auto pts = points(msh, cl);
            auto pt = pts.begin();
            size_t counter = 0;
            while( ( pt!= pts.end() ) && ( level_set_function(*pt,msh,cl) > 0 ) )
            {
                counter++;
                pt++;
                
            }
             
            if ( counter == pts.size() )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
            */
             
        }
        //MODIFICARE QUAAAA
        if (k == 2)
        {
            cl.user_data.location = element_location::ON_INTERFACE;
            auto p0 = cut_faces[0].second;
            auto p1 = cut_faces[1].second;
            auto pt = p1 - p0;
            auto pn = p0 + point<T,2>(-pt.y(), pt.x());
            auto pn_prova = (p0+p1)/2.0 + 0.5*point<T,2>(-pt.y(), pt.x());
            //if(offset(msh,cl)== 119)
            //    std::cout<<"p0 = "<<p0<< " , p1 ="<<p1<<std::endl;
            // PRIMA ERA DA p0 ->   MODIFCATO, ora è pt  medio!
            /*
            if( !pt_in_cell(msh, pn, cl) )
            {
                std::cout<<"I chose another pn to ordering interface_points in 'detect_cut_cells2'."<<std::endl;
                T m_half = ( ps1.y() - pm.y() )/( ps1.x() - pm.x() );
                T q = pm.y() - m_half * pm.x() ;
                auto pt_bdry = search_boundary( msh , cl , pm , m_half , q , lm , level_set_function ) ;
                auto lm_bdry = level_set_function( pt_bdry , msh , cl );
            }
            */
            if(offset(msh,cl)== 137 || offset(msh,cl)== 138 || offset(msh,cl)== 134||offset(msh,cl)== 103){
                std::cout<<yellow<<bold<<"offset(msh,cl) = "<<offset(msh,cl)<<reset<<std::endl;
                auto pn_bis = (p0+p1)/2.0 + point<T,2>(-pt.y(), pt.x());
                std::cout<<"pn_bis = "<<pn_bis<< " , level_set_function(pn_bis,msh,cl) ="<<level_set_function(pn_bis,msh,cl) <<std::endl;
                auto pn_bis0 = (p0+p1)/2.0 + 0.5* point<T,2>(-pt.y(), pt.x());
                std::cout<<"pn_bis0 = "<<pn_bis0<< " , level_set_function(pn_bis0,msh,cl) ="<<level_set_function(pn_bis0,msh,cl) <<std::endl;
                auto pn_bis1 = p0 + 0.5 * point<T,2>(-pt.y(), pt.x());
                std::cout<<"pn_bis1 = "<<pn_bis1<< " , level_set_function(pn_bis1,msh,cl) ="<<level_set_function(pn_bis1,msh,cl)<<'\n' <<std::endl;
                
                std::cout<<"pn = "<<pn<< " , p0 = "<<p0<< " , p1 = "<<p1<<std::endl;
                std::cout<<"level_set_function(pn,msh,cl) = "<<level_set_function(pn,msh,cl)<< " , level_set_function(p0,msh,cl) = "<<level_set_function(p0,msh,cl)<< " , level_set_function(pn,msh,cl) = "<<level_set_function(p1,msh,cl)<<std::endl;
                std::cout<<"p0 - point<T,2>(-pt.y(), pt.x()) = "<<p0 - point<T,2>(-pt.y(), pt.x())<< " , level_set_function(p0 - point<T,2>(-pt.y(), pt.x()),msh,cl) = "<<level_set_function(p0 - point<T,2>(-pt.y(), pt.x()),msh,cl)<<std::endl;
            }
            
            
            if( !(signbit(level_set_function(pn,msh,cl)) == signbit(level_set_function(pn_prova,msh,cl))) ){
                pn = pn_prova ;
                std::cout<<"pn = "<<pn<< " , pn_prova = "<<pn_prova<< " , level_set_function(pn,msh,cl) = "<<level_set_function(pn,msh,cl)<< " , level_set_function(pn_prova,msh,cl) = "<<level_set_function(pn_prova,msh,cl) <<std::endl;
            }
            
            if ( level_set_function(pn,msh,cl) >= 0 )
            {
                cl.user_data.p0 = p1;
                cl.user_data.p1 = p0;
            }
            else
            {
                cl.user_data.p0 = p0;
                cl.user_data.p1 = p1;
            }

            cl.user_data.interface.push_back(cl.user_data.p0);
            cl.user_data.interface.push_back(cl.user_data.p1);
        }

        if ( k != 0 && k != 2 ){
            auto pts = points(msh,cl);
            std::cout<<"Point[0] = "<<pts[0]<<" , point[1] = "<<pts[1]<<" , point[2] = "<<pts[2]<<" , point[3] = "<<pts[3]<<std::endl;
            std::cout<<"level_set_function(p0) = "<<level_set_function(pts[0],msh,cl) << " , level_set_function(p1) = "<<level_set_function(pts[1],msh,cl)<< " , level_set_function(p2) = "<<level_set_function(pts[2],msh,cl)<< " , level_set_function(p3) = "<<level_set_function(pts[3],msh,cl)<<std::endl;
            for (size_t i = 0; i < fcs.size(); i++)
            {
                if ( is_cut(msh, fcs[i]) )
                  std::cout<<"fcs[i].user_data.intersection_point = "<<fcs[i].user_data.intersection_point<<std::endl;
            }
           
            std::cout<<"ERROR: in cut cell "<<cell_i<<" there are k = "<<k<<" cuts!!!!"<<std::endl;
            throw std::logic_error(" --> Invalid number of cuts in cell");
            
        }

        cell_i++;
    }
}

/// New version of detect_cut_cells for discret level functions -> USING INTERFACE = 1/2
/// in cuthho_geom.hpp
template<typename T, size_t ET, typename Function>
void
detect_cut_cells3(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    std::cout<<"I AM IN DETECT CUT CELL3!!!!"<<std::endl;
    timecounter tc;
    tc.tic();
    //typedef typename cuthho_mesh<T, ET>::face_type  face_type;
    typedef typename cuthho_mesh<T, ET>::point_type point_type;
    //typedef typename cuthho_mesh<T, ET>::cell_type cell_type;
    T iso_val_interface = level_set_function.iso_val_interface ;
    std::cout<<"iso_val_interface = "<<iso_val_interface<<std::endl;
    size_t cell_i = 0;
    for (auto& cl : msh.cells)
    {
        auto fcs = faces(msh, cl);
        std::array< std::pair<size_t, point_type>, 2 >  cut_faces;

        size_t k = 0;
        for (size_t i = 0; i < fcs.size(); i++)
        {
            if ( is_cut(msh, fcs[i]) )
                cut_faces.at(k++) = std::make_pair(i, fcs[i].user_data.intersection_point);
        }

        /* If a face is cut, the cells that own the face are cut. Is this
         * unconditionally true? It should...fortunately this isn't avionics
         * software */

        if (k == 0)
        {
            
            auto is_positive = [&](const point_type& pt) -> bool {
            return level_set_function(pt,msh,cl) > iso_val_interface;
            };
            
            
            auto pts = points(msh, cl);
            
            if ( std::all_of(pts.begin(), pts.end(), is_positive) )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
     
            
            
            
            /*
            auto pts = points(msh, cl);
            auto pt = pts.begin();
            size_t counter = 0;
            while( ( pt!= pts.end() ) && ( level_set_function(*pt,msh,cl) > 0 ) )
            {
                counter++;
                pt++;
                
            }
             
            if ( counter == pts.size() )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
            */
             
        }
        //MODIFICARE QUAAAA
        if (k == 2)
        {
            cl.user_data.location = element_location::ON_INTERFACE;
            auto p0 = cut_faces[0].second;
            auto p1 = cut_faces[1].second;
            auto pt = p1 - p0;
            auto pn = p0 + point<T,2>(-pt.y(), pt.x());
            auto pn_prova = (p0+p1)/2.0 + 0.5*point<T,2>(-pt.y(), pt.x());
            //if(offset(msh,cl)== 119)
            //    std::cout<<"p0 = "<<p0<< " , p1 ="<<p1<<std::endl;
            // PRIMA ERA DA p0 ->   MODIFCATO, ora è pt  medio!
            /*
            if( !pt_in_cell(msh, pn, cl) )
            {
                std::cout<<"I chose another pn to ordering interface_points in 'detect_cut_cells2'."<<std::endl;
                T m_half = ( ps1.y() - pm.y() )/( ps1.x() - pm.x() );
                T q = pm.y() - m_half * pm.x() ;
                auto pt_bdry = search_boundary( msh , cl , pm , m_half , q , lm , level_set_function ) ;
                auto lm_bdry = level_set_function( pt_bdry , msh , cl );
            }
            */
            /*
            if(offset(msh,cl)== 137 || offset(msh,cl)== 138 || offset(msh,cl)== 134||offset(msh,cl)== 103){
                std::cout<<yellow<<bold<<"offset(msh,cl) = "<<offset(msh,cl)<<reset<<std::endl;
                auto pn_bis = (p0+p1)/2.0 + point<T,2>(-pt.y(), pt.x());
                std::cout<<"pn_bis = "<<pn_bis<< " , level_set_function(pn_bis,msh,cl) ="<<level_set_function(pn_bis,msh,cl) <<std::endl;
                auto pn_bis0 = (p0+p1)/2.0 + 0.5* point<T,2>(-pt.y(), pt.x());
                std::cout<<"pn_bis0 = "<<pn_bis0<< " , level_set_function(pn_bis0,msh,cl) ="<<level_set_function(pn_bis0,msh,cl) <<std::endl;
                auto pn_bis1 = p0 + 0.5 * point<T,2>(-pt.y(), pt.x());
                std::cout<<"pn_bis1 = "<<pn_bis1<< " , level_set_function(pn_bis1,msh,cl) ="<<level_set_function(pn_bis1,msh,cl)<<'\n' <<std::endl;
                
                std::cout<<"pn = "<<pn<< " , p0 = "<<p0<< " , p1 = "<<p1<<std::endl;
                std::cout<<"level_set_function(pn,msh,cl) = "<<level_set_function(pn,msh,cl)<< " , level_set_function(p0,msh,cl) = "<<level_set_function(p0,msh,cl)<< " , level_set_function(pn,msh,cl) = "<<level_set_function(p1,msh,cl)<<std::endl;
                std::cout<<"p0 - point<T,2>(-pt.y(), pt.x()) = "<<p0 - point<T,2>(-pt.y(), pt.x())<< " , level_set_function(p0 - point<T,2>(-pt.y(), pt.x()),msh,cl) = "<<level_set_function(p0 - point<T,2>(-pt.y(), pt.x()),msh,cl)<<std::endl;
            }
            */
            
            if( !(signbit(level_set_function(pn,msh,cl)-iso_val_interface) == signbit(level_set_function(pn_prova,msh,cl) -iso_val_interface) ) ){
                std::cout<<"p0 = "<<p0<< " , p1 = "<<p1<< std::endl;
                std::cout<<"pn = "<<pn<< " , pn_prova = "<<pn_prova<< " , level_set_function(pn,msh,cl) = "<<level_set_function(pn,msh,cl)<< " , level_set_function(pn_prova,msh,cl) = "<<level_set_function(pn_prova,msh,cl) <<std::endl;
                pn = pn_prova ;
            }
            
            if ( level_set_function(pn,msh,cl) >= iso_val_interface )
            {
                cl.user_data.p0 = p1;
                cl.user_data.p1 = p0;
            }
            else
            {
                cl.user_data.p0 = p0;
                cl.user_data.p1 = p1;
            }

            cl.user_data.interface.push_back(cl.user_data.p0);
            cl.user_data.interface.push_back(cl.user_data.p1);
        }

        if ( k != 0 && k != 2 ){
            auto pts = points(msh,cl);
            std::cout<<"Point[0] = "<<pts[0]<<" , point[1] = "<<pts[1]<<" , point[2] = "<<pts[2]<<" , point[3] = "<<pts[3]<<std::endl;
            std::cout<<"level_set_function(p0) = "<<level_set_function(pts[0],msh,cl) << " , level_set_function(p1) = "<<level_set_function(pts[1],msh,cl)<< " , level_set_function(p2) = "<<level_set_function(pts[2],msh,cl)<< " , level_set_function(p3) = "<<level_set_function(pts[3],msh,cl)<<std::endl;
            for (size_t i = 0; i < fcs.size(); i++)
            {
                if ( is_cut(msh, fcs[i]) )
                  std::cout<<"fcs[i].user_data.intersection_point = "<<fcs[i].user_data.intersection_point<<std::endl;
            }
           
            std::cout<<"ERROR: in cut cell "<<cell_i<<" there are k = "<<k<<" cuts!!!!"<<std::endl;
            throw std::logic_error(" --> Invalid number of cuts in cell");
            
        }

        cell_i++;
    }
    tc.toc();
    std::cout << bold << yellow << "detect_cut_cells3, time resolution: " << tc << " seconds" << reset << std::endl;
}


template<typename T, size_t ET, typename Function>
void
detect_cut_cells3_parallelized(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    std::cout<<"I AM IN DETECT CUT CELL3 PARALLELELIZED !!!!"<<std::endl;
    timecounter tc;
    tc.tic();
    //typedef typename cuthho_mesh<T, ET>::face_type  face_type;
    typedef typename cuthho_mesh<T, ET>::point_type point_type;
    //typedef typename cuthho_mesh<T, ET>::cell_type cell_type;
    T iso_val_interface = level_set_function.iso_val_interface ;
    std::cout<<"iso_val_interface = "<<iso_val_interface<<std::endl;
  
    
#ifdef HAVE_INTEL_TBB
    size_t n_cells = msh.cells.size();
    std::cout<<" I m in parallel zone"<<std::endl;
    tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
    [&] (size_t & cell_ind){
        auto& cl = msh.cells[cell_ind];
        
        auto fcs = faces(msh, cl);
        std::array< std::pair<size_t, point_type>, 2 >  cut_faces;
        
        size_t k = 0;
        for (size_t i = 0; i < fcs.size(); i++)
        {
            if ( is_cut(msh, fcs[i]) )
                cut_faces.at(k++) = std::make_pair(i, fcs[i].user_data.intersection_point);
        }

        /* If a face is cut, the cells that own the face are cut. Is this
        * unconditionally true? It should...fortunately this isn't avionics
        * software */

        if (k == 0)
        {
               
            auto is_positive = [&](const point_type& pt) -> bool {
            return level_set_function(pt) > iso_val_interface;
            };
               
               
            auto pts = points(msh, cl);
               
            if ( std::all_of(pts.begin(), pts.end(), is_positive) )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
        
               
               
               
            /*
            auto pts = points(msh, cl);
            auto pt = pts.begin();
            size_t counter = 0;
            while( ( pt!= pts.end() ) && ( level_set_function(*pt,msh,cl) > 0 ) )
            {
                counter++;
                pt++;
                   
            }
                
            if ( counter == pts.size() )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
            */
                
        }
        //MODIFICARE QUAAAA
        if (k == 2)
        {
            cl.user_data.location = element_location::ON_INTERFACE;
            auto p0 = cut_faces[0].second;
            auto p1 = cut_faces[1].second;
            auto pt = p1 - p0;
            auto pn = p0 + point<T,2>(-pt.y(), pt.x());
            auto pn_prova = (p0+p1)/2.0 + 0.5*point<T,2>(-pt.y(), pt.x());
            //if(offset(msh,cl)== 119)
            //    std::cout<<"p0 = "<<p0<< " , p1 ="<<p1<<std::endl;
            // PRIMA ERA DA p0 ->   MODIFCATO, ora è pt  medio!
            /*
            if( !pt_in_cell(msh, pn, cl) )
            {
                std::cout<<"I chose another pn to ordering interface_points in 'detect_cut_cells2'."<<std::endl;
                T m_half = ( ps1.y() - pm.y() )/( ps1.x() - pm.x() );
                T q = pm.y() - m_half * pm.x() ;
                auto pt_bdry = search_boundary( msh , cl , pm , m_half , q , lm , level_set_function ) ;
                auto lm_bdry = level_set_function( pt_bdry , msh , cl );
            }
            */
            /*
            if(offset(msh,cl)== 137 || offset(msh,cl)== 138 || offset(msh,cl)== 134||offset(msh,cl)== 103){
                std::cout<<yellow<<bold<<"offset(msh,cl) = "<<offset(msh,cl)<<reset<<std::endl;
                auto pn_bis = (p0+p1)/2.0 + point<T,2>(-pt.y(), pt.x());
                std::cout<<"pn_bis = "<<pn_bis<< " , level_set_function(pn_bis,msh,cl) ="<<level_set_function(pn_bis,msh,cl) <<std::endl;
                auto pn_bis0 = (p0+p1)/2.0 + 0.5* point<T,2>(-pt.y(), pt.x());
                std::cout<<"pn_bis0 = "<<pn_bis0<< " , level_set_function(pn_bis0,msh,cl) ="<<level_set_function(pn_bis0,msh,cl) <<std::endl;
                auto pn_bis1 = p0 + 0.5 * point<T,2>(-pt.y(), pt.x());
                std::cout<<"pn_bis1 = "<<pn_bis1<< " , level_set_function(pn_bis1,msh,cl) ="<<level_set_function(pn_bis1,msh,cl)<<'\n' <<std::endl;
                   
                std::cout<<"pn = "<<pn<< " , p0 = "<<p0<< " , p1 = "<<p1<<std::endl;
                std::cout<<"level_set_function(pn,msh,cl) = "<<level_set_function(pn,msh,cl)<< " , level_set_function(p0,msh,cl) = "<<level_set_function(p0,msh,cl)<< " , level_set_function(pn,msh,cl) = "<<level_set_function(p1,msh,cl)<<std::endl;
                std::cout<<"p0 - point<T,2>(-pt.y(), pt.x()) = "<<p0 - point<T,2>(-pt.y(), pt.x())<< " , level_set_function(p0 - point<T,2>(-pt.y(), pt.x()),msh,cl) = "<<level_set_function(p0 - point<T,2>(-pt.y(), pt.x()),msh,cl)<<std::endl;
            }
            */
               
            if( !(signbit(level_set_function(pn,msh,cl)-iso_val_interface) == signbit(level_set_function(pn_prova,msh,cl) -iso_val_interface) ) ){
                std::cout<<"p0 = "<<p0<< " , p1 = "<<p1<< std::endl;
                std::cout<<"pn = "<<pn<< " , pn_prova = "<<pn_prova<< " , level_set_function(pn,msh,cl) = "<<level_set_function(pn,msh,cl)<< " , level_set_function(pn_prova,msh,cl) = "<<level_set_function(pn_prova,msh,cl) <<std::endl;
                pn = pn_prova ;
            }
               
            if ( level_set_function(pn,msh,cl) >= iso_val_interface )
            {
                cl.user_data.p0 = p1;
                cl.user_data.p1 = p0;
            }
            else
            {
                cl.user_data.p0 = p0;
                cl.user_data.p1 = p1;
            }

            cl.user_data.interface.push_back(cl.user_data.p0);
            cl.user_data.interface.push_back(cl.user_data.p1);
        }

        if ( k != 0 && k != 2 ){
            auto pts = points(msh,cl);
            std::cout<<"Point[0] = "<<pts[0]<<" , point[1] = "<<pts[1]<<" , point[2] = "<<pts[2]<<" , point[3] = "<<pts[3]<<std::endl;
            std::cout<<"level_set_function(p0) = "<<level_set_function(pts[0],msh,cl) << " , level_set_function(p1) = "<<level_set_function(pts[1],msh,cl)<< " , level_set_function(p2) = "<<level_set_function(pts[2],msh,cl)<< " , level_set_function(p3) = "<<level_set_function(pts[3],msh,cl)<<std::endl;
            for (size_t i = 0; i < fcs.size(); i++)
            {
                if ( is_cut(msh, fcs[i]) )
                    std::cout<<"fcs[i].user_data.intersection_point = "<<fcs[i].user_data.intersection_point<<std::endl;
            }
              
            std::cout<<"ERROR: in cut cell "<<cell_ind<<" there are k = "<<k<<" cuts!!!!"<<std::endl;
            throw std::logic_error(" --> Invalid number of cuts in cell");
               
        }
    

    });
    tc.toc();
    std::cout << bold << yellow << "detect_cut_cells3_parallelized, time resolution: " << tc << " seconds" << reset << std::endl;
#else
      
    size_t cell_i = 0;
    for (auto& cl : msh.cells)
    {
        auto fcs = faces(msh, cl);
        std::array< std::pair<size_t, point_type>, 2 >  cut_faces;

        size_t k = 0;
        for (size_t i = 0; i < fcs.size(); i++)
        {
            if ( is_cut(msh, fcs[i]) )
                cut_faces.at(k++) = std::make_pair(i, fcs[i].user_data.intersection_point);
        }

        /* If a face is cut, the cells that own the face are cut. Is this
         * unconditionally true? It should...fortunately this isn't avionics
         * software */

        if (k == 0)
        {
            
            auto is_positive = [&](const point_type& pt) -> bool {
            return level_set_function(pt) > iso_val_interface;
            };
            
            
            auto pts = points(msh, cl);
            
            if ( std::all_of(pts.begin(), pts.end(), is_positive) )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
     
            
            
            
            /*
            auto pts = points(msh, cl);
            auto pt = pts.begin();
            size_t counter = 0;
            while( ( pt!= pts.end() ) && ( level_set_function(*pt,msh,cl) > 0 ) )
            {
                counter++;
                pt++;
                
            }
             
            if ( counter == pts.size() )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
            */
             
        }
        //MODIFICARE QUAAAA
        if (k == 2)
        {
            cl.user_data.location = element_location::ON_INTERFACE;
            auto p0 = cut_faces[0].second;
            auto p1 = cut_faces[1].second;
            auto pt = p1 - p0;
            auto pn = p0 + point<T,2>(-pt.y(), pt.x());
            auto pn_prova = (p0+p1)/2.0 + 0.5*point<T,2>(-pt.y(), pt.x());
            //if(offset(msh,cl)== 119)
            //    std::cout<<"p0 = "<<p0<< " , p1 ="<<p1<<std::endl;
            // PRIMA ERA DA p0 ->   MODIFCATO, ora è pt  medio!
            /*
            if( !pt_in_cell(msh, pn, cl) )
            {
                std::cout<<"I chose another pn to ordering interface_points in 'detect_cut_cells2'."<<std::endl;
                T m_half = ( ps1.y() - pm.y() )/( ps1.x() - pm.x() );
                T q = pm.y() - m_half * pm.x() ;
                auto pt_bdry = search_boundary( msh , cl , pm , m_half , q , lm , level_set_function ) ;
                auto lm_bdry = level_set_function( pt_bdry , msh , cl );
            }
            */
            /*
            if(offset(msh,cl)== 137 || offset(msh,cl)== 138 || offset(msh,cl)== 134||offset(msh,cl)== 103){
                std::cout<<yellow<<bold<<"offset(msh,cl) = "<<offset(msh,cl)<<reset<<std::endl;
                auto pn_bis = (p0+p1)/2.0 + point<T,2>(-pt.y(), pt.x());
                std::cout<<"pn_bis = "<<pn_bis<< " , level_set_function(pn_bis,msh,cl) ="<<level_set_function(pn_bis,msh,cl) <<std::endl;
                auto pn_bis0 = (p0+p1)/2.0 + 0.5* point<T,2>(-pt.y(), pt.x());
                std::cout<<"pn_bis0 = "<<pn_bis0<< " , level_set_function(pn_bis0,msh,cl) ="<<level_set_function(pn_bis0,msh,cl) <<std::endl;
                auto pn_bis1 = p0 + 0.5 * point<T,2>(-pt.y(), pt.x());
                std::cout<<"pn_bis1 = "<<pn_bis1<< " , level_set_function(pn_bis1,msh,cl) ="<<level_set_function(pn_bis1,msh,cl)<<'\n' <<std::endl;
                
                std::cout<<"pn = "<<pn<< " , p0 = "<<p0<< " , p1 = "<<p1<<std::endl;
                std::cout<<"level_set_function(pn,msh,cl) = "<<level_set_function(pn,msh,cl)<< " , level_set_function(p0,msh,cl) = "<<level_set_function(p0,msh,cl)<< " , level_set_function(pn,msh,cl) = "<<level_set_function(p1,msh,cl)<<std::endl;
                std::cout<<"p0 - point<T,2>(-pt.y(), pt.x()) = "<<p0 - point<T,2>(-pt.y(), pt.x())<< " , level_set_function(p0 - point<T,2>(-pt.y(), pt.x()),msh,cl) = "<<level_set_function(p0 - point<T,2>(-pt.y(), pt.x()),msh,cl)<<std::endl;
            }
            */
            
            if( !(signbit(level_set_function(pn,msh,cl)-iso_val_interface) == signbit(level_set_function(pn_prova,msh,cl) -iso_val_interface) ) ){
                std::cout<<"p0 = "<<p0<< " , p1 = "<<p1<< std::endl;
                std::cout<<"pn = "<<pn<< " , pn_prova = "<<pn_prova<< " , level_set_function(pn,msh,cl) = "<<level_set_function(pn,msh,cl)<< " , level_set_function(pn_prova,msh,cl) = "<<level_set_function(pn_prova,msh,cl) <<std::endl;
                pn = pn_prova ;
            }
            
            if ( level_set_function(pn,msh,cl) >= iso_val_interface )
            {
                cl.user_data.p0 = p1;
                cl.user_data.p1 = p0;
            }
            else
            {
                cl.user_data.p0 = p0;
                cl.user_data.p1 = p1;
            }

            cl.user_data.interface.push_back(cl.user_data.p0);
            cl.user_data.interface.push_back(cl.user_data.p1);
        }

        if ( k != 0 && k != 2 ){
            auto pts = points(msh,cl);
            std::cout<<"Point[0] = "<<pts[0]<<" , point[1] = "<<pts[1]<<" , point[2] = "<<pts[2]<<" , point[3] = "<<pts[3]<<std::endl;
            std::cout<<"level_set_function(p0) = "<<level_set_function(pts[0],msh,cl) << " , level_set_function(p1) = "<<level_set_function(pts[1],msh,cl)<< " , level_set_function(p2) = "<<level_set_function(pts[2],msh,cl)<< " , level_set_function(p3) = "<<level_set_function(pts[3],msh,cl)<<std::endl;
            for (size_t i = 0; i < fcs.size(); i++)
            {
                if ( is_cut(msh, fcs[i]) )
                  std::cout<<"fcs[i].user_data.intersection_point = "<<fcs[i].user_data.intersection_point<<std::endl;
            }
           
            std::cout<<"ERROR: in cut cell "<<cell_i<<" there are k = "<<k<<" cuts!!!!"<<std::endl;
            throw std::logic_error(" --> Invalid number of cuts in cell");
            
        }

        cell_i++;
    }
    
#endif
}

/// New version of refine_interface for discret level functions

template<typename T, size_t ET, typename Function>
void
refine_interface2(cuthho_mesh<T, ET>& msh, typename cuthho_mesh<T, ET>::cell_type& cl,
                 const Function& level_set_function, size_t min, size_t max)
{
    if ( (max-min) < 2 )
        return;

    typedef typename cuthho_mesh<T, ET>::point_type     point_type;

    size_t mid = (max+min)/2;
    auto p0 = cl.user_data.interface.at(min);
    auto p1 = cl.user_data.interface.at(max);
    auto pm = (p0+p1)/2.0;
    auto pt = p1 - p0;
    auto pn = point_type(-pt.y(), pt.x());
    auto ps1 = pm + pn;
    auto ps2 = pm - pn;

    auto lm = level_set_function(pm,msh,cl);
    auto ls1 = level_set_function(ps1,msh,cl);
    auto ls2 = level_set_function(ps2,msh,cl);

    point_type ip;
   // std::cout<<"the node of interface are "<<p0<<" and "<<p1<<". I search pm= "<<pm<<" in which phi = "<<lm<<" and ps1 e ps2 "<<ps1<<" and "<<ps2<<"equal to "<<ls1<<" , "<<ls2<<std::endl;
    if ( !((lm >= 0 && ls1 >= 0) || (lm < 0 && ls1 < 0)) )
    {
        auto threshold = diameter(msh, cl) / 1e20;
        ip = find_zero_crossing_in_cell(pm, ps1, level_set_function, threshold,msh,cl);
    }
    else if ( !((lm >= 0 && ls2 >= 0) || (lm < 0 && ls2 < 0)) )
    {
        auto threshold = diameter(msh, cl) / 1e20;
        ip = find_zero_crossing_in_cell(pm, ps2, level_set_function, threshold,msh,cl);
    }
    else
        throw std::logic_error("interface not found in search range");

    cl.user_data.interface.at(mid) = ip;

    refine_interface2(msh, cl, level_set_function, min, mid);
    refine_interface2(msh, cl, level_set_function, mid, max);
}

template<typename T, size_t ET, typename Function>
void
refine_interface2(cuthho_mesh<T, ET>& msh, const Function& level_set_function, size_t levels)
{
    if (levels == 0)
        return;

    size_t interface_points = iexp_pow(2, levels);

    for (auto& cl : msh.cells)
    {
        if ( !is_cut(msh, cl) )
            continue;

        
        std::cout<<yellow<<bold<<"--------------------> CELL = "<<offset(msh,cl)<<"<--------------------"<<reset<<std::endl;
        cl.user_data.interface.resize(interface_points+1);
        cl.user_data.interface.at(0)                = cl.user_data.p0;
        cl.user_data.interface.at(interface_points) = cl.user_data.p1;

        refine_interface2(msh, cl, level_set_function, 0, interface_points);
        
        std::cout<<"LIMIT CELL "<<offset(msh,cl)<<" are:"<<std::endl;
         std::cout<<"pt[0] = "<<points(msh,cl)[0]<<" , pt[1] = "<<points(msh,cl)[1]<<" , pt[2] = "<<points(msh,cl)[2]<<" , pt[3] = "<<points(msh,cl)[3]<<std::endl;
        
         for(size_t i_int = 0 ; i_int < interface_points + 1 ; i_int++ )
             std::cout<<"refined points are p = "<<cl.user_data.interface.at(i_int)<<std::endl;
         std::cout<<"--------------------> CELL = "<<offset(msh,cl)<<"<--------------------"<<std::endl;
    }
}



template<typename T, size_t ET ,typename Function >
typename cuthho_mesh<T, ET>::point_type
search_boundary( cuthho_mesh<T, ET>& msh, typename cuthho_mesh<T, ET>::cell_type& cl, typename cuthho_mesh<T, ET>::point_type& p_init ,T m , T q , T lm , const Function& level_set )
{
    typedef typename cuthho_mesh<T, ET>::point_type     point_type;
    auto pts = points(msh,cl);
    
    point_type pt_tmp0 = point_type( pts[0].x() , m* pts[0].x() + q ) ;
    point_type pt_tmp1 = point_type( pts[1].x() , m* pts[1].x() + q ) ;
    point_type pt_tmp2 = point_type( (pts[1].y()-q)/m , pts[1].y() ) ;
    point_type pt_tmp3 = point_type( (pts[2].y()-q)/m , pts[2].y() ) ;
    /*
    if( offset(msh,cl) == 1029 || offset(msh,cl) == 1082 )
    {
        std::cout<<yellow<<bold<<"search_boundary"<<reset<<std::endl;
        std::cout<<"pt_tmp0 = "<<pt_tmp0<<std::endl;
        std::cout<<"pt_tmp1 = "<<pt_tmp1<<std::endl;
        std::cout<<"pt_tmp2 = "<<pt_tmp2<<std::endl;
        std::cout<<"pt_tmp3 = "<<pt_tmp3<<std::endl;
    }
    */
    auto ls0 = level_set(pt_tmp0,msh,cl);
    auto ls1 = level_set(pt_tmp1,msh,cl);
    auto ls2 = level_set(pt_tmp2,msh,cl);
    auto ls3 = level_set(pt_tmp3,msh,cl);
    
    if ( pt_in_cell(msh , pt_tmp0 , cl) && ( !((lm >= 0 && ls0 >= 0) || (lm < 0 && ls0 < 0)) ) )
        return pt_tmp0 ;
    if ( pt_in_cell(msh , pt_tmp1 , cl) && ( !((lm >= 0 && ls1 >= 0) || (lm < 0 && ls1 < 0)) ) )
    return pt_tmp1 ;
    if ( pt_in_cell(msh , pt_tmp2 , cl) && ( !((lm >= 0 && ls2 >= 0) || (lm < 0 && ls2 < 0)) ) )
    return pt_tmp2 ;
    if ( pt_in_cell(msh , pt_tmp3 , cl) && ( !((lm >= 0 && ls3 >= 0) || (lm < 0 && ls3 < 0)) ) )
    return pt_tmp3 ;
    else{
        std::cout<<"In cell = "<<offset(msh,cl)<<" points(msh,cl)[0] = "<<points(msh,cl)[0]<<" points(msh,cl)[1] = "<<points(msh,cl)[1]<<" points(msh,cl)[2] = "<<points(msh,cl)[2]<<" points(msh,cl)[3] = "<<points(msh,cl)[3] <<std::endl;
        std::cout<<"m = "<<m<<" --> q = "<<q<<std::endl;
        std::cout<<"p_init = "<<p_init<<" --> pt_tmp0 = "<<pt_tmp0<<" , pt_tmp1 = "<<pt_tmp1<<" , pt_tmp2 = "<<pt_tmp2<<" , pt_tmp3 = "<<pt_tmp3<<std::endl;
        std::cout<<"ls0 = "<<ls0<<" , ls1 = "<<ls1<<" , ls2 = "<<ls2<<" , ls3 = "<<ls3<<" AND lm = "<<lm<<std::endl;
        std::cout<<"pt_in_cell( pt_tmp0 ) = "<<pt_in_cell(msh , pt_tmp0 , cl)<<" , pt_in_cell( pt_tmp1 ) = "<<pt_in_cell(msh , pt_tmp1 , cl)<<" , pt_in_cell( pt_tmp2 ) = "<<pt_in_cell(msh , pt_tmp2 , cl)<<" , pt_in_cel( pt_tmp3 ) = "<<pt_in_cell(msh , pt_tmp3 , cl)<<std::endl;
        T pp = pts[0].x();
        T dist = std::abs( pp - p_init.x() )/10.0;
        std::cout<<"DIST = "<<dist<< " and pp = "<<pp<< " and p_init.x() = "<<p_init.x() <<std::endl;
        point_type p0 = point_type( pp + dist , m* (pp-dist) + q ) ;
        point_type p1 = point_type( pp + (dist*2) , m* (pp+(dist*2)) + q ) ;
        point_type p2 = point_type( pp + (dist*3) , m* (pp+(dist*3)) + q ) ;
        point_type p3 = point_type( pp + (dist*4) , m* (pp+(dist*4)) + q ) ;
        point_type p4 = point_type( pp + (dist*5) , m* (pp+(dist*5)) + q ) ;
        point_type p5 = point_type( pp + (dist*6) , m* (pp+(dist*6)) + q ) ;
        point_type p6 = point_type( pp + (dist*7) , m* (pp+(dist*7)) + q ) ;
        point_type p7 = point_type( pp + (dist*8) , m* (pp+(dist*8)) + q ) ;
        point_type p8 = point_type( pp + (dist*9) , m* (pp+(dist*9)) + q ) ;
        std::cout<<"p0 = "<<p0<<" , level_set = "<<level_set(p0,msh,cl)<<" , p1 = "<<p1<<" , level_set = "<<level_set(p1,msh,cl)<<" , p2 = "<<p2<<" , level_set = "<<level_set(p2,msh,cl)<<" , p3 = "<<p3<<" , level_set = "<<level_set(p3,msh,cl)<<" ,p4 = "<<p4<<" , level_set = "<<level_set(p4,msh,cl)<<" ,p5 = "<<p5<<" , level_set = "<<level_set(p5,msh,cl)<<" , p6 = "<<p6<<" , level_set = "<<level_set(p6,msh,cl)<<", p7 = "<<p7<<" , level_set = "<<level_set(p7,msh,cl)<<" , p8 = "<<p8<<" , level_set = "<<level_set(p8,msh,cl)<<std::endl;
        
        //throw std::logic_error("search_boundary not find -> Stefano");
        //return p_init ;
        
        point_type ret ;
        T val_min = 1e10;
        if( pt_in_cell(msh , pt_tmp0 , cl) && std::abs(ls0) < val_min )
        {
            val_min = std::abs(ls0) ;
            ret = pt_tmp0 ;
            
        }
        if( pt_in_cell(msh , pt_tmp1 , cl) && std::abs(ls1) < val_min )
        {
            val_min = std::abs(ls1) ;
            ret = pt_tmp1 ;
                
        }
        if( pt_in_cell(msh , pt_tmp2 , cl) && std::abs(ls2) < val_min )
        {
            val_min = std::abs(ls2) ;
            ret = pt_tmp2 ;
                
        }
        if( pt_in_cell(msh , pt_tmp3 , cl) && std::abs(ls3) < val_min )
        {
            val_min = std::abs(ls3) ;
            ret = pt_tmp3 ;
                
        }
        return ret;
            
    }
    
  

}





template<typename T, size_t ET >
typename cuthho_mesh<T, ET>::point_type
search_boundary( cuthho_mesh<T, ET>& msh, typename cuthho_mesh<T, ET>::cell_type& cl, typename cuthho_mesh<T, ET>::point_type& p_init ,T m , T q )
{
    typedef typename cuthho_mesh<T, ET>::point_type     point_type;
    auto pts =points(msh,cl);
    
    point_type pt_tmp0 = point_type( pts[0].x() , m* pts[0].x() + q ) ;
    point_type pt_tmp1 = point_type( pts[1].x() , m* pts[1].x() + q ) ;
    point_type pt_tmp2 = point_type( (pts[1].y()-q)/m , pts[1].y() ) ;
    point_type pt_tmp3 = point_type( (pts[2].y()-q)/m , pts[2].y() ) ;
    
    
    if( pt_in_cell(msh , pt_tmp0 , cl) && !(p_init == pt_tmp0) )
        return pt_tmp0 ;
    if( pt_in_cell(msh , pt_tmp1 , cl) && !(p_init == pt_tmp1) )
        return pt_tmp1 ;
    if( pt_in_cell(msh , pt_tmp2 , cl) && !(p_init == pt_tmp2) )
        return pt_tmp2 ;
    if( pt_in_cell(msh , pt_tmp3 , cl) && !(p_init == pt_tmp3) )
        return pt_tmp3 ;
  

}



template<typename T, size_t ET, typename Function>
void
refine_interface_pro(cuthho_mesh<T, ET>& msh, typename cuthho_mesh<T, ET>::cell_type& cl,
                 const Function& level_set_function, size_t min, size_t max)
{
    if ( (max-min) < 2 )
        return;

    typedef typename cuthho_mesh<T, ET>::point_type     point_type;

    size_t mid = (max+min)/2;
    auto p0 = cl.user_data.interface.at(min);
    auto p1 = cl.user_data.interface.at(max);
    auto pm = (p0+p1)/2.0;
    auto pt = p1 - p0;
    auto pn = point_type(-pt.y(), pt.x());
    auto ps1 = pm + pn;
    auto ps2 = pm - pn;
    
    
    auto lm = level_set_function(pm,msh,cl);
    auto ls1 = level_set_function(ps1,msh,cl);
    auto ls2 = level_set_function(ps2,msh,cl);
    
    // CASE  MAX PT on the boudary
    T m_half = ( ps1.y() - pm.y() )/( ps1.x() - pm.x() );
    T q = pm.y() - m_half * pm.x() ;
    if( offset(msh,cl) == 119 )
    {
        std::cout<<yellow<<bold<<"CELL 119"<<reset<<std::endl;
        std::cout<<"p0 = "<<p0 << " , p1 = "<<p1<<std::endl;
        std::cout<<"ps1.y() = "<<ps1.y() << " , pm.y() = "<<pm.y()<<std::endl;
        std::cout<<"ps1.x() = "<<ps1.x() << " , pm.x() = "<<pm.x()<<std::endl;
        std::cout<<"ps1.x() = "<<ps1.x() << " , pm.x() = "<<pm.x()<<std::endl;
    }
    /*
    if( offset(msh,cl) == 118 )
    {
        T m_half_bis = ( ps2.y() - pm.y() )/( ps2.x() - pm.x() );
        T q_bis = pm.y() - m_half * pm.x() ;
        std::cout<<yellow<<bold<<"CELL 118"<<reset<<std::endl;
        std::cout<<"p0 = "<<p0 << " , p1 = "<<p1<<std::endl;
        std::cout<<"m_half = "<<m_half << " , m_half_bis = "<<m_half_bis<<std::endl;
        std::cout<<"q = "<<q << " , q_bis = "<<q_bis<<std::endl;
        
        std::cout<<"pm = "<<pm << " , level_set_function(pm) = "<<lm<<std::endl;
        std::cout<<"ps1 = "<<ps1 << " , level_set_function(ps1) = "<<ls1<<std::endl;
        std::cout<<"ps2 = "<<ps2 << " , level_set_function(ps2) = "<<ls2<<std::endl;
    }
    */
    auto pt_bdry = search_boundary( msh , cl , pm , m_half , q , lm , level_set_function ) ;
    auto lm_bdry = level_set_function( pt_bdry , msh , cl );
    /*
    if( offset(msh,cl) == 118 )
        std::cout<<"pt_bdry = "<<pt_bdry << " , level_set_function(lm_bdry) = "<<lm_bdry<<std::endl;
    */
    //std::cout<<"pm = "<<pm << " , level_set_function(pm) = "<<lm<<std::endl;
    //std::cout<<"ps1 = "<<ps1 << " , level_set_function(ps1) = "<<ls1<<std::endl;
    //std::cout<<"ps2 = "<<ps2 << " , level_set_function(ps2) = "<<ls2<<std::endl;
    
   
    point_type ip;
   // std::cout<<"the node of interface are "<<p0<<" and "<<p1<<". I search pm= "<<pm<<" in which phi = "<<lm<<" and ps1 e ps2 "<<ps1<<" and "<<ps2<<"equal to "<<ls1<<" , "<<ls2<<std::endl;
    if ( pt_in_cell(msh, ps1, cl) && ( !((lm >= 0 && ls1 >= 0) || (lm < 0 && ls1 < 0)) ) )
    {
        auto threshold = diameter(msh, cl) / 1e20;
        //auto threshold = diameter(msh, cl) / 1e10;
        ip = find_zero_crossing_in_cell(pm, ps1, level_set_function, threshold,msh,cl);
        //std::cout<<"OLD 1"<<std::endl;
    }
    else if ( pt_in_cell(msh, ps2, cl) && ( !((lm >= 0 && ls2 >= 0) || (lm < 0 && ls2 < 0)) ) )
    {
        auto threshold = diameter(msh, cl) / 1e20;
        //auto threshold = diameter(msh, cl) / 1e10;
        ip = find_zero_crossing_in_cell(pm, ps2, level_set_function, threshold,msh,cl);
        //std::cout<<"OLD 2"<<std::endl;
    }
    else if ( pt_in_cell(msh, pt_bdry, cl) && ( !((lm >= 0 && lm_bdry >= 0) || (lm < 0 && lm_bdry < 0)) ) )
    {
        auto threshold = diameter(msh, cl) / 1e20;
        //auto threshold = diameter(msh, cl) / 1e10;
        ip = find_zero_crossing_in_cell(pm, pt_bdry , level_set_function, threshold,msh,cl);
        //std::cout<<"BDRY NEW"<<std::endl;
    }
    else
    {
        //throw std::logic_error("interface not found in search range");
        //std::cout<<yellow<<bold<< "In cell "<<offset(msh,cl)<<" ---> implementing linear approximation. INTERFACE NOT FOUND."<<reset<<std::endl;
        //ip = pm;
        std::cout<<yellow<<bold<< "In cell "<<offset(msh,cl)<<" ---> implementing MINIMISATION ERROR APPROXIMATION. INTERFACE NOT FOUND."<<reset<<std::endl;
        point_type ret ;
        T val_min = 1e10;
        if( pt_in_cell(msh , ps1 , cl) && std::abs(ls1) < val_min )
        {
            val_min = std::abs(ls1) ;
            ret = ps1 ;
            std::cout<<"ps1 = "<<ps1 << " , ls1 = "<< ls1 <<std::endl;
            
        }
        if( pt_in_cell(msh , ps2 , cl) && std::abs(ls2) < val_min )
        {
            val_min = std::abs(ls2) ;
            ret = ps2 ;
            std::cout<<"ps2 = "<<ps2 << " , ls2 = "<< ls2 <<std::endl;
        }
        if( pt_in_cell(msh , pt_bdry , cl) && std::abs(lm_bdry) < val_min )
        {
            val_min = std::abs(lm_bdry) ;
            ret = pt_bdry ;
            std::cout<<"ppt_bdrys1 = "<<pt_bdry << " , lm_bdry = "<<lm_bdry<<std::endl;
        }
        if( pt_in_cell(msh , pm , cl) && std::abs(lm) < val_min )
        {
            val_min = std::abs(lm) ;
            ret = pm ;
            std::cout<<"pm = "<<ps1 << " , lm = "<<ls1<<std::endl;
        }
        std::cout<<"ret = "<<ret << std::endl;
        ip = ret;
        
    }
    /*
    if( offset(msh,cl) == 118 )
        std::cout<<"POINT INTERFACE Ip = "<<ip <<  " in pos = "<<mid<<std::endl;
    */
    cl.user_data.interface.at(mid) = ip;

    refine_interface_pro(msh, cl, level_set_function, min, mid);
    refine_interface_pro(msh, cl, level_set_function, mid, max);
}

template<typename T, size_t ET, typename Function>
void
refine_interface_pro(cuthho_mesh<T, ET>& msh, const Function& level_set_function, size_t levels)
{
    if (levels == 0)
        return;

    
    
    size_t interface_points = iexp_pow(2, levels);

    for (auto& cl : msh.cells)
    {
        if ( !is_cut(msh, cl) )
            continue;

        /*
        if( offset(msh,cl) == 118 )
        {
        std::cout<<yellow<<bold<<"--------------------> CELL = "<<offset(msh,cl)<<" <--------------------"<<reset<<std::endl;
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
        std::cout<<"INTERFACE_0 = "<<cl.user_data.p0 << " , INTERFACE_1 = "<<cl.user_data.p1 << std::endl;
        }
        */
        cl.user_data.interface.resize(interface_points+1);
        cl.user_data.interface.at(0)                = cl.user_data.p0;
        cl.user_data.interface.at(interface_points) = cl.user_data.p1;

        refine_interface_pro(msh, cl, level_set_function, 0, interface_points);
        /*
        if( offset(msh,cl) == 118 )
        {
        for(size_t i_int = 0 ; i_int < interface_points + 1 ; i_int++ )
            std::cout<<"refined points are p = "<<cl.user_data.interface.at(i_int)<<std::endl;
            
        std::cout<<"--------------------> FINE CELL <--------------------"<<std::endl;
        }
        */
    }
}

template<typename T, size_t ET ,typename Function >
typename cuthho_mesh<T, ET>::point_type
search_boundary3( cuthho_mesh<T, ET>& msh, typename cuthho_mesh<T, ET>::cell_type& cl, typename cuthho_mesh<T, ET>::point_type& p_init ,T m , T q , T lm , const Function& level_set , T iso_val_interface )
{
    typedef typename cuthho_mesh<T, ET>::point_type     point_type;
    auto pts = points(msh,cl);
    
    point_type pt_tmp0 = point_type( pts[0].x() , m* pts[0].x() + q ) ;
    point_type pt_tmp1 = point_type( pts[1].x() , m* pts[1].x() + q ) ;
    point_type pt_tmp2 = point_type( (pts[1].y()-q)/m , pts[1].y() ) ;
    point_type pt_tmp3 = point_type( (pts[2].y()-q)/m , pts[2].y() ) ;
    /*
    if( offset(msh,cl) == 1029 || offset(msh,cl) == 1082 )
    {
        std::cout<<yellow<<bold<<"search_boundary"<<reset<<std::endl;
        std::cout<<"pt_tmp0 = "<<pt_tmp0<<std::endl;
        std::cout<<"pt_tmp1 = "<<pt_tmp1<<std::endl;
        std::cout<<"pt_tmp2 = "<<pt_tmp2<<std::endl;
        std::cout<<"pt_tmp3 = "<<pt_tmp3<<std::endl;
    }
    */
    auto ls0 = level_set(pt_tmp0,msh,cl);
    auto ls1 = level_set(pt_tmp1,msh,cl);
    auto ls2 = level_set(pt_tmp2,msh,cl);
    auto ls3 = level_set(pt_tmp3,msh,cl);
    
    if ( pt_in_cell(msh , pt_tmp0 , cl) && ( !((lm >= iso_val_interface && ls0 >= iso_val_interface) || (lm < iso_val_interface && ls0 < iso_val_interface)) ) )
        return pt_tmp0 ;
    if ( pt_in_cell(msh , pt_tmp1 , cl) && ( !((lm >= iso_val_interface && ls1 >= iso_val_interface) || (lm < iso_val_interface && ls1 < iso_val_interface)) ) )
    return pt_tmp1 ;
    if ( pt_in_cell(msh , pt_tmp2 , cl) && ( !((lm >= iso_val_interface && ls2 >= iso_val_interface) || (lm < iso_val_interface && ls2 < iso_val_interface)) ) )
    return pt_tmp2 ;
    if ( pt_in_cell(msh , pt_tmp3 , cl) && ( !((lm >= iso_val_interface && ls3 >= iso_val_interface) || (lm < iso_val_interface && ls3 < iso_val_interface)) ) )
    return pt_tmp3 ;
    else{
        std::cout<<"In cell = "<<offset(msh,cl)<<" points(msh,cl)[0] = "<<points(msh,cl)[0]<<" points(msh,cl)[1] = "<<points(msh,cl)[1]<<" points(msh,cl)[2] = "<<points(msh,cl)[2]<<" points(msh,cl)[3] = "<<points(msh,cl)[3] <<std::endl;
        std::cout<<"m = "<<m<<" --> q = "<<q<<std::endl;
        std::cout<<"p_init = "<<p_init<<" --> pt_tmp0 = "<<pt_tmp0<<" , pt_tmp1 = "<<pt_tmp1<<" , pt_tmp2 = "<<pt_tmp2<<" , pt_tmp3 = "<<pt_tmp3<<std::endl;
        std::cout<<"ls0 = "<<ls0<<" , ls1 = "<<ls1<<" , ls2 = "<<ls2<<" , ls3 = "<<ls3<<" AND lm = "<<lm<<std::endl;
        std::cout<<"pt_in_cell( pt_tmp0 ) = "<<pt_in_cell(msh , pt_tmp0 , cl)<<" , pt_in_cell( pt_tmp1 ) = "<<pt_in_cell(msh , pt_tmp1 , cl)<<" , pt_in_cell( pt_tmp2 ) = "<<pt_in_cell(msh , pt_tmp2 , cl)<<" , pt_in_cel( pt_tmp3 ) = "<<pt_in_cell(msh , pt_tmp3 , cl)<<std::endl;
        T pp = pts[0].x();
        T dist = std::abs( pp - p_init.x() )/10.0;
        std::cout<<"DIST = "<<dist<< " and pp = "<<pp<< " and p_init.x() = "<<p_init.x() <<std::endl;
        point_type p0 = point_type( pp + dist , m* (pp-dist) + q ) ;
        point_type p1 = point_type( pp + (dist*2) , m* (pp+(dist*2)) + q ) ;
        point_type p2 = point_type( pp + (dist*3) , m* (pp+(dist*3)) + q ) ;
        point_type p3 = point_type( pp + (dist*4) , m* (pp+(dist*4)) + q ) ;
        point_type p4 = point_type( pp + (dist*5) , m* (pp+(dist*5)) + q ) ;
        point_type p5 = point_type( pp + (dist*6) , m* (pp+(dist*6)) + q ) ;
        point_type p6 = point_type( pp + (dist*7) , m* (pp+(dist*7)) + q ) ;
        point_type p7 = point_type( pp + (dist*8) , m* (pp+(dist*8)) + q ) ;
        point_type p8 = point_type( pp + (dist*9) , m* (pp+(dist*9)) + q ) ;
        std::cout<<"p0 = "<<p0<<" , level_set = "<<level_set(p0,msh,cl)<<" , p1 = "<<p1<<" , level_set = "<<level_set(p1,msh,cl)<<" , p2 = "<<p2<<" , level_set = "<<level_set(p2,msh,cl)<<" , p3 = "<<p3<<" , level_set = "<<level_set(p3,msh,cl)<<" ,p4 = "<<p4<<" , level_set = "<<level_set(p4,msh,cl)<<" ,p5 = "<<p5<<" , level_set = "<<level_set(p5,msh,cl)<<" , p6 = "<<p6<<" , level_set = "<<level_set(p6,msh,cl)<<", p7 = "<<p7<<" , level_set = "<<level_set(p7,msh,cl)<<" , p8 = "<<p8<<" , level_set = "<<level_set(p8,msh,cl)<<std::endl;
        
        //throw std::logic_error("search_boundary not find -> Stefano");
        //return p_init ;
        
        point_type ret ;
        T val_min = 1e10;
        if( pt_in_cell(msh , pt_tmp0 , cl) && std::abs(ls0-iso_val_interface) < val_min )
        {
            val_min = std::abs(ls0) ;
            ret = pt_tmp0 ;
            
        }
        if( pt_in_cell(msh , pt_tmp1 , cl) && std::abs(ls1-iso_val_interface) < val_min )
        {
            val_min = std::abs(ls1) ;
            ret = pt_tmp1 ;
                
        }
        if( pt_in_cell(msh , pt_tmp2 , cl) && std::abs(ls2-iso_val_interface) < val_min )
        {
            val_min = std::abs(ls2) ;
            ret = pt_tmp2 ;
                
        }
        if( pt_in_cell(msh , pt_tmp3 , cl) && std::abs(ls3-iso_val_interface) < val_min )
        {
            val_min = std::abs(ls3) ;
            ret = pt_tmp3 ;
                
        }
        return ret;
            
    }
    
  

}

// USING INTERFACE = 1/2
template<typename T, size_t ET, typename Function>
void
refine_interface_pro3(cuthho_mesh<T, ET>& msh, typename cuthho_mesh<T, ET>::cell_type& cl,
                 const Function& level_set_function, size_t min, size_t max)
{
    if ( (max-min) < 2 )
        return;

    typedef typename cuthho_mesh<T, ET>::point_type     point_type;
    T iso_val_interface = level_set_function.iso_val_interface ;
    size_t mid = (max+min)/2;
    auto p0 = cl.user_data.interface.at(min);
    auto p1 = cl.user_data.interface.at(max);
    auto pm = (p0+p1)/2.0;
    auto pt = p1 - p0;
    auto pn = point_type(-pt.y(), pt.x());
    auto ps1 = pm + pn;
    auto ps2 = pm - pn;
    
    
    auto lm = level_set_function(pm,msh,cl);
    auto ls1 = level_set_function(ps1,msh,cl);
    auto ls2 = level_set_function(ps2,msh,cl);
    
    // CASE  MAX PT on the boudary
    T m_half = ( ps1.y() - pm.y() )/( ps1.x() - pm.x() );
    T q = pm.y() - m_half * pm.x() ;
    if( offset(msh,cl) == 119 )
    {
        std::cout<<yellow<<bold<<"CELL 119"<<reset<<std::endl;
        std::cout<<"p0 = "<<p0 << " , p1 = "<<p1<<std::endl;
        std::cout<<"ps1.y() = "<<ps1.y() << " , pm.y() = "<<pm.y()<<std::endl;
        std::cout<<"ps1.x() = "<<ps1.x() << " , pm.x() = "<<pm.x()<<std::endl;
        std::cout<<"ps1.x() = "<<ps1.x() << " , pm.x() = "<<pm.x()<<std::endl;
    }
    /*
    if( offset(msh,cl) == 118 )
    {
        T m_half_bis = ( ps2.y() - pm.y() )/( ps2.x() - pm.x() );
        T q_bis = pm.y() - m_half * pm.x() ;
        std::cout<<yellow<<bold<<"CELL 118"<<reset<<std::endl;
        std::cout<<"p0 = "<<p0 << " , p1 = "<<p1<<std::endl;
        std::cout<<"m_half = "<<m_half << " , m_half_bis = "<<m_half_bis<<std::endl;
        std::cout<<"q = "<<q << " , q_bis = "<<q_bis<<std::endl;
        
        std::cout<<"pm = "<<pm << " , level_set_function(pm) = "<<lm<<std::endl;
        std::cout<<"ps1 = "<<ps1 << " , level_set_function(ps1) = "<<ls1<<std::endl;
        std::cout<<"ps2 = "<<ps2 << " , level_set_function(ps2) = "<<ls2<<std::endl;
    }
    */
    auto pt_bdry = search_boundary3( msh , cl , pm , m_half , q , lm , level_set_function , iso_val_interface) ;
    auto lm_bdry = level_set_function( pt_bdry , msh , cl );
    /*
    if( offset(msh,cl) == 118 )
        std::cout<<"pt_bdry = "<<pt_bdry << " , level_set_function(lm_bdry) = "<<lm_bdry<<std::endl;
    */
    //std::cout<<"pm = "<<pm << " , level_set_function(pm) = "<<lm<<std::endl;
    //std::cout<<"ps1 = "<<ps1 << " , level_set_function(ps1) = "<<ls1<<std::endl;
    //std::cout<<"ps2 = "<<ps2 << " , level_set_function(ps2) = "<<ls2<<std::endl;
    
   
    point_type ip;
   // std::cout<<"the node of interface are "<<p0<<" and "<<p1<<". I search pm= "<<pm<<" in which phi = "<<lm<<" and ps1 e ps2 "<<ps1<<" and "<<ps2<<"equal to "<<ls1<<" , "<<ls2<<std::endl;
    if ( pt_in_cell(msh, ps1, cl) && ( !((lm >= iso_val_interface && ls1 >= iso_val_interface) || (lm < iso_val_interface && ls1 < iso_val_interface)) ) )
    {
        auto threshold = diameter(msh, cl) / 1e20;
        //auto threshold = diameter(msh, cl) / 1e10;
        ip = find_zero_crossing_in_cell3(pm, ps1, level_set_function, threshold,msh,cl);
        //std::cout<<"OLD 1"<<std::endl;
    }
    else if ( pt_in_cell(msh, ps2, cl) && ( !((lm >= iso_val_interface && ls2 >= iso_val_interface) || (lm < iso_val_interface && ls2 < iso_val_interface)) ) )
    {
        auto threshold = diameter(msh, cl) / 1e20;
        //auto threshold = diameter(msh, cl) / 1e10;
        ip = find_zero_crossing_in_cell3(pm, ps2, level_set_function, threshold,msh,cl);
        //std::cout<<"OLD 2"<<std::endl;
    }
    else if ( pt_in_cell(msh, pt_bdry, cl) && ( !((lm >= iso_val_interface && lm_bdry >= iso_val_interface) || (lm < iso_val_interface && lm_bdry < iso_val_interface)) ) )
    {
        auto threshold = diameter(msh, cl) / 1e20;
        //auto threshold = diameter(msh, cl) / 1e10;
        ip = find_zero_crossing_in_cell3(pm, pt_bdry , level_set_function, threshold,msh,cl);
        //std::cout<<"BDRY NEW"<<std::endl;
    }
    else
    {
        //throw std::logic_error("interface not found in search range");
        //std::cout<<yellow<<bold<< "In cell "<<offset(msh,cl)<<" ---> implementing linear approximation. INTERFACE NOT FOUND."<<reset<<std::endl;
        //ip = pm;
        std::cout<<yellow<<bold<< "In cell "<<offset(msh,cl)<<" ---> implementing MINIMISATION ERROR APPROXIMATION. INTERFACE NOT FOUND."<<reset<<std::endl;
        point_type ret ;
        T val_min = 1e10;
        if( pt_in_cell(msh , ps1 , cl) && std::abs(ls1-iso_val_interface) < val_min )
        {
            val_min = std::abs(ls1) ;
            ret = ps1 ;
            std::cout<<"ps1 = "<<ps1 << " , ls1 = "<< ls1 <<std::endl;
            
        }
        if( pt_in_cell(msh , ps2 , cl) && std::abs(ls2-iso_val_interface) < val_min )
        {
            val_min = std::abs(ls2) ;
            ret = ps2 ;
            std::cout<<"ps2 = "<<ps2 << " , ls2 = "<< ls2 <<std::endl;
        }
        if( pt_in_cell(msh , pt_bdry , cl) && std::abs(lm_bdry-iso_val_interface) < val_min )
        {
            val_min = std::abs(lm_bdry) ;
            ret = pt_bdry ;
            std::cout<<"ppt_bdrys1 = "<<pt_bdry << " , lm_bdry = "<<lm_bdry<<std::endl;
        }
        if( pt_in_cell(msh , pm , cl) && std::abs(lm-iso_val_interface) < val_min )
        {
            val_min = std::abs(lm) ;
            ret = pm ;
            std::cout<<"pm = "<<ps1 << " , lm = "<<ls1<<std::endl;
        }
        std::cout<<"ret = "<<ret << std::endl;
        ip = ret;
        
    }
    /*
    if( offset(msh,cl) == 118 )
        std::cout<<"POINT INTERFACE Ip = "<<ip <<  " in pos = "<<mid<<std::endl;
    */
    cl.user_data.interface.at(mid) = ip;

    refine_interface_pro3(msh, cl, level_set_function, min, mid);
    refine_interface_pro3(msh, cl, level_set_function, mid, max);
}



template<typename T, size_t ET, typename Function>
void
refine_interface_pro3(cuthho_mesh<T, ET>& msh, const Function& level_set_function, size_t levels)
{
    if (levels == 0)
        return;

    
    
    size_t interface_points = iexp_pow(2, levels);

    for (auto& cl : msh.cells)
    {
        if ( !is_cut(msh, cl) )
            continue;

        /*
        if( offset(msh,cl) == 118 )
        {
        std::cout<<yellow<<bold<<"--------------------> CELL = "<<offset(msh,cl)<<" <--------------------"<<reset<<std::endl;
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
        std::cout<<"INTERFACE_0 = "<<cl.user_data.p0 << " , INTERFACE_1 = "<<cl.user_data.p1 << std::endl;
        }
        */
        cl.user_data.interface.resize(interface_points+1);
        cl.user_data.interface.at(0)                = cl.user_data.p0;
        cl.user_data.interface.at(interface_points) = cl.user_data.p1;

        refine_interface_pro3(msh, cl, level_set_function, 0, interface_points);
        /*
        if( offset(msh,cl) == 118 )
        {
        for(size_t i_int = 0 ; i_int < interface_points + 1 ; i_int++ )
            std::cout<<"refined points are p = "<<cl.user_data.interface.at(i_int)<<std::endl;
            
        std::cout<<"--------------------> FINE CELL <--------------------"<<std::endl;
        }
        */
    }
}


template<typename T, size_t ET, typename Function >
void
refine_interface_angle(cuthho_mesh<T, ET>& msh, typename cuthho_mesh<T, ET>::cell_type& cl,
                 const Function& level_set_function, size_t min, size_t max , typename cuthho_mesh<T, ET>::point_type& p_init , T h , bool pos , int multiplicity , T angle0 , T angle1 )
{
    if ( (max-min) < 2 )
        return;

    typedef typename cuthho_mesh<T, ET>::point_type     point_type;
   
     
    
    T angle_half = ( angle0 + angle1)/2.0;
    T m_half = tan(angle_half);
    
    std::cout<<bold<<yellow<<"CHECK ANGLE --------> "<< reset <<"angle0 = "<<angle0 * 180 / M_PI << " , angle1 = "<<angle1 * 180 / M_PI <<" and angle_half = "<<angle_half * 180 / M_PI <<std::endl;
    
    /*
    // In the case h is not long enough!
    T h_max = 0.0;
    if(multiplicity > 1  )
    {
        T cateto_min = 2*m_half/h ;
        h_max = std::max( h , sqrt( pow(cateto_min,2) + pow(h,2)/4.0 ) );
    }
    */
   
    // CASE h:
    T val = sqrt( pow(h,2)/( 1+pow(m_half,2) ) ) ;
    
    T x_new0 = p_init.x() + val ;
    T y_new0 = p_init.y() + ( x_new0 - p_init.x() )*m_half ;
    point_type pt_half0 = point_type(x_new0 , y_new0) ;
    
    auto lm0 = level_set_function( pt_half0 , msh , cl );
    
    T x_new1 = p_init.x() - val ;
    T y_new1 = p_init.y() + ( x_new1 - p_init.x() )*m_half ;
    point_type pt_half1 = point_type(x_new1,y_new1) ;
    
    auto lm1 = level_set_function(pt_half1,msh,cl);
    
    
    
    
    // CASE h_max = h*sqrt(2)
    T h_max = h*sqrt(2.0) ;
    T val_max = sqrt( pow(h_max,2)/( 1+pow(m_half,2) ) ) ;
    T x_new_max0 = p_init.x() + val_max ;
    T y_new_max0 = p_init.y() + ( x_new_max0 - p_init.x() )*m_half ;
    T x_new_max1 = p_init.x() - val_max ;
    T y_new_max1 = p_init.y() + ( x_new_max1 - p_init.x() )*m_half ;
    
    
    point_type pt_half_max0 = point_type(x_new_max0,y_new_max0) ;
    point_type pt_half_max1 = point_type(x_new_max1,y_new_max1) ;
     
    auto lm_max0 = level_set_function(pt_half_max0,msh,cl);
    auto lm_max1 = level_set_function(pt_half_max1,msh,cl);
    
    // CASE h_min = h/2
    T h_min = h/2.0 ;
    T val_min = sqrt( pow(h_min,2)/( 1+pow(m_half,2) ) ) ;
    T x_new_min0 = p_init.x() + val_min ;
    T y_new_min0 = p_init.y() + ( x_new_min0 - p_init.x() )*m_half ;
    T x_new_min1 = p_init.x() - val_min ;
    T y_new_min1 = p_init.y() + ( x_new_min1 - p_init.x() )*m_half ;
    
    
    point_type pt_half_min0 = point_type(x_new_min0,y_new_min0) ;
    point_type pt_half_min1 = point_type(x_new_min1,y_new_min1) ;
     
    auto lm_min0 = level_set_function(pt_half_min0,msh,cl);
    auto lm_min1 = level_set_function(pt_half_min1,msh,cl);
    
     
    
    // CASE PT on the boudary
    T q = p_init.y() - m_half * p_init.x() ;
    
    auto pt_bdry = search_boundary( msh , cl , p_init , m_half , q ) ;
    auto lm_bdry = level_set_function(pt_bdry,msh,cl);
    
    
    
    
    
    size_t mid = (max+min)/2;
    point_type ip;
    auto p0 = cl.user_data.interface.at(min);
    auto p1 = cl.user_data.interface.at(max);
    auto pm = (p0+p1)/2.0;
    //std::cout<<"p_init = "<<p_init<< " level_set(p_init) = "<<level_set_function(p_init,msh,cl)<<std::endl;
   
    
    if ( pt_in_cell(msh, pt_half0, cl) && !((lm0 >= 0 && pos == TRUE) || (lm0 < 0 && pos == FALSE ) ) )
    {
        
        auto threshold = diameter(msh, cl) / 1e20;
        ip = find_zero_crossing_in_cell(p_init, pt_half0, level_set_function, threshold,msh,cl);
        std::cout<<"NORMAL + --> pt_half0 = "<<pt_half0<< " and lm0 = "<<lm0<<" ------> ip = "<<ip<<std::endl;
    }
    else if ( pt_in_cell(msh, pt_half1, cl) && !((lm1 >= 0 && pos == TRUE) || (lm1 < 0 && pos == FALSE )) )
    {
        
        auto threshold = diameter(msh, cl) / 1e20;
        ip = find_zero_crossing_in_cell(p_init, pt_half1, level_set_function, threshold,msh,cl);
        std::cout<<"NORMAL - --> pt_half1 = "<<pt_half1<< " and lm1 = "<<lm1<<" ------> ip = "<<ip<<std::endl;
        
    }
    else if ( pt_in_cell(msh, pt_bdry, cl) && !((lm_bdry >= 0 && pos == TRUE) || (lm_bdry < 0 && pos == FALSE )) )
    {
        //std::cout<<"CHECK IF MIN POINT (WITH -) IS IN CELL"<<std::endl;
        
        auto threshold = diameter(msh, cl) / 1e20;
        ip = find_zero_crossing_in_cell(p_init, pt_bdry, level_set_function, threshold,msh,cl);
        std::cout<<"BDRY - --> pt_bdry = "<<pt_bdry<< " and lm_bdry = "<<lm_bdry<<" --> ip = "<<ip<<std::endl;
        
    }
    
    else if ( pt_in_cell(msh, pt_half_max0, cl) && !((lm_max0 >= 0 && pos == TRUE) || (lm_max0 < 0 && pos == FALSE )) )
    {
        //std::cout<<"CHECK IF MAX POINT (WITH +) IS IN CELL"<<std::endl;
        
        auto threshold = diameter(msh, cl) / 1e20;
        ip = find_zero_crossing_in_cell(p_init, pt_half_max0, level_set_function, threshold,msh,cl);
        std::cout<<"MAX + --> pt_max0 = "<<pt_half_max0<< " and lm_max0 = "<<lm_max0<<" --> ip = "<<ip<<std::endl;
        
    }
    else if ( pt_in_cell(msh, pt_half_max1, cl) && !((lm_max1 >= 0 && pos == TRUE) || (lm_max1 < 0 && pos == FALSE )) )
    {
        //std::cout<<"CHECK IF MAX POINT (WITH -) IS IN CELL"<<std::endl;
        
        auto threshold = diameter(msh, cl) / 1e20;
        ip = find_zero_crossing_in_cell(p_init, pt_half_max1, level_set_function, threshold,msh,cl);
        std::cout<<"MAX - --> pt_max1 = "<<pt_half_max1<< " and lm_max1 = "<<lm_max1<<" --> ip = "<<ip<<std::endl;
        
    }
    else if ( pt_in_cell(msh, pt_half_min0, cl) && !((lm_min0 >= 0 && pos == TRUE) || (lm_min0 < 0 && pos == FALSE )) )
    {
        //std::cout<<"CHECK IF MIN POINT (WITH +) IS IN CELL"<<std::endl;
        
        auto threshold = diameter(msh, cl) / 1e20;
        ip = find_zero_crossing_in_cell(p_init, pt_half_min0, level_set_function, threshold,msh,cl);
        std::cout<<"MIN + --> pt_min0 = "<<pt_half_min0<< " and lm_min0 = "<<lm_min0<<" --> ip = "<<ip<<std::endl;
        
    }
    else if ( pt_in_cell(msh, pt_half_min1, cl) && !((lm_min1 >= 0 && pos == TRUE) || (lm_min1 < 0 && pos == FALSE )) )
    {
        //std::cout<<"CHECK IF MIN POINT (WITH -) IS IN CELL"<<std::endl;
        
        auto threshold = diameter(msh, cl) / 1e20;
        ip = find_zero_crossing_in_cell(p_init, pt_half_min1, level_set_function, threshold,msh,cl);
        std::cout<<"MIN - --> pt_min1 = "<<pt_half_min1<< " and lm_min1 = "<<lm_min1<<" --> ip = "<<ip<<std::endl;
        
    }
    
    else
    {
        // IN THE CASE I DON'T FIND THE POINT I CONSIDER IT LINEAR
    
         std::cout<<"-----> ATTENTION: INTERFACE_REFINE3-> POINT DID NOT FIND, LINEAR APPROXIMATION EMPLOYED!"<<std::endl;
        std::cout<<"p_init = "<<p_init<< " level_set(p_init) = "<<level_set_function(p_init,msh,cl)<<std::endl;
        std::cout<<"CASE + : pt_half0 = "<<pt_half0<< " and lm0 = "<<lm0<<std::endl;
         std::cout<<"CASE - : pt_half1 = "<<pt_half1<< " and lm1 = "<<lm1<<std::endl;
        std::cout<<"CASE MAX+: pt_half_max0 = "<<pt_half_max0<< " and lm_max0 = "<<lm_max0<<std::endl;
        std::cout<<"CASE MAX-: pt_half_max1 = "<<pt_half_max1<< " and lm_max1 = "<<lm_max1<<std::endl;
        std::cout<<"CASE MIN +: pt_half_min0 = "<<pt_half_min0<< " and lm_min0 = "<<lm_min0<<std::endl;
        std::cout<<"CASE MIN -: pt_half_min1 = "<<pt_half_min1<< " and lm_min1 = "<<lm_min1<<std::endl;
        std::cout<<"CASE BDRY: pt_bdry = "<<pt_bdry<< " and lm_bdry = "<<lm_bdry<<std::endl;
        std::cout<<"--> ip = pm =  "<< pm <<std::endl;
        std::cout<<"ATTENTION: INTERFACE_REFINE3-> POINT DID NOT FIND, LINEAR APPROXIMATION EMPLOYED! <-------"<<std::endl;
        ip = pm ;
    }
    
   
    cl.user_data.interface.at(mid) = ip;
    
    refine_interface_angle( msh, cl, level_set_function, min , mid , p_init, h , pos, multiplicity , angle0 , angle_half );
    refine_interface_angle( msh, cl, level_set_function, mid , max , p_init , h , pos, multiplicity , angle_half , angle1 );
}






template<typename T, size_t ET, typename Function>
void
refine_interface_angle2(cuthho_mesh<T, ET>& msh, const Function& level_set_function, size_t levels)
{
    if (levels == 0)
        return;

    typedef typename cuthho_mesh<T, ET>::point_type point_type;
    size_t interface_points = iexp_pow(2, levels);

    for (auto& cl : msh.cells)
    {
        if ( !is_cut(msh, cl) )
            continue;

        cl.user_data.interface.resize(interface_points+1);
        cl.user_data.interface.at(0)                = cl.user_data.p0;
        cl.user_data.interface.at(interface_points) = cl.user_data.p1;
        
        // ADDED BY STEFANO
        point_type pos0 = cl.user_data.p0;
        point_type pos1 = cl.user_data.p1;
        std::vector< point_type >  pos_point;
        std::vector< point_type >  neg_point;
        
        bool positive = TRUE;
        int multiplicity = 1;
        std::vector<size_t> position_neg , position_pos ;
        T angle0 , angle1 ;
        point_type pm ;
        
        size_t counter = 0;
        for (auto& pt : points(msh,cl) )
        {
            if ( level_set_function(pt , msh , cl ) < 0 ){
                neg_point.push_back(pt);
                position_neg.push_back( counter );
            }
            else{
                pos_point.push_back(pt);
                position_pos.push_back( counter );
            }
            counter++;
        }
       
        /// FIND  EXTREME  ANGLES
        if( neg_point.size() == 1 ){
            pm = neg_point[0] ;
            positive = FALSE ;
            if(position_neg[0] == 0){
                angle0 = 0.0*M_PI;
                angle1 = M_PI/2.0 ;
            }
            else if(position_neg[0] == 1){
                angle0 = M_PI/2.0 ;
                angle1 = M_PI ;
            }
            else if(position_neg[0] == 2){
                angle0 = M_PI ;
                angle1 = 3.0*M_PI/2.0 ;
            }
            else if(position_neg[0] == 3){
                angle0 = 3.0*M_PI/2.0 ;
                angle1 = 0.0*M_PI ;
            }
            else{
                throw std::logic_error("POSITION ANGLE REFINE INTERFACE WRONG");
            }
        }
        else if( pos_point.size() == 1 ){
            pm = pos_point[0] ;
            if(position_pos[0] == 0){
                angle0 = 0.0*M_PI;
                angle1 = M_PI/2.0 ;
            }
            else if(position_pos[0] == 1){
                angle0 = M_PI/2.0 ;
                angle1 = M_PI ;
            }
            else if(position_pos[0] == 2){
                angle0 = M_PI ;
                angle1 = 3.0*M_PI/2.0 ;
            }
            else if(position_pos[0] == 3){
                angle0 = 3.0*M_PI/2.0 ;
                angle1 = 0.0*M_PI ;
            }
            else{
                throw std::logic_error("POSITION ANGLE REFINE INTERFACE WRONG");
            }
            
        }
        else
        {
            // MULTIPLICITY 2 -> #pos_point = #neg_point = 2
            if(position_pos[0] == 0 && position_pos[1] == 1){
                pm = (pos_point[0] +pos_point[1] )/2.0;
                multiplicity = 2 ;
                angle0 = atan( (pos0.y() - pm.y()) / (pos0.x() - pm.x()) );
                angle1 = atan( (pos1.y()-pm.y()) / (pos1.x()-pm.x()) );
                if(angle0 < 0 )
                    angle0 += M_PI ;
                else
                    angle1 += M_PI ;
            }
            else if(position_pos[0] == 1 && position_pos[1] == 2){
                pm = (pos_point[0] +pos_point[1] )/2.0;
                multiplicity = 2 ;
                angle0 = M_PI + atan( (pos0.y()-pm.y()) / (pos0.x()-pm.x()) );
                angle1 = M_PI + atan( (pos1.y()-pm.y()) / (pos1.x()-pm.x()) );
    
            }
            else if(position_pos[0] == 2 && position_pos[1] == 3){
                
                pm = (pos_point[0] +pos_point[1] )/2.0;
                multiplicity = 2 ;
                angle0 = atan( (pos0.y() - pm.y()) / (pos0.x() - pm.x()) );
                angle1 = atan( (pos1.y()-pm.y()) / (pos1.x()-pm.x()) );
                if(angle0 > 0 )
                    angle0 += M_PI ;
                else
                    angle1 += M_PI ;
                
            }
            else if(position_pos[0] == 0 && position_pos[1] == 3){
                
                pm = (pos_point[0] +pos_point[1] )/2.0;
                multiplicity = 2 ;
                angle0 = atan( (pos0.y() - pm.y()) / (pos0.x() - pm.x()) );
                angle1 = atan( (pos1.y()-pm.y()) / (pos1.x()-pm.x()) );
               
            }
            else if(position_pos[0] == 0 && position_pos[1] == 2){
                positive = FALSE ;
                pm = points(msh,cl)[1];
                //multiplicity = 2 ;
                angle0 = M_PI/2.0 ;
                angle1 = M_PI ;
               
            }
            else if(position_pos[0] == 1 && position_pos[1] == 3){
                positive = FALSE ;
                pm = points(msh,cl)[0];
                //multiplicity = 2 ;
                angle0 = 0.0*M_PI ;
                angle1 = M_PI/2.0 ;
               
            }
            else{
                throw std::logic_error("POSITION ANGLE REFINE INTERFACE WRONG");
            }
        }
        
        
        if(angle0 > angle1 )
        {
            T tmp = angle1 ;
            angle1 = angle0 ;
            angle0 = tmp ;
        }
        std::cout<<"angle0 = "<<angle0<< " and angle1 = "<<angle1<<std::endl;
            
        
        auto checK_sign = level_set_function(pm,msh,cl);
        
        
        if( !(signbit(checK_sign) && !positive) || !(!signbit(checK_sign) && positive) )
        {
            std::cout<<"LEVEL SET(Pm) = "<<checK_sign<< " and sign used is = "<<positive<<std::endl;
            throw std::logic_error("HO FATTO ERRORE IN POSITIVE SIGN CHECKING");
        }
            
        
        T h = std::min(level_set_function.params.hx() , level_set_function.params.hy() );
       
            
        refine_interface_angle( msh, cl, level_set_function, 0 , interface_points , pm , h , positive, multiplicity , angle0 , angle1 );
    }

}

template<typename T, size_t ET, typename Function>
void
refine_interface_angle(cuthho_mesh<T, ET>& msh, const Function& level_set_function, size_t levels)
{
    if (levels == 0)
        return;

    typedef typename cuthho_mesh<T, ET>::point_type point_type;
    //typedef typename cuthho_mesh<T, ET>::node_type node_type;
    size_t interface_points = iexp_pow(2, levels);

    for (auto& cl : msh.cells)
    {
        if ( !is_cut(msh, cl) )
            continue;

        cl.user_data.interface.resize(interface_points+1);
        cl.user_data.interface.at(0)                = cl.user_data.p0;
        cl.user_data.interface.at(interface_points) = cl.user_data.p1;
        std::cout<<yellow<<bold<<"--------------------> CELL = "<<offset(msh,cl)<<" <--------------------"<<reset<<std::endl;
        // ADDED BY STEFANO
        point_type pos0 = cl.user_data.p0;
        point_type pos1 = cl.user_data.p1;
       
        bool positive = TRUE;
        int multiplicity = 1;
        std::vector<size_t> position_neg , position_pos ;
        T angle0 , angle1 ;
        point_type pm ;
        
        size_t counter = 0;
        for (auto& nd :  nodes(msh,cl) )
        {
            
            if( nd.user_data.location == element_location::IN_NEGATIVE_SIDE ){
                position_neg.push_back( counter );
            }
            else{
               
                position_pos.push_back( counter );
            }
            counter++;
            //std::cout<<"nd = "<<nd.ptid<<std::endl;
        }
       
        /// FIND  EXTREME  ANGLES
        if( position_neg.size() == 1 )
        {
            
            pm = points(msh,cl)[position_neg[0]] ;
            std::cout<<"POINT"<<pm<<" , position_neg = "<<position_neg[0]<<std::endl;
            positive = FALSE ;
            if(position_neg[0] == 0){
                angle0 = 0.0*M_PI;
                angle1 = M_PI/2.0 ;
            }
            else if(position_neg[0] == 1){
                angle0 = M_PI/2.0 ;
                angle1 = M_PI ;
            }
            else if(position_neg[0] == 2){
                angle0 = M_PI ;
                angle1 = 3.0*M_PI/2.0 ;
            }
            else if(position_neg[0] == 3){
                angle0 = 3.0*M_PI/2.0 ;
                angle1 = 2.0*M_PI ;
            }
            else{
                throw std::logic_error("POSITION ANGLE REFINE INTERFACE WRONG");
            }
        }
        else if( position_pos.size() == 1 )
        {
            std::cout<<yellow<<bold<<"CASO POSITIVO MA TOGLIEREI E METTEREI SOLO CASI NEGATIVI!!!"<<reset<<std::endl;
            pm = points(msh,cl)[position_pos[0] ] ;
            std::cout<<"POINT"<<pm<<" , position_pos = "<<position_pos[0]<<std::endl;
            if(position_pos[0] == 0){
                angle0 = 0.0*M_PI;
                angle1 = M_PI/2.0 ;
            }
            else if(position_pos[0] == 1){
                angle0 = M_PI/2.0 ;
                angle1 = M_PI ;
            }
            else if(position_pos[0] == 2){
                angle0 = M_PI ;
                angle1 = 3.0*M_PI/2.0 ;
            }
            else if(position_pos[0] == 3){
                angle0 = 3.0*M_PI/2.0 ;
                angle1 = 2.0*M_PI ;
            }
            else{
                throw std::logic_error("POSITION ANGLE REFINE INTERFACE WRONG");
            }
            
        }
        else
        {
            //std::cout<<"sono qua 4 NEW"<<std::endl;
            //if(plus_close(p0,))
            // MULTIPLICITY 2 -> #pos_point = #neg_point = 2
            if(position_neg[0] == 0 && position_neg[1] == 1){
                positive = FALSE ;
                pm = (points(msh,cl)[position_neg[0] ] + points(msh,cl)[position_neg[1] ] )/2.0 ;
                std::cout<<"POINT "<<pm<<" , position_neg[0][1] --> "<<position_neg[0]<<" , "<<position_neg[1]<<std::endl;
                multiplicity = 2 ;
                angle0 = atan( (pos0.y() - pm.y()) / (pos0.x() - pm.x()) );
                angle1 = atan( (pos1.y()-pm.y()) / (pos1.x()-pm.x()) );
                if(angle0 < 0 )
                    angle0 += M_PI ;
                else
                    angle1 += M_PI ;
            }
            else if(position_neg[0] == 1 && position_neg[1] == 2){
                positive = FALSE ;
                pm = (points(msh,cl)[position_neg[0] ] + points(msh,cl)[position_neg[1] ] )/2.0 ;
                std::cout<<"POINT "<<pm<<" , position_neg[0][1] --> "<<position_neg[0]<<" , "<<position_neg[1]<<std::endl;
                multiplicity = 2 ;
                angle0 = M_PI + atan( (pos0.y()-pm.y()) / (pos0.x()-pm.x()) );
                angle1 = M_PI + atan( (pos1.y()-pm.y()) / (pos1.x()-pm.x()) );
    
            }
            else if(position_neg[0] == 2 && position_neg[1] == 3){
                positive = FALSE ;
                pm = (points(msh,cl)[position_neg[0] ] + points(msh,cl)[position_neg[1] ] )/2.0 ;
                std::cout<<"POINT "<<pm<<" , position_neg[0][1] --> "<<position_neg[0]<<" , "<<position_neg[1]<<std::endl;
                multiplicity = 2 ;
                angle0 = atan( (pos0.y() - pm.y()) / (pos0.x() - pm.x()) );
                angle1 = atan( (pos1.y()-pm.y()) / (pos1.x()-pm.x()) );
                if(angle0 > 0 )
                    angle0 += M_PI ;
                else
                    angle1 += M_PI ;
                
                if(angle0 < 0 )
                    angle0 = 2.0*M_PI + angle0 ;
                else
                    angle1 = 2.0*M_PI + angle1 ;
                
            }
            else if(position_neg[0] == 0 && position_neg[1] == 3){
                positive = FALSE ;
                pm = (points(msh,cl)[position_neg[0] ] + points(msh,cl)[position_neg[1] ] )/2.0 ;
                std::cout<<"POINT "<<pm<<" , position_neg[0][1] --> "<<position_neg[0]<<" , "<<position_neg[1]<<std::endl;
                multiplicity = 2 ;
                angle0 = atan( (pos0.y() - pm.y()) / (pos0.x() - pm.x()) );
                angle1 = atan( (pos1.y()-pm.y()) / (pos1.x()-pm.x()) );
               
            }
            else if(position_neg[0] == 0 && position_neg[1] == 2){
                //positive = FALSE ;
                pm = points(msh,cl)[1];
                std::cout<<"POINT "<<pm<<" , position_neg[0][1] --> "<<position_neg[0]<<" , "<<position_neg[1]<<std::endl;
                //multiplicity = 2 ;
                angle0 = M_PI/2.0 ;
                angle1 = M_PI ;
               
            }
            else if(position_neg[0] == 1 && position_neg[1] == 3){
                //positive = FALSE ;
                pm = points(msh,cl)[0];
                std::cout<<"POINT "<<pm<<" , position_neg[0][1] --> "<<position_neg[0]<<" , "<<position_neg[1]<<std::endl;
                //multiplicity = 2 ;
                angle0 = 0.0*M_PI ;
                angle1 = M_PI/2.0 ;
               
            }
            else{
                throw std::logic_error("POSITION ANGLE REFINE INTERFACE WRONG");
            }
        }
        
        
        if(angle0 > angle1 )
        {
            T tmp = angle1 ;
            angle1 = angle0 ;
            angle0 = tmp ;
        }
        //std::cout<<"CHECK ANGLE --------> angle0 = "<<angle0<< " and angle1 = "<<angle1<<std::endl;
            
        
        auto checK_sign = level_set_function(pm,msh,cl);
        
        if( (positive==FALSE && signbit(checK_sign) == 0 ) || (positive==TRUE && signbit(checK_sign) == 1 ) )
        {
            std::cout<<"LEVEL SET(Pm) = "<<checK_sign<< " and sign used is = "<<positive<<" and signbit is "<<signbit(checK_sign)<<std::endl;
            throw std::logic_error("HO FATTO ERRORE IN POSITIVE SIGN CHECKING");
        }
            
        
        T h = std::min(level_set_function.params.hx() , level_set_function.params.hy() );
       
        
        
            
        refine_interface_angle( msh, cl, level_set_function, 0 , interface_points , pm , h , positive, multiplicity , angle0 , angle1 );
        
        
        std::cout<<"LIMIT CELL "<<offset(msh,cl)<<" are:"<<std::endl;
        std::cout<<"pt[0] = "<<points(msh,cl)[0]<<" , pt[1] = "<<points(msh,cl)[1]<<" , pt[2] = "<<points(msh,cl)[2]<<" , pt[3] = "<<points(msh,cl)[3]<<std::endl;
       
        for(size_t i_int = 0 ; i_int < interface_points + 1 ; i_int++ )
            std::cout<<"refined points are p = "<<cl.user_data.interface.at(i_int)<<std::endl;
        std::cout<<"--------------------> CELL = "<<offset(msh,cl)<<"<--------------------"<<std::endl;
    }

}



template<typename T, typename Mesh>
bool
pt_in_cell(const Mesh& msh, const point<T,2>& , const typename Mesh::cell_type& );

template<typename T,typename Mesh>
std::vector< point<T,2> >
equidistriduted_nodes(const Mesh& , const typename Mesh::cell_type& , size_t );

template<typename T,typename Mesh>
std::vector< point<T,2> >
equidistriduted_nodes_subcell(const Mesh& ,
          const typename Mesh::cell_type& ,
                              size_t , const std::vector<size_t>& ) ;


template< typename FonctionD , typename Mesh , typename FonctionA >
void
testing_level_set(const Mesh msh , const FonctionD& ,  const FonctionA& );

template< typename FonctionD , typename Mesh >
void
test_new_method(const Mesh , const FonctionD&  , const typename Mesh::cell_type & );


// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh , typename FonctionA >
void
gradient_checking1(const Mesh msh , const FonctionD& level_set_disc , const FonctionA& level_set_anal , const typename Mesh::cell_type & cl)
{
    //typedef typename Mesh::point_type       point_type;
   
    double derD1x , derD1y , derAx , derAy , derD2x , derD2y ;
    Eigen::Matrix<double,2,1> derD1 ,derD2 , derA;
    point<double,2> node;

    
    auto pts = points(msh,cl);
    for(auto & node:pts)
    {
        derD1 = level_set_disc.gradient(node);
        derD2 = level_set_disc.gradient(node,msh,cl);
        derA = level_set_anal.gradient(node);
    
            derD1x = derD1(0);
            derD1y = derD1(1);
            derD2x = derD2(0);
            derD2y = derD2(1);
            derAx = derA(0);
            derAy = derA(1);

        
        /*
            if((derD1x-derD2x)>1e-2)
            {
            std::cout<<"Differnce between two x-evaluation system "<<(derD1x-derD2x)<<std::endl;
            }
        */
        
        if((derAx-derD2x)>1e-2)
        {
            std::cout<<"Differnce between analytic and NEW X-evaluation system "<<(derAx-derD2x)<<std::endl;
        }
   
        if((derAx-derD1x)>1e-2)
        {
            std::cout<<"Differnce between analytic and OLD X-evaluation system "<<(derAx-derD1x)<<std::endl;
        }
        
        
        /*
        if((derD1y-derD2y)>1e-2)
        {
            std::cout<<"Differnce between two y-evaluation system "<<(derD1y-derD2y)<<std::endl;
        }
         */
        
        
        if((derAy-derD2y)>1e-2)
        {
            std::cout<<"Differnce between analytic and NEW Y-evaluation system "<<(derAy-derD2y)<<std::endl;
        }
        
        if((derAy-derD1y)>1e-2)
        {
            std::cout<<"Differnce between analytic and OLD Y-evaluation system "<<(derAy-derD1y)<<std::endl;
        }
             
            
    }
        


}

// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh , typename FonctionA >
void
gradient_checking(const Mesh msh , const FonctionD& level_set_disc , const FonctionA& level_set_anal , const typename Mesh::cell_type & cl)
{
    //typedef typename Mesh::point_type       point_type;
   
    double derD1x , derD1y , derAx , derAy ; // , derD2x , derD2y ;
    Eigen::Matrix<double,2,1> derD1 ,derD2 , derA;
    point<double,2> node;

    auto pts = points(msh,cl);
    for(auto & node:pts)
    {
        derD1 = level_set_disc.gradient(node);
       // derD2 = level_set_disc.gradient(node,msh,cl);
        derA = level_set_anal.gradient(node);
    
            derD1x = derD1(0);
            derD1y = derD1(1);
         //   derD2x = derD2(0);
         //   derD2y = derD2(1);
            derAx = derA(0);
            derAy = derA(1);

        
        /*
            if((derD1x-derD2x)>1e-2)
            {
            std::cout<<"Differnce between two x-evaluation system "<<(derD1x-derD2x)<<std::endl;
            }
        */
        /*
        if((derAx-derD2x)>1e-2)
        {
            std::cout<<"Differnce between analytic and NEW X-evaluation system "<<(derAx-derD2x)<<std::endl;
        }
        */
        if((derAx-derD1x)>1e-2)
        {
            std::cout<<"Differnce between analytic and OLD X-evaluation system "<<(derAx-derD1x)<<std::endl;
        }
        
        
        /*
        if((derD1y-derD2y)>1e-2)
        {
            std::cout<<"Differnce between two y-evaluation system "<<(derD1y-derD2y)<<std::endl;
        }
         */
        
        /*
        if((derAy-derD2y)>1e-2)
        {
            std::cout<<"Differnce between analytic and NEW Y-evaluation system "<<(derAy-derD2y)<<std::endl;
        }
        */
        if((derAy-derD1y)>1e-2)
        {
            std::cout<<"Differnce between analytic and OLD Y-evaluation system "<<(derAy-derD1y)<<std::endl;
        }
             
            
    }
        


}




template< typename FonctionD , typename Mesh , typename FonctionA >
void
testing_velocity(const Mesh msh , const FonctionD& vel_disc , const FonctionA& vel_anal)
{
    //typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    
    Eigen::Matrix<double,2,1> valueA;
    point<double,2> node;
    size_t N, M;
    N = 40;
    M = 40;
    auto test_discx  = std::make_shared< gnuplot_output_object<double> >("vel_discX.dat");
    auto test_analx = std::make_shared< gnuplot_output_object<double> >("vel_analX.dat");
    auto test_discy  = std::make_shared< gnuplot_output_object<double> >("vel_discY.dat");
    auto test_analy = std::make_shared< gnuplot_output_object<double> >("vel_analY.dat");
    
    for(auto& cl:msh.cells)
    {
        auto pts = points(msh,cl);
        for(auto&pt : pts)
        {
            auto valueD = vel_disc(pt,msh,cl);
            valueA = vel_anal(pt);
            
            test_discx->add_data(pt,valueD.first);
            test_discy->add_data(pt,valueD.second);
            test_analx->add_data(pt,valueA(0));
            test_analy->add_data(pt,valueA(1));
            
            
        }
    }
   
    postoutput1.add_object(test_discx);
    postoutput1.add_object(test_analx);
    postoutput1.add_object(test_discy);
    postoutput1.add_object(test_analy);

    
    postoutput1.write();
    
}

template< typename FonctionD , typename Mesh  >
void
testing_level_set(const Mesh& msh , const FonctionD& level_set_disc )
{
    typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    double valueD , derDx , derDy ;
    Eigen::Matrix<double,2,1> derD ;
    point<double,2> node;
    size_t N, M;
    std::cout<<"In testing_level_set I need 80x80 points to see the interface. Now is 40x40, faster."<<std::endl;
    N = 4; //80 points to see also the interface!!!
    M = 4; //80 points to see also the interface!!!
    auto test_disc  = std::make_shared< gnuplot_output_object<double> >("testing_interface_disc.dat");
    
    
    auto test_disc_gradX  = std::make_shared< gnuplot_output_object<double> >("testing_der_discX.dat");
    
    auto test_disc_gradY  = std::make_shared< gnuplot_output_object<double> >("testing_der_discY.dat");
    
    auto test_profile_disc  = std::make_shared< gnuplot_output_object<double> >("test_profile_disc.dat");
    
    for(auto& cl : msh.cells)
    {
        auto pts = points(msh, cl);
        auto pt0_x = pts[0].x();
        auto pt1_x = pts[1].x();
        auto pt0_y = pts[0].y();
        auto pt1_y = pts[3].y();
        for(size_t i = 0 ; i<= N ; i++ )
        {
            for (size_t j = 0 ; j<= M ; j++ )
            {
                double px = pt0_x + i*( (pt1_x - pt0_x)/N);
                double py = pt0_y + j*( (pt1_y - pt0_y)/M);
                node = point_type(px,py);
            /*
            if (std::abs( level_set_disc(node)) <1e-2 ) {
                valueD = 1;
            }
            else
                valueD = 0;
            */
                valueD = level_set_disc(node,msh,cl);
            /*
            if (std::abs( level_set_anal(node)) <1e-2 ) {
                valueA = 1;
            }
            else
                valueA = 0;
            */
           
            
                derD = level_set_disc.gradient(node,msh,cl);
            
        
            
                if( py == 0.5 )
                {
                    test_profile_disc->add_data(node,valueD);
              
                }
            
                derDx = derD(0);
                derDy = derD(1);
           
            
                test_disc->add_data(node,valueD);
           
                test_disc_gradX->add_data(node,derDx);
                test_disc_gradY->add_data(node,derDy);
        
            
            }
        
        }
    }
    postoutput1.add_object(test_disc);
  
    
    postoutput1.add_object(test_disc_gradX);
   
    postoutput1.add_object(test_disc_gradY);
  
    postoutput1.add_object(test_profile_disc);
   
    
    postoutput1.write();
    
}



template< typename FonctionD , typename Mesh , typename FonctionA >
void
testing_level_set(const Mesh msh , const FonctionD& level_set_disc , const FonctionA& level_set_anal)
{
    typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    double valueD , valueA , derDx , derDy , derAx , derAy;
    Eigen::Matrix<double,2,1> derD , derA;
    point<double,2> node;
    size_t N, M;
    std::cout<<"In testing_level_set I need 80x80 points to see the interface. Now is 40x40, faster."<<std::endl;
    N = 40; //80 points to see also the interface!!!
    M = 40; //80 points to see also the interface!!!
    auto test_disc  = std::make_shared< gnuplot_output_object<double> >("testing_interface_disc.dat");
    auto test_anal = std::make_shared< gnuplot_output_object<double> >("testing_interface_anal.dat");
    
    auto test_disc_gradX  = std::make_shared< gnuplot_output_object<double> >("testing_der_discX.dat");
    auto test_anal_gradX = std::make_shared< gnuplot_output_object<double> >("testing_der_analX.dat");
    
    auto test_disc_gradY  = std::make_shared< gnuplot_output_object<double> >("testing_der_discY.dat");
    auto test_anal_gradY = std::make_shared< gnuplot_output_object<double> >("testing_der_analY.dat");
    
    auto test_profile_disc  = std::make_shared< gnuplot_output_object<double> >("test_profile_disc.dat");
    auto test_profile_anal = std::make_shared< gnuplot_output_object<double> >("test_profile_anal.dat");
    
    for(size_t i = 0 ; i<= N ; i++ )
    {
        for (size_t j = 0 ; j<= M ; j++ )
        {
            double px = i*(1.0/N);
            double py = j*(1.0/M);
            node = point_type(px,py);
            /*
            if (std::abs( level_set_disc(node)) <1e-2 ) {
                valueD = 1;
            }
            else
                valueD = 0;
            */
            valueD = level_set_disc(node);
            /*
            if (std::abs( level_set_anal(node)) <1e-2 ) {
                valueA = 1;
            }
            else
                valueA = 0;
            */
            valueA = level_set_anal(node);
            
            derD = level_set_disc.gradient(node);
            derA = level_set_anal.gradient(node);
        
            
            if( py == 0.5 )
            {
                test_profile_disc->add_data(node,valueD);
                test_profile_anal->add_data(node,valueA);
            }
            
            derDx = derD(0);
            derDy = derD(1);
            derAx = derA(0);
            derAy = derA(1);
            
            test_disc->add_data(node,valueD);
            test_anal->add_data(node,valueA);
            
            
            test_disc_gradX->add_data(node,derDx);
            test_anal_gradX->add_data(node,derAx);
            test_disc_gradY->add_data(node,derDy);
            test_anal_gradY->add_data(node,derAy);
            
        }
        
    }
    postoutput1.add_object(test_disc);
    postoutput1.add_object(test_anal);
    
    postoutput1.add_object(test_disc_gradX);
    postoutput1.add_object(test_anal_gradX);
    postoutput1.add_object(test_disc_gradY);
    postoutput1.add_object(test_anal_gradY);
    
    postoutput1.add_object(test_profile_disc);
    postoutput1.add_object(test_profile_anal);
    
    postoutput1.write();
    
}

// Qualitative testing of the discrete level set function wrt the analytical one


template< typename FonctionD , typename Mesh , typename T >
void
testing_level_set_time(const Mesh msh , const FonctionD& level_set_disc , T time)
{
    
    typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    
    point<double,2> node;
    size_t N ; //, M;
    std::cout<<"In testing_level_set_time I use 80 points to see the interface profile y = 0.5 interface."<<std::endl;
    N = 80; //80 points to see also the interface!!!
    
 
    std::string test_profile_disc = "‎⁨profile_" + std::to_string(time) + ".dat";
    auto test_profile  = std::make_shared< gnuplot_output_object<double> >(test_profile_disc);
   
    
    for(size_t i = 0 ; i<= N ; i++ )
    {
            double px = i*(1.0/N);
            double py = 0.5;
            node = point_type(px,py);
            
            T valueD = level_set_disc(node);
            test_profile->add_data(node,valueD);
                
            
           
        
        
    }
    
    postoutput1.add_object(test_profile);
   
    
    postoutput1.write();
    
}


template< typename FonctionD , typename Mesh  >
void
testing_level_set2_bis(const Mesh msh , const FonctionD& level_set_disc )
{
    typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    double valueD ; // , derDx , derDy ;
    Eigen::Matrix<double,2,1> derD ;
    point<double,2> node;
    size_t N, M;
    std::cout<<"In testing_level_set2 I need 80x80 points to see the interface. Now is 40x40, faster. No analytic solution."<<std::endl;
    N = 40; //80 points to see also the interface!!!
    M = 40; //80 points to see also the interface!!!
    double valueD_gamma ;
    auto test_disc  = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_disc_bis.dat");
    
    auto test_gamma_disc = std::make_shared< gnuplot_output_object<double> >("test_gamma_disc_bis.dat");
    
    auto test_profile_disc = std::make_shared< gnuplot_output_object<double> >("test_profile_disc_fin_bis.dat");
   
    double iso_val_interface = level_set_disc.iso_val_interface ;
    
    for(size_t i = 0 ; i<= N ; i++ )
    {
        for (size_t j = 0 ; j<= M ; j++ )
        {
            double px = i*(1.0/N);
            double py = j*(1.0/M);
            node = point_type(px,py);
            
            if (std::abs( level_set_disc(node)-iso_val_interface) <1*1e-3 ) {
                valueD_gamma = 1;
            }
            else
                valueD_gamma = 0;
            
            valueD = level_set_disc(node);
            
            
            
            
            
            
            if( py == 0.5 )
            {
                test_profile_disc->add_data(node,valueD);
               
            }
            
            
            test_disc->add_data(node,valueD);
            
            test_gamma_disc->add_data(node,valueD_gamma);
           
           
        }
        
    }
    postoutput1.add_object(test_disc);

    postoutput1.add_object(test_gamma_disc);
  
    postoutput1.add_object(test_profile_disc);
    
    postoutput1.write();
    
}
template< typename FonctionD , typename Mesh  >
void
testing_level_set2(const Mesh& msh , const FonctionD& level_set_disc )
{
    typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    double valueD  ; // , derDx , derDy ;
    Eigen::Matrix<double,2,1> derD ;
    point<double,2> node;
    size_t N, M;
    std::cout<<"In testing_level_set2 I need 80x80 points to see the interface. Now is 40x40, faster. No analytic solution."<<std::endl;
    N = 4; //80 points to see also the interface!!!
    M = 4; //80 points to see also the interface!!!
    double valueD_gamma ;
    auto test_disc  = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_disc.dat");
    
    //auto test_gamma_disc = std::make_shared< gnuplot_output_object<double> >("test_gamma_disc.dat");
    
    auto test_profile_disc = std::make_shared< gnuplot_output_object<double> >("test_profile_disc_fin.dat");
   
    double iso_val_interface = level_set_disc.iso_val_interface ;
    for(auto& cl : msh.cells)
    {
        auto pts = points(msh, cl);
        auto pt0_x = pts[0].x();
        auto pt1_x = pts[1].x();
        auto pt0_y = pts[0].y();
        auto pt1_y = pts[3].y();
        for(size_t i = 0 ; i<= N ; i++ )
        {
            double px = pt0_x + i*( (pt1_x - pt0_x)/N);
            for (size_t j = 0 ; j<= M ; j++ )
            {
                double py = pt0_y + j*( (pt1_y - pt0_y)/M);
                node = point_type(px,py);
                /*
                if (std::abs( level_set_disc(node)-iso_val_interface) <1*1e-3 ) {
                    valueD_gamma = 1;
                }
                else
                    valueD_gamma = 0;
                */
                valueD = level_set_disc(node,msh,cl);
            
    
            
                test_disc->add_data(node,valueD);
            
            
           
            }
            double py = 0.5;
            node = point_type(px,py);
            test_profile_disc->add_data(node,valueD);
        }
        
    }
    postoutput1.add_object(test_disc);
  
    postoutput1.add_object(test_profile_disc);
    
    postoutput1.write();
    
}

template< typename FonctionD , typename Mesh  >
void
testing_level_set2(const Mesh msh , const FonctionD& level_set_disc ,const FonctionD& level_set_fin)
{
    typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    double valueD , valueA ; //, derDx , derDy , derAx , derAy;
    Eigen::Matrix<double,2,1> derD , derA;
    point<double,2> node;
    size_t N, M;
    std::cout<<"In testing_level_set2 I need 80x80 points to see the interface. Now is 40x40, faster."<<std::endl;
    N = 40; //80 points to see also the interface!!!
    M = 40; //80 points to see also the interface!!!
    double valueA_gamma , valueD_gamma ;
    auto test_disc  = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_disc.dat");
    auto test_anal = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_anal.dat");
    auto test_gamma_disc = std::make_shared< gnuplot_output_object<double> >("test_gamma_disc.dat");
     auto test_gamma_anal = std::make_shared< gnuplot_output_object<double> >("test_gamma_anal.dat");
    
    auto test_profile_disc = std::make_shared< gnuplot_output_object<double> >("test_profile_disc_fin.dat");
    auto test_profile_anal = std::make_shared< gnuplot_output_object<double> >("test_profile_anal_fin.dat");
    /*
    auto test_disc_gradX  = std::make_shared< gnuplot_output_object<double> >("testing_der_discX.dat");
    auto test_anal_gradX = std::make_shared< gnuplot_output_object<double> >("testing_der_analX.dat");
    
    auto test_disc_gradY  = std::make_shared< gnuplot_output_object<double> >("testing_der_discY.dat");
    auto test_anal_gradY = std::make_shared< gnuplot_output_object<double> >("testing_der_analY.dat");
    */
    for(size_t i = 0 ; i<= N ; i++ )
    {
        for (size_t j = 0 ; j<= M ; j++ )
        {
            double px = i*(1.0/N);
            double py = j*(1.0/M);
            node = point_type(px,py);
            
            if (std::abs( level_set_disc(node)) <1*1e-3 ) {
                valueD_gamma = 1;
            }
            else
                valueD_gamma = 0;
            
            valueD = level_set_disc(node);
            
            if (std::abs( level_set_fin(node)) <1*1e-3 ) {
                valueA_gamma = 1;
            }
            else
                valueA_gamma = 0;
            
            valueA = level_set_fin(node);
            
            
            if( py == 0.5 )
            {
                test_profile_disc->add_data(node,valueD);
                test_profile_anal->add_data(node,valueA);
            }
            
            /*
            derD = level_set_disc.gradient(node);
            derA = level_set_fin.gradient(node);
        
            
            derDx = derD(0);
            derDy = derD(1);
            derAx = derA(0);
            derAy = derA(1);
            */
            test_disc->add_data(node,valueD);
            test_anal->add_data(node,valueA);
            test_gamma_disc->add_data(node,valueD_gamma);
            test_gamma_anal->add_data(node,valueA_gamma);
            /*
            test_disc_gradX->add_data(node,derDx);
            test_anal_gradX->add_data(node,derAx);
            test_disc_gradY->add_data(node,derDy);
            test_anal_gradY->add_data(node,derAy);
            */
        }
        
    }
    postoutput1.add_object(test_disc);
    postoutput1.add_object(test_anal);
    postoutput1.add_object(test_gamma_disc);
    postoutput1.add_object(test_gamma_anal);
    postoutput1.add_object(test_profile_disc);
    postoutput1.add_object(test_profile_anal);
    /*
    postoutput1.add_object(test_disc_gradX);
    postoutput1.add_object(test_anal_gradX);
    postoutput1.add_object(test_disc_gradY);
    postoutput1.add_object(test_anal_gradY);
    */
    postoutput1.write();
    
}

// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh  >
void
testing_level_set_mesh2(const Mesh msh , const FonctionD& level_set_disc ,const FonctionD& level_set_fin)
{
    typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    double valueD , valueA ; //, derDx , derDy , derAx , derAy;
    Eigen::Matrix<double,2,1> derD , derA;
    point<double,2> node;
    size_t N, M;
    N = 40;
    M = 40;
    auto test_disc  = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_disc_mesh2.dat");
    auto test_anal = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_anal_mesh2.dat");
    /*
    auto test_disc_gradX  = std::make_shared< gnuplot_output_object<double> >("testing_der_discX.dat");
    auto test_anal_gradX = std::make_shared< gnuplot_output_object<double> >("testing_der_analX.dat");
    
    auto test_disc_gradY  = std::make_shared< gnuplot_output_object<double> >("testing_der_discY.dat");
    auto test_anal_gradY = std::make_shared< gnuplot_output_object<double> >("testing_der_analY.dat");
    */
    for(size_t i = 0 ; i<= N ; i++ )
    {
        for (size_t j = 0 ; j<= M ; j++ )
        {
            double px = i*(1.0/N);
            double py = j*(1.0/M);
            node = point_type(px,py);
            /*
            if (std::abs( level_set_disc(node)) <1e-2 ) {
                valueD = 1;
            }
            else
                valueD = 0;
            */
            valueD = level_set_disc(node);
            /*
            if (std::abs( level_set_anal(node)) <1e-2 ) {
                valueA = 1;
            }
            else
                valueA = 0;
            */
            valueA = level_set_fin(node);
            /*
            derD = level_set_disc.gradient(node);
            derA = level_set_fin.gradient(node);
        
            
            derDx = derD(0);
            derDy = derD(1);
            derAx = derA(0);
            derAy = derA(1);
            */
            test_disc->add_data(node,valueD);
            test_anal->add_data(node,valueA);
            /*
            test_disc_gradX->add_data(node,derDx);
            test_anal_gradX->add_data(node,derAx);
            test_disc_gradY->add_data(node,derDy);
            test_anal_gradY->add_data(node,derAy);
            */
        }
        
    }
    postoutput1.add_object(test_disc);
    postoutput1.add_object(test_anal);
    /*
    postoutput1.add_object(test_disc_gradX);
    postoutput1.add_object(test_anal_gradX);
    postoutput1.add_object(test_disc_gradY);
    postoutput1.add_object(test_anal_gradY);
    */
    postoutput1.write();
    
}

// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh  >
void
testing_level_set_mesh1(const Mesh msh , const FonctionD& level_set_disc ,const FonctionD& level_set_fin)
{
    typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    double valueD , valueA ; //, derDx , derDy , derAx , derAy;
    Eigen::Matrix<double,2,1> derD , derA;
    point<double,2> node;
    size_t N, M;
    N = 40;
    M = 40;
    auto test_disc  = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_disc_mesh1.dat");
    auto test_anal = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_anal_mesh1.dat");
    /*
    auto test_disc_gradX  = std::make_shared< gnuplot_output_object<double> >("testing_der_discX.dat");
    auto test_anal_gradX = std::make_shared< gnuplot_output_object<double> >("testing_der_analX.dat");
    
    auto test_disc_gradY  = std::make_shared< gnuplot_output_object<double> >("testing_der_discY.dat");
    auto test_anal_gradY = std::make_shared< gnuplot_output_object<double> >("testing_der_analY.dat");
    */
    for(size_t i = 0 ; i<= N ; i++ )
    {
        for (size_t j = 0 ; j<= M ; j++ )
        {
            double px = i*(1.0/N);
            double py = j*(1.0/M);
            node = point_type(px,py);
            /*
            if (std::abs( level_set_disc(node)) <1e-2 ) {
                valueD = 1;
            }
            else
                valueD = 0;
            */
            valueD = level_set_disc(node);
            /*
            if (std::abs( level_set_anal(node)) <1e-2 ) {
                valueA = 1;
            }
            else
                valueA = 0;
            */
            valueA = level_set_fin(node);
            /*
            derD = level_set_disc.gradient(node);
            derA = level_set_fin.gradient(node);
        
            
            derDx = derD(0);
            derDy = derD(1);
            derAx = derA(0);
            derAy = derA(1);
            */
            test_disc->add_data(node,valueD);
            test_anal->add_data(node,valueA);
            /*
            test_disc_gradX->add_data(node,derDx);
            test_anal_gradX->add_data(node,derAx);
            test_disc_gradY->add_data(node,derDy);
            test_anal_gradY->add_data(node,derAy);
            */
        }
        
    }
    postoutput1.add_object(test_disc);
    postoutput1.add_object(test_anal);
    /*
    postoutput1.add_object(test_disc_gradX);
    postoutput1.add_object(test_anal_gradX);
    postoutput1.add_object(test_disc_gradY);
    postoutput1.add_object(test_anal_gradY);
    */
    postoutput1.write();
    
}

// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh  >
void
testing_level_set_mesh0(const Mesh msh , const FonctionD& level_set_disc ,const FonctionD& level_set_fin)
{
    typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    double valueD , valueA ; //, derDx , derDy , derAx , derAy;
    Eigen::Matrix<double,2,1> derD , derA;
    point<double,2> node;
    size_t N, M;
    N = 40;
    M = 40;
    auto test_disc  = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_disc_mesh0.dat");
    auto test_anal = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_anal_mesh0.dat");
    /*
    auto test_disc_gradX  = std::make_shared< gnuplot_output_object<double> >("testing_der_discX.dat");
    auto test_anal_gradX = std::make_shared< gnuplot_output_object<double> >("testing_der_analX.dat");
    
    auto test_disc_gradY  = std::make_shared< gnuplot_output_object<double> >("testing_der_discY.dat");
    auto test_anal_gradY = std::make_shared< gnuplot_output_object<double> >("testing_der_analY.dat");
    */
    for(size_t i = 0 ; i<= N ; i++ )
    {
        for (size_t j = 0 ; j<= M ; j++ )
        {
            double px = i*(1.0/N);
            double py = j*(1.0/M);
            node = point_type(px,py);
            /*
            if (std::abs( level_set_disc(node)) <1e-2 ) {
                valueD = 1;
            }
            else
                valueD = 0;
            */
            valueD = level_set_disc(node);
            /*
            if (std::abs( level_set_anal(node)) <1e-2 ) {
                valueA = 1;
            }
            else
                valueA = 0;
            */
            valueA = level_set_fin(node);
            /*
            derD = level_set_disc.gradient(node);
            derA = level_set_fin.gradient(node);
        
            
            derDx = derD(0);
            derDy = derD(1);
            derAx = derA(0);
            derAy = derA(1);
            */
            test_disc->add_data(node,valueD);
            test_anal->add_data(node,valueA);
            /*
            test_disc_gradX->add_data(node,derDx);
            test_anal_gradX->add_data(node,derAx);
            test_disc_gradY->add_data(node,derDy);
            test_anal_gradY->add_data(node,derAy);
            */
        }
        
    }
    postoutput1.add_object(test_disc);
    postoutput1.add_object(test_anal);
    /*
    postoutput1.add_object(test_disc_gradX);
    postoutput1.add_object(test_anal_gradX);
    postoutput1.add_object(test_disc_gradY);
    postoutput1.add_object(test_anal_gradY);
    */
    postoutput1.write();
    
}
// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh , typename FonctionA >
void
test_new_method(const Mesh msh , const FonctionD& level_set_disc , const FonctionA& level_set_anal , const typename Mesh::cell_type & cl)
{
    //typedef typename Mesh::point_type       point_type;
    double valueD1 , valueD2 ,valueA ;
    
    auto pts =points(msh,cl);
    
    
    
    
    timecounter tc1;
    tc1.tic();
    for(auto& pt : pts)
    {
        valueD1 = level_set_disc(pt,msh,cl);
       // valueA = level_set_anal(pt);
       // valueD2 = level_set_disc(pt);
       // std::cout<<"Differnce between two evaluation system "<<(valueD1-valueD2)<<std::endl;
      //  std::cout<<"Im in test_new_method";
    }
    
   
    
    
    tc1.toc();
    std::cout << bold << yellow << "Time for the new evaluation: " << tc1 << " seconds" << reset << std::endl;
    
   tc1.tic();
       for(auto& pt : pts)
       {
           valueD2 = level_set_disc(pt);
         //  std::cout<<"Im in test_new_method";
       }
    tc1.toc();
    std::cout << bold << yellow << "Time for the old evaluation: " << tc1 << " seconds" << reset << std::endl;
    
    tc1.tic();
         for(auto& pt : pts)
         {
             valueA = level_set_anal(pt);
         }
      tc1.toc();
      std::cout << bold << yellow << "Time for the analytic evaluation: " << tc1 << " seconds" << reset << std::endl;
    
    
}

// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh >
void
time_NEWdiscrete_testing(const Mesh msh , const FonctionD& level_set_disc , const typename Mesh::cell_type & cl)
{
    //typedef typename Mesh::point_type       point_type;
    double valueD1 ; //, valueD2 ,valueA ;
    
    auto pts =points(msh,cl);
    
  
    for(auto& pt : pts)
    {
        valueD1 = level_set_disc(pt,msh,cl);
       
    }
    
    
}

// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh >
void
time_OLDdiscrete_testing(const Mesh msh , const FonctionD& level_set_disc , const typename Mesh::cell_type & cl)
{
    //typedef typename Mesh::point_type       point_type;
    double valueD1 ; //, valueA ; // valueD2 ,
    
    auto pts =points(msh,cl);
    
  
    for(auto& pt : pts)
    {
        valueD1 = level_set_disc(pt);
       
    }
    
    
}

template< typename FonctionA , typename Mesh >
void
time_analytic_testing(const Mesh msh , const FonctionA& level_set_anal , const typename Mesh::cell_type & cl)
{
    //typedef typename Mesh::point_type       point_type;
    double valueA ; // valueD1 , valueD2 ,
    
    auto pts =points(msh,cl);
    
  
    for(auto& pt : pts)
    {
        valueA = level_set_anal(pt);
       
    }
    
    
}


template< typename FonctionD , typename Mesh >
void
time_face_testing(const Mesh msh , const FonctionD& level_set_disc , const typename Mesh::face_type & fc)
{
    
    //typedef typename Mesh::point_type       point_type;
    double valueD1 ;// , valueD2 ,valueA ;
       
    auto pts =points(msh,fc);
       
     
    for(auto& pt : pts)
    {
        valueD1 = level_set_disc(pt,msh,fc);
          
    }
       
    
    
}

template< typename FonctionA , typename Mesh >
void
time_faceANALITIC_testing(const Mesh msh , const FonctionA& level_set_anal , const typename Mesh::face_type & fc)
{
    
    //typedef typename Mesh::point_type       point_type;
    double valueA ; // valueD1 , valueD2 ,
       
    auto pts =points(msh,fc);
       
     
    for(auto& pt : pts)
    {
        valueA = level_set_anal(pt);
          
    }
       
    
    
}



// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh , typename FonctionA >
void
test_new_method(const Mesh msh , const FonctionD& level_set_disc , const FonctionA& level_set_anal , const typename Mesh::cell_type & cl , const typename Mesh::face_type & fc)
{
    //typedef typename Mesh::point_type       point_type;
    double valueD1  ,valueA , valueD3; //, valueD2
    
    auto pts =points(msh,fc);
    
    
    
    
    timecounter tc1;
    tc1.tic();
    for(auto& pt : pts)
    {
        valueD1 = level_set_disc(pt,msh,fc);
        valueA = level_set_anal(pt);
        //valueD2 = level_set_disc(pt,msh,cl);
        valueD3 = level_set_disc(pt);
        
        std::cout<<"Differnce between FACE and OLD evaluation system "<<(valueD1-valueD3)<<std::endl;
        std::cout<<"Error between analytic and face evaluation "<<(valueD1-valueA)<<std::endl;
    }
    

    
}





///////////////////////////////////////

template<typename Mesh, typename testType, typename meth , typename Fonction , typename Velocity , typename RealType >
stokes_test_info<typename Mesh::coordinate_type>
run_cuthho_interface_velocity_analytic(const Mesh& msh, size_t degree, meth method, testType test_case , Fonction & level_set_function , Velocity & velocity , bool sym_grad , RealType radius )
{
    //using RealType = typename Mesh::coordinate_type;
    
    //auto level_set_function = test_case.level_set_;

    auto iso_val_interface = level_set_function.iso_val_interface ;
    std::cout<<"Interface isovalue = "<<iso_val_interface<<std::endl;
    auto bcs_vel = test_case.bcs_vel;
    
    struct params<RealType> parms = test_case.parms;

    timecounter tc;

    bool sc = true; // static condensation


    // ************** ASSEMBLE PROBLEM **************
    hho_degree_info hdi(degree+1, degree);

    tc.tic();
    timecounter tc_bis2 ;
    tc_bis2.tic();
    auto assembler = make_stokes_interface_assembler(msh, bcs_vel, hdi);
    tc_bis2.toc();
    std::cout<<"-------> TIME assembler , time = "<<tc_bis2<<std::endl;
    tc_bis2.tic();
    auto assembler_sc = make_stokes_interface_condensed_assembler(msh, bcs_vel, hdi);
    tc_bis2.toc();
    std::cout<<"-------> TIME assembler_sc , time = "<<tc_bis2<<std::endl;
    for (auto& cl : msh.cells)
    {
        // ADD BY STE
        timecounter tc_bis ;
        tc_bis.tic();
        level_set_function.cell_assignment(cl);
        //auto test_case_cell = make_test_case_stokes_2(msh, level_set_function);
        auto prm = params<RealType>();
        prm.kappa_1 = 1.0;
        prm.kappa_2 = 1.0;
        auto test_case_cell = make_test_case_eshelby_analytic(msh, level_set_function, prm, sym_grad,radius);
        
        //auto test_case_cell = make_test_case_eshelby(msh, level_set_function, prm, sym_grad);
        //This stuff before it was before the loop in msh.cells and test_case and not test_case_cell
        auto rhs_fun = test_case_cell.rhs_fun;
        auto sol_vel = test_case_cell.sol_vel;
        auto sol_p = test_case_cell.sol_p;
        auto vel_grad = test_case_cell.vel_grad;
        //auto bcs_vel = test_case.bcs_vel;
        auto neumann_jump = test_case_cell.neumann_jump;
        assembler_sc.set_dir_func( bcs_vel);
        tc_bis.toc();
        std::cout<<"-------> TIME STOKES PARTE 0 , time = "<<tc_bis<<std::endl;
        tc_bis.tic();
        auto contrib = method.make_contrib(msh, cl, test_case_cell, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;
        tc_bis.toc();
        std::cout<<"-------> TIME STOKES PARTE 1 , time = "<<tc_bis<<std::endl;
         tc_bis.tic();
        if( sc )
            assembler_sc.assemble(msh, cl, lc, f);
        else
            assembler.assemble(msh, cl, lc, f);
        tc_bis.toc();
        std::cout<<"-------> TIME STOKES PARTE 2 , time = "<<tc_bis<<std::endl;
    }

    
    tc_bis2.tic();
    if( sc )
        assembler_sc.finalize();
    else
        assembler.finalize();

    tc_bis2.toc();
    std::cout<<"-------> TIME STOKES PARTE 3 , time = "<<tc_bis2<<std::endl;
    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

    if( sc )
        std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
    else
        std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;

    // ************** SOLVE **************
    tc.tic();
#if 1
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
#if 0
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
#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    // ************** POSTPROCESS **************


    postprocess_output<RealType>  postoutput;

    auto uT1_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT1.dat");
    auto uT2_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT2.dat");
    auto p_gp    = std::make_shared< gnuplot_output_object<RealType> >("interface_p.dat");

    tc.tic();
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
    RealType    L2_pressure_error = 0.0;
    
    std::cout<<"sono qua 0.0"<<std::endl;
    size_t i_global = 0 ; // ADD BY STE
    for (auto& cl : msh.cells)
    {
        
        
        vector_cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        
        
        cell_basis<cuthho_poly_mesh<RealType>, RealType> pb(msh, cl, hdi.face_degree());
        auto cbs = cb.size();
        auto pbs = pb.size();

        
        // ADD BY STE
        level_set_function.cell_assignment(cl);
        //auto test_case_cell = make_test_case_stokes_2(msh, level_set_function);
        
        auto prm = params<RealType>();
        prm.kappa_1 = 1.0;
        prm.kappa_2 = 1.0;
        auto test_case_cell = make_test_case_eshelby_analytic(msh, level_set_function, prm, sym_grad, radius);
        //auto test_case_cell = make_test_case_eshelby(msh, level_set_function,  prm , sym_grad);
        //This stuff before it was before the loop in msh.cells and test_case and not test_case_cell
        auto rhs_fun = test_case_cell.rhs_fun;
        auto sol_vel = test_case_cell.sol_vel;
        auto sol_p = test_case_cell.sol_p;
        auto vel_grad = test_case_cell.vel_grad;
        //auto bcs_vel = test_case.bcs_vel;
        auto neumann_jump = test_case_cell.neumann_jump;
        assembler_sc.set_dir_func( bcs_vel);
        
        
        Matrix<RealType, Dynamic, 1> vel_locdata_n, vel_locdata_p, vel_locdata;
        Matrix<RealType, Dynamic, 1> P_locdata_n, P_locdata_p, P_locdata;
        Matrix<RealType, Dynamic, 1> vel_cell_dofs_n, vel_cell_dofs_p, vel_cell_dofs;

        if (location(msh, cl) == element_location::ON_INTERFACE)
        {
            if( sc )
            {
                vel_locdata_n = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                vel_locdata_p = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata_n = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                P_locdata_p = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
            {
                vel_locdata_n = assembler.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                vel_locdata_p = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata_n = assembler.take_pressure(msh,cl, sol, element_location::IN_NEGATIVE_SIDE);
                P_locdata_p = assembler.take_pressure(msh,cl, sol, element_location::IN_POSITIVE_SIDE);
            }

            vel_cell_dofs_n = vel_locdata_n.head(cbs);
            vel_cell_dofs_p = vel_locdata_p.head(cbs);


            
            // Uploading velocity field by STE
            //std::cout<<"------------>>> CUT CELL"<<std::endl;
            //std::cout<<"subcells.size() = "<<level_set_function.subcells.size()<<std::endl;
            
            // NOT AGGLO CELL
            if ( level_set_function.subcells.size()<1 )
            {
                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells.size() == 2);
                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                {
                    if( level_set_function(ln_Qk) > iso_val_interface )
                    {
                        auto phi_HHO = cb.eval_basis( ln_Qk );
                        auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                        velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                        //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
                    }
                    else
                    {
                        auto phi_HHO = cb.eval_basis( ln_Qk );
                        auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                        velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                        //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
                        //  velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
                        //  velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                    }
                }
                
            }
            else // AGGLO CELL
            {
                for(size_t i_subcell = 0 ; i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size() ; i_subcell++ )
                {
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
                     std::cout<<"offset_old = "<<offset_old<<std::endl;
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                    size_t i_local = 0;
                    for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                    {
                        if( level_set_function(ln_Qk) > iso_val_interface )
                        {
                            auto phi_HHO = cb.eval_basis( ln_Qk );
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                            velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                            //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
                        }
                        else
                        {
                            auto phi_HHO = cb.eval_basis( ln_Qk );
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                            velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                            //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
                            //  velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
                            //  velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                        }
                    }
                    
                }
            }
                
            
            
            auto qps_n = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                // Compute H1-error //
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += vel_cell_dofs_n(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                H1_error += qp.second * inner_product(grad_diff , grad_diff);


                 // Compute L2-error //
                auto t_phi = cb.eval_basis( qp.first );
                auto v = t_phi.transpose() * vel_cell_dofs_n;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                uT1_gp->add_data( qp.first, v(0) );
                uT2_gp->add_data( qp.first, v(1) );

                // L2 - pressure - error //
                auto p_phi = pb.eval_basis( qp.first );
                RealType p_num = p_phi.dot(P_locdata_n);
                RealType p_diff = test_case_cell.sol_p( qp.first ) - p_num; // era test_case STE
                L2_pressure_error += qp.second * p_diff * p_diff;

                p_gp->add_data( qp.first, p_num );
            }

            auto qps_p = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                // Compute H1-error //
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += vel_cell_dofs_p(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                H1_error += qp.second * inner_product(grad_diff , grad_diff);

                // Compute L2-error //
                auto t_phi = cb.eval_basis( qp.first );
                auto v = t_phi.transpose() * vel_cell_dofs_p;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                uT1_gp->add_data( qp.first, v(0) );
                uT2_gp->add_data( qp.first, v(1) );

                 // L2 - pressure - error //
                auto p_phi = pb.eval_basis( qp.first );
                RealType p_num = p_phi.dot(P_locdata_p);
                RealType p_diff = test_case_cell.sol_p( qp.first ) - p_num; // era test_case STE
                L2_pressure_error += qp.second * p_diff * p_diff;

                p_gp->add_data( qp.first, p_num );
            }
        }
        else
        {
            if( sc )
            {
                vel_locdata = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
            {
                vel_locdata = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata = assembler.take_pressure(msh,cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            vel_cell_dofs = vel_locdata.head(cbs);

            //std::cout<<"------------>>> NOT CUT CELL!!!!!"<<std::endl;
            //std::cout<<"subcells.size() = "<<level_set_function.subcells.size()<<std::endl;
            /*
            for(size_t i_subcell = 0 ; i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size() ; i_subcell++ )
            {
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
                 std::cout<<"offset_old = "<<offset_old<<std::endl;
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                {
                    auto phi_HHO = cb.eval_basis( ln_Qk );
                    auto vel = phi_HHO.transpose() * vel_cell_dofs;
                    velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                    velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                    //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                    i_local++;
                  
                }
                
            }
            */
            
            // NOT AGGLO CELL
            if ( level_set_function.subcells.size()<1 )
            {
                assert(level_set_function.agglo_LS_cl.user_data.offset_subcells.size()==2);
                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                {
                    auto phi_HHO = cb.eval_basis( ln_Qk );
                    auto vel = phi_HHO.transpose() * vel_cell_dofs;
                    velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                    velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                    //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                    i_local++;
                  
                }
                
            }
            else // AGGLO CELL
            {
                for(size_t i_subcell = 0 ; i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size() ; i_subcell++ )
                {
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
                     std::cout<<"offset_old = "<<offset_old<<std::endl;
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                    size_t i_local = 0;
                    for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                    {
                        auto phi_HHO = cb.eval_basis( ln_Qk );
                        auto vel = phi_HHO.transpose() * vel_cell_dofs;
                        velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                        //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
                      
                    }
                    
                }
            }
            
            
       
            
            auto qps = integrate(msh, cl, 2*hdi.cell_degree());
            for (auto& qp : qps)
            {
                // Compute H1-error //
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += vel_cell_dofs(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                H1_error += qp.second * inner_product(grad_diff , grad_diff);

                // Compute L2-error //
                auto t_phi = cb.eval_basis( qp.first );
                auto v = t_phi.transpose() * vel_cell_dofs;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                uT1_gp->add_data( qp.first, v(0) );
                uT2_gp->add_data( qp.first, v(1) );

                // L2 - pressure - error //
                auto p_phi = pb.eval_basis( qp.first );
                RealType p_num = p_phi.dot(P_locdata);
                RealType p_diff = test_case_cell.sol_p( qp.first ) - p_num; // era test_case STE
                L2_pressure_error += qp.second * p_diff * p_diff;

                p_gp->add_data( qp.first, p_num );
            }
        }
        
        i_global++;
    }
    //std::cout<<"velocity.sol_HHO.first"<<'\n'<<velocity.sol_HHO.first<<std::endl;
    //std::cout<<"velocity.sol_HHO.second"<<'\n'<<velocity.sol_HHO.second<<std::endl;

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:               " << std::sqrt(L2_error) << std::endl;
    std::cout << bold << green << "Pressure L2-norm absolute error:      " << std::sqrt(L2_pressure_error) << std::endl;

    postoutput.add_object(uT1_gp);
    postoutput.add_object(uT2_gp);
    postoutput.add_object(p_gp);
    postoutput.write();



    stokes_test_info<RealType> TI;
    TI.H1_vel = std::sqrt(H1_error);
    TI.L2_vel = std::sqrt(L2_error);
    TI.L2_p = std::sqrt(L2_pressure_error);

    if (false)
    {
        /////////////// compute condition number
        SparseMatrix<RealType> Mat;
        // Matrix<RealType, Dynamic, Dynamic> Mat;
        if (sc)
            Mat = assembler_sc.LHS;
        else
            Mat = assembler.LHS;


        RealType sigma_max, sigma_min;

        // Construct matrix operation object using the wrapper class SparseSymMatProd
        Spectra::SparseSymMatProd<RealType> op(Mat);
        // Construct eigen solver object, requesting the largest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::LARGEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > max_eigs(&op, 1, 10);
        max_eigs.init();
        max_eigs.compute();
        if(max_eigs.info() == Spectra::SUCCESSFUL)
            sigma_max = max_eigs.eigenvalues()(0);


        // Construct eigen solver object, requesting the smallest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::SMALLEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > min_eigs(&op, 1, 10);

        min_eigs.init();
        min_eigs.compute();
        if(min_eigs.info() == Spectra::SUCCESSFUL)
            sigma_min = min_eigs.eigenvalues()(0);

        // compute condition number
        RealType cond = sigma_max / sigma_min;
        TI.cond = cond;
        std::cout << "sigma_max = " << sigma_max << "   sigma_min = "
                  << sigma_min << "  cond = " << cond
                  << std::endl;
    }
    else
        TI.cond = 0.0;

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


    return TI;
}

template<typename Mesh, typename testType, typename meth , typename Fonction , typename Velocity>
stokes_test_info<typename Mesh::coordinate_type>
run_cuthho_interface_velocity(const Mesh& msh, size_t degree, meth method, testType test_case , Fonction & level_set_function , Velocity & velocity , bool sym_grad )
{
    using RealType = typename Mesh::coordinate_type;
    
    //auto level_set_function = test_case.level_set_;

    auto iso_val_interface = level_set_function.iso_val_interface ;
    std::cout<<"Interface isovalue = "<<iso_val_interface<<std::endl;
    auto bcs_vel = test_case.bcs_vel;
    
    struct params<RealType> parms = test_case.parms;

    timecounter tc;

    bool sc = true; // static condensation


    // ************** ASSEMBLE PROBLEM **************
    hho_degree_info hdi(degree+1, degree);

    tc.tic();
    timecounter tc_bis2 ;
    tc_bis2.tic();
    auto assembler = make_stokes_interface_assembler(msh, bcs_vel, hdi);
    tc_bis2.toc();
    std::cout<<"-------> TIME assembler , time = "<<tc_bis2<<std::endl;
    tc_bis2.tic();
    auto assembler_sc = make_stokes_interface_condensed_assembler(msh, bcs_vel, hdi);
    tc_bis2.toc();
     std::cout<<"-------> TIME assembler_sc , time = "<<tc_bis2<<std::endl;
    for (auto& cl : msh.cells)
    {
        // ADD BY STE
        timecounter tc_bis ;
        tc_bis.tic();
        level_set_function.cell_assignment(cl);
        //auto test_case_cell = make_test_case_stokes_2(msh, level_set_function);
        auto prm = params<RealType>();
        prm.kappa_1 = 1.0;
        prm.kappa_2 = 1.0;
        auto test_case_cell = make_test_case_eshelby_2(msh, level_set_function, prm, sym_grad);
        tc_bis.toc();
        std::cout<<"-------> TIME STOKES 0 , time = "<<tc_bis<<std::endl;
        tc_bis.tic();
        //auto test_case_cell = make_test_case_eshelby(msh, level_set_function, prm, sym_grad);
        //This stuff before it was before the loop in msh.cells and test_case and not test_case_cell
        //auto rhs_fun = test_case_cell.rhs_fun;
        //auto sol_vel = test_case_cell.sol_vel;
        //auto sol_p = test_case_cell.sol_p;
        //auto vel_grad = test_case_cell.vel_grad;
        ///---> QUESTO NO auto bcs_vel = test_case.bcs_vel;
        //auto neumann_jump = test_case_cell.neumann_jump;
        assembler_sc.set_dir_func( bcs_vel);
        tc_bis.toc();
        std::cout<<"-------> TIME STOKES 1 , time = "<<tc_bis<<std::endl;
        tc_bis.tic();
        auto contrib = method.make_contrib(msh, cl, test_case_cell, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;
        tc_bis.toc();
        std::cout<<"-------> TIME STOKES 2 , time = "<<tc_bis<<std::endl;
        tc_bis.tic();
        if( sc )
            assembler_sc.assemble(msh, cl, lc, f);
        else
            assembler.assemble(msh, cl, lc, f);
        
        tc_bis.toc();
        std::cout<<"-------> TIME STOKES 3 , time = "<<tc_bis<<std::endl;
    }
    tc_bis2.tic();
    
    if( sc )
        assembler_sc.finalize();
    else
        assembler.finalize();

    tc_bis2.toc();
    std::cout<<"------------------> TIME STOKES 4 , time = "<<tc_bis2<<std::endl;
    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

    if( sc )
        std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
    else
        std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;

    // ************** SOLVE **************
    tc.tic();
#if 1
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
#if 0
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
#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    // ************** POSTPROCESS **************


    postprocess_output<RealType>  postoutput;

    auto uT1_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT1.dat");
    auto uT2_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT2.dat");
    auto p_gp    = std::make_shared< gnuplot_output_object<RealType> >("interface_p.dat");

    tc.tic();
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
    RealType    L2_pressure_error = 0.0;
    
    std::cout<<"sono qua 0.0"<<std::endl;
    size_t i_global = 0 ; // ADD BY STE
    for (auto& cl : msh.cells)
    {
        
        
        vector_cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        
        
        cell_basis<cuthho_poly_mesh<RealType>, RealType> pb(msh, cl, hdi.face_degree());
        auto cbs = cb.size();
        auto pbs = pb.size();

        
        // ADD BY STE
        level_set_function.cell_assignment(cl);
        //auto test_case_cell = make_test_case_stokes_2(msh, level_set_function);
        
        auto prm = params<RealType>();
        prm.kappa_1 = 1.0;
        prm.kappa_2 = 1.0;
        auto test_case_cell = make_test_case_eshelby_2(msh, level_set_function, prm, sym_grad);
        //auto test_case_cell = make_test_case_eshelby(msh, level_set_function,  prm , sym_grad);
        //This stuff before it was before the loop in msh.cells and test_case and not test_case_cell
        auto rhs_fun = test_case_cell.rhs_fun;
        auto sol_vel = test_case_cell.sol_vel;
        auto sol_p = test_case_cell.sol_p;
        auto vel_grad = test_case_cell.vel_grad;
        //auto bcs_vel = test_case.bcs_vel;
        auto neumann_jump = test_case_cell.neumann_jump;
        assembler_sc.set_dir_func( bcs_vel);
        
        
        Matrix<RealType, Dynamic, 1> vel_locdata_n, vel_locdata_p, vel_locdata;
        Matrix<RealType, Dynamic, 1> P_locdata_n, P_locdata_p, P_locdata;
        Matrix<RealType, Dynamic, 1> vel_cell_dofs_n, vel_cell_dofs_p, vel_cell_dofs;

        if (location(msh, cl) == element_location::ON_INTERFACE)
        {
            if( sc )
            {
                vel_locdata_n = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                vel_locdata_p = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata_n = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                P_locdata_p = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
            {
                vel_locdata_n = assembler.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                vel_locdata_p = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata_n = assembler.take_pressure(msh,cl, sol, element_location::IN_NEGATIVE_SIDE);
                P_locdata_p = assembler.take_pressure(msh,cl, sol, element_location::IN_POSITIVE_SIDE);
            }

            vel_cell_dofs_n = vel_locdata_n.head(cbs);
            vel_cell_dofs_p = vel_locdata_p.head(cbs);


            
            // Uploading velocity field by STE
            //std::cout<<"------------>>> CUT CELL"<<std::endl;
            //std::cout<<"subcells.size() = "<<level_set_function.subcells.size()<<std::endl;
            
            // NOT AGGLO CELL
            if ( level_set_function.subcells.size()<1 )
            {
                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells.size() == 2);
                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                {
                    if( level_set_function(ln_Qk) > iso_val_interface )
                    {
                        auto phi_HHO = cb.eval_basis( ln_Qk );
                        auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                        velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                        //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
                    }
                    else
                    {
                        auto phi_HHO = cb.eval_basis( ln_Qk );
                        auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                        velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                        //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
                        //  velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
                        //  velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                    }
                }
                
            }
            else // AGGLO CELL
            {
                for(size_t i_subcell = 0 ; i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size() ; i_subcell++ )
                {
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
                     std::cout<<"offset_old = "<<offset_old<<std::endl;
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                    size_t i_local = 0;
                    for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                    {
                        if( level_set_function(ln_Qk) > iso_val_interface )
                        {
                            auto phi_HHO = cb.eval_basis( ln_Qk );
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                            velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                            //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
                        }
                        else
                        {
                            auto phi_HHO = cb.eval_basis( ln_Qk );
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                            velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                            //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
                            //  velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
                            //  velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                        }
                    }
                    
                }
            }
                
            
            
            auto qps_n = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                // Compute H1-error //
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += vel_cell_dofs_n(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                H1_error += qp.second * inner_product(grad_diff , grad_diff);


                 // Compute L2-error //
                auto t_phi = cb.eval_basis( qp.first );
                auto v = t_phi.transpose() * vel_cell_dofs_n;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                uT1_gp->add_data( qp.first, v(0) );
                uT2_gp->add_data( qp.first, v(1) );

                // L2 - pressure - error //
                auto p_phi = pb.eval_basis( qp.first );
                RealType p_num = p_phi.dot(P_locdata_n);
                RealType p_diff = test_case_cell.sol_p( qp.first ) - p_num; // era test_case STE
                L2_pressure_error += qp.second * p_diff * p_diff;

                p_gp->add_data( qp.first, p_num );
            }

            auto qps_p = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                // Compute H1-error //
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += vel_cell_dofs_p(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                H1_error += qp.second * inner_product(grad_diff , grad_diff);

                // Compute L2-error //
                auto t_phi = cb.eval_basis( qp.first );
                auto v = t_phi.transpose() * vel_cell_dofs_p;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                uT1_gp->add_data( qp.first, v(0) );
                uT2_gp->add_data( qp.first, v(1) );

                 // L2 - pressure - error //
                auto p_phi = pb.eval_basis( qp.first );
                RealType p_num = p_phi.dot(P_locdata_p);
                RealType p_diff = test_case_cell.sol_p( qp.first ) - p_num; // era test_case STE
                L2_pressure_error += qp.second * p_diff * p_diff;

                p_gp->add_data( qp.first, p_num );
            }
        }
        else
        {
            if( sc )
            {
                vel_locdata = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
            {
                vel_locdata = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata = assembler.take_pressure(msh,cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            vel_cell_dofs = vel_locdata.head(cbs);

            //std::cout<<"------------>>> NOT CUT CELL!!!!!"<<std::endl;
            //std::cout<<"subcells.size() = "<<level_set_function.subcells.size()<<std::endl;
            /*
            for(size_t i_subcell = 0 ; i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size() ; i_subcell++ )
            {
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
                 std::cout<<"offset_old = "<<offset_old<<std::endl;
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                {
                    auto phi_HHO = cb.eval_basis( ln_Qk );
                    auto vel = phi_HHO.transpose() * vel_cell_dofs;
                    velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                    velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                    //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                    i_local++;
                  
                }
                
            }
            */
            
            // NOT AGGLO CELL
            if ( level_set_function.subcells.size()<1 )
            {
                assert(level_set_function.agglo_LS_cl.user_data.offset_subcells.size()==2);
                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                {
                    auto phi_HHO = cb.eval_basis( ln_Qk );
                    auto vel = phi_HHO.transpose() * vel_cell_dofs;
                    velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                    velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                    //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                    i_local++;
                  
                }
                
            }
            else // AGGLO CELL
            {
                for(size_t i_subcell = 0 ; i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size() ; i_subcell++ )
                {
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
                     std::cout<<"offset_old = "<<offset_old<<std::endl;
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                    size_t i_local = 0;
                    for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                    {
                        auto phi_HHO = cb.eval_basis( ln_Qk );
                        auto vel = phi_HHO.transpose() * vel_cell_dofs;
                        velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                        //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
                      
                    }
                    
                }
            }
            
            
       
            
            auto qps = integrate(msh, cl, 2*hdi.cell_degree());
            for (auto& qp : qps)
            {
                // Compute H1-error //
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += vel_cell_dofs(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                H1_error += qp.second * inner_product(grad_diff , grad_diff);

                // Compute L2-error //
                auto t_phi = cb.eval_basis( qp.first );
                auto v = t_phi.transpose() * vel_cell_dofs;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                uT1_gp->add_data( qp.first, v(0) );
                uT2_gp->add_data( qp.first, v(1) );

                // L2 - pressure - error //
                auto p_phi = pb.eval_basis( qp.first );
                RealType p_num = p_phi.dot(P_locdata);
                RealType p_diff = test_case_cell.sol_p( qp.first ) - p_num; // era test_case STE
                L2_pressure_error += qp.second * p_diff * p_diff;

                p_gp->add_data( qp.first, p_num );
            }
        }
        
        i_global++;
    }
    //std::cout<<"velocity.sol_HHO.first"<<'\n'<<velocity.sol_HHO.first<<std::endl;
    //std::cout<<"velocity.sol_HHO.second"<<'\n'<<velocity.sol_HHO.second<<std::endl;

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:               " << std::sqrt(L2_error) << std::endl;
    std::cout << bold << green << "Pressure L2-norm absolute error:      " << std::sqrt(L2_pressure_error) << std::endl;

    postoutput.add_object(uT1_gp);
    postoutput.add_object(uT2_gp);
    postoutput.add_object(p_gp);
    postoutput.write();



    stokes_test_info<RealType> TI;
    TI.H1_vel = std::sqrt(H1_error);
    TI.L2_vel = std::sqrt(L2_error);
    TI.L2_p = std::sqrt(L2_pressure_error);

    if (false)
    {
        /////////////// compute condition number
        SparseMatrix<RealType> Mat;
        // Matrix<RealType, Dynamic, Dynamic> Mat;
        if (sc)
            Mat = assembler_sc.LHS;
        else
            Mat = assembler.LHS;


        RealType sigma_max, sigma_min;

        // Construct matrix operation object using the wrapper class SparseSymMatProd
        Spectra::SparseSymMatProd<RealType> op(Mat);
        // Construct eigen solver object, requesting the largest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::LARGEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > max_eigs(&op, 1, 10);
        max_eigs.init();
        max_eigs.compute();
        if(max_eigs.info() == Spectra::SUCCESSFUL)
            sigma_max = max_eigs.eigenvalues()(0);


        // Construct eigen solver object, requesting the smallest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::SMALLEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > min_eigs(&op, 1, 10);

        min_eigs.init();
        min_eigs.compute();
        if(min_eigs.info() == Spectra::SUCCESSFUL)
            sigma_min = min_eigs.eigenvalues()(0);

        // compute condition number
        RealType cond = sigma_max / sigma_min;
        TI.cond = cond;
        std::cout << "sigma_max = " << sigma_max << "   sigma_min = "
                  << sigma_min << "  cond = " << cond
                  << std::endl;
    }
    else
        TI.cond = 0.0;

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


    return TI;
}

template<typename Mesh, typename testType, typename meth , typename Fonction , typename Velocity>
stokes_test_info<typename Mesh::coordinate_type>
run_cuthho_interface_velocity_prova(const Mesh& msh, size_t degree, meth method, testType test_case , Fonction & level_set_function , Velocity & velocity , bool sym_grad )
{
    using RealType = typename Mesh::coordinate_type;
    
    //auto level_set_function = test_case.level_set_;

    auto iso_val_interface = level_set_function.iso_val_interface ;
    std::cout<<"Interface isovalue = "<<iso_val_interface<<std::endl;
    auto bcs_vel = test_case.bcs_vel;
    
    struct params<RealType> parms = test_case.parms;

    timecounter tc;

    bool sc = true; // static condensation


    // ************** ASSEMBLE PROBLEM **************
    hho_degree_info hdi(degree+1, degree);

    tc.tic();
    //timecounter tc_bis2 ;
 
    auto assembler = make_stokes_interface_assembler(msh, bcs_vel, hdi);
   
    auto assembler_sc = make_stokes_interface_condensed_assembler(msh, bcs_vel, hdi);
    std::cout<<yellow<<bold<<"assembler_sc.set_dir_func ---> INTO the CELLS' LOOP." <<reset<<std::endl;
    
    
    test_case.test_case_mesh_assignment(msh) ;
   
    for (auto& cl : msh.cells)
    {
        // ADD BY STE
        //std::cout<<yellow<<bold<<"CELL = "<<offset(msh,cl) <<reset<<std::endl;
        //timecounter tc_bis ;
        //tc_bis.tic();
        //level_set_function.cell_assignment(cl);
        //auto test_case_cell = make_test_case_stokes_2(msh, level_set_function);
        //auto prm = params<RealType>();
        //prm.kappa_1 = 1.0;
        //prm.kappa_2 = 1.0;
        //auto test_case_cell = make_test_case_eshelby_2(msh, level_set_function, prm, sym_grad);
        test_case.test_case_cell_assignment(cl);
        test_case.refresh_lambdas(level_set_function, parms, sym_grad);
       
        //std::cout<<"test_case.cl = "<<offset(msh,test_case.cl)<<std::endl;
        //std::cout<<"test_case.i = "<<test_case.i <<std::endl;
       // std::cout<<"-------> TIME STOKES 0 , time = "<<tc_bis<<std::endl;
        //tc_bis.tic();
        //auto test_case_cell = make_test_case_eshelby(msh, level_set_function, prm, sym_grad);
        //This stuff before it was before the loop in msh.cells and test_case and not test_case_cell
        //auto rhs_fun = test_case_cell.rhs_fun;
        //auto sol_vel = test_case_cell.sol_vel;
        //auto sol_p = test_case_cell.sol_p;
        //auto vel_grad = test_case_cell.vel_grad;
        ///---> QUESTO NO auto bcs_vel = test_case.bcs_vel;
        //auto neumann_jump = test_case_cell.neumann_jump;
        assembler_sc.set_dir_func( bcs_vel ); // DOVE VA? INTO LOOP cl? SE CAMBIASSE bcs_vel in spazio forse si!
        //tc_bis.toc();
        //std::cout<<"-------> TIME STOKES 1 , time = "<<tc_bis<<std::endl;
        //tc_bis.tic();
        auto contrib = method.make_contrib(msh, cl, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;
        //tc_bis.toc();
        //std::cout<<"-------> TIME STOKES 2 , time = "<<tc_bis<<std::endl;
        //tc_bis.tic();
        if( sc )
            assembler_sc.assemble(msh, cl, lc, f);
        else
            assembler.assemble(msh, cl, lc, f);
        
        //tc_bis.toc();
        //std::cout<<"-------> TIME STOKES 3 , time = "<<tc_bis<<std::endl;
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

    // ************** SOLVE **************
    tc.tic();
#if 1
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
#if 0
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
#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    // ************** POSTPROCESS **************


    postprocess_output<RealType>  postoutput;

    auto uT1_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT1.dat");
    auto uT2_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT2.dat");
    auto p_gp    = std::make_shared< gnuplot_output_object<RealType> >("interface_p.dat");

    tc.tic();
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
    RealType    L2_pressure_error = 0.0;
    
    std::cout<<"sono qua 0.0"<<std::endl;
    size_t i_global = 0 ; // ADD BY STE
    for (auto& cl : msh.cells)
    {
        
        
        vector_cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        
        
        cell_basis<cuthho_poly_mesh<RealType>, RealType> pb(msh, cl, hdi.face_degree());
        auto cbs = cb.size();
        //auto pbs = pb.size();

        
        // ADD BY STE
        level_set_function.cell_assignment(cl);
        //auto test_case_cell = make_test_case_stokes_2(msh, level_set_function);
        
        //auto prm = params<RealType>();
        //prm.kappa_1 = 1.0;
        //prm.kappa_2 = 1.0;
        //auto test_case_cell = make_test_case_eshelby_2(msh, level_set_function, prm, sym_grad);
        test_case.test_case_cell_assignment(cl) ;
        test_case.refresh_lambdas(level_set_function, parms, sym_grad);
        //auto test_case_cell = make_test_case_eshelby(msh, level_set_function,  prm , sym_grad);
        //This stuff before it was before the loop in msh.cells and test_case and not test_case_cell
        //auto rhs_fun = test_case.rhs_fun;
        auto sol_vel = test_case.sol_vel;
        auto sol_p = test_case.sol_p;
        auto vel_grad = test_case.vel_grad;
        //auto bcs_vel = test_case.bcs_vel;
        //auto neumann_jump = test_case.neumann_jump;
        assembler_sc.set_dir_func( bcs_vel);
        
        
        Matrix<RealType, Dynamic, 1> vel_locdata_n, vel_locdata_p, vel_locdata;
        Matrix<RealType, Dynamic, 1> P_locdata_n, P_locdata_p, P_locdata;
        Matrix<RealType, Dynamic, 1> vel_cell_dofs_n, vel_cell_dofs_p, vel_cell_dofs;

        if (location(msh, cl) == element_location::ON_INTERFACE)
        {
            if( sc )
            {
                vel_locdata_n = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                vel_locdata_p = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata_n = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                P_locdata_p = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
            {
                vel_locdata_n = assembler.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                vel_locdata_p = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata_n = assembler.take_pressure(msh,cl, sol, element_location::IN_NEGATIVE_SIDE);
                P_locdata_p = assembler.take_pressure(msh,cl, sol, element_location::IN_POSITIVE_SIDE);
            }

            vel_cell_dofs_n = vel_locdata_n.head(cbs);
            vel_cell_dofs_p = vel_locdata_p.head(cbs);


            
            // Uploading velocity field by STE
            //std::cout<<"------------>>> CUT CELL"<<std::endl;
            //std::cout<<"subcells.size() = "<<level_set_function.subcells.size()<<std::endl;
            
            // NOT AGGLO CELL
            if ( level_set_function.subcells.size()<1 )
            {
                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells.size() == 2);
                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                {
                    if( level_set_function(ln_Qk) > iso_val_interface )
                    {
                        auto phi_HHO = cb.eval_basis( ln_Qk );
                        auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                        velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                        //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
                    }
                    else
                    {
                        auto phi_HHO = cb.eval_basis( ln_Qk );
                        auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                        velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                        //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
                        //  velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
                        //  velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                    }
                }
                
            }
            else // AGGLO CELL
            {
                for(size_t i_subcell = 0 ; i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size() ; i_subcell++ )
                {
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
                     std::cout<<"offset_old = "<<offset_old<<std::endl;
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                    size_t i_local = 0;
                    for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                    {
                        if( level_set_function(ln_Qk) > iso_val_interface )
                        {
                            auto phi_HHO = cb.eval_basis( ln_Qk );
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                            velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                            //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
                        }
                        else
                        {
                            auto phi_HHO = cb.eval_basis( ln_Qk );
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                            velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                            //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
                            //  velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
                            //  velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                        }
                    }
                    
                }
            }
                
            
            
            auto qps_n = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                // Compute H1-error //
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += vel_cell_dofs_n(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                H1_error += qp.second * inner_product(grad_diff , grad_diff);


                 // Compute L2-error //
                auto t_phi = cb.eval_basis( qp.first );
                auto v = t_phi.transpose() * vel_cell_dofs_n;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                uT1_gp->add_data( qp.first, v(0) );
                uT2_gp->add_data( qp.first, v(1) );

                // L2 - pressure - error //
                auto p_phi = pb.eval_basis( qp.first );
                RealType p_num = p_phi.dot(P_locdata_n);
                RealType p_diff = test_case.sol_p( qp.first ) - p_num; // era test_case STE
                L2_pressure_error += qp.second * p_diff * p_diff;

                p_gp->add_data( qp.first, p_num );
            }

            auto qps_p = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                // Compute H1-error //
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += vel_cell_dofs_p(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                H1_error += qp.second * inner_product(grad_diff , grad_diff);

                // Compute L2-error //
                auto t_phi = cb.eval_basis( qp.first );
                auto v = t_phi.transpose() * vel_cell_dofs_p;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                uT1_gp->add_data( qp.first, v(0) );
                uT2_gp->add_data( qp.first, v(1) );

                 // L2 - pressure - error //
                auto p_phi = pb.eval_basis( qp.first );
                RealType p_num = p_phi.dot(P_locdata_p);
                RealType p_diff = test_case.sol_p( qp.first ) - p_num; // era test_case STE
                L2_pressure_error += qp.second * p_diff * p_diff;

                p_gp->add_data( qp.first, p_num );
            }
        }
        else
        {
            if( sc )
            {
                vel_locdata = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
            {
                vel_locdata = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata = assembler.take_pressure(msh,cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            vel_cell_dofs = vel_locdata.head(cbs);

            //std::cout<<"------------>>> NOT CUT CELL!!!!!"<<std::endl;
            //std::cout<<"subcells.size() = "<<level_set_function.subcells.size()<<std::endl;
            /*
            for(size_t i_subcell = 0 ; i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size() ; i_subcell++ )
            {
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
                 std::cout<<"offset_old = "<<offset_old<<std::endl;
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                {
                    auto phi_HHO = cb.eval_basis( ln_Qk );
                    auto vel = phi_HHO.transpose() * vel_cell_dofs;
                    velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                    velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                    //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                    i_local++;
                  
                }
                
            }
            */
            
            // NOT AGGLO CELL
            if ( level_set_function.subcells.size()<1 )
            {
                assert(level_set_function.agglo_LS_cl.user_data.offset_subcells.size()==2);
                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                {
                    auto phi_HHO = cb.eval_basis( ln_Qk );
                    auto vel = phi_HHO.transpose() * vel_cell_dofs;
                    velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                    velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                    //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                    i_local++;
                  
                }
                
            }
            else // AGGLO CELL
            {
                for(size_t i_subcell = 0 ; i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size() ; i_subcell++ )
                {
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
                     std::cout<<"offset_old = "<<offset_old<<std::endl;
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                    size_t i_local = 0;
                    for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                    {
                        auto phi_HHO = cb.eval_basis( ln_Qk );
                        auto vel = phi_HHO.transpose() * vel_cell_dofs;
                        velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                        //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
                      
                    }
                    
                }
            }
            
            
       
            
            auto qps = integrate(msh, cl, 2*hdi.cell_degree());
            for (auto& qp : qps)
            {
                // Compute H1-error //
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += vel_cell_dofs(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                H1_error += qp.second * inner_product(grad_diff , grad_diff);

                // Compute L2-error //
                auto t_phi = cb.eval_basis( qp.first );
                auto v = t_phi.transpose() * vel_cell_dofs;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                uT1_gp->add_data( qp.first, v(0) );
                uT2_gp->add_data( qp.first, v(1) );

                // L2 - pressure - error //
                auto p_phi = pb.eval_basis( qp.first );
                RealType p_num = p_phi.dot(P_locdata);
                RealType p_diff = test_case.sol_p( qp.first ) - p_num; // era test_case STE
                L2_pressure_error += qp.second * p_diff * p_diff;

                p_gp->add_data( qp.first, p_num );
            }
        }
        
        i_global++;
    }
    //std::cout<<"velocity.sol_HHO.first"<<'\n'<<velocity.sol_HHO.first<<std::endl;
    //std::cout<<"velocity.sol_HHO.second"<<'\n'<<velocity.sol_HHO.second<<std::endl;

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:               " << std::sqrt(L2_error) << std::endl;
    std::cout << bold << green << "Pressure L2-norm absolute error:      " << std::sqrt(L2_pressure_error) << std::endl;

    postoutput.add_object(uT1_gp);
    postoutput.add_object(uT2_gp);
    postoutput.add_object(p_gp);
    postoutput.write();



    stokes_test_info<RealType> TI;
    TI.H1_vel = std::sqrt(H1_error);
    TI.L2_vel = std::sqrt(L2_error);
    TI.L2_p = std::sqrt(L2_pressure_error);

    if (false)
    {
        /////////////// compute condition number
        SparseMatrix<RealType> Mat;
        // Matrix<RealType, Dynamic, Dynamic> Mat;
        if (sc)
            Mat = assembler_sc.LHS;
        else
            Mat = assembler.LHS;


        RealType sigma_max, sigma_min;

        // Construct matrix operation object using the wrapper class SparseSymMatProd
        Spectra::SparseSymMatProd<RealType> op(Mat);
        // Construct eigen solver object, requesting the largest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::LARGEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > max_eigs(&op, 1, 10);
        max_eigs.init();
        max_eigs.compute();
        if(max_eigs.info() == Spectra::SUCCESSFUL)
            sigma_max = max_eigs.eigenvalues()(0);


        // Construct eigen solver object, requesting the smallest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::SMALLEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > min_eigs(&op, 1, 10);

        min_eigs.init();
        min_eigs.compute();
        if(min_eigs.info() == Spectra::SUCCESSFUL)
            sigma_min = min_eigs.eigenvalues()(0);

        // compute condition number
        RealType cond = sigma_max / sigma_min;
        TI.cond = cond;
        std::cout << "sigma_max = " << sigma_max << "   sigma_min = "
                  << sigma_min << "  cond = " << cond
                  << std::endl;
    }
    else
        TI.cond = 0.0;

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


    return TI;
}


template<typename Mesh, typename testType, typename meth , typename Fonction , typename Velocity>
stokes_test_info<typename Mesh::coordinate_type>
run_cuthho_interface_velocity_parallel(const Mesh& msh, size_t degree, meth method, testType test_case , Fonction & level_set_function , Velocity & velocity , bool sym_grad )
{
    using RealType = typename Mesh::coordinate_type;
    
    //auto level_set_function = test_case.level_set_;

    auto iso_val_interface = level_set_function.iso_val_interface ;
    std::cout<<"Interface isovalue = "<<iso_val_interface<<std::endl;
    auto bcs_vel = test_case.bcs_vel;
    
    struct params<RealType> parms = test_case.parms;

    timecounter tc;

    bool sc = true; // static condensation


    // ************** ASSEMBLE PROBLEM **************
    hho_degree_info hdi(degree+1, degree);

    tc.tic();
    timecounter tc_bis2 ;
    tc_bis2.tic();
    auto assembler = make_stokes_interface_assembler(msh, bcs_vel, hdi);
    tc_bis2.toc();
    std::cout<<"-------> TIME assembler , time = "<<tc_bis2<<std::endl;
    tc_bis2.tic();
    auto assembler_sc = make_stokes_interface_condensed_assembler(msh, bcs_vel, hdi);
    tc_bis2.toc();
    std::cout<<"-------> TIME assembler_sc , time = "<<tc_bis2<<std::endl;
    size_t n_cells = msh.cells.size();
    std::cout<<" I m in parallel zone"<<std::endl;
    tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
    [&] (size_t  cell_ind){
        auto& cl = msh.cells[cell_ind];
            
    //for (auto& cl : msh.cells)
    //{
        // ADD BY STE
        timecounter tc_bis ;
        tc_bis.tic();
        level_set_function.cell_assignment(cl);
        //auto test_case_cell = make_test_case_stokes_2(msh, level_set_function);
        auto prm = params<RealType>();
        prm.kappa_1 = 1.0;
        prm.kappa_2 = 1.0;
        auto test_case_cell = make_test_case_eshelby_2(msh, level_set_function, prm, sym_grad);
        
        //auto test_case_cell = make_test_case_eshelby(msh, level_set_function, prm, sym_grad);
        //This stuff before it was before the loop in msh.cells and test_case and not test_case_cell
        auto rhs_fun = test_case_cell.rhs_fun;
        auto sol_vel = test_case_cell.sol_vel;
        auto sol_p = test_case_cell.sol_p;
        auto vel_grad = test_case_cell.vel_grad;
        //auto bcs_vel = test_case.bcs_vel;
        auto neumann_jump = test_case_cell.neumann_jump;
        assembler_sc.set_dir_func( bcs_vel);
        tc_bis.toc();
        std::cout<<"-------> TIME STOKES 0 , time = "<<tc_bis<<std::endl;
        tc_bis.tic();
        auto contrib = method.make_contrib(msh, cl, test_case_cell, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;
        tc_bis.toc();
        std::cout<<"-------> TIME STOKES 1 , time = "<<tc_bis<<std::endl;
        tc_bis.tic();
        if( sc )
            assembler_sc.assemble(msh, cl, lc, f);
        else
            assembler.assemble(msh, cl, lc, f);
        
        tc_bis.toc();
        std::cout<<"-------> TIME STOKES 2 , time = "<<tc_bis<<std::endl;
    
    
       
    });
    
    tc_bis2.tic();
    if( sc )
        assembler_sc.finalize();
    else
        assembler.finalize();

    tc_bis2.toc();
    std::cout<<"-------> TIME STOKES 3 , time = "<<tc_bis2<<std::endl;
    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

    if( sc )
        std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
    else
        std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;

    // ************** SOLVE **************
    tc.tic();
#if 1
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
#if 0
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
#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    // ************** POSTPROCESS **************


    postprocess_output<RealType>  postoutput;

    auto uT1_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT1.dat");
    auto uT2_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT2.dat");
    auto p_gp    = std::make_shared< gnuplot_output_object<RealType> >("interface_p.dat");

    tc.tic();
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
    RealType    L2_pressure_error = 0.0;
    
    std::cout<<"sono qua 0.0"<<std::endl;
    size_t i_global = 0 ; // ADD BY STE
    for (auto& cl : msh.cells)
    {
        
        
        vector_cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        
        
        cell_basis<cuthho_poly_mesh<RealType>, RealType> pb(msh, cl, hdi.face_degree());
        auto cbs = cb.size();
        auto pbs = pb.size();

        
        // ADD BY STE
        level_set_function.cell_assignment(cl);
        //auto test_case_cell = make_test_case_stokes_2(msh, level_set_function);
        
        auto prm = params<RealType>();
        prm.kappa_1 = 1.0;
        prm.kappa_2 = 1.0;
        auto test_case_cell = make_test_case_eshelby_2(msh, level_set_function, prm, sym_grad);
        //auto test_case_cell = make_test_case_eshelby(msh, level_set_function,  prm , sym_grad);
        //This stuff before it was before the loop in msh.cells and test_case and not test_case_cell
        auto rhs_fun = test_case_cell.rhs_fun;
        auto sol_vel = test_case_cell.sol_vel;
        auto sol_p = test_case_cell.sol_p;
        auto vel_grad = test_case_cell.vel_grad;
        //auto bcs_vel = test_case.bcs_vel;
        auto neumann_jump = test_case_cell.neumann_jump;
        assembler_sc.set_dir_func( bcs_vel);
        
        
        Matrix<RealType, Dynamic, 1> vel_locdata_n, vel_locdata_p, vel_locdata;
        Matrix<RealType, Dynamic, 1> P_locdata_n, P_locdata_p, P_locdata;
        Matrix<RealType, Dynamic, 1> vel_cell_dofs_n, vel_cell_dofs_p, vel_cell_dofs;

        if (location(msh, cl) == element_location::ON_INTERFACE)
        {
            if( sc )
            {
                vel_locdata_n = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                vel_locdata_p = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata_n = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                P_locdata_p = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
            {
                vel_locdata_n = assembler.take_velocity(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                vel_locdata_p = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata_n = assembler.take_pressure(msh,cl, sol, element_location::IN_NEGATIVE_SIDE);
                P_locdata_p = assembler.take_pressure(msh,cl, sol, element_location::IN_POSITIVE_SIDE);
            }

            vel_cell_dofs_n = vel_locdata_n.head(cbs);
            vel_cell_dofs_p = vel_locdata_p.head(cbs);


            
            // Uploading velocity field by STE
            //std::cout<<"------------>>> CUT CELL"<<std::endl;
            //std::cout<<"subcells.size() = "<<level_set_function.subcells.size()<<std::endl;
            
            // NOT AGGLO CELL
            if ( level_set_function.subcells.size()<1 )
            {
                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells.size() == 2);
                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                {
                    if( level_set_function(ln_Qk) > iso_val_interface )
                    {
                        auto phi_HHO = cb.eval_basis( ln_Qk );
                        auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                        velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                        //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
                    }
                    else
                    {
                        auto phi_HHO = cb.eval_basis( ln_Qk );
                        auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                        velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                        //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
                        //  velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
                        //  velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                    }
                }
                
            }
            else // AGGLO CELL
            {
                for(size_t i_subcell = 0 ; i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size() ; i_subcell++ )
                {
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
                     std::cout<<"offset_old = "<<offset_old<<std::endl;
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                    size_t i_local = 0;
                    for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                    {
                        if( level_set_function(ln_Qk) > iso_val_interface )
                        {
                            auto phi_HHO = cb.eval_basis( ln_Qk );
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_p;
                            velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                            //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
                        }
                        else
                        {
                            auto phi_HHO = cb.eval_basis( ln_Qk );
                            auto vel = phi_HHO.transpose() * vel_cell_dofs_n;
                            velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                            velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                            //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                            i_local++;
                            //  velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
                            //  velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                        }
                    }
                    
                }
            }
                
            
            
            auto qps_n = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                // Compute H1-error //
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += vel_cell_dofs_n(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                H1_error += qp.second * inner_product(grad_diff , grad_diff);


                 // Compute L2-error //
                auto t_phi = cb.eval_basis( qp.first );
                auto v = t_phi.transpose() * vel_cell_dofs_n;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                uT1_gp->add_data( qp.first, v(0) );
                uT2_gp->add_data( qp.first, v(1) );

                // L2 - pressure - error //
                auto p_phi = pb.eval_basis( qp.first );
                RealType p_num = p_phi.dot(P_locdata_n);
                RealType p_diff = test_case_cell.sol_p( qp.first ) - p_num; // era test_case STE
                L2_pressure_error += qp.second * p_diff * p_diff;

                p_gp->add_data( qp.first, p_num );
            }

            auto qps_p = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                // Compute H1-error //
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += vel_cell_dofs_p(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                H1_error += qp.second * inner_product(grad_diff , grad_diff);

                // Compute L2-error //
                auto t_phi = cb.eval_basis( qp.first );
                auto v = t_phi.transpose() * vel_cell_dofs_p;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                uT1_gp->add_data( qp.first, v(0) );
                uT2_gp->add_data( qp.first, v(1) );

                 // L2 - pressure - error //
                auto p_phi = pb.eval_basis( qp.first );
                RealType p_num = p_phi.dot(P_locdata_p);
                RealType p_diff = test_case_cell.sol_p( qp.first ) - p_num; // era test_case STE
                L2_pressure_error += qp.second * p_diff * p_diff;

                p_gp->add_data( qp.first, p_num );
            }
        }
        else
        {
            if( sc )
            {
                vel_locdata = assembler_sc.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata = assembler_sc.take_pressure(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
            {
                vel_locdata = assembler.take_velocity(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
                P_locdata = assembler.take_pressure(msh,cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            vel_cell_dofs = vel_locdata.head(cbs);

            //std::cout<<"------------>>> NOT CUT CELL!!!!!"<<std::endl;
            //std::cout<<"subcells.size() = "<<level_set_function.subcells.size()<<std::endl;
            /*
            for(size_t i_subcell = 0 ; i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size() ; i_subcell++ )
            {
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
                 std::cout<<"offset_old = "<<offset_old<<std::endl;
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                {
                    auto phi_HHO = cb.eval_basis( ln_Qk );
                    auto vel = phi_HHO.transpose() * vel_cell_dofs;
                    velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                    velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                    //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                    i_local++;
                  
                }
                
            }
            */
            
            // NOT AGGLO CELL
            if ( level_set_function.subcells.size()<1 )
            {
                assert(level_set_function.agglo_LS_cl.user_data.offset_subcells.size()==2);
                assert( level_set_function.agglo_LS_cl.user_data.offset_subcells[0] == level_set_function.agglo_LS_cl.user_data.offset_subcells[1] );
                auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = velocity.msh.cells[offset_old];
                auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                size_t i_local = 0;
                for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                {
                    auto phi_HHO = cb.eval_basis( ln_Qk );
                    auto vel = phi_HHO.transpose() * vel_cell_dofs;
                    velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                    velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                    //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                    i_local++;
                  
                }
                
            }
            else // AGGLO CELL
            {
                for(size_t i_subcell = 0 ; i_subcell < level_set_function.agglo_LS_cl.user_data.offset_subcells.size() ; i_subcell++ )
                {
                    auto offset_old = level_set_function.agglo_LS_cl.user_data.offset_subcells[i_subcell];
                     std::cout<<"offset_old = "<<offset_old<<std::endl;
                    auto cl_old = velocity.msh.cells[offset_old];
                    auto Lagrange_nodes_Qk = equidistriduted_nodes_ordered_bis<RealType,Mesh> (velocity.msh,cl_old,velocity.degree_FEM);
                    size_t i_local = 0;
                    for ( const auto & ln_Qk : Lagrange_nodes_Qk)
                    {
                        auto phi_HHO = cb.eval_basis( ln_Qk );
                        auto vel = phi_HHO.transpose() * vel_cell_dofs;
                        velocity.sol_HHO.first(i_local,offset_old) = vel(0);
                        velocity.sol_HHO.second(i_local,offset_old) = vel(1);
                        //std::cout<<"In pt = "<<ln_Qk<<"-> vel(0) = "<<vel(0)<<" and vel(1) = "<<vel(1)<<std::endl;
                        i_local++;
                      
                    }
                    
                }
            }
            
            
       
            
            auto qps = integrate(msh, cl, 2*hdi.cell_degree());
            for (auto& qp : qps)
            {
                // Compute H1-error //
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 2, 2> grad = Matrix<RealType, 2, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += vel_cell_dofs(i) * t_dphi[i].block(0, 0, 2, 2);

                Matrix<RealType, 2, 2> grad_diff = vel_grad(qp.first) - grad;
                H1_error += qp.second * inner_product(grad_diff , grad_diff);

                // Compute L2-error //
                auto t_phi = cb.eval_basis( qp.first );
                auto v = t_phi.transpose() * vel_cell_dofs;
                Matrix<RealType, 2, 1> sol_diff = sol_vel(qp.first) - v;
                L2_error += qp.second * sol_diff.dot(sol_diff);

                uT1_gp->add_data( qp.first, v(0) );
                uT2_gp->add_data( qp.first, v(1) );

                // L2 - pressure - error //
                auto p_phi = pb.eval_basis( qp.first );
                RealType p_num = p_phi.dot(P_locdata);
                RealType p_diff = test_case_cell.sol_p( qp.first ) - p_num; // era test_case STE
                L2_pressure_error += qp.second * p_diff * p_diff;

                p_gp->add_data( qp.first, p_num );
            }
        }
        
        i_global++;
    }
    //std::cout<<"velocity.sol_HHO.first"<<'\n'<<velocity.sol_HHO.first<<std::endl;
    //std::cout<<"velocity.sol_HHO.second"<<'\n'<<velocity.sol_HHO.second<<std::endl;

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:               " << std::sqrt(L2_error) << std::endl;
    std::cout << bold << green << "Pressure L2-norm absolute error:      " << std::sqrt(L2_pressure_error) << std::endl;

    postoutput.add_object(uT1_gp);
    postoutput.add_object(uT2_gp);
    postoutput.add_object(p_gp);
    postoutput.write();



    stokes_test_info<RealType> TI;
    TI.H1_vel = std::sqrt(H1_error);
    TI.L2_vel = std::sqrt(L2_error);
    TI.L2_p = std::sqrt(L2_pressure_error);

    if (false)
    {
        /////////////// compute condition number
        SparseMatrix<RealType> Mat;
        // Matrix<RealType, Dynamic, Dynamic> Mat;
        if (sc)
            Mat = assembler_sc.LHS;
        else
            Mat = assembler.LHS;


        RealType sigma_max, sigma_min;

        // Construct matrix operation object using the wrapper class SparseSymMatProd
        Spectra::SparseSymMatProd<RealType> op(Mat);
        // Construct eigen solver object, requesting the largest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::LARGEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > max_eigs(&op, 1, 10);
        max_eigs.init();
        max_eigs.compute();
        if(max_eigs.info() == Spectra::SUCCESSFUL)
            sigma_max = max_eigs.eigenvalues()(0);


        // Construct eigen solver object, requesting the smallest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::SMALLEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > min_eigs(&op, 1, 10);

        min_eigs.init();
        min_eigs.compute();
        if(min_eigs.info() == Spectra::SUCCESSFUL)
            sigma_min = min_eigs.eigenvalues()(0);

        // compute condition number
        RealType cond = sigma_max / sigma_min;
        TI.cond = cond;
        std::cout << "sigma_max = " << sigma_max << "   sigma_min = "
                  << sigma_min << "  cond = " << cond
                  << std::endl;
    }
    else
        TI.cond = 0.0;

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;


    return TI;
}




///// test_case_kink_velocity --> DISCRETE LEVEL SET
// !! available for circle_level_set only !!
// exact solution : u(r) sin(theta) in the whole domain for vel_component 1
//                 -u(r) cos(theta) in the whole domain for vel_component 2
//         with u(r) = r^6 / kappa_1 in Omega_1
//              u(r) = (r^6 - R^6)/kappa_2 + R^6/kappa_1 in Omega_2
//                   sin(x+y)     in the whole domain for p
// \kappa_1 , \kappa_2 given
template<typename T, typename Mesh, typename Function >
class test_case_eshelby: public test_case_stokes<T, Function , Mesh>
{
  public:
   test_case_eshelby(const Function & level_set__, params<T> parms_, bool sym_grad)
       : test_case_stokes<T, Function , Mesh>
       (level_set__, parms_,
        [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> { // sol_vel
           Matrix<T, 2, 1> ret;
           T C = 2.0;
           T mid_point = 1.0/2.0;
           T x1 = pt.x() - mid_point ;
           T y1 = mid_point - pt.y() ;
           ret(0) = C*x1 ;
           ret(1) = C*y1 ;
           
           
           return ret;},
        [level_set__](const typename Mesh::point_type& pt) -> T { // p
            //return 1.0;
           
           T k = 1.0;
           T R = 1.0/3.0;
           if(level_set__(pt) < 0)
               return k / R - M_PI * R * k;
            else
                return -M_PI * R * k;
           
       },
        [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> { // rhs
            Matrix<T, 2, 1> ret;
            ret(0) = 0.0 ;
            ret(1) = 0.0 ;
            return ret;},
        [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> { // bcs
           Matrix<T, 2, 1> ret;
           T C = 2.0;
           T mid_point = 1.0/2.0;
           T x1 = pt.x() - mid_point ;
           T y1 = mid_point - pt.y() ;
           ret(0) = C*x1 ;
           ret(1) = C*y1 ;
           return ret;},
        [](const typename Mesh::point_type& pt) -> auto { // grad
           
           Matrix<T, 2, 2> ret;
           T C = 2.0;
           ret(0,0) = C*1.0;
           ret(0,1) = C*0.0;
           ret(1,0) = C*0.0;
           ret(1,1) = C*(-1.0);
           return ret;},
        [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
            Matrix<T, 2, 1> ret;
            ret(0) = 0.0;
            ret(1) = 0.0;
            return ret;},
        [level_set__,parms_,sym_grad](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> {/* Neu */
            Matrix<T, 2, 1> ret;
            if(sym_grad)
            {
                T gamma = 1.0;
                //T k = 1.0;
                //T R = 1.0/3.0;
                
                //T H = level_set__.normal(pt)
                ret(0) = 2.0*gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                ret(1) = 2.0*gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);
                
                //ret(0) = - k / R * level_set__.normal(pt)(0);
                //ret(1) = - k / R * level_set__.normal(pt)(1);
                
            }
            else
            {
                ret(0) = 0.0;
                ret(1) = 0.0;
            }
            return ret;})
       {}
};

template<typename Mesh, typename T, typename Function>
auto make_test_case_eshelby(const Mesh& msh, const Function& level_set_function, params<T> parms_, bool sym_grad)
{
   return test_case_eshelby<typename Mesh::coordinate_type, Mesh , Function>(level_set_function,parms_,sym_grad);
}

/// FATTO BY STEFANO
// Starting from an elliptic level set -> final circle equilibrium
// exact solution : u = 0
//                  constant = \kappa    in the whole domain for p
// \kappa  given

template<typename T, typename Mesh, typename Function >
class test_case_eshelby_2: public test_case_stokes<T, Function , Mesh>
{
  public:
   test_case_eshelby_2(const Function & level_set__, params<T> parms_, bool sym_grad)
       : test_case_stokes<T, Function , Mesh>
       (level_set__, parms_,
        [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> {
           // sol_vel
           Matrix<T, 2, 1> ret;
          
           ret(0) = 0.0;
           ret(1) = 0.0;
           
           return ret;},
        [level_set__](const typename Mesh::point_type& pt) -> T { // p
          
           return 0.0;
           
       },
        [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> { // rhs
            Matrix<T, 2, 1> ret;
            ret(0) = 0.0 ;
            ret(1) = 0.0 ;
            return ret;},
        [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> { // bcs
           Matrix<T, 2, 1> ret;
           
           ret(0) = 0.0;
           ret(1) = 0.0;
           return ret;},
        [](const typename Mesh::point_type& pt) -> auto { // grad
           
           Matrix<T, 2, 2> ret;
           ret(0,0) = 0.0;
           ret(0,1) = 0.0;
           ret(1,0) = 0.0;
           ret(1,1) = (0.0);
           return ret;},
        [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
            Matrix<T, 2, 1> ret;
            ret(0) = 0.0;
            ret(1) = 0.0;
            return ret;},
        [level_set__,sym_grad](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> {/* Neu */
            Matrix<T, 2, 1> ret;
            if(sym_grad)
            {
                T gamma = 1.0;
                //T k = 1.0;
                //T R = 1.0/3.0;
                
                //T H = level_set__.normal(pt)
                
                ret(0) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                ret(1) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);
               
                
                //ret(0) = - k / R * level_set__.normal(pt)(0);
                //ret(1) = - k / R * level_set__.normal(pt)(1);
                
            }
            else
            {
                T gamma = 1.0;
                            
                ret(0) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                ret(1) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);
            }
            return ret;})
       {}
};

template<typename Mesh, typename T, typename Function>
auto make_test_case_eshelby_2(const Mesh& msh, const Function& level_set_function, params<T> parms_, bool sym_grad)
{
   return test_case_eshelby_2<typename Mesh::coordinate_type, Mesh , Function>(level_set_function,parms_,sym_grad);
}



template<typename T, typename Mesh, typename Function >
class test_case_eshelby_2_prova: public test_case_stokes<T, Function , Mesh>
{
  
public:
    
    Mesh m_msh  ;
    typename Mesh::cell_type m_cl ;
//    Mesh* msh_pt ;
//    typename Mesh::cell_type* cl_pt ;
//    size_t i = 2;
    //std::shared_ptr<typename Mesh::cell_type> cl_pointer;
    
    explicit test_case_eshelby_2_prova( Function & level_set__, params<T> parms_, bool sym_grad)
       : test_case_stokes<T, Function , Mesh>
       (level_set__, parms_,
        [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> {
           // sol_vel
           Matrix<T, 2, 1> ret;
          
           ret(0) = 0.0;
           ret(1) = 0.0;
           
           return ret;},
        [level_set__](const typename Mesh::point_type& pt) -> T { // p
          
           return 0.0;
           
       },
        [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> { // rhs
            Matrix<T, 2, 1> ret;
            ret(0) = 0.0 ;
            ret(1) = 0.0 ;
            return ret;},
        [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> { // bcs
           Matrix<T, 2, 1> ret;
           
           ret(0) = 0.0;
           ret(1) = 0.0;
           return ret;},
        [](const typename Mesh::point_type& pt) -> auto { // grad
           
           Matrix<T, 2, 2> ret;
           ret(0,0) = 0.0;
           ret(0,1) = 0.0;
           ret(1,0) = 0.0;
           ret(1,1) = (0.0);
           return ret;},
        [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
            Matrix<T, 2, 1> ret;
            ret(0) = 0.0;
            ret(1) = 0.0;
            return ret;},
        [level_set__,sym_grad,this](const typename Mesh::point_type& pt) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */
            Matrix<T, 2, 1> ret;
            if(sym_grad)
            {
                T gamma = 1.0;
                //T k = 1.0;
                //T R = 1.0/3.0;
                //cl = cl_pointer.get() ;
                //std::cout<<"SONO  IN NEUMANN CONDITION."<<std::endl;
//                std::cout<<"i = "<<i<<std::endl;
                //auto cl_uploaded = this->cl ;
//                i++;
//                std::cout<<"i = "<<i<<std::endl;
//                Mesh msh_prova = *msh_pt ;
//                typename Mesh::cell_type cl_prova = *cl_pt ;
                
                
//                std::cout<<"-----> test_case_eshelby_2_prova : CELL PROVA = "<<offset(msh_prova,cl_prova)<<std::endl;
                //if(i==3)
                //   this->cl = msh.cells[227];
                //std::cout<<"-----> test_case_eshelby_2_prova : CELL = "<<offset(m_msh,m_cl)<<std::endl;
                /*
                auto cl_new = this->cl ;
                std::cout<<"-----> test_case_eshelby_2_prova : CELL = "<<offset(msh,cl_new)<<std::endl;
                
                auto cl_new2 = upload_cl();
                std::cout<<"-----> test_case_eshelby_2_prova : CELL = "<<offset(msh,cl_new2)<<std::endl;
                
                auto cl_new3 = upload_cl2();
                std::cout<<"-----> test_case_eshelby_2_prova : CELL = "<<offset(msh,cl_new3)<<std::endl;
                */
                level_set__.cell_assignment(m_cl);
                //T H = level_set__.normal(pt)
                
                ret(0) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                ret(1) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);
               
                
                //ret(0) = - k / R * level_set__.normal(pt)(0);
                //ret(1) = - k / R * level_set__.normal(pt)(1);
                
            }
            else
            {
                T gamma = 1.0;
                level_set__.cell_assignment(m_cl);
                ret(0) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                ret(1) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);
            }
            return ret;})
       {}
    
    test_case_eshelby_2_prova(const test_case_eshelby_2_prova & other) : test_case_stokes<T, Function , Mesh>(other) {
        m_msh = other.m_msh;
        m_cl = other.m_cl;
    }
    
//    void test_case_cell_assignment(const Mesh& msh , const typename Mesh::cell_type& cl_new )
//    {
//        std::cout<<"sono qua 0"<<std::endl;
//        std::cout<<"-----> test_case_cell_assignment : CELL OLD = "<<offset(msh,cl)<<std::endl;
//        std::cout<<"sono qua 1"<<std::endl;
//        std::cout<<"-----> test_case_cell_assignment : CELL NEW = "<<offset(msh,cl_new)<<std::endl;
//        cl = cl_new ;
//        cl_pt = & cl;
//        //msh = msh ;
//        std::cout<<"----------------> test_case_cell_assignment : CELL NEW = "<<offset(msh,cl_new)<< " and CELL UPLOADED = "<<offset(msh,cl)<<std::endl;
//        std::cout<<"----------------> test_case_cell_assignment : CELL NEW PUNTATORE = "<<offset(msh,*cl_pt)<<std::endl;
//        i+=8;
//        //cl_pointer = std::make_shared<typename Mesh::cell_type>(cl_new);
//        //std::cout<<"----------------> test_case_cell_assignment : CELL POINTER = "<<offset(msh, cl_pointer.get())<< " and CELL UPLOADED = "<<offset(msh,cl)<<std::endl;
//        //level_set__.cell_assignment(cl_new);
//    }
    
    void test_case_cell_assignment(const typename Mesh::cell_type& cl_new )
    {
        m_cl = cl_new ;
    }
    
    void refresh_lambdas(Function & level_set__, params<T> parms_, bool sym_grad){
        
       this->neumann_jump = [level_set__,sym_grad,this](const typename Mesh::point_type& pt) mutable -> Eigen::Matrix<T, 2, 1> {/* Neu */
                    Matrix<T, 2, 1> ret;
                    if(sym_grad)
                    {
                        T gamma = 1.0;
                        //T k = 1.0;
                        //T R = 1.0/3.0;
                        //cl = cl_pointer.get() ;
                        //std::cout<<"SONO  IN NEUMANN CONDITION."<<std::endl;
        //                std::cout<<"i = "<<i<<std::endl;
                        //auto cl_uploaded = this->cl ;
        //                i++;
        //                std::cout<<"i = "<<i<<std::endl;
        //                Mesh msh_prova = *msh_pt ;
        //                typename Mesh::cell_type cl_prova = *cl_pt ;
                        
                        
        //                std::cout<<"-----> test_case_eshelby_2_prova : CELL PROVA = "<<offset(msh_prova,cl_prova)<<std::endl;
                        //if(i==3)
                        //   this->cl = msh.cells[227];
                        //std::cout<<"-----> test_case_eshelby_2_prova : CELL = "<<offset(m_msh,m_cl)<<std::endl;
                        /*
                        auto cl_new = this->cl ;
                        std::cout<<"-----> test_case_eshelby_2_prova : CELL = "<<offset(msh,cl_new)<<std::endl;
                        
                        auto cl_new2 = upload_cl();
                        std::cout<<"-----> test_case_eshelby_2_prova : CELL = "<<offset(msh,cl_new2)<<std::endl;
                        
                        auto cl_new3 = upload_cl2();
                        std::cout<<"-----> test_case_eshelby_2_prova : CELL = "<<offset(msh,cl_new3)<<std::endl;
                        */
                        level_set__.cell_assignment(m_cl);
                        //T H = level_set__.normal(pt)
                        
                        ret(0) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                        ret(1) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);
                       
                        
                        //ret(0) = - k / R * level_set__.normal(pt)(0);
                        //ret(1) = - k / R * level_set__.normal(pt)(1);
                        
                    }
                    else
                    {
                        T gamma = 1.0;
                        level_set__.cell_assignment(m_cl);
                        ret(0) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                        ret(1) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);
                    }
           return ret;
       };
        
    }
    
    void test_case_mesh_assignment(const Mesh& msh_new )
    {
       
        std::cout<<"-----> test_case_mesh_assignment "<<std::endl;
        m_msh = msh_new ;
//        msh_pt = & msh ;
        
    }
    
    typename Mesh::cell_type& upload_cl()
    {
        return m_cl ;
    }
    
    typename Mesh::cell_type upload_cl2()
    {
        return m_cl ;
    }
    

    
};

template<typename Mesh, typename T, typename Function>
auto make_test_case_eshelby_2_prova(const Mesh& msh, Function& level_set_function, params<T> parms_, bool sym_grad)
{
   return test_case_eshelby_2_prova<typename Mesh::coordinate_type, Mesh , Function>(level_set_function,parms_,sym_grad);
}



template<typename T, typename Mesh, typename Function >
class test_case_eshelby_analytic: public test_case_stokes<T, Function , Mesh>
{
  public:
   test_case_eshelby_analytic(const Function & level_set__, params<T> parms_, bool sym_grad , T radius)
       : test_case_stokes<T, Function , Mesh>
       (level_set__, parms_,
        [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> {
           // sol_vel
           Matrix<T, 2, 1> ret;
          
           ret(0) = 0.0;
           ret(1) = 0.0;
           
           return ret;},
        [level_set__](const typename Mesh::point_type& pt) -> T { // p
          
           return 0.0;
           
       },
        [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> { // rhs
            Matrix<T, 2, 1> ret;
            ret(0) = 0.0 ;
            ret(1) = 0.0 ;
            return ret;},
        [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> { // bcs
           Matrix<T, 2, 1> ret;
           
           ret(0) = 0.0;
           ret(1) = 0.0;
           return ret;},
        [](const typename Mesh::point_type& pt) -> auto { // grad
           
           Matrix<T, 2, 2> ret;
           ret(0,0) = 0.0;
           ret(0,1) = 0.0;
           ret(1,0) = 0.0;
           ret(1,1) = (0.0);
           return ret;},
        [](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> {/* Dir */
            Matrix<T, 2, 1> ret;
            ret(0) = 0.0;
            ret(1) = 0.0;
            return ret;},
        [level_set__,parms_,sym_grad,radius](const typename Mesh::point_type& pt) -> Eigen::Matrix<T, 2, 1> {/* Neu */
            Matrix<T, 2, 1> ret;
            if(sym_grad)
            {
                T gamma = 1.0;
                //T k = 1.0;
                //T R = 1.0/3.0;
                
                //T H = level_set__.normal(pt)
                //ret(0) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(0);
                //ret(1) = 2.0 * gamma * level_set__.divergence(pt) * level_set__.normal(pt)(1);
                
                ret(0) = gamma * radius * level_set__.normal(pt)(0);
                       
                ret(1) = gamma * radius * level_set__.normal(pt)(1);
                
                //ret(0) = - k / R * level_set__.normal(pt)(0);
                //ret(1) = - k / R * level_set__.normal(pt)(1);
                
            }
            else
            {
                T gamma = 1.0;
                            
                ret(0) =  gamma * radius * level_set__.normal(pt)(0);
                ret(1) =  gamma * radius * level_set__.normal(pt)(1);
            }
            return ret;})
       {}
};




template<typename Mesh, typename T, typename Function>
auto make_test_case_eshelby_analytic(const Mesh& msh, const Function& level_set_function, params<T> parms_, bool sym_grad, T radius)
{
   return test_case_eshelby_analytic<typename Mesh::coordinate_type, Mesh , Function>(level_set_function,parms_,sym_grad,radius);
}

