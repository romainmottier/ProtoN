
#ifndef PrintIntegrationRule_hpp
#define PrintIntegrationRule_hpp

template<typename Mesh>
void PrintIntegrationRule(const Mesh& msh, hho_degree_info & hdi);


template<typename Mesh>
void PrintIntegrationRule(const Mesh& msh, hho_degree_info & hdi){
    
    std::ofstream int_rule_file("cut_integration_rule.txt");
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
            
            auto qps_n = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                int_rule_file << qp.first.x() << " " << qp.first.y() << std::endl;
            }
            
            
            auto qps_p = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                int_rule_file << qp.first.x() << " " << qp.first.y() << std::endl;
            }
        }
        else
        {

            auto qps = integrate(msh, cl, 2*hdi.cell_degree());
            for (auto& qp : qps)
            {
                int_rule_file << qp.first.x() << " " << qp.first.y() << std::endl;
            }
        }
    }
    int_rule_file.flush();
}


#endif
