
#ifndef PrintAgglomeratedCells_hpp
#define PrintAgglomeratedCells_hpp

template<typename Mesh>
void PrintAgglomeratedCells(const Mesh& msh);

template<typename Mesh>
void PrintAgglomeratedCells(const Mesh& msh){
    
    std::ofstream agglo_cells_file("agglomerated_cells.txt");
    for (auto& cl : msh.cells)
    {

        if (location(msh, cl) == element_location::ON_INTERFACE)
        {
            auto pts = points(msh, cl);
            if (pts.size() == 4) {
                continue;
            }
            for (auto point : pts) {
                agglo_cells_file << " ";
                agglo_cells_file << point.x() << " " << point.y();
            }
            agglo_cells_file << std::endl;
        }
    }
    agglo_cells_file.flush();
}

#endif
