//
//  preprocessor.hpp
//  acoustics
//
//  Created by Omar Durán on 4/10/20.
//

#pragma once
#ifndef preprocessor_hpp
#define preprocessor_hpp

#include <getopt.h>

class simulation_data
{
    
public:
    
    size_t m_k_degree;
    
    size_t m_n_divs;
    
    bool m_hdg_stabilization_Q;
    
    bool m_scaled_stabilization_Q;
    
    bool m_sc_Q;
    
    size_t m_nt_divs;
    
    bool m_render_silo_files_Q;
    
    bool m_report_energy_Q;
    
    bool m_quadratic_function_Q;
    
    simulation_data() : m_k_degree(0), m_n_divs(0), m_hdg_stabilization_Q(false), m_scaled_stabilization_Q(false), m_sc_Q(false), m_nt_divs(0), m_render_silo_files_Q(false), m_report_energy_Q(false), m_quadratic_function_Q(false){
        
    }
    
    simulation_data(const simulation_data & other){
        
        m_k_degree              = other.m_k_degree;
        m_n_divs                = other.m_n_divs;
        m_hdg_stabilization_Q   = other.m_hdg_stabilization_Q;
        m_scaled_stabilization_Q = other.m_scaled_stabilization_Q;
        m_sc_Q                  = other.m_sc_Q;
        m_nt_divs               = other.m_nt_divs;
        m_render_silo_files_Q   = other.m_render_silo_files_Q;
        m_report_energy_Q       = other.m_report_energy_Q;
        m_quadratic_function_Q  = other.m_quadratic_function_Q;
    }

    const simulation_data & operator=(const simulation_data & other){
        
        // check for self-assignment
        if(&other == this){
            return *this;
        }
        
        m_k_degree              = other.m_k_degree;
        m_n_divs                = other.m_n_divs;
        m_hdg_stabilization_Q   = other.m_hdg_stabilization_Q;
        m_scaled_stabilization_Q = other.m_scaled_stabilization_Q;
        m_sc_Q                  = other.m_sc_Q;
        m_nt_divs               = other.m_nt_divs;
        m_render_silo_files_Q   = other.m_render_silo_files_Q;
        m_report_energy_Q       = other.m_report_energy_Q;
        m_quadratic_function_Q  = other.m_quadratic_function_Q;
        
        return *this;
    }
    
    ~simulation_data(){
        
    }
    
    void print_simulation_data(){
        std::cout << bold << red << "face degree : " << m_k_degree << reset << std::endl;
        std::cout << bold << red << "refinements : " << m_n_divs << reset << std::endl;
        std::cout << bold << red << "stabilization type ? : " << m_hdg_stabilization_Q << reset << std::endl;
        std::cout << bold << red << "scaled stabilization ? : " << m_scaled_stabilization_Q << reset << std::endl;
        std::cout << bold << red << "condensation  ? : " << m_sc_Q << reset << std::endl;
        std::cout << bold << red << "time refinements : " << m_nt_divs << reset << std::endl;
        std::cout << bold << red << "write silo files ? : " << m_render_silo_files_Q << reset << std::endl;
        std::cout << bold << red << "report energy file ? : " << m_report_energy_Q << reset << std::endl;
        std::cout << bold << red << "quadratic spatial function ? : " << m_quadratic_function_Q << reset << std::endl;
    }
    
};

class preprocessor {
    
public:
    
    

    static void PrintHelp()
    {
        std::cout <<
                "-k <int>:  Face polynomial degree: default 0\n"
                "-l <int>:  Number of uniform space refinements: default 0\n"
                "-s <0-1>:  Stabilization type 0 -> HHO, 1 -> HDG-like: default 0 \n"
                "-r <0-1>:  Scaled stabilization 0 -> without, 1 -> with: default 0 \n"
                "-n <int>:  Number of uniform time refinements: default 0\n"
                "-f <0-1>:  Write silo files: default 0\n"
                "-e <0-1>:  Report (time,energy) pairs: default 0\n"
                "-c <0-1>:  Static Condensation (implicit schemes): default 0 \n"
                "-help:     Show help\n";
        exit(1);
    }

    static simulation_data process_args(int argc, char** argv)
    {
        const char* const short_opts = "k:l:s:r:n:c:f:e:";
        const option long_opts[] = {
                {"degree", required_argument, nullptr, 'k'},
                {"xref", required_argument, nullptr, 'l'},
                {"stab", required_argument, nullptr, 's'},
                {"scal", required_argument, nullptr, 'r'},
                {"tref", required_argument, nullptr, 'n'},
                {"c", optional_argument, nullptr, 'c'},
                {"file", optional_argument, nullptr, 'f'},
                {"energy", optional_argument, nullptr, 'e'},
                {"help", no_argument, nullptr, 'h'},
                {nullptr, no_argument, nullptr, 0}
        };

        size_t k_degree = 0;
        size_t n_divs   = 0;
        size_t nt_divs   = 0;
        bool hdg_Q = false;
        bool scaled_Q = false;
        bool sc_Q = false;
        bool silo_files_Q = false;
        bool report_energy_Q = false;
        
        while (true)
        {
            const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

            if (-1 == opt)
                break;

            switch (opt)
            {
            case 'k':
                k_degree = std::stoi(optarg);
                break;

            case 'l':
                n_divs = std::stoi(optarg);
                break;

            case 's':
                hdg_Q = std::stoi(optarg);
                break;
                    
            case 'r':
                scaled_Q = std::stoi(optarg);
                break;
                    
            case 'n':
                nt_divs = std::stoi(optarg);
                break;
                    
            case 'c':
                sc_Q = std::stoi(optarg);
                break;
                    
            case 'f':
                silo_files_Q = std::stoi(optarg);
                break;

            case 'e':
                report_energy_Q = std::stoi(optarg);
                break;
                    

            case 'h': // -h or --help
            case '?': // Unrecognized option
            default:
                preprocessor::PrintHelp();
                break;
            }
        }
        
        // populating simulation data
        simulation_data sim_data;
        sim_data.m_k_degree = k_degree;
        sim_data.m_n_divs = n_divs;
        sim_data.m_hdg_stabilization_Q = hdg_Q;
        sim_data.m_scaled_stabilization_Q = scaled_Q;
        sim_data.m_sc_Q = sc_Q;
        sim_data.m_nt_divs = nt_divs;
        sim_data.m_render_silo_files_Q = silo_files_Q;
        sim_data.m_report_energy_Q = report_energy_Q;
        return sim_data;
    }
    
    static void PrintTestHelp()
    {
        std::cout <<
                "-k <int>:  Maximum Face polynomial degree: default 0\n"
                "-l <int>:  Maximum Number of uniform space refinements: default 0\n"
                "-s <0-1>:  Stabilization type 0 -> HHO, 1 -> HDG-like: default 0 \n"
                "-r <0-1>:  Scaled stabilization 0 -> without, 1 -> with: default 0 \n"
                "-q <0-1>:  Quadratic function type 0 -> non-polynomial, 1 -> quadratic: default 0 \n"
                "-f <0-1>:  Write silo files : default 0\n"
                "-c <0-1>:  Static Condensation (implicit schemes): default 0 \n"
                "-help:     Show help\n";
        exit(1);
    }
    
    static simulation_data process_convergence_test_args(int argc, char** argv)
    {
        const char* const short_opts = "k:l:s:r:c:q:f:";
        const option long_opts[] = {
                {"degree", required_argument, nullptr, 'k'},
                {"xref", required_argument, nullptr, 'l'},
                {"stab", required_argument, nullptr, 's'},
                {"scal", required_argument, nullptr, 'r'},
                {"file", optional_argument, nullptr, 'f'},
                {"qfunc", optional_argument, nullptr, 'q'},
                {"cond", optional_argument, nullptr, 'c'},
                {"help", no_argument, nullptr, 'h'},
                {nullptr, no_argument, nullptr, 0}
        };

        size_t k_degree = 0;
        size_t n_divs   = 0;
        bool hdg_Q = false;
        bool scaled_Q = false;
        bool sc_Q = false;
        bool quadratic_func_Q = false;
        bool silo_files_Q = false;
        
        while (true)
        {
            const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

            if (-1 == opt)
                break;

            switch (opt)
            {
            case 'k':
                k_degree = std::stoi(optarg);
                break;

            case 'l':
                n_divs = std::stoi(optarg);
                break;

            case 's':
                hdg_Q = std::stoi(optarg);
                break;
                    
            case 'r':
                scaled_Q = std::stoi(optarg);
                break;
                    
            case 'c':
                sc_Q = std::stoi(optarg);
                break;

            case 'q':
                    quadratic_func_Q = std::stoi(optarg);
                break;
                    
            case 'f':
                silo_files_Q = std::stoi(optarg);
                break;

            case 'h': // -h or --help
            case '?': // Unrecognized option
            default:
                preprocessor::PrintTestHelp();
                break;
            }
        }
        
        // populating simulation data
        simulation_data sim_data;
        sim_data.m_k_degree = k_degree;
        sim_data.m_n_divs = n_divs;
        sim_data.m_hdg_stabilization_Q = hdg_Q;
        sim_data.m_scaled_stabilization_Q = scaled_Q;
        sim_data.m_sc_Q = sc_Q;
        sim_data.m_render_silo_files_Q = silo_files_Q;
        sim_data.m_quadratic_function_Q = quadratic_func_Q;
        return sim_data;
    }
};

#endif /* preprocessor_hpp */
