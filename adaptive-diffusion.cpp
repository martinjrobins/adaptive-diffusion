#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#define BOOST_MPL_LIMIT_VECTOR_SIZE 40
#include "Aboria.h"
using namespace Aboria;

#include <boost/math/constants/constants.hpp>
#include "boost/program_options.hpp" 
namespace po = boost::program_options;
#include <chrono>
typedef std::chrono::system_clock Clock;

const double PI = boost::math::constants::pi<double>();
enum linear_solver {CG, BiCGSTAB, GMRES};

template<typename Kernel,typename VectorType>
void solve(Kernel &&kernel, VectorType &&result, VectorType &&source, size_t max_iter=10, size_t restart=10, linear_solver solver=CG) {
    switch (solver) {
        case CG: {
            Eigen::ConjugateGradient<
                typename std::remove_reference<Kernel>::type, 
                         Eigen::Lower|Eigen::Upper, Eigen::DiagonalPreconditioner<double>> cg;
            cg.setMaxIterations(max_iter);
            cg.compute(kernel);
            result = cg.solveWithGuess(source,result);
            std::cout << "CG:    #iterations: " << cg.iterations() << ", estimated error: " << cg.error() << std::endl;
            break;
                 }
        case BiCGSTAB: {
            Eigen::BiCGSTAB<
                typename std::remove_reference<Kernel>::type, 
                     Eigen::DiagonalPreconditioner<double>> bicg;
            bicg.setMaxIterations(max_iter);
            bicg.compute(kernel);
            result = bicg.solveWithGuess(source,result);
            std::cout << "BiCGSTAB:    #iterations: " << bicg.iterations() << ", estimated error: " << bicg.error() << std::endl;
            break;
               }
        case GMRES: {
            Eigen::GMRES<
                typename std::remove_reference<Kernel>::type, 
                    Eigen::DiagonalPreconditioner<double>> gmres;
            gmres.set_restart(restart);
            gmres.setMaxIterations(max_iter);
            gmres.compute(kernel);
            result = gmres.solveWithGuess(source,result);
            std::cout << "GMRES:    #iterations: " << gmres.iterations() << ", estimated error: " << gmres.error() << std::endl;
            break;
                    }
    }

}

int main(int argc, char **argv) {

    unsigned int nout,max_iter_linear,restart_linear,nx,nr,nmiddle,ngrid;
    double dt_aim,c0,factorr;
    unsigned int solver_in;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("max_iter_linear", po::value<unsigned int>(&max_iter_linear)->default_value(20), "maximum iterations for linear solve")
        ("restart_linear", po::value<unsigned int>(&restart_linear)->default_value(20), "iterations until restart for linear solve")
        ("linear_solver", po::value<unsigned int>(&solver_in)->default_value(0), "linear solver")
        ("nout", po::value<unsigned int>(&nout)->default_value(10), "number of output points")
        ("c0", po::value<double>(&c0)->default_value(1.0), "kernel constant")
        ("nx", po::value<unsigned int>(&nx)->default_value(10), "nx")
        ("nmiddle", po::value<unsigned int>(&nmiddle)->default_value(1), "nx")
        ("ngrid", po::value<unsigned int>(&ngrid)->default_value(10), "ngrid")
        ("nr", po::value<unsigned int>(&nr)->default_value(10), "nr")
        ("factorr", po::value<double>(&factorr)->default_value(1.5), "factorr")
        ("dt", po::value<double>(&dt_aim)->default_value(0.0001), "timestep")
    ;
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);  

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }
    

    ABORIA_VARIABLE(boundary,uint8_t,"is boundary knot")
    ABORIA_VARIABLE(temperature,double,"temperature")
    ABORIA_VARIABLE(temperature_change,double,"temperature change")
    ABORIA_VARIABLE(velocity,double,"velocity")
    ABORIA_VARIABLE(temperature_weights,double,"temperature weights")
    ABORIA_VARIABLE(kernel_constant,double,"kernel constant")
    ABORIA_VARIABLE(resting_separation,double,"resting separation")

    typedef Particles<std::tuple<temperature,temperature_change,velocity,temperature_weights,boundary,kernel_constant,resting_separation>,2> ParticlesType;
    typedef position_d<2> position;
    ParticlesType knots;

    const double k = 0.5;
    const double Tf = 10.0;
    const int max_iter = 100;
    const int restart = 100;
    double2 periodic(false);

    const double L = 1.0;
    const double h0 = 2.0*L/nx;
    const double delta = 1.0/nx;
    typename ParticlesType::value_type p;
    const double id_theta = 0.2;
    const double id_alpha = 30.0;
    for (int i=0; i<=nx; ++i) {
        for (int j=0; j<=nx; ++j) {
            get<position>(p) = double2(i*delta,j*delta);
            if ((i==0)||(i==nx)||(j==0)||(j==nx)) {
                get<boundary>(p) = true;
                if (i==0) {
                    get<temperature>(p) = 0.5*(tanh(id_alpha*(get<position>(p)[1]-id_theta))+tanh(id_alpha*(1-id_theta-get<position>(p)[1])));
                } else {
                    get<temperature>(p) = 0;
                }
            } else {
                get<boundary>(p) = false;
                get<temperature>(p) = 0;
            }
            get<kernel_constant>(p) = h0;
            knots.push_back(p);
        }
    }


    knots.init_neighbour_search(double2(0-L/10),double2(L+L/10),2*h0,bool2(false));
    std::cout << "added "<<knots.size()<<" knots" << std::endl;

    Symbol<boundary> is_b;
    Symbol<position> r;
    Symbol<kernel_constant> h;
    Symbol<temperature> u;
    Symbol<temperature_weights> w;
    Symbol<velocity> drdt;
    Symbol<resting_separation> d;
    Label<0,ParticlesType> a(knots);
    Label<1,ParticlesType> b(knots);
    auto dx = create_dx(a,b);
    Accumulate<std::plus<double> > sum;

    auto kernel = deep_copy(
            //exp(-4*dot(dx,dx)/pow(h[a]+h[b],2))
            pow(h[a]+h[b]-norm(dx),4)*(16*(h[a]+h[b]) + 64*norm(dx))/pow(h[a]+h[b],5)
            );

    auto laplace_kernel = deep_copy(
            //(-16*pow(h[a]+h[b],2)+64*dot(dx,dx))*exp(-4*dot(dx,dx)/pow(h[a]+h[b],2))/pow(h[a]+h[b],4)
            65536.0*pow(h[a]+h[b]-norm(dx),2)*(0.001953125*pow(dx[0],2)*(-2.5*(h[a]+h[b]) + 10.0*norm(dx)) + 0.0048828125*pow(dx[0],2)*(h[a]+h[b]-norm(dx)) + 0.001953125*pow(dx[1],2)*(-2.5*(h[a]+h[b])+10.0*norm(dx)) + 0.0048828125*pow(dx[1],2)*(h[a]+h[b]-norm(dx)) - 0.009765625*(pow(norm(dx),2))*(h[a]+h[b]-norm(dx)))
            );

    auto force_kernel = deep_copy(
            (-k*(d[a]+d[b]-norm(dx))/norm(dx))*dx
            );
    
    typedef Eigen::Matrix<double,Eigen::Dynamic,1> vector_type; 
    typedef Eigen::Map<vector_type> map_type;

#ifdef HAVE_VTK
    vtkWriteGrid("init",1,knots.get_grid(true));
#endif

    const int timesteps = Tf/dt_aim;
    const double dt = Tf/timesteps;
    auto t0 = Clock::now();
    for (int i=0; i < timesteps; i++) {
        auto t1 = Clock::now();
        std::chrono::duration<double> dt_timestep = t1 - t0;
        t0 = Clock::now();
        std::cout << "timestep "<<i<<"/"<<timesteps<<". Will finish in "<<dt_timestep.count()*(timesteps-i)/std::pow(60,2)<<" hours"<<std::endl;

        //implicit euler step
        // Ku_n - Ku_n-1 = dt*K_lu_n
        // Ku_n - dt*K_lu_n =  Ku_n-1
        // (K-dt*K_l)*u_n =  K*u_n-1
        // A*u_n =  K*u_n-1
        auto A = create_eigen_operator(a,b,
                    if_else(is_b[a],
                       kernel,
                       kernel - dt*laplace_kernel
                    )
                    ,norm(dx) < h[a]+h[b]
                );
        solve(A,map_type(get<temperature_weights>(knots).data(),knots.size()),
                map_type(get<temperature>(knots).data(),knots.size()),
                max_iter_linear,restart_linear,(linear_solver)solver_in);

        //u[a] = if_else(is_b[a],
        //            u[a],
        //            //sum(b,norm(dx)<h[a]+h[b],kernel*w[b])
        //            sum(b,true,kernel*w[b])
        //            );
        //u[a] = sum(b,true,kernel*w[b]);
        sum(b,norm(dx)<h[a]+h[b],kernel*w[b]);

#ifdef HAVE_VTK
        vtkWriteGrid("explicit",i,knots.get_grid(true));
#endif
        
    }
}

