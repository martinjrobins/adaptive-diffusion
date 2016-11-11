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
            Eigen::ConjugateGradient<Kernel, Eigen::Lower|Eigen::Upper, Eigen::DiagonalPreconditioner<double>> cg;
            cg.setMaxIterations(max_iter);
            cg.compute(kernel);
            result = cg.solveWithGuess(source,result);
            std::cout << "CG:    #iterations: " << cg.iterations() << ", estimated error: " << cg.error() << std::endl;
            break;
                 }
        case BiCGSTAB: {
            Eigen::BiCGSTAB<Kernel, Eigen::DiagonalPreconditioner<double>> bicg;
            bicg.setMaxIterations(max_iter);
            bicg.compute(kernel);
            result = bicg.solveWithGuess(source,result);
            std::cout << "BiCGSTAB:    #iterations: " << bicg.iterations() << ", estimated error: " << bicg.error() << std::endl;
            break;
               }
        case GMRES: {
            Eigen::GMRES<Kernel, Eigen::DiagonalPreconditioner<double>> gmres;
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
    const int max_iter = 100;
    const int restart = 100;
    double2 periodic(false);

    constexpr int N = (nx+1)*(nx+1);
    const double delta = 1.0/nx;
    typename ParticlesType::value_type p;
    for (int i=0; i<=nx; ++i) {
        for (int j=0; j<=nx; ++j) {
            get<position>(p) = double2(i*delta,j*delta);
            if ((i==0)||(i==nx)||(j==0)||(j==nx)) {
                get<boundary>(p) = true;
                if (i==0) {
                    get<temperature> = 1;
                } else {
                    get<temperature> = 0;
                }

            } else {
                get<boundary>(p) = false;
                get<temperature> = 0;
            }
            get<kernel_constant>(p) = h0;
            knots.push_back(p);
        }
    }

    Symbol<boundary> is_b;
    Symbol<position> r;
    Symbol<kernel_constant> h;
    Symbol<temperature> u;
    Symbol<temperature_weights> w;
    Symbol<temperature_change> dudt;
    Symbol<velocity> drdt;
    Symbol<resting_separation> d;
    Label<0,ParticlesType> a(knots);
    Label<1,ParticlesType> b(knots);
    auto dx = create_dx(a,b);
    Accumulate<std::plus<double> > sum;

    auto kernel = deep_copy(
            //exp(-pow(norm(dx),2)/c2[b])
            //sqrt(pow(norm(dx),2) + c2[b])
            pow(h[a]+h[b]-dx.norm(),4)*(16*(h[a]+h[b]) + 64*dx.norm())/pow(h[a]+h[b],5)
            );

    auto laplace_kernel = deep_copy(
            //(2*c2[b] + pow(norm(dx),2)) / pow(pow(norm(dx),2) + c2[b],1.5)
            //4*(pow(norm(dx),2) - c2[b]) * exp(-pow(norm(dx),2)/c2[b])/pow(c2[a],2)
            65536*pow(h[a]+h[b]-dx.norm(),2)*(0.001953125*pow(dx[0],2)*(-2.5*(h[a]+h[b]) + 10*dx.norm()) + 0.0048828125*pow(dx[0],2)*(h[a]+h[b]-dx.norm()) + 0.001953125*pow(dx[1],2)*(-2.5*(h[a]+h[b])+10*dx.norm()) + 0.0048828125*pow(dx[1],2)*(h[a]+h[b]-dx.norm()) - 0.009765625*(pow(dx.norm(),2))*(h[a]+h[b]-dx.norm()))
            );

    auto force_kernel = deep_copy(
            (-k*(d[a]+d[b]-dx.norm())/dx.norm())*dx
            );
    
    typedef Eigen::Matrix<double,Eigen::Dynamic,1> vector_type; 
    typedef Eigen::Map<vector_type> map_type;

 
    solve(create_eigen_operator(a,b,kernel),
            map_type(get<temperature_weights>(knots).data(),knots.size()),
            map_type(get<temperature>(knots).data(),knots.size()),max_iter_linear,restart_linear,(linear_solver)solver_in);
    
#ifdef HAVE_VTK
    vtkWriteGrid("init",1,knots.get_grid(true));
#endif

    auto t0 = Clock::now();
    for (int i=0; i < timesteps; i++) {
        auto t1 = Clock::now();
        std::chrono::duration<double> dt_timestep = t1 - t0;
        t0 = Clock::now();
        std::cout << "timestep "<<i<<"/"<<timesteps<<". Will finish in "<<dt_timestep.count()*(timesteps-i)/std::pow(60,2)<<" hours"<<std::endl;
        //evaluate temporaries
        dudt[a] = if_else(is_b[b],
                    0,
                    sum(b,dx.norm()<2*h,laplace_kernel*w[b])
                    );
        drdt[a] = if_else(is_b[a],
                    0,
                    sum(b,dx.norm()<d[a]+d[b],force_kernel)
                    );
                    

        //implicit euler step
        // Ku_n - Ku_n-1 = dt*K_lu_n
        // Ku_n - dt*K_lu_n =  Ku_n-1
        // (K-dt*K_l)*u_n =  K*u_n-1
        r[a] += dt*drdt[a];
        u[a] += dt*dudt[a];

        solve(create_eigen_operator(a,b,kernel),
            map_type(get<temperature_weights>(knots).data(),knots.size()),
            map_type(get<temperature>(knots).data(),knots.size()),max_iter_linear,restart_linear,(linear_solver)solver_in);
    
#ifdef HAVE_VTK
        vtkWriteGrid("explicit",i,knots.get_grid(true));
#endif
        
    }
}

