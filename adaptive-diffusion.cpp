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
            //result = bicg.solve(source);
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
    double dt_aim,c0,h0_factor,k,gamma,force_a,force_b;
    unsigned int solver_in;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("max_iter_linear", po::value<unsigned int>(&max_iter_linear)->default_value(100), "maximum iterations for linear solve")
        ("restart_linear", po::value<unsigned int>(&restart_linear)->default_value(20), "iterations until restart for linear solve")
        ("linear_solver", po::value<unsigned int>(&solver_in)->default_value(1), "linear solver")
        ("nout", po::value<unsigned int>(&nout)->default_value(10), "number of output points")
        ("k", po::value<double>(&k)->default_value(0.1), "spring constant")
        ("force_a", po::value<double>(&force_a)->default_value(60), "force_a constant")
        ("force_b", po::value<double>(&force_b)->default_value(80), "force_b constant")
        ("k", po::value<double>(&k)->default_value(0.1), "spring constant")
        ("gamma", po::value<double>(&gamma)->default_value(0.44721359), "spring constant")
        ("nx", po::value<unsigned int>(&nx)->default_value(19), "nx")
        ("h0_factor", po::value<double>(&h0_factor)->default_value(4.0), "h0 factor")
        ("dt", po::value<double>(&dt_aim)->default_value(0.1), "timestep")
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
    ABORIA_VARIABLE(temperature_weights,double,"temperature weights")
    ABORIA_VARIABLE(kernel_constant,double,"kernel constant")
    ABORIA_VARIABLE(resting_separation,double,"resting separation")

    typedef Particles<std::tuple<temperature,temperature_weights,boundary,kernel_constant,resting_separation>,2> ParticlesType;
    typedef position_d<2> position;
    ParticlesType knots;

    const double Tf = 2.0;
    const double R = 0.5;
    const double delta = 2*R/nx;
    const double h0 = h0_factor*delta;
    const int timesteps = Tf/dt_aim;
    const double dt = Tf/timesteps;
    const double dt_adapt = (1.0/100.0)*PI/sqrt(2*k);


    typename ParticlesType::value_type p;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> uniform(-R,R);
    for (int i=0; i<nx*nx; ++i) {
        get<position>(p) = double2(uniform(generator),uniform(generator));
        get<boundary>(p) = false;
        if (get<position>(p).norm()<R) { 
            knots.push_back(p);
        }
    }

    const double dtheta_aim = delta/R;
    const int ntheta = 2*PI/dtheta_aim;
    const double dtheta = 2*PI/ntheta;
    for (int i=0; i<ntheta; ++i) {
        get<position>(p) = R*double2(cos(i*dtheta),sin(i*dtheta));
        get<boundary>(p) = true;
        knots.push_back(p);
    }

    knots.init_neighbour_search(double2(-1.5*R),double2(1.5*R),4*h0,bool2(false));
    std::cout << "added "<<knots.size()<<" knots with h0 = " <<h0<< std::endl;

    Symbol<boundary> is_b;
    Symbol<position> r;
    Symbol<kernel_constant> h;
    Symbol<temperature> u;
    Symbol<temperature_weights> w;
    Symbol<resting_separation> s;
    Label<0,ParticlesType> a(knots);
    Label<1,ParticlesType> b(knots);
    auto dx = create_dx(a,b);
    Accumulate<std::plus<double> > sum;
    Accumulate<std::plus<double2> > sumv;
    Accumulate<Aboria::max<double> > max;
    max.set_init(0);
    VectorSymbolic<double,2> vector;      

    auto kernel = deep_copy(
        pow(h[a]+h[b]-norm(dx),4)*(16.0*(h[a]+h[b]) + 64.0*norm(dx))/pow(h[a]+h[b],5)
        );

    auto gradient_kernel = deep_copy(
        -320.0*dx*pow(h[a]+h[b]-norm(dx),3)/pow(h[a]+h[b],5)
        );

    auto laplace_kernel = deep_copy(
        //if_else(dot(dx,dx)==0.0
            //,-2.0*320.0/pow(h[a]+h[b],2)
            //,65536.0*pow(h[a]+h[b]-norm(dx),2)*(0.001953125*pow(dx[0],2)*(-2.5*(h[a]+h[b]) + 10.0*norm(dx)) + 0.0048828125*pow(dx[0],2)*(h[a]+h[b]-norm(dx)) + 0.001953125*pow(dx[1],2)*(-2.5*(h[a]+h[b])+10.0*norm(dx)) + 0.0048828125*pow(dx[1],2)*(h[a]+h[b]-norm(dx)) - 0.009765625*dot(dx,dx)*(h[a]+h[b]-norm(dx)))/(pow(h[a]+h[b],5)*dot(dx,dx))
            pow(h[a]+h[b]-norm(dx),2)/pow(h[a]+h[b],5)*(128.0*(-2.5*(h[a]+h[b]) + 10.0*norm(dx)) - 320.0*(h[a]+h[b]-norm(dx)))

            //,65536.0*pow(h[a]+h[b]-norm(dx),2)*(0.001953125*(-2.5*(h[a]+h[b]) + 10.0*norm(dx)) - 0.0048828125*(h[a]+h[b]-norm(dx)))/(pow(h[a]+h[b],5))
            //,320.0*pow(h[a]+h[b]-norm(dx),2)*(dot(dx,dx)*(-2.5*(h[a]+h[b]) + 10.0*norm(dx)))/(pow(h[a]+h[b],5)*dot(dx,dx))

            //)
        );

    auto force_kernel = deep_copy(
        if_else(dot(dx,dx)==0
            ,0
            ,(-k*((s[a]+s[b])-norm(dx))/norm(dx))*dx
            )
        );

    auto forcing = deep_copy(
        (-pow(force_a,2)*pow(2.0*r[a][0],2) + 2.0*force_a - pow(force_b,2)*pow(2.0*r[a][1],2) + 2*force_b)*exp(-force_a*pow(r[a][0],2)-force_b*pow(r[a][1],2))
        );

    auto solution_eval = deep_copy( 
        exp(-force_a*pow(r[a][0],2)-force_b*pow(r[a][1],2))
        );

    auto boundary_force = deep_copy(
        10*k*r[a]*if_else(norm(r[a])<R
                ,0
                ,R-norm(r[a]))/norm(r[a])
        );


#ifdef HAVE_VTK
    vtkWriteGrid("init",1,knots.get_grid(true));
#endif

    s[a] = delta;

    // adapt knot locations
    for (int j=0; j<1000; j++) {
        r[a] += dt_adapt*if_else(is_b[a],
                vector(0,0),
                sumv(b,norm(dx)<s[a]+s[b],force_kernel)
                + boundary_force
                );
    }

    w[a] = 0;
    h[a] = h0;
    u[a] = solution_eval;

    auto t0 = Clock::now();
    std::cout << "dt = " <<dt<< std::endl;
    for (int i=0; i < timesteps; i++) {
        auto t1 = Clock::now();
        std::chrono::duration<double> dt_timestep = t1 - t0;
        t0 = Clock::now();
        std::cout << "timestep "<<i<<"/"<<timesteps<<". Will finish in "<<dt_timestep.count()*(timesteps-i)/std::pow(60,2)<<" hours"<<std::endl;

        // add forcing term
        u[a] += if_else(is_b[a],0,dt*forcing);

        // implicit euler step
        // Ku_n - Ku_n-1 = dt*K_lu_n + dt * forcing_n
        // Ku_n - dt*K_lu_n =  Ku_n-1 + dt * forcing_n
        // (K-dt*K_l)*u_n =  K*u_n-1 + dt * forcing_n
        // A*u_n =  K*u_n-1 + dt * forcing_n
        auto A = create_eigen_operator(a,b,
                    if_else(is_b[a],
                       kernel,
                       kernel - dt*laplace_kernel
                    )
                    ,norm(dx) < h[a]+h[b] 
                );

        typedef Eigen::Matrix<double,Eigen::Dynamic,1> vector_type; 
        typedef Eigen::Map<vector_type> map_type;
        solve(A,map_type(get<temperature_weights>(knots).data(),knots.size()),
                map_type(get<temperature>(knots).data(),knots.size()),
                max_iter_linear,restart_linear,(linear_solver)solver_in);

        // calculate solution
        u[a] = sum(b,norm(dx)<h[a]+h[b],kernel*w[b]);

        // compare against true solution
        const double norm2 = eval(sqrt(sum(a,true,pow(solution_eval-u[a],2))));
        const double scale = eval(sqrt(sum(a,true,pow(solution_eval,2))));
        std::cout << "norm2 error is "<<norm2/scale<<std::endl;

        // write knots to file
        #ifdef HAVE_VTK
        vtkWriteGrid("explicit-random",i,knots.get_grid(true));
        #endif

        // adapt knot locations
        for (int j=0; j<100; j++) {
            r[a] += dt_adapt*if_else(is_b[a],
                            vector(0,0),
                            sumv(b,norm(dx)<s[a]+s[b],force_kernel)
                            - pow(delta,2)*sumv(b,norm(dx)<h[a]+h[b],gradient_kernel*w[b])
                            + boundary_force
                    );
        }
    }
}

