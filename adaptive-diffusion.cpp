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
            //result = bicg.solveWithGuess(source,result);
            result = bicg.solve(source);
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
    double dt_aim,c0,h0_factor,k;
    unsigned int solver_in;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("max_iter_linear", po::value<unsigned int>(&max_iter_linear)->default_value(40), "maximum iterations for linear solve")
        ("restart_linear", po::value<unsigned int>(&restart_linear)->default_value(20), "iterations until restart for linear solve")
        ("linear_solver", po::value<unsigned int>(&solver_in)->default_value(1), "linear solver")
        ("nout", po::value<unsigned int>(&nout)->default_value(10), "number of output points")
        ("k", po::value<double>(&k)->default_value(1.0), "spring constant")
        ("nx", po::value<unsigned int>(&nx)->default_value(10), "nx")
        ("h0_factor", po::value<double>(&h0_factor)->default_value(1.5), "h0 factor")
        ("dt", po::value<double>(&dt_aim)->default_value(0.01), "timestep")
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
    ABORIA_VARIABLE(kernel_test,double,"kernel")
    ABORIA_VARIABLE(laplace_kernel_test,double,"laplace kernel")
    ABORIA_VARIABLE(velocity,double,"velocity")
    ABORIA_VARIABLE(gradient,double2,"gradient")
    ABORIA_VARIABLE(change_in_position,double2,"change in position")
    ABORIA_VARIABLE(temperature_weights,double,"temperature weights")
    ABORIA_VARIABLE(kernel_constant,double,"kernel constant")
    ABORIA_VARIABLE(resting_separation,double,"resting separation")

    typedef Particles<std::tuple<temperature,kernel_test,laplace_kernel_test,velocity,temperature_weights,boundary,kernel_constant,gradient,change_in_position>,2> ParticlesType;
    typedef position_d<2> position;
    ParticlesType knots;

    const double Tf = 1.0;
    const int max_iter = 100;
    const int restart = 100;
    double2 periodic(false);

    const double L = 1.0;
    const double h0 = h0_factor*L/nx;
    const double delta = L/nx;
    typename ParticlesType::value_type p;
    const double id_theta = 0.2;
    const double id_alpha = 30.0;
    for (int i=-1; i<=(int)nx+1; ++i) {
        for (int j=-1; j<=(int)nx+1; ++j) {
            get<position>(p) = double2(i*delta,j*delta);
            if ((i<=0)||(i>=nx)||(j<=0)||(j>=nx)) {
                get<boundary>(p) = true;
                if (i<=0) {
                    get<temperature>(p) = 0.5*(tanh(id_alpha*(get<position>(p)[1]-id_theta))+tanh(id_alpha*(1-id_theta-get<position>(p)[1])));
                    get<temperature>(p) = 0;
                } else {
                    get<temperature>(p) = 0;
                }
            } else {
                get<boundary>(p) = false;
                get<temperature>(p) = 0;
            }
            get<kernel_constant>(p) = h0;
            get<temperature_weights>(p) = 0;
            knots.push_back(p);
        }
    }


    knots.init_neighbour_search(double2(0-L/2),double2(L+L/2),2*h0,bool2(false));
    std::cout << "added "<<knots.size()<<" knots with h0 = " <<h0<< std::endl;

    Symbol<boundary> is_b;
    Symbol<position> r;
    Symbol<kernel_constant> h;
    Symbol<kernel_test> ktest;
    Symbol<laplace_kernel_test> lktest;
    Symbol<gradient> g;
    Symbol<change_in_position> dr;
    Symbol<temperature> u;
    Symbol<temperature_weights> w;
    Symbol<velocity> drdt;
    Symbol<resting_separation> d;
    Label<0,ParticlesType> a(knots);
    Label<1,ParticlesType> b(knots);
    auto dx = create_dx(a,b);
    Accumulate<std::plus<double> > sum;
    Accumulate<std::plus<double2> > sumv;

    auto kernel = deep_copy(
            //exp(-4*dot(dx,dx)/pow(h[a]+h[b],2))
            pow(h[a]+h[b]-norm(dx),4)*(16*(h[a]+h[b]) + 64*norm(dx))/pow(h[a]+h[b],5)
            );

    auto laplace_kernel = deep_copy(
            //(-16*pow(h[a]+h[b],2)+64*dot(dx,dx))*exp(-4*dot(dx,dx)/pow(h[a]+h[b],2))/pow(h[a]+h[b],4)
            if_else(dot(dx,dx)==0.0,
            -640/pow(h[a]+h[b],2),
            65536.0*pow(h[a]+h[b]-norm(dx),2)*(0.001953125*pow(dx[0],2)*(-2.5*(h[a]+h[b]) + 10.0*norm(dx)) + 0.0048828125*pow(dx[0],2)*(h[a]+h[b]-norm(dx)) + 0.001953125*pow(dx[1],2)*(-2.5*(h[a]+h[b])+10.0*norm(dx)) + 0.0048828125*pow(dx[1],2)*(h[a]+h[b]-norm(dx)) - 0.009765625*dot(dx,dx)*(h[a]+h[b]-norm(dx)))/(pow(h[a]+h[b],5)*dot(dx,dx))

            )
            );

    auto adapt_kernel = deep_copy(
            320.0*pow(h[a]+h[b]-norm(dx),2)*abs(h[a]+h[b]-norm(dx))*norm(dx)
            );

    auto gradient_kernel = deep_copy(
            320.0*dx*pow(h[a]+h[b]-norm(dx),3)/pow(h[a]+h[b],5)
            );

    auto force_kernel = deep_copy(
            if_else(dot(dx,dx)==0,
                    0,
                    (-k*(h[a]+h[b]-norm(dx))/norm(dx))
                )*dx
            );

    auto forcing = deep_copy(
            if_else(pow(r[a][0]-0.5,2) + pow(r[a][1]-0.5,2) < 0.04,
                10000,0)
            );

    
    typedef Eigen::Matrix<double,Eigen::Dynamic,1> vector_type; 
    typedef Eigen::Map<vector_type> map_type;


    get<temperature_weights>(knots)[std::pow(nx+1,2)/2] = 1.0;
    ktest[a] = sum(b,norm(dx) < h[a]+h[b],kernel*w[b]);
    lktest[a] = sum(b,norm(dx) < h[a]+h[b],laplace_kernel*w[b]);
    get<temperature_weights>(knots)[std::pow(nx+1,2)/2] = 0.0;


#ifdef HAVE_VTK
    vtkWriteGrid("init",1,knots.get_grid(true));
#endif

    const int timesteps = Tf/dt_aim;
    const double dt = Tf/timesteps;
    const double dt_adapt_aim = (1.0/100.0)*PI/sqrt(2*k);
    const int timesteps_adapt = std::ceil(dt/dt_adapt_aim);
    const double dt_apapt = dt/timesteps_adapt;
    std::cout << "doing "<<timesteps_adapt<<" adaptive timesteps"<<std::endl;
    CHECK(timesteps_adapt==1,"assuming one adaptive timestep")

    // semi-implicit euler step
    // dKdt u + Kdudt = dt*k_lu u
    // dt*dKdt u_n-1 + Ku_n - Ku_n-1 = dt*K_lu_n
    // (K-dt*K_l)*u_n =  K*u_n-1 - dt*dKdt u_n-1
    // A*u_n =  K*u_n-1 - dt*dKdt u_n-1
    // A*u_n =  (K-dt*dKdt)*u_n-1
    // A*u_n =  B*u_n-1
    // dKdt = dKdx * dxdt + dKdy * dydt
    auto B = deep_copy(
                if_else(is_b[a],
                   kernel,
                   kernel + dt*dot(gradient_kernel,force_kernel)
                )
            );

    // fully-implicit euler step
    // dKdt u + Kdudt = dt*k_lu u
    // dt*dKdt u_n + Ku_n - Ku_n-1 = dt*K_lu_n
    // (dt*dKdt + K-dt*K_l)*u_n =  K*u_n-1
    // C*u_n =  K*u_n-1
    // dKdt = dKdx * dxdt + dKdy * dydt
    auto C = create_eigen_operator(a,b,
                if_else(is_b[a],
                   kernel,
                   kernel - dt*(laplace_kernel + dot(gradient_kernel,force_kernel))
                )
            );

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

    auto t0 = Clock::now();
    std::cout << "dt = " <<dt<< std::endl;
    for (int i=0; i < timesteps; i++) {
        auto t1 = Clock::now();
        std::chrono::duration<double> dt_timestep = t1 - t0;
        t0 = Clock::now();
        std::cout << "timestep "<<i<<"/"<<timesteps<<". Will finish in "<<dt_timestep.count()*(timesteps-i)/std::pow(60,2)<<" hours"<<std::endl;

        solve(A,map_type(get<temperature_weights>(knots).data(),knots.size()),
                map_type(get<temperature>(knots).data(),knots.size()),
                max_iter_linear,restart_linear,(linear_solver)solver_in);

        g[a] = sumv(b,norm(dx)<h[a]+h[b],gradient_kernel*w[b]);
        h[a] = h0/(1+0.01*norm(g[a]));
        r[a] += if_else(is_b[a],0,1)
                        *dt*sumv(b,norm(dx)<h[a]+h[b],force_kernel);

        u[a] = sum(b,norm(dx)<h[a]+h[b],B*w[b]) + dt*forcing;

        #ifdef HAVE_VTK
        vtkWriteGrid("explicit",i,knots.get_grid(true));
#endif
        
    }
}

