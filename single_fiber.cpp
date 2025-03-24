#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <omp.h> // OpenMP header
#include <amgcl/backend/builtin.hpp>  // Use built-in backend instead of omp.hpp
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <iostream>
#include <tuple>
#include <amgcl/solver/gmres.hpp>
#include <string>
#include <sstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace std;

/************* Global variables ******************/
const double pi = 4 * atan(1);
// N is the numeber of cylinders
const int N = 5;
const double dt = 0.1;
const double T = 10;
const double E = 1e9;
const double d = 0.01;
const double l = 1;
const double Ia = pi* pow(d,4)/64;
const double K = E*Ia;
// Number of variables per cylinder 5 is for 2D 
const int vars = 5;
/*  Fluid parameters (for now uniform vertical velocity)*/
const double u = 0;
const double v = 1;  
const double omega = 0;
const double c1 = 1;
const double c2 = 1;

/**********************************************************************/
/* All the variables are with unit so in the future adimensionalize plz*/
/*************************************************************************/


void init(vector<double>* rx, vector<double>* ry, vector<double>* Xy, vector<double>* Xx, vector<double>* theta){
	
	for (int i=0; i < N; i++)
	{
		(*rx)[i] = l * (i - floor(N/2)); 
	
	}

}


int system_coefficients_first_half(Eigen::SparseMatrix<double>& A,vector<double> theta)
{
	/// This function includes also the center
        int indx = 0;
	for (int i = 0; i < (int)floor(N/2) + 1; i++)
        {
                for (int j = 0; j < vars; j++)
                {
                indx = i * vars + j;
                // Equation for momentum x
                if (j == 0)
                {
                        if (i == 0)
                        {
                                A.insert(indx, indx) = 1;
                                A.insert(indx, indx + vars + 1) = -dt/c1;
                        }


                        // The middle cylinder has a set of constraints that keeps it still
			// It has only the connectivity equations
                        else if (i == (int)floor(N/2))
                        {

                                A.insert(indx, indx) = 1;
			       // A.insert(indx, indx + 2) = 1; cos(theta)	


                        }
                        
			else
                        {
                                A.insert(indx, indx - 2) = 1;
                                A.insert(indx, indx + 1) = dt/c1;
                                A.insert(indx, indx + vars + 1) = -dt/c1;
                        }

                   cout << "- Matrix A row " << indx << endl;
                }
                // Equation for momentum y
                else if (j == 1)
                {
                         if (i == 0)
                         {
                                A.insert(indx, indx) = 1;
                                A.insert(indx, indx + vars + 1) = -dt/c1;
			 }

			 else if (i == (int)floor(N/2))
                        {

                                A.insert(indx, indx) = 1;
                               // A.insert(indx, indx + 1) = 1; sin(theta)
				return 0;	
                        }
                        else
                        {
                                A.insert(indx, indx - 2) = 1;
                                A.insert(indx, indx + 1) = dt/c1;
                                A.insert(indx, indx + vars + 1) = -dt/c1;
                        }

                         cout << "-- Matrix A row " << indx << endl;
                }
                // Equation for momentum theta (angular)
                else if (j == 2)
                {
                        if (i == 0)
                        {
                                A.insert(indx, indx) = 1 - 2*K*dt/c2;
                                A.insert(indx, indx + vars) = -l*dt/c2;
                                A.insert(indx, indx + vars - 1) = -l*dt/c2;
                                A.insert(indx, indx + vars - 2) = K*dt/c2;

                                cout << "--- Matrix A row " << indx << endl;
                        }
                
			else
                        {

                                A.insert(indx, indx - 2) = 1 - 2*K*dt/c2;
                                A.insert(indx, indx - 1) = -l*dt/c2;  // ADD THE COSINE TERM???
                                A.insert(indx, indx) = -l*dt/c2;        // ADD THE SINE TERM?

                                A.insert(indx, indx + vars - 2) = K*dt/c2;
                                A.insert(indx, indx + vars - 1)  = -l*dt/c2;
                                A.insert(indx, indx + vars)  = -l*dt/c2;

                                if (i == 1){
                                        A.insert(indx, indx - vars) = -K*dt/c2;
                                }
                                else {
                                        A.insert(indx, indx- 2 - vars) = -K*dt/c2;
                                }
                                cout << "--- Matrix A row " << indx << endl;
                        }
                }
                                // Connectivity equation x
                 else if (j == 3)
                        {
				if (i == 0){
                                        A.insert(indx, indx - 3) = 1;
                                        A.insert(indx, indx + vars - 5) = -1;         // ADD THE coSINE TERM?
                                        cout << "---- Matrix A row " << indx << endl;
                                }

				else
				{
					A.insert(indx, indx - 3 - 2) = 1;
                                        A.insert(indx, indx + vars - 5) = -1;         // ADD THE coSINE TERM?
                                        cout << "---- Matrix A row " << indx << endl;
				
				}
			}                                // Connectivity equation y
                  else if (j == 4)
                        {
				if (i == 0){
                                        A.insert(indx, indx - 3) = 1;           // ADD THE SINE TERM???
                                        A.insert(indx, indx + vars - 5) = -1;
                                        cout << "----- Matrix A row " << indx << endl;
                                }
				else
				{
                        		A.insert(indx, indx - 3 - 2) = 1;          // ADD THE SINE TERM???
                                        A.insert(indx, indx + vars - 5) = -1;
                                        cout << "----- Matrix A row " << indx << endl;
				
				}
			}	
                        
                	
        	}
	}
	return 1;
}


int system_coefficients_second_half(Eigen::SparseMatrix<double>& A,vector<double> theta)
{
        int indx = 0;
        for (int i = (int)floor(N/2) + 1; i < N; i++)
        {
                for (int j = 0; j < vars; j++)
                {
                indx = i * vars + j - 3;
                // Equation for momentum x
                if (j == 0)
                {
                        // The last cylinder has the boundary condition of X_i+1 = 0
                        // so it doesn't have the final term
                        if (i == N-1)
                        {
                                A.insert(indx, indx - 2) = 1;
                                A.insert(indx, indx + 1) = dt/c1;
                        }


                        else
                        {
                                A.insert(indx, indx - 2) = 1;
                                A.insert(indx, indx + 1) = dt/c1;
                                A.insert(indx, indx + vars + 1) = -dt/c1;
                        }

                   cout << "- Matrix A row " << indx << endl;
                }
                // Equation for momentum y
                else if (j == 1)
                {
                        if (i == N-1)
                        {
                                A.insert(indx, indx - 2) = 1;
                                A.insert(indx, indx + 1) = dt/c1;
                        }
                        
			else
                        {
                                A.insert(indx, indx - 2) = 1;
                                A.insert(indx, indx + 1) = dt/c1;
                                A.insert(indx, indx + vars + 1) = -dt/c1;
                        }

                         cout << "-- Matrix A row " << indx << endl;
                }
                // Equation for momentum theta (angular)
                else if (j == 2)
                {
                        // Last cylinder
                        if (i == N - 1)
                        {
                                A.insert(indx, indx - 2) = 1 - 2*K*dt/c2;
                                A.insert(indx, indx - 1) = -l*dt/c2;  // ADD THE COSINE TERM???
                                A.insert(indx, indx) = -l*dt/c2;  // ADD THE SINE TERM?
                                A.insert(indx, indx - 2 - vars) = -K*dt/c2;
                                cout << "--- Matrix A row " << indx << endl;

				return 0;
                        }

                        else
                        {

                                A.insert(indx, indx - 2) = 1 - 2*K*dt/c2;
                                A.insert(indx, indx - 1) = -l*dt/c2;  // ADD THE COSINE TERM???
				A.insert(indx, indx) = -l*dt/c2;        // ADD THE SINE TERM?

                                A.insert(indx, indx + vars - 2) = K*dt/c2;
                                A.insert(indx, indx + vars - 1)  = -l*dt/c2;
                                A.insert(indx, indx + vars)  = -l*dt/c2;

				A.insert(indx, indx- 2 - vars) = -K*dt/c2;
                                cout << "--- Matrix A row " << indx << endl;
                        }

		}
		// connectivity x
		else if (j == 3)
                                {
                                        A.insert(indx, indx - 3 - 2) = 1;
                                        A.insert(indx, indx + vars - 5) = -1;         // ADD THE coSINE TERM?
                                        cout << "---- Matrix A row " << indx << endl;
                                }
                                
		// Connectivity equation y
                else if (j == 4)
                                {
                                        A.insert(indx, indx - 3 - 2) = 1;          // ADD THE SINE TERM???
                                        A.insert(indx, indx + vars - 5) = -1;
                                        cout << "----- Matrix A row " << indx << endl;
                                }

        	}
	}

	return 1;
}





Eigen::SparseMatrix<double> system_coefficients(Eigen::SparseMatrix<double>& A,vector<double> theta)
{
	cout << "Total rows " << A.rows() << endl;
	int check = system_coefficients_first_half(A, theta);
	if (check != 0)
	{
		throw std::runtime_error("The first half has an error!");
	}
	check = system_coefficients_second_half(A, theta);	
	if (check != 0)
        {
                throw std::runtime_error("The second half has an error!");
        }
	return A;
}


Eigen::VectorXd rhs(Eigen::VectorXd& b, vector<double>& rx, vector<double>& ry, vector<double>& theta)
{
	int indx = 0;
	for (int i = 0; i < N; i++)
	{
		if (i <= floor(N/2)){
			for (int j = 0; j < vars; j++)
			{
				indx = i * vars + j;
				if (i == floor(N/2))
				{
					if (j == 0)
                                                {
                                                        b[indx] = -l/2 * (1 + cos(theta[i+1]));
                                                }
                                        else if (j == 1)
                                                {

                                                        b[indx] = b[indx] = -l/2 * (0 + sin(theta[i+1]));
							// Need to exit the inner loop when the midpoint is addressed
							break;

                                                }
				}
				else{
					if (j == 0)
						{
							b[indx] = dt * u + rx[i];
						}
					else if (j == 1)
						{
					
							b[indx] = dt * v + ry[i];
					
						}
					else if (j == 2)
						{

							b[indx] = dt * omega + theta[i];

						}
					
					else if (j == 3)
							{

								b[indx] = -l/2 * (cos(theta[i]) + cos(theta[i+1]));

							}
					else if (j == 4)
							{

								b[indx] = -l/2 * (sin(theta[i]) + sin(theta[i+1]));

							}	 
				}
			}

		}

		else{
			for (int j = 0; j < vars; j++)
                        {
                                indx = i * vars + j - 3;
                                if (j == 0)
                                        {
                                                b[indx] = dt * u + rx[i];
                                        }
                                else if (j == 1)
                                        {

                                                b[indx] = dt * v + ry[i];

                                        }
                                else if (j == 2)
                                        {

                                                b[indx] = dt * omega + theta[i];

                                        }
                                if (indx < N*vars - 5){
                                        if (j == 3)
                                                {

                                                        b[indx] = -l/2 * (cos(theta[i]) + cos(theta[i+1]));

                                                }
                                        else if (j == 4)
                                                {

                                                        b[indx] = -l/2 * (sin(theta[i]) + sin(theta[i+1]));

                                                }
                                }
                        }		
		
		}
	}
	return b;

}


void writeMatrixToFile(const Eigen::MatrixXd& matrix, const string& filename) {
            // Open a file in write mode
               ofstream outFile(filename);

               if (!outFile.is_open()) {
                       cerr << "Error: Could not open file " << filename << " for writing." << endl;
                       return;
                       }

               // Iterate through the matrix and write its elements to the file
                for (int i = 0; i < matrix.rows(); ++i) {
                          for (int j = 0; j < matrix.cols(); ++j) {
                                      outFile << matrix(i,j) ;
                                      if (j < matrix.cols() - 1) {
                                            outFile << " "; // Add space between elements in a row
                                       }
                            }
                            outFile << "\n"; // Add newline after each row
               }
               outFile.close(); // Close the file
               //cout << "Matrix successfully written to " << filename << endl;

}


int main(){
	vector<double> rx(N, 0);
	vector<double> ry(N, 0);
	vector<double> Xx(N, 0);
	vector<double> Xy(N, 0);
	vector<double> theta(N, 0);
	/// Since there are 5 variables in 2D per cylinder the matrix is 5*N
	/// 5 equations are revomed since the last cylinder doesn't have connectivity and the middle one 
	/// doesn't have equilibrium eq
	Eigen::SparseMatrix<double> A(vars*N - 2 - 3,vars*N -2 - 3);
	Eigen::VectorXd b(vars*N - 2 - 3);
	


	init(&rx, &ry, &Xy, &Xx, &theta);
	A.setZero();
	A = system_coefficients(A, theta);
	writeMatrixToFile(A, "Linear_system_matrix_2.txt");
	b = rhs(b, rx, ry, theta);
	writeMatrixToFile(b, "rhs_linear_system_2.txt");
	
	
	/* Check for the correct inizialization*/
	/*
	for (int i=0; i < N; i++)
        {
                cout << rx[i] << endl;

        }
	*/
	/*        Linear system definition    */




	/********** Start of the time cycle **********/
/*
	for (double t = 0; t < T; t += dt)
	{
				
	
	
	
	
	
	
	
	
		
	}

	
*/

	return 0;
}

























