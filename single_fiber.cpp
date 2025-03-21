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



//Define two functions that writes a matrix and a vector for the linear system Ax = b

Eigen::SparseMatrix<double> system_coefficients(Eigen::SparseMatrix<double>& A,vector<double> theta)
{
	cout << "Total rows " << A.rows() << endl;
	for (int i = 0; i < A.rows(); i++)
	{
		// Equation for momentum x
		if (i % N == 0)	
		{
			if (i == 0)
			{
				A.insert(i, i) = 1;
				A.insert(i, i + vars + 1) = -dt/c1;
			}
			
			
			// The last cylinder has the boundary condition of X_i+1 = 0
			// so it doesn't have the final term
			else if (i == A.rows() - (vars - 2))
			{
				A.insert(i, i - 2) = 1;
				A.insert(i, i + 1) = dt/c1;				
			}
			// All cylinders that are neither the first nor the last 
			else
			{
				A.insert(i, i - 2) = 1;
		        	A.insert(i, i + 1) = dt/c1;
				A.insert(i, i + vars + 1) = -dt/c1;
			}

		   cout << "- Matrix A row " << i << endl;
		}
		// Equation for momentum y
		else if (i % N == 1)
                {
			 if (i == 1)
			 {
                                A.insert(i, i) = 1;
                                A.insert(i, i + vars + 1) = -dt/c1;
                         }

                        // The last cylinder has the boundary condition of X_i+1 = 0
                        // so it doesn't have the final term
                        else if (i == A.rows() - (vars - 3))
                        {
                                A.insert(i, i - 2) = 1;
                                A.insert(i, i + 1) = dt/c1;
                        }
                        // All cylinders that are neither the first nor the last
                        else
                        {
                        	A.insert(i, i - 2) = 1;
                        	A.insert(i, i + 1) = dt/c1;
                        	A.insert(i, i + vars + 1) = -dt/c1;
                        }	

			 cout << "-- Matrix A row " << i << endl;
                }
		// Equation for momentum theta (angular)
		else if (i % N == 2)
                {
			if (i == 2) 
			{
				A.insert(i, i) = 1 - 2*K*dt/c2;
                        	//A.insert(i, i + 1) = -l*dt/c2;  // ADD THE COSINE TERM???
                        	//A.insert(i, i + 2) = -l*dt/c2;  // ADD THE SINE TERM?
				//The previous terms are 0 since the boundary condition is 0 for the force
				A.insert(i, i + vars) = -l*dt/c2;
                                A.insert(i, i + vars - 1) = -l*dt/c2;
				A.insert(i, i + vars - 2) = K*dt/c2;
			 	
				cout << "--- Matrix A row " << i << endl;
			}
			// Last cylinder
			else if (i == A.rows() - 1)
			{
				A.insert(i, i - 2) = 1 - 2*K*dt/c2;
                                A.insert(i, i - 1) = -l*dt/c2;  // ADD THE COSINE TERM???
                                A.insert(i, i) = -l*dt/c2;  // ADD THE SINE TERM?
				A.insert(i, i - 2 - vars) = -K*dt/c2;
				cout << "--- Matrix A row " << i << endl;
			}

			else
			{

				A.insert(i, i - 2) = 1 - 2*K*dt/c2;
                        	A.insert(i, i - 1) = -l*dt/c2;  // ADD THE COSINE TERM???
                        	A.insert(i, i) = -l*dt/c2;	// ADD THE SINE TERM?
			
				A.insert(i, i + vars - 2) = K*dt/c2;
				A.insert(i, i + vars - 1)  = -l*dt/c2;
                        	A.insert(i, i + vars)  = -l*dt/c2;		

			        if (i == 7){
					A.insert(i, i - vars) = -K*dt/c2;
				}
				else {
					A.insert(i, i - 2 - vars) = -K*dt/c2;
				}
			 	cout << "--- Matrix A row " << i << endl;
			}
                }
		
		if (i < vars){
			// Connectivity equation x
			if (i % N == 3)
                	{ 
				A.insert(i, i - 3) = 1;
				A.insert(i, i + vars - 5) = -1;         // ADD THE coSINE TERM?
		 		cout << "---- Matrix A row " << i << endl;
                	}
			// Connectivity equation y
			else if (i % N == 4)
                	{
				A.insert(i, i - 3) = 1;		// ADD THE SINE TERM???
                        	A.insert(i, i + vars - 5) = -1;
				cout << "----- Matrix A row " << i << endl;
                	}
		}
		else
		{
		
			 if (i % N == 3)
                        {
                                A.insert(i, i - 3 - 2) = 1;
                                A.insert(i, i + vars - 5) = -1;         // ADD THE coSINE TERM?
                                cout << "---- Matrix A row " << i << endl;
                        }
                        // Connectivity equation y
                        else if (i % N == 4)
                        {
                                A.insert(i,i - 3 - 2) = 1;          // ADD THE SINE TERM???
                                A.insert(i,i + vars - 5) = -1;
                                cout << "----- Matrix A row " << i << endl;
                        }
		
		}
		
	}
		return A;
		
}


Eigen::VectorXd rhs(Eigen::VectorXd& b, vector<double>& rx, vector<double>& ry, vector<double>& theta)
{
	int indx = 0;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < vars; j++)
		{
			indx = i * vars + j;
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

				 	b[indx] = dt * omega + theta[i] + i;

                        	}
			if (indx < N*vars - 2){
				if (j == 3)
                        		{

						b[indx] = -l/2 * (cos(theta[i]) + cos(theta[i+1])) + i;

                        		}
				else if (j == 4)
                        		{

						b[indx] = -l/2 * (sin(theta[i]) + sin(theta[i+1])) + i;

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
	// Since there are 5 variables in 2D per cylinder the matrix is 5*N
	Eigen::SparseMatrix<double> A(vars*N - 2,vars*N -2);
	Eigen::VectorXd b(vars*N - 2);
	


	init(&rx, &ry, &Xy, &Xx, &theta);
	A.setZero();
	A = system_coefficients(A, theta);
	writeMatrixToFile(A, "Linear_system_matrix.txt");
	b = rhs(b, rx, ry, theta);
	writeMatrixToFile(b, "rhs_linear_system.txt");
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

























