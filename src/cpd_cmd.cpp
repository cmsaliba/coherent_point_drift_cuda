// standard includes
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

// CLI11
#include <CLI11.hpp>

// cpd_cuda
#include "cpd_cuda.h"

// ************************************************************************** //
// Load point set from a file.
// Overlaod to accept single or double precision floating point arguments.
// ************************************************************************** //

void LoadPoints(const std::string& points_filename,
                std::vector<std::vector<float>>& points)
{
	std::ifstream points_file(points_filename.c_str());
	if (points_file.is_open() == false)
		std::cerr << "File not found: " << points_filename << std::endl;

	std::string csv_line, csv_val;
	float val;
	for (int i = 0; std::getline(points_file, csv_line); ++i)
	{
		std::vector<float> tmp_point;
		std::istringstream csv_line_stream(csv_line);
		for (int j = 0; std::getline(csv_line_stream, csv_val, ','); ++j)
		{
			std::istringstream csv_val_stream(csv_val);
			csv_val_stream >> val;
			tmp_point.push_back(val);
		}
		points.push_back(tmp_point);
	}
}

void LoadPoints(const std::string& points_filename,
                	std::vector<std::vector<double>>& points)
{
	std::ifstream points_file(points_filename.c_str());
	if (points_file.is_open() == false)
		std::cerr << "File not found: " << points_filename << std::endl;

	std::string csv_line, csv_val;
	double val;
	for (int i = 0; std::getline(points_file, csv_line); ++i)
	{
		std::vector<double> tmp_point;
		std::istringstream csv_line_stream(csv_line);
		for (int j = 0; std::getline(csv_line_stream, csv_val, ','); ++j)
		{
			std::istringstream csv_val_stream(csv_val);
			csv_val_stream >> val;
			tmp_point.push_back(val);
		}
		points.push_back(tmp_point);
	}
}

template<typename cudafloat>
bool RunCPD(std::string Xptsfile, std::string Yptsfile, std::string Tptsfile, 
			std::string Cfile, cudafloat w, cudafloat beta, cudafloat lambda, 
			cudafloat tol, int max_iter)
{
	bool b_return = false;

	// load the unnormalized X point set
	std::vector<std::vector<cudafloat>> Xraw_vec;
	LoadPoints(Xptsfile, Xraw_vec);
	// number of X points and dimensionality
	const int N = (int)Xraw_vec.size();
	const int D = (int)Xraw_vec[0].size();

	// load the unnormalized Y point set
	std::vector<std::vector<cudafloat>> Yraw_vec;
	LoadPoints(Yptsfile, Yraw_vec);
	// number of Y points
	const int M = (int)Yraw_vec.size();
	// check dimensionality
	bool b_valid_dimensions = true;
	if (Yraw_vec[0].size() != D)
	{
		std::cerr << "Point set dimensionality must agree." << std::endl;
		b_valid_dimensions = false;
	}

	if (b_valid_dimensions)
	{
		// create unnormalized X point set array, column-major storage
		cudafloat* Xraw = (cudafloat*)malloc(N * D * sizeof(cudafloat));
		for (int n = 0; n < N; ++n)
		{
			for (int d = 0; d < D; ++d)
				Xraw[d * N + n] = Xraw_vec[n][d];
		}

		// create unnormalized Y point set array, column-major storage
		cudafloat* Yraw = (cudafloat*)malloc(M * D * sizeof(cudafloat));
		for (int m = 0; m < M; ++m)
		{
			for (int d = 0; d < D; ++d)
				Yraw[d * M + m] = Yraw_vec[m][d];
		}

		// transformed point set
		cudafloat* Traw = (cudafloat*)malloc(M * D * sizeof(cudafloat));

		// if requested, correspondence vector
		int* C = NULL;
		if (Cfile != "")
		{
			C = (int*)malloc(M * sizeof(int));
		}

		CoherentPointDrift(Xraw, Yraw, M, N, D, w, beta, lambda, tol, max_iter, 
			Traw, C);

		// save transformed point set to file
		std::ofstream T_file;
		T_file.open(Tptsfile);
		for (int i = 0; i < M; ++i)
		{
			for (int j = 0; j < D; ++j)
			{
				T_file << Traw[i + j*M];
				if ( j < D - 1) T_file << ",";
			}
			T_file << std::endl;
		}
		T_file.close();

		// save correspondence vector to file
		if (C)
		{
			std::ofstream C_file;
			C_file.open(Cfile);
			for (int i = 0; i < M; ++i)
			{
				C_file << C[i] << std::endl;
			}
			C_file.close();
		}
			
		// free memory
		free(Xraw);
		free(Yraw);
		free(Traw);
		free(C);

		b_return = true;
	}

	return b_return;
}

// ************************************************************************** //
// Main function.
// Parses input and calls RunCPD.
// ************************************************************************** //

int main(int argc, char* argv[])
{
	// setup and parse command line inputs

	CLI::App app{"CPD_CUDA Command Line"};

	std::string xpts_file;
	std::string ypts_file;
	std::string tpts_file;
	std::string correspondence_file = "";
	bool b_use_double_precision = false;
	double w = 0.1;
	double beta = 2.0;
	double lambda = 3.0;
	double tol = 1e-5;
	int max_iter = 500;

	app.add_option("xpts_file", xpts_file, 
		"File containing the target point set.")->required();
	app.add_option("ypts_file", ypts_file, 
		"File containing the point set to transform.")->required();
	app.add_option("tpts_file", tpts_file, 
		"File to write the transformed point set to.")->required();
	app.add_option("correspondence_file", correspondence_file, 
		"File to write the correspondence vector to.");
	app.add_flag("--use_double_precision", b_use_double_precision, 
		"Use double precision math.");
	app.add_option("--w", w, "Point set noise parameter (0 <= w <= 1).", true);
	app.add_option("--beta", beta,
		"Smoothness and regularization parameter (beta > 0).", true);
	app.add_option("--lambda", lambda, 
		"Smoothness and regularization parameter (lambda > 0).", true);
	app.add_option("--tol", tol, "Convergence tolerance (tol >= 0).", true);
	app.add_option("--max_iter", max_iter, "Maximum iterations to run.", true);
	
	CLI11_PARSE(app, argc, argv);

	// check options for valid values
	if ( w < 0 || w > 1 || beta <= 0 || lambda <= 0 || tol < 0)
	{
		std::cerr 
			<< "INVALID INPUT: 0 <= w <= 1, beta > 0, lambda > 0, tol >= 0." 
			<< std::endl;
		return 1;
	}

	// run CPD
	if (b_use_double_precision)
	{
		RunCPD(xpts_file, ypts_file, tpts_file, correspondence_file, (double)w, 
			(double)beta, (double)lambda, (double)tol, max_iter);
	}
	else
	{
		RunCPD(xpts_file, ypts_file, tpts_file, correspondence_file, (float)w, 
			(float)beta, (float)lambda, (float)tol, max_iter);
	}		

	return 0;
}