// Simple k-means calculation with 2 centroids using ParallelVector

#include <iostream>
#include <vector>
#include "ParallelVector.hpp"

int main(int argc, char *argv[]) {
	const unsigned num_points = 100000;
	const unsigned num_iterations = 10;
	
	// generate random points
	std::vector<float> std_points_X(num_points);
	std::vector<float> std_points_Y(num_points);
	
	srand(time(0));
	for (unsigned i = 0; i < num_points; ++i) {
		std_points_X[i] = (float)rand() / (float)RAND_MAX;
		std_points_Y[i] = (float)rand() / (float)RAND_MAX;
	}
	
	// initialize centroids to two first points
	float centroid1_X = std_points_X[0];
	float centroid1_Y = std_points_Y[0];
	float centroid2_X = std_points_X[1];
	float centroid2_Y = std_points_Y[1];
	
	// import point data into ParallelVector
	PV::Vector<float> points_X(std_points_X);
	PV::Vector<float> points_Y(std_points_Y);
	
	for (unsigned itteration = 0; itteration < num_iterations; ++itteration) {
		printf("itteration %d\n", itteration + 1);
		printf("centroid 1 starting location: (%g, %g)\n", centroid1_X, centroid1_Y);
		printf("centroid 2 starting location: (%g, %g)\n", centroid2_X, centroid2_Y);
		
		// import centroid data into ParallelVector
		PV::Vector<float> centroid1_X_vec(num_points, centroid1_X);
		PV::Vector<float> centroid1_Y_vec(num_points, centroid1_Y);
		PV::Vector<float> centroid2_X_vec(num_points, centroid2_X);
		PV::Vector<float> centroid2_Y_vec(num_points, centroid2_Y);
		
		// calculate squared distances from points to both centroids
		PV::Vector<float> distance1 = (points_X - centroid1_X_vec) * (points_X - centroid1_X_vec) +
		                              (points_Y - centroid1_Y_vec) * (points_Y - centroid1_Y_vec);
		PV::Vector<float> distance2 = (points_X - centroid2_X_vec) * (points_X - centroid2_X_vec) +
		                              (points_Y - centroid2_Y_vec) * (points_Y - centroid2_Y_vec);
		
		// get points closest to each centroid
		PV::Vector<float> points1_X = points_X.filterBy(distance1 < distance2);
		PV::Vector<float> points1_Y = points_Y.filterBy(distance1 < distance2);
		PV::Vector<float> points2_X = points_X.filterBy(distance1 >= distance2);
		PV::Vector<float> points2_Y = points_Y.filterBy(distance1 >= distance2);
		
		// get new coordinates for each centroid
		centroid1_X = points1_X.sum() / points1_X.size();
		centroid1_Y = points1_Y.sum() / points1_Y.size();
		centroid2_X = points2_X.sum() / points2_X.size();
		centroid2_Y = points2_Y.sum() / points2_Y.size();
	}
	
	// print out final centroid coordinates
	printf("centroid 1 final location: (%g, %g)\n", centroid1_X, centroid1_Y);
	printf("centroid 2 final location: (%g, %g)\n", centroid2_X, centroid2_Y);
}