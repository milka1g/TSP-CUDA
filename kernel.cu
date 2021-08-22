﻿#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include "device_launch_parameters.h"
#include <float.h>
#include <time.h>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <sstream>
#include "utils.h"

using namespace std;

#define BLOCKSIZE 256
#define PERM_SIZE 12
#define REAL_PERM 12
#define NUMTR 1871100

__constant__ float distanceMap[REAL_PERM][REAL_PERM];

int* permCPU(unsigned long long m)
{
	int i, ind;
	int* permuted = new int[REAL_PERM];
	int* elems = new int[REAL_PERM];

	for (i = 0; i < REAL_PERM; i++) elems[i] = i;

	for (i = 0; i < REAL_PERM; i++)
	{
		ind = m % (REAL_PERM - i);
		m = m / (REAL_PERM - i);
		permuted[i] = elems[ind];
		elems[ind] = elems[REAL_PERM - i - 1];
	}
	return permuted;
}


__global__ void kernelReduce(float* distance, unsigned long long* step, unsigned int* index) {
	__shared__ float distances[BLOCKSIZE];
	__shared__ unsigned int realindex[BLOCKSIZE];
	unsigned int tid = threadIdx.x;
	unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < 479001600) {
		unsigned int i, ind;
		unsigned long long m = id + (*step);
		unsigned int permuted[REAL_PERM];
		unsigned int elems[REAL_PERM];
		float len = 0;

		for (i = 0; i < REAL_PERM; i++) elems[i] = i;

		for (i = 0; i < REAL_PERM; i++)
		{
			ind = m % (REAL_PERM - i);
			m = m / (REAL_PERM - i);
			permuted[i] = elems[ind];
			elems[ind] = elems[REAL_PERM - i - 1];
		}

		for (i = 0; i < REAL_PERM - 1; i++)
			len = len + distanceMap[permuted[i]][permuted[i + 1]];

		distances[tid] = len;
		realindex[tid] = id;

		__syncthreads();
		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s) {
				if (distances[tid] > distances[tid + s]) {
					distances[tid] = distances[tid + s];
					realindex[tid] = realindex[tid + s];
				}
			}
			__syncthreads();
		}

		if (tid == 0) {
			distance[blockIdx.x] = distances[0];
			index[blockIdx.x] = realindex[0];
		};
	}

}

int main(int argc, char* argv[]) {

	float points[15][2] = {
		{62.0f, 58.4f},
		{57.5f, 56.0f},
		{51.7f, 56.0f},
		{67.9f, 19.6f},
		{57.7f, 42.1f},
		{54.2f, 29.1f},
		{46.0f, 45.1f},
		{34.7f, 45.1f},
		{45.7f, 25.1f},
		{34.7f, 26.4f},
		{28.4f, 31.7f},
		{33.4f, 60.5f}
	};

	vector<vector<float>> holes;
	int numHoles = 0;

	for (int i = 1; i < argc; i++) {
		holes.clear(); 
		numHoles = 0;
		string fileName = argv[i];
		int numHoles = 0;
		try {
			std::ifstream file("C:\\Users\\mn170387d\\Desktop\\clusters\\" + fileName);
			std::string str;
			while (std::getline(file, str)) {
				std::istringstream in(str);
				float x, y;
				in >> x >> y;
				//cout << "x: " << x << ", y: " << y << '\n';
				vector<float> hole{ x, y };
				std::cout << hole[0] << ", " << hole[1] << "\n";
				holes.push_back(hole);
				numHoles++;
			}
		}
		catch (const std::exception& e) {
			cout << "File doesn't exist!";
		}

		//compute distances
		vector<vector<float>> distances(numHoles, vector<float>(numHoles));
		vector<float> bestPerm(numHoles);

		for (int i = 0; i < numHoles; i++) {
			for (int j = 0; j < numHoles; j++) {
				if (i == j)
					distances[i][j] = 0.0f;
				else
					distances[i][j] = sqrt(pow(holes[i][0] - holes[j][0], 2.0f) + pow(holes[i][1] - holes[j][1], 2.0f));
			}
		}

		
		if (numHoles <= 12) { //no need to do CUDA
			auto start = std::chrono::high_resolution_clock::now();

			float shortestPathLength = DBL_MAX;
			float currCost = 0.0f;
			int* P = new int[numHoles];
			for (int i = 0; i < numHoles; i++) {
				P[i] = i; //holes from 1-numHoles
			}

			do {
				currCost = 0.0f;
				for (int i = 0; i < numHoles - 1; i++) {
					currCost += distances[P[i]][P[i+1]];
				}

				if (currCost < shortestPathLength) {
					shortestPathLength = currCost;
					/*for (int i = 0; i < numHoles; i++) {
						bestPerm[i] = P[i];
					}*/
				}
			} while (next_permutation(numHoles, P));

			cout << "Najkraca putanja iznosi " << fixed << shortestPathLength << " mm.\n" << endl;
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<float> duration = end - start;
			cout << "Time elapsed: " << duration.count() * 1000 << "ms\n";
		}
		else { //doing cuda for 12+
			//auto start = std::chrono::high_resolution_clock::now();
		
			//cudaError_t err;
			//// 

			//err = cudaMemcpyToSymbol(distanceMap, distances, REAL_PERM * REAL_PERM * sizeof(float), 0, cudaMemcpyHostToDevice);
			//if (err != cudaSuccess) {
			//	std::cout << "Nije uspelo kopija matrice\n";
			//}
			//// //
			//float* h_distance, * d_distance;
			//unsigned int* h_index, * d_index;
			//unsigned long long* h_step = new unsigned long long, * d_step;
			//*h_step = 0;
			//err = cudaMalloc(&d_step, sizeof(unsigned long long));
			//if (err != cudaSuccess) {
			//	std::cout << "Nije uspelo malloc d_step\n";
			//}
			//err = cudaMemcpy(d_step, h_step, sizeof(unsigned long long), cudaMemcpyHostToDevice);
			//if (err != cudaSuccess) {
			//	std::cout << "Nije uspelo distancecpy d_step\n";
			//}

			//err = cudaMalloc(&d_index, sizeof(unsigned int) * NUMTR);
			//if (err != cudaSuccess) {
			//	std::cout << "Nije uspelo malloc d_index\n";
			//}

			//h_distance = new float[NUMTR];
			//h_index = new unsigned int[NUMTR];

			//err = cudaMalloc(&d_distance, sizeof(float) * NUMTR);
			//if (err != cudaSuccess) {
			//	std::cout << "Nije uspelo malloc d_distance\n";
			//}


			//float min = FLT_MAX;
			//unsigned long long bestind = 0;

			//for (int i = 0; i < factorial(REAL_PERM) / factorial(PERM_SIZE); i++) {
			//	err = cudaMemcpy(d_step, h_step, sizeof(unsigned long long), cudaMemcpyHostToDevice);
			//	if (err != cudaSuccess) {
			//		std::cout << "Nije uspelo, setovanje stepa u petlji\n";
			//	}

			//	kernelReduce << <NUMTR, BLOCKSIZE >> > (d_distance, d_step, d_index);

			//	cudaDeviceSynchronize();
			//	err = cudaMemcpy(h_distance, d_distance, sizeof(float) * NUMTR, cudaMemcpyDeviceToHost);
			//	if (err != cudaSuccess) {
			//		std::cout << "Nije uspelo kopiranje u petlji u h_distance\n";
			//	}
			//	err = cudaMemcpy(h_index, d_index, sizeof(unsigned int) * NUMTR, cudaMemcpyDeviceToHost);
			//	if (err != cudaSuccess) {
			//		std::cout << "Nije uspelo kopiranje u petlji u h_index\n";
			//	}
			//	for (int i = 0; i < NUMTR; i++) {
			//		if (h_distance[i] < min) {
			//			min = h_distance[i];
			//			bestind = h_index[i] + *h_step;
			//		}
			//	}
			//	(*h_step) += factorial(PERM_SIZE);
			//}
			//float newmin = 0;
			//int* rez = permCPU(bestind);

			//for (int i = 0; i < REAL_PERM - 1; i++)
			//	newmin = newmin + distances[rez[i]][rez[i + 1]];

			//std::cout << "Najbolji put je:\n";
			//for (int i = 0; i < REAL_PERM; i++) {
			//	std::cout << rez[i] + 1 << " ";
			//}
			//std::cout << std::endl;


			//auto end = std::chrono::high_resolution_clock::now();
			//std::chrono::duration<float> duration = end - start;

			//std::cout << " Minimal path from kernel " << min << ", calculated " << newmin << ", execution time: " << duration.count() * 1000 << "ms" << std::endl;

			//cudaFree(d_distance);
			//delete[] h_distance;
			//delete[] rez;
			//delete h_step;
			//delete[] h_index;

			//cudaDeviceReset();

		}


	}


	return 0;
}