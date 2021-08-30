#include <cuda.h>
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


using namespace std;

#define BLOCKSIZE 256
#define PERM_SIZE 12
#define REAL_PERM 13 //max size we will need for cuda
#define THREAD_BLOCKS 1871100 //12! / 256 = 479001600/256 = 1871100

__constant__ float distanceMap[REAL_PERM + 2][REAL_PERM + 2];

int next_permutation(const int N, int* P) {
	int s;
	int* first = &P[0];
	int* last = &P[N - 1];
	int* k = last - 1;
	int* l = last;
	//find larges k so that P[k]<P[k+1]
	while (k > first) {
		if (*k < *(k + 1)) {
			break;
		}
		k--;
	}
	//if no P[k]<P[k+1], P is the last permutation in lexicographic order
	if (*k > *(k + 1)) {
		return 0;
	}
	//find largest l so that P[k]<P[l]
	while (l > k) {
		if (*l > *k) {
			break;
		}
		l--;
	}
	//swap P[l] and P[k]
	s = *k;
	*k = *l;
	*l = s;
	//reverse the remaining P[k+1]...P[N-1]
	first = k + 1;
	while (first < last) {
		s = *first;
		*first = *last;
		*last = s;

		first++;
		last--;
	}

	return 1;
}


unsigned long long factorial(int n)
{
	unsigned long long factorial = 1;
	for (int i = 1; i <= n; ++i)
	{
		factorial *= i;
	}
	return factorial;
}


int* permCPU(unsigned long long m)
{
	int i, ind;
	int* permuted = new int[REAL_PERM];
	int* elems = new int[REAL_PERM];

	for (i = 0; i < REAL_PERM; i++) elems[i] = i + 1;  //first and last hole is fixed, we permute the numHoles-2 in between

	for (i = 0; i < REAL_PERM; i++)
	{
		ind = m % (REAL_PERM - i);
		m = m / (REAL_PERM - i);
		permuted[i] = elems[ind];
		elems[ind] = elems[REAL_PERM - i - 1];
	}
	delete[] elems;
	return permuted;
}


__global__ void kernelReduce(float* distance, unsigned long long* step, unsigned int* index) {
	//extern __shared__ float shared[];
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

		for (i = 0; i < REAL_PERM; i++) elems[i] = i + 1; //first and last hole is fixed, we permute the numHoles-2 in between

		for (i = 0; i < REAL_PERM; i++)
		{
			ind = m % (REAL_PERM - i);
			m = m / (REAL_PERM - i);
			permuted[i] = elems[ind];
			elems[ind] = elems[REAL_PERM - i - 1];
		}

		len = len + distanceMap[0][permuted[0]];
		for (i = 0; i < REAL_PERM - 1; i++)
			len = len + distanceMap[permuted[i]][permuted[i + 1]];
		len = len + distanceMap[permuted[REAL_PERM - 1]][REAL_PERM + 1];

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

	vector<vector<float>> holes;

	string fileName = argv[1];
	int numHoles = 0;
	int num_holes = 0; //from file first line check if correct
	try {
		std::ifstream file("C:\\Users\\mn170387d\\Desktop\\clusters\\" + fileName);
		std::string str;
		std::getline(file, str);
		std::istringstream in(str);
		in >> num_holes;
		while (std::getline(file, str)) {
			std::istringstream in(str);
			float x, y;
			in >> x >> y;
			vector<float> hole{ x, y };
			std::cout << hole[0] << ", " << hole[1] << "\n";
			holes.push_back(hole);
			numHoles++;
		}
	}
	catch (const std::exception&) {
		cout << "File doesn't exist!";
	}

	if (num_holes == numHoles) //we read from second to second last
		std::cout << "Numbers of holes are good: " << num_holes << '\n';

	//compute distances
	vector<vector<float>> distances(numHoles, vector<float>(numHoles));
	vector<int> bestPerm(numHoles);

	for (int i = 0; i < numHoles; i++) {
		for (int j = 0; j < numHoles; j++) {
			if (i == j)
				distances[i][j] = 0.0f;
			else
				distances[i][j] = sqrt(pow(holes[i][0] - holes[j][0], 2.0f) + pow(holes[i][1] - holes[j][1], 2.0f));
		}
	}

	//cout << "BR RUPA: " << numHoles << endl;

	if (numHoles <= 14) { //no need to do CUDA
		auto start = std::chrono::high_resolution_clock::now();
		int permSize = numHoles - 2; //0 1..11 12 for 13==numHoles
		float shortestPathLength = FLT_MAX;
		float currCost = 0.0f;
		int* P = new int[permSize];
		for (int i = 0; i < permSize; i++) {
			P[i] = i + 1; 
		}
	
		do {
			currCost = 0.0f;
			currCost += distances[0][P[0]];
			for (int i = 0; i < permSize - 1; i++) {
				currCost += distances[P[i]][P[i + 1]];
			}
			currCost += distances[P[permSize - 1]][numHoles - 1];
			if (currCost < shortestPathLength) {
				shortestPathLength = currCost;
				for (int i = 0; i < permSize ; i++) {
					bestPerm[i] = P[i];
				}
			}
		} while (next_permutation(permSize, P));

		cout << "Best cost: " << fixed << shortestPathLength << " mm.\n" << endl;

		ofstream out;
		out.open("C:\\Users\\mn170387d\\Desktop\\clusters\\solved" + fileName);
		out << numHoles << '\n';
		out << holes[0][0] << ' ' << holes[0][1] << " 0" << '\n';
		for (int i = 0; i < permSize; i++) {
			out << holes[i + 1][0] << ' ' << holes[i + 1][1] << ' ' << bestPerm[i] << '\n';
		}
		out << holes[numHoles - 1][0] << ' ' << holes[numHoles - 1][1] << ' ' << numHoles - 1;
		out.close();

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> duration = end - start;
		cout << "Time elapsed: " << duration.count() << "ms\n";
	}
	else { //doing cuda for 12+
		auto start = std::chrono::high_resolution_clock::now();
		//must copy contiguous arr to device 
		float distancesCont[REAL_PERM + 2][REAL_PERM + 2]; //first and last hole are fixed
		for (int i = 0; i < numHoles; i++) {
			for (int j = 0; j < numHoles; j++) {
				distancesCont[i][j] = distances[i][j];
			}
		}

		cudaError_t err;
		// 
		err = cudaMemcpyToSymbol(distanceMap, distancesCont, (REAL_PERM + 2) * (REAL_PERM + 2) * sizeof(float), 0, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cout << "Copying distanceMap failed\n";
		}

		float* h_distance, * d_distance;
		unsigned int* h_index, * d_index;
		unsigned long long* h_step = new unsigned long long, * d_step;
		*h_step = 0;
		err = cudaMalloc(&d_step, sizeof(unsigned long long));
		if (err != cudaSuccess) {
			std::cout << "cudaMalloc failed d_step \n";
		}
		err = cudaMemcpy(d_step, h_step, sizeof(unsigned long long), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			std::cout << "cudaMemcpy failed d_step\n";
		}

		err = cudaMalloc(&d_index, sizeof(unsigned int) * THREAD_BLOCKS);
		if (err != cudaSuccess) {
			std::cout << "cudaMalloc failed d_index\n";
		}

		h_distance = new float[THREAD_BLOCKS];
		h_index = new unsigned int[THREAD_BLOCKS];

		err = cudaMalloc(&d_distance, sizeof(float) * THREAD_BLOCKS);
		if (err != cudaSuccess) {
			std::cout << "cudaMalloc failed d_distance\n";
		}


		float min = FLT_MAX;
		unsigned long long bestind = 0;

		for (int i = 0; i < factorial(REAL_PERM) / factorial(PERM_SIZE); i++) {
			err = cudaMemcpy(d_step, h_step, sizeof(unsigned long long), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "cudaMemcpy failed setting of step in loop\n";
			}

			kernelReduce << <THREAD_BLOCKS, BLOCKSIZE >> > (d_distance, d_step, d_index);

			cudaDeviceSynchronize();
			err = cudaMemcpy(h_distance, d_distance, sizeof(float) * THREAD_BLOCKS, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) {
				std::cout << "cudaMemcpy failed in loop d_distance->h_distance\n";
			}
			err = cudaMemcpy(h_index, d_index, sizeof(unsigned int) * THREAD_BLOCKS, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) {
				std::cout << "cudaMemcpy failed in loop d_index->h_index\n";
			}
			for (int i = 0; i < THREAD_BLOCKS; i++) {
				if (h_distance[i] < min) {
					min = h_distance[i];
					bestind = h_index[i] + *h_step;
				}
			}
			(*h_step) += factorial(PERM_SIZE);
		}
		float newmin = 0;
		int* rez = permCPU(bestind);

		newmin += distancesCont[0][rez[0]];
		for (int i = 0; i < REAL_PERM - 1; i++)
			newmin = newmin + distancesCont[rez[i]][rez[i + 1]];
		newmin += distancesCont[rez[REAL_PERM - 1]][REAL_PERM + 1];

		std::cout << "Best path is:\n";
		std::cout << 0 << " ";
		for (int i = 0; i < REAL_PERM; i++) {
			std::cout << rez[i] + 1 << " ";
		}
		std::cout << REAL_PERM + 1 << " ";
		std::cout << std::endl;


		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> duration = end - start;
		cout << "Time elapsed: " << duration.count() << "ms\n";

		std::cout << " Minimal path from kernel " << min << ", calculated " << newmin << ", execution time: " << duration.count() << "ms" << std::endl;

		ofstream out;
		out.open("C:\\Users\\mn170387d\\Desktop\\clusters\\solved" + fileName);
		out << numHoles << '\n';
		out << holes[0][0] << ' ' << holes[0][1] << " 0" << '\n';
		for (int i = 0; i < REAL_PERM; i++) {
			out << holes[i + 1][0] << ' ' << holes[i + 1][1] << ' ' << rez[i] << '\n';
		}
		out << holes[numHoles - 1][0] << ' ' << holes[numHoles - 1][1] << ' ' << numHoles - 1 << '\n';
		out.close();

		cudaFree(d_distance);
		cudaFree(d_index);
		cudaFree(d_step);
		delete[] h_distance;
		delete h_step;
		delete[] h_index;
		delete[] rez;

		cudaDeviceReset();
	}

	return 0;
}