#pragma once

#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <random>

unsigned long long ull_rand_max = 1ull << 60;
unsigned long long ull_rand() {
	unsigned long long res = 0;
	if (RAND_MAX == 32767)
		for (int i = 0; i < 4; i++)
			res = (res << 15) | rand();
	else {
		res = rand() >> 2;
		res = res << 31 | rand();
	}
	return res;
}

void rand_distribution(int* arr, int n, int div) {
	for (int i = 0; i < n; i++)
		arr[i] = i;
	std::shuffle(arr, arr + n, std::default_random_engine(0));
	int sub = n / div;
	for (int i = 0; i < n; i++)
		arr[i] /= sub;
}