#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <algorithm>
#include <ctime>
#include <cstring>
#include <cctype>
#include <omp.h>
#include "qrand.h"
#include "qcirc.h"
#include "qsubnet.h"

using std::cout;
using std::cerr;
using std::endl;

const char input_file_name[]="TRI_6.json";
const int n_nodes = 6;
const int n_layers = 10;
const int n_qBits = n_nodes + 1;
const int n_params = n_qBits * n_layers;
const int n_Rs = pow(3, n_qBits);
const int n_CXs = n_qBits;
const int n_graphs = 1000;
const int pool_size = 256;
const int n_init = 8;
const int n_compete = 16;
const int n_train = 256;
const int n_threads = 16;

const int n_edge_label = 2;
const int n_node_label = 2;
const int n_graph_label = 2;

const double threshold = log(1.0 / n_graph_label + 0.001);

int edge_label[n_graphs][n_nodes][n_nodes];
int node_label[n_graphs][n_nodes];
int graph_label[n_graphs];

qState<n_qBits> encoded_states[n_graphs];

char output_file_name[2048], record_file_name[2048];

typedef qState<n_qBits, n_params> autoGrad_state;

int valid_CNOTs[n_CXs][2];
void valid_CNOTs_init() {
	for (int i = 0; i < n_CXs; i++) {
		valid_CNOTs[i][0] = i;
		valid_CNOTs[i][1] = (i + 1) % n_qBits;
	}
}

void read_dataset(const char* s) {
	FILE* fin = fopen(s, "r");
	if (fin == nullptr) {
		cerr << "File not found." << endl;
		exit(1);
	}
	for (;;)
		if (fgetc(fin) == '[')
			break;
	for (int i = 0; i < n_graphs; i++)
		for (int j = 0; j < n_nodes; j++)
			for (int k = 0; k < n_nodes; k++)
				for (;;) {
					int c = fgetc(fin);
					if (isdigit(c)) {
						edge_label[i][j][k] = c - '0';
						break;
					}
				}
	for (int i = 0; i < n_graphs; i++)
		for (;;) {
			int c = fgetc(fin);
			if (isdigit(c)) {
				graph_label[i] = c - '0';
				break;
			}
		}
	fclose(fin);
}

template<int paramN>
void encode_layer(qState<n_qBits, paramN>& qs, int graph_idx) {
	for (int i = 0; i < n_nodes; i++)
		qs.h_gate(i);
//		qs.ry_param_gate(i, node_label[graph_idx][i] * 2.0 * Pi / n_node_label);
	for (int i = 0; i < n_nodes; i++)
		for (int j = i + 1; j < n_nodes; j++)
			if (edge_label[graph_idx][i][j])
				qs.cz_gate(i, j);
//			qs.crz_gate(i, j, edge_label[graph_idx][i][j] * 2.0 * Pi / n_edge_label);
	for (int i = 0; i < n_nodes; i++)
		qs.cx_gate(i, n_nodes);
}

void encoded_states_init() {
	for (int i = 0; i < n_graphs; i++)
		encode_layer(encoded_states[i], i);
}

autoGrad_state state[100];
double temp_exp[100];
double temp_grad[100][n_params];

int measurements = 0;
struct Model {
	subnet_union sub[n_layers];
	double param[n_params];
	double grad[n_params];
	double grad_norm;
	double loss;
	double valid_loss;
	double acc;
	double valid_acc;
	double learning_rate;
	int survived;

	void init() {
		subnet<n_qBits> n[n_layers];
		n[0] = random_subnet_generate(n[0]);
		for (int i = 1; i < n_layers; i++)
			n[i] = random_subnet_generate(n[i - 1]);
		for (int i = 0; i < n_layers; i++) {
			if (n[i].sub.id == 0)
				return init();
			sub[i] = n[i].sub;
		}
		for (int i = 0; i < n_params; i++)
			param[i] = ull_rand() * 2.0 * Pi / ull_rand_max;
		learning_rate = 64.0;
		survived = 1;
	}

	int param_cnt()const {
		int cnt = 0;
		for (int i = 0; i < n_layers; i++)
			for (int j = 0; j < n_qBits; j++)
				if ((sub[i].state[1] | sub[i].state[2]) & 1 << j)
					cnt++;
		return cnt;
	}

	void run(autoGrad_state& qs, bool need_grad = 0)const {
		int paramIdx = 0;
		for (int L = 0; L < n_layers; L++) {
			for (int i = 0; i < n_qBits; i++) {
				int paramCnt = need_grad ? paramIdx + 1 : 0;
				if (sub[L].state[1] & 1 << i) {
					qs.ry_param_gate(i, param[paramIdx], paramIdx, paramCnt);
					paramIdx++;
				}
				if (sub[L].state[2] & 1 << i) {
					qs.rz_param_gate(i, param[paramIdx], paramIdx, paramCnt);
					paramIdx++;
				}
			}
			int paramCnt = need_grad ? paramIdx : 0;
			for (int i = 0; i < n_CXs; i++)
				if (sub[L].state[0] & 1 << i)
					qs.cx_gate(valid_CNOTs[i][0], valid_CNOTs[i][1], paramCnt);
		}
	}

	double get_exp(autoGrad_state& qs, int graph_idx, double* temp_grad = nullptr)const {
		bool need_grad = temp_grad != nullptr;
		if (need_grad)
			memset(temp_grad, 0, sizeof(grad));
		qs.load(encoded_states[graph_idx].qVec);
		run(qs, need_grad);
		double expval = 0;
		for (int i = 0; i < 1 << n_qBits; i++) {
			int label = 0;
			for (int j = i; j > 0; j >>= 1)
				label ^= j;
			if ((label & 1) != graph_label[graph_idx])
				continue;
			expval += std::norm(qs.qVec[i]);
			if (need_grad)
				for (int j = 0; j < n_params; j++)
					temp_grad[j] += 2 * qs.qVec[i].real() * qs.grad[i][j].real(),
					temp_grad[j] += 2 * qs.qVec[i].imag() * qs.grad[i][j].imag();
		}
		return expval;
	}

	double get_loss(int input_begin, int input_end, bool need_grad, double* new_acc) {
		double new_loss = 0;
		int correct = 0;
		measurements++;
		if (need_grad)
			memset(grad, 0, sizeof(grad)),
			measurements += 2 * param_cnt();

		for (int p = input_begin; p < input_end; p += 100) {
#pragma omp parallel for
			for (int i = 0; i < 100; i++) {
				if (need_grad) {
					temp_exp[i] = get_exp(state[i], p + i, temp_grad[i]);
					for (int j = 0; j < n_params; j++)
						temp_grad[i][j] /= temp_exp[i];
				}
				else
					temp_exp[i] = get_exp(state[i], p + i);
				temp_exp[i] = log(temp_exp[i]);
			}

			for (int i = 0; i < 100; i++) {
				if (need_grad)
					for (int j = 0; j < n_params; j++)
						grad[j] -= temp_grad[i][j];
				new_loss -= temp_exp[i];
				if (temp_exp[i] > threshold)
					correct++;
			}
		}

		if (need_grad) {
			grad_norm = 0;
			for (int i = 0; i < n_params; i++)
				grad[i] /= (input_end - input_begin),
				grad_norm += grad[i] * grad[i];
		}
		if (new_acc != nullptr)
			*new_acc = (double)correct / (input_end - input_begin);
		return new_loss / (input_end - input_begin);
	}

	void gradient_descent(int input_begin, int input_end) {
		double old_loss = loss;
		double old_param[n_params];
		memcpy(old_param, param, sizeof(param));
		for (int i = 0; i < 32; i++) {
			for (int j = 0; j < n_params; j++)
				param[j] = old_param[j] - learning_rate * grad[j];
			loss = get_loss(input_begin, input_end, false, nullptr);
			if (loss < old_loss - 0.9 * learning_rate * grad_norm)
				break;
			learning_rate *= 0.618;
		}
		learning_rate *= 2;
	}
}models[pool_size];

int main(int argn, char** args) {
	int seed = 0;
	if (argn > 1)
		sscanf(args[1], "%d", &seed);
	else
		std::cin >> seed;
	srand(seed);
	valid_CNOTs_init();
	read_dataset(input_file_name);
	encoded_states_init();
	omp_set_num_threads(n_threads);

	sprintf(record_file_name, "Records_%10u_%d", (unsigned)time(nullptr), seed);
	freopen(record_file_name, "w", stdout);

	for (int i = 0; i < pool_size; i++)
		models[i].init();

	const int train_begin = 0, train_end = 800;
	const int valid_begin = 800, valid_end = 900;
	const int test_begin = 900, test_end = 1000;

	for (int epoch = 0; epoch < n_init; epoch++) {
		cerr << epoch << endl;
		for (int i = 0; i < pool_size; i++) {
			models[i].loss = models[i].get_loss(train_begin, train_end, true, &models[i].acc);
			cout<< i << '\t' << models[i].loss << '\t' << models[i].grad_norm << '\t'
				<< models[i].learning_rate << '\t' << models[i].acc << '\t' << measurements << endl;
			models[i].gradient_descent(train_begin, train_end);
		}
	}

	for (int epoch = 0; epoch < n_train; epoch++) {
		cerr << epoch << endl;
		int grad_stable = 1;
		for (int i = 0; i < pool_size; i++) {
			if (models[i].survived == 0)
				continue;
			models[i].loss = models[i].get_loss(train_begin, train_end, true, &models[i].acc);
			cout << i << '\t' << models[i].loss << '\t' << models[i].grad_norm << '\t'
				<< models[i].learning_rate << '\t' << models[i].acc << '\t' << measurements << endl;
			models[i].gradient_descent(train_begin, train_end);

			if (models[i].grad_norm > 1e-6)
				grad_stable = 0;

			models[i].valid_loss = models[i].get_loss(valid_begin, valid_end, false, &models[i].valid_acc);
			cout << i << '\t' << models[i].valid_loss << '\t' << models[i].valid_acc << '\t' << measurements << endl;
		}

		for (int i = 0; i < pool_size; i++) {
			for (int j = 0; j < pool_size; j++)
				if (
					i != j && models[i].survived && models[j].survived
					&& models[i].loss > models[j].loss
					&& models[i].valid_loss > models[j].valid_loss
					&& models[i].grad_norm < models[j].grad_norm
					&& models[i].acc < models[j].acc
					&& models[i].valid_acc < models[j].valid_acc
				)
					models[i].survived = 0;
		}

		if (grad_stable)
			break;

		if (epoch >= n_compete)
			continue;
		int n_survivals = pool_size;
		for (int i = 0; i < pool_size; i++) {
			if (models[i].survived)
				continue;
			n_survivals--;
			models[i].init();
		}
		cerr << n_survivals << endl;
	}

	puts("\nTest:");
	sprintf(output_file_name, "Final_result_%10u_%d", (unsigned)time(nullptr), seed);
	FILE* fout = fopen(output_file_name, "w");

	for (int i = 0; i < pool_size; i++)
		if (models[i].survived) {
			for (int j = 0; j < n_layers; j++)
				fprintf(fout, "%16llx ", models[i].sub[j].id),
				printf("%16llx ", models[i].sub[j].id);
			models[i].loss = models[i].get_loss(test_begin, test_end, false, &models[i].acc);
			fprintf(fout, "\nloss: %.6lf, acc: %.3lf\n", models[i].loss, models[i].acc);
			printf("\nloss: %.6lf, acc: %.3lf\n", models[i].loss, models[i].acc);
			for (int j = 0; j < n_params; j++)
				fprintf(fout, "%.15lf\n", models[i].param[j]);
		}

	fclose(fout);

	return 0;
}
