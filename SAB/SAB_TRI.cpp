#define _CRT_SECURE_NO_WARNINGS

#include "qcirc.h"
#include <iostream>
#include <algorithm>
#include <ctime>
#include <cstring>
#include <omp.h>

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
const int n_epochs = 2048, n_searches = 2048, n_finals = 16384;
const int warmup_epochs = 20;
const double learning_rate = 0.5;
const int n_threads = 16;

const double threshold = log(1.0 / 2.0 + 0.001);

bool adjacency_matrix[n_graphs][n_nodes][n_nodes];
bool has_triangle[n_graphs];
char output_file_name[2048];
char record_file_name[2048];

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
	for (int i = 0; i < n_graphs; i++)
		for (int j = 0; j < n_nodes; j++)
			for (int k = 0; k < n_nodes; k++)
				for (;;) {
					int c = fgetc(fin);
					if (isdigit(c)) {
						adjacency_matrix[i][j][k] = c - '0';
						break;
					}
				}
	for (int i = 0; i < n_graphs; i++)
		for (;;) {
			int c = fgetc(fin);
			if (isdigit(c)) {
				has_triangle[i] = c - '0';
				break;
			}
		}
	fclose(fin);
}

void encode_layer(autoGrad_state& qs, int graph_idx) {
	for (int i = 0; i < n_nodes; i++)
		qs.h_gate(i);
	for (int i = 0; i < n_nodes; i++)
		for (int j = i + 1; j < n_nodes; j++)
			if (adjacency_matrix[graph_idx][i][j])
				qs.cz_gate(i, j);
	for (int i = 0; i < n_nodes; i++)
		qs.cx_gate(i, n_nodes);
}

void qas_layer(autoGrad_state& qs, double params[], int subnet_id, int paramCount = -1) {
	for (int i = 0, Rs_id = subnet_id >> n_CXs; i < n_qBits; i++, Rs_id /= 3) {
		int paramIdx = paramCount == -1 ? -1 : paramCount + i;
		switch (Rs_id % 3) {
			case 0:
				qs.rx_param_gate(i, params[i], paramIdx, paramIdx + 1); break;
			case 1:
				qs.ry_param_gate(i, params[i], paramIdx, paramIdx + 1); break;
			case 2:
				qs.rz_param_gate(i, params[i], paramIdx, paramIdx + 1); break;
		}
	}
	paramCount = paramCount == -1 ? 0 : paramCount + n_qBits;
	for (int i = 0; i < n_CXs; i++, subnet_id >>= 1)
		if (subnet_id & 1)
			qs.cx_gate(valid_CNOTs[i][0], valid_CNOTs[i][1], paramCount);
}

struct model {
	int n_experts;
	int expert_idx;
	int subnet_ids[n_layers];
	double(*params)[n_layers][n_qBits];

	model(int init_n_experts, int* subnet = nullptr) {
		n_experts = init_n_experts;
		expert_idx = 0;
		if (subnet == nullptr)
			memset(subnet_ids, 0, sizeof(subnet_ids));
		else
			memcpy(subnet_ids, subnet, sizeof(subnet_ids));
		params = new double[n_experts][n_layers][n_qBits];
		for (int i = 0; i < n_experts; i++)
			for (int j = 0; j < n_layers; j++)
				for (int k = 0; k < n_qBits; k++)
					params[i][j][k] = ull_rand() * 2.0 * Pi / ull_rand_max;
	}

	~model() {
		delete[] params;
	}

	void run(autoGrad_state& qs, int graph_idx, bool need_grad = 0)const {
		encode_layer(qs, graph_idx);
		for (int i = 0; i < n_layers; i++) {
			int paramCount = need_grad ? n_qBits * i : -1;
			qas_layer(qs, params[expert_idx][i], subnet_ids[i], paramCount);
		}
	}
};

autoGrad_state state[100];
double temp_exp[100];
double temp_grad[100][n_params];

double get_expval(autoGrad_state& qs, const model& mdl, int graph_idx, double* grad = nullptr) {
	bool need_grad = grad != nullptr;
	if (need_grad)
		memset(grad, 0, sizeof(double) * n_params);
	qs.load();
	mdl.run(qs, graph_idx, need_grad);
	double expval = 0;
	for (int i = 0; i < (1 << n_qBits); i++) {
		int label = 0;
		for (int j = i; j > 0; j >>= 1)
			label ^= j;
		if ((label & 1) != has_triangle[graph_idx])
			continue;
		expval += std::norm(qs.qVec[i]);
		if (need_grad)
			for (int j = 0; j < n_params; j++)
				grad[j] += 2 * qs.qVec[i].real() * qs.grad[i][j].real(),
				grad[j] += 2 * qs.qVec[i].imag() * qs.grad[i][j].imag();
	}
	return expval;
}

int measurements = 0;

double get_loss(const model& mdl, int input_begin, int input_end, double* grad = nullptr, double* acc = nullptr) {
	measurements++;
	double loss = 0;
	double correct = 0;
	bool need_grad = grad != nullptr;
	if (need_grad)
		measurements += 2 * n_params,
		memset(grad, 0, sizeof(double) * n_params);

	for (int p = input_begin; p < input_end; p += 100) {
#pragma omp parallel for
		for (int i = 0; i < 100; i++) {
			if (need_grad) {
				temp_exp[i] = get_expval(state[i], mdl, p + i, temp_grad[i]);
				for (int j = 0; j < n_params; j++)
					temp_grad[i][j] /= temp_exp[i];
			}
			else
				temp_exp[i] = get_expval(state[i], mdl, p + i);
			temp_exp[i] = log(temp_exp[i]);
		}

		for (int i = 0; i < 100; i++) {
			if (need_grad)
				for (int j = 0; j < n_params; j++)
					grad[j] -= temp_grad[i][j];
			loss -= temp_exp[i];
			if (temp_exp[i] > threshold)
				correct++;
		}
	}

	if (acc != nullptr)
		*acc = correct / (input_end - input_begin);
	if (need_grad) {
		for (int i = 0; i < n_params; i++)
			grad[i] /= (input_end - input_begin);
	}
	return loss / (input_end - input_begin);
}

int expert_evaluator(model& mdl, int input_begin, int input_end) {
	int target_expert = 0;
	double target_loss = -1;
	for (int i = 0; i < mdl.n_experts; i++) {
		mdl.expert_idx = i;
		double loss = get_loss(mdl, input_begin, input_end);
		if (target_loss < 0 || loss < target_loss) {
			target_loss = loss;
			target_expert = i;
		}
	}
	return target_expert;
}

struct search_result {
	int subnet_ids[n_layers];
	double loss;
	double acc;
}results[n_searches];

bool operator<(const search_result& r1, const search_result& r2) {
	return r1.loss < r2.loss;
}

void get_best_circuit() {
	model mdl(5);
	const int search_begin = 0, search_end = 400;

	for (int epoch = 0; epoch < n_epochs; epoch++) {
		cerr << epoch << endl;
		for (int i = 0; i < n_layers; i++)
			mdl.subnet_ids[i] = ull_rand() % (n_Rs << n_CXs);
		mdl.expert_idx = epoch % mdl.n_experts;
		if (epoch >= warmup_epochs)
			mdl.expert_idx = expert_evaluator(mdl, search_begin, search_end);

		double grad[n_params];
		get_loss(mdl, search_begin, search_end, grad);
		double norm_grad = 0;
		for (int i = 0; i < n_layers; i++)
			for (int j = 0; j < n_qBits; j++) {
				mdl.params[mdl.expert_idx][i][j] -= learning_rate * grad[i * n_qBits + j];
				norm_grad += grad[i * n_qBits + j] * grad[i * n_qBits + j];
			}
		
		double acc;
		double loss = get_loss(mdl, search_begin, search_end, nullptr, &acc);
		cout << loss << '\t' << acc << '\t' << norm_grad << '\t' << measurements << endl;
	}

	for (int i = 0; i < n_searches; i++) {
		for (int i = 0; i < n_layers; i++)
			mdl.subnet_ids[i] = ull_rand() % (n_Rs << n_CXs);
		mdl.expert_idx = expert_evaluator(mdl, search_begin, search_end);
		results[i].loss = get_loss(mdl, search_begin, search_end, nullptr, &results[i].acc);
		memcpy(results[i].subnet_ids, mdl.subnet_ids, sizeof(mdl.subnet_ids));
	}

	std::sort(results, results + n_searches);
}

int main(int argn, char** argv) {
	int seed;
	if (argn != 1)
		sscanf(argv[1], "%d", &seed);
	else
		scanf("%d", &seed);
	srand(seed);
	valid_CNOTs_init();
	read_dataset(input_file_name);
	omp_set_num_threads(n_threads);

	sprintf(record_file_name, "Records_%10u_%d", (unsigned)time(nullptr), seed);
	freopen(record_file_name, "w", stdout);

	get_best_circuit();

	model mdl(1, results[0].subnet_ids);
	const int train_begin = 400, train_end = 800;
	const int valid_begin = 800, valid_end = 1000;

	for (int epoch = 0; epoch < n_finals; epoch++) {
		cerr << epoch << endl;
		double grad[n_params];
		get_loss(mdl, train_begin, train_end, grad);
		double norm_grad = 0;
		for (int i = 0; i < n_layers; i++)
			for (int j = 0; j < n_qBits; j++) {
				mdl.params[0][i][j] -= learning_rate * grad[i * n_qBits + j];
				norm_grad += grad[i * n_qBits + j] * grad[i * n_qBits + j];
			}
		double acc;
		double loss = get_loss(mdl, train_begin, train_end, nullptr, &acc);
		cout << loss << '\t' << acc << '\t' << norm_grad << '\t' << measurements << endl;
		if (norm_grad < 1e-6)
			break;
	}

	puts("\nValid:");
	double acc;
	double loss = get_loss(mdl, valid_begin, valid_end, nullptr, &acc);
	printf("loss: %.7lf, acc: %.3lf\n", loss, acc);

	sprintf(output_file_name, "Final_result_%10u_%d", (unsigned)time(nullptr), seed);
	FILE* fout = fopen(output_file_name, "w");
	for (int i = 0; i < n_layers; i++)
		fprintf(fout, "%8x ", mdl.subnet_ids[i]);
	for (int i = 0; i < n_layers; i++) {
		fputs("\n", fout);
		for (int j = 0; j < n_qBits; j++)
			fprintf(fout, "%.15lf\n", mdl.params[0][i][j]);
	}
	fclose(fout);
	return 0;
}
