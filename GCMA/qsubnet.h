#pragma once
#include <vector>
#include <cstring>
#include <iostream>
#include <array>
#include "qrand.h"

//	0: CX	1: RY	2: RZ
union subnet_union {
	int64_t id;
	short state[4];
	subnet_union(int64_t id = 0) :id(id) { }
};

template<int qN>
struct subnet {
	subnet_union sub;
	short genes[4];

	subnet() {
		memset(genes, -1, sizeof(genes));
	}

	void set_genes(const short parent_genes[]) {
		if (sub.id == 0) {
			memset(genes, 0, sizeof(genes));
			return;
		}
		genes[0] = parent_genes[0];
		for (int i = 0; i < qN; i++) {
			mutation_cx(i);
			mutation_ry(i);
			mutation_rz(i);
		}
	}

	void mutation_cx(int qBit) {
		if (sub.state[0] & 1 << qBit) {
			genes[0] &= ~(1 << qBit);
			return;
		}
		if (sub.state[0] & (1 << (qBit + 1) % qN | 1 << (qBit + qN - 1) % qN))
			genes[0] |= 1 << qBit;
		if ((sub.state[1] | sub.state[2]) & (1 << qBit | 1 << (qBit + 1) % qN))
			genes[0] |= 1 << qBit;
	}

	void mutation_ry(int qBit) {
		if (sub.state[0] & (1 << qBit | 1 << (qBit + qN - 1) % qN)) {
			genes[1] |= 1 << qBit;
			return;
		}
		if (sub.state[2] & 1 << qBit)
			genes[1] |= 1 << qBit;
		else
			genes[1] &= ~(1 << qBit);
	}

	void mutation_rz(int qBit) {
		if (sub.state[0] & 1 << (qBit + qN - 1) % qN) {
			genes[2] |= 1 << qBit;
			return;
		}
		if (sub.state[1] & 1 << qBit)
			genes[2] |= 1 << qBit;
		else
			genes[2] &= ~(1 << qBit);
	}
};

template<int qN>
subnet_union random_r_gate_generate(const subnet<qN>& p) {
	subnet_union r_gate = 0;
	for (int i = 0; i < qN; i++) {
		subnet_union pool[3];
		int pool_size = 1;
		if (p.genes[1] & 1 << i)
			pool[pool_size++].state[1] = 1 << i;
		if (p.genes[2] & 1 << i)
			pool[pool_size++].state[2] = 1 << i;
		r_gate.id |= pool[rand() % pool_size].id;
	}
	return r_gate;
}

void cx_pool_generate(int* begin, int* end, int qN, int64_t msk, std::vector<int64_t>& v) {
	if (begin >= end) {
		v.push_back(msk);
		return;
	}
	cx_pool_generate(begin + 1, end, qN, msk, v);
	int64_t m = 1ll << begin[0];
	if (begin[0] == 0 && end[-1] == qN - 1)
		end--;
	if (begin[1] == begin[0] + 1)
		begin++;
	cx_pool_generate(begin + 1, end, qN, msk | m, v);
}

template<int qN>
subnet_union random_cx_gate_generate(const subnet<qN>& p, subnet_union r_gate) {
	short cx_gene = p.genes[0];
	int available_cx[qN];
	int available_cx_size = 0;
	std::vector<int64_t> pool;
	for (int i = 0; i < qN; i++) {
		if ((r_gate.state[1] | r_gate.state[2]) & (1 << i | 1 << (i + 1) % qN))
			cx_gene |= 1 << i;
		if (cx_gene & 1 << i)
			available_cx[available_cx_size++] = i;;
	}
	cx_pool_generate(available_cx, available_cx + available_cx_size, qN, 0, pool);
	return pool[rand() % pool.size()];
}

template<int qN>
subnet<qN> random_subnet_generate(const subnet<qN>& p) {
	subnet<qN> n;
	n.sub = random_r_gate_generate(p);
	n.sub.id |= random_cx_gate_generate(p, n.sub.id).id;
	n.set_genes(p.genes);
	return n;
}

template<int qN>
std::ostream& operator<<(std::ostream& out, const subnet<qN>& n) {
	for (int i = 0; i < qN; i++) {
		if (n.sub.state[2] & 1 << i)
			out << 'Z';
		else if (n.sub.state[1] & 1 << i)
			out << 'Y';
		else
			out << ' ';
		out << ' ';
		if (n.sub.state[0] & 1 << i)
			out << 'C';
		else if (n.sub.state[0] & 1 << (i + qN - 1) % qN)
			out << 'X';
		else
			out << ' ';
		out << '\n';
	}
	for (int i = 0; i < 16; i++)
		out << '-';
	return out;
}