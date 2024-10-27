#pragma once

#include <complex>
#include <iostream>
#include <cstdlib>
#include "qrand.h"

const double Pi = acos(-1.0);

typedef std::complex<double> qValue;
const qValue Im = qValue(0, 1);

template<int qN, int paramN = 1>
struct qState {
	static const int vecSize = 1 << qN;
	qValue qVec[vecSize];
	qValue grad[vecSize][paramN];

	qState(const qValue* init_list = nullptr) {
		load(init_list);
	}

	void clear_grad() {
		memset(grad, 0, sizeof(grad));
	}

	void load(const qValue* load_list = nullptr) {
		clear_grad();
		if (load_list == nullptr) {
			memset(qVec, 0, sizeof(qVec));
			qVec[0] = 1;
			return;
		}
		memcpy(qVec, load_list, sizeof(qVec));
	}

	void copy(qValue* copy_list) {
		memcpy(copy_list, qVec, sizeof(qVec));
	}

	void single_gate_base(int qBit, const qValue mat[2][2], const qValue matGrad[2][2], int paramIdx, int paramCount) {
		for (int i = 0; i < vecSize; i++, i += i & 1 << qBit) {
			int j = i | 1 << qBit;
			for (int k = 0; k < paramCount; k++) {
				qValue g0 = mat[0][0] * grad[i][k] + mat[0][1] * grad[j][k];
				qValue g1 = mat[1][0] * grad[i][k] + mat[1][1] * grad[j][k];
				if (k == paramIdx && matGrad != nullptr) {
					g0 += matGrad[0][0] * qVec[i] + matGrad[0][1] * qVec[j];
					g1 += matGrad[1][0] * qVec[i] + matGrad[1][1] * qVec[j];
				}
				grad[i][k] = g0;
				grad[j][k] = g1;
			}
			qValue y0 = mat[0][0] * qVec[i] + mat[0][1] * qVec[j];
			qValue y1 = mat[1][0] * qVec[i] + mat[1][1] * qVec[j];
			qVec[i] = y0;
			qVec[j] = y1;
		}
	}

	void h_gate(int qBit, int paramCount = 0) {
		double m = sqrt(0.5);
		for (int i = 0; i < vecSize; i++, i += i & 1 << qBit) {
			int j = i | 1 << qBit;
			for (int k = 0; k < paramCount; k++) {
				qValue g0 = grad[i][k] + grad[j][k];
				qValue g1 = grad[i][k] - grad[j][k];
				grad[i][k] = m * g0;
				grad[j][k] = m * g1;
			}
			qValue y0 = qVec[i] + qVec[j];
			qValue y1 = qVec[i] - qVec[j];
			qVec[i] = m * y0;
			qVec[j] = m * y1;
		}
	}

	void x_gate(int qBit, int paramCount = 0) {
		for (int i = 0; i < vecSize; i++, i += i & 1 << qBit) {
			int j = i | 1 << qBit;
			for (int k = 0; k < paramCount; k++)
				std::swap(grad[i][k], grad[j][k]);
			std::swap(qVec[i], qVec[j]);
		}
	}

	void y_gate(int qBit, int paramCount = 0) {
		for (int i = 0; i < vecSize; i++, i += i & 1 << qBit) {
			int j = i | 1 << qBit;
			for (int k = 0; k < paramCount; k++) {
				qValue g = grad[i][k];
				grad[i][k] = -Im * grad[j][k];
				grad[j][k] = Im * g;
			}
			qValue y = qVec[i];
			qVec[i] = -Im * qVec[j];
			qVec[j] = Im * y;
		}
	}

	void z_gate(int qBit, int paramCount = 0) {
		for (int i = 0; i < vecSize; i++, i += i & 1 << qBit) {
			int j = i | 1 << qBit;
			for (int k = 0; k < paramCount; k++)
				grad[j][k] = -grad[j][k];
			qVec[j] = -qVec[j];
		}
	}

	void rx_param_gate(int qBit, double param, int paramIdx = -1, int paramCount = 0) {
		double sin_t = sin(0.5 * param);
		double cos_t = cos(0.5 * param);
		for (int i = 0; i < vecSize; i++, i += i & 1 << qBit) {
			int j = i | 1 << qBit;
			for (int k = 0; k < paramCount; k++) {
				qValue g0 = cos_t * grad[i][k] - Im * sin_t * grad[j][k];
				qValue g1 = -Im * sin_t * grad[i][k] + cos_t * grad[j][k];
				if (k == paramIdx) {
					g0 += -0.5 * (sin_t * qVec[i] + Im * cos_t * qVec[j]);
					g1 += -0.5 * (Im * cos_t * qVec[i] + sin_t * qVec[j]);
				}
				grad[i][k] = g0;
				grad[j][k] = g1;
			}
			qValue y0 = cos_t * qVec[i] - Im * sin_t * qVec[j];
			qValue y1 = -Im * sin_t * qVec[i] + cos_t * qVec[j];
			qVec[i] = y0;
			qVec[j] = y1;
		}
	}

	void ry_param_gate(int qBit, double param, int paramIdx = -1, int paramCount = 0) {
		double sin_t = sin(0.5 * param);
		double cos_t = cos(0.5 * param);
		for (int i = 0; i < vecSize; i++, i += i & 1 << qBit) {
			int j = i | 1 << qBit;
			for (int k = 0; k < paramCount; k++) {
				qValue g0 = cos_t * grad[i][k] - sin_t * grad[j][k];
				qValue g1 = sin_t * grad[i][k] + cos_t * grad[j][k];
				if (k == paramIdx) {
					g0 += -0.5 * (sin_t * qVec[i] + cos_t * qVec[j]);
					g1 += 0.5 * (cos_t * qVec[i] - sin_t * qVec[j]);
				}
				grad[i][k] = g0;
				grad[j][k] = g1;
			}
			qValue y0 = cos_t * qVec[i] - sin_t * qVec[j];
			qValue y1 = sin_t * qVec[i] + cos_t * qVec[j];
			qVec[i] = y0;
			qVec[j] = y1;
		}
	}

	void rz_param_gate(int qBit, double param, int paramIdx = -1, int paramCount = 0) {
		qValue exp_t = exp(0.5 * Im * param);
		qValue exp_neg_t = exp(-0.5 * Im * param);
		for (int i = 0; i < vecSize; i++, i += i & 1 << qBit) {
			int j = i | 1 << qBit;
			for (int k = 0; k < paramCount; k++) {
				grad[i][k] *= exp_neg_t;
				grad[j][k] *= exp_t;
				if (k == paramIdx) {
					grad[i][k] += -0.5 * Im * exp_neg_t * qVec[i];
					grad[j][k] += 0.5 * Im * exp_t * qVec[j];
				}
			}
			qVec[i] *= exp_neg_t;
			qVec[j] *= exp_t;
		}
	}

	void cx_gate(int ctrl, int targ, int paramCount = 0) {
		if (ctrl == targ)
			return;
		int qBit0 = std::min(ctrl, targ);
		int qBit1 = std::max(ctrl, targ);
		for (int i = 0; i < vecSize; i++, i += i & 1 << qBit0, i += i & 1 << qBit1) {
			int j01 = i | 1 << ctrl;
			int j11 = i | 1 << ctrl | 1 << targ;
			for (int k = 0; k < paramCount; k++)
				std::swap(grad[j01][k], grad[j11][k]);
			std::swap(qVec[j01], qVec[j11]);
		}
	}

	void cz_gate(int qBit0, int qBit1, int paramCount = 0) {
		if (qBit0 == qBit1)
			return;
		if (qBit0 > qBit1)
			std::swap(qBit0, qBit1);
		for (int i = 0; i < vecSize; i++, i += i & 1 << qBit0, i += i & 1 << qBit1) {
			int j = i | 1 << qBit0 | 1 << qBit1;
			for (int k = 0; k < paramCount; k++)
				grad[j][k] = -grad[j][k];
			qVec[j] = -qVec[j];
		}
	}

	void crz_gate(int ctrl, int targ, double param, int paramCount = 0) {
		if (ctrl == targ)
			return;
		int qBit0 = std::min(ctrl, targ);
		int qBit1 = std::max(ctrl, targ);
		qValue exp_t = exp(0.5 * Im * param);
		qValue exp_neg_t = exp(-0.5 * Im * param);
		for (int i = 0; i < vecSize; i++, i += i & 1 << qBit0, i += i & 1 << qBit1) {
			int j01 = i | 1 << ctrl;
			int j11 = i | 1 << ctrl | 1 << targ;
			for (int k = 0; k < paramCount; k++) {
				grad[j01][k] *= exp_neg_t;
				grad[j11][k] *= exp_t;
			}
			qVec[j01] *= exp_neg_t;
			qVec[j11] *= exp_t;
		}
	}
};

template<int qN, int paramN>
std::ostream& operator<<(std::ostream& out, const qState<qN, paramN>& qs) {
	for (int i = 0; i < qs.vecSize; i++)
		out << qs.qVec[i] << '\n';
	return out;
}