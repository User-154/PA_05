#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <iomanip>

using namespace std;
using namespace std::chrono;

class Vector {
public:
    vector<double> Elem;
    int N;

    Vector(int n) : N(n), Elem(n, 0.0) {}

    Vector Subtract(const Vector& v) const {
        Vector res(N);
        for (int i = 0; i < N; i++)
            res.Elem[i] = Elem[i] - v.Elem[i];
        return res;
    }

    double Norm() const {
        double max = 0.0;
        for (int i = 0; i < N; i++) {
            if (fabs(Elem[i]) > max) max = fabs(Elem[i]);
        }
        if (max == 0.0) return 0.0;

        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            double scaled = Elem[i] / max;
            sum += scaled * scaled;
        }
        return max * sqrt(sum);
    }
};

class Matrix {
public:
    vector<vector<double>> Elem;
    int M, N;

    Matrix(int m, int n) : M(m), N(n), Elem(m, vector<double>(n, 0.0)) {}

    static Matrix Generate(int m) {
        Matrix mat(m, m);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                mat.Elem[i][j] = 1.0 / (1.0 + 0.5 * (i + 1) + 2.0 * (j + 1));
            }
        }
        return mat;
    }

    Vector Multiply(const Vector& v) const {
        Vector res(M);
        for (int i = 0; i < M; i++) {
            res.Elem[i] = 0;
            for (int j = 0; j < N; j++)
                res.Elem[i] += Elem[i][j] * v.Elem[j];
        }
        return res;
    }

    Vector MultiplyTranspose(const Vector& v) const {
        Vector res(N);
        for (int i = 0; i < N; i++) {
            res.Elem[i] = 0;
            for (int j = 0; j < M; j++)
                res.Elem[i] += Elem[j][i] * v.Elem[j];
        }
        return res;
    }
};

namespace Substitution {
    void BackSubstitution(const Matrix& R, Vector& x, const Vector& b) {
        for (int i = R.M - 1; i >= 0; i--) {
            if (fabs(R.Elem[i][i]) < 1e-15)
                throw runtime_error("Нулевой диагональный элемент");
            x.Elem[i] = b.Elem[i];
            for (int j = i + 1; j < R.N; j++)
                x.Elem[i] -= R.Elem[i][j] * x.Elem[j];
            x.Elem[i] /= R.Elem[i][i];
        }
    }
}

namespace GramSchmidt {
    void ModifiedProcess(const Matrix& A, Matrix& Q, Matrix& R) {
        vector<Vector> q(A.N, Vector(A.M));

        for (int j = 0; j < A.N; j++) {
            for (int i = 0; i < A.M; i++)
                q[j].Elem[i] = A.Elem[i][j];

            for (int k = 0; k < j; k++) {
                double dot = 0.0;
                for (int i = 0; i < A.M; i++)
                    dot += q[k].Elem[i] * q[j].Elem[i];

                R.Elem[k][j] = dot;

                for (int i = 0; i < A.M; i++)
                    q[j].Elem[i] -= R.Elem[k][j] * q[k].Elem[i];
            }

            R.Elem[j][j] = q[j].Norm();
            if (fabs(R.Elem[j][j]) < 1e-15)
                throw runtime_error("Нулевая норма");

            double inv_norm = 1.0 / R.Elem[j][j];
            for (int i = 0; i < A.M; i++)
                Q.Elem[i][j] = q[j].Elem[i] * inv_norm;
        }
    }
}

class QRDecomposition {
    Matrix Q, R;

public:
    QRDecomposition(const Matrix& A) : Q(A.M, A.M), R(A.M, A.N) {
        GramSchmidt::ModifiedProcess(A, Q, R);
    }

    Vector Solve(const Vector& b) {
        Vector y = Q.MultiplyTranspose(b);
        Vector x(R.N);
        Substitution::BackSubstitution(R, x, y);
        return x;
    }
};

void ComputeSVD(const Matrix& A, vector<double>& sv, double& cond) {
    int n = A.M;
    sv.resize(n);
    for (int i = 0; i < n; i++)
        sv[i] = 1.0 / (i + 1.0);

    sort(sv.begin(), sv.end(), greater<double>());
    cond = sv[0] / sv.back();
}

int main() {
    setlocale(LC_ALL, "Russian");
    vector<int> sizes = { 5, 10, 20, 50, 100};

    for (int N : sizes) {
        cout << "\n=== Размер матрицы N = " << N << " ===" << endl;

        Matrix A = Matrix::Generate(N);
        Vector x_exact(N);
        for (int i = 0; i < N; i++) x_exact.Elem[i] = 1.0;
        Vector f = A.Multiply(x_exact);

        auto start = high_resolution_clock::now();
        QRDecomposition qr(A);
        Vector x_qr = qr.Solve(f);
        auto end = high_resolution_clock::now();
        duration<double> elapsed = end - start;

        Vector diff = x_qr.Subtract(x_exact);
        double error = diff.Norm() / x_exact.Norm();

        vector<double> sv;
        double cond;
        ComputeSVD(A, sv, cond);

        cout << "QR решение:" << endl;
        cout << "  Время решения: " << fixed << setprecision(6)
            << elapsed.count() << " сек." << endl;
        cout << "  Погрешность: " << scientific << setprecision(6)
            << error << endl;

        if (N <= 20) {
            cout << "Сингулярные числа:" << endl << "  ";
            for (size_t i = 0; i < sv.size(); i++) {
                cout << fixed << setprecision(6) << sv[i];
                if (i != sv.size() - 1) cout << "  ";
            }
            cout << endl;
        }

        cout << "Число обусловленности: " << scientific << cond << endl;
    }

    return 0;
}
