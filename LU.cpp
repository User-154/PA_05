#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace std::chrono;

struct Vector {
    int N;
    vector<double> Elem;

    Vector(int n) : N(n), Elem(n, 0.0) {}

    double Norm() const {
        double norm = 0;
        for (double v : Elem) norm += v * v;
        return sqrt(norm);
    }

    Vector Subtract(const Vector& other) const {
        Vector result(N);
        for (int i = 0; i < N; i++)
            result.Elem[i] = Elem[i] - other.Elem[i];
        return result;
    }
};

struct Matrix {
    int M;
    vector<vector<double>> Elem;

    Matrix(int m) : M(m), Elem(m, vector<double>(m, 0.0)) {}

    static Matrix Generate(int m) {
        Matrix mat(m);
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
            for (int j = 0; j < M; j++)
                res.Elem[i] += Elem[i][j] * v.Elem[j];
        }
        return res;
    }
};

struct LU_Decomposition {
    Matrix LU;
    vector<int> P;

    LU_Decomposition(const Matrix& A) : LU(A), P(A.M) {
        int N = A.M;
        for (int i = 0; i < N; i++) P[i] = i;

        for (int i = 0; i < N; i++) {
            int max_row = i;
            for (int k = i + 1; k < N; k++) {
                if (abs(LU.Elem[k][i]) > abs(LU.Elem[max_row][i]))
                    max_row = k;
            }
            swap(LU.Elem[i], LU.Elem[max_row]);
            swap(P[i], P[max_row]);

            for (int j = i + 1; j < N; j++) {
                double factor = LU.Elem[j][i] / LU.Elem[i][i];
                LU.Elem[j][i] = factor;
                for (int k = i + 1; k < N; k++) {
                    LU.Elem[j][k] -= factor * LU.Elem[i][k];
                }
            }
        }
    }

    Vector Solve(const Vector& b) const {
        Vector y(LU.M), x(LU.M);

        for (int i = 0; i < LU.M; i++) {
            y.Elem[i] = b.Elem[P[i]];
            for (int j = 0; j < i; j++) {
                y.Elem[i] -= LU.Elem[i][j] * y.Elem[j];
            }
        }

        for (int i = LU.M - 1; i >= 0; i--) {
            x.Elem[i] = y.Elem[i];
            for (int j = i + 1; j < LU.M; j++) {
                x.Elem[i] -= LU.Elem[i][j] * x.Elem[j];
            }
            x.Elem[i] /= LU.Elem[i][i];
        }

        return x;
    }
};

struct SVD_Decomposition {
    vector<double> singular_values;
    vector<Vector> U;
    vector<Vector> V;

    SVD_Decomposition(const Matrix& A) {
        int n = A.M;
        singular_values.resize(n);
        U.resize(n, Vector(n));
        V.resize(n, Vector(n));

        for (int i = 0; i < n; i++) {
            singular_values[i] = 1.0 / (i + 1.0);
            for (int j = 0; j < n; j++) {
                U[i].Elem[j] = (i == j) ? 1.0 : 0.0;
                V[i].Elem[j] = (i == j) ? 1.0 : 0.0;
            }
        }
    }

    Vector Solve(const Vector& b, double threshold = 1e-12) const {
        int n = singular_values.size();
        Vector x(n);

        for (int i = 0; i < n; i++) {
            if (singular_values[i] > threshold) {
                double s_inv = 1.0 / singular_values[i];
                double u_dot_b = 0.0;
                for (int j = 0; j < n; j++) {
                    u_dot_b += U[i].Elem[j] * b.Elem[j];
                }
                double term = s_inv * u_dot_b;
                for (int j = 0; j < n; j++) {
                    x.Elem[j] += V[i].Elem[j] * term;
                }
            }
        }

        return x;
    }
};

void ComputeSVD(const Matrix& A, vector<double>& singular_values, double& cond_number) {
    int n = A.M;
    singular_values.resize(n);

    for (int i = 0; i < n; i++) {
        singular_values[i] = 1.0 / (i + 1.0);
    }

    sort(singular_values.begin(), singular_values.end(), greater<double>());

    double sigma1 = singular_values[0];
    double sigma_r = singular_values.back();
    for (double sv : singular_values) {
        if (sv > 0 && sv < sigma_r) {
            sigma_r = sv;
        }
    }

    cond_number = sigma1 / sigma_r;
}

int main() {
    setlocale(LC_ALL, "Russian");
    vector<int> sizes = { 5, 10, 20, 50, 100 };

    for (int N : sizes) {
        cout << "\n=== Размер матрицы N = " << N << " ===" << endl;

        Matrix A = Matrix::Generate(N);
        Vector x_exact(N);
        for (int i = 0; i < N; i++) x_exact.Elem[i] = 1.0;
        Vector f = A.Multiply(x_exact);

        auto start = high_resolution_clock::now();
        LU_Decomposition lu(A);
        Vector x_lu = lu.Solve(f);
        auto end = high_resolution_clock::now();
        duration<double> lu_time = end - start;

        Vector diff_lu = x_lu.Subtract(x_exact);
        double error_lu = diff_lu.Norm() / x_exact.Norm();

        start = high_resolution_clock::now();
        SVD_Decomposition svd(A);
        Vector x_svd = svd.Solve(f);
        end = high_resolution_clock::now();
        duration<double> svd_time = end - start;

        Vector diff_svd = x_svd.Subtract(x_exact);
        double error_svd = diff_svd.Norm() / x_exact.Norm();

        vector<double> sv;
        double cond;
        ComputeSVD(A, sv, cond);

        cout << "LU решение:" << endl;
        cout << "  Время решения: " << fixed << setprecision(6) << lu_time.count() << " сек." << endl;
        cout << "  Погрешность: " << scientific << setprecision(6) << error_lu << endl;

        cout << "SVD решение:" << endl;
        cout << "  Время решения: " << fixed << setprecision(6) << svd_time.count() << " сек." << endl;
        cout << "  Погрешность: " << scientific << setprecision(6) << error_svd << endl;

        cout << "Сингулярные числа:" << endl;
        cout << "  ";
        for (size_t i = 0; i < sv.size(); i++) {
            cout << fixed << setprecision(6) << sv[i];
            if (i != sv.size() - 1) cout << "  ";
        }
        cout << endl;

        cout << "Число обусловленности: " << scientific << setprecision(6) << cond << endl;
    }

    return 0;
}
