//#define DEBUG_OUTPUT
#define MPS_HS
#define MPS_HL

#include <iostream>
#include <array>
#include <limits>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/lu.hpp>

enum class Type
{
	Water,
	Wall,
	Dummy
};

// 粒子数密度
template<typename N, typename X, typename TYPE>
void UpdateNeighborDensity(N&& n, const X& x, const TYPE& type, const std::size_t count,
	const double r_e)
{
	for (auto i = decltype(count)(0); i < count; ++i)
	{
		if (type[i] != Type::Dummy)
		{
			auto n_tmp = double(0);
			for (auto j = decltype(count)(0); j < count; ++j)
			{
				if (j != i)
				{
					const auto r = std::abs(x[j] - x[i]);
					if (r < r_e)
					{
						n_tmp += r_e / r - 1;
					}
				}
			}
			n[i] = n_tmp;
		}
	}
}

// 自由落下
template<typename U, typename X, typename TYPE>
void Fall(U&& u, X&& x, const TYPE& type, const std::size_t count,
	const double g, const double dt)
{
	for (auto i = decltype(count)(0); i < count; ++i)
	{
		if (type[i] == Type::Water) // 水以外は運動しない
		{
			u[i] += -g * dt;
			x[i] += u[i] * dt;
		}
	}
}

// 圧力の求解
template<typename P, typename N, typename X, typename TYPE
#ifdef MPS_HS
	, typename U
#endif
>
void SolvePressure(P&& p, const N& n, const X& x,
#ifdef MPS_HS
	const U& u,	
#endif
	const TYPE& type, const std::size_t count,
	const double beta, const double n0, const double r_e, const double rho, const double dt, const double lambda)
{
	// 圧力ポアソン方程式を作成
	using Matrix = boost::numeric::ublas::matrix<double>;
	using Vector = boost::numeric::ublas::vector<double>;
	Matrix A(count, count);
	Vector b(count);
	Vector xx(count);
	for (auto i = decltype(count)(0); i < count; ++i)
	{
		if ((type[i] == Type::Dummy) || (n[i] < beta * n0)) // ダミー粒子と自由表面は方程式を解かない
		{
			for (auto j = decltype(count)(0); j < count; ++j)
			{
				A(i, j) = (i == j) ? 1 : 0;
			}
			b(i) = 0;
			xx(i) = 0;
		}
		else
		{
			auto a_ii = double(0);
			for (auto j = decltype(count)(0); j < count; ++j)
			{
				if ((j != i) && (type[j] != Type::Dummy)) // ダミー粒子とは相互作用しない
				{
					const auto r = std::abs(x[j] - x[i]);
					if (r < r_e)
					{
						constexpr auto DIM = double(1.0);
#ifdef MPS_HL
						const auto a_ij = (5 - DIM) * r_e / n0 / (r*r*r);
#else
						const auto w = r_e / r - 1; // 重み関数
						const auto a_ij = (2 * DIM / (lambda * n0)) * w;
#endif
						A(i, j) = a_ij;
						a_ii += -a_ij;
					}
					else
					{
						A(i, j) = 0;
					}
				}
				else
				{
					A(i, j) = 0;
				}
			}
			A(i, i) = a_ii;

#ifdef MPS_HS
			auto b_i = double(0);
			for (auto j = decltype(count)(0); j < count; ++j)
			{
				if (j != i)
				{
					const auto dx = x[j] - x[i];
					const auto r = std::abs(dx);
					if (r < r_e)
					{
						const auto du = u[j] - u[i];
						b_i += dx * du / (r*r*r);
					}
				}
			}
			b(i) = -r_e * b_i;
#else
			b(i) = (n[i] - n0) / dt;
#endif
			b(i) *= -rho / (dt * n0);
		}
	}

#ifdef DEBUG_OUTPUT
	for (auto i = decltype(count)(0); i < count; ++i)
	{
		for (auto j = decltype(count)(0); j < count; ++j)
		{
			std::cout << A(i, j) << ", ";
		}
		std::cout << ",," << b(i) << std::endl;
	}
#endif

	// 方程式を解く
	auto tmp = boost::numeric::ublas::permutation_matrix<>(count);
	boost::numeric::ublas::lu_factorize(A, tmp);
	boost::numeric::ublas::lu_substitute(A, tmp, b);

#ifdef DEBUG_OUTPUT
	std::cout << std::endl;
	for (auto i = decltype(count)(0); i < count; ++i)
	{
		std::cout << b(i) << std::endl;
	}
#endif

	// 解を代入
	for (auto i = decltype(count)(0); i < count; ++i)
	{
		p[i] = std::max(b(i), 0.0); // 負圧は考慮しない
	}
}

template<typename U, typename X, typename TYPE, typename P>
void ModifyByPressrureGradient(U&& u, X&& x, const TYPE& type, const P& p, const std::size_t count,
	const double n0, const double r_e, const double rho, const double dt)
{
	// 速度修正量
	std::remove_reference_t<U> du;
	for (auto i = decltype(count)(0); i < count; ++i)
	{
		if (type[i] == Type::Water) // 水以外は運動しない
		{
			auto d = double(0);
			for (auto j = decltype(count)(0); j < count; ++j)
			{
				if ((j != i) && (type[j] != Type::Dummy)) // ダミー粒子とは相互作用しない
				{
					const auto dx = x[j] - x[i];
					const auto r = std::abs(dx);
					if (r < r_e) // 影響範囲内
					{
						const auto w = r_e / r - 1;
						d += (p[j] + p[i]) / (r*r) * w * dx;
					}
				}
			}
			constexpr auto DIM = double(1.0);
			du[i] = -(dt / rho * DIM / n0) * d;
		}
		else
		{
			du[i] = 0;
		}
	}

	for (auto i = decltype(count)(0); i < count; ++i)
	{
		if (type[i] == Type::Water) // 水以外は運動しない
		{
			u[i] += du[i];
			x[i] += du[i] * dt;
		}
	}
}

int main()
{
	// 設定値
	constexpr auto COUNT = std::size_t(11); // 粒子数
	constexpr auto l0 = double(1e-3); // 初期粒子間距離
	constexpr auto r_eByL0 = double(2.4); // 影響半径と初期粒子間距離の比
	constexpr auto g = double(9.8); // 重力加速度
	constexpr auto dt = double(0.00000001); // 時間刻み
	constexpr auto beta = double(0.98); // 自由表面判定係数
	constexpr auto rho = double(998.20); // 水の密度
	constexpr auto OUTPUT_LOOP = std::size_t(100); // 出力回数
	constexpr auto LOOP = std::size_t(1000); // 計算回数


	// 計算に必要なパラメーターなど
	constexpr auto r_e = r_eByL0 * l0;
	constexpr auto COUNT_RE = static_cast<decltype(COUNT)>(r_eByL0);
	constexpr auto N = COUNT + 1 + COUNT_RE;
	auto x = std::array<double, N>(); // 位置ベクトル
	auto u = std::array<double, N>(); // 流速ベクトル
	auto p = std::array<double, N>(); // 圧力
	auto n = std::array<double, N>(); // 粒子数密度
	auto type = std::array<Type, N>();

	// 粒子配置
	for (auto i = decltype(COUNT)(0); i < COUNT; ++i)
	{
		x[i]    = (COUNT - 1 - i) * l0;
		u[i]    = 0;
		p[i]    = 0;
		n[i]    = 0;
		type[i] = Type::Water;
	}
	for(auto i = decltype(N)(0); COUNT + i < N; ++i)
	{
		x[i + COUNT]    = (i + 1) * -l0;
		u[i + COUNT]    = 0;
		p[i + COUNT]    = 0;
		n[i + COUNT]    = 0;
		type[i + COUNT] = i == 0 ? Type::Wall : Type::Dummy;
	}

	// 基準粒子数密度
	auto n0 = double(0);
	auto lambda = double(0);
	for (auto i = decltype(COUNT_RE)(0); i < COUNT_RE; ++i)
	{
		const auto r = (i + 1)*l0;
		const auto w = r_e / r - 1;
		n0 += w;
		lambda += w * r * r;
	}
	n0 *= 2;
	lambda *= 2 / n0;

	const auto output = [&ar = x, N, n0]()
	{
		for (auto i = decltype(N)(0); i < N; ++i)
		{
			std::cout << ar[i] << ", ";
		}
		std::cout << std::endl;
	};

	// 初期状態
	UpdateNeighborDensity(n, x, type, N,
		r_e);
	output();

	for (auto i = decltype(OUTPUT_LOOP)(0); i < OUTPUT_LOOP; ++i)
	{
		for (auto j = decltype(LOOP)(0); j < LOOP; ++j)
		{
			// 自由落下
			Fall(u, x, type, N,
				g, dt);

			// 粒子数密度
			UpdateNeighborDensity(n, x, type, N,
				r_e);

			// 圧力の求解
			SolvePressure(p, n, x,
#ifdef MPS_HS
				u,
#endif
				type, N,
				beta, n0, r_e, rho, dt, lambda);

			// 圧力勾配
			ModifyByPressrureGradient(u, x, type, p, N,
				n0, r_e, rho, dt);
		}

		// 出力
		output();
	}

	return 0;
}
