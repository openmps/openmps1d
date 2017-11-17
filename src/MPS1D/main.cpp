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

// ���q�����x
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

// ���R����
template<typename U, typename X, typename TYPE>
void Fall(U&& u, X&& x, const TYPE& type, const std::size_t count,
	const double g, const double dt)
{
	for (auto i = decltype(count)(0); i < count; ++i)
	{
		if (type[i] == Type::Water) // ���ȊO�͉^�����Ȃ�
		{
			u[i] += -g * dt;
			x[i] += u[i] * dt;
		}
	}
}

// ���͂̋���
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
	// ���̓|�A�\�����������쐬
	using Matrix = boost::numeric::ublas::matrix<double>;
	using Vector = boost::numeric::ublas::vector<double>;
	Matrix A(count, count);
	Vector b(count);
	Vector xx(count);
	for (auto i = decltype(count)(0); i < count; ++i)
	{
		if ((type[i] == Type::Dummy) || (n[i] < beta * n0)) // �_�~�[���q�Ǝ��R�\�ʂ͕������������Ȃ�
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
				if ((j != i) && (type[j] != Type::Dummy)) // �_�~�[���q�Ƃ͑��ݍ�p���Ȃ�
				{
					const auto r = std::abs(x[j] - x[i]);
					if (r < r_e)
					{
						constexpr auto DIM = double(1.0);
#ifdef MPS_HL
						const auto a_ij = (5 - DIM) * r_e / n0 / (r*r*r);
#else
						const auto w = r_e / r - 1; // �d�݊֐�
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

	// ������������
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

	// ������
	for (auto i = decltype(count)(0); i < count; ++i)
	{
		p[i] = std::max(b(i), 0.0); // �����͍l�����Ȃ�
	}
}

template<typename U, typename X, typename TYPE, typename P>
void ModifyByPressrureGradient(U&& u, X&& x, const TYPE& type, const P& p, const std::size_t count,
	const double n0, const double r_e, const double rho, const double dt)
{
	// ���x�C����
	std::remove_reference_t<U> du;
	for (auto i = decltype(count)(0); i < count; ++i)
	{
		if (type[i] == Type::Water) // ���ȊO�͉^�����Ȃ�
		{
			auto d = double(0);
			for (auto j = decltype(count)(0); j < count; ++j)
			{
				if ((j != i) && (type[j] != Type::Dummy)) // �_�~�[���q�Ƃ͑��ݍ�p���Ȃ�
				{
					const auto dx = x[j] - x[i];
					const auto r = std::abs(dx);
					if (r < r_e) // �e���͈͓�
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
		if (type[i] == Type::Water) // ���ȊO�͉^�����Ȃ�
		{
			u[i] += du[i];
			x[i] += du[i] * dt;
		}
	}
}

int main()
{
	// �ݒ�l
	constexpr auto COUNT = std::size_t(11); // ���q��
	constexpr auto l0 = double(1e-3); // �������q�ԋ���
	constexpr auto r_eByL0 = double(2.4); // �e�����a�Ə������q�ԋ����̔�
	constexpr auto g = double(9.8); // �d�͉����x
	constexpr auto dt = double(0.00000001); // ���ԍ���
	constexpr auto beta = double(0.98); // ���R�\�ʔ���W��
	constexpr auto rho = double(998.20); // ���̖��x
	constexpr auto OUTPUT_LOOP = std::size_t(100); // �o�͉�
	constexpr auto LOOP = std::size_t(1000); // �v�Z��


	// �v�Z�ɕK�v�ȃp�����[�^�[�Ȃ�
	constexpr auto r_e = r_eByL0 * l0;
	constexpr auto COUNT_RE = static_cast<decltype(COUNT)>(r_eByL0);
	constexpr auto N = COUNT + 1 + COUNT_RE;
	auto x = std::array<double, N>(); // �ʒu�x�N�g��
	auto u = std::array<double, N>(); // �����x�N�g��
	auto p = std::array<double, N>(); // ����
	auto n = std::array<double, N>(); // ���q�����x
	auto type = std::array<Type, N>();

	// ���q�z�u
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

	// ����q�����x
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

	// �������
	UpdateNeighborDensity(n, x, type, N,
		r_e);
	output();

	for (auto i = decltype(OUTPUT_LOOP)(0); i < OUTPUT_LOOP; ++i)
	{
		for (auto j = decltype(LOOP)(0); j < LOOP; ++j)
		{
			// ���R����
			Fall(u, x, type, N,
				g, dt);

			// ���q�����x
			UpdateNeighborDensity(n, x, type, N,
				r_e);

			// ���͂̋���
			SolvePressure(p, n, x,
#ifdef MPS_HS
				u,
#endif
				type, N,
				beta, n0, r_e, rho, dt, lambda);

			// ���͌��z
			ModifyByPressrureGradient(u, x, type, p, N,
				n0, r_e, rho, dt);
		}

		// �o��
		output();
	}

	return 0;
}
