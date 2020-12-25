#ifndef MATHEMATICS_TESTS_HPP
#define MATHEMATICS_TESTS_HPP
#include <cstdio>
#include <cmath>
#include <string>
#include <stdexcept>
#include <functional>
#include <sstream>

namespace mathtests
{
    void RunTests();

	constexpr int SEED = 1234;

	struct TestRunner
	{
		template<typename TResult>
		struct callable_base { virtual TResult operator()() = 0; [[nodiscard]] virtual std::string arguments() const = 0; };

		template<typename TResult, typename... Args>
		struct Callable : callable_base<TResult>
		{
		private:
			std::tuple<Args...> args;
			std::function<TResult(Args...)> callable;

			template<size_t... N>
			TResult call(std::index_sequence<N...>)
			{
				return callable(std::get<N>(args)...);
			}

			template<size_t... N>
			std::string arguments(std::index_sequence<N...>)
			{
				std::stringstream stream;
				((stream << std::get<N>(args)),...);
				return stream.str();
			}
		public:

			TResult operator()() override
			{
				call(std::make_index_sequence<sizeof...(Args)>{});
			}

			[[nodiscard]] std::string arguments() const override
			{
				return arguments(std::make_index_sequence<sizeof...(Args)>{});
			}
		};

		template<typename TResult>
		struct Callable<TResult> : callable_base<TResult>
		{
		private:
			std::function<TResult()> callable;
		public:

			TResult operator()() override
			{
				return callable();
			}

			[[nodiscard]] std::string arguments() const override
			{
				return "none";
			}
		};

		struct Fail : std::runtime_error { using runtime_error::runtime_error; };
		struct Case
		{
			char const* name;
			bool verbose{false};

			template<size_t N>
			explicit Case(char (&name)[N]) : name(name) {}

			template<size_t N>
			explicit Case(char (&name)[N], bool verbose) : name(name), verbose(verbose) {}

			template<typename F, typename Equality>
			void assert_equals(F&& actual, F&& expected, Equality&& equals)
			{
				auto expected_val = expected();
				auto actual_val = actual();
				if (!equals(expected, actual))
				{
					char buf[1024];
					fprintf(stderr, "%s failed!\nExpected: %s\nActual:%s\n", name, stringify(expected_val).c_str(), stringify(actual_val).c_str());
					throw Fail(name);
				}
				fprintf(stdout, "%s passed!\n");
			}

			template<typename TResult, typename Equality>
			void assert_equals(callable_base<TResult>& actual, callable_base<TResult>& expected, Equality&& equals)
			{
				auto expected_val = expected();
				auto actual_val = actual();
				if (verbose)
				{
					fprintf(stdout, "[Actual] Arguments: %s\n", actual.arguments().c_str());
					fprintf(stdout, "[Expected] Arguments: %s\n", expected.arguments().c_str());
				}
				if (!equals(expected, actual))
				{
					char buf[1024];
					fprintf(stderr, "%s failed!\nExpected: %s\nActual:%s\n", name, stringify(expected_val).c_str(), stringify(actual_val).c_str());
					throw Fail(name);
				}
				fprintf(stdout, "%s passed!\n");
			}

			template<typename T>
			std::string stringify(T&& arg) { return arg; }
		};


	};
}

#endif //MATHEMATICS_TESTS_HPP
