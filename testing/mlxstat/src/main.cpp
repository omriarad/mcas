#include <boost/program_options.hpp>
#include <common/logging.h>
#include <gsl/span>
#include <chrono>
#include <cstddef> /* uint16_t, uint64_t */
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>

struct field
{
	const char *name;
	std::uint64_t base;
	field(const char *name_)
		: name(name_)
		, base(0)
	{}
};

enum print_e
{
	pr_err,
	pr_log,
	pr_cout
};

namespace
{
	print_e destination = pr_cout;

	void print(const std::string &s)
	{
		switch ( destination )
		{
		case pr_err:
			::fprintf(stderr, "%s\n", s.c_str()); fflush(stderr);
			break;
		case pr_log:
			PLOG("%s", s.c_str()); ::fflush(stderr);
			break;
		default:
			std::cout << s << std::endl;
			break;
		}
	}

	void line(const std::string counters_dir_, std::string pfx, const gsl::span<field> fields_, const bool delta_)
	{
		std::ostringstream os;
		for ( auto &f : fields_ )
		{
			std::ifstream fs(counters_dir_ + "/" + f.name);
			std::uint64_t st = 2;
			fs >> st;
			os << pfx;
			if ( fs.good() )
			{
				os << st - (delta_ ? f.base : 0);
				f.base = st;
			}
			else
			{
				os << "-";
			}
			pfx = " ";
		}
		print(os.str());
	}

	void header_line(gsl::span<field> fields_)
	{
		std::ostringstream os;
		std::string sep;
		for ( auto d : fields_ )
		{
			os << sep << d.name;
			sep = " ";
		}
		print(os.str());
	}

	void utc_line()
	{
		timespec ts;
		timespec_get(&ts, TIME_UTC);
		std::ostringstream os;
		os << "UTC time " << std::fixed << double(double(ts.tv_sec) + double(ts.tv_nsec)/1e9);
		print(os.str());
	}
}

int main(int argc, char *argv[])
{
	namespace po = boost::program_options;

	try {
		po::options_description desc("Options");

		desc.add_options()
			("help", "Show help")
		        ("device", po::value<std::string>()->default_value("mlx5_0"), "Device (e.g. mlx5_0)")
		        ("port", po::value<std::uint16_t>()->default_value(1), "Device port.")
		        ("delta", "Show delta counts since last line.")
		        ("stderr", "write output to stderr.")
		        ("log", "write output using MCAS PLOG interface.")
		        ("debug", po::value<unsigned>()->default_value(0), "Debug level")
		        ("interval", po::value<double>(), "Sample interval (seconds)")
		        ("count", po::value<unsigned>(), "Sample count")
		;

		po::positional_options_description pos{};
		pos.add("interval", 1).add("count", 2);

		po::variables_map vm;

		po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);

		if ( vm.count("help") != 0 )
	       	{
			std::cout << desc << "\n";
			return 1;
		}

		std::string counters =
			"/sys/class/infiniband/"
			+ vm["device"].as<std::string>()
			+ "/ports/"
			+ std::to_string(vm["port"].as<std::uint16_t>())
			+ "/counters";

		unsigned count = 0;
		auto interval = 0.0;

		if ( vm.count("stderr") ) destination = pr_err;
		if ( vm.count("log") ) destination = pr_log;

		if ( vm.count("interval") == 0 )
		{
			count = 1;
		}
		else
		{
			interval = vm["interval"].as<double>();
			if ( vm.count("count") )
			{
				count = vm["count"].as<unsigned>();
			}
		}

		auto delta = vm.count("delta") != 0;

		auto time_base = std::chrono::steady_clock::now();

		auto chrono_interval = std::chrono::nanoseconds(std::uint64_t(interval * 1e9));

		field fields[4] = { "port_rcv_data", "port_rcv_packets", "port_xmit_data", "port_xmit_packets" };

		header_line(fields);

		utc_line();

		for ( unsigned n = 0; count == 0 || n != count; ++n )
		{
			std::this_thread::sleep_until(time_base + n * chrono_interval);
			line(counters, "mlx: ", fields, delta);
		}
	}

	catch (const po::error &e) {
		std::cerr << "bad command line option: " << e.what() << "\n";
		return -1;
	}

	return 0;
}
