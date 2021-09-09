#ifndef FABRIC_ENTER_EXIT_TRACE
#define FABRIC_ENTER_EXIT_TRACE
#include <common/string_view.h>
#include <string>
#include <iostream>

/* 
 * Fabric enter/exit trace
 */
struct enter_exit_trace
{
private:
	std::string _func;
	std::string _file;
	unsigned _line;
	void write(common::string_view id_) const
	{
		std::cerr << "FABRIC FUNC " << id_ << " " << _func << " " << _file << ":" << _line << "\n";
	}
public:
	enter_exit_trace(common::string_view func_, common::string_view file_, unsigned line_)
		: _func(func_)
		, _file(file_)
		, _line(line_)
	{
		write("begin");
	}
	~enter_exit_trace()
	{
		write("end");
	}
};

#define ENTER_EXIT_TRACE static enter_exit_trace x0(__func__, __FILE__, __LINE__);
#endif
