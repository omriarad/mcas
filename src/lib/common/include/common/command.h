/*
   Copyright [2021] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

/*
 * Authors:
 *
 */

#ifndef MCAS_COMMON_COMMAND_H
#define MCAS_COMMON_COMMAND_H

#include <cerrno>
#include <csignal> /* SIGTERM */
#include <cstdlib> /* exit */
#include <unistd.h> /* fork, execle */

namespace common
{
	/* A single Linux command; destuctor waits for command completion */
	struct command
	{
	private:
		int _err;
		pid_t _pid;
		int _status;
		static const char *env0[]; // = { static_cast<char *>(0) };
	public:
		template<typename... Args>
			command(const char *env[], Args ... args)
				: _err(0)
				, _pid(::fork())
				, _status(0)
			{
				switch ( _pid )
				{
				case 0:
					::exit(::execle(args..., static_cast<char *>(0), env));
				case -1:
					_err = errno;
					break;
				default:
					;
				}
			}

		template<typename... Args>
			command(Args ... args)
				: command(env0, args...)
			{}

		~command();

		int kill(int sig_);
		void wait();
	};

	/* Run a command which needs to by killed to force an exit, e.g. "mpstat 1" */
	struct command_killed
		: public command
	{
	private:
		int _signal;
	public:
		template<typename... Args>
			command_killed(int signal_, Args ... args)
				: command(args...)
				, _signal(signal_)
			{}

		template<typename... Args>
			command_killed(Args ... args)
				: command_killed(SIGTERM, args...)
			{}

		~command_killed();
	};
}

#endif
