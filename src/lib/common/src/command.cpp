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

#include <common/command.h>
#include <common/logging.h>

#include <csignal> /* kill */
#include <sys/wait.h> /* waitpid */

using command = common::command;

const char *command::env0[] = { static_cast<char *>(0) };

int command::kill(int sig_)
{
	FLOGM("killing {} with {}", _pid, sig_);
	return _pid
		? ::kill(_pid, sig_)
		: 0
		;
}

void command::wait()
{
	if ( _pid )
	{
		::waitpid(_pid, &_status, 0);
		FLOGM(" status {}", _status);
	}
}

command::~command()
{
	wait();
}


common::command_killed::~command_killed()
{
	FLOGM("killing with {}", _signal);
	kill(_signal);
}
