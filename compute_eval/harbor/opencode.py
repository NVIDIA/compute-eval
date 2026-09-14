import shlex

from harbor.agents.installed.base import NonZeroAgentExitCodeError, with_prompt_template
from harbor.agents.installed.opencode import OpenCode as HarborOpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

AGENT_IMPORT_PATH = "compute_eval.harbor.opencode:OpenCode"


class OpenCode(HarborOpenCode):
    """Harbor OpenCode runner with a custom command body and quoted model names.

    Provider-specific details stay in regular OpenCode config files passed via
    ``--opencode-config``. This wrapper exists because Harbor's stock adapter
    (still true as of 0.20.0) interpolates the model name without shell quoting
    and assumes an nvm installation. Our task image installs OpenCode globally.
    """

    # Delete this shim once Harbor's stock OpenCode shell-quotes model names and
    # exposes the run command we need.

    def get_version_command(self) -> str:
        # The CLI is baked into the image and available on PATH. Harbor's stock
        # command sources ~/.nvm/nvm.sh unconditionally, which pollutes the
        # captured version when an image does not use nvm.
        return "opencode --version"

    async def install(self, environment: BaseEnvironment) -> None:
        # Override Harbor's live installer; the CLI must be baked into the task image.
        return None

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        self._instruction = instruction
        escaped_instruction = shlex.quote(instruction)

        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")
        model_arg = shlex.quote(self.model_name)

        env = {
            "OPENCODE_FAKE_VCS": "git",
            "XDG_DATA_HOME": "/logs/agent/opencode/xdg-data",
            "XDG_STATE_HOME": "/logs/agent/opencode/xdg-state",
        }

        skills_command = self._build_register_skills_command()
        if skills_command:
            await self.exec_as_agent(environment, command=skills_command, env=env)

        mcp_command = self._build_register_config_command()
        if mcp_command:
            await self.exec_as_agent(environment, command=mcp_command, env=env)

        cli_flags = self.build_cli_flags()
        cli_flags_arg = f"{cli_flags} " if cli_flags else ""
        resume_flag = "--continue " if self._resume else ""

        await self.exec_as_agent(
            environment,
            command=(
                "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                f"opencode --model={model_arg} run --format=json "
                f"{resume_flag}{cli_flags_arg}--thinking "
                f"--dangerously-skip-permissions -- {escaped_instruction} "
                "2>&1 </dev/null | stdbuf -oL tee /logs/agent/opencode.txt"
            ),
            env=env,
        )

        if messages := self._error_messages():
            raise NonZeroAgentExitCodeError("OpenCode emitted error event(s): " + "; ".join(messages[:3]))
