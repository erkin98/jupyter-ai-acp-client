import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

from acp import (
    PROTOCOL_VERSION,
    Client,
    RequestError,
    connect_to_agent,
    text_block,
)
from acp.core import ClientSideConnection
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AudioContentBlock,
    AvailableCommandsUpdate,
    ClientCapabilities,
    CreateTerminalResponse,
    CurrentModeUpdate,
    EmbeddedResourceContentBlock,
    EnvVariable,
    ImageContentBlock,
    Implementation,
    KillTerminalCommandResponse,
    NewSessionResponse,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    ResourceContentBlock,
    TerminalOutputResponse,
    TextContentBlock,
    ToolCall,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
    AllowedOutcome
)
from jupyter_ai_persona_manager import BasePersona
from jupyterlab_chat.ychat import YChat
from typing import Awaitable
from asyncio.subprocess import Process

class JaiAcpClient(Client):
    """
    The default ACP client. The client should be stored as a class attribute on each
    ACP persona, such that each ACP agent subprocess is communicated through
    exactly one ACP client (an instance of this class).
    """

    agent_subprocess: Process
    _connection_future: Awaitable[ClientSideConnection] | None
    event_loop: asyncio.AbstractEventLoop
    _personas_by_session: dict[str, BasePersona]

    def __init__(self, *args, agent_subprocess: Awaitable[Process], event_loop: asyncio.AbstractEventLoop, **kwargs):
        """
        :param agent_subprocess: The ACP agent subprocess
        (`asyncio.subprocess.Process`) assigned to this client.

        :param event_loop: The `asyncio` event loop running this process.
        """
        self.agent_subprocess = agent_subprocess
        self._connection_future = event_loop.create_task(self._init_connection())
        self.event_loop = event_loop
        self._personas_by_session = {}
        super().__init__(*args, **kwargs)
    

    async def _init_connection(self) -> ClientSideConnection:
        proc = self.agent_subprocess
        conn = connect_to_agent(self, proc.stdin, proc.stdout)
        await conn.initialize(
            protocol_version=PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(),
            client_info=Implementation(name="Jupyter AI", title="Jupyter AI ACP Client", version="0.1.0"),
        )
        return conn
    
    async def get_connection(self) -> ClientSideConnection:
        return await self._connection_future

    async def create_session(self, persona: BasePersona) -> NewSessionResponse:
        """
        Create an ACP agent session through this client scoped to a
        `BasePersona` instance.
        """
        conn = await self.get_connection()
        # TODO: change this to Jupyter preferred dir
        session = await conn.new_session(mcp_servers=[], cwd=os.getcwd())
        self._personas_by_session[session.session_id] = persona
        return session
    
    async def session_update(
        self,
        session_id: str,
        update: UserMessageChunk
        | AgentMessageChunk
        | AgentThoughtChunk
        | ToolCallStart
        | ToolCallProgress
        | AgentPlanUpdate
        | AvailableCommandsUpdate
        | CurrentModeUpdate,
        **kwargs: Any,
    ) -> None:
        """
        Handles `session/update` requests from the ACP agent.
        """

        if not isinstance(update, AgentMessageChunk):
            return

        content = update.content
        text: str
        if isinstance(content, TextContentBlock):
            text = content.text
        elif isinstance(content, ImageContentBlock):
            text = "<image>"
        elif isinstance(content, AudioContentBlock):
            text = "<audio>"
        elif isinstance(content, ResourceContentBlock):
            text = content.uri or "<resource>"
        elif isinstance(content, EmbeddedResourceContentBlock):
            text = "<resource>"
        else:
            text = "<content>"
        
        persona = self._personas_by_session[session_id]
        persona.send_message(text)
        print(text)

    async def request_permission(
        self, options: list[PermissionOption], session_id: str, tool_call: ToolCall, **kwargs: Any
    ) -> RequestPermissionResponse:
        """
        Handles `session/request_permission` requests from the ACP agent.

        TODO: This currently always gives the agent permission. We will need to
        add some tool call approval UI and handle permission requests properly.
        """
        option_id = ""
        for o in options:
            if "allow" in o.option_id.lower():
                option_id = o.option_id
                break

        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=option_id, outcome='selected')
        )

    ##############################
    # Unimplemented methods below
    ##############################

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: Any
    ) -> WriteTextFileResponse | None:
        raise RequestError.method_not_found("fs/write_text_file")

    async def read_text_file(
        self, path: str, session_id: str, limit: int | None = None, line: int | None = None, **kwargs: Any
    ) -> ReadTextFileResponse:
        raise RequestError.method_not_found("fs/read_text_file")

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> TerminalOutputResponse:
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> ReleaseTerminalResponse | None:
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> WaitForTerminalExitResponse:
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: Any
    ) -> KillTerminalCommandResponse | None:
        raise RequestError.method_not_found("terminal/kill")

    async def ext_method(self, method: str, params: dict) -> dict:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict) -> None:
        raise RequestError.method_not_found(method)

