from .base_acp_persona import BaseAcpPersona
from jupyter_ai_persona_manager import PersonaDefaults

class TestAcpPersona(BaseAcpPersona):
    def __init__(self, *args, **kwargs):
        executable = ["python", "jupyter-ai-acp-client/examples/agent.py"]
        super().__init__(*args, executable=executable, **kwargs)
    
    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="ACP-Test",
            description="A test ACP persona",
            avatar_path="TODO",
            system_prompt="unused"
        )