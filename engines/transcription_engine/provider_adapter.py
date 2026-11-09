
import os

from providers.assemblyai_provider import AssemblyAIProvider


def get_provider():

    provider_name = os.getenv(
        "TRANSCRIPTION_PROVIDER",
        "assemblyai"
    ).lower()

    if provider_name == "assemblyai":
        return AssemblyAIProvider()

    raise ValueError(
        f"Unsupported transcription provider: {provider_name}"
    )
