"""Vidence — gestión local, offline y cifrada de evidencia multimedia personal.

El guardia offline se instala al importar el paquete, antes de que cualquier
módulo cargue dependencias con capacidad de red: `huggingface_hub` (traído por
faster-whisper) lee HF_HUB_OFFLINE en el momento de su importación, y ningún
punto de entrada puede "olvidarse" de activarlo.
"""

from app import offline_guard

offline_guard.install()
