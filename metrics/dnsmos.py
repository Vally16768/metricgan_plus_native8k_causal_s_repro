from __future__ import annotations


def dnsmos_wav(path: str) -> dict[str, float]:
    raise RuntimeError(
        "DNSMOS este optional si nu este inclus implicit in proiectul standalone. "
        "Activeaza-l separat doar daca adaugi modelele ONNX si dependentele aferente."
    )
