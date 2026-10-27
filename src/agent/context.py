"""Contexto de cada llamada a una tool. Rol, planta y línea vienen de la sesión del servidor, nunca del texto."""
import sqlite3
from dataclasses import dataclass
from typing import Literal

Role = Literal["supervisor", "quality", "maintenance", "ml_engineer", "plant_manager"]
ROLES: tuple[str, ...] = ("supervisor", "quality", "maintenance", "ml_engineer", "plant_manager")


@dataclass
class ToolContext:
    user_id: str
    role: Role
    plant_id: str
    line_id: str
    session_id: str
    conn: sqlite3.Connection
    vision: "object"  # VisionClient o un doble de pruebas con la misma interfaz
    image_dir: str = "appdata/images"
