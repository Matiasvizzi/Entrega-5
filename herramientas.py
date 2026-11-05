"""Herramientas del agente: una "base de datos" simulada de clientes y pedidos.

El docstring de cada @tool es lo ÚNICO que el LLM lee para decidir cuándo usarla,
por eso son deliberadamente específicos: qué hace, qué recibe, cuándo usarla y qué
devuelve en caso de error.

Hay dos tablas separadas a propósito: el usuario menciona al cliente por NOMBRE, pero
los pedidos se consultan por ID. El agente debe razonar en varios pasos
(nombre -> ID -> pedidos) sin que nadie le indique el orden.
"""
from typing import TypedDict

from langchain_core.tools import BaseTool, tool

CLIENTES_DB: dict[str, int] = {
    "ana garcía": 102,
    "juan pérez": 205,
    "maría lópez": 310,
}


class Pedido(TypedDict):
    id: str
    fecha: str
    monto: int


PEDIDOS_DB: dict[int, list[Pedido]] = {
    102: [
        {"id": "P-1001", "fecha": "2026-08-14", "monto": 4000},
        {"id": "P-1002", "fecha": "2026-09-30", "monto": 5500},
        {"id": "P-1003", "fecha": "2026-10-21", "monto": 5000},
    ],
    205: [
        {"id": "P-2001", "fecha": "2026-07-02", "monto": 3200},
    ],
    310: [
        {"id": "P-3001", "fecha": "2026-05-11", "monto": 5000},
        {"id": "P-3002", "fecha": "2026-06-23", "monto": 6000},
        {"id": "P-3003", "fecha": "2026-08-05", "monto": 4800},
        {"id": "P-3004", "fecha": "2026-09-17", "monto": 7000},
        {"id": "P-3005", "fecha": "2026-10-29", "monto": 5000},
    ],
}


@tool
def buscar_cliente_por_nombre(nombre: str) -> str:
    """Busca el ID interno (cliente_id) de un cliente a partir de su nombre completo.
    Usar esta herramienta SIEMPRE que el usuario mencione a un cliente por su nombre
    y necesites su cliente_id antes de poder consultar sus pedidos.
    Devuelve un mensaje que empieza con ERROR si el nombre no coincide exactamente con
    ningún cliente registrado; en ese caso hay que pedirle al usuario que confirme
    el nombre en lugar de inventar un ID."""
    clave = nombre.strip().lower()
    if clave not in CLIENTES_DB:
        return (
            f"ERROR: no se encontró ningún cliente con el nombre '{nombre}'. "
            "Verificá que el nombre esté completo y bien escrito."
        )
    return f"Cliente encontrado: '{nombre}' -> cliente_id={CLIENTES_DB[clave]}"


@tool
def buscar_pedidos(cliente_id: int) -> str:
    """Devuelve la cantidad de pedidos y el monto total gastado por un cliente, dado su
    cliente_id NUMÉRICO (no su nombre: si solo tenés el nombre, primero usá
    buscar_cliente_por_nombre para obtener el ID).
    Devuelve un mensaje que empieza con ERROR si el cliente_id no existe."""
    if cliente_id not in PEDIDOS_DB:
        return f"ERROR: no existe ningún cliente con id={cliente_id}."
    pedidos = PEDIDOS_DB[cliente_id]
    total = sum(p["monto"] for p in pedidos)
    return f"cliente_id={cliente_id}: pedidos={len(pedidos)}, total=${total}"


@tool
def buscar_ultimo_pedido(cliente_id: int) -> str:
    """Devuelve el pedido MÁS RECIENTE de un cliente (id del pedido, fecha y monto),
    dado su cliente_id NUMÉRICO. Usar cuando el usuario pregunte por el último pedido
    o la última compra de un cliente. Si solo tenés el nombre, primero usá
    buscar_cliente_por_nombre. Devuelve un mensaje que empieza con ERROR si el
    cliente_id no existe."""
    if cliente_id not in PEDIDOS_DB:
        return f"ERROR: no existe ningún cliente con id={cliente_id}."
    ultimo = max(PEDIDOS_DB[cliente_id], key=lambda p: p["fecha"])
    return (
        f"cliente_id={cliente_id}: último pedido {ultimo['id']} "
        f"del {ultimo['fecha']} por ${ultimo['monto']}"
    )


HERRAMIENTAS: list[BaseTool] = [
    buscar_cliente_por_nombre,
    buscar_pedidos,
    buscar_ultimo_pedido,
]
