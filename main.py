"""Punto de entrada: demo con traza ReAct y chat interactivo con memoria persistente.

Uso:
    python main.py demo                       # corre los escenarios y escribe traza_ejecucion.json
    python main.py chat --thread mi-sesion    # chat; si repetís el thread, recuerda la charla
"""
import argparse
import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agente import config_hilo, construir_grafo, crear_llm

DB_PATH = "checkpoints.sqlite"
TRAZA_PATH = Path(__file__).resolve().parent / "traza_ejecucion.json"


def extraer_texto(mensaje: BaseMessage) -> str:
    """Normaliza el content (Gemini a veces devuelve una lista de bloques)."""
    contenido = mensaje.content
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "".join(b.get("text", "") for b in contenido if isinstance(b, dict))
    return str(contenido)


def serializar(mensajes: list[BaseMessage]) -> list[dict[str, Any]]:
    """Mensajes de LangGraph -> JSON simple (la traza ReAct pedida en la consigna)."""
    traza: list[dict[str, Any]] = []
    for m in mensajes:
        entrada: dict[str, Any] = {"tipo": type(m).__name__, "contenido": extraer_texto(m)}
        if isinstance(m, AIMessage) and m.tool_calls:
            entrada["tool_calls"] = [
                {"herramienta": tc["name"], "argumentos": tc["args"]} for tc in m.tool_calls
            ]
        if isinstance(m, ToolMessage):
            entrada["herramienta"] = m.name
        traza.append(entrada)
    return traza


def imprimir_pasos(pasos: list[dict[str, Any]]) -> None:
    for p in pasos:
        if p["tipo"] == "HumanMessage":
            print(f"🧑 Usuario: {p['contenido']}")
        elif p["tipo"] == "AIMessage" and "tool_calls" in p:
            for tc in p["tool_calls"]:
                print(f"   🔧 Decide usar: {tc['herramienta']}({tc['argumentos']})")
        elif p["tipo"] == "ToolMessage":
            print(f"   ↩️  {p['herramienta']} devolvió: {p['contenido']}")
        elif p["tipo"] == "AIMessage":
            print(f"🤖 Respuesta: {p['contenido']}")


async def turno(app: Any, thread_id: str, pregunta: str) -> list[dict[str, Any]]:
    """Ejecuta un turno y devuelve solo los mensajes NUEVOS de ese turno."""
    config = config_hilo(thread_id)
    previos = await app.aget_state(config)
    n_previos = len(previos.values.get("messages", [])) if previos.values else 0

    resultado = await app.ainvoke({"messages": [HumanMessage(content=pregunta)]}, config=config)
    return serializar(resultado["messages"][n_previos:])


def contar_llamadas(pasos: list[dict[str, Any]]) -> list[str]:
    return [tc["herramienta"] for p in pasos for tc in p.get("tool_calls", [])]


async def demo() -> None:
    llm = crear_llm()
    sufijo = uuid.uuid4().hex[:6]  # hilos nuevos en cada corrida -> traza reproducible
    hilo_demo, hilo_error = f"demo-{sufijo}", f"error-{sufijo}"

    escenarios = [
        ("1_multi_paso", hilo_demo,
         "¿Cuántos pedidos tuvo Ana García y cuál fue el total?"),
        ("2_memoria_mismo_thread", hilo_demo,
         "¿Y Juan Pérez?"),
        ("3_recuerda_contexto", hilo_demo,
         "Volviendo a Ana García: ¿cuál fue su último pedido?"),
        ("4_error_y_aclaracion", hilo_error,
         "¿Cuántos pedidos tuvo el cliente Roberto Sánchez?"),
    ]

    traza: dict[str, Any] = {
        "meta": {
            "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "modelo": getattr(llm, "model", None) or getattr(llm, "model_name", "desconocido"),
            "recursion_limit": config_hilo("x")["recursion_limit"],
            "threads": {"demo": hilo_demo, "error": hilo_error},
        },
        "escenarios": {},
    }

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = construir_grafo(llm).compile(checkpointer=checkpointer)

        for nombre, hilo, pregunta in escenarios:
            print("\n" + "=" * 80)
            print(f"▶ Escenario {nombre}  (thread_id={hilo})")
            print("=" * 80)
            pasos = await turno(app, hilo, pregunta)
            imprimir_pasos(pasos)
            traza["escenarios"][nombre] = {
                "thread_id": hilo,
                "llamadas_a_herramientas": contar_llamadas(pasos),
                "pasos": pasos,
            }

    # Resiliencia de estado: reabrimos la conexión desde cero (como si el proceso
    # se hubiera reiniciado) y verificamos que el historial sigue en SQLite.
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = construir_grafo(llm).compile(checkpointer=checkpointer)
        estado = await app.aget_state(config_hilo(hilo_demo))
        n = len(estado.values["messages"])
        print(f"\n💾 Tras reabrir {DB_PATH}: el thread '{hilo_demo}' conserva {n} mensajes.")
        traza["persistencia"] = {"thread_id": hilo_demo, "mensajes_recuperados": n}

    TRAZA_PATH.write_text(json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Traza guardada en {TRAZA_PATH.name}")


async def chat(thread_id: str) -> None:
    llm = crear_llm()
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        app = construir_grafo(llm).compile(checkpointer=checkpointer)
        previos = await app.aget_state(config_hilo(thread_id))
        n = len(previos.values.get("messages", [])) if previos.values else 0
        print(f"💬 thread_id='{thread_id}' ({n} mensajes previos). Escribí 'salir' para terminar.\n")

        while True:
            pregunta = (await asyncio.to_thread(input, "🧑 Vos: ")).strip()
            if pregunta.lower() in ("", "salir", "exit", "quit"):
                print("👋 Hasta luego.")
                break
            imprimir_pasos(await turno(app, thread_id, pregunta))
            print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Agente ReAct con memoria persistente")
    sub = ap.add_subparsers(dest="comando", required=True)
    sub.add_parser("demo", help="corre los escenarios y escribe traza_ejecucion.json")
    p_chat = sub.add_parser("chat", help="chat interactivo")
    p_chat.add_argument("--thread", default="sesion-1", help="thread_id (misma id = misma memoria)")
    args = ap.parse_args()

    asyncio.run(demo() if args.comando == "demo" else chat(args.thread))
