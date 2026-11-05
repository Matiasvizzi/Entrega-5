"""Agente ReAct con LangGraph: grafo cíclico modelo <-> herramientas.

    START -> modelo --(¿tool_calls?)--> herramientas --+
               ^                                        |
               +----------------------------------------+
               |
               +--(sin tool_calls)--> END

El propio LLM decide en cada vuelta si necesita una herramienta o ya puede responder:
no hay ningún if/else manual sobre la pregunta del usuario.
"""
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage, trim_messages
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from herramientas import HERRAMIENTAS

# En redes corporativas con inspección TLS, usar el almacén de certificados del
# sistema evita errores CERTIFICATE_VERIFY_FAILED. Es opcional.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

load_dotenv(Path(__file__).resolve().parent / ".env")

RECURSION_LIMIT: int = 10          # techo de pasos por invocación: evita bucles infinitos
MAX_MENSAJES_CONTEXTO: int = 30    # ventana de mensajes que ve el LLM (evita "estado sucio")

SYSTEM_PROMPT: str = (
    "Eres un asistente de atención comercial. Respondes preguntas sobre clientes y "
    "pedidos usando EXCLUSIVAMENTE las herramientas disponibles. "
    "Si una herramienta devuelve un ERROR, no inventes datos: informá el problema y "
    "pedile al usuario que aclare o corrija la información. "
    "Respondé en español, de forma breve."
)


def crear_llm() -> BaseChatModel:
    """Crea el LLM según LLM_PROVIDER (gemini | openai | anthropic)."""
    proveedor = os.getenv("LLM_PROVIDER", "gemini").lower()

    if proveedor == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    if proveedor == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model_name=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"), temperature=0
        )
    if proveedor == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
            temperature=0,
            max_retries=6,  # el modelo a veces responde 503 por alta demanda
        )
    raise SystemExit(f"❌ LLM_PROVIDER desconocido: '{proveedor}' (usar gemini, openai o anthropic)")


def construir_grafo(llm: BaseChatModel) -> StateGraph:
    """Arma el StateGraph (sin compilar). El estado hereda de MessagesState, cuyo
    reducer `add_messages` acumula los mensajes en lugar de reemplazarlos."""
    llm_con_herramientas = llm.bind_tools(HERRAMIENTAS)

    async def nodo_modelo(state: MessagesState) -> dict[str, list[BaseMessage]]:
        """Le pasa el historial al LLM; su respuesta puede traer tool_calls o ser final."""
        # El checkpoint guarda TODO el historial; al modelo solo le mostramos los
        # últimos N mensajes (arrancando siempre en un mensaje humano, para no cortar
        # un par tool_call/tool_result a la mitad).
        recientes: list[AnyMessage] = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=len,
            max_tokens=MAX_MENSAJES_CONTEXTO,
            start_on="human",
            allow_partial=False,
        )
        respuesta = await llm_con_herramientas.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *recientes]
        )
        return {"messages": [respuesta]}

    grafo = StateGraph(MessagesState)
    grafo.add_node("modelo", nodo_modelo)
    # handle_tool_errors=True: si una tool lanza una excepción, el error vuelve al LLM
    # como un mensaje más (no rompe el programa) y el agente puede reintentar.
    grafo.add_node("herramientas", ToolNode(HERRAMIENTAS, handle_tool_errors=True))

    grafo.add_edge(START, "modelo")
    grafo.add_conditional_edges(
        "modelo",
        tools_condition,  # mira si el último mensaje trae tool_calls
        {"tools": "herramientas", END: END},
    )
    grafo.add_edge("herramientas", "modelo")  # el ciclo ReAct
    return grafo


def config_hilo(thread_id: str) -> dict[str, Any]:
    """Configuración de invocación: thread_id (memoria) + recursion_limit (techo de pasos)."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
