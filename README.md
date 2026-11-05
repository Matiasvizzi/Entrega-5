# Agente ReAct con memoria persistente — Pre-entrega 5

Agente de razonamiento cíclico (patrón **ReAct**: Reasoning + Acting) construido con **LangGraph**. A diferencia de un flujo fijo, aquí el propio LLM decide en cada vuelta si necesita llamar a una herramienta o ya puede responder. El historial se guarda en **SQLite** (`AsyncSqliteSaver`), por lo que el agente recuerda la conversación por `thread_id`, incluso después de reiniciar el programa.

## Arquitectura

```
START ─► modelo ──(¿tool_calls?)──► herramientas ─┐
           ▲                                       │
           └───────────────────────────────────────┘   (ciclo ReAct)
           │
           └──(sin tool_calls)──► END
```

| Archivo | Rol |
|---|---|
| `herramientas.py` | 3 herramientas `@tool` sobre una "base de datos" simulada de clientes y pedidos. |
| `agente.py` | `StateGraph` (hereda de `MessagesState`), `bind_tools`, `tools_condition`, `recursion_limit`. |
| `main.py` | Demo con traza (`demo`) y chat interactivo con memoria (`chat`). Todo asíncrono. |
| `traza_ejecucion.json` | Traza ReAct real de la demo (incluida en el repo). |

### Cómo cumple los criterios

- **Autonomía:** no hay `if/else` sobre la pregunta; el LLM elige la herramienta a partir de los docstrings.
- **Ciclo de retorno:** si una herramienta devuelve `ERROR`, el mensaje vuelve al LLM, que pide una aclaración en lugar de inventar datos (`ToolNode(handle_tool_errors=True)` además captura las excepciones).
- **Resiliencia de estado:** con el mismo `thread_id` el checkpointer devuelve el historial; se verifica reabriendo la base SQLite.
- **Código limpio:** Python 3.12, type hints y `asyncio` (`ainvoke`, `AsyncSqliteSaver`).
- **Bucles infinitos:** `recursion_limit = 10` en cada invocación.
- **Estado sucio:** el checkpoint guarda todo el historial, pero al LLM solo se le pasan los últimos 30 mensajes (`trim_messages`, cortando siempre en un mensaje humano para no separar un par tool_call/tool_result).

### Herramientas

Dos tablas separadas a propósito: el usuario habla de **nombres**, pero los pedidos se consultan por **ID**, así que el agente debe razonar en varios pasos.

| Herramienta | Entrada | Uso |
|---|---|---|
| `buscar_cliente_por_nombre` | `nombre: str` | Obtiene el `cliente_id`. |
| `buscar_pedidos` | `cliente_id: int` | Cantidad de pedidos y total gastado. |
| `buscar_ultimo_pedido` | `cliente_id: int` | Pedido más reciente (id, fecha, monto). |

## Cómo levantar el entorno

Requiere **Python 3.12+**.

```bash
# 1. Entorno virtual y dependencias
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate en Linux/macOS)
pip install -r requirements.txt

# 2. Variables de entorno
copy .env.example .env            # Windows  (cp en Linux/macOS)
# editar .env: elegir LLM_PROVIDER (gemini | openai | anthropic) y pegar la API key correspondiente

# 3. Demo: corre los escenarios y regenera traza_ejecucion.json
python main.py demo

# 4. Chat interactivo con memoria persistente
python main.py chat --thread mi-sesion
#   cortá el programa, volvé a ejecutarlo con el mismo --thread y el agente recuerda la charla
```

## Traza de ejecución

La demo ejecuta 4 escenarios y guarda la traza completa en [`traza_ejecucion.json`](traza_ejecucion.json). Resumen:

| # | Escenario | Resultado |
|---|---|---|
| 1 | *"¿Cuántos pedidos tuvo Ana García y cuál fue el total?"* | **Multi-paso:** `buscar_cliente_por_nombre` → `buscar_pedidos` → "3 pedidos, $14.500". |
| 2 | *"¿Y Juan Pérez?"* (mismo thread) | Entiende por el historial que se pregunta por pedidos y total → 1 pedido, $3.200. |
| 3 | *"Volviendo a Ana García: ¿cuál fue su último pedido?"* | `buscar_cliente_por_nombre` → `buscar_ultimo_pedido` → P-1003, $5.000. |
| 4 | *"¿Cuántos pedidos tuvo el cliente Roberto Sánchez?"* | La herramienta devuelve `ERROR`; el agente pide verificar el nombre sin inventar nada. |

Además la demo reabre la base SQLite desde cero y comprueba que el thread conserva su historial (18 mensajes).

Ejemplo del ciclo (escenario 1):

```
🧑 Usuario: ¿Cuántos pedidos tuvo Ana García y cuál fue el total?
   🔧 Decide usar: buscar_cliente_por_nombre({'nombre': 'Ana García'})
   ↩️  buscar_cliente_por_nombre devolvió: Cliente encontrado: 'Ana García' -> cliente_id=102
   🔧 Decide usar: buscar_pedidos({'cliente_id': 102})
   ↩️  buscar_pedidos devolvió: cliente_id=102: pedidos=3, total=$14500
🤖 Respuesta: Ana García tuvo 3 pedidos y el monto total gastado fue de $14,500.
```

## Seguridad

Las API keys se leen de `.env`, que está en `.gitignore` junto con `checkpoints.sqlite`. Solo se sube `.env.example`.
