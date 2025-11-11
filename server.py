# server.py
"""
FastAPI + Google ADK (warm-session, stateless requests) with minimal logging.

Behavior:
- On startup: create runner + one pre-warmed session (ready to use).
- On each /flight request:
    * Atomically take (consume) the pre-warmed session.
    * Run the agent with that session (request is independent; no state reuse).
    * Delete the consumed session.
    * Fire-and-forget to warm a new session for the next request.
- If no pre-warmed session exists when a request arrives, create one synchronously (fallback).
"""

import os
import logging
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Response
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Minimal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("flight_agent")

# ADK / Gemini
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

class Query(BaseModel):
    q: str  # e.g., "AI201 2025-08-25"

AGENT_INSTRUCTION = """You are a Flight Info agent.
The user provides a flight number and date. Use Google Search grounding (via the google_search tool)
to gather details (airline, origin, destination, scheduled/actual times, status, terminal/gate, aircraft).
CRITICAL NOTE: all times must be converted and returned in GST (Gulf Standard Time). Do NOT rely on a search result to
do the conversion for you — compute/convert the times yourself before returning them.
Then return a concise JSON answer containing the key data fields found. Do not return anything except a json response , no "```json .....```" like text formating wrappers or decorators, no need to mention sources, only pure data.
Example of wrong output: this is unacceptable
```json
{
 "airline": "Emirates",......... "aircraft_type": "Boeing 777-300ER"
}
Example of correct output (as it does not have the ```json``` formating): { "airline": "Emirates",......... "aircraft_type": "Boeing 777-300ER"}
In case of finding conflicting timings or other info mention it clearly in the response. like departure_time: 13:40 (80% confidence, conflicting info found 13:00 and 12:00)
Dont include the fields with null or empty values.
CRITICAL field that must be included always in the terminal no. with the airport.
Almost all requests will be for future flights so dont include actual_departure_time or actual_arrival_time as they mean that the flight has already taken place, just include the schedules time for arrival and departure.
If there is no flight of given flight number scheduled on that day return a messages saying "Please ask the user again to confirm the date and flight number via send_whatsapp_message, <mention reason here like 'no flight found' or 'flight appears to be canceled'>"
"""

# Build agent + runner once (fast path)
agent = Agent(
    name="flight_agent",
    model="gemini-2.5-flash",
    instruction=AGENT_INSTRUCTION,
    tools=[google_search],
)
runner = InMemoryRunner(app_name="flight_app", agent=agent)

# One warm session + lock
_shared_session = None  # google.adk.sessions.Session
_session_lock = asyncio.Lock()
_background_warm_task: Optional[asyncio.Task] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize runner and create first warm session on startup."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not set. Put it in .env or export it.")
    logger.info("Starting flight agent runner and creating initial warm session.")
    await _create_and_set_session()
    logger.info("Flight agent runner started and initial session warmed.")
    yield

app = FastAPI(title="ADK Flight Info Agent", lifespan=lifespan)

async def _create_and_set_session(user_id: str = "anonymous_user"):
    """
    Create a session and set it as the warm shared session (if empty).
    If one already exists, delete the extra we just created.
    """
    global _shared_session
    try:
        logger.info("Creating a new ADK session (warm).")
        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
        )
        async with _session_lock:
            if _shared_session is None:
                _shared_session = session
                logger.info("Warm session created and set for reuse.")
            else:
                # Another warm session already exists; delete this one
                try:
                    await runner.session_service.delete_session(
                        app_name=runner.app_name,
                        user_id=session.user_id,        # REQUIRED
                        session_id=session.id,          # REQUIRED
                    )
                    logger.info("Extra session deleted (another warm session exists).")
                except Exception as e:
                    logger.warning("Failed to delete extra session: %s", e)
    except Exception as e:
        logger.exception("Failed to create warm session: %s", e)

@app.post("/flight")
async def get_flight_info(body: Query):
    """
    Use a pre-warmed session, run the agent, delete the session, and warm a new one.
    """
    global _shared_session, _background_warm_task
    logger.info(f"Incoming request: {body.q}")

    # Consume a warm session (or create synchronously if absent)
    async with _session_lock:
        if _shared_session is None:
            logger.info("No warm session; creating synchronously for this request.")
            session = await runner.session_service.create_session(
                app_name=runner.app_name,
                user_id="anonymous_user",
            )
        else:
            session = _shared_session
            _shared_session = None  # mark as consumed

    # Run the agent with this session
    content = types.Content(role="user", parts=[types.Part(text=body.q)])
    final_text: Optional[str] = None
    try:
        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=content,
        ):
            # Look for final response event; then extract first text part
            if hasattr(event, "is_final_response") and event.is_final_response():
                if getattr(event, "content", None) and getattr(event.content, "parts", None):
                    for part in event.content.parts:
                        if getattr(part, "text", None):
                            final_text = part.text
                            break
                break
            else:
                logger.info(f"Agent thinking... event={event.__class__.__name__}")
    except Exception as e:
        logger.exception("Error while running agent: %s", e)
        # Best-effort delete of the consumed session
        try:
            await runner.session_service.delete_session(
                app_name=runner.app_name,
                user_id=session.user_id,     # REQUIRED
                session_id=session.id,       # REQUIRED
            )
        except Exception as e2:
            logger.warning("Failed to delete session after run error: %s", e2)
        raise

    if not final_text:
        logger.info("No final response produced by agent.")
        return Response(status_code=204)

    logger.info(f"Agent final response (raw): {final_text}")

    # Detect JSON and set media type
    media_type = "text/plain"
    try:
        json.loads(final_text)
        media_type = "application/json"
    except Exception:
        media_type = "text/plain"

    # Create response object
    response = Response(content=final_text, media_type=media_type)
    
    # Schedule session cleanup and warming to happen after response is sent
    async def cleanup_and_warm():
        # Delete the consumed session
        try:
            await runner.session_service.delete_session(
                app_name=runner.app_name,
                user_id=session.user_id,         # REQUIRED
                session_id=session.id,           # REQUIRED
            )
            logger.info("Consumed session deleted.")
        except Exception as e:
            logger.warning("Failed to delete consumed session: %s", e)

        # Warm a new session in the background
        try:
            global _background_warm_task
            if _background_warm_task is None or _background_warm_task.done():
                _background_warm_task = asyncio.create_task(_create_and_set_session())
                logger.info("Launched background warm-session task.")
        except Exception as e:
            logger.warning("Failed to spawn background warm-session task: %s", e)
    
    # Schedule cleanup to run after response is sent
    asyncio.create_task(cleanup_and_warm())
    
    return response

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")