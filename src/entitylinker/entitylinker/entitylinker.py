import reflex as rx
import asyncio
import sys,os,json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from entitylinker.entity_linker import EntityLinker

class State(rx.State):
    text: str = ""
    updates: list[str] = []
    config: dict = {
        "elasticsearch": "http://localhost:9222",
        "sparql_endpoint": "http://localhost:8897/sparql"
    }
    entity_linker = EntityLinker(config)

    async def send_text(self):
        self.updates = []  # Clear previous output
        for i in range(5):
            await asyncio.sleep(1)  # simulate delay like WebSocket push
            self.updates.append(f"Step {i+1}/5 for: {self.text}")
            yield  # Triggers reactive update
        self.updates.append("✅ Done!")
        yield

def index():
    return rx.container(
        rx.heading("Real-Time Text Streamer", size="4"),
        rx.text_area(
            placeholder="Enter text...",
            on_change=State.set_text,
            width="100%",
            height="100px"
        ),
        rx.button("Submit", on_click=State.send_text, mt="2"),
        rx.box(
            rx.foreach(State.updates, lambda msg: rx.text(msg)),
            border="1px solid #ccc",
            padding="2",
            mt="4",
            height="200px",
            overflow_y="scroll"
        ),
        spacing="4",
        padding="4",
        max_width="600px",
        margin="auto"
    )

app = rx.App()
app.add_page(index)
