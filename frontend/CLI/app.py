"""Qchat TUI — a Claude-Code-style chat interface."""

import json

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Input, Markdown, Static
from textual import work

from frontend.CLI.client import QchatClient

WELCOME = """\
# Qchat

Type a message below to start chatting.
Use `/model <name>` to switch models, `/refresh` to update model list, `/providers` to list providers, `/provider <name>` to switch provider.
"""


class ChatMessage(Static):
    """A single message bubble."""

    DEFAULT_CSS = """
    ChatMessage {
        margin: 1 0;
        padding: 0 2;
    }
    """


class QchatApp(App):
    TITLE = "Qchat"
    CSS = """
    #chat-view {
        height: 1fr;
        padding: 1 2;
    }
    #chat-input {
        dock: bottom;
        margin: 0 2 1 2;
    }
    .user-msg {
        color: $accent;
    }
    .assistant-msg {
        color: $text;
    }
    .system-msg {
        color: $text-muted;
        text-style: italic;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear"),
    ]

    def __init__(self, backend_url: str = "http://127.0.0.1:10000") -> None:
        super().__init__()
        self.client = QchatClient(backend_url)
        self.model: str | None = None
        self.context: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="chat-view"):
            yield Markdown(WELCOME, id="welcome")
        yield Input(placeholder="Send a message…", id="chat-input")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#chat-input", Input).focus()
        self._load_initial_config()

    @work()
    async def _load_initial_config(self) -> None:
        try:
            config = await self.client.get_config()
            provider = config.get("provider")
            if provider:
                models = await self.client.refresh_models()
                if models:
                    self.model = models[0]["model_name"]
                    self._post_system(f"Provider: **{provider}** | Model: **{self.model}**")
                else:
                    self._post_system(f"Provider: **{provider}** | No models available. Use `/refresh`.")
            else:
                self._post_system("No provider configured. Use `/provider <name>` to select one.")
        except Exception as e:
            self._post_system(f"Failed to connect to backend: `{e}`")

    # ── input handling ──

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.clear()

        if text.startswith("/"):
            self._handle_command(text)
            return

        self._post_user(text)
        self.context.append({
            "role": "user",
            "content": [{"type": "text", "text": text}],
        })
        self._send_message()

    # ── commands ──

    @work()
    async def _handle_command(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        try:
            match cmd:
                case "/model":
                    if not arg:
                        models = await self.client.get_models()
                        names = [m["model_name"] for m in models]
                        self._post_system("Models: " + ", ".join(f"`{n}`" for n in names) if names else "No models cached. Run `/refresh` first.")
                    else:
                        self.model = arg
                        self._post_system(f"Model set to **{arg}**")
                case "/refresh":
                    self._post_system("Refreshing models…")
                    models = await self.client.refresh_models()
                    names = [m["model_name"] for m in models]
                    self._post_system(f"Loaded {len(names)} models: " + ", ".join(f"`{n}`" for n in names[:20]))
                case "/providers":
                    providers = await self.client.get_providers()
                    self._post_system("Providers: " + ", ".join(f"`{p}`" for p in providers))
                case "/provider":
                    if not arg:
                        config = await self.client.get_config()
                        self._post_system(f"Current provider: **{config.get('provider', 'None')}**")
                    else:
                        await self.client.update_config(provider=arg)
                        models = await self.client.refresh_models()
                        self.model = models[0]["model_name"] if models else None
                        info = f"Switched to **{arg}**"
                        if self.model:
                            info += f" | Model: **{self.model}**"
                        self._post_system(info)
                case "/clear":
                    self.action_clear_chat()
                case _:
                    self._post_system(f"Unknown command: `{cmd}`")
        except Exception as e:
            self._post_system(f"Error: `{e}`")

    # ── send message & stream response ──

    @work()
    async def _send_message(self) -> None:
        if not self.model:
            self._post_system("No model selected. Use `/model <name>` or `/refresh`.")
            return

        md_widget = Markdown("", classes="assistant-msg")
        chat_view = self.query_one("#chat-view", VerticalScroll)
        chat_view.mount(md_widget)
        chat_view.scroll_end(animate=False)

        full_text = ""
        try:
            async for raw in self.client.post_message_stream(self.model, self.context):
                chunk = json.loads(raw)
                delta = chunk.get("delta", "")
                full_text += delta
                md_widget.update(full_text)
                chat_view.scroll_end(animate=False)
        except Exception as e:
            full_text += f"\n\n*Error: {e}*"
            md_widget.update(full_text)

        if full_text:
            self.context.append({
                "role": "assistant",
                "content": [{"type": "text", "text": full_text}],
            })

    # ── helpers ──

    def _post_user(self, text: str) -> None:
        chat_view = self.query_one("#chat-view", VerticalScroll)
        msg = ChatMessage(f"**> {text}**", classes="user-msg")
        chat_view.mount(msg)
        chat_view.scroll_end(animate=False)

    def _post_system(self, text: str) -> None:
        chat_view = self.query_one("#chat-view", VerticalScroll)
        msg = Markdown(text, classes="system-msg")
        chat_view.mount(msg)
        chat_view.scroll_end(animate=False)

    def action_clear_chat(self) -> None:
        chat_view = self.query_one("#chat-view", VerticalScroll)
        chat_view.remove_children()
        self.context.clear()
        self._post_system("Chat cleared.")

    async def action_quit(self) -> None:
        await self.client.close()
        self.exit()
