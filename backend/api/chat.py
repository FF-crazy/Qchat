from ..main import app

@app.post("/v1/chat/completion")
async def post_openai():
  # TODO
  pass

@app.post("/v1/message")
async def post_anthropic():
  # TODO
  pass