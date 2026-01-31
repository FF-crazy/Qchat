from ..main import app

@app.post("/v1/chat/completions")
async def post_openai():
  # TODO
  pass

@app.post("/v1/message")
async def post_anthropic():
  # TODO
  pass

@app.post("/v1/models")
async def get_model_openai():
  # TODO
  pass

