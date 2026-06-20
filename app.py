import ollama



response: ollama.ChatResponse = ollama.chat(
  model='qwen3:0.6b',
  messages=[
    {
      'role': 'user',
      'content': 'Explain drift in Machine Learning in simple terms.',
    },
  ],
  tools=[
    
  ]
)
print(response['message']['content'])
print(response.message.content)