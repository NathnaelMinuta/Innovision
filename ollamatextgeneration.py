import ollama

try:
    user_input = input("🗣️ You: ")

    response = ollama.chat(
        model='phi3:mini',
        messages=[
            {'role': 'system', 'content': 'You are an assistant who responds briefly, no more than 10 words.'},
            {'role': 'user', 'content': user_input}
        ]
    )

    print("\n🤖 Ollama (phi3:mini):")
    print(response['message']['content'])

except Exception as e:
    print("❌ Error:", e)

