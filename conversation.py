import ollama

class Conversation:
    def __init__(self, model='phi3:mini', max_words=30):
        self.model = model
        self.history = []
        self.system_prompt = f"You are an assistant who responds briefly, no more than {max_words} words."

    def chat(self, user_input):
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            *self.history,
            {'role': 'user', 'content': user_input}
        ]

        response = ollama.chat(model=self.model, messages=messages)
        self.history.append({'role': 'user', 'content': user_input})
        self.history.append({'role': 'assistant', 'content': response['message']['content']})

        return response['message']['content']

    def reset(self):
        self.history.clear()

