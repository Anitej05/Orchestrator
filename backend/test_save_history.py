from orchestrator.graph import save_conversation_history
from langchain_core.messages import HumanMessage, AIMessage


def run_test():
    state = {
        'original_prompt': 'Hello',
        'messages': [HumanMessage(content='Hi'), AIMessage(content='Hello!')],
        'completed_tasks': []
    }
    class Config(dict):
        def get(self, k, default=None):
            return super().get(k, default)
    config = {'configurable': {'thread_id': 'test-thread-123'}}
    save_conversation_history(state, config)
    print('Saved history to conversation_history/test-thread-123.json')

if __name__ == '__main__':
    run_test()