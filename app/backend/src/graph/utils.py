from langchain.chat_models import init_chat_model

from graph.configuration import get_config


def load_chat_model():
    settings = get_config()
    return init_chat_model(
        model=settings.model,
        model_provider=settings.provider,
        api_key=settings.openai_api_key,
        temperature=settings.temperature,
    )
