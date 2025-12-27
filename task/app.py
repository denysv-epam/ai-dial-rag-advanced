from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import SearchMode, TextProcessor
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role

SYSTEM_PROMPT = """
You are a RAG powered assistant

## Structure of user message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions
- use information from the 'RAG CONTEXT' as a context when answering the `USER QUESTION`
- Cite particular sources whenever referencing information from the context.
- Answer ONLY based on RAG context or conversation history.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """
##RAG CONTEXT:
{context}

##USER QUESTION: 
{query}
"""


embeddings_client = DialEmbeddingsClient(
    deployment_name="text-embedding-3-small-1", api_key=API_KEY
)
completion_client = DialChatCompletionClient(deployment_name="gpt-4o", api_key=API_KEY)
text_processor = TextProcessor(
    embeddings_client=embeddings_client,
    db_config={
        "host": "localhost",
        "port": 5433,
        "database": "vectordb",
        "user": "postgres",
        "password": "postgres",
    },
)


def main():
    """
    $ python -m task.app

    NOTE:
    PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
    $ docker compose up -d

    IF dimensions are changes
    MAKE SURE to wipe DB and recreate doc vectors again
    $ docker compose down -v
    $ docker compose up -d
    """

    print("Microwave RAG")
    print("=" * 100)

    # Populate vector db if needed
    load_context = input("\nLoad context to VectorDB (y/n)? > ").strip()
    if load_context.lower().strip() in ["y", "yes"]:
        text_processor.process_text_file(
            file_name="microwave_manual.txt",
            chunk_size=400,
            overlap=40,
            dimensions=384,
        )
        print("=" * 100)

    # Set-up SYSTEM_PROMPT
    conversation = Conversation()
    conversation.add_message(Message(Role.SYSTEM, SYSTEM_PROMPT))

    while True:
        user_request = input("\n➡️ ").strip()

        if user_request.lower().strip() in ["quit", "exit"]:
            print("👋 Goodbye")
            break

        # Step 1: Retrieval
        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        context = text_processor.search(
            search_mode=SearchMode.EUCLIDIAN_DISTANCE,
            user_request=user_request,
            top_k=5,
            score_threshold=0.5,
            dimensions=384,
        )

        # Step 2: Augmentation
        print(f"\n{'=' * 100}\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")
        augmented_prompt = USER_PROMPT.format(
            context="\n\n".join(context), query=user_request
        )
        conversation.add_message(Message(Role.USER, augmented_prompt))
        print(f"Prompt:\n{augmented_prompt}")

        # Step 3: Generation
        print(f"\n{'=' * 100}\n🤖 STEP 3: GENERATION\n{'-' * 100}")
        ai_message = completion_client.get_completion(conversation.get_messages())
        print(f"✅ RESPONSE:\n{ai_message.content}")
        print("=" * 100)
        conversation.add_message(ai_message)


main()
