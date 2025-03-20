from langgraph_sdk import get_client

client = get_client(url="http://localhost:2024")
assistant_id = "paralleliization-llm-agent"

async def main():
    thread = await client.threads.create()

    input_question = {"question": "AI Agent 시대에 커머스 기업의 대응 전략은?"}
    
    async for event in client.runs.stream(thread["thread_id"],
                       assistant_id=assistant_id,
                       input = input_question
                       ):
        answer = event.data.get('answer', None)
        if answer:
            print(answer['content'])
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
