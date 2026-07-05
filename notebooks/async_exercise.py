import asyncio
import time


# 1. This coroutine simulates a single network call to an embedding API
async def mock_embedding_call(chunk_id: int, sem: asyncio.Semaphore) -> str:
    # Use the semaphore as an async context manager to enforce rate limits
    async with sem:
        print(f"[Start] Processing chunk {chunk_id}...")

        # Simulate a 1-second network I/O block
        await asyncio.sleep(1)

        print(f"[Complete] chunk {chunk_id} finished.")

    return f"vector_for_chunk {chunk_id}"


# 2. This is the master coroutine that orchasterates the concurrent run
async def main() -> None:
    # Set semaphore value to 3
    sem = asyncio.Semaphore(3)

    start_time = time.perf_counter()

    tasks = [mock_embedding_call(i, sem) for i in range(10)]
    results = await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    print()
    print("All 10 chunks embedded successfully.")
    print(f"Total time taken{total_time:.2f} seconds.")


if __name__ == "__main__":
    asyncio.run(main())
