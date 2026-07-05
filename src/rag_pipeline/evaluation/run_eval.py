import asyncio

from rag_pipeline.evaluation.harness import EvaluationHarness


async def main() -> None:
    print()
    print("         LAUNCHING AUTOMATED RAGAS EVALUATION HARNESS       ")
    print()

    # Initialize the evaluation harness
    harness = EvaluationHarness(collection_name="corporate_leave_policy")

    # Run the full live evaluation benchmark
    await harness.run_evaluation()


if __name__ == "__main__":
    asyncio.run(main())
