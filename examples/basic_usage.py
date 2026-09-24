"""Basic AMACS usage example.

Run with:
    python examples/basic_usage.py

Or via the CLI:
    amacs run examples/basic_usage.py
"""

from amacs import amacs


# ── Simple sync example ──────────────────────────────────────────────────

@amacs(max_agents=8, strategy="performance")
def research_task(topic: str) -> str:
    """Conduct comprehensive research on the given topic."""
    return f"Research on {topic}"


# ── Async example ─────────────────────────────────────────────────────────

@amacs(max_agents=4, strategy="speed", adaptive=True)
async def async_research(topic: str) -> str:
    """Async variant — same decorator, auto-detected."""
    return f"Async research on {topic}"


# ── Cost-optimised example ────────────────────────────────────────────────

@amacs(max_agents=2, strategy="cost", retry_limit=1)
def quick_summary(topic: str) -> str:
    """Generate a brief summary with minimal resources."""
    return f"Summarise {topic}"


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    print("=" * 60)
    print("AMACS Basic Usage Example")
    print("=" * 60)

    # Sync call
    print("\n▸ Running research_task('Renewable Energy')...")
    result = research_task("Renewable Energy")
    print(f"\n✓ Result ({len(result)} chars):\n{result[:500]}...")

    print("\n" + "=" * 60)

    # Sync cost-optimised call
    print("\n▸ Running quick_summary('Quantum Computing')...")
    result2 = quick_summary("Quantum Computing")
    print(f"\n✓ Result ({len(result2)} chars):\n{result2[:500]}...")

    print("\n" + "=" * 60)

    # Async call
    import asyncio

    async def run_async() -> None:
        print("\n▸ Running async_research('Climate Change')...")
        result3 = await async_research("Climate Change")
        print(f"\n✓ Result ({len(result3)} chars):\n{result3[:500]}...")

    asyncio.run(run_async())
    print("\nDone!")
