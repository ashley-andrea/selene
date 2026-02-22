"""
Simulator Node — calls the Simulator Model API for each pill the agent selected.

The agent has already decided WHICH pills to simulate (via risk_assessor).
This node executes those API calls concurrently and collects results.
Results accumulate across iterations so the agent can compare all seen pills.
"""

import asyncio
import logging

from agent.state import SystemState
from agent.tools.simulator_api import call_simulator
from agent.pill_database import get_pill_by_id

logger = logging.getLogger(__name__)


def run(state: SystemState) -> dict:
    """
    Calls the Simulator API for each pill selected by the agent.
    Uses pills_to_simulate (agent's choice), falls back to full candidate_pool.
    Merges new results with any existing results from prior iterations.
    """
    pills = state.get("pills_to_simulate") or state.get("candidate_pool", [])
    patient_data = state["patient_data"]

    if not pills:
        logger.warning("No pills to simulate")
        return {"simulated_results": {}}

    logger.info("Simulating %d pills: %s", len(pills), pills)

    # Fetch full pill records for each pill ID
    pill_records = {}
    for pill_id in pills:
        pill_record = get_pill_by_id(pill_id)
        if pill_record:
            pill_records[pill_id] = pill_record
        else:
            logger.warning("Pill %s not found in database — skipping", pill_id)

    async def run_all():
        tasks = [
            call_simulator(pill_records[pill_id], patient_data) 
            for pill_id in pills 
            if pill_id in pill_records
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    # Run the async simulation loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            results_list = pool.submit(lambda: asyncio.run(run_all())).result()
    else:
        results_list = asyncio.run(run_all())

    # Merge with prior iteration results (accumulate across iterations)
    simulated_results = dict(state.get("simulated_results") or {})
    new_count = 0
    # Filter pills to only those that were found in the database
    valid_pills = [pill_id for pill_id in pills if pill_id in pill_records]
    for pill_id, result in zip(valid_pills, results_list):
        if isinstance(result, Exception):
            logger.error("Simulation failed for %s: %s", pill_id, result)
            continue
        if result is None:
            logger.warning("Simulation returned None for %s — skipping", pill_id)
            continue
        simulated_results[pill_id] = result
        new_count += 1

    logger.info(
        "Simulated %d/%d pills this iteration (total results across iterations: %d)",
        new_count,
        len(pills),
        len(simulated_results),
    )

    return {"simulated_results": simulated_results}
