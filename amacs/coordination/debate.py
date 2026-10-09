"""DebateCoordinator — multi-agent consensus and critique protocol.

Runs a multi-turn debate between a proposer agent and a critic agent, followed
by a synthesizer pass to produce a high-consensus result.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from amacs.agents.analysis_agent import AnalysisAgent
from amacs.agents.base_agent import AgentResult, SubTask
from amacs.agents.writer_agent import WriterAgent
from amacs.communication import CommunicationBus
from amacs.coordination.base import Coordinator
from amacs.integrations.llm_providers import Message

logger = logging.getLogger("amacs.coordination.debate")


class DebateCoordinator(Coordinator):
    """Executes a multi-turn Proposer vs. Critic debate protocol."""

    def coordinate(
        self,
        sub_task: SubTask,
        context: Dict[str, Any],
        bus: CommunicationBus,
    ) -> AgentResult:
        proposer = WriterAgent(provider=self._provider, config=self._config)
        critic = AnalysisAgent(provider=self._provider, config=self._config)

        # Step 1: Initial proposal
        prop_res = proposer.run(sub_task, context, bus=None)
        if not prop_res.success:
            return prop_res

        # Step 2: Critique
        critique_task = SubTask(
            id=f"{sub_task.id}_critique",
            label="analyze",
            description=f"Critique and identify weaknesses or flaws in the following proposal:\n{prop_res.content}",
        )
        crit_res = critic.run(critique_task, context, bus=None)

        # Step 3: Consensus synthesis
        synth_messages = [
            Message(
                role="system",
                content="You are a consensus synthesizer. Reconcile the initial proposal with the critique to produce a polished final version.",
            ),
            Message(
                role="user",
                content=(
                    f"Initial Proposal:\n{prop_res.content}\n\n"
                    f"Critique:\n{crit_res.content if crit_res.success else 'No major issues'}\n\n"
                    "Produce the reconciled final output:"
                ),
            ),
        ]

        try:
            resp = self._provider.chat(synth_messages)
            final_content = resp.content
        except Exception as exc:
            logger.warning("Debate synthesis failed: %s. Using initial proposal.", exc)
            final_content = prop_res.content

        return AgentResult(
            sub_task_id=sub_task.id,
            agent_name="debate_consensus",
            content=final_content,
            success=True,
        )
