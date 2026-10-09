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
        proposer = self.prepare_agent(WriterAgent(provider=self._provider, config=self._config))
        critic = self.prepare_agent(AnalysisAgent(provider=self._provider, config=self._config))

        # Step 1: Initial proposal
        prop_res = proposer.run(sub_task, context, bus=bus)
        if not prop_res.success:
            return prop_res

        # Step 2: Critique
        critique_task = SubTask(
            id=f"{sub_task.id}_critique",
            label="analyze",
            description=f"Critique and identify weaknesses or flaws in the following proposal:\n{prop_res.content}",
        )
        crit_res = critic.run(critique_task, context, bus=bus)

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

        p_tokens = prop_res.token_usage.get("prompt_tokens", 0) + crit_res.token_usage.get("prompt_tokens", 0)
        c_tokens = prop_res.token_usage.get("completion_tokens", 0) + crit_res.token_usage.get("completion_tokens", 0)

        try:
            resp = self._provider.chat(synth_messages)
            final_content = resp.content
            p_tokens += resp.usage.get("prompt_tokens", 0)
            c_tokens += resp.usage.get("completion_tokens", 0)
        except Exception as exc:
            logger.warning("Debate synthesis failed: %s. Using initial proposal.", exc)
            final_content = prop_res.content

        if bus:
            bus.publish(sub_task.id, final_content, writer="debate_consensus")

        return AgentResult(
            sub_task_id=sub_task.id,
            agent_name="debate_consensus",
            content=final_content,
            success=True,
            token_usage={
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
                "total_tokens": p_tokens + c_tokens,
            },
        )

    async def acoordinate(self, sub_task: SubTask, context: Dict[str, Any], bus: CommunicationBus) -> AgentResult:
        proposer = self.prepare_agent(WriterAgent(provider=self._provider, config=self._config))
        critic = self.prepare_agent(AnalysisAgent(provider=self._provider, config=self._config))
        prop_res = await proposer.arun(sub_task, context, bus=bus)
        if not prop_res.success:
            return prop_res
        critique_task = SubTask(id=f"{sub_task.id}_critique", label="analyze", description=f"Critique:\n{prop_res.content}")
        crit_res = await critic.arun(critique_task, context, bus=bus)
        messages = [Message(role="system", content="Reconcile the proposal and critique."), Message(role="user", content=f"Proposal:\n{prop_res.content}\n\nCritique:\n{crit_res.content}")]
        try:
            response = await self._provider.achat(messages)
            content = response.content
            usage = response.usage
        except Exception:
            content, usage = prop_res.content, {}
        if bus:
            bus.publish(sub_task.id, content, writer="debate_consensus")
        prompt = prop_res.token_usage.get("prompt_tokens", 0) + crit_res.token_usage.get("prompt_tokens", 0) + usage.get("prompt_tokens", 0)
        completion = prop_res.token_usage.get("completion_tokens", 0) + crit_res.token_usage.get("completion_tokens", 0) + usage.get("completion_tokens", 0)
        return AgentResult(sub_task.id, "debate_consensus", content, token_usage={"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}, metadata={"model": getattr(response, "model", "unknown") if 'response' in locals() else "unknown"})
