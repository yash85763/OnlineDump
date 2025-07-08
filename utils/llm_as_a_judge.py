############################################################
# LLM Judge Pipeline                                      #
# ------------------------------------------------------- #
# A reference implementation of the three‑stage cascade   #
# described in the “LLM‑as‑a‑Judge” blueprint (J1 Pairwise,#
# J1 Pointwise‑Consistency, JudgeLRM Deep‑Reason Judge).   #
#                                                         #
# This script shows *one* clean way to wire up existing    #
# open‑sourced checkpoints from HuggingFace using the      #
# 🤗 Transformers API.  Replace the model IDs with the     #
# exact checkpoints you have access to (or fine‑tuned     #
# variants).                                              #
############################################################

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    pipeline,
)

# ----------------------------------------------------------------------------------
# Utility classes & helpers
# ----------------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


@dataclass
class JudgeResult:
    verdict: str  # "good" | "bad" | "tie"
    score: float  # higher ⇒ better/higher confidence that answer is good
    thoughts: List[str]  # chain‑of‑thought traces (may be empty)
    elapsed: float  # seconds


# ----------------------------------------------------------------------------------
# Base wrapper that loads any causal‑LM in chat format and provides a generate() API.
# ----------------------------------------------------------------------------------

class ChatModel:
    """Thin convenience wrapper around causal‑LM checkpoints in chat template form."""

    def __init__(self, model_name: str, system_prompt: str = "", max_new_tokens: int = 512):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
            device_map="auto",
            trust_remote_code=True,
        )
        self.system_prompt = system_prompt.strip()
        self.max_new_tokens = max_new_tokens

        # A simple streamer prints tokens as they generate – handy for debugging.
        self.debug_streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # High‑level helpers
    # ------------------------------------------------------------------

    def build_chat_prompt(self, user_content: str) -> str:
        if self.system_prompt:
            return f"<|system|>\n{self.system_prompt}\n<|user|>\n{user_content}\n<|assistant|>"
        return f"<|user|>\n{user_content}\n<|assistant|>"

    @torch.inference_mode()
    def generate(self, user_content: str, temperature: float = 0.1, top_p: float = 0.9, stream: bool = False) -> str:
        """Generate one response for *user_content* and return raw text."""
        prompt = self.build_chat_prompt(user_content)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            streamer=self.debug_streamer if stream else None,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Return only the assistant's final turn (after the last <|assistant|>)
        split_token = "<|assistant|>"
        if split_token in text:
            text = text.split(split_token)[-1].strip()
        return text


# ----------------------------------------------------------------------------------
# 1️⃣  Quick Pairwise Judge (J1‑pairwise)
# ----------------------------------------------------------------------------------

PAIRWISE_SYSTEM_PROMPT = """You are an expert judge. Compare Answer A and Answer B to the USER question.  \
Write a short chain‑of‑thought ending with exactly one line of the form  \
`VERDICT: A` if Answer A is better,  \
`VERDICT: B` if Answer B is better, or  \
`VERDICT: TIE` if they are equally good.  \
Be rigorous – factuality, reasoning and style all matter."""


class PairwiseJudge:
    def __init__(self, model_name: str = "meta‑llm/J1‑8B‑Pairwise"):
        self.model = ChatModel(model_name, system_prompt=PAIRWISE_SYSTEM_PROMPT, max_new_tokens=512)

    def score(self, prompt: str, answer_a: str, answer_b: str, temperature: float = 0.0) -> JudgeResult:
        """Return JudgeResult; verdict ∈ {A, B, TIE}."""
        user_query = (
            f"USER QUESTION:\n{prompt}\n\nAnswer A:\n{answer_a}\n\nAnswer B:\n{answer_b}\n"
        )
        start = time.perf_counter()
        raw = self.model.generate(user_query, temperature=temperature)
        elapsed = time.perf_counter() - start

        # Parse verdict
        verdict_line = next((l for l in raw.splitlines() if l.strip().upper().startswith("VERDICT:")), "VERDICT: TIE")
        verdict = verdict_line.split(":", 1)[1].strip().upper()
        verdict = "tie" if "TIE" in verdict else ("good" if verdict == "A" else "bad")

        score = 1.0 if verdict == "good" else (0.0 if verdict == "bad" else 0.5)
        return JudgeResult(verdict=verdict, score=score, thoughts=[raw], elapsed=elapsed)


# ----------------------------------------------------------------------------------
# 2️⃣  Pointwise + Self‑Consistency Judge (J1‑pointwise)
# ----------------------------------------------------------------------------------

POINTWISE_SYSTEM_PROMPT = """You are a rigorous evaluator.  Given the USER question and one ANSWER, think step‑by‑step and output  \
`SCORE: <p>` where <p> is an integer 1‑10 (10 = perfect)."""


class PointwiseJudge:
    def __init__(self, model_name: str = "meta‑llm/J1‑8B‑Pointwise", self_consistency_samples: int = 16):
        self.model = ChatModel(model_name, system_prompt=POINTWISE_SYSTEM_PROMPT, max_new_tokens=512)
        self.n = self_consistency_samples

    def _single_score(self, prompt: str, answer: str, temperature: float = 0.7) -> Tuple[int, str]:
        user_query = f"USER QUESTION:\n{prompt}\n\nANSWER:\n{answer}\n"
        raw = self.model.generate(user_query, temperature=temperature)
        score_line = next((l for l in raw.splitlines() if l.strip().upper().startswith("SCORE:")), "SCORE: 5")
        try:
            score = int(score_line.split(":", 1)[1].strip())
        except ValueError:
            score = 5
        return score, raw

    def score(self, prompt: str, answer: str) -> JudgeResult:
        scores, thoughts = [], []
        start = time.perf_counter()
        for _ in range(self.n):
            s, t = self._single_score(prompt, answer)
            scores.append(s)
            thoughts.append(t)
        elapsed = time.perf_counter() - start
        avg = sum(scores) / len(scores)
        # linear‑map 1‑10 onto 0‑1 convenience score
        norm_score = (avg - 1) / 9.0
        verdict = "good" if norm_score >= 0.6 else ("bad" if norm_score <= 0.4 else "tie")
        return JudgeResult(verdict=verdict, score=norm_score, thoughts=thoughts, elapsed=elapsed)


# ----------------------------------------------------------------------------------
# 3️⃣  Deep‑Reason Judge (JudgeLRM‑7B)
# ----------------------------------------------------------------------------------

DEEP_SYSTEM_PROMPT = """You are JudgeLRM, an advanced reasoning model.  Provide a thorough critique and then output  \
`FINAL SCORE: <p>` where <p> is an integer 1‑10.  Use full chain‑of‑thought reasoning."""


class DeepJudge:
    def __init__(self, model_name: str = "nus‑ai/JudgeLRM‑7B"):
        self.model = ChatModel(model_name, system_prompt=DEEP_SYSTEM_PROMPT, max_new_tokens=1024)

    def score(self, prompt: str, answer: str, temperature: float = 0.3) -> JudgeResult:
        user_query = f"USER QUESTION:\n{prompt}\n\nANSWER:\n{answer}\n"
        start = time.perf_counter()
        raw = self.model.generate(user_query, temperature=temperature)
        elapsed = time.perf_counter() - start

        score_line = next((l for l in raw.splitlines() if l.upper().startswith("FINAL SCORE:")), "FINAL SCORE: 5")
        try:
            score = int(score_line.split(":", 1)[1].strip())
        except ValueError:
            score = 5
        norm_score = (score - 1) / 9.0
        verdict = "good" if norm_score >= 0.7 else ("bad" if norm_score <= 0.3 else "tie")
        return JudgeResult(verdict=verdict, score=norm_score, thoughts=[raw], elapsed=elapsed)


# ----------------------------------------------------------------------------------
# Cascaded evaluator orchestrating the three judges
# ----------------------------------------------------------------------------------

class CascadedEvaluator:
    def __init__(
        self,
        pairwise_model: str | None = None,
        pointwise_model: str | None = None,
        deep_model: str | None = None,
        self_consistency_samples: int = 16,
        quick_confidence: float = 0.9,
        pointwise_threshold: float = 0.6,
        deep_threshold: float = 0.7,
    ):
        self.pairwise = PairwiseJudge(pairwise_model) if pairwise_model or pairwise_model is None else pairwise_model
        self.pointwise = PointwiseJudge(pointwise_model, self_consistency_samples)
        self.deep = DeepJudge(deep_model)
        self.quick_confidence = quick_confidence
        self.pointwise_threshold = pointwise_threshold
        self.deep_threshold = deep_threshold

    def evaluate(
        self,
        prompt: str,
        candidate_answer: str,
        baseline_answer: Optional[str] = None,
    ) -> JudgeResult:
        """Return a final JudgeResult plus a metadata dict for logging."""

        meta = {}

        # ---------------------------------------------
        # Stage 1: Quick pairwise vs. baseline answer
        # ---------------------------------------------
        if baseline_answer is None:
            baseline_answer = "I don't know."  # simplest bad baseline

        pair_res = self.pairwise.score(prompt, candidate_answer, baseline_answer)
        meta["pairwise"] = pair_res
        if pair_res.verdict == "good" and pair_res.score >= self.quick_confidence:
            return pair_res  # early exit

        # ---------------------------------------------
        # Stage 2: Pointwise + self‑consistency
        # ---------------------------------------------
        point_res = self.pointwise.score(prompt, candidate_answer)
        meta["pointwise"] = point_res
        if point_res.score >= self.pointwise_threshold:
            return point_res

        # ---------------------------------------------
        # Stage 3: Deep reasoning judge
        # ---------------------------------------------
        deep_res = self.deep.score(prompt, candidate_answer)
        meta["deep"] = deep_res
        # Pass through deep verdict or override with tie if uncertain
        final = deep_res
        final.thoughts.extend(pair_res.thoughts + point_res.thoughts)
        return final


# ----------------------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------------------

def demo():
    evaluator = CascadedEvaluator()

    prompt = "Explain photosynthesis in one paragraph."
    candidate = (
        "Photosynthesis is the process by which green plants use sunlight to produce glucose and oxygen from carbon "
        "dioxide and water.  In the chloroplasts, chlorophyll absorbs light energy, driving the light‑dependent "
        "reactions that split water into oxygen, protons and electrons.  The energized electrons move through the "
        "electron transport chain, generating ATP and NADPH.  In the Calvin cycle, these energy carriers power the "
        "fixation of CO₂ into 3‑carbon sugars, which are later assembled into glucose.  Oxygen is released as a "
        "by‑product."
    )

    result = evaluator.evaluate(prompt, candidate)

    print("\n=========  FINAL VERDICT  =========")
    print(f"Verdict:  {result.verdict.upper()}  |  Score: {result.score:.2f}")
    print(f"(Evaluated in {result.elapsed:.1f} s across all stages)")

    # Optional: dump abbreviated chain‑of‑thoughts
    print("\nSample reasoning (first judge):\n")
    print(result.thoughts[0][:1000])


if __name__ == "__main__":
    demo()
