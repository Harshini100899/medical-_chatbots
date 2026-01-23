"""Utility for tracking LLM latency and token metrics with Ollama."""
import time
from dataclasses import dataclass, field
from typing import Optional, Callable
from contextlib import contextmanager

@dataclass
class LLMMetrics:
    """Stores metrics from an LLM call."""
    latency_seconds: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model: str = ""
    
    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.completion_tokens and self.latency_seconds > 0:
            return self.completion_tokens / self.latency_seconds
        return None
    
    def __str__(self) -> str:
        tps = self.tokens_per_second
        tps_str = f"{tps:.1f} tok/s" if tps else "N/A"
        return (
            f"⏱️ Latency: {self.latency_seconds:.2f}s | "
            f"📊 Tokens: {self.total_tokens or 'N/A'} | "
            f"🚀 Speed: {tps_str}"
        )


class MetricsTracker:
    """Tracks metrics across multiple LLM calls."""
    
    def __init__(self):
        self.calls: list[LLMMetrics] = []
    
    @contextmanager
    def track(self, model: str = "ollama"):
        """Context manager to track a single LLM call."""
        metrics = LLMMetrics(model=model)
        start = time.perf_counter()
        try:
            yield metrics
        finally:
            metrics.latency_seconds = time.perf_counter() - start
            self.calls.append(metrics)
    
    def get_last(self) -> Optional[LLMMetrics]:
        return self.calls[-1] if self.calls else None
    
    def get_summary(self) -> dict:
        if not self.calls:
            return {"total_calls": 0}
        
        total_latency = sum(c.latency_seconds for c in self.calls)
        total_tokens = sum(c.total_tokens or 0 for c in self.calls)
        
        return {
            "total_calls": len(self.calls),
            "total_latency_seconds": round(total_latency, 2),
            "avg_latency_seconds": round(total_latency / len(self.calls), 2),
            "total_tokens": total_tokens,
            "avg_tokens_per_call": round(total_tokens / len(self.calls), 1) if total_tokens else None,
        }


# Global tracker instance
_global_tracker = MetricsTracker()

def get_tracker() -> MetricsTracker:
    return _global_tracker


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars ≈ 1 token for English)."""
    return len(text) // 4
