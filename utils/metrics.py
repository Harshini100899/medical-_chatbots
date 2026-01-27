"""Simple metrics tracking for LLM calls."""
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class CallMetrics:
    model: str = ""
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __str__(self):
        return f"Model: {self.model} | Latency: {self.latency_seconds:.2f}s | Tokens: {self.total_tokens}"


class MetricsTracker:
    def __init__(self):
        self.calls: List[CallMetrics] = []
    
    @contextmanager
    def track(self, model: str = "unknown"):
        metrics = CallMetrics(model=model)
        start = time.perf_counter()
        try:
            yield metrics
        finally:
            metrics.latency_seconds = time.perf_counter() - start
            self.calls.append(metrics)
    
    def get_last(self) -> Optional[CallMetrics]:
        return self.calls[-1] if self.calls else None
    
    def get_summary(self) -> dict:
        if not self.calls:
            return {"total_calls": 0, "total_latency_seconds": 0, "avg_latency_seconds": 0, "total_tokens": 0}
        
        total_latency = sum(c.latency_seconds for c in self.calls)
        total_tokens = sum(c.total_tokens for c in self.calls)
        return {
            "total_calls": len(self.calls),
            "total_latency_seconds": round(total_latency, 2),
            "avg_latency_seconds": round(total_latency / len(self.calls), 2),
            "total_tokens": total_tokens,
            "avg_tokens_per_call": round(total_tokens / len(self.calls), 1) if total_tokens else 0
        }


_tracker = MetricsTracker()

def get_tracker() -> MetricsTracker:
    return _tracker

def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token)."""
    return len(text) // 4 if text else 0
