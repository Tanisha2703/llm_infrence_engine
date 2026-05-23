"""Core inference engine wrapping llama-cpp-python."""
import time
import threading
from pathlib import Path
from typing import Optional


class InferenceEngine:
    """Thread-safe wrapper around a llama-cpp-python Llama model."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_threads: int = 4,
        verbose: bool = False,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Download TinyLlama GGUF models — see SETUP.md."
            )
        from llama_cpp import Llama

        self.model_path = model_path
        self.n_ctx = n_ctx
        self._lock = threading.Lock()
        self._model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose,
        )

    @property
    def model(self):
        return self._model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[list] = None,
    ) -> dict:
        """Generate text; return text plus TTFT, total_time, token_count, tps."""
        with self._lock:
            t_start = time.perf_counter()
            first_token_t: Optional[float] = None
            chunks: list[str] = []

            for chunk in self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                stream=True,
                echo=False,
            ):
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                chunks.append(chunk["choices"][0]["text"])

            t_end = time.perf_counter()
            text = "".join(chunks)
            ttft = (first_token_t - t_start) if first_token_t else (t_end - t_start)
            total_time = t_end - t_start
            n_tokens = len(chunks)
            tps = n_tokens / total_time if total_time > 0 else 0.0

            return {
                "text": text,
                "ttft": ttft,
                "total_time": total_time,
                "token_count": n_tokens,
                "tps": tps,
            }

    def generate_single(
        self,
        prompt: str,
        max_tokens: int = 1,
        temperature: float = 0.0,
    ) -> str:
        """Generate a single completion (non-streaming). Used by speculative decoding."""
        with self._lock:
            result = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
            )
            return result["choices"][0]["text"] if result["choices"] else ""
