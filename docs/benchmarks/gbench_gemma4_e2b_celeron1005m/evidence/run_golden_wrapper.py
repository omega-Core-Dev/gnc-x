#!/usr/bin/env python3
"""
Launcher fino e auditavel para executar o google-deepmind/gbench oficial
contra o servidor GNC-X sem modificar nenhum arquivo de /tmp/gbench_official.

Ajustes exclusivos em memoria:
  1) golden.DEFAULT_TIMEOUT_S = 600 e
     golden.GoldenBenchmarkRunner._request_json.__defaults__ = (600,)
     (necessario em Intel Celeron 1005M @ 1.90GHz SSE4.2 sem AVX/GPU: decode ~1.76 tok/s,
      encoder visual ~14.6s, evitando falso timeout no limite fixo de 30s do Golden Set).
  2) Hook pass-through em GoldenBenchmarkRunner._query_remote_endpoint apenas para medir
     o wall_sec real e o tamanho da resposta de cada requisicao em task_timings.json,
     sem alterar payload, resposta nem avaliacao.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from gbench.cli import main as gbench_main
from gbench.runners import golden


TIMINGS_PATH = Path(
    "/home/glc/Documentos/nucleo_memory-001/motor_memoserve/workspace_memory/benchmarks/gbench_gemma4_e2b/task_timings.json"
)


def main() -> int:
    repo = "/tmp/gbench_official"
    st = subprocess.check_output(["git", "-C", repo, "status", "--porcelain"], text=True).strip()
    head = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    assert st == "", f"Repositorio oficial {repo} modificado: {st!r}"

    fp_const = golden.GOLDEN_SCORING_FP
    fp_runtime = golden.scoring_fingerprint()

    golden.DEFAULT_TIMEOUT_S = 600
    golden.GoldenBenchmarkRunner._request_json.__defaults__ = (600,)

    orig_query = golden.GoldenBenchmarkRunner._query_remote_endpoint
    timings = []

    def timed_query(self, messages, endpoint_url, tools=None, tool_choice=None, sampling=None, model_id=None):
        t0 = time.monotonic()
        first_txt = ""
        has_img = False
        has_aud = False
        for m in (messages or []):
            media = m.get("media") or []
            for item in media:
                mt = (item.get("type") or "").lower()
                if mt == "image":
                    has_img = True
                elif mt == "audio":
                    has_aud = True
            if m.get("role") == "user" and not first_txt:
                first_txt = str(m.get("content") or "")[:90]
        res = orig_query(
            self,
            messages,
            endpoint_url,
            tools=tools,
            tool_choice=tool_choice,
            sampling=sampling,
            model_id=model_id,
        )
        dt = time.monotonic() - t0
        timings.append(
            {
                "index": len(timings) + 1,
                "wall_sec": round(dt, 3),
                "has_tools": bool(tools),
                "has_image": has_img,
                "has_audio": has_aud,
                "prompt_preview": first_txt,
                "served_model": res.get("served_model", ""),
                "response_chars": len(res.get("content") or ""),
                "tool_calls_count": len(res.get("tool_calls") or []),
            }
        )
        TIMINGS_PATH.write_text(json.dumps(timings, indent=2, ensure_ascii=False), encoding="utf-8")
        return res

    golden.GoldenBenchmarkRunner._query_remote_endpoint = timed_query

    print(
        f"[run_golden_wrapper] repo={repo} commit={head[:12]} clean=True "
        f"scoring_version={golden.SCORING_VERSION} fp_const={fp_const} "
        f"fp_runtime={fp_runtime} socket_timeout_s=600 "
        f"sandbox={os.environ.get('GBENCH_SANDBOX', 'required')}",
        flush=True,
    )
    return int(gbench_main() or 0)


if __name__ == "__main__":
    sys.exit(main())
