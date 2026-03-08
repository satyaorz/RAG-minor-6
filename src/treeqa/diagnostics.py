from __future__ import annotations

from dataclasses import asdict, dataclass, field

from treeqa.backends import build_graph_backend, build_llm_client, build_vector_backend
from treeqa.config import TreeQASettings


@dataclass(slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass(slots=True)
class DiagnosticReport:
    settings: dict[str, str]
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "settings": self.settings,
            "checks": [asdict(check) for check in self.checks],
        }


def run_diagnostics(
    settings: TreeQASettings | None = None, live_llm_probe: bool = False
) -> DiagnosticReport:
    settings = settings or TreeQASettings.from_env()
    report = DiagnosticReport(settings=_settings_summary(settings))
    report.checks.append(_check_llm(settings, live_llm_probe))
    report.checks.append(_check_vector(settings))
    report.checks.append(_check_graph(settings))
    return report


def _check_llm(settings: TreeQASettings, live_probe: bool) -> CheckResult:
    try:
        client = build_llm_client(settings)
    except Exception as exc:
        return CheckResult(name="llm", ok=False, detail=str(exc))

    if client is None:
        return CheckResult(name="llm", ok=True, detail="LLM provider disabled; using local fallbacks.")

    if not live_probe:
        return CheckResult(
            name="llm",
            ok=True,
            detail=f"Configured provider `{settings.llm_provider}` with model `{settings.llm_model}`.",
        )

    try:
        response = client.generate_text("Reply with exactly: OK", "Test")
    except Exception as exc:
        return CheckResult(name="llm", ok=False, detail=f"Live probe failed: {exc}")

    normalized = response.strip()
    if not normalized:
        return CheckResult(name="llm", ok=False, detail="Live probe returned an empty response.")
    if normalized == "OK":
        return CheckResult(name="llm", ok=True, detail="Live probe succeeded and matched the expected response.")
    return CheckResult(
        name="llm",
        ok=True,
        detail=(
            "Live probe reached the model successfully, but the response did not match the "
            f"expected literal output. Returned: {response}"
        ),
    )


def _check_vector(settings: TreeQASettings) -> CheckResult:
    try:
        backend = build_vector_backend(settings)
    except Exception as exc:
        return CheckResult(name="vector", ok=False, detail=str(exc))
    if settings.vector_provider.strip().lower() == "local" and hasattr(backend, "documents"):
        if not backend.documents:
            return CheckResult(
                name="vector",
                ok=False,
                detail="Configured local vector provider, but the index is empty. Add files under data/documents and run `python -m treeqa.cli ingest`.",
            )

    return CheckResult(
        name="vector",
        ok=True,
        detail=f"Configured vector provider `{settings.vector_provider}` ({backend.__class__.__name__}).",
    )


def _check_graph(settings: TreeQASettings) -> CheckResult:
    try:
        backend = build_graph_backend(settings)
    except Exception as exc:
        return CheckResult(name="graph", ok=False, detail=str(exc))
    if settings.graph_provider.strip().lower() == "local" and hasattr(backend, "facts"):
        if not backend.facts:
            return CheckResult(
                name="graph",
                ok=False,
                detail="Configured local graph provider, but the index is empty. Add facts under data/graph and run `python -m treeqa.cli ingest`.",
            )

    return CheckResult(
        name="graph",
        ok=True,
        detail=f"Configured graph provider `{settings.graph_provider}` ({backend.__class__.__name__}).",
    )


def _settings_summary(settings: TreeQASettings) -> dict[str, str]:
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model or "<unset>",
        "vector_provider": settings.vector_provider,
        "graph_provider": settings.graph_provider,
        "data_dir": settings.data_dir,
    }
