"""The dashboard and alert rules are code that talks to the server through metric names. If a metric is renamed and
the dashboard is not, Grafana just shows 'No data' and nobody notices until an incident. These tests make that drift
a failing test instead (the queries were also run for real against Prometheus and Grafana: docs/RESULTS.md)."""
import json
import re
from pathlib import Path

import pytest
import torch
import yaml
from prometheus_client import generate_latest

from miniserve.loader import DEFAULT_MODEL, load_model
from miniserve.server import Server

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
SERVER_SRC = (Path(__file__).resolve().parents[1] / "src/miniserve/server.py").read_text(encoding="utf8")


@pytest.fixture(scope="module")
def exported():
    """Every metric family name the server can export, over both engine kinds (the KV gauges differ)."""
    model, tok = load_model(DEFAULT_MODEL, dtype=torch.float32)
    names = set()
    for kind in ("continuous", "paged"):
        srv = Server(model, tok, engine_kind=kind, max_slots=2, max_len=64)
        for line in generate_latest(srv.registry).decode().splitlines():
            if line.startswith("# TYPE "):
                names.add(line.split()[2])
            elif line and not line.startswith("#"):
                names.add(re.split(r"[{ ]", line)[0])
    return names


def referenced(expr: str) -> set[str]:
    return {re.sub(r"_(bucket|sum|count)$", "", n) for n in re.findall(r"miniserve_[a-z0-9_]+", expr)}


def families(exported_names: set[str]) -> set[str]:
    return {re.sub(r"_(bucket|sum|count|created)$", "", n) for n in exported_names} | exported_names


def dashboard():
    return json.loads((DEPLOY / "grafana/dashboards/miniserve.json").read_text(encoding="utf8"))


def alert_exprs():
    doc = yaml.safe_load((DEPLOY / "alerts.yml").read_text(encoding="utf8"))
    return [(r["alert"], r["expr"]) for g in doc["groups"] for r in g["rules"]]


def test_every_dashboard_query_uses_a_metric_the_server_exports(exported):
    known = families(exported)
    missing = {(p["title"], m) for p in dashboard()["panels"] for t in p["targets"] for m in referenced(t["expr"]) if m not in known}
    assert not missing, f"dashboard references metrics the server does not export: {missing}"


def test_every_alert_rule_uses_a_metric_the_server_exports(exported):
    known = families(exported)
    missing = {(name, m) for name, e in alert_exprs() for m in referenced(e) if m not in known}
    assert not missing, f"alerts reference metrics the server does not export: {missing}"


def test_status_label_values_in_queries_are_ones_the_server_emits():
    exprs = [t["expr"] for p in dashboard()["panels"] for t in p["targets"]] + [e for _, e in alert_exprs()]
    for e in exprs:
        for group in re.findall(r'status=~?"([^"]+)"', e):
            for value in group.split("|"):
                assert f'"{value}"' in SERVER_SRC, f"status {value!r} is never emitted by the server"


def test_dashboard_structure_is_sane():
    d = dashboard()
    ids = [p["id"] for p in d["panels"]]
    assert len(ids) == len(set(ids)), "duplicate panel ids"
    for p in d["panels"]:
        g = p["gridPos"]
        assert 0 <= g["x"] and g["x"] + g["w"] <= 24, f'{p["title"]} overflows the 24-column grid'
        assert p["datasource"]["uid"] == "prometheus"
    ds = yaml.safe_load((DEPLOY / "grafana/provisioning/datasources/prometheus.yml").read_text(encoding="utf8"))
    assert ds["datasources"][0]["uid"] == "prometheus"  # panels point at the provisioned datasource by this uid


def test_histogram_quantile_queries_aggregate_by_le():
    """histogram_quantile needs the 'le' label; dropping it (a classic mistake) silently yields NaN."""
    for p in dashboard()["panels"]:
        for t in p["targets"]:
            if "histogram_quantile" in t["expr"]:
                assert "sum by (le)" in t["expr"], p["title"]
    for name, e in alert_exprs():
        if "histogram_quantile" in e:
            assert "sum by (le)" in e, name


def test_compose_wires_the_stack_together_and_every_mounted_file_exists():
    root = DEPLOY.parent
    compose = yaml.safe_load((root / "docker-compose.yml").read_text(encoding="utf8"))
    services = compose["services"]
    assert set(services) == {"miniserve", "prometheus", "grafana"}
    assert "8000:8000" in services["miniserve"]["ports"]
    # Prometheus scrapes the server by its compose service name and port
    cfg = yaml.safe_load((DEPLOY / "prometheus.yml").read_text(encoding="utf8"))
    assert cfg["rule_files"] == ["alerts.yml"]
    assert cfg["scrape_configs"][0]["static_configs"][0]["targets"] == ["miniserve:8000"]
    # every host file the compose file mounts must exist (a typo here only shows up when the stack is started)
    for name in ("prometheus", "grafana"):
        for vol in services[name]["volumes"]:
            host = vol.split(":")[0]
            assert (root / host).exists(), f"{name} mounts {host}, which does not exist"
    assert (root / "Dockerfile").exists()
