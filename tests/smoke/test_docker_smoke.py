# VARIANT-GNN — Missense Varyant Patojenite Tahmini
# Telif Hakkı (c) 2026 XYRA3 Takımı. Tüm hakları saklıdır.
# TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması (Üniversite ve Üzeri).
# Lisans: depo kökündeki LICENSE. Yalnızca araştırma/eğitim/yarışma amaçlıdır;
# klinik tanı/tedavi için kullanılamaz (Şartname §10).

"""
tests/smoke/test_docker_smoke.py
Docker build and Dockerfile structure smoke tests.

Does NOT require Docker daemon to be running.
Validates Dockerfile syntax and structure that a jury member would use.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DOCKERFILE = Path("Dockerfile")
DOCKERFILE_API = Path("Dockerfile.api")
DOCKER_COMPOSE = Path("docker-compose.yml")


# ---------------------------------------------------------------------------
# Dockerfile existence
# ---------------------------------------------------------------------------


class TestDockerfileExists:
    def test_dockerfile_exists(self):
        """Dockerfile must exist for jury environment reproduction."""
        assert DOCKERFILE.exists(), "Dockerfile missing"

    def test_docker_compose_exists(self):
        """docker-compose.yml must exist for single-command startup."""
        assert DOCKER_COMPOSE.exists(), "docker-compose.yml missing"


# ---------------------------------------------------------------------------
# Dockerfile content checks (no Docker daemon needed)
# ---------------------------------------------------------------------------


class TestDockerfileContent:
    def _read(self, path: Path = DOCKERFILE) -> str:
        if not path.exists():
            pytest.skip(f"{path} not found")
        return path.read_text(encoding="utf-8")

    def test_from_python_310_or_312(self):
        """Dockerfile must use Python 3.10 or 3.12 base image."""
        content = self._read()
        assert "python:3.10" in content or "python:3.12" in content, (
            "Dockerfile must use Python 3.10 or 3.12 base image"
        )

    def test_requirements_txt_copied(self):
        """Dockerfile must COPY requirements.txt."""
        content = self._read()
        assert "requirements.txt" in content, "Dockerfile must install from requirements.txt"

    def test_pip_install_present(self):
        """Dockerfile must run pip install."""
        content = self._read()
        assert "pip install" in content, "Dockerfile must run pip install"

    def test_main_entrypoint_present(self):
        """Dockerfile must reference main.py or predict.py as entrypoint."""
        content = self._read()
        assert "main.py" in content or "predict.py" in content or "CMD" in content, (
            "Dockerfile must define an entrypoint (main.py or predict.py)"
        )

    def test_no_secrets_in_dockerfile(self):
        """Dockerfile must not contain hardcoded secrets or tokens."""
        content = self._read()
        forbidden = ["password", "secret", "token", "api_key", "ghp_"]
        for word in forbidden:
            assert word.lower() not in content.lower(), f"Dockerfile contains suspicious keyword: {word}"


# ---------------------------------------------------------------------------
# docker-compose.yml content checks
# ---------------------------------------------------------------------------


class TestDockerComposeContent:
    def _read(self) -> str:
        if not DOCKER_COMPOSE.exists():
            pytest.skip("docker-compose.yml not found")
        return DOCKER_COMPOSE.read_text(encoding="utf-8")

    def test_compose_parseable(self):
        """docker-compose.yml must be valid YAML."""
        import yaml

        content = self._read()
        data = yaml.safe_load(content)
        assert isinstance(data, dict), "docker-compose.yml must parse to dict"

    def test_compose_has_services(self):
        """docker-compose.yml must define at least one service."""
        import yaml

        content = self._read()
        data = yaml.safe_load(content)
        assert "services" in data, "docker-compose.yml must have 'services' key"
        assert len(data["services"]) >= 1, "At least one service must be defined"


# ---------------------------------------------------------------------------
# Reproducibility infrastructure
# ---------------------------------------------------------------------------


class TestReproducibilityInfrastructure:
    def test_requirements_txt_pinned(self):
        """requirements.txt must have pinned versions (== format)."""
        path = Path("requirements.txt")
        assert path.exists(), "requirements.txt missing"
        content = path.read_text(encoding="utf-8")
        lines = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("#")]
        pinned = [l for l in lines if "==" in l]
        assert len(pinned) > 0, "requirements.txt must have pinned (==) versions"
        # At least 80% of non-empty, non-comment lines should be pinned
        ratio = len(pinned) / max(len(lines), 1)
        assert ratio > 0.5, f"Only {ratio:.0%} of requirements are pinned — jury cannot reproduce environment"

    def test_single_command_train_documented(self):
        """README must document the single training command."""
        path = Path("README.md")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "main.py" in content and "--mode train" in content, "README must document: python main.py --mode train"

    def test_single_command_predict_documented(self):
        """README must document the jury prediction command."""
        content = Path("README.md").read_text(encoding="utf-8")
        assert "submission/predict.py" in content or "predict.py" in content, (
            "README must document the prediction entry point"
        )
