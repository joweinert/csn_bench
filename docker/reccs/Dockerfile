FROM condaforge/miniforge3:latest

ARG RECCS_REF
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    openmpi-bin libopenmpi-dev \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/illinois-or-research-analytics/lanne2_networks.git /opt/lanne2_networks \
  && cd /opt/lanne2_networks \
  && git checkout ${RECCS_REF}

# Patch for RECCS repo: parses 2-column TSV (where column 2 may contain spaces or tabs)
# -> its the Mr Hi special treatment
RUN python - <<'PY'
from pathlib import Path

p = Path("/opt/lanne2_networks/generate_synthetic_networks/clean_outliers.py")
txt = p.read_text(encoding="utf-8")

old = "node, _ = line.strip().split()"
new = "node, _ = line.rstrip('\\n').split('\\t', 1)"

if old not in txt:
    raise SystemExit("Patch failed: target line not found (file changed upstream).")

p.write_text(txt.replace(old, new), encoding="utf-8")
print("Patched:", p)
PY

COPY docker/reccs/environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml && conda clean -afy

ENV PATH=/opt/conda/envs/reccs/bin:$PATH
WORKDIR /workspace
CMD ["bash"]
