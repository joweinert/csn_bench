FROM condaforge/miniforge3:latest

ARG ECSBM_REF

ENV DEBIAN_FRONTEND=noninteractive

# build tools + MPI: needed for pymincut
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
    build-essential cmake \
    openmpi-bin libopenmpi-dev \
  && rm -rf /var/lib/apt/lists/*

# conda-forge only -> avoids repo.anaconda.com warnings + channel conflicts
RUN conda config --remove channels defaults || true \
 && conda config --add channels conda-forge \
 && conda config --set channel_priority strict

# Cloning EC-SBM and checkout pinned commit
RUN git clone https://github.com/illinois-or-research-analytics/ec-sbm.git /opt/ec-sbm \
 && cd /opt/ec-sbm \
 && git checkout "${ECSBM_REF}"

#env creation
COPY docker/ecsbm/environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml \
 && conda clean -afy

RUN python - <<'PY'
from pathlib import Path

p = Path("/opt/ec-sbm/ec-sbm/clean_outlier.py")
txt = p.read_text(encoding="utf-8")

old = "node, _ = line.strip().split()"
new = "node, _ = line.rstrip('\\n').split('\\t', 1)"

if old not in txt:
    raise SystemExit("Patch failed: target line not found (file changed upstream).")

p.write_text(txt.replace(old, new), encoding="utf-8")
print("Patched:", p)
PY

ENV PATH=/opt/conda/envs/ec-sbm/bin:$PATH

WORKDIR /workspace
CMD ["bash"]
