FROM python:3.10-slim

LABEL maintainer="Oracle1 <fleet@cocapn.ai>"
LABEL description="PLATO self-training rooms — 21 AI training methods as grab-and-go rooms"
LABEL version="0.5.0a1"

WORKDIR /app

# Copy package
COPY src/ /app/src/
COPY setup.py README.md LICENSE /app/

# Ensure room_base is available in presets for relative imports
RUN cp /app/src/room_base.py /app/src/presets/room_base.py

# Install
RUN pip install --no-cache-dir .

# Create dirs for tiles and ensigns
RUN mkdir -p /data/tiles /data/ensigns

# Health check
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python3 -c "from presets import PRESET_MAP; print(f'{len(PRESET_MAP)} presets')" || exit 1

# Default: run tests
CMD ["python3", "-c", "import sys; sys.path.insert(0, '/app/src'); from presets import PRESET_MAP; print(f'plato-torch: {len(PRESET_MAP)} presets ready')"]
