FROM mcr.microsoft.com/devcontainers/base:bookworm

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

USER vscode

RUN set -eux; \
    case "$(dpkg --print-architecture)" in \
        amd64) miniforge_arch=x86_64 ;; \
        arm64) miniforge_arch=aarch64 ;; \
        ppc64el) miniforge_arch=ppc64le ;; \
        s390x) miniforge_arch=s390x ;; \
        *) echo "Unsupported architecture: $(dpkg --print-architecture)" >&2; exit 1 ;; \
    esac; \
    curl -fsSL \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${miniforge_arch}.sh" \
        -o /tmp/miniforge.sh; \
    bash /tmp/miniforge.sh -b -p /home/vscode/miniforge3; \
    rm /tmp/miniforge.sh
