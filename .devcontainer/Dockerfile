FROM python:3.13-bookworm

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libgl1 systemctl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid $USER_GID $USERNAME \
 && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

USER $USERNAME
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

RUN pip install --upgrade pip --user \
 && pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
 && pip install --user ultralytics

WORKDIR /home/${USERNAME}/piml-in-metal-am
COPY --chown=${USERNAME}:${USERNAME} pyproject.toml README.md ./
COPY --chown=${USERNAME}:${USERNAME} src/ ./src/

RUN pip install --upgrade pip --user \
 && pip install --user -e ".[dev]"

CMD ["tail", "-f", "/dev/null"]
