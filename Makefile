.PHONY: help setup download-models run-dev run-prod test clean

help:
	@echo "Available commands:"
	@echo "  setup          - Set up development environment"
	@echo "  download-models - Download all required models"
	@echo "  run-dev        - Run all services in development mode"
	@echo "  run-stt        - Run only STT service"
	@echo "  run-tts        - Run only TTS service"
	@echo "  run-llm        - Run only LLM service"
	@echo "  run-kg         - Run only KG service"
	@echo "  test           - Run all tests"
	@echo "  clean          - Clean build artifacts"

setup:
	@echo "Setting up Rust environment..."
	cd rust-services && cargo build
	@echo "Setting up Python environment..."
	cd python-services && uv sync --all-extras
	@echo "Creating necessary directories..."
	mkdir -p data/models logs

download-models:
	bash scripts/download_models.sh

run-dev:
	@echo "Starting all services in development mode..."
	@tmux new-session -d -s audio-guide
	@tmux send-keys -t audio-guide:0 'make run-stt' C-m
	@tmux split-window -t audio-guide:0 -h
	@tmux send-keys -t audio-guide:0 'make run-tts' C-m
	@tmux split-window -t audio-guide:0 -v
	@tmux send-keys -t audio-guide:0 'make run-llm' C-m
	@tmux select-pane -t audio-guide:0.0
	@tmux split-window -t audio-guide:0 -v
	@tmux send-keys -t audio-guide:0 'make run-kg' C-m
	@tmux attach -t audio-guide

run-stt:
	cd rust-services/stt-service && RUST_LOG=info cargo run

run-tts:
	cd rust-services/tts-service && RUST_LOG=info cargo run

run-llm:
	cd python-services && uv run python -m llm_engine.src.main

run-kg:
	cd python-services && uv run python -m knowledge_graph.src.main