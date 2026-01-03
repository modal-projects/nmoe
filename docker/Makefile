# Docker image build chain
#
# Usage:
#   make all        # build full chain up to runner
#   make train      # build up to train image
#   make infer      # build inference image (DeepEP/DeepGEMM/FlashMLA)
#   make deps       # rebuild from deps down (after adding pip packages)
#   make push       # push all images
#   make push-runner # push only runner to GHCR
#
# Images:
#   base   → xjdr/nmoe:base       (CUDA + system, ~6GB, rarely changes)
#   system → xjdr/nmoe:system     (torch + triton + vendors, ~24GB, almost never)
#   deps   → xjdr/nmoe:deps       (pip deps, ~500MB, sometimes)
#   train  → xjdr/nmoe_train      (built kernels, changes with code)
#   infer  → xjdr/nmoe_infer      (inference: DeepEP/DeepGEMM/FlashMLA/nixl)
#   runner → xjdr/nmoe:runner     (CI runner)

.PHONY: all build base system deps train infer runner push push-runner push-infer clean

# Default target
.DEFAULT_GOAL := all

DOCKER_BUILD = docker build
CONTEXT = ..

# Base: CUDA + system packages + uv + venv
base:
	$(DOCKER_BUILD) -f Dockerfile.base -t xjdr/nmoe:base $(CONTEXT)

# System: torch + triton + vendors (depends on base)
system: base
	$(DOCKER_BUILD) -f Dockerfile.system -t xjdr/nmoe:system $(CONTEXT)

# Deps: pip packages (depends on system)
deps: system
	$(DOCKER_BUILD) -f Dockerfile.deps -t xjdr/nmoe:deps $(CONTEXT)

# Train: copy source + build kernels (depends on deps)
train: deps
	$(DOCKER_BUILD) -f Dockerfile.train -t xjdr/nmoe_train:latest $(CONTEXT)

# Infer: inference with DeepEP/DeepGEMM/full FlashMLA (depends on deps)
infer: deps
	$(DOCKER_BUILD) -f Dockerfile.infer -t xjdr/nmoe_infer:latest $(CONTEXT)

# Runner: add GitHub Actions runner (depends on train)
runner: train
	$(DOCKER_BUILD) -f Dockerfile.runner -t xjdr/nmoe:runner $(CONTEXT)

# Build everything
build: runner

# Build and push everything
all: build push

# Push all images to registries
push: push-base push-system push-deps push-train push-infer push-runner

push-base:
	docker push xjdr/nmoe:base

push-system:
	docker push xjdr/nmoe:system

push-deps:
	docker push xjdr/nmoe:deps

push-train:
	docker push xjdr/nmoe_train:latest

push-infer:
	docker push xjdr/nmoe_infer:latest

push-runner:
	docker push xjdr/nmoe:runner

# Force rebuild from a specific stage (ignores dependencies)
force-%:
	$(DOCKER_BUILD) --no-cache -f Dockerfile.$* -t xjdr/nmoe:$* $(CONTEXT)

# Show image sizes
sizes:
	@docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" | grep -E "nmoe|REPOSITORY"

# Clean up dangling images
clean:
	docker image prune -f
