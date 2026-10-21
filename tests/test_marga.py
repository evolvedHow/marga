"""Tests for marga."""

import pytest

from marga import HookContext, Proxy


@pytest.mark.asyncio
async def test_basic_routing():
    """Test basic router initialization and routing."""
    proxy = Proxy(backends={
        "test": {
            "base_url": "http://localhost:8080/v1",
            "models": ["test-model"]
        }
    })

    ctx = HookContext(messages=[], model="test-model")
    backend, model = await proxy.router.route(ctx)

    assert backend.name == "test"
    assert model == "test-model"


@pytest.mark.asyncio
async def test_custom_router():
    """Test custom routing function."""
    proxy = Proxy(backends={
        "fast": {"base_url": "http://fast/v1", "models": ["fast-model"]},
        "slow": {"base_url": "http://slow/v1", "models": ["slow-model"]}
    })

    @proxy.route
    async def custom(ctx):
        if ctx.estimated_tokens < 100:
            return "fast/fast-model"
        return "slow/slow-model"

    # Small request
    ctx = HookContext(messages=[{"role": "user", "content": "hi"}], model="")
    backend, model = await proxy.router.route(ctx)
    assert backend.name == "fast"

    # Large request
    long_msg = "x" * 1000
    ctx = HookContext(messages=[{"role": "user", "content": long_msg}], model="")
    backend, model = await proxy.router.route(ctx)
    assert backend.name == "slow"


@pytest.mark.asyncio
async def test_before_hooks():
    """Test before hooks modify context."""
    proxy = Proxy(backends={"test": {"base_url": "http://test/v1"}})

    @proxy.before
    async def add_system(ctx):
        ctx.messages.insert(0, {"role": "system", "content": "Be helpful"})

    ctx = HookContext(messages=[{"role": "user", "content": "hello"}], model="test")
    await proxy.hooks.run_before(ctx)

    assert len(ctx.messages) == 2
    assert ctx.messages[0]["role"] == "system"


@pytest.mark.asyncio
async def test_after_hooks():
    """Test after hooks modify response."""
    proxy = Proxy(backends={"test": {"base_url": "http://test/v1"}})

    @proxy.after
    async def add_metadata(ctx, response):
        response["custom_field"] = "added"
        return response

    ctx = HookContext(messages=[], model="test")
    response = {"choices": []}
    result = await proxy.hooks.run_after(ctx, response)

    assert "custom_field" in result
    assert result["custom_field"] == "added"


@pytest.mark.asyncio
async def test_usage_tracking():
    """Test usage statistics tracking."""
    proxy = Proxy(backends={
        "backend1": {"base_url": "http://b1/v1"},
        "backend2": {"base_url": "http://b2/v1"}
    })

    await proxy.router.record_usage("backend1", tokens=100)
    await proxy.router.record_usage("backend1", tokens=200)
    await proxy.router.record_usage("backend2", tokens=50)

    stats = await proxy.router.get_stats()

    assert stats["backend1"]["calls"] == 2
    assert stats["backend1"]["tokens"] == 300
    assert stats["backend2"]["calls"] == 1
    assert stats["backend2"]["tokens"] == 50


@pytest.mark.asyncio
async def test_least_used_routing():
    """Test least-used backend selection."""
    from marga import least_used

    proxy = Proxy(backends={
        "backend1": {"base_url": "http://b1/v1", "models": ["m1"]},
        "backend2": {"base_url": "http://b2/v1", "models": ["m2"]}
    })
    proxy.route(least_used())

    # Record usage on backend1
    await proxy.router.record_usage("backend1", tokens=100)

    # Should route to backend2 (least used)
    ctx = HookContext(messages=[], model="")
    backend, model = await proxy.router.route(ctx)

    assert backend.name == "backend2"


def test_token_estimation():
    """Test token estimation."""
    messages = [
        {"role": "user", "content": "a" * 400},  # ~100 tokens
        {"role": "assistant", "content": "b" * 800}  # ~200 tokens
    ]

    ctx = HookContext(messages=messages, model="test")
    assert ctx.estimated_tokens == 300  # (400 + 800) / 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
