# tests/test_engine.py

import os
import sys
import asyncio
import torch
import logging
import pytest

# Add the project root directory to the Python path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.engine import InferenceEngine, EngineConfig

# ------------------------------------------------------------------------------
# Fixture: Provide an event loop for synchronous tests
# ------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def event_loop_sync():
    """
    Creates and sets an event loop for tests that are not marked as async.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ------------------------------------------------------------------------------
# Test Cases
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_inference():
    """Test a single asynchronous inference call."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Linear(10, 2).to(device)
    config = EngineConfig(
        input_shape=[1, 10],
        trt_input_shape=[([1, 10], [32, 10], [128, 10])],
        use_tensorrt=False,
    )
    engine = InferenceEngine(model=model, config=config)
    
    # Create an input tensor with a batch dimension.
    input_tensor = torch.randn(1, 10, device=engine.device)
    output = await engine.run_inference_async(input_tensor)
    engine.logger.info(f"Async inference output shape: {output.shape}")
    
    # Check that the output has the expected shape.
    assert output.shape == (1, 2), "Async inference output shape mismatch"
    
    # Cleanup background tasks and await their cancellation.
    engine.cleanup()
    await asyncio.gather(engine.batch_processor_task, engine.autoscale_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_multiple_async_inference():
    """Test running multiple asynchronous inference calls concurrently."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Linear(10, 2).to(device)
    config = EngineConfig(
        input_shape=[1, 10],
        trt_input_shape=[([1, 10], [32, 10], [128, 10])],
        use_tensorrt=False,
    )
    engine = InferenceEngine(model=model, config=config)
    
    # Generate 100 valid inputs (each tensor has 10 features).
    inputs = [torch.randn(10, device=engine.device) for _ in range(100)]
    tasks = [engine.run_inference_async(x.unsqueeze(0)) for x in inputs]
    results = await asyncio.gather(*tasks)
    
    engine.logger.info(f"Received {len(results)} asynchronous inference results.")
    assert len(results) == 100, "Did not receive 100 results"
    for output in results:
        assert output.shape == (1, 2), "Async batch inference output shape mismatch"
    
    engine.cleanup()
    await asyncio.gather(engine.batch_processor_task, engine.autoscale_task, return_exceptions=True)


def test_batch_inference(event_loop_sync):
    """Test synchronous batch inference."""
    loop = event_loop_sync  # Use the provided event loop.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Linear(10, 2).to(device)
    config = EngineConfig(
        input_shape=[1, 10],
        trt_input_shape=[([1, 10], [32, 10], [128, 10])],
        use_tensorrt=False,
    )
    engine = InferenceEngine(model=model, config=config)
    
    # Generate 50 valid inputs.
    inputs = [torch.randn(10, device=engine.device) for _ in range(50)]
    batched_inputs = [x.unsqueeze(0) for x in inputs]  # Add batch dimension.
    outputs = engine.run_batch_inference(batched_inputs)
    
    engine.logger.info(f"Synchronous batch inference output shape: {outputs.shape}")
    assert outputs.shape[0] == 50, "Batch inference did not process 50 inputs"
    assert outputs.shape[1] == 2, "Batch inference output shape mismatch"
    
    engine.cleanup()
    # Await the cancellation of background tasks.
    loop.run_until_complete(
        asyncio.gather(engine.batch_processor_task, engine.autoscale_task, return_exceptions=True)
    )


def test_profile_inference(event_loop_sync):
    """Test the profiling functionality of the inference engine."""
    loop = event_loop_sync
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Linear(10, 2).to(device)
    config = EngineConfig(
        input_shape=[1, 10],
        trt_input_shape=[([1, 10], [32, 10], [128, 10])],
        use_tensorrt=False,
    )
    engine = InferenceEngine(model=model, config=config)
    
    input_tensor = torch.randn(1, 10, device=engine.device)
    metrics = engine.profile_inference(input_tensor)
    engine.logger.info(f"Profile metrics: {metrics}")
    
    # Check for expected profiling metric keys.
    expected_keys = {"preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"}
    missing_keys = expected_keys - metrics.keys()
    assert not missing_keys, f"Profile metrics missing keys: {missing_keys}"
    for key in expected_keys:
        assert isinstance(metrics[key], float), f"Metric {key} should be a float"
    
    engine.cleanup()
    loop.run_until_complete(
        asyncio.gather(engine.batch_processor_task, engine.autoscale_task, return_exceptions=True)
    )


def test_cleanup(event_loop_sync):
    """Test that cleanup runs without errors."""
    loop = event_loop_sync
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Linear(10, 2).to(device)
    config = EngineConfig(
        input_shape=[1, 10],
        trt_input_shape=[([1, 10], [32, 10], [128, 10])],
        use_tensorrt=False,
    )
    engine = InferenceEngine(model=model, config=config)
    
    # Simply ensure that cleanup does not raise any exceptions.
    try:
        engine.cleanup()
        loop.run_until_complete(
            asyncio.gather(engine.batch_processor_task, engine.autoscale_task, return_exceptions=True)
        )
    except Exception as e:
        pytest.fail(f"Cleanup raised an exception: {e}")


# ------------------------------------------------------------------------------
# Optional: Run tests if this file is executed directly.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
