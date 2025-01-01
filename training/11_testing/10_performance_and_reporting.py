"""
Demonstration of performance testing, load testing, and test reporting.
"""

import pytest
import time
import statistics
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import asyncio
import aiohttp
import json
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

# Performance metrics collection
@dataclass
class PerformanceMetrics:
    """Container for performance test metrics."""
    operation: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    success: bool
    error: Optional[str] = None

class PerformanceMonitor:
    """Monitor and collect performance metrics."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
    
    def record(self, metrics: PerformanceMetrics) -> None:
        """Record a performance metric."""
        self.metrics.append(metrics)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics from collected metrics."""
        if not self.metrics:
            return {}
        
        durations = [m.duration_ms for m in self.metrics if m.success]
        success_count = sum(1 for m in self.metrics if m.success)
        
        return {
            "total_operations": len(self.metrics),
            "successful_operations": success_count,
            "failure_rate": (len(self.metrics) - success_count) / len(self.metrics),
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "avg_duration_ms": statistics.mean(durations) if durations else 0,
            "median_duration_ms": statistics.median(durations) if durations else 0
        }

# Test subject: A simulated database
class SimulatedDatabase:
    """Simulate database operations with artificial delays."""
    
    def __init__(self, avg_delay: float = 0.1):
        self.avg_delay = avg_delay
        self.data: Dict[str, Any] = {}
    
    async def read(self, key: str) -> Any:
        """Simulate reading from database."""
        await asyncio.sleep(self.avg_delay * (0.5 + random.random()))
        return self.data.get(key)
    
    async def write(self, key: str, value: Any) -> None:
        """Simulate writing to database."""
        await asyncio.sleep(self.avg_delay * (0.8 + random.random()))
        self.data[key] = value

# Load testing helpers
async def load_test_operation(
    operation: Callable,
    monitor: PerformanceMonitor,
    operation_name: str,
    *args
) -> None:
    """Execute and monitor a single operation."""
    start_time = datetime.now()
    try:
        await operation(*args)
        success = True
        error = None
    except Exception as e:
        success = False
        error = str(e)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() * 1000
    
    monitor.record(PerformanceMetrics(
        operation=operation_name,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration,
        success=success,
        error=error
    ))

async def run_load_test(
    operation: Callable,
    monitor: PerformanceMonitor,
    operation_name: str,
    num_operations: int,
    concurrency: int,
    *args
) -> None:
    """Run load test with specified concurrency."""
    tasks = []
    for _ in range(num_operations):
        task = load_test_operation(operation, monitor, operation_name, *args)
        tasks.append(task)
        
        if len(tasks) >= concurrency:
            await asyncio.gather(*tasks)
            tasks = []
    
    if tasks:
        await asyncio.gather(*tasks)

# Test fixtures
@pytest.fixture
def performance_monitor() -> PerformanceMonitor:
    """Provide a performance monitor."""
    return PerformanceMonitor()

@pytest.fixture
def db() -> SimulatedDatabase:
    """Provide a simulated database."""
    return SimulatedDatabase()

# Performance tests
@pytest.mark.asyncio
async def test_database_performance(
    db: SimulatedDatabase,
    performance_monitor: PerformanceMonitor
):
    """Test database performance under load."""
    # Test write performance
    await run_load_test(
        db.write,
        performance_monitor,
        "write",
        num_operations=100,
        concurrency=10,
        *["test_key", "test_value"]
    )
    
    write_stats = performance_monitor.get_statistics()
    assert write_stats["failure_rate"] == 0
    assert write_stats["avg_duration_ms"] < 200  # Adjust threshold as needed
    
    # Test read performance
    await run_load_test(
        db.read,
        performance_monitor,
        "read",
        num_operations=100,
        concurrency=10,
        *["test_key"]
    )
    
    read_stats = performance_monitor.get_statistics()
    assert read_stats["failure_rate"] == 0
    assert read_stats["avg_duration_ms"] < 150  # Adjust threshold as needed

# Generate HTML report
def generate_report(monitor: PerformanceMonitor, output_path: str) -> None:
    """Generate HTML performance report."""
    stats = monitor.get_statistics()
    
    html = f"""
    <html>
    <head>
        <title>Performance Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Performance Test Report</h1>
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Operations</td>
                <td>{stats['total_operations']}</td>
            </tr>
            <tr>
                <td>Successful Operations</td>
                <td>{stats['successful_operations']}</td>
            </tr>
            <tr>
                <td>Failure Rate</td>
                <td>{stats['failure_rate']:.2%}</td>
            </tr>
            <tr>
                <td>Average Duration (ms)</td>
                <td>{stats['avg_duration_ms']:.2f}</td>
            </tr>
            <tr>
                <td>Median Duration (ms)</td>
                <td>{stats['median_duration_ms']:.2f}</td>
            </tr>
            <tr>
                <td>Min Duration (ms)</td>
                <td>{stats['min_duration_ms']:.2f}</td>
            </tr>
            <tr>
                <td>Max Duration (ms)</td>
                <td>{stats['max_duration_ms']:.2f}</td>
            </tr>
        </table>
        
        <h2>Detailed Metrics</h2>
        <table>
            <tr>
                <th>Operation</th>
                <th>Start Time</th>
                <th>Duration (ms)</th>
                <th>Success</th>
                <th>Error</th>
            </tr>
    """
    
    for metric in monitor.metrics:
        html += f"""
            <tr>
                <td>{metric.operation}</td>
                <td>{metric.start_time.isoformat()}</td>
                <td>{metric.duration_ms:.2f}</td>
                <td>{'Yes' if metric.success else 'No'}</td>
                <td>{metric.error or ''}</td>
            </tr>
        """
    
    html += """
        </table>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)

# Run tests with reporting
def pytest_sessionfinish(session, exitstatus):
    """Generate report after test session."""
    for item in session.items:
        if hasattr(item, 'fixturenames') and 'performance_monitor' in item.fixturenames:
            monitor = item.funcargs['performance_monitor']
            report_path = Path('performance_report.html')
            generate_report(monitor, str(report_path))
            print(f"\nPerformance report generated: {report_path.absolute()}")

if __name__ == '__main__':
    pytest.main([__file__, "-v"]) 