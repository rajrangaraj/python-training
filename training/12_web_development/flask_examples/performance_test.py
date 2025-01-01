"""
Performance testing script for Flask application.
"""

import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

def measure_response_time(url):
    """Measure response time for a single request."""
    start_time = time.time()
    response = requests.get(url)
    end_time = time.time()
    
    return {
        'time': end_time - start_time,
        'status': response.status_code
    }

def run_load_test(url, num_requests=100, concurrent=10):
    """Run load test with concurrent requests."""
    results = []
    
    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = [
            executor.submit(measure_response_time, url)
            for _ in range(num_requests)
        ]
        
        for future in futures:
            result = future.result()
            results.append(result)
    
    return results

def analyze_results(results):
    """Analyze test results."""
    times = [r['time'] for r in results]
    successful = sum(1 for r in results if r['status'] == 200)
    
    analysis = {
        'total_requests': len(results),
        'successful_requests': successful,
        'min_time': min(times),
        'max_time': max(times),
        'mean_time': statistics.mean(times),
        'median_time': statistics.median(times),
        'std_dev': statistics.stdev(times)
    }
    
    return analysis, times

def plot_results(times, title='Response Time Distribution'):
    """Plot response time distribution."""
    plt.figure(figsize=(10, 6))
    plt.hist(times, bins=30, edgecolor='black')
    plt.title(title)
    plt.xlabel('Response Time (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('response_times.png')
    plt.close()

def main():
    """Run performance tests."""
    base_url = 'http://localhost:5000'
    endpoints = [
        '/api/products',
        '/api/products/1'
    ]
    
    for endpoint in endpoints:
        url = f'{base_url}{endpoint}'
        print(f'\nTesting endpoint: {endpoint}')
        
        # First request (cold start)
        print('\nCold start request:')
        cold_result = measure_response_time(url)
        print(f'Response time: {cold_result["time"]:.4f}s')
        
        # Load test
        print('\nRunning load test...')
        results = run_load_test(url)
        analysis, times = analyze_results(results)
        
        print('\nResults:')
        print(f'Total requests: {analysis["total_requests"]}')
        print(f'Successful requests: {analysis["successful_requests"]}')
        print(f'Min time: {analysis["min_time"]:.4f}s')
        print(f'Max time: {analysis["max_time"]:.4f}s')
        print(f'Mean time: {analysis["mean_time"]:.4f}s')
        print(f'Median time: {analysis["median_time"]:.4f}s')
        print(f'Standard deviation: {analysis["std_dev"]:.4f}s')
        
        # Plot results
        plot_results(times, f'Response Times - {endpoint}')
        print(f'\nPlot saved as response_times_{endpoint.replace("/", "_")}.png')

if __name__ == '__main__':
    main() 