"""
Demonstration of distributed systems concepts and consensus patterns.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Any
from enum import Enum
import time
import random
import asyncio
import json
import hashlib
from datetime import datetime
import threading
from queue import Queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Node States for Raft Consensus
class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

@dataclass
class LogEntry:
    """Represents a log entry in the distributed log."""
    term: int
    index: int
    command: str
    data: Dict[str, Any]

class RaftNode:
    """Implementation of Raft consensus algorithm."""
    
    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        self.state = NodeState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        self.leader_id: Optional[str] = None
        self.election_timeout = random.uniform(150, 300)  # milliseconds
        self.last_heartbeat = time.time() * 1000
        self._lock = threading.Lock()
        
        # Leader-specific state
        self.next_index: Dict[str, int] = {peer: 1 for peer in peers}
        self.match_index: Dict[str, int] = {peer: 0 for peer in peers}
    
    def reset_election_timeout(self):
        """Reset election timeout with random duration."""
        self.election_timeout = random.uniform(150, 300)
        self.last_heartbeat = time.time() * 1000
    
    async def start_election(self):
        """Start leader election process."""
        with self._lock:
            self.state = NodeState.CANDIDATE
            self.current_term += 1
            self.voted_for = self.node_id
            votes_received = 1
            
            # Request votes from peers
            for peer in self.peers:
                try:
                    vote_granted = await self.request_vote(peer)
                    if vote_granted:
                        votes_received += 1
                except Exception as e:
                    logger.error(f"Error requesting vote from {peer}: {e}")
            
            # Check if won election
            if votes_received > (len(self.peers) + 1) / 2:
                self.state = NodeState.LEADER
                self.leader_id = self.node_id
                await self.send_heartbeat()
            else:
                self.state = NodeState.FOLLOWER
    
    async def request_vote(self, peer: str) -> bool:
        """Request vote from a peer."""
        # Simulate network request
        await asyncio.sleep(random.uniform(0.01, 0.05))
        return random.random() > 0.3  # 70% chance of getting vote
    
    async def send_heartbeat(self):
        """Send heartbeat to all peers."""
        if self.state != NodeState.LEADER:
            return
        
        for peer in self.peers:
            try:
                await self.append_entries(peer)
            except Exception as e:
                logger.error(f"Error sending heartbeat to {peer}: {e}")
    
    async def append_entries(self, peer: str):
        """Send AppendEntries RPC to peer."""
        next_idx = self.next_index[peer]
        prev_log_index = next_idx - 1
        prev_log_term = 0
        
        if prev_log_index > 0:
            prev_log_term = self.log[prev_log_index - 1].term
        
        entries = self.log[next_idx - 1:]
        
        # Simulate network request
        await asyncio.sleep(random.uniform(0.01, 0.05))
        success = random.random() > 0.1  # 90% success rate
        
        if success:
            self.next_index[peer] = len(self.log) + 1
            self.match_index[peer] = len(self.log)
        else:
            self.next_index[peer] = max(1, self.next_index[peer] - 1)

# Distributed Key-Value Store
class DistributedKVStore:
    """Distributed key-value store using eventual consistency."""
    
    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        self.data: Dict[str, Any] = {}
        self.vector_clock: Dict[str, int] = {node: 0 for node in [node_id] + peers}
        self.conflict_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def _increment_vector_clock(self):
        """Increment vector clock for current node."""
        self.vector_clock[self.node_id] += 1
    
    def _merge_vector_clocks(self, other_clock: Dict[str, int]):
        """Merge vector clocks using element-wise maximum."""
        for node, count in other_clock.items():
            self.vector_clock[node] = max(self.vector_clock[node], count)
    
    async def put(self, key: str, value: Any):
        """Put value into distributed store."""
        with self._lock:
            self._increment_vector_clock()
            timestamp = datetime.now().isoformat()
            
            self.data[key] = {
                'value': value,
                'timestamp': timestamp,
                'vector_clock': self.vector_clock.copy(),
                'node_id': self.node_id
            }
            
            # Propagate to peers
            for peer in self.peers:
                try:
                    await self._replicate_to_peer(peer, key)
                except Exception as e:
                    logger.error(f"Error replicating to {peer}: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from distributed store."""
        with self._lock:
            if key not in self.data:
                return None
            return self.data[key]['value']
    
    async def _replicate_to_peer(self, peer: str, key: str):
        """Replicate key-value pair to peer."""
        # Simulate network request
        await asyncio.sleep(random.uniform(0.01, 0.05))
        
        # Simulate peer update
        if random.random() > 0.1:  # 90% success rate
            logger.info(f"Replicated {key} to {peer}")
        else:
            raise Exception("Replication failed")

# Distributed Lock
class DistributedLock:
    """Distributed lock implementation."""
    
    def __init__(self, lock_id: str, ttl_seconds: int = 30):
        self.lock_id = lock_id
        self.ttl_seconds = ttl_seconds
        self.owner: Optional[str] = None
        self.expiry: Optional[float] = None
        self._lock = threading.Lock()
    
    async def acquire(self, requester_id: str) -> bool:
        """Attempt to acquire the lock."""
        with self._lock:
            now = time.time()
            
            # Check if lock is free or expired
            if self.owner is None or (self.expiry and now > self.expiry):
                self.owner = requester_id
                self.expiry = now + self.ttl_seconds
                return True
            
            return False
    
    async def release(self, requester_id: str) -> bool:
        """Release the lock if owned by requester."""
        with self._lock:
            if self.owner == requester_id:
                self.owner = None
                self.expiry = None
                return True
            return False
    
    async def refresh(self, requester_id: str) -> bool:
        """Refresh lock TTL if owned by requester."""
        with self._lock:
            if self.owner == requester_id:
                self.expiry = time.time() + self.ttl_seconds
                return True
            return False

async def demonstrate_distributed_systems():
    """Demonstrate distributed systems concepts."""
    
    # Initialize Raft cluster
    nodes = [
        RaftNode("node1", ["node2", "node3"]),
        RaftNode("node2", ["node1", "node3"]),
        RaftNode("node3", ["node1", "node2"])
    ]
    
    # Initialize distributed KV store
    kv_stores = [
        DistributedKVStore("store1", ["store2", "store3"]),
        DistributedKVStore("store2", ["store1", "store3"]),
        DistributedKVStore("store3", ["store1", "store2"])
    ]
    
    # Initialize distributed lock
    lock = DistributedLock("resource_lock")
    
    # Simulate distributed operations
    logger.info("Starting distributed operations simulation...")
    
    # Simulate Raft leader election
    await nodes[0].start_election()
    
    # Simulate distributed KV store operations
    await kv_stores[0].put("key1", "value1")
    result = await kv_stores[1].get("key1")
    logger.info(f"Retrieved value: {result}")
    
    # Simulate distributed lock operations
    acquired = await lock.acquire("client1")
    logger.info(f"Lock acquired by client1: {acquired}")
    
    acquired = await lock.acquire("client2")
    logger.info(f"Lock acquired by client2: {acquired}")
    
    released = await lock.release("client1")
    logger.info(f"Lock released by client1: {released}")

if __name__ == "__main__":
    asyncio.run(demonstrate_distributed_systems()) 