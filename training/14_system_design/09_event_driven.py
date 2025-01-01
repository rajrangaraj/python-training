"""
Demonstration of event-driven architecture and message queue patterns.
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import asyncio
import aiohttp
from abc import ABC, abstractmethod
import logging
from enum import Enum
import uuid
import redis
from kafka import KafkaProducer, KafkaConsumer
import pika
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Event Types
class EventType(Enum):
    ORDER_CREATED = "order.created"
    ORDER_UPDATED = "order.updated"
    PAYMENT_PROCESSED = "payment.processed"
    INVENTORY_UPDATED = "inventory.updated"
    NOTIFICATION_SENT = "notification.sent"

@dataclass
class Event:
    """Base event class."""
    id: str
    type: EventType
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime

# Event Bus Interface
class EventBus(ABC):
    @abstractmethod
    async def publish(self, event: Event) -> None:
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], None]
    ) -> None:
        pass

# Kafka Implementation
class KafkaEventBus(EventBus):
    """Kafka-based event bus implementation."""
    
    def __init__(self, bootstrap_servers: List[str]):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode('utf-8'))
        )
        self.handlers: Dict[EventType, List[Callable]] = {}
    
    async def publish(self, event: Event) -> None:
        try:
            self.producer.send(
                event.type.value,
                {
                    "id": event.id,
                    "type": event.type.value,
                    "data": event.data,
                    "metadata": event.metadata,
                    "timestamp": event.timestamp.isoformat()
                }
            )
            logger.info(
                "event_published",
                event_id=event.id,
                event_type=event.type.value
            )
        except Exception as e:
            logger.error(
                "event_publish_failed",
                event_id=event.id,
                error=str(e)
            )
            raise
    
    async def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], None]
    ) -> None:
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        self.consumer.subscribe([event_type.value])
        
        # Start consuming messages
        asyncio.create_task(self._consume())
    
    async def _consume(self):
        """Consume messages from Kafka."""
        for message in self.consumer:
            event_data = message.value
            event = Event(
                id=event_data["id"],
                type=EventType(event_data["type"]),
                data=event_data["data"],
                metadata=event_data["metadata"],
                timestamp=datetime.fromisoformat(event_data["timestamp"])
            )
            
            if event.type in self.handlers:
                for handler in self.handlers[event.type]:
                    try:
                        await handler(event)
                        logger.info(
                            "event_processed",
                            event_id=event.id,
                            event_type=event.type.value
                        )
                    except Exception as e:
                        logger.error(
                            "event_processing_failed",
                            event_id=event.id,
                            error=str(e)
                        )

# RabbitMQ Implementation
class RabbitMQEventBus(EventBus):
    """RabbitMQ-based event bus implementation."""
    
    def __init__(self, connection_params: pika.ConnectionParameters):
        self.connection = pika.BlockingConnection(connection_params)
        self.channel = self.connection.channel()
        self.handlers: Dict[EventType, List[Callable]] = {}
    
    async def publish(self, event: Event) -> None:
        try:
            self.channel.basic_publish(
                exchange='',
                routing_key=event.type.value,
                body=json.dumps({
                    "id": event.id,
                    "type": event.type.value,
                    "data": event.data,
                    "metadata": event.metadata,
                    "timestamp": event.timestamp.isoformat()
                })
            )
            logger.info(
                "event_published",
                event_id=event.id,
                event_type=event.type.value
            )
        except Exception as e:
            logger.error(
                "event_publish_failed",
                event_id=event.id,
                error=str(e)
            )
            raise
    
    async def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Event], None]
    ) -> None:
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        
        self.channel.queue_declare(queue=event_type.value)
        self.channel.basic_consume(
            queue=event_type.value,
            on_message_callback=self._handle_message
        )
        
        # Start consuming messages
        asyncio.create_task(self._consume())
    
    def _handle_message(self, ch, method, properties, body):
        """Handle incoming RabbitMQ message."""
        event_data = json.loads(body)
        event = Event(
            id=event_data["id"],
            type=EventType(event_data["type"]),
            data=event_data["data"],
            metadata=event_data["metadata"],
            timestamp=datetime.fromisoformat(event_data["timestamp"])
        )
        
        if event.type in self.handlers:
            for handler in self.handlers[event.type]:
                try:
                    asyncio.create_task(handler(event))
                    logger.info(
                        "event_processed",
                        event_id=event.id,
                        event_type=event.type.value
                    )
                except Exception as e:
                    logger.error(
                        "event_processing_failed",
                        event_id=event.id,
                        error=str(e)
                    )
    
    async def _consume(self):
        """Start consuming messages."""
        self.channel.start_consuming()

# Event Handlers
class OrderService:
    """Service for handling order-related events."""
    
    async def handle_order_created(self, event: Event):
        """Handle order.created event."""
        order_id = event.data["order_id"]
        # Process order logic...
        logger.info("order_processing_started", order_id=order_id)
        
        # Publish inventory update event
        await event_bus.publish(Event(
            id=str(uuid.uuid4()),
            type=EventType.INVENTORY_UPDATED,
            data={"order_id": order_id},
            metadata={"source": "order_service"},
            timestamp=datetime.utcnow()
        ))

class PaymentService:
    """Service for handling payment-related events."""
    
    async def handle_payment_processed(self, event: Event):
        """Handle payment.processed event."""
        order_id = event.data["order_id"]
        # Process payment logic...
        logger.info("payment_processed", order_id=order_id)
        
        # Update order status
        await event_bus.publish(Event(
            id=str(uuid.uuid4()),
            type=EventType.ORDER_UPDATED,
            data={
                "order_id": order_id,
                "status": "paid"
            },
            metadata={"source": "payment_service"},
            timestamp=datetime.utcnow()
        ))

class NotificationService:
    """Service for handling notification events."""
    
    async def handle_order_updated(self, event: Event):
        """Handle order.updated event."""
        order_id = event.data["order_id"]
        # Send notification logic...
        logger.info("notification_sent", order_id=order_id)
        
        await event_bus.publish(Event(
            id=str(uuid.uuid4()),
            type=EventType.NOTIFICATION_SENT,
            data={
                "order_id": order_id,
                "type": "order_status_update"
            },
            metadata={"source": "notification_service"},
            timestamp=datetime.utcnow()
        ))

async def demonstrate_event_driven():
    """Demonstrate event-driven architecture."""
    
    # Initialize services
    order_service = OrderService()
    payment_service = PaymentService()
    notification_service = NotificationService()
    
    # Subscribe to events
    await event_bus.subscribe(
        EventType.ORDER_CREATED,
        order_service.handle_order_created
    )
    await event_bus.subscribe(
        EventType.PAYMENT_PROCESSED,
        payment_service.handle_payment_processed
    )
    await event_bus.subscribe(
        EventType.ORDER_UPDATED,
        notification_service.handle_order_updated
    )
    
    # Simulate order creation
    order_id = str(uuid.uuid4())
    await event_bus.publish(Event(
        id=str(uuid.uuid4()),
        type=EventType.ORDER_CREATED,
        data={"order_id": order_id},
        metadata={"source": "api"},
        timestamp=datetime.utcnow()
    ))
    
    # Wait for event processing
    await asyncio.sleep(5)

if __name__ == "__main__":
    # Initialize event bus (Kafka example)
    event_bus = KafkaEventBus(["localhost:9092"])
    
    # Run demonstration
    asyncio.run(demonstrate_event_driven()) 