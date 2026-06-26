from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable

try:
    import pika
    HAS_PIKA = True
except ImportError:
    HAS_PIKA = False


class MessageBus:
    """Enterprise-grade message bus supporting RabbitMQ and in-memory fallback.

    All configurations default to RabbitMQ on localhost with password '123456'.
    """

    def __init__(self) -> None:
        self.bus_type = os.getenv("MESSAGE_BUS_TYPE", "in_memory").lower()
        self.rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:123456@localhost:5673/%2F")
        
        self._local_subscribers: dict[str, list[Callable[[dict], None]]] = {}
        self._local_lock = threading.Lock()
        
        connected = False
        if self.bus_type == "rabbitmq":
            if HAS_PIKA:
                try:
                    # Test connection to RabbitMQ
                    parameters = pika.URLParameters(self.rabbitmq_url)
                    parameters.connection_attempts = 1
                    parameters.retry_delay = 1
                    parameters.socket_timeout = 2.0
                    connection = pika.BlockingConnection(parameters)
                    connection.close()
                    connected = True
                except Exception as e:
                    print(f"Warning: Failed to connect to RabbitMQ: {e}. Falling back to in-memory queue.")
                    self.bus_type = "in_memory"
            else:
                print("Warning: pika package not found. Falling back to in-memory queue.")
                self.bus_type = "in_memory"

        if not connected:
            self.bus_type = "in_memory"

    def publish(self, channel: str, message: dict[str, Any]) -> None:
        """Publish a structured message dictionary to a channel."""
        payload = json.dumps(message, ensure_ascii=False)
        if self.bus_type == "rabbitmq" and HAS_PIKA:
            try:
                parameters = pika.URLParameters(self.rabbitmq_url)
                connection = pika.BlockingConnection(parameters)
                ch = connection.channel()
                # Declare exchange with name `channel` (fanout)
                ch.exchange_declare(exchange=channel, exchange_type='fanout')
                ch.basic_publish(exchange=channel, routing_key='', body=payload.encode('utf-8'))
                connection.close()
            except Exception as e:
                print(f"Error publishing to RabbitMQ: {e}")
        else:
            # Local Pub/Sub
            with self._local_lock:
                callbacks = list(self._local_subscribers.get(channel, []))
            
            for callback in callbacks:
                # Dispatch in a separate thread to mimic network async behavior and prevent blocking
                threading.Thread(target=self._safe_invoke, args=(callback, message), daemon=True).start()

    def subscribe(self, channel: str, callback: Callable[[dict], None]) -> None:
        """Subscribe to a channel with a callback function."""
        if self.bus_type == "rabbitmq" and HAS_PIKA:
            threading.Thread(target=self._rabbitmq_listen_loop, args=(channel, callback), daemon=True).start()
        else:
            with self._local_lock:
                if channel not in self._local_subscribers:
                    self._local_subscribers[channel] = []
                self._local_subscribers[channel].append(callback)

    def _safe_invoke(self, callback: Callable[[dict], None], message: dict) -> None:
        try:
            callback(message)
        except Exception as e:
            print(f"Error executing callback on message bus: {e}")

    def _rabbitmq_listen_loop(self, channel_name: str, callback: Callable[[dict], None]) -> None:
        try:
            parameters = pika.URLParameters(self.rabbitmq_url)
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            
            # Declare exchange
            channel.exchange_declare(exchange=channel_name, exchange_type='fanout')
            
            # Declare exclusive queue
            result = channel.queue_declare(queue='', exclusive=True)
            queue_name = result.method.queue
            
            # Bind queue to exchange
            channel.queue_bind(exchange=channel_name, queue=queue_name, routing_key='')
            
            def on_message(ch, method, properties, body):
                try:
                    data = json.loads(body.decode('utf-8'))
                    self._safe_invoke(callback, data)
                except Exception as e:
                    print(f"Error parsing RabbitMQ message data: {e}")
                    
            channel.basic_consume(queue=queue_name, on_message_callback=on_message, auto_ack=True)
            channel.start_consuming()
        except Exception as e:
            print(f"RabbitMQ Pub/Sub listening error on channel '{channel_name}': {e}")
