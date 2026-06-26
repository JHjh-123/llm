from __future__ import annotations

import os
import time
import threading

# We configure test environments for the integration test
os.environ["DATABASE_TYPE"] = os.getenv("DATABASE_TYPE", "postgres")
os.environ["MESSAGE_BUS_TYPE"] = os.getenv("MESSAGE_BUS_TYPE", "rabbitmq")

from multi_agent.memory import SharedMemory
from multi_agent.state_exchange import StateStore
from multi_agent.message_bus import MessageBus


def test_postgres_memory():
    print("--- [1] Testing PostgreSQL/SQLite SharedMemory ---")
    memory = SharedMemory(reset=True)
    print(f"Active SharedMemory backend: {memory.db_type.upper()}")
    
    # Test record writing
    record = memory.add(
        source_agent="test_suite",
        task_topic="verification",
        summary="Testing backend middleware integration.",
        tags=["test", "middleware"]
    )
    print(f"Memory Record written successfully. ID: {record.memory_id}")
    
    # Test record searching
    results = memory.search("middleware integration", limit=1)
    if results:
        print(f"Search match successful. Matched topic: '{results[0].task_topic}'")
    else:
        print("Error: Search yielded no matches.")
        
    memory.close()


def test_postgres_states():
    print("\n--- [2] Testing PostgreSQL/SQLite StateStore ---")
    store = StateStore(reset=True)
    print(f"Active StateStore backend: {store.db_type.upper()}")
    
    # Test state writing
    record = store.put(
        producer_agent="test_suite",
        task_id="test_task_id",
        state_type="embedding",
        payload=[0.1, 0.2, 0.3, 0.4],
        metadata={"test": True}
    )
    print(f"State Record written successfully. ID: {record.state_id}")
    
    # Test state reading
    retrieved = store.get(record.state_id)
    if retrieved:
        print(f"State retrieval successful. Producer: {retrieved.producer_agent}")
    else:
        print("Error: State retrieval failed.")
        
    store.close()


def test_rabbitmq_message_bus():
    print("\n--- [3] Testing RabbitMQ/In-Memory MessageBus ---")
    bus = MessageBus()
    print(f"Active MessageBus backend: {bus.bus_type.upper()}")
    
    received_messages = []
    event = threading.Event()
    
    def callback(msg):
        print(f"Received message on bus: {msg}")
        received_messages.append(msg)
        event.set()
        
    bus.subscribe("verification_channel", callback)
    
    # Wait for subscription thread to spin up
    time.sleep(0.5)
    
    test_payload = {"sender": "test_suite", "message": "hello middleware"}
    bus.publish("verification_channel", test_payload)
    
    # Wait for callback execution
    completed = event.wait(timeout=2.0)
    if completed and received_messages:
        print("Message publication and receipt verified successfully!")
    else:
        print("Error: Message receipt timed out or failed.")


def main():
    print("====================================================")
    print("Multi-Agent Backend Integration Verification Suite")
    print("====================================================")
    
    test_postgres_memory()
    test_postgres_states()
    test_rabbitmq_message_bus()
    
    print("\nAll integration verification tasks executed.")
    print("====================================================")


if __name__ == "__main__":
    main()
