from dataclasses import dataclass
from collections import deque


@dataclass
class Request:
    request_id: str
    remaining_tokens: int


# Requests waiting to be scheduled
waiting_queue = deque([
    Request("A", 5),
    Request("B", 2),
    Request("C", 4),
    Request("D", 3),
])

# Currently active requests
active_requests = []

MAX_BATCH_SIZE = 3

step = 0

while waiting_queue or active_requests:

    step += 1

    # Fill available batch slots
    while (
        waiting_queue
        and len(active_requests) < MAX_BATCH_SIZE
    ):
        request = waiting_queue.popleft()
        active_requests.append(request)

    print(f"\nStep {step}")
    print(
        "Active batch:",
        [r.request_id for r in active_requests]
    )

    # Simulate generating ONE token for every active request
    for request in active_requests:
        request.remaining_tokens -= 1

    # Remove completed requests
    completed = [
        r for r in active_requests
        if r.remaining_tokens == 0
    ]

    active_requests = [
        r for r in active_requests
        if r.remaining_tokens > 0
    ]

    if completed:
        print(
            "Completed:",
            [r.request_id for r in completed]
        )